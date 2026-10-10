"""Fixed-K conditional Gaussian regression mixtures with joint Louis information.

An explicit outcome-unit sigma_min defines a constrained likelihood. The best
converged constrained endpoint is retained even when its bound is active and
ordinary interior inference is unavailable. No variance jitter or component
regression covariance is substituted for the joint model information.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.tsworkflows.sspace import _cpu_call
from openecon.econometrics.tsworkflows.ssdiffuse import (
    _digest,
    _index_admission,
    _index_record,
    _json_admission,
    _json_value,
    _load_record,
    _response_dtype_admission,
    _restore_index,
)
from openecon.resources import plan_workspace

FLOAT = torch.float64
EPS = torch.finfo(FLOAT).eps
SCHEMA = "openecon.finite-mixture.gaussian.v1"
RESULT_SCHEMA = "openecon.finite-mixture.result.v1"
DEFAULT_WORK = 5_000_000_000
MAX_WORK = 200_000_000_000
MAX_JSON = 64 * 1024**2
MAX_TARGETS = 512
PROVENANCE = "EM parameter iterates and all numerical endpoints are replayed; bounded Newton value histories are optimizer provenance, not regenerated optimizer paths"


def _error(code, message):
    raise AnalysisError(code, message)


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        _error("system_budget", f"{name} must be an integer in {low}..{high}.")


def _real(value, low, high, name):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        _error("invalid_option", f"{name} must be finite in {low}..{high}.")


def _plan(n, d, k, settings):
    for value, low, high, name in (
        (n, 3, 20000, "rows"),
        (d, 1, 33, "design columns"),
        (k, 1, 4, "components"),
        (settings["restarts"], 1, 32, "restarts"),
        (settings["max_em_iterations"], 1, 1000, "EM iterations"),
        (settings["max_polish_iterations"], 1, 200, "Newton iterations"),
        (settings["max_work"], 1, MAX_WORK, "max_work"),
        (settings["seed"], 0, 2**63 - 1, "seed"),
    ):
        _integer(value, low, high, name)
    _real(settings["sigma_min"], torch.finfo(FLOAT).tiny ** 0.25, 1e100, "sigma_min")
    _real(settings["tolerance"], 1e-12, 1e-3, "tolerance")
    _real(settings["alpha"], 1e-8, 1 - 1e-8, "alpha")
    q = k * d + 2 * k - 1
    if q > 160 or n < max(d + 1, k * (d + 2)):
        _error(
            "insufficient_sample",
            "The bounded mixture requires n >= max(d+1,K*(d+2)) and at most 160 free parameters.",
        )
    s, em, polish = (
        settings["restarts"],
        settings["max_em_iterations"],
        settings["max_polish_iterations"],
    )
    em_pass = 32 * n * k * d * d + 32 * k * d**3 + 32 * n * k
    louis_pass = 64 * n * k * q * q + 32 * q**3
    # At most 40 exact-likelihood line trials per Newton iteration; every start
    # endpoint and optimizer-free replay is included, not only the winning one.
    work = s * ((em + 2) * em_pass + (41 * polish + 8) * louis_pass) + 8 * louis_pass
    if work > settings["max_work"]:
        _error(
            "system_budget",
            f"Complete EM/joint-information work {work} exceeds max_work={settings['max_work']}.",
        )
    serialized = n * (d + 7 * k + 6) * 96 + s * (em + 1) * (q + 6) * 96 + q * q * 320 + 6 * 1024**2
    if serialized > MAX_JSON:
        _error(
            "metadata_limit",
            "Complete source/trajectories/numerical state exceed the 64 MiB portable-state plan.",
        )
    record = plan_workspace(
        "Gaussian mixture EM, Louis information and complete replay",
        {
            "source, complete design and typed identities": n * (d + 6) * 96 + 4 * 1024**2,
            "responsibilities, residuals and weighted QR": n * k * (d + 8) * 32 + k * d * d * 64,
            "chunked complete/missing information and score": min(n, 256) * k * q * 24 + q * q * 96,
            "all recorded EM paths and bounded optimizer provenance": s * (em + 1) * (q + 6) * 96
            + s * (polish + 1) * 128,
            "portable JSON and tables": serialized * 4,
        },
    ).record()
    record.update(
        estimated_work=work,
        max_work=settings["max_work"],
        free_parameters=q,
        estimated_json_bytes=serialized,
    )
    return record


def _tensor_admit(x, y):
    if (
        not isinstance(x, torch.Tensor)
        or not isinstance(y, torch.Tensor)
        or x.ndim != 2
        or y.ndim != 1
        or len(x) != len(y)
        or any(v.dtype != FLOAT or v.device.type != "cpu" for v in (x, y))
    ):
        _error("invalid_data", "Use resident CPU float64 design [n,d] and response [n].")
    return len(y), x.shape[1]


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        _error(
            "nonfinite_data",
            f"{name} must be finite; this complete-sample mixture does not delete missing rows.",
        )


def _qr(x, y, weights=None):
    if weights is not None:
        root = weights.sqrt()
        x, y = x * root[:, None], y * root
    # Unit-equilibrated rank admission; no normal-equation inverse or ridge.
    norms = torch.linalg.vector_norm(x, dim=0)
    if bool((norms == 0).any()):
        _error("rank_deficient", "Every component weighted design must have full rank.")
    normalized = x / norms
    q, r = torch.linalg.qr(normalized, mode="reduced")
    singular = torch.linalg.svdvals(r)
    if float(singular[-1]) <= 128 * EPS * max(x.shape) * float(singular[0]):
        _error(
            "rank_deficient",
            "Component weighted design rank is unresolved in its own column units.",
        )
    return torch.linalg.solve_triangular(r, (q.T @ y)[:, None], upper=True)[:, 0] / norms


class GaussianMixture:
    def __init__(self, d, k, sigma_min):
        self.d, self.k, self.sigma_min = d, k, sigma_min
        self.q = k * d + 2 * k - 1
        self.scale_indices = list(range(k * d, k * d + k))

    def unpack(self, theta):
        if (
            not isinstance(theta, torch.Tensor)
            or theta.dtype != FLOAT
            or theta.device.type != "cpu"
            or tuple(theta.shape) != (self.q,)
        ):
            _error(
                "invalid_parameter",
                "Supply the full CPU float64 beta/log-scale/mixing-logit chart.",
            )
        _finite(theta, "mixture chart")
        beta = theta[: self.k * self.d].reshape(self.k, self.d)
        sigma = theta[self.k * self.d : self.k * self.d + self.k].exp()
        pi = torch.softmax(
            torch.cat((theta[self.k * self.d + self.k :], torch.zeros(1, dtype=FLOAT))), 0
        )
        if bool((sigma < self.sigma_min * (1 - 16 * EPS)).any()) or bool((pi <= 0).any()):
            _error(
                "parameter_domain",
                "All mixing weights must be positive and scales must satisfy the declared sigma_min.",
            )
        _finite(sigma, "component scales")
        return beta, sigma, pi

    def pack(self, beta, sigma, pi):
        return torch.cat((beta.reshape(-1), sigma.log(), (pi[:-1] / pi[-1]).log()))

    def physical(self, theta):
        beta, sigma, pi = self.unpack(theta)
        return torch.cat((beta.reshape(-1), sigma, pi))

    def jacobian(self, theta):
        _, sigma, pi = self.unpack(theta)
        j = torch.zeros((self.q + 1, self.q), dtype=FLOAT)
        j[: self.k * self.d, : self.k * self.d] = torch.eye(self.k * self.d, dtype=FLOAT)
        j[
            self.k * self.d : self.k * self.d + self.k, self.k * self.d : self.k * self.d + self.k
        ] = torch.diag(sigma)
        if self.k > 1:
            j[-self.k :, -self.k + 1 :] = (torch.diag(pi) - torch.outer(pi, pi))[:, :-1]
        return j

    def evaluate(self, x, y, theta, *, derivatives=False):
        beta, sigma, pi = self.unpack(theta)
        mu = x @ beta.T
        residual = y[:, None] - mu
        scaled = residual / sigma
        log_component = -0.5 * math.log(2 * math.pi) - sigma.log() - 0.5 * scaled.square()
        log_joint = log_component + pi.log()
        log_density = torch.logsumexp(log_joint, 1)
        responsibility = (log_joint - log_density[:, None]).exp()
        _finite(log_density, "row joint likelihood")
        mixture_mean = mu @ pi
        output = dict(
            log_likelihood=log_density.sum(),
            log_density=log_density,
            density=log_density.exp(),
            responsibility=responsibility,
            component_mean=mu,
            component_log_density=log_component,
            prior=pi.expand(len(y), -1),
            mean=mixture_mean,
            variance=(sigma.square() + (mu - mixture_mean[:, None]).square()) @ pi,
        )
        if not derivatives:
            return output
        g = torch.zeros(self.q, dtype=FLOAT)
        h = torch.zeros((self.q, self.q), dtype=FLOAT)
        # Louis: E[complete Hessian] + Cov(complete score | observed row).
        # Chunking bounds n*K*q score memory; all off-diagonal blocks survive.
        mixing_hessian = -(torch.diag(pi) - torch.outer(pi, pi))[:-1, :-1]
        for lo in range(0, len(y), 256):
            hi = min(lo + 256, len(y))
            xx = x[lo:hi]
            rr = residual[lo:hi]
            zz = responsibility[lo:hi]
            length = hi - lo
            complete = torch.zeros((length, self.k, self.q), dtype=FLOAT)
            for component in range(self.k):
                sl = slice(component * self.d, (component + 1) * self.d)
                si = self.k * self.d + component
                complete[:, component, sl] = (
                    xx * (rr[:, component] / sigma[component].square())[:, None]
                )
                complete[:, component, si] = (
                    rr[:, component].square() / sigma[component].square() - 1
                )
                if self.k > 1:
                    complete[:, component, -self.k + 1 :] = -pi[:-1]
                    if component < self.k - 1:
                        complete[:, component, -self.k + 1 + component] += 1
                w = zz[:, component] / sigma[component].square()
                h[sl, sl] -= xx.T @ (xx * w[:, None])
                cross = -2 * (xx.T @ (w * rr[:, component]))
                h[sl, si] += cross
                h[si, sl] += cross
                h[si, si] -= 2 * (w * rr[:, component].square()).sum()
            if self.k > 1:
                h[-self.k + 1 :, -self.k + 1 :] += length * mixing_hessian
            row_score = (zz[:, :, None] * complete).sum(1)
            g += row_score.sum(0)
            flat = complete.reshape(-1, self.q)
            h += flat.T @ (flat * zz.reshape(-1, 1)) - row_score.T @ row_score
        _finite(g, "joint score")
        _finite(h, "joint Hessian")
        output.update(score=g, information=-(h + h.T) / 2)
        return output

    def em_update(self, x, y, theta):
        old = self.evaluate(x, y, theta)
        z = old["responsibility"]
        mass = z.sum(0)
        if bool((mass <= 0).any()):
            _error(
                "component_collapse",
                "A component has zero posterior mass; no empty component is hidden or reset.",
            )
        beta = torch.stack([_qr(x, y, z[:, j]) for j in range(self.k)])
        residual = y[:, None] - x @ beta.T
        variance = (z * residual.square()).sum(0) / mass
        # This maximum is the exact constrained M-step, not numerical jitter.
        sigma = variance.sqrt().maximum(torch.full_like(variance, self.sigma_min))
        next_theta = self.pack(beta, sigma, mass / len(y))
        return next_theta, old


def _information(information):
    diagonal = information.diagonal()
    if bool((diagonal <= 0).any()):
        _error(
            "weak_identification",
            "The full observed information has nonpositive coordinate curvature.",
        )
    scale = diagonal.rsqrt()
    normalized = information * scale[:, None] * scale[None, :]
    eigen = torch.linalg.eigvalsh(normalized)
    if float(eigen[0]) <= max(128 * EPS * len(eigen), 1e-10) * float(eigen[-1]):
        _error(
            "weak_identification",
            "Full joint information is singular or too weakly resolved for interior inference.",
        )
    chol = torch.linalg.cholesky(normalized)
    covariance = torch.cholesky_inverse(chol) * scale[:, None] * scale[None, :]
    return (covariance + covariance.T) / 2, float(eigen[0] / eigen[-1])


def _polish(model, x, y, theta, settings):
    history = []
    tolerance = max(settings["tolerance"] ** 2, 1e-10)
    minimum = math.log(model.sigma_min)
    for iteration in range(settings["max_polish_iterations"] + 1):
        out = model.evaluate(x, y, theta, derivatives=True)
        value = float(out["log_likelihood"])
        history.append(value)
        g, info = out["score"], out["information"]
        active = [
            j
            for j in model.scale_indices
            if float(theta[j]) <= minimum + 64 * EPS * max(1, abs(minimum)) and float(g[j]) <= 0
        ]
        free = [j for j in range(model.q) if j not in active]
        try:
            covariance, _ = _information(info[free][:, free])
        except AnalysisError:
            return theta, dict(
                status="em_fixed_point",
                reason="weak_identification",
                iterations=iteration,
                values=history,
                scaled_score=None,
                active_scale_indices=active,
            )
        step = covariance @ g[free]
        scaled = float(g[free] @ step)
        if scaled <= tolerance:
            return theta, dict(
                status="projected_score",
                reason=None,
                iterations=iteration,
                values=history,
                scaled_score=scaled,
                active_scale_indices=active,
            )
        if iteration == settings["max_polish_iterations"]:
            _error(
                "nonconvergence",
                "Native constrained Newton polishing exhausted its declared iterations.",
            )
        direction = torch.zeros_like(theta)
        direction[free] = step
        accepted = False
        for trial in range(40):
            candidate = theta + direction * (2.0**-trial)
            candidate[model.scale_indices] = candidate[model.scale_indices].clamp_min(minimum)
            try:
                other = model.evaluate(x, y, candidate, derivatives=True)
            except AnalysisError:
                continue
            new_value = float(other["log_likelihood"])
            # A late indistinguishable objective is accepted only with a lower
            # projected score; no downhill endpoint is chosen for convenience.
            band = 128 * EPS * max(len(y), abs(value), 1.0)
            if new_value >= value and (
                new_value > value + band
                or float(other["score"].square().sum()) < float(g.square().sum())
            ):
                theta, accepted = candidate, True
                break
        if not accepted:
            _error(
                "nonconvergence",
                "Constrained native Newton could not admit a likelihood-increasing trial.",
            )
    raise AssertionError("bounded Newton loop")


def _start(model, x, y, index, seed):
    base = _qr(x, y)
    residual = y - x @ base
    scale = float(residual.square().mean().sqrt())
    if scale <= model.sigma_min:
        scale = model.sigma_min
    if model.k == 1:
        return model.pack(
            base[None], torch.tensor([scale], dtype=FLOAT), torch.ones(1, dtype=FLOAT)
        )
    generator = torch.Generator(device="cpu").manual_seed((seed + 104729 * index) % (2**63))
    order = (
        torch.argsort(y, stable=True) if index == 0 else torch.randperm(len(y), generator=generator)
    )
    beta, sigma = [], []
    for j in range(model.k):
        rows = order[j * len(y) // model.k : (j + 1) * len(y) // model.k]
        b = _qr(x[rows], y[rows])
        beta.append(b)
        s = float((y[rows] - x[rows] @ b).square().mean().sqrt())
        sigma.append(max(s, model.sigma_min))
    return model.pack(
        torch.stack(beta),
        torch.tensor(sigma, dtype=FLOAT),
        torch.full((model.k,), 1 / model.k, dtype=FLOAT),
    )


def _canonical(model, theta):
    beta, sigma, pi = model.unpack(theta)
    permutation = sorted(
        range(model.k), key=lambda j: (*beta[j].tolist(), float(sigma[j]), float(pi[j]))
    )
    return model.pack(beta[permutation], sigma[permutation], pi[permutation]), permutation


def _inference(model, x, y, theta, settings):
    out = model.evaluate(x, y, theta, derivatives=True)
    beta, sigma, pi = model.unpack(theta)
    reasons = []
    if bool((sigma <= model.sigma_min * (1 + 128 * EPS)).any()):
        reasons.append("active_sigma_bound")
    if bool((pi <= 1e-8).any()):
        reasons.append("near_zero_membership")
    covariance = None
    resolution = None
    try:
        covariance, resolution = _information(out["information"])
        scaled = float(out["score"] @ covariance @ out["score"])
        if scaled > max(settings["tolerance"] ** 2, 1e-10):
            reasons.append("unresolved_endpoint_score")
    except AnalysisError:
        reasons.append("weak_identification")
        scaled = None
    j = model.jacobian(theta)
    return dict(
        available=not reasons,
        reasons=reasons,
        score=out["score"],
        information=out["information"],
        chart_covariance=covariance if not reasons else None,
        jacobian=j,
        covariance=j @ covariance @ j.T if not reasons else None,
        information_resolution=resolution,
        scaled_score=scaled,
        reference="normal" if not reasons else None,
        df=None,
        covariance_kind="joint observed information" if not reasons else None,
    )


@_cpu_call
def fit_gaussian_mixture(
    x,
    y,
    *,
    components=2,
    sigma_min,
    starts=None,
    restarts=4,
    seed=1729,
    max_em_iterations=100,
    max_polish_iterations=25,
    tolerance=1e-8,
    alpha=0.05,
    max_work=DEFAULT_WORK,
):
    n, d = _tensor_admit(x, y)
    if starts is not None:
        if type(starts) is not list or not 1 <= len(starts) <= 32:
            _error(
                "invalid_start",
                "Explicit starts require a bounded plain list of physical parameter objects.",
            )
        restarts = len(starts)
    settings = dict(
        restarts=restarts,
        seed=seed,
        max_em_iterations=max_em_iterations,
        max_polish_iterations=max_polish_iterations,
        tolerance=tolerance,
        alpha=alpha,
        sigma_min=sigma_min,
        max_work=max_work,
    )
    admission = _plan(n, d, components, settings)
    if starts is not None:
        _json_admission(starts, limit=1024**2)
    x, y = x.detach(), y.detach()
    _finite(x, "design")
    _finite(y, "response")
    _qr(x, y)
    model = GaussianMixture(d, components, sigma_min)
    records = []
    for index in range(restarts):
        item = dict(
            start=index,
            seed=(seed + 104729 * index) % (2**63),
            initial=None,
            status="failed",
            failure=None,
            em_path=[],
            polish=None,
            terminal=None,
        )
        try:
            theta = (
                _physical_start(model, starts[index])
                if starts is not None
                else _start(model, x, y, index, seed)
            )
            item["initial"] = theta.tolist()
            value = float(model.evaluate(x, y, theta)["log_likelihood"])
            item["em_path"].append(dict(theta=theta.tolist(), log_likelihood=value))
            converged = False
            for iteration in range(max_em_iterations):
                candidate, _ = model.em_update(x, y, theta)
                new_value = float(model.evaluate(x, y, candidate)["log_likelihood"])
                band = 128 * EPS * max(n, abs(value), 1.0)
                if new_value < value - band:
                    _error(
                        "nonmonotone_em", "Exact constrained EM decreased the observed likelihood."
                    )
                step = float(((candidate - theta) / (1 + theta.abs())).abs().max())
                item["em_path"].append(dict(theta=candidate.tolist(), log_likelihood=new_value))
                theta = candidate
                if abs(new_value - value) <= tolerance * (1 + abs(value)) and step <= math.sqrt(
                    tolerance
                ):
                    converged = True
                    break
                value = new_value
            if not converged:
                _error(
                    "nonconvergence",
                    "EM exhausted its declared iterations before its likelihood/fixed-point criteria.",
                )
            theta, polish = _polish(model, x, y, theta, settings)
            item["polish"] = polish
            item["terminal"] = theta.tolist()
            item["log_likelihood"] = float(model.evaluate(x, y, theta)["log_likelihood"])
            item["status"] = "converged"
        except AnalysisError as error:
            item["failure"] = dict(code=error.code, message=str(error))
        records.append(item)
    successes = [i for i, item in enumerate(records) if item["status"] == "converged"]
    if not successes:
        error = AnalysisError(
            "all_starts_failed", "Every declared mixture start failed; no attempt was omitted."
        )
        error.start_diagnostics = _json_value(records)
        raise error
    chosen = max(successes, key=lambda i: (records[i]["log_likelihood"], -i))
    theta, permutation = _canonical(model, torch.tensor(records[chosen]["terminal"], dtype=FLOAT))
    state = dict(
        schema=SCHEMA,
        components=components,
        design_columns=d,
        settings=settings,
        initialization=dict(
            kind="explicit" if starts is not None else "local_seeded", physical_starts=starts
        ),
        x=x.tolist(),
        y=y.tolist(),
        starts=records,
        chosen_start=chosen,
        label_permutation=permutation,
        theta=theta,
        parameters=model.physical(theta),
        output=model.evaluate(x, y, theta),
        inference=_inference(model, x, y, theta, settings),
        admission=admission,
        optimizer_provenance=PROVENANCE,
        attempted_starts=restarts,
        converged_starts=len(successes),
        failed_starts=restarts - len(successes),
    )
    state = _json_value(state)
    _json_admission(state)
    state["digest"] = _digest(state)
    return state


def _physical_start(model, record):
    if type(record) is not dict or set(record) != {"coefficients", "scales", "weights"}:
        _error("invalid_start", "Start keys are coefficients[K,d], scales[K], weights[K].")
    _json_admission(record)
    beta = _numeric_tensor(record["coefficients"], (model.k, model.d), "start coefficients")
    sigma = _numeric_tensor(record["scales"], (model.k,), "start scales")
    pi = _numeric_tensor(record["weights"], (model.k,), "start membership")
    if bool((pi <= 0).any()) or abs(float(pi.sum()) - 1) > 64 * EPS * model.k:
        _error("invalid_start", "Start membership must be strictly positive and sum to one.")
    theta = model.pack(beta, sigma, pi)
    model.unpack(theta)
    return theta


def _numeric_cells(value, shape, name, *, floats_only=False):
    def visit(item, depth=0):
        if depth == len(shape):
            if (
                type(item) not in ((float,) if floats_only else (int, float))
                or not math.isfinite(item)
                or isinstance(item, int)
                and int(float(item)) != item
            ):
                _error(
                    "invalid_state",
                    f"{name} needs finite, exactly representable real numeric cells.",
                )
        elif type(item) is not list or len(item) != shape[depth]:
            _error("invalid_state", f"{name} has invalid bounded shape.")
        else:
            for child in item:
                visit(child, depth + 1)

    visit(value)


def _numeric_tensor(value, shape, name):
    _numeric_cells(value, shape, name)
    return torch.tensor(value, dtype=FLOAT, device="cpu")


def _same(saved, expected, path="state"):
    if type(expected) is dict:
        if type(saved) is not dict or set(saved) != set(expected):
            _error("invalid_state", f"{path} keys do not reproduce the declared model.")
        for key in expected:
            _same(saved[key], expected[key], f"{path}.{key}")
    elif type(expected) is list:
        if type(saved) is not list or len(saved) != len(expected):
            _error("invalid_state", f"{path} array geometry is invalid.")
        for i, (a, b) in enumerate(zip(saved, expected)):
            _same(a, b, f"{path}[{i}]")
    elif type(expected) is float:
        if type(saved) is not float or not math.isfinite(saved):
            _error("invalid_state", f"{path} needs its exact finite numeric type.")
        # Structural zeros (including the K=1 fixed mixing weight) stay exact;
        # a dimensionful minimum-normal floor would admit subnormal forgeries.
        bound = 4096 * EPS * max(abs(saved), abs(expected))
        if abs(saved - expected) > bound:
            _error("invalid_state", f"{path} disagrees with optimizer-free numerical replay.")
    elif type(saved) is not type(expected) or saved != expected:
        _error("invalid_state", f"{path} value or type is invalid.")


def _cached_admission(state, n, d, k, settings):
    """Validate every numerical cache before tensors, QR or likelihood replay."""
    q = k * d + 2 * k - 1

    def numeric(value, shape, name):
        _numeric_cells(value, shape, name, floats_only=True)

    for key, shape in (("x", (n, d)), ("y", (n,)), ("theta", (q,)), ("parameters", (q + 1,))):
        numeric(state[key], shape, f"cached {key}")
    _integer(state["chosen_start"], 0, settings["restarts"] - 1, "cached chosen start")
    if (
        type(state["label_permutation"]) is not list
        or len(state["label_permutation"]) != k
        or any(type(v) is not int for v in state["label_permutation"])
        or sorted(state["label_permutation"]) != list(range(k))
    ):
        _error("invalid_state", "Cached labels require a complete integer permutation.")
    for key in ("attempted_starts", "converged_starts", "failed_starts"):
        _integer(state[key], 0, settings["restarts"], f"cached {key}")
    output_shapes = {
        "log_likelihood": (),
        "log_density": (n,),
        "density": (n,),
        "responsibility": (n, k),
        "component_mean": (n, k),
        "component_log_density": (n, k),
        "prior": (n, k),
        "mean": (n,),
        "variance": (n,),
    }
    output = state["output"]
    if type(output) is not dict or set(output) != set(output_shapes):
        _error("invalid_state", "Cached numerical output has invalid keys.")
    for key, shape in output_shapes.items():
        numeric(output[key], shape, f"cached output.{key}")
    inference = state["inference"]
    inference_keys = {
        "available",
        "reasons",
        "score",
        "information",
        "chart_covariance",
        "jacobian",
        "covariance",
        "information_resolution",
        "scaled_score",
        "reference",
        "df",
        "covariance_kind",
    }
    reason_keys = {
        "active_sigma_bound",
        "near_zero_membership",
        "unresolved_endpoint_score",
        "weak_identification",
    }
    if (
        type(inference) is not dict
        or set(inference) != inference_keys
        or type(inference["available"]) is not bool
        or type(inference["reasons"]) is not list
        or len(inference["reasons"]) > len(reason_keys)
        or any(type(v) is not str or v not in reason_keys for v in inference["reasons"])
        or len(set(inference["reasons"])) != len(inference["reasons"])
        or inference["available"] != (not inference["reasons"])
        or inference["df"] is not None
    ):
        _error("invalid_state", "Cached joint inference has invalid keys/types/availability.")
    for key, shape in (("score", (q,)), ("information", (q, q)), ("jacobian", (q + 1, q))):
        numeric(inference[key], shape, f"cached inference.{key}")
    for key in ("information_resolution", "scaled_score"):
        if inference[key] is not None:
            numeric(inference[key], (), f"cached inference.{key}")
    if inference["available"]:
        numeric(inference["chart_covariance"], (q, q), "cached inference.chart_covariance")
        numeric(inference["covariance"], (q + 1, q + 1), "cached inference.covariance")
        if (
            inference["information_resolution"] is None
            or inference["scaled_score"] is None
            or inference["reference"] != "normal"
            or inference["covariance_kind"] != "joint observed information"
        ):
            _error("invalid_state", "Available joint inference requires its full interior record.")
    elif any(
        inference[key] is not None
        for key in (
            "chart_covariance",
            "covariance",
            "reference",
            "covariance_kind",
        )
    ):
        _error("invalid_state", "Unavailable joint inference cannot retain interior covariance.")
    initialization = state["initialization"]
    if type(initialization) is not dict or set(initialization) != {"kind", "physical_starts"}:
        _error("invalid_state", "Saved initialization protocol is invalid.")
    if initialization["kind"] == "explicit":
        if (
            type(initialization["physical_starts"]) is not list
            or len(initialization["physical_starts"]) != settings["restarts"]
        ):
            _error("invalid_state", "Every explicit physical start must be retained.")
    elif initialization["kind"] != "local_seeded" or initialization["physical_starts"] is not None:
        _error("invalid_state", "Saved local initialization protocol is invalid.")
    records = state["starts"]
    if type(records) is not list or len(records) != settings["restarts"]:
        _error("invalid_state", "Every declared start must be retained.")
    base_keys = {"start", "seed", "initial", "status", "failure", "em_path", "polish", "terminal"}
    for item in records:
        if type(item) is not dict or set(item) not in (base_keys, base_keys | {"log_likelihood"}):
            _error("invalid_state", "A saved start has invalid keys.")
        _integer(item["start"], 0, settings["restarts"] - 1, "cached start number")
        _integer(item["seed"], 0, 2**63 - 1, "cached start seed")
        if (
            item["status"] not in ("converged", "failed")
            or type(item["em_path"]) is not list
            or len(item["em_path"]) > settings["max_em_iterations"] + 1
        ):
            _error("invalid_state", "Saved start status/history is invalid.")
        if item["initial"] is not None:
            numeric(item["initial"], (q,), "cached initial chart")
        for point in item["em_path"]:
            if type(point) is not dict or set(point) != {"theta", "log_likelihood"}:
                _error("invalid_state", "Cached EM point needs a complete chart and likelihood.")
            numeric(point["theta"], (q,), "cached EM chart")
            numeric(point["log_likelihood"], (), "cached EM likelihood")
        if item["status"] == "converged":
            if item["failure"] is not None or set(item) != base_keys | {"log_likelihood"}:
                _error(
                    "invalid_state", "A converged start requires its complete numerical endpoint."
                )
            numeric(item["terminal"], (q,), "cached terminal chart")
            numeric(item["log_likelihood"], (), "cached terminal likelihood")
            polish = item["polish"]
            if type(polish) is not dict or set(polish) != {
                "status",
                "reason",
                "iterations",
                "values",
                "scaled_score",
                "active_scale_indices",
            }:
                _error("invalid_state", "Cached Newton endpoint provenance has invalid keys.")
            _integer(
                polish["iterations"],
                0,
                settings["max_polish_iterations"],
                "cached Newton iterations",
            )
            numeric(polish["values"], (polish["iterations"] + 1,), "cached Newton values")
            if polish["scaled_score"] is not None:
                numeric(polish["scaled_score"], (), "cached Newton scaled score")
            if (
                polish["status"] not in ("projected_score", "em_fixed_point")
                or polish["reason"] not in (None, "weak_identification")
                or type(polish["active_scale_indices"]) is not list
                or len(polish["active_scale_indices"]) > k
                or any(
                    type(v) is not int or not k * d <= v < k * d + k
                    for v in polish["active_scale_indices"]
                )
                or len(set(polish["active_scale_indices"])) != len(polish["active_scale_indices"])
            ):
                _error(
                    "invalid_state", "Cached Newton endpoint status/active coordinates are invalid."
                )
        else:
            failure = item["failure"]
            if (
                type(failure) is not dict
                or set(failure) != {"code", "message"}
                or any(
                    type(failure[key]) is not str or not failure[key] for key in ("code", "message")
                )
                or item["terminal"] is not None
                or item["polish"] is not None
                or "log_likelihood" in item
            ):
                _error("invalid_state", "Failed-start provenance cannot masquerade as an endpoint.")


def _polish_provenance(record, model, x, y, terminal, settings, em_terminal):
    keys = {"status", "reason", "iterations", "values", "scaled_score", "active_scale_indices"}
    if type(record) is not dict or set(record) != keys:
        _error("invalid_state", "Native constrained Newton provenance has invalid keys.")
    _integer(record["iterations"], 0, settings["max_polish_iterations"], "saved Newton iterations")
    if (
        type(record["values"]) is not list
        or len(record["values"]) != record["iterations"] + 1
        or any(type(v) is not float or not math.isfinite(v) for v in record["values"])
        or any(b < a for a, b in zip(record["values"][:-1], record["values"][1:]))
    ):
        _error("invalid_state", "Saved Newton history must be bounded, finite and nondecreasing.")
    initial_value = float(model.evaluate(x, y, em_terminal)["log_likelihood"])
    out = model.evaluate(x, y, terminal, derivatives=True)
    _same(record["values"][0], initial_value, "Newton initial objective")
    _same(record["values"][-1], float(out["log_likelihood"]), "Newton terminal objective")
    minimum = math.log(model.sigma_min)
    active = [
        j
        for j in model.scale_indices
        if float(terminal[j]) <= minimum + 64 * EPS * max(1, abs(minimum))
        and float(out["score"][j]) <= 0
    ]
    _same(record["active_scale_indices"], active, "active scale indices")
    free = [j for j in range(model.q) if j not in active]
    try:
        covariance, _ = _information(out["information"][free][:, free])
    except AnalysisError:
        if (
            record["status"] != "em_fixed_point"
            or record["reason"] != "weak_identification"
            or record["scaled_score"] is not None
        ):
            _error(
                "invalid_state",
                "Unresolved endpoint information requires explicit nonregular provenance.",
            )
        _same(terminal.tolist(), em_terminal.tolist(), "nonregular EM endpoint")
    else:
        scaled = float(out["score"][free] @ covariance @ out["score"][free])
        _same(record["scaled_score"], scaled, "Newton scaled score")
        if (
            record["status"] != "projected_score"
            or record["reason"] is not None
            or scaled > max(settings["tolerance"] ** 2, 1e-10)
        ):
            _error(
                "invalid_state",
                "A converged regular endpoint must reproduce its projected joint score.",
            )


@_cpu_call
def restore_gaussian_mixture_state(value):
    state = _load_record(value)
    keys = {
        "schema",
        "components",
        "design_columns",
        "settings",
        "initialization",
        "x",
        "y",
        "starts",
        "chosen_start",
        "label_permutation",
        "theta",
        "parameters",
        "output",
        "inference",
        "admission",
        "optimizer_provenance",
        "attempted_starts",
        "converged_starts",
        "failed_starts",
        "digest",
    }
    if (
        set(state) != keys
        or state["schema"] != SCHEMA
        or state["optimizer_provenance"] != PROVENANCE
    ):
        _error("invalid_state", "Use the complete versioned Gaussian-mixture state.")
    unsigned = {key: item for key, item in state.items() if key != "digest"}
    if type(state["digest"]) is not str or state["digest"] != _digest(unsigned):
        _error("invalid_state", "Mixture state digest is invalid.")
    settings = state["settings"]
    expected_settings = {
        "restarts",
        "seed",
        "max_em_iterations",
        "max_polish_iterations",
        "tolerance",
        "alpha",
        "sigma_min",
        "max_work",
    }
    if (
        type(settings) is not dict
        or set(settings) != expected_settings
        or type(state["y"]) is not list
    ):
        _error("invalid_state", "Saved mixture settings or response geometry are invalid.")
    n, d, k = len(state["y"]), state["design_columns"], state["components"]
    admission = _plan(n, d, k, settings)
    # A different caller workspace ceiling may be higher or lower; re-admit
    # now, then verify the original plan's numerical quantities independently.
    old_plan = state["admission"]
    if type(old_plan) is not dict or set(old_plan) != set(admission):
        _error("invalid_state", "Saved admission keys are invalid.")
    for key in set(admission) - {"budget_bytes"}:
        _same(old_plan[key], admission[key], f"admission.{key}")
    _integer(
        old_plan["budget_bytes"],
        old_plan["estimated_workspace_bytes"],
        MAX_WORK,
        "original workspace budget",
    )
    _cached_admission(state, n, d, k, settings)
    x = _numeric_tensor(state["x"], (n, d), "saved design")
    y = _numeric_tensor(state["y"], (n,), "saved response")
    _qr(x, y)
    model = GaussianMixture(d, k, settings["sigma_min"])
    initialization = state["initialization"]
    if type(initialization) is not dict or set(initialization) != {"kind", "physical_starts"}:
        _error("invalid_state", "Saved initialization protocol is invalid.")
    if initialization["kind"] == "explicit":
        if (
            type(initialization["physical_starts"]) is not list
            or len(initialization["physical_starts"]) != settings["restarts"]
        ):
            _error("invalid_state", "Every explicit physical start must be retained.")
    elif initialization["kind"] != "local_seeded" or initialization["physical_starts"] is not None:
        _error(
            "invalid_state", "Use an explicit physical-start or deterministic local-seed protocol."
        )
    records = state["starts"]
    if type(records) is not list or len(records) != settings["restarts"]:
        _error("invalid_state", "Every declared start must be retained.")
    successes = []
    for index, item in enumerate(records):
        base_keys = {
            "start",
            "seed",
            "initial",
            "status",
            "failure",
            "em_path",
            "polish",
            "terminal",
        }
        if type(item) is not dict or set(item) not in (base_keys, base_keys | {"log_likelihood"}):
            _error("invalid_state", "A saved start has invalid keys.")
        _same(item["start"], index, "start number")
        _same(item["seed"], (settings["seed"] + 104729 * index) % (2**63), "local start seed")
        if (
            item["status"] not in {"converged", "failed"}
            or type(item["em_path"]) is not list
            or len(item["em_path"]) > settings["max_em_iterations"] + 1
        ):
            _error("invalid_state", "Saved start status/history is invalid.")
        try:
            initial = (
                _physical_start(model, initialization["physical_starts"][index])
                if initialization["kind"] == "explicit"
                else _start(model, x, y, index, settings["seed"])
            )
        except AnalysisError as error:
            if (
                item["initial"] is not None
                or item["status"] != "failed"
                or type(item["failure"]) is not dict
                or item["failure"].get("code") != error.code
            ):
                _error(
                    "invalid_state", "Failed initialization does not reproduce its declared input."
                )
        else:
            _same(item["initial"], initial.tolist(), "declared initialization")
        if item["initial"] is None:
            if item["em_path"] or item["status"] != "failed":
                _error("invalid_state", "An uninitialized start cannot have fitted iterates.")
        else:
            theta = _numeric_tensor(item["initial"], (model.q,), "saved initial chart")
            model.unpack(theta)
            for iteration, point in enumerate(item["em_path"]):
                if type(point) is not dict or set(point) != {"theta", "log_likelihood"}:
                    _error(
                        "invalid_state",
                        "EM iterates need complete parameter and objective records.",
                    )
                if iteration:
                    theta, _ = model.em_update(x, y, theta)
                _same(point["theta"], theta.tolist(), "exact EM iterate")
                _same(
                    point["log_likelihood"],
                    float(model.evaluate(x, y, theta)["log_likelihood"]),
                    "exact EM likelihood",
                )
        if item["status"] == "converged":
            if (
                item["failure"] is not None
                or set(item) != base_keys | {"log_likelihood"}
                or len(item["em_path"]) < 2
            ):
                _error(
                    "invalid_state",
                    "A converged start requires a complete endpoint and successful EM record.",
                )
            previous, last = item["em_path"][-2:]
            old_theta = _numeric_tensor(previous["theta"], (model.q,), "previous EM chart")
            step = float(((theta - old_theta) / (1 + old_theta.abs())).abs().max())
            if abs(last["log_likelihood"] - previous["log_likelihood"]) > settings["tolerance"] * (
                1 + abs(previous["log_likelihood"])
            ) or step > math.sqrt(settings["tolerance"]):
                _error("invalid_state", "Saved EM stopping criteria do not reproduce convergence.")
            terminal = _numeric_tensor(item["terminal"], (model.q,), "saved terminal chart")
            _polish_provenance(item["polish"], model, x, y, terminal, settings, theta)
            _same(
                item["log_likelihood"],
                float(model.evaluate(x, y, terminal)["log_likelihood"]),
                "start terminal likelihood",
            )
            successes.append(index)
        else:
            failure = item["failure"]
            if (
                type(failure) is not dict
                or set(failure) != {"code", "message"}
                or type(failure["code"]) is not str
                or type(failure["message"]) is not str
                or not failure["code"]
                or not failure["message"]
                or item["terminal"] is not None
                or item["polish"] is not None
                or "log_likelihood" in item
            ):
                _error(
                    "invalid_state",
                    "Failed-start provenance must be bounded and distinct from a fitted endpoint.",
                )
    if not successes:
        _error("invalid_state", "A saved fitted mixture requires at least one converged start.")
    chosen = max(successes, key=lambda i: (records[i]["log_likelihood"], -i))
    _same(state["chosen_start"], chosen, "chosen best constrained objective")
    theta, permutation = _canonical(
        model, _numeric_tensor(records[chosen]["terminal"], (model.q,), "winning terminal chart")
    )
    _same(state["label_permutation"], permutation, "canonical labels")
    _same(state["theta"], theta.tolist(), "canonical chart")
    _same(state["parameters"], model.physical(theta).tolist(), "full physical parameters")
    _same(state["output"], _json_value(model.evaluate(x, y, theta)), "cached numerical output")
    _same(
        state["inference"],
        _json_value(_inference(model, x, y, theta, settings)),
        "full joint inference",
    )
    for key, expected in (
        ("attempted_starts", len(records)),
        ("converged_starts", len(successes)),
        ("failed_starts", len(records) - len(successes)),
    ):
        _same(state[key], expected, key)
    return state


def _probe(data, names):
    import numpy as np
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        if (
            data.shape[1] > 1024
            or not data.columns.is_unique
            or any(name not in data for name in names)
        ):
            _error(
                "invalid_data",
                "Project distinct existing columns from at most 1024 resident source columns.",
            )
        return len(data), data, data.index
    if not isinstance(data, Mapping) or any(name not in data for name in names):
        _error(
            "unsupported_input",
            "Use a resident DataFrame or bounded column mapping; Dataset and iterators are unsupported.",
        )
    projected = {name: data[name] for name in names}
    allowed = (list, tuple, range, pd.Series, pd.Index, np.ndarray)
    if any(
        not isinstance(v, allowed) or isinstance(v, np.ndarray) and v.ndim != 1
        for v in projected.values()
    ):
        _error(
            "unsupported_input",
            "Columns must be resident one-dimensional arrays; no generator is consumed.",
        )
    lengths = [len(v) for v in projected.values()]
    if len(set(lengths)) != 1:
        _error("invalid_data", "All declared columns require identical resident lengths.")
    series = [v for v in projected.values() if isinstance(v, pd.Series)]
    if series and any(not series[0].index.equals(v.index) for v in series[1:]):
        _error("invalid_data", "Series-backed columns require identical row identities.")
    return lengths[0], projected, series[0].index if series else pd.RangeIndex(lengths[0])


def _source(data, y, predictors, intercept, settings, components):
    import pandas as pd
    from openecon import analysis

    if (
        type(y) is not str
        or not y
        or len(y) > 256
        or type(predictors) is not list
        or len(predictors) > 32
        or any(type(name) is not str or not name or len(name) > 256 for name in predictors)
        or len(set([y, *predictors])) != len(predictors) + 1
        or type(intercept) is not bool
    ):
        _error(
            "invalid_data",
            "Declare one distinct outcome and up to32 distinct named predictors with boolean intercept.",
        )
    names = [y, *predictors]
    n, projected, index = _probe(data, names)
    _plan(n, len(predictors) + int(intercept), components, settings)
    _index_admission(index)
    for name in names:
        for cell in projected[name]:
            if isinstance(cell, bool) or not isinstance(cell, Real):
                _error(
                    "non_numeric_column",
                    "Complete Gaussian mixture columns require real numeric values.",
                )
            try:
                converted = float(cell)
            except (ValueError, OverflowError):
                _error("precision_loss", "Source values exceed finite float64.")
            if not math.isfinite(converted):
                _error(
                    "missing_values",
                    "Mixture data require complete finite rows; none are silently dropped.",
                )
            if isinstance(cell, Integral) and int(converted) != int(cell):
                _error(
                    "precision_loss",
                    "Integer source values must be exactly representable in float64.",
                )
    frame = analysis._coerce_frame(
        projected.loc[:, names] if isinstance(projected, pd.DataFrame) else projected
    )
    rows = frame[names].astype("float64").values.tolist()
    dtypes = [str(frame[name].dtype) for name in names]
    _response_dtype_admission(dtypes, rows)
    source = dict(
        names=names,
        response=y,
        predictors=predictors,
        dtypes=dtypes,
        values=rows,
        index=_index_record(frame.index),
        positions=list(range(n)),
        sample=[True] * n,
        intercept=intercept,
        sample_assumption="independent complete rows; fixed K conditional regression likelihood",
    )
    _json_admission(source, limit=16 * 1024**2)
    source["digest"] = _digest(source)
    tensor = torch.tensor(rows, dtype=FLOAT)
    x = tensor[:, 1:]
    if intercept:
        x = torch.cat((torch.ones((n, 1), dtype=FLOAT), x), 1)
    return source, x, tensor[:, 0]


def _terms(state, source):
    names = (["_cons"] if source["intercept"] else []) + source["predictors"]
    k = state["components"]
    return (
        [f"component[{j + 1}]:{name}" for j in range(k) for name in names]
        + [f"scale[{j + 1}]" for j in range(k)]
        + [f"weight[{j + 1}]" for j in range(k)]
    )


def _tables(record):
    state, source = record["state"], record["source"]
    n, k, d = len(state["y"]), state["components"], state["design_columns"]
    model = GaussianMixture(d, k, state["settings"]["sigma_min"])
    theta = torch.tensor(state["theta"], dtype=FLOAT)
    params = torch.tensor(state["parameters"], dtype=FLOAT)
    terms = _terms(state, source)
    inference = state["inference"]
    available = inference["available"]
    critical = float(
        torch.distributions.Normal(
            torch.tensor(0.0, dtype=FLOAT), torch.tensor(1.0, dtype=FLOAT)
        ).icdf(torch.tensor(1 - state["settings"]["alpha"] / 2, dtype=FLOAT))
    )
    rows = []
    if available:
        covariance = torch.tensor(inference["covariance"], dtype=FLOAT)
        chart_cov = torch.tensor(inference["chart_covariance"], dtype=FLOAT)
        jacobian = model.jacobian(theta)
    for i, term in enumerate(terms):
        estimate = float(params[i])
        se = z = p = low = high = None
        if available:
            se = float(covariance[i, i].sqrt())
            if i < k * d:
                z = estimate / se
                p = float(torch.erfc(torch.tensor(abs(z) / math.sqrt(2), dtype=FLOAT)))
                low, high = estimate - critical * se, estimate + critical * se
            elif i < k * d + k:
                chart_se = float(chart_cov[i, i].sqrt())
                low, high = (
                    math.exp(float(theta[i]) - critical * chart_se),
                    math.exp(float(theta[i]) + critical * chart_se),
                )
            elif k > 1:
                j = i - k * d - k
                pi = estimate
                delta = jacobian[i] / (pi * (1 - pi))
                logit_se = float((delta @ chart_cov @ delta).sqrt())
                logit = math.log(pi / (1 - pi))
                low = float(torch.sigmoid(torch.tensor(logit - critical * logit_se, dtype=FLOAT)))
                high = float(torch.sigmoid(torch.tensor(logit + critical * logit_se, dtype=FLOAT)))
            else:
                se, low, high = 0.0, 1.0, 1.0
        rows.append([estimate, se, z, p, low, high])
    coeff = table(
        rows,
        columns=["estimate", "std_error", "z", "pvalue", "ci_lower", "ci_upper"],
        index=terms,
        reference=inference["reference"],
        df=None,
        weight_zero_test="unavailable: class-count boundary null",
    )
    index = _restore_index(source["index"], n)
    out = state["output"]
    membership = {
        "observed": state["y"],
        "mean": out["mean"],
        "conditional_variance": out["variance"],
        "log_density": out["log_density"],
    }
    for j in range(k):
        membership[f"prior[{j + 1}]"] = [row[j] for row in out["prior"]]
        membership[f"posterior[{j + 1}]"] = [row[j] for row in out["responsibility"]]
        membership[f"component_mean[{j + 1}]"] = [row[j] for row in out["component_mean"]]
    starts = table(
        [
            [
                v["start"],
                v["status"],
                len(v["em_path"]) - 1,
                v.get("log_likelihood"),
                v["failure"]["code"] if v["failure"] else None,
            ]
            for v in state["starts"]
        ],
        columns=["start", "status", "em_iterations", "log_likelihood", "failure"],
    )
    covariance_table = table(
        inference["covariance"] if available else [[None] * len(terms) for _ in terms],
        columns=terms,
        index=terms,
        available=available,
        reasons=inference["reasons"],
        simplex_constraint="sum(weights)=1",
    )
    membership_table = table(membership, index=index)
    membership_table.index = index.copy()
    return MixtureResult(
        {
            "parameters": coeff,
            "covariance": covariance_table,
            "membership": membership_table,
            "starts": starts,
        },
        title="Fixed-K Gaussian regression mixture",
        **record,
    )


class MixtureResult(TableSet):
    @_cpu_call
    def to_json(self, *, indent=None):
        restored = finite_mixture_restore(self)
        return json.dumps(restored.attrs, indent=indent, allow_nan=False)


@_cpu_call
def finite_mixture(
    data,
    y,
    x,
    *,
    components=2,
    sigma_min,
    intercept=True,
    starts=None,
    restarts=4,
    seed=1729,
    max_em_iterations=100,
    max_polish_iterations=25,
    tolerance=1e-8,
    alpha=0.05,
    max_work=DEFAULT_WORK,
    device="cpu",
    weights=None,
    missing="raise",
) -> MixtureResult:
    if device != "cpu" or weights is not None or missing != "raise":
        _error(
            "unsupported_domain",
            "This stage admits CPU complete unweighted rows; weight, missing-drop and device extensions have separate contracts.",
        )
    if starts is not None:
        if type(starts) is not list or not 1 <= len(starts) <= 32:
            _error("invalid_start", "Use a bounded list of physical starts.")
        restarts = len(starts)
        _json_admission(starts, limit=1024**2)
    settings = dict(
        restarts=restarts,
        seed=seed,
        max_em_iterations=max_em_iterations,
        max_polish_iterations=max_polish_iterations,
        tolerance=tolerance,
        alpha=alpha,
        sigma_min=sigma_min,
        max_work=max_work,
    )
    source, design, response = _source(data, y, x, intercept, settings, components)
    state = fit_gaussian_mixture(design, response, components=components, starts=starts, **settings)
    record = dict(
        schema=RESULT_SCHEMA,
        source=source,
        state=state,
        likelihood_target="fixed-K conditional Gaussian regression; explicit component sigma lower bound",
        inference_available=state["inference"]["available"],
        inference_reasons=state["inference"]["reasons"],
        log_likelihood=state["output"]["log_likelihood"],
        df=None,
        notes=[
            "The highest converged constrained objective is retained; multi-start does not prove a global maximum.",
            "Prior membership conditions on X; posterior membership additionally conditions on observed y.",
            PROVENANCE,
        ],
    )
    _json_admission(record)
    record["digest"] = _digest(record)
    return _tables(record)


@_cpu_call
def finite_mixture_restore(result) -> MixtureResult:
    record = _load_record(result.attrs if isinstance(result, MixtureResult) else result)
    keys = {
        "schema",
        "source",
        "state",
        "likelihood_target",
        "inference_available",
        "inference_reasons",
        "log_likelihood",
        "df",
        "notes",
        "digest",
    }
    if set(record) != keys or record["schema"] != RESULT_SCHEMA:
        _error("invalid_state", "Use a complete public Gaussian-mixture result.")
    if record["digest"] != _digest(
        {key: value for key, value in record.items() if key != "digest"}
    ):
        _error("invalid_state", "Public mixture digest is invalid.")
    state = restore_gaussian_mixture_state(record["state"])
    source = record["source"]
    source_keys = {
        "names",
        "response",
        "predictors",
        "dtypes",
        "values",
        "index",
        "positions",
        "sample",
        "intercept",
        "sample_assumption",
        "digest",
    }
    if type(source) is not dict or set(source) != source_keys:
        _error("invalid_state", "The complete typed source is required for replay.")
    if source["digest"] != _digest(
        {key: value for key, value in source.items() if key != "digest"}
    ):
        _error("invalid_state", "Source digest is invalid.")
    n, d = len(state["y"]), state["design_columns"]
    names, predictors = source["names"], source["predictors"]
    if (
        type(source["intercept"]) is not bool
        or type(source["response"]) is not str
        or not source["response"]
        or len(source["response"]) > 256
        or type(predictors) is not list
        or len(predictors) + int(source["intercept"]) != d
        or any(type(v) is not str or not v or len(v) > 256 for v in predictors)
        or type(names) is not list
        or names != [source["response"], *predictors]
        or len(set(names)) != len(names)
        or type(source["dtypes"]) is not list
        or len(source["dtypes"]) != len(names)
    ):
        _error("invalid_state", "Saved source columns/types/intercept are invalid.")
    raw = _numeric_tensor(source["values"], (n, len(names)), "typed source")
    _response_dtype_admission(source["dtypes"], source["values"])
    _restore_index(source["index"], n)
    _same(source["positions"], list(range(n)), "source positions")
    _same(source["sample"], [True] * n, "complete sample")
    _same(
        source["sample_assumption"],
        "independent complete rows; fixed K conditional regression likelihood",
        "sample assumption",
    )
    x = raw[:, 1:]
    if source["intercept"]:
        x = torch.cat((torch.ones((n, 1), dtype=FLOAT), x), 1)
    _same(state["x"], x.tolist(), "source design")
    _same(state["y"], raw[:, 0].tolist(), "source response")
    _same(
        record["inference_available"],
        state["inference"]["available"],
        "public inference availability",
    )
    _same(record["inference_reasons"], state["inference"]["reasons"], "public inference reasons")
    _same(record["log_likelihood"], state["output"]["log_likelihood"], "public likelihood")
    _same(record["df"], None, "normal inference df")
    expected_notes = [
        "The highest converged constrained objective is retained; multi-start does not prove a global maximum.",
        "Prior membership conditions on X; posterior membership additionally conditions on observed y.",
        PROVENANCE,
    ]
    _same(record["notes"], expected_notes, "public scientific notes")
    _same(
        record["likelihood_target"],
        "fixed-K conditional Gaussian regression; explicit component sigma lower bound",
        "likelihood target",
    )
    return _tables(record)


def _predict_value(model, design, theta, target, observed=None):
    beta, sigma, pi = model.unpack(theta)
    mu = design @ beta.T
    mean = mu @ pi
    if target == "mean":
        return mean
    if target == "variance":
        return (sigma.square() + (mu - mean[:, None]).square()) @ pi
    if target == "component_mean":
        return mu.reshape(-1)
    if target == "prior":
        return pi.expand(len(design), -1).reshape(-1)
    scaled = (observed[:, None] - mu) / sigma
    if target == "cdf":
        return (0.5 * torch.erfc(-scaled / math.sqrt(2))) @ pi
    joint = -0.5 * math.log(2 * math.pi) - sigma.log() - 0.5 * scaled.square() + pi.log()
    log_density = torch.logsumexp(joint, 1)
    if target == "log_density":
        return log_density
    if target == "density":
        return log_density.exp()
    if target == "posterior":
        return (joint - log_density[:, None]).exp().reshape(-1)
    raise AssertionError("validated prediction target")


def _quantiles(model, design, theta, probability):
    beta, sigma, pi = model.unpack(theta)
    mu = design @ beta.T
    z = torch.distributions.Normal(
        torch.tensor(0.0, dtype=FLOAT), torch.tensor(1.0, dtype=FLOAT)
    ).icdf(torch.tensor(probability, dtype=FLOAT))
    component = mu + sigma * z
    lower, upper = component.min(1).values, component.max(1).values
    for _ in range(96):
        middle = lower / 2 + upper / 2
        cdf = (0.5 * torch.erfc(-(middle[:, None] - mu) / sigma / math.sqrt(2))) @ pi
        lower = torch.where(cdf < probability, middle, lower)
        upper = torch.where(cdf >= probability, middle, upper)
    value = lower / 2 + upper / 2
    residual = _predict_value(model, design, theta, "cdf", value) - probability
    if float(residual.abs().max()) > 256 * EPS:
        _error(
            "prediction_precision",
            "Mixture quantiles cannot be resolved at the requested float64 CDF tolerance.",
        )
    return value


@_cpu_call
def finite_mixture_predict(
    result,
    data,
    *,
    target="mean",
    y=None,
    probability=None,
    parameter_uncertainty=True,
    alpha=None,
    max_work=DEFAULT_WORK,
) -> TableSet:
    import pandas as pd
    from openecon import analysis

    targets = {
        "mean",
        "variance",
        "component_mean",
        "prior",
        "posterior",
        "log_density",
        "density",
        "cdf",
        "quantile",
    }
    if target not in targets or type(parameter_uncertainty) is not bool:
        _error(
            "invalid_prediction", "Use a declared mixture target and boolean parameter_uncertainty."
        )
    record = _load_record(result.attrs if isinstance(result, MixtureResult) else result)
    if (
        record.get("schema") != RESULT_SCHEMA
        or type(record.get("state")) is not dict
        or type(record.get("source")) is not dict
    ):
        _error("invalid_state", "Prediction requires the complete public saved mixture.")
    state, source = record["state"], record["source"]
    if type(source.get("predictors")) is not list or type(state.get("settings")) is not dict:
        _error("invalid_state", "Saved prediction geometry is invalid.")
    predictors = source["predictors"]
    if any(type(v) is not str for v in predictors) or len(predictors) > 32:
        _error("invalid_state", "Saved predictors are invalid.")
    requires_y = target in {"posterior", "log_density", "density", "cdf"}
    if requires_y:
        if type(y) is not str or y in predictors or not y or len(y) > 256:
            _error(
                "invalid_prediction",
                "This conditional target requires an explicit distinct observed-y/threshold column.",
            )
    elif y is not None:
        _error(
            "invalid_prediction",
            "Unconditional targets must not consume a supplied future outcome.",
        )
    if target == "quantile":
        _real(probability, 1e-8, 1 - 1e-8, "quantile probability")
    elif probability is not None:
        _error("invalid_prediction", "probability is only a quantile option.")
    names = predictors + ([y] if requires_y else [])
    n, projected, index = _probe(data, names)
    _integer(n, 1, 20000, "prediction rows")
    _integer(max_work, 1, MAX_WORK, "prediction max_work")
    k, d = state.get("components"), state.get("design_columns")
    fit_plan = _plan(len(state.get("y", [])), d, k, state["settings"])
    q = fit_plan["free_parameters"]
    cells = n * k if target in {"component_mean", "prior", "posterior"} else n
    if cells > MAX_TARGETS:
        _error("prediction_budget", "Full joint prediction admits at most512 target cells.")
    work = (
        fit_plan["estimated_work"] + 512 * n * k * q * q * d + 64 * cells * cells * q + 256 * n * k
    )
    if work > max_work:
        _error("prediction_budget", "Complete state replay and joint prediction exceed max_work.")
    admission = plan_workspace(
        "Gaussian mixture full joint saved prediction",
        {
            "complete fit replay": fit_plan["estimated_workspace_bytes"],
            "prediction source, Jacobian and joint covariance": n * (d + 4) * 96
            + cells * q * 64
            + cells * cells * 64
            + 2 * 1024**2,
            "bounded prediction autodiff": n * k * q * d * 256,
        },
    ).record()
    admission.update(estimated_work=work, max_work=max_work, target_cells=cells)
    _index_admission(index)
    # The prediction budget is admitted before numeric scans or DataFrame copy.
    for name in names:
        for cell in projected[name]:
            if isinstance(cell, bool) or not isinstance(cell, Real):
                _error("non_numeric_column", "Prediction cells must be real numeric.")
            try:
                converted = float(cell)
            except (ValueError, OverflowError):
                _error("precision_loss", "Prediction values exceed finite float64.")
            if not math.isfinite(converted):
                _error("missing_values", "Prediction columns must be complete finite rows.")
            if isinstance(cell, Integral) and int(converted) != int(cell):
                _error(
                    "precision_loss",
                    "Prediction integers must be exactly representable in float64.",
                )
    restored = finite_mixture_restore(record)
    state, source = restored.attrs["state"], restored.attrs["source"]
    if parameter_uncertainty and not state["inference"]["available"]:
        _error(
            "unavailable_inference",
            "This best constrained fit has no regular joint parameter covariance; request fixed-parameter predictions explicitly.",
        )
    frame = analysis._coerce_frame(
        projected.loc[:, names] if isinstance(projected, pd.DataFrame) else projected
    )
    if not names:
        # A zero-predictor model still has the caller's explicit resident rows.
        frame = pd.DataFrame(index=index)
    values = frame[names].astype("float64").values.tolist()
    dtypes = [str(frame[name].dtype) for name in names]
    _response_dtype_admission(dtypes, values)
    query = torch.tensor(frame[predictors].astype("float64").values.tolist(), dtype=FLOAT).reshape(
        n, len(predictors)
    )
    if source["intercept"]:
        query = torch.cat((torch.ones((n, 1), dtype=FLOAT), query), 1)
    observed = (
        torch.tensor(frame[y].astype("float64").tolist(), dtype=FLOAT) if requires_y else None
    )
    theta = torch.tensor(state["theta"], dtype=FLOAT)
    model = GaussianMixture(d, k, state["settings"]["sigma_min"])
    value = (
        _quantiles(model, query, theta, probability)
        if target == "quantile"
        else _predict_value(model, query, theta, target, observed)
    )
    covariance = torch.zeros((cells, cells), dtype=FLOAT)
    jacobian = torch.zeros((cells, q), dtype=FLOAT)
    if parameter_uncertainty:
        with torch.enable_grad():
            if target == "quantile":
                jacobian = torch.autograd.functional.jacobian(
                    lambda z: _predict_value(model, query, z, "cdf", value.detach()), theta
                )
                density = _predict_value(model, query, theta, "density", value.detach())
                if bool((density <= 0).any()):
                    _error(
                        "prediction_precision",
                        "Quantile delta information requires a resolved positive mixture density.",
                    )
                jacobian = -jacobian / density[:, None]
            else:
                jacobian = torch.autograd.functional.jacobian(
                    lambda z: _predict_value(model, query, z, target, observed), theta
                )
        chart_cov = torch.tensor(state["inference"]["chart_covariance"], dtype=FLOAT)
        covariance = jacobian @ chart_cov @ jacobian.T
        covariance = (covariance + covariance.T) / 2
    _finite(value, "prediction target")
    _finite(covariance, "full target covariance")
    alpha = state["settings"]["alpha"] if alpha is None else alpha
    _real(alpha, 1e-8, 1 - 1e-8, "prediction alpha")
    critical = float(
        torch.distributions.Normal(
            torch.tensor(0.0, dtype=FLOAT), torch.tensor(1.0, dtype=FLOAT)
        ).icdf(torch.tensor(1 - alpha / 2, dtype=FLOAT))
    )
    multi = target in {"component_mean", "prior", "posterior"}
    rows, labels = [], []
    for cell in range(cells):
        position, component = (cell // k, cell % k + 1) if multi else (cell, None)
        estimate = float(value[cell])
        se = low = high = None
        if parameter_uncertainty:
            variance = float(covariance[cell, cell])
            if variance < 0:
                _error("prediction_precision", "Target variance is unresolved; it is not clipped.")
            se = math.sqrt(variance)
            if target in {"prior", "posterior", "cdf"} and 0 < estimate < 1:
                logit = math.log(estimate / (1 - estimate))
                spread = critical * se / (estimate * (1 - estimate))
                low = float(torch.sigmoid(torch.tensor(logit - spread, dtype=FLOAT)))
                high = float(torch.sigmoid(torch.tensor(logit + spread, dtype=FLOAT)))
            elif target == "variance" and estimate > 0:
                low, high = (
                    math.exp(math.log(estimate) - critical * se / estimate),
                    math.exp(math.log(estimate) + critical * se / estimate),
                )
            else:
                low, high = estimate - critical * se, estimate + critical * se
        labels.append(f"row[{position}]:component[{component}]" if multi else f"row[{position}]")
        rows.append([position, component, estimate, se, low, high])
    query_source = dict(
        names=names,
        dtypes=dtypes,
        values=values,
        index=_index_record(frame.index),
        positions=list(range(n)),
    )
    attrs = dict(
        schema="openecon.finite-mixture.prediction.v1",
        model_digest=record["digest"],
        target=target,
        conditioning="X and explicitly supplied observed y"
        if target == "posterior"
        else "X; density/CDF evaluate an explicit outcome argument",
        parameter_uncertainty=parameter_uncertainty,
        uncertainty="joint frequentist delta covariance"
        if parameter_uncertainty
        else "fixed fitted parameters",
        predictive_distribution="fixed-parameter Gaussian mixture; no Bayesian parameter integration",
        alpha=alpha,
        probability=probability,
        source=query_source,
        values=value.tolist(),
        jacobian=jacobian.tolist(),
        covariance=covariance.tolist(),
        admission=admission,
    )
    _json_admission(attrs)
    attrs["digest"] = _digest(attrs)
    return TableSet(
        {
            "prediction": table(
                rows,
                columns=["position", "component", "estimate", "std_error", "ci_lower", "ci_upper"],
                index=labels,
            ),
            "covariance": table(covariance.tolist(), columns=labels, index=labels),
        },
        title="Saved Gaussian mixture prediction",
        **attrs,
    )
