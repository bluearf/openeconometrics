"""Native CPU float64 Weibull interval likelihood, complete OIM and delta maps."""

from __future__ import annotations

import math

import torch

from openecon.engines.contracts import KernelError
from openecon.engines.optimize import maximize_bfgs
from . import weibull_regression_common as c

DT = torch.float64


def tensor(value):
    return torch.tensor(value, dtype=DT, device="cpu")


def endpoints(source, spec):
    cols = source["columns"]
    lj, uj = cols.index(spec["lower"]), cols.index(spec["upper"])
    kinds, lo, hi = [], [], []
    for row in source["values"]:
        lower, upper = c.real(row[lj], "lower endpoint"), row[uj]
        if lower < 0 or lower > 1e12 or 0 < lower < 1e-12:
            c.fail("Lower requires zero or a finite time in [1e-12,1e12].", "invalid_interval")
        if upper is None:
            if lower == 0:
                c.fail("(0,infinity) contains no survival information.", "invalid_interval")
            kinds.append("right")
            hi.append(1.0)  # unused finite placeholder, never a substituted endpoint
        else:
            upper = c.real(upper, "upper endpoint")
            if not 1e-12 <= upper <= 1e12 or upper < lower:
                c.fail(
                    "Finite upper must lie in [1e-12,1e12] and at least lower.", "invalid_interval"
                )
            kinds.append("exact" if lower == upper else "left" if lower == 0 else "interval")
            hi.append(upper)
        lo.append(lower)
    return lo, hi, kinds


def prepare(source, spec):
    lo, hi, kinds = endpoints(source, spec)
    n, p = len(lo), len(spec["x"])
    if n < p + 3:
        c.fail(
            "An interior Weibull fit needs more rows than location/scale parameters.",
            "unidentified_model",
        )
    if all(v == "left" for v in kinds) or all(v == "right" for v in kinds):
        c.fail(
            "All-left/all-right observations have no finite interior Weibull MLE.", "no_finite_mle"
        )
    x = tensor(
        [[row[source["columns"].index(col)] for col in spec["x"]] for row in source["values"]]
    ).reshape(n, p)
    location = x.mean(0)
    centered = x - location
    scale = torch.linalg.vector_norm(centered, dim=0) / math.sqrt(n)
    if p and (not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any())):
        c.fail("Every covariate requires finite nonzero variation.", "unidentified_model")
    design = torch.cat((torch.ones((n, 1), dtype=DT, device="cpu"), centered / scale), dim=1)
    singular = torch.linalg.svdvals(design)
    ratio = float(singular[-1] / singular[0])
    if ratio < 1e-6:
        c.fail(
            "Normalized design singular-value ratio is below 1e-6; no ridge/term dropping.",
            "unidentified_model",
        )
    representatives = [
        lo[i]
        if kind == "exact"
        else hi[i] / 2
        if kind == "left"
        else lo[i] * 1.5
        if kind == "right"
        else math.sqrt(lo[i]) * math.sqrt(hi[i])
        for i, kind in enumerate(kinds)
    ]
    time_center = sum(map(math.log, representatives)) / n
    loglo = tensor([math.log(v) - time_center if v else 0.0 for v in lo])
    loghi = tensor([math.log(v) - time_center for v in hi])
    masks = {
        kind: torch.tensor([v == kind for v in kinds], dtype=torch.bool, device="cpu")
        for kind in ("exact", "left", "interval", "right")
    }
    q = p + 2
    jacobian = torch.eye(q, dtype=DT, device="cpu")
    if p:
        jacobian[0, 1 : p + 1] = -location / scale
        jacobian[1 : p + 1, 1 : p + 1] = torch.diag(scale.reciprocal())
    chart = dict(
        location=location.tolist(),
        scale=scale.tolist(),
        time_log_center=time_center,
        normalized_design_singular_ratio=ratio,
        original_jacobian=jacobian.tolist(),
    )
    return dict(
        design=design,
        loglo=loglo,
        loghi=loghi,
        masks=masks,
        kinds=kinds,
        lo=tensor(lo),
        hi=tensor(hi),
        chart=chart,
        jacobian=jacobian,
        n=n,
    )


def log_cdf_hazard(logh):
    out = torch.empty_like(logh)
    small, large = logh < -36.0, logh > math.log(745.0)
    middle = ~(small | large)
    out[small] = logh[small]
    out[large] = logh[large] * 0
    out[middle] = torch.log(-torch.expm1(-torch.exp(logh[middle])))
    return out


def log_expm1(value):
    out = torch.empty_like(value)
    large = value > 36.0
    out[large] = value[large] + torch.log1p(-torch.exp(-value[large]))
    out[~large] = torch.log(torch.expm1(value[~large]))
    return out


def contributions(theta, data):
    eta = data["design"] @ theta[:-1]
    shape = torch.exp(-theta[-1])
    out = torch.empty(data["n"], dtype=DT, device="cpu")
    for kind, mask in data["masks"].items():
        if not bool(mask.any()):
            continue
        logtime = data["loghi"][mask] if kind == "left" else data["loglo"][mask]
        z = shape * (logtime - eta[mask])
        if kind == "exact":
            out[mask] = -theta[-1] - torch.log(data["lo"][mask]) + z - torch.exp(z)
        elif kind == "left":
            out[mask] = log_cdf_hazard(z)
        elif kind == "right":
            out[mask] = -torch.exp(z)
        else:
            width = shape * torch.log1p((data["hi"][mask] - data["lo"][mask]) / data["lo"][mask])
            out[mask] = -torch.exp(z) + log_cdf_hazard(z + log_expm1(width))
    return out


def inside(theta):
    return (
        bool(torch.isfinite(theta).all())
        and bool((theta[:-1].abs() < 40).all())
        and abs(float(theta[-1])) < math.log(1000)
    )


def evaluate(theta, data):
    if not inside(theta):
        c.fail(
            "Saved/estimated normalized parameters exceed the open numerical domain.",
            "boundary_fit",
        )
    with torch.enable_grad():
        point = theta.detach().clone().requires_grad_(True)
        values = contributions(point, data)
        value = values.sum()
        if not bool(torch.isfinite(values).all()):
            c.fail("Weibull likelihood exceeds float64 support; no clipping.", "numerical_failure")
        score = torch.autograd.grad(value, point)[0]
        information = -torch.autograd.functional.hessian(
            lambda v: contributions(v, data).sum(), point
        )
    information = (information + information.T) / 2
    if not bool(torch.isfinite(score).all()) or not bool(torch.isfinite(information).all()):
        c.fail("Nonfinite full ML derivatives.", "numerical_failure")
    diagonal = information.diagonal()
    if bool((diagonal <= 0).any()):
        c.fail("Observed information is not positive definite.", "unidentified_model")
    units = diagonal.rsqrt()
    normalized = information * units[:, None] * units[None, :]
    eig = torch.linalg.eigvalsh(normalized)
    if float(eig[0]) <= 1e-8 * float(eig[-1]):
        c.fail(
            "Equilibrated full OIM is nonpositive/ill-conditioned; no ridge.", "unidentified_model"
        )
    chol = torch.linalg.cholesky(normalized)
    inv = torch.cholesky_inverse(chol)
    covariance = inv * units[:, None] * units[None, :]
    covariance = (covariance + covariance.T) / 2
    quadratic = float(score @ covariance @ score)
    if quadratic > 1e-12 or float(score.abs().max()) > 1e-7 * data["n"]:
        c.fail(
            "Saved/estimated endpoint fails the normalized full OIM score certificate.",
            "nonconvergence",
        )
    return dict(
        log_likelihood=float(value.detach()),
        contributions=values.detach().tolist(),
        score=score.detach().tolist(),
        information=information.detach().tolist(),
        covariance=covariance.detach().tolist(),
        score_quadratic=quadratic,
        equilibrated_information_eigenvalues=eig.tolist(),
    )


def fit(data, spec):
    representative = tensor(
        [
            float(data["loglo"][i])
            if kind in ("exact", "right")
            else float(data["loghi"][i]) - math.log(2)
            if kind == "left"
            else (float(data["loglo"][i]) + float(data["loghi"][i])) / 2
            for i, kind in enumerate(data["kinds"])
        ]
    )
    qr, rr = torch.linalg.qr(data["design"], mode="reduced")
    beta = torch.linalg.solve_triangular(rr, (qr.T @ representative)[:, None], upper=True)[:, 0]
    spread = max(0.2, min(3.0, float((representative - data["design"] @ beta).std(unbiased=False))))
    successes, failures = [], []

    def objective(theta):
        if not inside(theta):
            return tensor(-math.inf), torch.full_like(theta, math.nan)
        with torch.enable_grad():
            point = theta.detach().clone().requires_grad_(True)
            value = contributions(point, data).sum()
            if not bool(torch.isfinite(value)):
                return value.detach(), torch.full_like(theta, math.nan)
            score = torch.autograd.grad(value, point)[0]
        return value.detach(), score.detach()

    def hessian(theta):
        with torch.enable_grad():
            return torch.autograd.functional.hessian(
                lambda v: contributions(v, data).sum(), theta
            ).detach()

    with torch.device("cpu"):
        for multiplier in (0.7, 1.0, 1.5):
            start = torch.cat((beta, tensor([math.log(spread * multiplier)])))
            try:
                result = maximize_bfgs(
                    objective,
                    start,
                    max_iter=spec["maxiter"],
                    gradient_tol=1e-10,
                    scaled_gradient_tol=1e-13,
                    hessian_fn=hessian,
                )
                check = evaluate(result.theta, data)
                successes.append((result, check))
            except (KernelError, c.AnalysisError) as exc:
                failures.append(dict(code=exc.code, message=str(exc)[:512]))
    if not successes:
        c.fail(
            "No accepted interior Weibull ML endpoint within the declared starts/iteration budget.",
            "nonconvergence",
        )
    result, check = max(successes, key=lambda pair: pair[1]["log_likelihood"])
    provenance = dict(
        starts=3,
        accepted_starts=len(successes),
        iterations=result.iterations,
        method="bfgs_with_observed_hessian",
        failures=failures,
        global_optimum_certified=False,
    )
    return result.theta.detach().tolist(), check, provenance


def normal_quantile(level):
    level = c.real(level, "level")
    if not 0 < level < 1:
        c.fail("level requires a strict finite probability.", "invalid_argument")
    z = float(torch.special.ndtri(tensor((1 + level) / 2)))
    if not math.isfinite(z):
        c.fail("Confidence level exceeds float64 quantile resolution.", "invalid_argument")
    return z


def original(theta, data):
    raw = data["jacobian"] @ theta
    raw[0] += data["chart"]["time_log_center"]
    return raw


def ph_map(raw):
    shape = torch.exp(-raw[-1])
    return torch.cat((-shape * raw[:-1], (-raw[-1]).reshape(1)))


def inverse_chart(chart):
    scale, location = tensor(chart["scale"]), tensor(chart["location"])
    inverse = torch.eye(len(scale) + 2, dtype=DT, device="cpu")
    if len(scale):
        inverse[0, 1:-1] = location
        inverse[1:-1, 1:-1] = torch.diag(scale)
    return inverse


def parameter_result(theta, endpoint, data, spec):
    point = tensor(theta)
    raw = original(point, data)
    nc = tensor(endpoint["covariance"])
    cov = data["jacobian"] @ nc @ data["jacobian"].T
    cov = (cov + cov.T) / 2
    with torch.enable_grad():
        jac = torch.autograd.functional.jacobian(ph_map, raw)
    ph = ph_map(raw)
    combined = jac @ data["jacobian"]
    phcov = combined @ nc @ combined.T
    phcov = (phcov + phcov.T) / 2
    inverse = inverse_chart(data["chart"])
    info = inverse.T @ tensor(endpoint["information"]) @ inverse
    info = (info + info.T) / 2
    z = normal_quantile(spec["level"])

    def rows(values, covariance, names):
        out = []
        for i, name in enumerate(names):
            se = float(covariance[i, i].sqrt())
            out.append(
                dict(
                    parameter=name,
                    estimate=float(values[i]),
                    se=se,
                    lower=float(values[i]) - z * se,
                    upper=float(values[i]) + z * se,
                    level=spec["level"],
                )
            )
        return out

    return dict(
        aft_parameters=raw.tolist(),
        ph_parameters=ph.tolist(),
        aft_covariance=cov.tolist(),
        ph_covariance=phcov.tolist(),
        ph_jacobian=jac.tolist(),
        aft_information=info.tolist(),
        aft_score=(inverse.T @ tensor(endpoint["score"])).tolist(),
        sigma=float(raw[-1].exp()),
        shape=float((-raw[-1]).exp()),
        aft_table=rows(raw, cov, ["Intercept", *spec["x"], "log_sigma"]),
        ph_table=rows(ph, phcov, ["Intercept", *spec["x"], "log_shape"]),
    )


def query_values(raw, design, times, time_center=0.0):
    eta = design @ raw[:-1]
    shape = torch.exp(-raw[-1])
    loghazard = torch.zeros((len(design), len(times)), dtype=DT, device="cpu")
    positive = times > 0
    loghazard[:, positive] = shape * (times[positive].log()[None, :] - time_center - eta[:, None])
    hazard = torch.zeros_like(loghazard)
    hazard[:, positive] = torch.exp(loghazard[:, positive])
    survival = torch.exp(-hazard)
    return torch.cat((survival.flatten(), hazard.flatten())), loghazard


def prediction(theta, covariance, chart, source, times, level):
    raw, covariance = tensor(theta), tensor(covariance)
    n = len(source["values"])
    x = tensor(source["values"]).reshape(n, len(raw) - 2)
    location, scale = tensor(chart["location"]), tensor(chart["scale"])
    design = torch.cat((torch.ones((n, 1), dtype=DT, device="cpu"), (x - location) / scale), dim=1)
    tt = tensor(times)
    center = chart["time_log_center"]
    with torch.enable_grad():
        estimates, loghazard = query_values(raw, design, tt, center)
        jac = torch.autograd.functional.jacobian(
            lambda v: query_values(v, design, tt, center)[0], raw
        )
        lj = torch.autograd.functional.jacobian(
            lambda v: query_values(v, design, tt, center)[1].flatten(), raw
        )
    full = jac @ covariance @ jac.T
    full = (full + full.T) / 2
    if (
        not bool(torch.isfinite(estimates).all())
        or not bool(torch.isfinite(jac).all())
        or not bool(torch.isfinite(full).all())
    ):
        c.fail(
            "Complete query values/derivatives/covariance exceed float64; no clipping/extrapolation substitution.",
            "numerical_failure",
        )
    if bool((full.diagonal() < 0).any()):
        c.fail("Negative full delta variance.", "numerical_failure")
    logvar = (lj @ covariance * lj).sum(1)
    z = normal_quantile(level)
    count = n * len(times)
    rows = []
    for r in range(n):
        for j, time in enumerate(times):
            i = r * len(times) + j
            sv, hz = float(estimates[i]), float(estimates[count + i])
            if time == 0:
                sl = su = 1.0
                hl = hu = 0.0
            else:
                logh, se = float(loghazard[r, j]), float(logvar[i].sqrt())
                lo, hi = logh - z * se, logh + z * se
                if hi > 709 or lo < -745:
                    c.fail(
                        "Log-hazard confidence limits exceed float64; no clipping.",
                        "numerical_failure",
                    )
                hl, hu = math.exp(lo), math.exp(hi)
                sl, su = math.exp(-hu), math.exp(-hl)
            rows.append(
                dict(
                    position=r,
                    time=time,
                    survival=sv,
                    survival_se=float(full[i, i].sqrt()),
                    survival_lower=sl,
                    survival_upper=su,
                    cumulative_hazard=hz,
                    hazard_se=float(full[count + i, count + i].sqrt()),
                    hazard_lower=hl,
                    hazard_upper=hu,
                    level=level,
                )
            )
    names = [
        f"{family}[{r},{j}]"
        for family in ("survival", "cumulative_hazard")
        for r in range(n)
        for j in range(len(times))
    ]
    original_j = tensor(chart["original_jacobian"])
    inverse_j = inverse_chart(chart)
    return dict(
        rows=rows,
        component_names=names,
        estimates=estimates.tolist(),
        jacobian=(jac @ inverse_j).tolist(),
        normalized_jacobian=jac.tolist(),
        covariance=full.tolist(),
        parameter_query_covariance=(original_j @ covariance @ jac.T).tolist(),
        log_hazard_jacobian=(lj @ inverse_j).tolist(),
    )
