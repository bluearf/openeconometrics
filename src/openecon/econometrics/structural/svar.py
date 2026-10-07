"""Exactly identified impact restrictions and Haar sign-restricted SVAR sets."""

from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.advanced_common import regular_frame
from openecon.econometrics.core import build_result, column_list, kernel_call, make_spec, table
from openecon.econometrics.structural.connectedness import _ma
from openecon.econometrics.var import kernels
from openecon.econometrics.var.estimators import var
from openecon.econometrics.var.postestimation import _system
from openecon.engines.optimize import maximize_bfgs
from openecon.models import ResultBundle


def _rotation(theta, k):
    q = torch.eye(k, dtype=torch.float64)
    index = 0
    for i in range(k):
        for j in range(i + 1, k):
            c, s = theta[index].cos(), theta[index].sin()
            left, right = q[:, i].clone(), q[:, j].clone()
            columns = list(q.unbind(1))
            columns[i], columns[j] = c * left + s * right, -s * left + c * right
            q = torch.stack(columns, 1)
            index += 1
    return q


def _restriction_list(short, long, k):
    restrictions = []
    for name, matrix in (("short_run", short), ("long_run", long)):
        if matrix is None:
            continue
        if (
            not isinstance(matrix, list)
            or len(matrix) != k
            or any(not isinstance(row, list) or len(row) != k for row in matrix)
        ):
            raise AnalysisError(
                "invalid_restrictions", f"{name} must be a K-by-K list of zero or null entries."
            )
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0:
                    raise AnalysisError(
                        "invalid_restrictions",
                        "Exact restrictions use zero; null means free. Nonzero constraints are outside this impact normalization.",
                    )
                restrictions.append((name, i, j))
    return restrictions


def _long_multiplier(a):
    k = a.shape[1]
    if max(abs(complex(v["real"], v["imag"])) for v in kernels.companion_eigenvalues(a)) >= 1:
        raise AnalysisError("unstable_long_run", "Long-run SVAR restrictions need a stable VAR.")
    matrix = torch.eye(k, dtype=torch.float64) - a.sum(0)
    if float(torch.linalg.svdvals(matrix).min()) <= 1e-10 * float(
        torch.linalg.svdvals(matrix).max()
    ):
        raise AnalysisError("unidentified_long_run", "The long-run multiplier is singular.")
    return torch.linalg.solve(matrix, torch.eye(k, dtype=torch.float64))


def _normalize(impact, long_multiplier, restrictions):
    # Column signs are observationally equivalent for zero restrictions.
    anchor = long_multiplier @ impact if any(a[0] == "long_run" for a in restrictions) else impact
    signs, anchors = [], []
    for j in range(len(impact)):
        row = j if abs(float(anchor[j, j])) > 1e-10 else int(torch.argmax(anchor[:, j].abs()))
        value = float(anchor[row, j])
        if abs(value) <= 1e-10:
            raise AnalysisError("unidentified_sign", "A shock has no nonzero sign anchor.")
        signs.append(1.0 if value > 0 else -1.0)
        anchors.append(row)
    return impact * torch.tensor(signs, dtype=torch.float64)[None, :], anchors


def _identify(a, sigma, short, long, seed):
    k = sigma.shape[0]
    restrictions = _restriction_list(short, long, k)
    q = k * (k - 1) // 2
    if len(restrictions) != q:
        raise AnalysisError(
            "unidentified_svar",
            f"An exactly identified impact system needs {q} independent zero restrictions; got {len(restrictions)}.",
        )
    f = _long_multiplier(a) if long is not None else torch.eye(k, dtype=torch.float64)
    p = kernel_call(kernels.spd_factor, sigma)

    def residual(theta):
        impact = p @ _rotation(theta, k)
        lf = f @ impact
        return torch.stack(
            [impact[i, j] if kind == "short_run" else lf[i, j] for kind, i, j in restrictions]
        )

    @torch.enable_grad()
    def objective(theta):
        t = theta.detach().requires_grad_(True)
        value = -0.5 * residual(t).square().sum()
        return value.detach(), torch.autograd.grad(value, t)[0].detach()

    def hessian(theta):
        return torch.autograd.functional.hessian(lambda t: -0.5 * residual(t).square().sum(), theta)

    generator = torch.Generator().manual_seed(seed)
    for attempt in range(20):
        initial = (
            torch.zeros(q, dtype=torch.float64)
            if attempt == 0
            else (torch.rand(q, generator=generator, dtype=torch.float64) - 0.5) * 2 * math.pi
        )
        try:
            solution = kernel_call(
                maximize_bfgs,
                objective,
                initial,
                hessian_fn=hessian,
                max_iter=500,
                gradient_tol=1e-10,
                scaled_gradient_tol=1e-12,
            )
        except AnalysisError as error:
            if error.code not in (
                "nonconvergence",
                "singular_information",
                "nonconcave_information",
                "nonconcave_objective",
            ):
                raise
            continue
        if float(residual(solution.theta).abs().max()) > 1e-7 * max(float(p.abs().max()), 1):
            continue
        jac = torch.autograd.functional.jacobian(residual, solution.theta)
        if int(torch.linalg.matrix_rank(jac, rtol=1e-8)) != q:
            raise AnalysisError(
                "unidentified_svar",
                "Zero restrictions fail the local rotation identification rank condition.",
            )
        impact, anchors = _normalize(p @ _rotation(solution.theta, k), f, restrictions)
        return impact, {
            "type": "exact_zero",
            "restriction_count": q,
            "rotation_rank": q,
            "seed": seed,
            "attempt": attempt,
            "sign_anchor_rows": anchors,
            "max_restriction_error": float(residual(solution.theta).abs().max()),
            "converged": solution.converged,
        }
    raise AnalysisError(
        "infeasible_restrictions",
        "No identified rotation satisfying these restrictions was found in the declared 20-start protocol.",
    )


def _sign_draws(a, sigma, names, signs, seed, draws, accepted):
    k = len(names)
    if not isinstance(signs, list) or not signs:
        raise AnalysisError(
            "invalid_restrictions",
            "signs must be a nonempty list of horizon/response/shock/sign records.",
        )
    parsed = []
    for r in signs:
        if not isinstance(r, dict) or set(r) != {"horizon", "response", "shock", "sign"}:
            raise AnalysisError(
                "invalid_restrictions", "A sign record needs horizon, response, shock and sign."
            )
        h = r["horizon"]
        if (
            isinstance(h, bool)
            or not isinstance(h, int)
            or not 0 <= h <= 200
            or r["sign"] not in (-1, 1)
            or isinstance(r["sign"], bool)
            or r["response"] not in names
            or r["shock"] not in names
        ):
            raise AnalysisError(
                "invalid_restrictions",
                "Use integer horizons 0..200, known variable names and signs -1 or +1.",
            )
        parsed.append((h, names.index(r["response"]), names.index(r["shock"]), r["sign"]))
    phi = _ma(a, max(r[0] for r in parsed) + 1)
    p = kernel_call(kernels.spd_factor, sigma)
    generator = torch.Generator().manual_seed(seed)
    impacts = []
    for attempt in range(draws):
        q, r = torch.linalg.qr(torch.randn((k, k), generator=generator, dtype=torch.float64))
        q *= torch.where(r.diagonal() >= 0, 1.0, -1.0)[None, :]
        impact = p @ q
        responses = phi @ impact
        if all(float(responses[h, i, j]) * sign > 0 for h, i, j, sign in parsed):
            impacts.append(impact)
            if len(impacts) == accepted:
                break
    if len(impacts) < accepted:
        raise AnalysisError(
            "insufficient_admissible_draws",
            f"Only {len(impacts)} admissible Haar rotations in {draws} declared draws; requested {accepted}.",
        )
    return impacts, {
        "type": "sign_identified_set",
        "seed": seed,
        "attempted": attempt + 1,
        "accepted": len(impacts),
        "rotation_distribution": "Haar on O(K), QR diagonal signs corrected",
        "interval_meaning": "rotation-set quantiles, not sampling confidence intervals",
        "signs": signs,
    }


def fit_svar(spec, data):
    frame, _ = regular_frame(spec, data)
    names = frame.role("system")
    if not 2 <= len(names) <= 6 or spec.outcome != names[0]:
        raise AnalysisError(
            "invalid_system", "Use 2..6 endogenous variables, with the first one as outcome."
        )
    result = var(
        data=frame.sample,
        y=names,
        time=spec.time,
        lags=frame.option("lags"),
        maxlag=frame.option("lags"),
        irf_steps=1,
        irf_kinds=["simple"],
        lm_lags=0,
        alpha=spec.alpha,
    )
    system = _system(result)
    short, long, signs = (frame.option(s) for s in ("short_run", "long_run", "signs"))
    if signs is not None:
        if short is not None or long is not None:
            raise AnalysisError(
                "invalid_restrictions",
                "This version separates exact zero restrictions from sign-identified sets.",
            )
        impacts, identification = _sign_draws(
            system["a"],
            system["sigma"],
            names,
            signs,
            frame.option("seed"),
            frame.option("draws"),
            frame.option("accepted"),
        )
        impact = impacts[0]
        ensemble = [c.tolist() for c in impacts]
    else:
        impact, identification = _identify(
            system["a"], system["sigma"], short, long, frame.option("seed")
        )
        ensemble = None
    original_positions = frame.positions.copy()
    keep = torch.zeros(frame.n, dtype=torch.bool)
    keep[result.sample_positions] = True
    frame.restrict(keep, "Excluded initial VAR lag boundaries without compressing the calendar.")
    assembled = build_result(
        frame,
        terms=[c.term for c in result.coefficients],
        equations=[c.equation for c in result.coefficients],
        params=torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64),
        covariance=torch.tensor(result.covariance_matrix, dtype=torch.float64),
        title="Structural vector autoregression",
        use_t=result.inference["use_t"],
        df_inference=result.inference.get("df_inference"),
        df_resid=result.inference.get("df_resid"),
        metrics=result.metrics,
        tests=result.tests,
        inference=result.inference,
        solver="float64_VAR_QR_and_rotation_identification",
        extra={
            **result.extra,
            "structural": {
                "impact": impact.tolist(),
                "identification": identification,
                "short_run": short,
                "long_run": long,
                "accepted_impacts": ensemble,
                "coefficient_table": "reduced-form VAR coefficients; identified impact matrix and structural responses stored separately",
            },
        },
    )
    predictions = [{**r, "row": original_positions[r["row"]]} for r in result.predictions]
    return assembled.model_copy(update={"predictions": predictions})


def svar(
    *,
    data,
    y,
    time=None,
    lags=1,
    short_run=None,
    long_run=None,
    signs=None,
    seed=0,
    draws=2000,
    accepted=100,
    missing="raise",
    alpha=0.05,
):
    """SVAR with K(K-1)/2 identified zero restrictions, or declared sign-rotation sets."""
    from openecon.analysis import fit

    names = column_list(y, "y")
    if not names:
        raise AnalysisError("invalid_spec", "y needs at least two endogenous variables.")
    return fit(
        make_spec(
            "svar",
            outcome=names[0],
            columns={"system": names},
            time=time,
            missing=missing,
            alpha=alpha,
            options={
                "lags": lags,
                "short_run": short_run,
                "long_run": long_run,
                "signs": signs,
                "seed": seed,
                "draws": draws,
                "accepted": accepted,
            },
        ),
        data=data,
    )


def svar_irf(result, steps=8, *, draws=200, seed=0, alpha=0.05):
    """Structural responses; exact models use joint asymptotic Gaussian/Wishart simulation.

    Sign sets return rotation quantiles conditional on the fitted reduced form.
    Exact models simulate the reduced-form coefficient covariance and innovation
    covariance jointly (Gaussian ML independence) and re-identify every draw.
    """
    if not isinstance(result, ResultBundle) or result.spec.estimator != "svar":
        raise AnalysisError("invalid_result", "svar_irf needs a saved SVAR result.")
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or not 0 <= steps <= 200
        or isinstance(draws, bool)
        or not isinstance(draws, int)
        or not 20 <= draws <= 2000
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or not 0 < alpha < 1
    ):
        raise AnalysisError(
            "invalid_option",
            "Use steps 0..200, draws 20..2000, a nonnegative seed and alpha in (0,1).",
        )
    system, state = _system(result), result.extra["structural"]
    a, sigma = system["a"], system["sigma"]
    names, k = system["names"], len(system["names"])
    phi = _ma(a, steps + 1)
    point = phi @ torch.tensor(state["impact"], dtype=torch.float64)
    if state["accepted_impacts"] is not None:
        samples = torch.stack(
            [phi @ torch.tensor(c, dtype=torch.float64) for c in state["accepted_impacts"]]
        )
        point = samples.median(0).values
        meaning = "conditional identified-set quantiles; not sampling confidence intervals"
    else:
        generator = torch.Generator().manual_seed(seed)
        covariance = torch.tensor(result.covariance_matrix, dtype=torch.float64)
        root = torch.linalg.cholesky(covariance)
        p = torch.linalg.cholesky(sigma)
        original = torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)
        n = system["n_obs"]
        samples = []
        for _ in range(draws):
            beta = original + root @ torch.randn(
                len(original), generator=generator, dtype=torch.float64
            )
            coef = beta.reshape(k, -1)
            ad = kernels.lag_matrices(coef, k, a.shape[0])
            errors = torch.randn((n, k), generator=generator, dtype=torch.float64) @ p.T
            sd = errors.T @ errors / n
            impact, _ = _identify(ad, sd, state["short_run"], state["long_run"], seed)
            samples.append(_ma(ad, steps + 1) @ impact)
        samples = torch.stack(samples)
        meaning = "joint asymptotic Gaussian VAR coefficient / Wishart innovation simulation; percentile intervals"
    low, high = (
        torch.quantile(samples, alpha / 2, dim=0),
        torch.quantile(samples, 1 - alpha / 2, dim=0),
    )
    rows = [
        [names[j], names[i], h, float(point[h, i, j]), float(low[h, i, j]), float(high[h, i, j])]
        for h in range(steps + 1)
        for i in range(k)
        for j in range(k)
    ]
    return table(
        rows,
        columns=["shock", "response", "horizon", "irf", "ci_low", "ci_high"],
        identification=state["identification"],
        seed=seed,
        draws=len(samples),
        interval_method=meaning,
        alpha=alpha,
    )
