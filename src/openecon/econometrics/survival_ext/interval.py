"""Intercept-only iid interval-censored ML and complete saved survival inference.

Exact endpoints contribute the density in actual time units. Left, right and
finite interval endpoints contribute CDF, survival, and interval probability.
SciPy is a development oracle only; all runtime likelihoods use Torch float64.
"""

from __future__ import annotations

import math
import hashlib
import json
from numbers import Real

import torch
import torch.nn.functional as F

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.nonparametric.common import procedure
from openecon.engines.contracts import KernelError
from openecon.engines.optimize import maximize_bfgs
from .common import confidence, controls, interval_data, output, prediction_times, workspace

DT = torch.float64
DISTRIBUTIONS = ("exponential", "weibull", "lognormal", "loglogistic")
LIMIT = math.log(1000.0)


def _log_cdf_hazard(logh):
    result = torch.empty_like(logh)
    small, large = logh < -36.0, logh > math.log(745.0)
    middle = ~(small | large)
    # At these tails the omitted correction is below float64 resolution. These
    # branches avoid both exp underflow and 0*infinity derivatives of saturation.
    result[small] = logh[small]
    result[large] = logh[large] * 0
    result[middle] = torch.log(-torch.expm1(-logh[middle].exp()))
    return result


def _parts(distribution, theta, t):
    """Stable log CDF/SF/density for strictly positive actual-time values."""
    logt = t.log()
    if distribution == "exponential":
        logh = theta[0] + logt
        h = logh.exp()
        return _log_cdf_hazard(logh), -h, theta[0] - h
    if distribution == "weibull":
        shape = theta[0].exp()
        z = shape * (logt - theta[1])
        h = z.exp()
        return _log_cdf_hazard(z), -h, theta[0] - logt + z - h
    if distribution == "lognormal":
        z = (logt - theta[0]) * torch.exp(-theta[1])
        return (
            torch.special.log_ndtr(z),
            torch.special.log_ndtr(-z),
            -logt - theta[1] - 0.5 * z.square() - 0.5 * math.log(2 * math.pi),
        )
    shape = theta[0].exp()
    z = shape * (logt - theta[1])
    return -F.softplus(-z), -F.softplus(z), theta[0] - logt - F.softplus(z) - F.softplus(-z)


def _log_difference(a, b):
    """log(exp(a)-exp(b)), a>b; expm1 preserves narrow interval masses."""
    return a + torch.log(-torch.expm1(b - a))


def _log_expm1_positive(value):
    result = torch.empty_like(value)
    large = value > 36
    result[large] = value[large]
    result[~large] = torch.expm1(value[~large]).log()
    return result


def _narrow_probability(distribution, lower_z, width):
    def quadrature(nodes, weights):
        node = torch.tensor(nodes, dtype=DT, device="cpu")
        weight = torch.tensor(weights, dtype=DT, device="cpu")
        z = lower_z[:, None] + width[:, None] * ((node + 1) / 2)
        density = (
            -0.5 * z.square() - 0.5 * math.log(2 * math.pi)
            if distribution == "lognormal"
            else -F.softplus(z) - F.softplus(-z)
        )
        return width.log() + torch.logsumexp(density + weight.log() - math.log(2), dim=1)

    four = quadrature(
        [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
        [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538],
    )
    eight = quadrature(
        [
            -0.9602898564975363,
            -0.7966664774136267,
            -0.525532409916329,
            -0.1834346424956498,
            0.1834346424956498,
            0.525532409916329,
            0.7966664774136267,
            0.9602898564975363,
        ],
        [
            0.1012285362903763,
            0.2223810344533745,
            0.3137066458778873,
            0.362683783378362,
            0.362683783378362,
            0.3137066458778873,
            0.2223810344533745,
            0.1012285362903763,
        ],
    )
    if not bool(torch.isfinite(eight).all()) or bool(
        ((four - eight).abs() > 2e-12 * eight.abs().clamp_min(1)).any()
    ):
        raise AnalysisError(
            "numerical_failure", "Narrow interval 4/8-node log-probability agreement gate failed."
        )
    return eight


def _contributions(distribution, theta, data):
    values = torch.zeros_like(data.lo)
    for kind in ("exact", "left", "right", "interval"):
        mask = torch.tensor([v == kind for v in data.kind], dtype=torch.bool, device="cpu")
        if not bool(mask.any()):
            continue
        if kind == "exact":
            contribution = _parts(distribution, theta, data.lo[mask])[2]
        elif kind == "left":
            contribution = _parts(distribution, theta, data.hi[mask])[0]
        elif kind == "right":
            contribution = _parts(distribution, theta, data.lo[mask])[1]
        else:
            if distribution in ("exponential", "weibull"):
                lo, hi = data.lo[mask], data.hi[mask]
                if distribution == "exponential":
                    h_lo = torch.exp(theta[0] + lo.log())
                    log_delta_h = theta[0] + (hi - lo).log()
                else:
                    shape = theta[0].exp()
                    log_h_lo = shape * (lo.log() - theta[1])
                    h_lo = log_h_lo.exp()
                    log_delta_h = log_h_lo + _log_expm1_positive(
                        shape * torch.log1p((hi - lo) / lo)
                    )
                values[mask] = -h_lo + _log_cdf_hazard(log_delta_h)
                continue
            lo, hi = data.lo[mask], data.hi[mask]
            if distribution == "lognormal":
                lower_z = (lo.log() - theta[0]) * torch.exp(-theta[1])
                width = torch.log1p((hi - lo) / lo) * torch.exp(-theta[1])
            else:
                lower_z = (lo.log() - theta[1]) * theta[0].exp()
                width = torch.log1p((hi - lo) / lo) * theta[0].exp()
            narrow = width * (1 + lower_z.abs()) < 1e-3
            contribution = torch.empty_like(lo)
            if bool(narrow.any()):
                contribution[narrow] = _narrow_probability(
                    distribution, lower_z[narrow], width[narrow]
                )
            if not bool((~narrow).any()):
                values[mask] = contribution
                continue
            lc_lo, ls_lo, _ = _parts(distribution, theta, lo[~narrow])
            lc_hi, ls_hi, _ = _parts(distribution, theta, hi[~narrow])
            # Choose each interval's less saturated tail. Indexing rather than
            # torch.where avoids evaluating undefined derivatives in the unused
            # saturated CDF difference for high upper-tail intervals.
            lower_tail = lc_hi < -math.log(2)
            wide = torch.empty_like(lc_hi)
            if bool(lower_tail.any()):
                wide[lower_tail] = _log_difference(lc_hi[lower_tail], lc_lo[lower_tail])
            if bool((~lower_tail).any()):
                wide[~lower_tail] = _log_difference(ls_lo[~lower_tail], ls_hi[~lower_tail])
            contribution[~narrow] = wide
        values[mask] = contribution
    return values


def _inside(distribution, theta):
    if not bool(torch.isfinite(theta).all()):
        return False
    if distribution == "exponential":
        return bool((theta.abs() < 40).all())
    if distribution == "lognormal":
        return abs(float(theta[0])) < 40 and abs(float(theta[1])) < LIMIT
    return abs(float(theta[0])) < LIMIT and abs(float(theta[1])) < 40


def _names(distribution):
    if distribution == "exponential":
        return ["log_rate"], ["rate"]
    if distribution == "lognormal":
        return ["meanlog", "log_sdlog"], ["meanlog", "sdlog"]
    return ["log_shape", "log_scale"], ["shape", "scale"]


def _natural(distribution, theta):
    if distribution == "lognormal":
        return torch.stack((theta[0], theta[1].exp()))
    return theta.exp()


def _fit(distribution, data, maxiter):
    if isinstance(maxiter, bool) or not isinstance(maxiter, int) or not 1 <= maxiter <= 2000:
        raise AnalysisError("invalid_option", "maxiter must be an integer in [1,2000].")
    if data.n < (2 if distribution == "exponential" else 3):
        raise AnalysisError(
            "unidentified_model", "Too few observations for an interior distribution ML fit."
        )
    if all(v == "right" for v in data.kind) or all(v == "left" for v in data.kind):
        raise AnalysisError(
            "no_finite_mle", "All-left or all-right censoring has no finite interior MLE."
        )
    if distribution != "exponential":
        exact = [float(data.lo[i]) for i, v in enumerate(data.kind) if v == "exact"]
        if exact and max(exact) == min(exact):
            value = exact[0]
            if all(
                (kind == "exact" or (float(data.lo[i]) <= value <= float(data.hi[i])))
                for i, kind in enumerate(data.kind)
            ):
                raise AnalysisError(
                    "no_finite_mle",
                    "A common exact time permits unbounded concentration; no finite MLE.",
                )
        if not exact and float(data.lo.max()) <= float(data.hi.min()):
            raise AnalysisError(
                "no_finite_mle",
                "Common-overlap censored intervals permit boundary concentration; no finite MLE.",
            )
    representative = torch.tensor(
        [
            float(data.lo[i])
            if kind == "exact"
            else float(data.hi[i]) / 2
            if kind == "left"
            else float(data.lo[i]) * 1.5
            if kind == "right"
            else math.sqrt(float(data.lo[i])) * math.sqrt(float(data.hi[i]))
            for i, kind in enumerate(data.kind)
        ],
        dtype=DT,
        device="cpu",
    )
    center = float(representative.log().mean())
    spread = max(0.25, min(3.0, float(representative.log().std(unbiased=False))))
    starts = (
        [[-center]]
        if distribution == "exponential"
        else [[center, math.log(spread * factor)] for factor in (0.7, 1.0, 1.5)]
        if distribution == "lognormal"
        else [[math.log(shape), center] for shape in (0.7, 1.3, 2.0)]
    )

    def loglike(theta):
        return _contributions(distribution, theta, data).sum()

    def evaluate(theta):
        if not _inside(distribution, theta):
            return torch.tensor(-math.inf, dtype=DT, device="cpu"), torch.full_like(theta, math.nan)
        with torch.enable_grad():
            trial = theta.detach().clone().requires_grad_(True)
            try:
                value = loglike(trial)
            except AnalysisError:
                return torch.tensor(-math.inf, dtype=DT, device="cpu"), torch.full_like(
                    theta, math.nan
                )
            if not bool(torch.isfinite(value)):
                return value.detach(), torch.full_like(theta, math.nan)
            gradient = torch.autograd.grad(value, trial)[0]
        return value.detach(), gradient.detach()

    def hessian(theta):
        with torch.enable_grad():
            return torch.autograd.functional.hessian(loglike, theta).detach()

    accepted, failures = [], []
    # The CPU context also protects the shared optimizer's generic constructors
    # from a caller's global default device, without changing that default.
    with torch.device("cpu"):
        for start in starts:
            try:
                fit = maximize_bfgs(
                    evaluate,
                    torch.tensor(start, dtype=DT, device="cpu"),
                    max_iter=maxiter,
                    gradient_tol=1e-10,
                    scaled_gradient_tol=1e-12,
                    hessian_fn=hessian,
                )
                accepted.append(fit)
            except KernelError as error:
                failures.append(dict(code=error.code, message=str(error)))
    if not accepted:
        raise AnalysisError(
            "nonconvergence",
            "No finite stationary interval ML fit within the parameter/iteration budget.",
        )
    fit = max(accepted, key=lambda result: result.value)
    theta = fit.theta.detach()
    if not _inside(distribution, theta) or not _inside(distribution, theta * (1 + 1e-8)):
        raise AnalysisError(
            "boundary_fit", "The interval ML fit touches the supported parameter interior."
        )
    information = -(fit.hessian + fit.hessian.T) / 2
    eig = torch.linalg.eigvalsh(information)
    if not bool(torch.isfinite(information).all()) or float(eig[0]) <= max(
        1e-8, float(eig[-1]) * 1e-10
    ):
        raise AnalysisError(
            "unidentified_model",
            "Observed information is not positive definite or is ill-conditioned; no ridge.",
        )
    inverse = torch.linalg.inv(information)
    covariance = ((inverse + inverse.T) / 2).contiguous()
    score = fit.gradient.detach()
    score_gate = float(score @ covariance @ score)
    if score_gate > 1e-10 or float(score.abs().max()) > 1e-6 * max(1.0, data.n):
        raise AnalysisError(
            "nonconvergence", "Interval ML score/observed-information convergence gate failed."
        )
    return theta, covariance, information, score, fit, failures, len(accepted)


def _survival(distribution, theta, t):
    value = torch.ones_like(t)
    positive = t > 0
    if bool(positive.any()):
        value[positive] = _parts(distribution, theta, t[positive])[1].exp()
    return value


def _log_cumulative_hazard(distribution, theta, t):
    value = torch.zeros_like(t)
    positive = t > 0
    if not bool(positive.any()):
        return value
    logt = t[positive].log()
    if distribution == "exponential":
        transformed = theta[0] + logt
    elif distribution == "weibull":
        transformed = theta[0].exp() * (logt - theta[1])
    elif distribution == "lognormal":
        standard = (logt - theta[0]) * torch.exp(-theta[1])
        logcdf = torch.special.log_ndtr(standard)
        small = logcdf < -36
        transformed = torch.empty_like(standard)
        transformed[small] = logcdf[small]
        transformed[~small] = (-torch.special.log_ndtr(-standard[~small])).log()
    else:
        standard = theta[0].exp() * (logt - theta[1])
        small = standard < -36
        transformed = torch.empty_like(standard)
        transformed[small] = standard[small]
        transformed[~small] = F.softplus(standard[~small]).log()
    value[positive] = transformed
    return value


def _curves(distribution, theta, covariance, times, level, z):
    with torch.enable_grad():
        survival = _survival(distribution, theta, times)
        gradient = torch.autograd.functional.jacobian(
            lambda point: _survival(distribution, point, times), theta
        )
        transformed = _log_cumulative_hazard(distribution, theta, times)
        transformed_gradient = torch.autograd.functional.jacobian(
            lambda point: _log_cumulative_hazard(distribution, point, times), theta
        )
    joint = gradient @ covariance @ gradient.T
    joint = (joint + joint.T) / 2
    variance = joint.diag()
    if not bool(torch.isfinite(joint).all()) or bool((variance < -1e-12).any()):
        raise AnalysisError("numerical_failure", "Nonfinite or negative survival delta covariance.")
    se = variance.clamp_min(0).sqrt()
    transformed_variance = (transformed_gradient @ covariance * transformed_gradient).sum(dim=1)
    if (
        not bool(torch.isfinite(transformed).all())
        or not bool(torch.isfinite(transformed_variance).all())
        or bool((transformed_variance < -1e-12).any())
    ):
        raise AnalysisError(
            "numerical_failure", "Nonfinite complementary-log-log curve uncertainty."
        )
    transformed_se = transformed_variance.clamp_min(0).sqrt()
    rows = []
    for i, time in enumerate(times.tolist()):
        s = float(survival[i])
        if time == 0:
            lower = upper = s
        else:
            # Complementary-log-log limits stay in [0,1]. Their SE follows the
            # full raw-coordinate covariance, including parameter correlations.
            g, se_g = float(transformed[i]), float(transformed_se[i])
            lower = math.exp(-math.exp(min(709.0, g + z * se_g)))
            upper = math.exp(-math.exp(max(-745.0, g - z * se_g)))
        rows.append(
            dict(
                time=time,
                survival=s,
                se=float(se[i]),
                lower=lower,
                upper=upper,
                cdf=1 - s,
                cdf_se=float(se[i]),
                cdf_lower=1 - upper,
                cdf_upper=1 - lower,
                log_cumulative_hazard=None if time == 0 else float(transformed[i]),
                log_cumulative_hazard_se=float(transformed_se[i]),
                level=level,
            )
        )
    names = [f"survival({v:.17g})" for v in times.tolist()]
    covrows = [
        dict(row=names[i], column=names[j], covariance=float(joint[i, j]))
        for i in range(len(times))
        for j in range(len(times))
    ]
    return rows, covrows, joint


def _run(distribution, lower, upper, *, times, level, device, weights, maxiter):
    data = interval_data(lower, upper, device=device, weights=weights)
    level, z = confidence(level)
    t = prediction_times(
        times,
        default=sorted(
            set(
                [
                    0.0,
                    *[float(v) for v in data.lo if v > 0],
                    *[float(v) for v in data.hi if torch.isfinite(v)],
                ]
            )
        ),
    )
    budget = workspace(data.n, len(t))
    theta, covariance, information, score, fit, failures, successes = _fit(
        distribution, data, maxiter
    )
    raw_names, natural_names = _names(distribution)
    estimates = _natural(distribution, theta)
    diagonal = estimates.clone()
    if distribution == "lognormal":
        diagonal[0] = 1
    natural_covariance = covariance * diagonal[:, None] * diagonal[None, :]
    parameters = []
    for i, name in enumerate(natural_names):
        se_raw = float(covariance[i, i].sqrt())
        natural = float(estimates[i])
        lo, hi = float(theta[i]) - z * se_raw, float(theta[i]) + z * se_raw
        if not (distribution == "lognormal" and i == 0) and (
            lo < math.log(float.fromhex("0x0.0000000000001p-1022"))
            or hi > math.log(torch.finfo(DT).max)
        ):
            raise AnalysisError(
                "numerical_failure",
                "Natural-parameter confidence limits exceed float64 representation; no clipping.",
            )
        ci = (lo, hi) if distribution == "lognormal" and i == 0 else (math.exp(lo), math.exp(hi))
        parameters.append(
            dict(
                parameter=name,
                estimate=natural,
                se=float(natural_covariance[i, i].sqrt()),
                lower=ci[0],
                upper=ci[1],
                level=level,
            )
        )
    curves, curve_covariance, joint = _curves(distribution, theta, covariance, t, level, z)
    state = dict(
        schema="interval_survival_v1",
        distribution=distribution,
        raw_names=raw_names,
        raw_parameters=theta.tolist(),
        raw_covariance=covariance.tolist(),
        natural_names=natural_names,
        natural_parameters=estimates.tolist(),
        natural_covariance=natural_covariance.tolist(),
        level=level,
    )
    state["checksum"] = _checksum(state)
    terms = _contributions(distribution, theta, data).detach().tolist()
    frames = dict(
        parameters=parameters,
        covariance=[
            dict(
                row=natural_names[i],
                column=natural_names[j],
                covariance=float(natural_covariance[i, j]),
            )
            for i in range(len(theta))
            for j in range(len(theta))
        ],
        log_parameters=[
            dict(parameter=name, estimate=float(theta[i]), se=float(covariance[i, i].sqrt()))
            for i, name in enumerate(raw_names)
        ],
        log_covariance=[
            dict(row=raw_names[i], column=raw_names[j], covariance=float(covariance[i, j]))
            for i in range(len(theta))
            for j in range(len(theta))
        ],
        survival=curves,
        survival_covariance=curve_covariance,
        observations=[
            dict(
                observation=i,
                lower=data.settings["lower"][i],
                upper=data.settings["upper"][i],
                kind=data.kind[i],
                log_likelihood=terms[i],
            )
            for i in range(data.n)
        ],
    )
    settings = dict(
        data.settings,
        **budget,
        distribution=distribution,
        level=level,
        times=t.tolist(),
        log_likelihood=fit.value,
        likelihood_units="actual time density for exact observations; probability for censored intervals",
        n_parameters=len(theta),
        observed_information=information.tolist(),
        score=score.tolist(),
        scaled_score=float(score @ covariance @ score),
        converged=True,
        iterations=fit.iterations,
        optimizer="BFGS strong Wolfe with exact Torch autodiff score/Hessian and checked Newton polish",
        start_failures=failures,
        successful_starts=successes,
        prediction_state=state,
        survival_covariance=joint.tolist(),
        covariance="inverse full observed information; unregularized; natural-parameter Jacobian transform",
        inference="iid asymptotic normal ML; delta survival covariance and complementary-log-log survival CI; no df correction",
        inference_df=None,
        parameter_null=None,
        parameter_p_values="not reported; no parameter null supplied",
        survival_tail_policy="probability/delta SE may round to 0 or 1; CI comes directly from stable log cumulative hazard and its full raw-coordinate covariance",
        parameter_interior="|log rate|, |log scale/meanlog|<40; 0.001<shape/sdlog<1000",
        interval_probability="exact expm1 log-hazard differences; log-CDF/SF tail subtraction; only standardized width*(1+abs(lower_z))<1e-3 uses transformed-density 4/8-node log-mass agreement <=2e-12*max(1,abs(logmass))",
        score_gate="score.T @ raw_covariance @ score<=1e-10; positive-definite information condition<1e10",
        maxiter=maxiter,
        references=[
            "https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html",
            "https://stat.ethz.ch/R-manual/R-devel/library/survival/html/Surv.html",
        ],
    )
    return output("stinterval_" + distribution, frames, settings)


def _checksum(state):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in state.items() if k != "checksum"},
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _restore(state):
    expected = {
        "schema",
        "distribution",
        "raw_names",
        "raw_parameters",
        "raw_covariance",
        "natural_names",
        "natural_parameters",
        "natural_covariance",
        "level",
        "checksum",
    }
    if (
        set(state) != expected
        or state.get("schema") != "interval_survival_v1"
        or state.get("distribution") not in DISTRIBUTIONS
    ):
        raise AnalysisError(
            "invalid_result", "Unsupported saved interval survival schema/fields/distribution."
        )
    try:
        if _checksum(state) != state["checksum"]:
            raise AnalysisError(
                "invalid_result", "Saved interval survival checksum does not match."
            )
    except (TypeError, ValueError, OverflowError) as error:
        raise AnalysisError(
            "invalid_result", "Saved interval survival state is not finite JSON."
        ) from error
    distribution = state["distribution"]
    k = 1 if distribution == "exponential" else 2
    names, natural_names = _names(distribution)
    if state["raw_names"] != names or state["natural_names"] != natural_names:
        raise AnalysisError(
            "invalid_result",
            "Saved interval survival parameter names do not match the distribution.",
        )

    def vector(x):
        return (
            isinstance(x, list)
            and len(x) == k
            and all(
                isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(float(v))
                for v in x
            )
        )

    def matrix(x):
        return isinstance(x, list) and len(x) == k and all(vector(row) for row in x)

    if (
        not vector(state["raw_parameters"])
        or not vector(state["natural_parameters"])
        or not matrix(state["raw_covariance"])
        or not matrix(state["natural_covariance"])
    ):
        raise AnalysisError(
            "invalid_result",
            "Saved interval survival arrays must have finite numeric values and complete shapes.",
        )
    theta = torch.tensor(state["raw_parameters"], dtype=DT, device="cpu")
    covariance = torch.tensor(state["raw_covariance"], dtype=DT, device="cpu")
    if not _inside(distribution, theta):
        raise AnalysisError(
            "invalid_result", "Saved interval survival parameters exceed the supported interior."
        )
    if not torch.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
        raise AnalysisError(
            "invalid_covariance",
            "Saved raw covariance must be symmetric positive definite; no projection.",
        )
    eigen = torch.linalg.eigvalsh(covariance)
    if float(eigen[0]) <= 0 or float(eigen[-1]) / float(eigen[0]) >= 1e10:
        raise AnalysisError(
            "invalid_covariance", "Saved raw covariance is singular or ill-conditioned; no ridge."
        )
    estimates = _natural(distribution, theta)
    derivative = estimates.clone()
    if distribution == "lognormal":
        derivative[0] = 1
    natural_cov = covariance * derivative[:, None] * derivative[None, :]
    if not torch.allclose(
        torch.tensor(state["natural_parameters"], dtype=DT, device="cpu"),
        estimates,
        rtol=1e-12,
        atol=1e-14,
    ) or not torch.allclose(
        torch.tensor(state["natural_covariance"], dtype=DT, device="cpu"),
        natural_cov,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise AnalysisError(
            "invalid_result",
            "Saved natural estimates/covariance are inconsistent with the complete raw coordinates.",
        )
    confidence(state["level"])
    return distribution, theta, covariance


@procedure
def stinterval_exponential(
    lower, upper, *, times=None, level=0.95, device="cpu", weights=None, maxiter=500
):
    """Intercept-only exponential rate ML with exact/left/interval/right censoring."""
    return _run(
        "exponential",
        lower,
        upper,
        times=times,
        level=level,
        device=device,
        weights=weights,
        maxiter=maxiter,
    )


@procedure
def stinterval_weibull(
    lower, upper, *, times=None, level=0.95, device="cpu", weights=None, maxiter=500
):
    """Intercept-only Weibull shape/scale ML with mixed censoring and full OIM covariance."""
    return _run(
        "weibull",
        lower,
        upper,
        times=times,
        level=level,
        device=device,
        weights=weights,
        maxiter=maxiter,
    )


@procedure
def stinterval_lognormal(
    lower, upper, *, times=None, level=0.95, device="cpu", weights=None, maxiter=500
):
    """Intercept-only lognormal meanlog/sdlog ML with mixed censoring and full OIM covariance."""
    return _run(
        "lognormal",
        lower,
        upper,
        times=times,
        level=level,
        device=device,
        weights=weights,
        maxiter=maxiter,
    )


@procedure
def stinterval_loglogistic(
    lower, upper, *, times=None, level=0.95, device="cpu", weights=None, maxiter=500
):
    """Intercept-only loglogistic shape/scale ML with mixed censoring and full OIM covariance."""
    return _run(
        "loglogistic",
        lower,
        upper,
        times=times,
        level=level,
        device=device,
        weights=weights,
        maxiter=maxiter,
    )


@procedure
def interval_survival_predict(result, *, times, level=None, device="cpu", weights=None):
    """Restore complete saved raw covariance for survival curves; no refit or live model required."""
    controls(device, weights)
    attrs = result.attrs if hasattr(result, "attrs") else result
    if not isinstance(attrs, dict) or not isinstance(attrs.get("prediction_state"), dict):
        raise AnalysisError(
            "invalid_result", "Saved interval survival prediction_state is required."
        )
    state = attrs["prediction_state"]
    distribution, theta, covariance = _restore(state)
    level, z = confidence(state.get("level", 0.95) if level is None else level)
    t = prediction_times(times)
    budget = workspace(0, len(t))
    curves, covrows, joint = _curves(distribution, theta, covariance, t, level, z)
    settings = dict(
        **budget,
        source_method=attrs.get("method"),
        distribution=distribution,
        times=t.tolist(),
        level=level,
        prediction_state=state,
        survival_covariance=joint.tolist(),
        restored_complete_covariance=True,
        refit=False,
        device="cpu",
        inference="full saved ML parameter covariance; delta survival covariance and complementary-log-log CI",
        survival_tail_policy="probability/delta SE may round to 0 or 1; CI comes directly from stable log cumulative hazard and its full raw-coordinate covariance",
    )
    return output(
        "interval_survival_predict", dict(survival=curves, survival_covariance=covrows), settings
    )
