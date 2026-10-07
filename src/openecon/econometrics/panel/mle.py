"""Random-effects maximum likelihood (Stata's xtreg, mle).

The Gaussian random-effects model y_i = X_i beta + u_i 1 + e_i has, per panel,
the covariance sigma_e^2 I + sigma_u^2 11' whose inverse and determinant are
closed form, so the log likelihood is O(N) (``kernels.RandomEffectsLikelihood``).
It is maximized over (beta, ln sigma_u, ln sigma_e) by Newton-Raphson with the
analytic gradient and Hessian, starting from the Swamy-Arora GLS estimates.

Conditioning. The likelihood is maximized on centered and scaled data: every
slope column is replaced by (x - xbar) / s_x and the outcome by (y - ybar) / s_y
(s = root mean square of the centered column). This is an exact affine
reparameterization, so the estimates, the covariance (A V A') and the log
likelihood (minus N ln s_y) are mapped back without approximation, and a
regressor with a 1e5 offset converges exactly like the GLS models do.

sigma_u and sigma_e are reported as the ancillary terms ``/sigma_u`` and
``/sigma_e`` with delta-method standard errors from the ln parameterization
(the covariance in reporting units is J V J with J = diag(1, ..., sigma_u,
sigma_e)); their confidence intervals are exp(ln sigma -/+ z se_ln), the
asymmetric interval Stata prints (its ``/sigma`` rows carry no z or P>|z|; the
z statistic reported here is sigma / se, the delta-method one). The ln-scale
estimates and standard errors are in ``extra['ln_sigma']``. The covariance is
the observed information (OIM, Stata's default); vce(robust)/vce(cluster),
which Stata 19 added for this model, are not offered.

Tests: ``model`` is the LR chi2(k) test against the constant-only random-effects
model; ``sigma_u`` is the LR test of sigma_u = 0 against pooled OLS, a
boundary test whose reference distribution is chibar2(01) (the chi2(1) tail
halved).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, build_result, information_criteria, kernel_call, lr_test, ml_covariance,
)
from openecon.econometrics.panel import kernels
from openecon.econometrics.panel.common import PanelSample, random_effects_gls
from openecon.engines import optimize
from openecon.engines.distributions import chi2_sf
from openecon.engines.inference import critical_value
from openecon.models import ResultBundle

# ln(sigma_u / sigma_e) below which a non-converged iteration is a boundary solution.
_BOUNDARY = math.log(1e-4)


def _maximize(like: kernels.RandomEffectsLikelihood, start: Tensor,
              what: str) -> optimize.OptimResult:
    try:
        result = optimize.maximize_newton(like, start, value_fn=like.value, raise_on_failure=False)
    except Exception as exc:          # KernelError and numerical failures of the kernel
        raise AnalysisError("numerical_failure",
                            f"The {what} likelihood could not be evaluated: {exc}") from exc
    if result.converged:
        return result
    if float(result.theta[-2] - result.theta[-1]) < _BOUNDARY:
        raise AnalysisError(
            "boundary_solution",
            f"The {what} maximum likelihood estimate of sigma_u lies at zero (no panel-level "
            "variance), where the likelihood is flat in ln sigma_u. Use model='re' (whose "
            "Breusch-Pagan test will not reject sigma_u = 0) or model='pooled'.")
    raise AnalysisError("nonconvergence",
                        f"The {what} likelihood maximization did not converge: "
                        f"{result.diagnostics.get('message')}")


def _rms(values: Tensor) -> Tensor:
    """Root mean square of each centered column, 1 where a column is constant."""
    scale = values.square().mean(dim=0).sqrt()
    return torch.where(scale > 0, scale, torch.ones_like(scale))


def fit_mle(sample: PanelSample) -> ResultBundle:
    """Fit ``xtreg, mle`` on a validated panel sample (see the module docstring).

    Raises ``boundary_solution`` when the maximum lies at sigma_u = 0,
    ``nonconvergence`` when Newton-Raphson stops elsewhere without converging and
    ``no_within_variation`` when the outcome leaves no idiosyncratic variance.
    """
    frame, spec, codes, n, nobs = sample.frame, sample.spec, sample.codes, sample.n, sample.nobs
    design = frame.drop_collinear(frame.design())
    y = frame.numeric(spec.outcome)
    kk = design.x.shape[1]
    if nobs - kk - 2 <= 0:
        raise AnalysisError("insufficient_observations", "The random-effects likelihood needs "
                            "more observations than parameters.")
    # Condition the problem: centered, unit-RMS slopes and outcome (exact reparameterization).
    centre_x = design.x[:, 1:].mean(dim=0)
    scaled_x = design.x[:, 1:] - centre_x
    scale_x = _rms(scaled_x)
    scaled_x /= scale_x
    centre_y = float(y.mean())
    scale_y = float(_rms((y - centre_y)[:, None])[0])
    scaled = Design(torch.cat([torch.ones((frame.n, 1), dtype=torch.float64), scaled_x], dim=1),
                    design.terms, design.categories, True)
    ys = (y - centre_y) / scale_y
    # The rounding yardstick of the degenerate-fit checks is the level of y, in scaled units.
    gls = random_effects_gls(sample, scaled, ys, sa=False,
                             scale=float(y.square().sum()) / scale_y ** 2)
    sigma_u2 = max(gls.sigma_u2, 1e-2 * gls.sigma_e2)
    scales = torch.tensor([0.5 * math.log(sigma_u2), 0.5 * math.log(gls.sigma_e2)],
                          dtype=torch.float64)
    like = kernel_call(kernels.RandomEffectsLikelihood, scaled.x, ys, codes, n)
    result = _maximize(like, torch.cat([gls.beta, scales]), "random-effects")
    ones = scaled.x[:, :1]
    null = kernel_call(kernels.RandomEffectsLikelihood, ones, ys, codes, n)
    restricted = _maximize(null, torch.cat([ys.mean().reshape(1), result.theta[-2:]]),
                           "constant-only random-effects")
    # Map theta = (b0, b, ln su, ln se) back to the original units: theta = A theta_s + c.
    ratio = scale_y / scale_x
    jacobian = torch.eye(kk + 2, dtype=torch.float64)
    jacobian[0, 0] = scale_y
    jacobian[0, 1:kk] = -centre_x * ratio
    jacobian[1:kk, 1:kk] = torch.diag(ratio)
    shift = torch.zeros(kk + 2, dtype=torch.float64)
    shift[0], shift[kk:] = centre_y, math.log(scale_y)
    theta = jacobian @ result.theta + shift
    v_scaled, info = ml_covariance(frame, hessian=result.hessian, kind="nonrobust")
    v_log = jacobian @ v_scaled @ jacobian.T
    sigma_u, sigma_e = float(torch.exp(theta[kk])), float(torch.exp(theta[kk + 1]))
    delta = torch.ones(kk + 2, dtype=torch.float64)
    delta[kk], delta[kk + 1] = sigma_u, sigma_e
    v = v_log * delta[:, None] * delta
    info["correction"] = ("observed information (OIM); /sigma_u and /sigma_e by the delta method "
                          "from ln sigma, confidence intervals exp(ln sigma -/+ z se_ln)")
    info["df_inference"] = None
    offset = nobs * math.log(scale_y)
    ll = result.value - offset
    ll_ols = kernel_call(kernels.ols_log_likelihood, gls.pooled_ssr, nobs) - offset
    lr_sigma = max(0.0, 2 * (ll - ll_ols))
    tests = {
        "model": lr_test(ll, restricted.value - offset, kk - 1, label="LR chi2 test of the slopes"),
        "sigma_u": {"statistic": lr_sigma, "df": 1, "p_value": 0.5 * chi2_sf(lr_sigma, 1),
                    "distribution": "chibar2",
                    "label": "LR test of sigma_u = 0 against pooled OLS (chibar2(01): chi2(1) "
                             "tail halved)"},
    }
    metrics = {"sigma_u": sigma_u, "sigma_e": sigma_e,
               "rho": sigma_u ** 2 / (sigma_u ** 2 + sigma_e ** 2),
               **information_criteria(ll, kk + 2, nobs), **sample.structure()}
    se_ln = v_log.diagonal()[kk:].sqrt()
    extra = {"model": "mle",
             "ln_sigma": {"sigma_u": {"estimate": float(theta[kk]), "std_error": float(se_ln[0])},
                          "sigma_e": {"estimate": float(theta[kk + 1]),
                                      "std_error": float(se_ln[1])}},
             "null_log_likelihood": restricted.value - offset,
             "starting_values": {"sigma_u": scale_y * math.sqrt(sigma_u2),
                                 "sigma_e": scale_y * math.sqrt(gls.sigma_e2),
                                 "method": "swamy_arora_gls"},
             "conditioning": {"outcome_scale": scale_y, "regressor_scale_min": float(scale_x.min()),
                              "regressor_scale_max": float(scale_x.max())}}
    optimizer = {"method": result.method, "iterations": result.iterations,
                 "converged": result.converged,
                 "gradient_max": result.diagnostics.get("gradient_max"),
                 "message": result.diagnostics.get("message")}
    bundle = build_result(
        frame, terms=[*design.terms, "/sigma_u", "/sigma_e"],
        params=torch.cat([theta[:kk], torch.tensor([sigma_u, sigma_e], dtype=torch.float64)]),
        covariance=v, title="Random-effects ML regression",
        equations=[spec.outcome] * kk + [None, None], use_t=False, df_resid=nobs - kk,
        metrics=metrics, fitted=design.x @ theta[:kk], solver="newton_random_effects",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer, inference=info, tests=tests, extra=extra,
        categories=design.categories, nobs=nobs, provenance={"model": "mle"},
    )
    # Stata's /sigma rows: the interval of ln sigma transformed back (asymmetric).
    critical = critical_value(spec.alpha, None)
    for position, sigma, se in ((kk, sigma_u, float(se_ln[0])), (kk + 1, sigma_e, float(se_ln[1]))):
        term = bundle.coefficients[position]
        term.ci_low = sigma * math.exp(-critical * se)
        term.ci_high = sigma * math.exp(critical * se)
    return bundle
