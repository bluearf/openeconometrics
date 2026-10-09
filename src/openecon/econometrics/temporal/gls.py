"""Fixed-shape Gaussian GLS temporal disaggregation on native CPU float64.

The high-frequency covariance shape is a caller/model assumption, rather
than an estimated autoregressive parameter.  The GLS fit and redistribution
follow Sax and Steiner (2013), equations 3 and 5; covariance shapes are their
equation 6 and section 2.2.  Joint reconstruction error covariance follows the
fixed Gaussian GLS linear-prediction derivation, including beta estimation.
Integrated processes have zero presample innovations/level.
"""
from __future__ import annotations

import math
from numbers import Real

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.multivariate.common import procedure
from openecon.engines.distributions import t_isf, t_sf
from .common import checked_solve, check_constraints, output, prepare


SOURCE = "https://journal.r-project.org/articles/RJ-2013-028/"


def _rho(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not -.95 <= value <= .95:
        raise AnalysisError("invalid_option", "rho must be a fixed finite number in [-0.95,0.95]; automatic rho estimation is unsupported.")
    return float(value)


def _options(intercept, alpha):
    if type(intercept) is not bool:
        raise AnalysisError("invalid_option", "intercept must be a boolean.")
    if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha) or not 1e-8 <= alpha <= .5:
        raise AnalysisError("invalid_option", "alpha must lie in [1e-8,0.5].")
    return float(alpha)


def _covariance(method, n, rho):
    """Return covariance shape and its causal innovation factor.

    The first Chow--Lin state has its stationary distribution.  Fernandez
    and Litterman instead anchor the level and AR increment before the first
    observation at zero; no diffuse or stationary-integrated initialization.
    """
    indices = torch.arange(n, dtype=torch.int64, device="cpu")
    lag = indices[:, None] - indices[None, :]
    if method == "fernandez":
        factor = torch.ones((n, n), dtype=torch.float64, device="cpu").tril()
        shape = torch.minimum(indices[:, None], indices[None, :]).to(torch.float64) + 1.
    else:
        base = torch.tensor(rho, dtype=torch.float64, device="cpu")
        autoregression = torch.pow(base, lag.clamp_min(0)).tril()
        if method == "chow_lin":
            factor = autoregression.clone()
            factor[:, 0] /= math.sqrt(1.-rho*rho)
            shape = torch.pow(base, lag.abs()) / (1.-rho*rho)
        else:
            # D^-1 H^-1 maps independent innovations into the anchored
            # integrated AR(1) process; V = (D' H' H D)^-1.
            factor = torch.cumsum(autoregression, dim=0)
            shape = factor @ factor.T
    return shape, factor


def _design(x, intercept):
    n, indicators = x.shape
    parts = [torch.ones((n, 1), dtype=torch.float64, device="cpu"), x] if intercept else [x]
    original = torch.cat(parts, dim=1)
    scaled = original.clone()
    transform = torch.eye(original.shape[1], dtype=torch.float64, device="cpu")
    for j in range(indicators):
        column = j + int(intercept)
        center = float(x[:, j].mean()) if intercept else 0.
        scale = float((x[:, j] - center).square().mean().sqrt())
        if not math.isfinite(scale) or scale <= 0:
            raise AnalysisError("singular_design", "An indicator is zero or constant in an intercept model; no predictor is silently removed.")
        scaled[:, column] = (x[:, j] - center) / scale
        transform[column, column] = 1. / scale
        if intercept:
            transform[0, column] = -center / scale
    return original, scaled, transform, [*(["_cons"] if intercept else []), *[f"x{i+1}" for i in range(indicators)]]


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("numerical_failure", f"Nonfinite {name}; rescale the supplied inputs.")


@torch.no_grad()
def _run(method, low, indicator, *, rho, low_periods, high_periods,
         low_frequency, high_frequency, aggregation, as_of, low_releases,
         high_releases, device, weights, intercept, alpha):
    alpha = _options(intercept, alpha)
    p = prepare(low, indicator, low_periods=low_periods, high_periods=high_periods,
                low_frequency=low_frequency, high_frequency=high_frequency,
                aggregation=aggregation, as_of=as_of, low_releases=low_releases,
                high_releases=high_releases, device=device, weights=weights)
    original, x, transform, terms = _design(p.x, intercept)
    parameters = len(terms)
    df = p.m - parameters
    if df <= 0:
        raise AnalysisError("insufficient_observations", "GLS needs more low-frequency observations than fitted coefficients, including the intercept.")
    shape, factor = _covariance(method, p.n, rho)
    W = p.C @ shape @ p.C.T
    z = p.C @ x
    # A checked solve validates the covariance condition and backward error;
    # whitening and QR avoid squaring the scaled low-frequency design's
    # condition number during coefficient estimation.
    distribution = checked_solve(W, p.C @ shape, "low-frequency covariance").T
    try:
        cholesky = torch.linalg.cholesky(W)
    except torch.linalg.LinAlgError as error:
        raise AnalysisError("numerical_failure", "The declared low-frequency covariance is not positive definite.") from error
    white_x = torch.linalg.solve_triangular(cholesky, z, upper=False)
    white_y = torch.linalg.solve_triangular(cholesky, p.y[:, None], upper=False).squeeze(1)
    singular = torch.linalg.svdvals(white_x)
    if not bool(torch.isfinite(singular).all()) or float(singular[-1]) <= float(singular[0]) * 1e-10:
        raise AnalysisError("singular_design", "The aggregated whitened indicators are collinear or ill-conditioned after scaling; no ridge fallback.")
    Q, R = torch.linalg.qr(white_x, mode="reduced")
    beta_scaled = checked_solve(R, Q.T @ white_y, "scaled GLS coefficient system")
    R_inverse = checked_solve(R, torch.eye(parameters, dtype=torch.float64, device="cpu"), "scaled GLS covariance system")
    bread_scaled = R_inverse @ R_inverse.T
    beta = transform @ beta_scaled
    residual = p.y - z @ beta_scaled
    white_residual = white_y - white_x @ beta_scaled
    sse = float(white_residual @ white_residual)
    energy = float(white_y @ white_y)
    if not math.isfinite(sse) or sse <= 0 or sse <= 1e-24 * energy:
        raise AnalysisError("degenerate_inference", "Residual innovation variance is zero or numerically degenerate (weighted SSE <= 1e-24 of response energy); GLS uncertainty is unavailable.")
    sigma2, sigma2_ml = sse / df, sse / p.m
    coefficient_covariance = sigma2 * (transform @ bread_scaled @ transform.T)
    coefficient_covariance = (coefficient_covariance + coefficient_covariance.T) / 2.
    preliminary = x @ beta_scaled
    values = preliminary + distribution @ residual
    observed_positions = p.C.argmax(dim=1) if aggregation in ("first", "last") else None
    if observed_positions is not None:
        values[observed_positions] = p.y
    check_constraints(p, values)
    # This factor form equals V - VC'W^-1 CV plus the complete GLS beta
    # uncertainty term.  It avoids subtractive loss of positive semidefiniteness.
    conditional_factor = factor - distribution @ (p.C @ factor)
    beta_error_factor = (x - distribution @ z) @ R_inverse
    # First/last aggregation observes the corresponding high-frequency value
    # exactly.  Its latent reconstruction error is identically zero.
    if observed_positions is not None:
        conditional_factor[observed_positions, :] = 0.
        beta_error_factor[observed_positions, :] = 0.
    high_covariance = sigma2 * (conditional_factor @ conditional_factor.T + beta_error_factor @ beta_error_factor.T)
    high_covariance = (high_covariance + high_covariance.T) / 2.
    _finite(beta, "GLS coefficients")
    _finite(coefficient_covariance, "coefficient covariance")
    _finite(high_covariance, "complete reconstruction error covariance")
    if bool((coefficient_covariance.diagonal() <= 0).any()):
        raise AnalysisError("numerical_failure", "The coefficient covariance must have positive diagonal entries.")
    constrained_factor = p.C @ torch.cat((conditional_factor, beta_error_factor), dim=1)
    error_scale = max(float(conditional_factor.abs().max()), float(beta_error_factor.abs().max()), 1e-300)
    if float(constrained_factor.abs().max()) > 2e-9 * error_scale * p.n:
        raise AnalysisError("numerical_failure", "The complete reconstruction error covariance failed the exact-aggregation gate.")
    sign, logdet = torch.linalg.slogdet(W)
    if float(sign) != 1 or not math.isfinite(float(logdet)):
        raise AnalysisError("numerical_failure", "The low-frequency covariance determinant is not positive finite.")
    log_likelihood = -.5 * (p.m * (math.log(2. * math.pi) + math.log(sigma2_ml) + 1.) + float(logdet))
    critical = t_isf(alpha / 2., df)
    se = coefficient_covariance.diagonal().sqrt()
    statistic = beta / se
    high_se = high_covariance.diagonal().clamp_min(0).sqrt()
    coefficients = table(dict(term=terms, coefficient=beta.tolist(), standard_error=se.tolist(),
                              t=statistic.tolist(), p_value=[2. * t_sf(abs(float(t)), df) for t in statistic],
                              ci_lower=(beta-critical*se).tolist(), ci_upper=(beta+critical*se).tolist()))
    high_periods = p.settings["high_periods"]
    fit = dict(weighted_sse=sse, sigma2=sigma2, sigma2_ml=sigma2_ml,
               log_likelihood=log_likelihood, df_residual=df, n_parameters=parameters,
               n_low=p.m, n_high=p.n)
    covariance_shape = {"chow_lin": "stationary AR(1): rho^abs(i-j)/(1-rho^2)",
                        "fernandez": "anchored random walk: inverse(D' D), V_ij=min(i,j) for one-based indices",
                        "litterman": "anchored integrated AR(1): inverse(D' H' H D), H diagonal 1 and subdiagonal -rho"}[method]
    extra_tables = dict(
        coefficients=coefficients,
        coefficient_covariance=table(coefficient_covariance.tolist(), columns=terms, index=terms),
        high_covariance=table(high_covariance.tolist(), columns=high_periods, index=high_periods),
        latent_intervals=table(dict(high_period=high_periods, value=values.tolist(),
                                   standard_error=high_se.tolist(), ci_lower=(values-critical*high_se).tolist(),
                                   ci_upper=(values+critical*high_se).tolist())),
        low_fit=table(dict(low_period=p.settings["low_periods"], observed=p.y.tolist(),
                           fitted=(p.C@preliminary).tolist(), residual=residual.tolist())),
        fit=table([fit]),
    )
    return output(method, p, values, tables=extra_tables,
                  settings=dict(**fit, rho=rho, intercept=intercept, alpha=alpha,
                                covariance_shape=covariance_shape, scale_convention="innovation variance",
                                rho_estimation="fixed caller value; no rho estimation or rho uncertainty" if method != "fernandez" else "rho=0 by model definition",
                                initialization="stationary AR(1) state" if method == "chow_lin" else "zero presample level and increment; anchored process",
                                inference="Gaussian GLS with fixed covariance shape; coefficient and marginal latent reconstruction error t intervals with low-frequency residual df",
                                uncertainty="complete joint model-dependent BLUP reconstruction error covariance, including estimated GLS coefficients; exact low observations; no forecast/future observed-value intervals or simultaneous coverage",
                                likelihood="low-frequency Gaussian ML scale SSE/n_low at fixed rho; inference separately uses SSE/df_residual",
                                solver="centered/scaled indicators; Cholesky whitening and reduced QR; condition/backward-error and exact-aggregation gates; no ridge",
                                source=SOURCE,
                                notes=["The high-frequency relationship and covariance process are model assumptions; reconstructed values are latent estimates rather than newly observed data.",
                                       "Fixed rho inference excludes covariance-parameter uncertainty, parameter selection and uncertainty in the observed low-frequency totals.",
                                       "First/last-constrained high values have zero reconstruction uncertainty. Marginal t intervals do not provide simultaneous coverage."]))


@procedure
def chow_lin(low, indicator, *, rho, low_periods, high_periods,
             low_frequency="Y", high_frequency="Q", aggregation="sum", as_of=None,
             low_releases=None, high_releases=None, device="cpu", weights=None,
             intercept=True, alpha=.05) -> TableSet:
    """Disaggregate exact low totals using fixed stationary AR(1) Gaussian GLS.

    rho is required and fixed in [-.95,.95]; V_ij=rho^abs(i-j)/(1-rho²).
    Scale is innovation variance, fitted as weighted SSE/(n_low-k) for t
    inference and SSE/n_low for the reported Gaussian ML log likelihood.
    Complete coefficient and latent BLUP reconstruction error covariance
    include GLS coefficient estimation and respect every exact low constraint.
    No rho estimation, forecasting, sampling weights, missing-row deletion or
    automatic calendar inference. See the complete common calendar contract.
    """
    return _run("chow_lin", low, indicator, rho=_rho(rho), low_periods=low_periods,
                high_periods=high_periods, low_frequency=low_frequency, high_frequency=high_frequency,
                aggregation=aggregation, as_of=as_of, low_releases=low_releases,
                high_releases=high_releases, device=device, weights=weights,
                intercept=intercept, alpha=alpha)


@procedure
def fernandez(low, indicator, *, low_periods, high_periods,
              low_frequency="Y", high_frequency="Q", aggregation="sum", as_of=None,
              low_releases=None, high_releases=None, device="cpu", weights=None,
              intercept=True, alpha=.05) -> TableSet:
    """Disaggregate exact low totals with anchored Gaussian random-walk GLS.

    The presample level is zero and V=(D'D)^-1, with first-difference D
    diagonal 1 and subdiagonal -1. This is exactly litterman(rho=0), including
    coefficients, scale, likelihood and complete latent error covariance.
    Inference uses t(n_low-k) at the declared covariance shape and excludes
    forecasting, estimated process parameters and low-total uncertainty.
    """
    return _run("fernandez", low, indicator, rho=0., low_periods=low_periods,
                high_periods=high_periods, low_frequency=low_frequency, high_frequency=high_frequency,
                aggregation=aggregation, as_of=as_of, low_releases=low_releases,
                high_releases=high_releases, device=device, weights=weights,
                intercept=intercept, alpha=alpha)


@procedure
def litterman(low, indicator, *, rho, low_periods, high_periods,
              low_frequency="Y", high_frequency="Q", aggregation="sum", as_of=None,
              low_releases=None, high_releases=None, device="cpu", weights=None,
              intercept=True, alpha=.05) -> TableSet:
    """Disaggregate exact low totals with fixed anchored integrated AR(1) GLS.

    Level u_t=u_(t-1)+v_t and increment v_t=rho*v_(t-1)+epsilon_t start
    with u_0=v_0=0. The shape is (D'H'H D)^-1 and rho is required in
    [-.95,.95]. Inference and the complete joint latent reconstruction error
    covariance condition on this fixed shape; no rho fitting/diffuse start,
    forecasting, low-total uncertainty or future observed-value prediction.
    """
    return _run("litterman", low, indicator, rho=_rho(rho), low_periods=low_periods,
                high_periods=high_periods, low_frequency=low_frequency, high_frequency=high_frequency,
                aggregation=aggregation, as_of=as_of, low_releases=low_releases,
                high_releases=high_releases, device=device, weights=weights,
                intercept=intercept, alpha=alpha)
