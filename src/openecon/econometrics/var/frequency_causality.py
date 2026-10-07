"""Breitung--Candelon (2006) pointwise frequency noncausality restrictions."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch

from openecon.analysis import _coerce_frame, _frame_hasher, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.unitroot.common import check_count, check_flag, check_magnitude
from openecon.econometrics.var import kernels
from openecon.econometrics.var.common import check_alpha
from openecon.econometrics.var.estimators import _screen_system
from openecon.econometrics.var.toda_yamamoto import _budget, _ordered_sample
from openecon.engines.distributions import f_sf

SOURCE = "https://doi.org/10.1016/j.jeconom.2005.02.004"


def _frequencies(values: Any) -> list[float]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 1000:
        raise AnalysisError("invalid_frequency", "frequencies must contain 1..1000 distinct "
                            "radian frequencies strictly between 0 and pi.")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real) \
                or not math.isfinite(value) or not 0 < value < math.pi:
            raise AnalysisError("invalid_frequency", "Each frequency must be finite and "
                                "strictly between 0 and pi; endpoint inference is unavailable.")
        result.append(float(value))
    if len(set(result)) != len(result):
        raise AnalysisError("invalid_frequency", "frequencies must be distinct.")
    return result


def bccaustest(*, data: Any, y: str, x: str, frequencies: list[float], lags: int = 3,
              time: str | None = None, constant: bool = True,
              integration_order: int = 0, alpha: float = 0.05) -> TableSet:
    """Breitung--Candelon pointwise test of x not predicting y at each frequency.

    Fits a complete bivariate VAR(p) and tests the two restrictions
    sum_j beta_j cos(j*omega) = sum_j beta_j sin(j*omega) = 0 on x's lags
    in y's equation. Frequencies are radians per observation (0 < omega < pi),
    and the associated cycle length is 2*pi/omega observations. At least three
    fixed lags are required for distinct frequency restrictions.

    This validated route assumes jointly stationary series, a correctly
    specified fixed lag order, and iid homoskedastic innovations. The caller
    declares integration_order=0; integrated/cointegrated, trend, endpoint,
    multivariate, HAC and lag-selection inference are not supplied. A fitted
    unstable VAR is rejected, but fitted roots are not a stationarity pretest.
    Each F = W/2 uses SSR/(n-p-2p-constant) covariance and an approximate
    F(2, residual_df) reference, not an exact finite-sample dynamic-regression
    law. Decisions are pointwise, with no simultaneous frequency-band claim.

    Complete regular observations are sorted by time when provided, otherwise
    row order supplies time. No missing rows are silently dropped. In-memory
    CPU float64 QR is bounded to 512 MiB workspace. Returns tests, coefficients,
    and restrictions tables with direction, sample hash, df and assumptions;
    tables support saved console output and publication LaTeX.

    Parameters
    ----------
    data : table
        Complete regularly spaced in-memory observations.
    y, x : str
        Effect and cause columns; direction is x to y.
    frequencies : list[float]
        Distinct radians per observation, strictly between 0 and pi.
    lags : int
        Caller-chosen fixed VAR lag order p >= 3.
    time : str or None
        Regular calendar or exact consecutive integer key; row order if None.
    constant : bool
        Include an intercept in both equations; consumes one residual df.
    integration_order : int
        Declared upper integration order; only stationary order 0 is validated.
    alpha : float
        Significance level of each separate pointwise frequency decision.

    Returns
    -------
    TableSet
        Tests, both equations' coefficients, and frequency restrictions.
    """
    if not all(isinstance(name, str) and name for name in (y, x)) or y == x:
        raise AnalysisError("invalid_spec", "y and x must name two distinct nonempty columns.")
    p = check_count(lags, "lags", minimum=3)
    check_flag(constant, "constant")
    if check_count(integration_order, "integration_order") != 0:
        raise AnalysisError("unsupported_integration_order", "This validated frequency-test "
                            "route supports declared stationary series only (integration_order=0).")
    omega = _frequencies(frequencies)
    alpha = check_alpha(alpha)
    if time is not None and (not isinstance(time, str) or not time or time in (y, x)):
        raise AnalysisError("invalid_spec", "time must name a distinct nonempty column.")
    raw = _coerce_frame(data)
    usable, m, work_bytes = _budget(len(raw), 2, p, int(constant))
    chosen = _ordered_sample(raw, [y, x], time)
    with torch.device("cpu"), torch.no_grad():
        levels = torch.stack([_numeric(chosen[name], name) for name in (y, x)], dim=1)
        check_magnitude(levels, [y, x])
        z = kernels.lag_block(levels, p)
        terms = [f"L{lag}.{name}" for name in (y, x) for lag in range(1, p + 1)]
        mean_z, mean_y = z.new_zeros(2 * p), levels.new_zeros(2)
        outcome = levels[p:]
        if constant:
            mean_z, mean_y = z.mean(0), outcome.mean(0)
            z = torch.cat([z - mean_z, torch.ones((usable, 1), dtype=torch.float64)], dim=1)
            outcome = outcome - mean_y
            terms.append("Intercept")
        _screen_system(z, terms, constant)
        fit = kernel_call(kernels.fit_system, z, outcome, [y, x])
        # Variable-major design -> lag-major companion blocks.
        companion = torch.zeros((2 * p, 2 * p), dtype=torch.float64)
        companion[:2] = fit.coef[:, :2 * p].reshape(2, 2, p).transpose(1, 2).reshape(2, 2 * p)
        companion[2:, :-2] = torch.eye(2 * p - 2, dtype=torch.float64)
        largest_root = float(torch.linalg.eigvals(companion).abs().max())
        if not math.isfinite(largest_root) or largest_root >= 1:
            raise AnalysisError("unstable_var", "The fitted VAR is unstable; stationary "
                                "frequency-test inference is unavailable for this sample.")
        df = usable - m
        covariance = fit.xtx_inv * (fit.resid[:, 0].square().sum() / df)
        beta = fit.coef[0, p:2 * p]
        covariance_beta = covariance[p:2 * p, p:2 * p]
        lag_numbers = torch.arange(1, p + 1, dtype=torch.float64)
        rows, restrictions = [], []
        for frequency in omega:
            restriction = torch.stack([(frequency * lag_numbers).cos(),
                                       (frequency * lag_numbers).sin()])
            # Orthonormalize the row space. This avoids squaring the restriction
            # condition number near endpoints and never changes the null.
            _, singular_values, vh = torch.linalg.svd(restriction, full_matrices=False)
            if float(singular_values[-1]) <= 1e-10 * float(singular_values[0]):
                raise AnalysisError("singular_frequency_restriction", "A requested frequency "
                                    "is too close to an endpoint for two independent restrictions.")
            basis = vh[:2]
            difference = basis @ beta
            restricted_cov = basis @ covariance_beta @ basis.T
            factor = kernel_call(kernels.spd_factor, restricted_cov,
                                 code="singular_test_covariance", what="frequency restriction covariance")
            statistic = float((difference[:, None].T @ torch.cholesky_solve(
                difference[:, None], factor)).squeeze()) / 2
            if not math.isfinite(statistic) or statistic < 0:
                raise AnalysisError("numerical_failure", "Frequency test produced a nonfinite statistic.")
            probability = kernel_call(f_sf, statistic, 2, df)
            rows.append({"effect": y, "cause": x, "frequency": frequency,
                         "cycle_period": 2 * math.pi / frequency, "statistic": statistic,
                         "df_num": 2, "df_den": df, "p_value": probability,
                         "reject": probability < alpha})
            restrictions.extend({"frequency": frequency, "term": f"L{lag}.{x}",
                                 "cosine": float(restriction[0, lag - 1]),
                                 "sine": float(restriction[1, lag - 1])}
                                for lag in range(1, p + 1))
        coef = fit.coef.clone()
        change = torch.eye(m, dtype=torch.float64)
        if constant:
            coef[:, -1] += mean_y - coef[:, :-1] @ mean_z
            change[-1, :-1] = -mean_z
        bread = change @ fit.xtx_inv @ change.T
        sigma_df = fit.resid.T @ fit.resid / df
        coefficients = [{"equation": name, "term": term, "estimate": float(coef[i, j]),
                         "std_error": math.sqrt(float(sigma_df[i, i] * bread[j, j]))}
                        for i, name in enumerate((y, x)) for j, term in enumerate(terms)]
        return TableSet({"tests": table(rows, title="Breitung--Candelon frequency tests"),
                         "coefficients": table(coefficients, title="Bivariate VAR coefficients"),
                         "restrictions": table(restrictions, title="Frequency restrictions on cause lags")},
                        title="Breitung--Candelon frequency noncausality", source=SOURCE,
                        effect=y, cause=x, lags=p, constant=constant, alpha=alpha,
                        integration_order=0, observations=usable, levels=len(chosen),
                        regressors=m, residual_df=df, covariance_divisor=df,
                        covariance="classical df-corrected", distribution="F",
                        reference="approximate F(2, residual_df)", pointwise=True,
                        maximum_companion_root=largest_root, sample_sha256=_frame_hasher(chosen).hexdigest(),
                        time_order=f"sorted by {time}" if time else "input row order",
                        estimated_workspace_bytes=work_bytes, device="cpu", dtype="float64",
                        null="cosine and sine weighted sums of cause lag coefficients are zero",
                        alternative="predictive causality at the specified frequency",
                        inference_scope="stationary bivariate fixed-lag VAR; iid homoskedastic innovations",
                        notes=["Frequency decisions are pointwise; no multiple-frequency adjustment.",
                               "Fitted stability does not establish stationarity of the data-generating process.",
                               "Predictive noncausality tests do not identify structural causal effects."])
