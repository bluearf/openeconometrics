"""Exact full-source univariate residual tests with bounded lag histories."""
from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import chi2_sf
from openecon.engines.execution import qr_factor
from openecon.engines.linalg import least_squares
from openecon.engines.streaming_filter import LagState
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments


def diagnostics(factory, n, lags, *, fitted_parameters=0, arch_lags=1, label_suffix="residuals", allow_degenerate_aux=False):
    if not 0 < lags < n:
        raise AnalysisError("invalid_lags", "Residual diagnostic lags must be positive and smaller than the sample.")
    moments = _WeightedMoments(2, intercept=True)
    for values in factory():
        moments.add(torch.stack((torch.ones_like(values), values), 1), torch.ones_like(values), False)
    mean = float(moments.anchor[1]+moments.magnitude[1]*moments.mean[1])
    sums, history = _CompensatedSum((lags+3,)), LagState(lags)
    if type(arch_lags) is not int or not 0<=arch_lags<n:
        raise AnalysisError("invalid_lags", "ARCH-LM lags must be a nonnegative integer smaller than the sample.")
    tree, square_history, target_moments, seen = _TSQRTree(), LagState(arch_lags), _WeightedMoments(2, intercept=True), 0
    for values in factory():
        centered = values-mean
        lagged = history(centered, list(range(1, lags+1)))
        sums.add(torch.stack([centered.square().sum(), centered.pow(3).sum(), centered.pow(4).sum(), *[centered@other for other in lagged]]))
        square = values.square()
        previous = square_history(square, list(range(1, arch_lags+1))) if arch_lags else []
        skipped = min(len(values), max(0, arch_lags-seen))
        seen += len(values)
        if arch_lags and skipped < len(values):
            target = square[skipped:]
            x = torch.stack((torch.ones_like(target), *[row[skipped:] for row in previous]), 1)
            tree.add(qr_factor(torch.cat((x, target[:, None]), 1)))
            target_moments.add(torch.stack((torch.ones_like(target), target), 1), torch.ones_like(target), False)
    m2, m3, m4 = [float(value/n) for value in sums.value[:3]]
    if not m2 > 0.:
        return {}
    skew, kurtosis = m3/m2**1.5, m4/(m2*m2)
    jb = n/6.*(skew*skew+(kurtosis-3.)**2/4.)
    r = sums.value[3:]/sums.value[0]
    q = float(n*(n+2.)*(r.square()/torch.arange(n-1, n-lags-1, -1, dtype=torch.float64)).sum())
    result = {"ljung_box": {"statistic": q, "df": lags, "p_value": chi2_sf(q, lags), "distribution": "chi2",
                            "label": f"Ljung-Box Q({lags}) test of the {label_suffix}", "lags": lags},
              "jarque_bera": {"statistic": jb, "df": 2, "p_value": chi2_sf(jb, 2), "distribution": "chi2",
                               "label": f"Jarque-Bera normality test of the {label_suffix}", "skewness": skew, "kurtosis": kurtosis, "nobs": n}}
    if fitted_parameters and lags > fitted_parameters:
        adjusted = lags-fitted_parameters
        result["ljung_box"].update(df_adjusted=adjusted, p_value_adjusted=chi2_sf(q, adjusted))
    if arch_lags and n >= max(6, 2*arch_lags+3):
        factor = tree.finish()
        fit = least_squares(factor[:, :arch_lags+1], factor[:, -1])
        total = float(target_moments.m2.value[1]*target_moments.magnitude[1]**2)
        if not total > 0.:
            if allow_degenerate_aux:
                return result
            raise AnalysisError("constant_series", "Squared residuals are constant; ARCH-LM is undefined.")
        r2 = max(0., 1.-float(fit.ssr)/total)
        statistic = (n-arch_lags)*r2
        result["arch_lm"] = {"statistic": statistic, "df": arch_lags, "p_value": chi2_sf(statistic, arch_lags),
                             "distribution": "chi2", "label": f"ARCH-LM({arch_lags}) test of the {label_suffix}", "lags": arch_lags, "r_squared": r2, "nobs": n-arch_lags}
    return result
