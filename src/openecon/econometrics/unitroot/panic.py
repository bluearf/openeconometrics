"""Bai--Ng (2004) PANIC: fixed-factor decomposition and component unit-root tests."""

from __future__ import annotations

from functools import wraps
import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.unitroot.cips import _sample
from openecon.econometrics.unitroot.common import check_choice, check_count, check_flag
from openecon.econometrics.unitroot.critical import mackinnon_critical, mackinnon_p
from openecon.econometrics.unitroot.panic_kernels import (
    SOURCE, adf_statistics, common_sequence, decompose, idiosyncratic_probabilities,
)
from openecon.engines.distributions import normal_sf
from openecon.econometrics.unitroot.panic_selection import bridge_probabilities, select_factors
from openecon.econometrics.unitroot.series import adf_lag_choice


def _cpu_scope(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with torch.no_grad(), torch.device("cpu"):
            return function(*args, **kwargs)
    return wrapped


def _workspace(n: int, total: int, factors: int, lags: int, var_lags: int, requested: int, memory_mb: int):
    length, q = total - 1, min(n, total - 1)
    # This is an exact dense thin SVD, not a streaming/randomized approximation.
    # Explicitly count thin U/V and conservative live decomposition buffers.
    decomposition = 8 * (6 * n * length + q * (n + length) + q * q
                         + total * factors * (12 + 4 * var_lags)
                         + 4 * (factors * max(1, var_lags)) ** 2)
    per_panel = 8 * (total * (4 * (lags + 1) + 5) + 4 * (lags + 1) ** 2)
    budget = memory_mb * 1024**2
    if decomposition + per_panel > budget:
        raise AnalysisError("workspace_limit", "The dense PANIC PCA and one ADF block exceed memory_mb; increase the budget or restrict the panel.")
    work = n * length * q
    if work > 250_000_000:
        raise AnalysisError("work_limit", "This release caps exact dense PCA work at N*(T-1)*min(N,T-1) <=250000000; restrict the sample.")
    block = min(n, requested, (budget - decomposition) // per_panel)
    return block, decomposition + block * per_panel, work


@_cpu_scope
def xtpanic(
    data: Any,
    y: str,
    panel: str,
    time: str,
    *,
    factors: int | str = 1,
    lags: int | str = 0,
    max_factors: int = 6,
    maxlag: int | None = None,
    trend: str = "constant",
    method: str = "mqc",
    bandwidth: int | None = None,
    var_lags: int = 1,
    inference: str = "asymptotic",
    level: float = .05,
    pooling: str = "none",
    components: bool = False,
    block_size: int = 128,
    memory_mb: int = 64,
) -> TableSet:
    """PANIC tests of estimated common and idiosyncratic components (Bai--Ng 2004).

    ``factors`` is a positive fixed count or 'icp1'/'icp2'/'icp3' for the
    Bai--Ng (2002) information criteria over 0..max_factors. The selection
    sample is the PANIC difference matrix. Complete balanced common-date panels are
    required. ``trend='constant'`` extracts PCA from raw first differences;
    ``'trend'`` first removes each unit's mean difference. The cumulative
    estimated levels are t=2..T. Every idiosyncratic ADF has no deterministic
    terms. Its asymptotic no-constant DF p-value is valid only in the constant
    case. ``inference='bridge'`` explicitly calibrates trend-case idiosyncratic
    statistics by a recorded Brownian-bridge Monte Carlo limit. Fixed-count
    defaults remain unchanged. ``lags='aic'/'bic'/'t'`` selects each component's
    ADF lag on one common maximum-lag comparison sample, then refits it.

    With one factor, the common test is ADF with constant or constant+trend.
    With multiple factors, ``method='mqc'`` uses Bartlett-corrected VAR(1),
    ``'mqf'`` filters a finite VAR(var_lags) in differences. The published
    sequential rule estimates the number of stochastic trends. Table I
    supports m=1..6 and levels .01/.05/.10; no MQ p-value is fabricated.
    ``inference='none'`` returns all requested MQ statistics without decisions.

    ``pooling='independent'`` explicitly assumes independent idiosyncratic
    innovations across units for the standardized Fisher upper-tail normal
    limit. It is available only with constant/asymptotic individual inference.
    Weak cross-sectional dependence alone does not justify this pooled test.

    ``components=True`` adds estimated factors, loadings and idiosyncratic
    levels, measured from a zero initial anchor; they are not absolute fitted
    levels or an out-of-sample model. Float64 CPU computation is scoped to this
    call. ``memory_mb`` bounds conservative tensor-workspace estimates, not
    input/index/output frames or private BLAS allocation. Exact dense PCA has
    an additional cubic-work guard; this is not an out-of-core procedure.
    """
    factor_criterion = factors if isinstance(factors, str) else None
    if factor_criterion:
        check_choice(factors, "factors", ("icp1", "icp2", "icp3"))
        max_factors = check_count(max_factors, "max_factors", minimum=1)
        factors = max_factors
    else:
        factors = check_count(factors, "factors", minimum=1)
    lag_method = lags if isinstance(lags, str) else None
    if lag_method:
        check_choice(lags, "lags", ("aic", "bic", "t"))
        lags = 0 if maxlag is None else check_count(maxlag, "maxlag")
    else:
        lags = check_count(lags, "lags")
        if maxlag is not None:
            raise AnalysisError("invalid_option", "maxlag requires automatic lags.")
    var_lags = check_count(var_lags, "var_lags")
    requested = check_count(block_size, "block_size", minimum=1)
    memory_mb = check_count(memory_mb, "memory_mb", minimum=1)
    check_choice(trend, "trend", ("constant", "trend"))
    check_choice(method, "method", ("mqc", "mqf"))
    check_choice(inference, "inference", ("asymptotic", "none", "bridge"))
    if inference == "bridge" and trend != "trend":
        raise AnalysisError("unsupported_inference", "inference='bridge' requires trend='trend'.")
    check_choice(pooling, "pooling", ("none", "independent"))
    check_flag(components, "components")
    if not isinstance(level, (int, float)) or isinstance(level, bool) or level not in (.01, .05, .1):
        raise AnalysisError("invalid_option", "level must be .01, .05 or .10.")
    if requested > 512 or memory_mb > 1024 or factors > 32 or lags > 64 or var_lags > 16:
        raise AnalysisError("invalid_option", "Bounds are block_size<=512, memory_mb<=1024, factors<=32, lags<=64 and var_lags<=16.")
    if bandwidth is not None:
        bandwidth = check_count(bandwidth, "bandwidth")
    if method == "mqc" and var_lags != 1:
        raise AnalysisError("invalid_option", "var_lags is only configurable for method='mqf'.")
    if method == "mqf" and bandwidth is not None:
        raise AnalysisError("invalid_option", "bandwidth is only used for method='mqc'.")
    if not factor_criterion and factors == 1 and (bandwidth is not None or var_lags != 1 or method != "mqc"):
        raise AnalysisError("invalid_option", "With one factor the common test is ADF; use default MQ options (method='mqc', bandwidth=None, var_lags=1).")
    if pooling != "none" and not ((trend == "constant" and inference == "asymptotic") or inference == "bridge"):
        raise AnalysisError("unsupported_inference", "Pooled PANIC requires constant-case asymptotic idiosyncratic DF p-values and the independence opt-in.")
    if inference != "none" and factors > 6:
        raise AnalysisError("critical_values_unavailable", "Published MQ quantiles cover at most six common factors; use inference='none' for statistics only.")
    try:
        values, labels, time_range = _sample(data, y, panel, time)
    except AnalysisError as exc:
        # Keep the shared complete-calendar checks, but name this procedure.
        raise AnalysisError(exc.code, str(exc).replace("CADF/CIPS", "PANIC")) from exc
    n, total = values.shape
    if lag_method and maxlag is None:
        lags = min(12, int(12 * ((total-1)/100)**.25), (total-6)//2)
    factor_selection = None
    if factor_criterion:
        _workspace(n, total, factors, lags, var_lags, requested, memory_mb)
        factors, factor_selection = select_factors(values, trend, factor_criterion, max_factors)
    if factors >= min(n, total - 1):
        raise AnalysisError("invalid_factor_count", "factors must be smaller than both N and T-1 so idiosyncratic variation remains.")
    if total - 2 - lags <= 1 + lags + (2 if factors == 1 and trend == "trend" else 1 if factors == 1 else 0):
        raise AnalysisError("insufficient_observations", "The estimated component ADF sample cannot identify the requested lag/deterministic terms.")
    block, estimated, work = _workspace(n, total, factors, lags, var_lags, requested, memory_mb)
    if inference == "bridge":
        # Persistent sorted simulation plus one 128-by-512 float64 draw block,
        # squared draws, and sorting/quantile buffers: conservative 4 MiB.
        estimated += 4*1024**2
        if estimated > memory_mb*1024**2:
            raise AnalysisError("workspace_limit", "PANIC Brownian-bridge calibration needs an additional 4 MiB within memory_mb.")
    if bandwidth is None:
        bandwidth = min(total - 2, int(4 * (total / 100) ** (2 / 9)))
    if factors > 1 and method == "mqc" and bandwidth >= total - 1:
        raise AnalysisError("invalid_lags", "Bartlett bandwidth must be smaller than T-1.")
    if factors > 1 and method == "mqf" and total - 1 - var_lags <= factors * var_lags:
        raise AnalysisError("insufficient_observations", "The common-factor difference VAR has too few observations for its requested order.")
    try:
        scores, loadings, idio, spectrum, scale = decompose(values, factors, trend)
        idio_statistics = torch.empty(n, dtype=torch.float64)
        idio_lags = [adf_lag_choice(row, "n", lag_method, lags) for row in idio] if lag_method else [lags]*n
        idio_nobs = [total-2-lag for lag in idio_lags]
        for chosen in sorted(set(idio_lags)):
            indexes = torch.tensor([i for i, lag in enumerate(idio_lags) if lag == chosen], dtype=torch.int64)
            for first in range(0, len(indexes), block):
                index = indexes[first:first+block]
                idio_statistics[index], _ = adf_statistics(idio[index], chosen, "none")
        common_levels = scores.cumsum(dim=0)
        common_lags = adf_lag_choice(common_levels[:, 0], "c" if trend == "constant" else "ct", lag_method, lags) if lag_method and factors == 1 else lags
        if factors == 0:
            common_rows, selected = [], 0
        elif factors == 1:
            statistics, common_nobs = adf_statistics(common_levels.T, common_lags, trend)
            statistic = float(statistics[0])
            case = "c" if trend == "constant" else "ct"
            p_value = mackinnon_p(statistic, case) if inference != "none" else None
            critical = mackinnon_critical(case, 1, math.inf) if inference != "none" else None
            reject = statistic < critical[{.01: "1%", .05: "5%", .1: "10%"}[level]] if critical else None
            common_rows = [{"test": "ADF", "stochastic_trends_null": 1, "statistic": statistic,
                            "p_value": p_value, "nobs": common_nobs,
                            **{f"critical_{key}": value for key, value in (critical or {"1%": None, "5%": None, "10%": None}).items()},
                            "reject": reject}]
            selected = (0 if reject else 1) if reject is not None else None
        else:
            common_rows, selected = common_sequence(common_levels, total, trend, method, bandwidth, var_lags, "asymptotic" if inference == "bridge" else inference, float(level))
    except torch.linalg.LinAlgError as exc:
        raise AnalysisError("numerical_failure", "PANIC's factor/VAR matrices could not be resolved in float64; rescale or restrict the sample.") from exc
    bridge_metadata = None
    idio_critical = None
    if inference == "bridge":
        idio_p, idio_logp, idio_critical, bridge_metadata = bridge_probabilities(idio_statistics)
    elif trend == "constant" and inference == "asymptotic":
        idio_p, idio_logp = idiosyncratic_probabilities(idio_statistics)
        idio_critical = mackinnon_critical("n", 1, math.inf)
    else:
        idio_p = idio_logp = None
    idio_rows = [{"panel": label, "statistic": float(idio_statistics[i]),
                  "p_value": float(idio_p[i]) if idio_p is not None else None,
                  "nobs": idio_nobs[i], "lags": idio_lags[i],
                  **{f"critical_{key}": value for key, value in (idio_critical or {"1%": None, "5%": None, "10%": None}).items()},
                  "reject": float(idio_statistics[i]) < idio_critical[{.01: "1%", .05: "5%", .1: "10%"}[level]] if idio_critical else None}
                 for i, label in enumerate(labels)]
    notes = [
        "Large-N, large-T inference conditional on a fixed, correctly specified number of pervasive factors; no finite-sample or Stata parity claim.",
        "Individual tests permit the paper's weak idiosyncratic cross-sectional dependence; pooled inference additionally assumes independence.",
        "ADF uses estimated cumulative levels t=2..T; MQ uses a zero t=1 anchor, the same deterministic fit extrapolated to that anchor, original T scaling, and valid lag rows only.",
    ]
    if factor_selection is not None:
        notes.append("ICp factor selection assumes pervasive factors and large N,T; selection uncertainty is not a finite-sample confidence adjustment.")
    if inference == "bridge":
        notes.append("Brownian-bridge calibration is a Monte Carlo asymptotic limit with recorded CDF simulation error; pooled Fisher additionally requires independent idiosyncratic components.")
    if trend == "trend" and inference != "bridge":
        notes.append("Trend-case idiosyncratic ADF has a Brownian-bridge limit, not a Dickey-Fuller limit; p-values and critical decisions are unavailable here.")
    if factors > 1:
        notes.append("MQ has only the published 1%, 5%, 10% asymptotic quantiles, no interpolated p-value.")
        if method == "mqf":
            notes.append("MQ_f requires a finite-order VAR representation with the supplied order at least the true order.")
        else:
            notes.append("MQ_c uses Bartlett one-sided VAR(1)-residual covariances; the bandwidth must grow sufficiently slowly under the paper's asymptotics.")
        if trend == "trend" and factors >= 5 and inference == "asymptotic":
            notes.append("Table I's printed trend/m=5/10% quantile -55.286 is preserved; it is not silently amended.")
    if inference == "none":
        notes.append("Inference disabled: component statistics have no p-values, quantiles or stochastic-trend estimate.")
    result = TableSet({
        "common": table(common_rows, distribution="Dickey-Fuller asymptotic" if factors == 1 else "Bai-Ng Table I asymptotic MQ", label="PANIC common-component tests"),
        "idiosyncratic": table(idio_rows, distribution="Dickey-Fuller no constant" if trend == "constant" else "Brownian-bridge Monte Carlo" if inference == "bridge" else "Brownian-bridge functional (not calibrated)", label="PANIC idiosyncratic-component tests"),
    }, title="Bai-Ng PANIC", procedure="xtpanic", n_panels=n, n_periods=total,
        nobs=n * total, time_range=time_range, factor_count=factors,
        stochastic_trends=selected, lags=lags, trend=trend,
        common_method="adf" if factors == 1 else method,
        bandwidth=bandwidth if factors > 1 and method == "mqc" else None,
        var_lags=var_lags if factors > 1 and method == "mqf" else None,
        inference=inference, level=float(level), pooling=pooling,
        factor_selection=factor_selection, bridge_calibration=bridge_metadata,
        lag_selection={"method": lag_method or "fixed", "maximum": lags if lag_method else None,
                       "chosen_idiosyncratic": idio_lags, "chosen_common_adf": common_lags if factors == 1 else None,
                       "comparison_common_sample": [lags+3, total] if lag_method else None},
        notes=notes, workspace={"panel_block_size": block,
                              "estimated_tensor_workspace_bytes": estimated,
                              "budget_bytes": memory_mb * 1024**2,
                              "dense_pca_work_proxy": work,
                              "dense_pca_work_limit": 250_000_000,
                              "excludes": ["caller input", "panel index arrays", "result frames", "private BLAS workspace"]},
        provenance={"source": SOURCE, "critical_table": "Table I, p.1136" if factors > 1 else "Asymptotic Dickey-Fuller / MacKinnon response surface",
                    "dtype": "float64", "device": "cpu", "stata_parity_validated": False})
    if pooling == "independent":
        # Do not clip an approximate distribution's zero p-value to a fake
        # finite tail or emit non-finite JSON. Require statistics-only use.
        if not bool(torch.isfinite(idio_logp).all()):
            raise AnalysisError("pooled_tail_unavailable", "An individual asymptotic p-value is zero at this response surface's tail boundary; a finite pooled Fisher statistic cannot be recovered. Use pooling='none'.")
        fisher = float(-2 * idio_logp.sum())
        standardized = (fisher - 2 * n) / math.sqrt(4 * n)
        result["pooled"] = table([{"test": "standardized Fisher", "statistic": standardized,
                                   "p_value": normal_sf(standardized), "fisher": fisher,
                                   "n_panels": n}], distribution="asymptotic standard normal upper tail", null="All idiosyncratic components have unit roots", independence_assumed=True)
    if components:
        restored_loadings, restored_idio = loadings * scale, idio * scale
        if not bool(torch.isfinite(restored_loadings).all()) or not bool(torch.isfinite(restored_idio).all()):
            raise AnalysisError("numerical_failure", "Original-unit components overflow float64; rescale before requesting components.")
        result["factors"] = table(common_levels.tolist(), columns=[f"factor_{j + 1}" for j in range(factors)], index=list(range(2, total + 1)), index_label="original period position")
        result["loadings"] = table({"panel": labels, **{f"factor_{j + 1}": restored_loadings[:, j].tolist() for j in range(factors)}})
        result["idiosyncratic_levels"] = table({"panel": [label for label in labels for _ in range(total - 1)],
                                              "period_position": list(range(2, total + 1)) * n,
                                              "component": restored_idio.flatten().tolist()}, zero_initial_anchor=True)
        result["spectrum"] = table({"singular_value_scaled": spectrum.tolist()}, global_measurement_scale=scale)
    return result
