"""One externally identified VAR shock with explicit response normalization.

Montiel Olea, Stock and Watson (2021), equations (2.7)--(2.9):
Gamma = Cov(u,z) is proportional to one impact column; its scale is set by
the response of a declared anchor. Point responses only, no weak-IV inference.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral

import torch

from openecon.analysis import _coerce_frame, _frame_hasher, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from openecon.econometrics.var import kernels
from openecon.econometrics.var.common import load_sample, require_result
from openecon.resources import plan_workspace, tensor_bytes


def _count(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
        raise AnalysisError("invalid_option", f"{name} must be an integer in {minimum}..{maximum}.")
    return int(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise AnalysisError("invalid_option", f"{name} requires nonblank text up to 2000 characters.")
    return value


def _keys(frame, name):
    if name not in frame or frame.columns.has_duplicates:
        raise AnalysisError("invalid_proxy_keys", "A complete unique key column must be present.")
    raw = frame[name].tolist()
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in raw):
        raise AnalysisError("invalid_proxy_keys", "Proxy alignment requires exact integer time keys; no rounding or string/date coercion is performed.")
    keys = [int(v) for v in raw]
    if len(keys) != len(set(keys)):
        raise AnalysisError("invalid_proxy_keys", "Proxy time keys must be unique.")
    return keys


def _responses(a, impact, steps, max_work):
    k, p = a.shape[1], a.shape[0]
    required = 8 * (steps + 1) * p * k**3
    if required > max_work:
        raise AnalysisError("work_limit", "Requested proxy responses exceed max_work; no horizons are truncated.")
    plan_workspace("saved proxy VAR impulse responses", {
        "ma_matrices_and_response_state": tensor_bytes((steps + 1, k, k), itemsize=40),
    })
    phi = [torch.eye(k, dtype=torch.float64)]
    for h in range(1, steps + 1):
        value = torch.zeros((k, k), dtype=torch.float64)
        for lag in range(1, min(h, p) + 1):
            value += a[lag - 1] @ phi[h - lag]
        phi.append(value)
    responses = torch.stack(phi) @ impact
    if not bool(torch.isfinite(responses).all()):
        raise AnalysisError("numerical_failure", "Proxy VAR propagation exceeds finite float64 arithmetic.")
    return responses


def _table(state, steps, max_work):
    names = state["variables"]
    values = _responses(torch.tensor(state["a"], dtype=torch.float64),
                        torch.tensor(state["impact_vector"], dtype=torch.float64), steps, max_work)
    return table(
        [[h, name, float(values[h, j])] for h in range(steps + 1) for j, name in enumerate(names)],
        columns=["horizon", "response", "irf"],
        shock="externally identified single shock",
        normalization=state["normalization"],
        inference="unavailable; weak-proxy robust uncertainty requires separate validation",
        fevd_available=False,
        source_var_refitted=False,
    )


@resident_cpu
def proxy_svar(result, *, data, proxy_data, proxy, key, normalize, source,
               exogeneity_assumed, impact=1.0, steps=8, tolerance=1e-12,
               max_work=100000000):
    """Identify one saved VAR impact column from a complete external proxy.

    Requires the exact hashed original VAR table and a separate proxy table
    with exactly the retained estimation integer time keys, in any order.
    Numeric constant-only VARs with no trend/exogenous variables, 2..12
    equations, 1..12 lags and 20..8192 retained rows are admitted. Original
    missing='drop' lag boundaries are preserved; missing proxies are refused.
    ``exogeneity_assumed=True`` declares exclusion of all other structural
    shocks; it does not establish exclusion or relevance from the data.
    ``normalize`` names the response whose impact is exactly ``impact``.
    No other shock, FEVD, historical decomposition or confidence band is
    supplied. Raw covariance and descriptive anchor R-squared are saved,
    without a strength cutoff or a weak-instrument calibration claim.
    """
    result = require_result(result, ("var",), "proxy_svar")
    s = result.spec
    if (s.predictors or s.categorical or not s.time or s.panel or s.weights
            or s.options.get("trend", False) or not s.intercept
            or not s.options.get("constant", True)):
        raise AnalysisError("unsupported_spec", "Proxy identification supports numeric constant-only VARs with a saved integer time column; no exogenous terms or trend.")
    source = _text(source, "source")
    proxy, key, normalize = (_text(v, name) for v, name in ((proxy, "proxy"), (key, "key"), (normalize, "normalize")))
    if key != s.time or proxy == key:
        raise AnalysisError("invalid_proxy_keys", "key must equal the saved VAR time column and differ from proxy.")
    if exogeneity_assumed is not True:
        raise AnalysisError("identification_assumption_required", "Declare exogeneity_assumed=True only when the supplied proxy is assumed orthogonal to every other structural shock.")
    if isinstance(impact, bool) or not isinstance(impact, (int, float)) or not math.isfinite(impact) or impact == 0:
        raise AnalysisError("invalid_normalization", "impact must be a finite nonzero response normalization.")
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance) or not 1e-14 <= tolerance <= 1e-6):
        raise AnalysisError("invalid_option", "tolerance must lie in 1e-14..1e-6; this is a numerical denominator guard, not a weak-IV cutoff.")
    steps = _count(steps, "steps", 0, 200)
    max_work = _count(max_work, "max_work", 1, 1000000000000)
    if isinstance(data, Dataset) or isinstance(proxy_data, Dataset):
        raise AnalysisError("streaming_unsupported", "Proxy VAR identification requires resident tables; Dataset inputs are never collected.")
    layout = result.extra.get("layout", {})
    names = layout.get("variables", [])
    k, p, n = len(names), layout.get("lags"), result.nobs
    original_rows = result.nobs_original
    if (not 2 <= k <= 12 or isinstance(p, bool) or not isinstance(p, int)
            or not 1 <= p <= 12 or not 20 <= n <= 8192
            or not n + p <= original_rows <= 32768):
        raise AnalysisError("resource_limit", "Proxy identification admits 2..12 equations, 1..12 lags, 20..8192 retained rows and at most 32768 original rows.")
    if normalize not in names:
        raise AnalysisError("invalid_normalization", "normalize must name one saved endogenous response.")
    m = k*p + 1
    required = 8 * (original_rows*(k+2) + n*m*k + n*(k+1)**2 + (steps+1)*p*k**3)
    if required > max_work:
        raise AnalysisError("work_limit", "Saved residual reconstruction and proxy responses exceed max_work.")
    plan_workspace("proxy VAR keyed alignment, residuals and complete state", {
        "original_input_copies": tensor_bytes((original_rows, k + 1), itemsize=64),
        "lagged_design_and_residuals": tensor_bytes((n, m + k), itemsize=64),
        "complete_proxy_residual_json_state": tensor_bytes((n, k + 3), itemsize=96),
        "full_moments": tensor_bytes((k + 1, k + 1), itemsize=64),
    })
    original = _coerce_frame(data)
    if len(original) > 32768:
        raise AnalysisError("resource_limit", "Proxy source verification supports at most 32768 original rows, including excluded endpoints.")
    if len(original) != result.nobs_original:
        raise AnalysisError("sample_mismatch", "Supply the original VAR model rows, not only its estimation sample.")
    sample = load_sample(s, original)
    if _frame_hasher(sample.frame.original).hexdigest() != result.provenance.get("data_hash"):
        raise AnalysisError("sample_mismatch", "Original VAR model columns, physical rows/order and values must match the saved data hash.")
    if sample.frame.positions[p:] != result.sample_positions:
        raise AnalysisError("sample_mismatch", "Saved VAR estimation positions differ from the original lagged sample.")
    retained_keys = _keys(sample.frame.sample.iloc[p:], key)
    instrument = _coerce_frame(proxy_data)
    supplied_keys = _keys(instrument, key)
    if len(supplied_keys) != n or set(supplied_keys) != set(retained_keys):
        raise AnalysisError("sample_mismatch", "Proxy table must contain exactly every retained VAR estimation key; extra, missing or duplicated periods are refused.")
    if proxy not in instrument:
        raise AnalysisError("missing_columns", "The named external proxy column is absent.")
    index = {value: row for row, value in enumerate(supplied_keys)}
    aligned = instrument.iloc[[index[value] for value in retained_keys]][[key, proxy]].reset_index(drop=True)
    z = _numeric(aligned[proxy], proxy)
    if not bool(torch.isfinite(z).all()):
        raise AnalysisError("missing_values", "Every retained proxy value must be finite; the saved VAR sample is never rescreened.")
    levels = sample.levels
    design = torch.cat((kernels.lag_block(levels, p), torch.ones((n, 1), dtype=torch.float64)), dim=1)
    coefficients = torch.tensor([entry.estimate for entry in result.coefficients], dtype=torch.float64)
    if coefficients.numel() != k*m or layout.get("n_regressors") != m:
        raise AnalysisError("invalid_result", "Saved VAR coefficient layout is incompatible with the admitted constant-only design.")
    coefficients = coefficients.reshape(k, m)
    u = levels[p:] - design @ coefficients.T
    centered_z, centered_u = z-z.mean(), u-u.mean(0)
    joined = torch.cat((centered_z[:, None], centered_u), dim=1)
    moments = joined.T @ joined / n
    if not bool(torch.isfinite(moments).all()):
        raise AnalysisError("numerical_failure", "Proxy and residual moments exceed finite float64 arithmetic.")
    proxy_variance = float(moments[0, 0])
    gamma = moments[0, 1:]
    anchor = names.index(normalize)
    scale = math.sqrt(proxy_variance)*math.sqrt(float(moments[anchor + 1, anchor + 1]))
    denominator = float(gamma[anchor])
    if scale == 0 or abs(denominator) <= tolerance*scale:
        raise AnalysisError("unidentified_normalization", "Anchor-proxy covariance is zero within the declared relative numerical tolerance; no ratio is returned. This is not a calibrated weak-instrument test.")
    vector = gamma / denominator * float(impact)
    if not bool(torch.isfinite(vector).all()):
        raise AnalysisError("numerical_failure", "Normalized proxy impact ratios exceed finite float64 arithmetic.")
    a = kernels.lag_matrices(coefficients, k, p)
    state = {
        "schema": "openecon.proxy_svar.v1",
        "variables": names, "lags": p, "a": a.tolist(),
        "impact_vector": vector.tolist(), "gamma": gamma.tolist(),
        "normalization": {"response": normalize, "impact": float(impact),
                          "denominator": denominator, "relative_numerical_tolerance": float(tolerance)},
        "complete_joint_centered_moments": moments.tolist(),
        "moment_order": [proxy, *names],
        "proxy_mean": float(z.mean()), "residual_means": u.mean(0).tolist(),
        "proxy_values": z.tolist(), "residuals": u.tolist(),
        "sample_keys": retained_keys, "sample_positions": result.sample_positions,
        "source_var_id": result.id, "source_data_hash": result.provenance["data_hash"],
        "proxy_data_hash": _frame_hasher(aligned).hexdigest(),
        "source_result_json_hash": hashlib.sha256(result.model_dump_json().encode()).hexdigest(),
        "proxy_source": source, "proxy_column": proxy, "key_column": key,
        "nobs": n, "nobs_original": original_rows, "exogeneity_assumed": True,
        "assumptions": ["External proxy is uncorrelated with every non-target structural shock",
                        "Nonzero covariance between proxy and target shock",
                        "Reduced-form VAR correctly spans the innovations; structural shock is recoverable",
                        "Constant impact relationship over the retained sample"],
        "anchor_correlation": denominator/scale,
        "descriptive_anchor_r_squared": (denominator/scale)**2,
        "instrument_strength_calibrated": False,
        "inference_available": False, "fevd_available": False,
        "source_var_refitted": False, "device": "cpu", "precision": "float64",
        "stability": result.extra.get("stability"),
    }
    output = TableSet({
        "responses": _table(state, steps, max_work),
        "impact": table([[name, float(vector[j]), float(gamma[j])] for j, name in enumerate(names)],
                         columns=["response", "impact", "covariance_with_proxy"]),
        "diagnostics": table([["nobs", n], ["proxy_variance", proxy_variance],
                              ["anchor_covariance", denominator], ["anchor_correlation", denominator/scale],
                              ["descriptive_anchor_r_squared", (denominator/scale)**2]],
                             columns=["diagnostic", "value"]),
    }, title="Externally identified single VAR shock", method="proxy_svar", proxy_state=state,
        inference_available=False, fevd_available=False, source_var_refitted=False)
    return saved_summary(output)


@resident_cpu
def proxy_svar_irf(output, *, steps=8, max_work=100000000):
    """Replay point responses from complete saved single-shock state; no fitting."""
    steps = _count(steps, "steps", 0, 200)
    max_work = _count(max_work, "max_work", 1, 1000000000000)
    if not isinstance(output, TableSet) or output.attrs.get("method") != "proxy_svar":
        raise AnalysisError("invalid_result", "Supply a proxy_svar TableSet or its restored summary state.")
    state = output.attrs.get("proxy_state", {})
    try:
        names, p = state["variables"], state["lags"]
        if (not isinstance(names, list) or not 2 <= len(names) <= 12
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)
                or isinstance(p, bool) or not isinstance(p, int) or not 1 <= p <= 12
                or len(state["a"]) != p
                or any(len(block) != len(names) or any(len(row) != len(names) for row in block) for block in state["a"])
                or len(state["impact_vector"]) != len(names)):
            raise ValueError("Invalid state dimensions")
        a = torch.tensor(state["a"], dtype=torch.float64)
        impact = torch.tensor(state["impact_vector"], dtype=torch.float64)
        normal = state["normalization"]
        if (state["schema"] != "openecon.proxy_svar.v1"
                or a.shape != (p, len(names), len(names)) or impact.shape != (len(names),)
                or not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(impact).all())
                or normal["response"] not in names or isinstance(normal["impact"], bool)
                or not isinstance(normal["impact"], (int, float)) or not math.isfinite(normal["impact"])
                or normal["impact"] == 0
                or abs(float(impact[names.index(normal["response"])])-normal["impact"]) > 1e-12*abs(normal["impact"])):
            raise ValueError("Invalid state")
    except (KeyError, ValueError, TypeError, RuntimeError):
        raise AnalysisError("invalid_state", "Saved proxy schema, variable order or finite matrix dimensions are invalid.") from None
    return _table(state, steps, max_work)
