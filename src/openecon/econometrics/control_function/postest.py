"""Complete saved conditional means with full generated-control uncertainty."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

import pandas as pd
import torch

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import kernel_call, table
from openecon.econometrics.mi.common import _decode_index
from openecon.engines.inference import critical_value
from openecon.models import ResultBundle
from openecon.resources import plan_workspace

from .kernels import mean_link
from .state import MAX_ROWS, MAX_WORK, _admit, _design, _metadata_size, _real, _tensor, fail, validate_state, work_limit


def _restore(result, max_work):
    if isinstance(result, str):
        if len(result) > 64 * 1024**2:
            fail("Saved control-function JSON exceeds its bounded input size.", "resource_limit")
        plan_workspace("control-function JSON admission", {"JSON decoding and model restoration": 8 * len(result)})
        try:
            result = json.loads(result)
        except (ValueError, TypeError, RecursionError, OverflowError) as exc:
            raise AnalysisError("invalid_result", "Supply finite saved control-function JSON.") from exc
    if isinstance(result, ResultBundle):
        state = result.extra.get("control_function_state") if isinstance(result.extra, Mapping) else None
        _admit(state, max_work)
        if (len(result.coefficients) > 36 or len(result.predictions) > 400
                or len(result.covariance_matrix) > 36 or len(result.spec.options) > 5):
            fail("The saved result envelope exceeds declared coefficient/preview/option bounds.")
        return result
    if not isinstance(result, Mapping):
        fail("cf_predict requires a control-function result, saved dictionary or JSON.", "invalid_result")
    extra = result.get("extra")
    state = extra.get("control_function_state") if isinstance(extra, Mapping) else None
    _admit(state, max_work)
    size = _metadata_size(result)
    plan_workspace("control-function ResultBundle restoration", {"complete finite JSON and validated model copies": 4 * size})
    try:
        return ResultBundle.model_validate(result)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise AnalysisError("invalid_result", "Saved control-function ResultBundle is malformed.") from exc


def _query_shape(data):
    """Admit a resident envelope before pandas conversion, projection or copies."""
    if isinstance(data, Dataset):
        fail("Conditional mean prediction needs a resident table; Dataset is not collected.", "streaming_unsupported")
    if isinstance(data, pd.DataFrame):
        n, columns = data.shape
    elif isinstance(data, Mapping):
        if not data or len(data) > 128:
            fail("Prediction data need a bounded resident column mapping.", "invalid_data")
        lengths = []
        for value in data.values():
            if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
                fail("Prediction columns must contain complete row sequences.", "invalid_data")
            lengths.append(len(value))
        if len(set(lengths)) != 1:
            fail("Prediction columns must have equal lengths.", "invalid_data")
        n, columns = lengths[0], len(data)
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        n = len(data)
        if not 1 <= n <= MAX_ROWS or any(not isinstance(v, Mapping) or len(v) > 128 for v in data):
            fail("Prediction records must contain 1..5000 bounded resident rows.", "invalid_data")
        columns = len({name for row in data for name in row})
    else:
        fail("Prediction data must be a resident DataFrame, columns or row records.", "invalid_data")
    if not 1 <= n <= MAX_ROWS or not 1 <= columns <= 128:
        fail("Conditional mean prediction admits 1..5000 resident rows and at most 128 input columns.", "dimension_limit")
    plan_workspace("control-function resident prediction input", {"resident conversion, projection and missing admission": 192 * n * columns})
    return n


@torch.no_grad()
def cf_predict(*, result: ResultBundle | Mapping | str, data: Any = None,
               missing: str = "raise", alpha: float | None = None,
               max_work: int = MAX_WORK) -> pd.DataFrame:
    """Saved conditional means, joint Gamma/Beta delta SEs and link-scale CIs.

    New data require the observed endogenous column, included exogenous
    predictors and excluded instruments. Confidence intervals concern a
    conditional mean; they are not outcome prediction or intervention intervals.
    The full row covariance is represented exactly by J and joint C in attrs.
    """
    with torch.device("cpu"):
        return _predict(result, data, missing, alpha, max_work)


def _predict(result, data, missing, alpha, max_work):
    max_work = work_limit(max_work)
    if not isinstance(missing, str) or missing not in {"raise", "drop"}:
        fail("missing must be 'raise' or 'drop'.", "invalid_option")
    if alpha is not None and (not _real(alpha) or not 0 < alpha < 1):
        fail("alpha must be a finite real probability in (0,1).", "invalid_option")
    bundle = _restore(result, max_work)
    state = bundle.extra["control_function_state"]
    from .stream_state import is_stream_state
    if is_stream_state(state):
        if data is None or isinstance(data, Dataset):
            fail("Compact Dataset fits use predict(result, explicit evaluation data); cf_predict requires a resident query table.", "unsupported_prediction")
        return _predict_compact(bundle, data, missing, alpha, max_work)
    n, kz, kx, replay_work, _ = _admit(state, max_work)
    count = n if data is None else _query_shape(data)
    width = kz + kx + 1
    query_work = 32 * count * (width * width + width + kz + kx)
    if replay_work + query_work > max_work:
        fail("Complete semantic replay and conditional prediction exceed max_work.", "resource_limit")
    plan = plan_workspace("control-function full conditional mean prediction", {
        "source replay and query designs": 256 * (n * width + count * (kz + kx + 2)),
        "joint covariance, both Jacobians and output rows": 256 * (width * width + count * (2 * width + 16)),
    })
    state = validate_state(bundle, max_work=max_work - query_work)
    spec = bundle.spec
    alpha = spec.alpha if alpha is None else float(alpha)
    gamma, beta = _tensor(state["gamma"], "gamma"), _tensor(state["beta"], "beta")
    covariance = _tensor(state["joint_covariance"], "joint covariance")
    if data is None:
        positions = list(state["sample_positions"])
        source_index = _decode_index(state["source_index"])
        index = source_index.take(positions)
        z, x, d = (_tensor(state[name], name) for name in ("z", "x", "d"))
        original_count = state["n_original"]
    else:
        source = _coerce_frame(data)
        if source.columns.has_duplicates:
            fail("Prediction data column names must be unique.", "invalid_data")
        endogenous = spec.columns["endogenous"]
        required = list(dict.fromkeys([endogenous, *spec.predictors, *spec.columns["instruments"]]))
        if any(name not in source for name in required):
            fail("New data need the observed endogenous, exogenous and excluded-instrument columns.", "missing_column")
        selected = source.loc[:, required]
        complete = ~selected.isna().any(axis=1)
        if missing == "raise" and not bool(complete.all()):
            fail("Prediction inputs contain missing endogenous/predictor/instrument values.", "missing_values")
        positions = [int(i) for i in complete.to_numpy().nonzero()[0]]
        if not positions:
            fail("No complete rows remain for conditional prediction.", "insufficient_data")
        index = source.index.take(positions)
        selected = selected.iloc[positions]
        z = _design(selected, state["z_terms"], spec.intercept)
        x = _design(selected, state["x_terms"], spec.intercept)
        d = _numeric(selected[endogenous], endogenous)
        original_count = len(source)
    residual = d - z @ gamma
    q = torch.cat([x, residual[:, None]], 1)
    eta = q @ beta
    if not bool(torch.isfinite(eta).all()) or not bool(torch.isfinite(residual).all()):
        fail("Conditional residual/index exceeds finite float64 support.", "numerical_domain")
    mean, derivative = kernel_call(mean_link, state["kind"], eta)
    jacobian = torch.cat([-beta[-1] * z, q], 1)
    variance = ((jacobian @ covariance) * jacobian).sum(1)
    zero_gradient = (jacobian == 0).all(1)
    if (not bool(torch.isfinite(variance).all()) or not bool((variance >= 0).all())
            or bool(((variance == 0) & ~zero_gradient).any())):
        fail("Conditional mean uncertainty is nonfinite or unresolved; covariance is not repaired.", "numerical_domain")
    eta_se = variance.sqrt()
    mean_se = derivative * eta_se
    mean_jacobian = derivative[:, None] * jacobian
    critical = kernel_call(critical_value, alpha, None)
    low = kernel_call(mean_link, state["kind"], eta - critical * eta_se)[0]
    high = kernel_call(mean_link, state["kind"], eta + critical * eta_se)[0]
    if (any(not bool(torch.isfinite(v).all()) for v in (mean_se, mean_jacobian, jacobian, low, high))
            or bool(((mean_se == 0) & ~zero_gradient).any())):
        fail("Conditional mean uncertainty or confidence bounds leave finite link support.", "numerical_domain")
    result_table = table({"position": positions, "mean": mean.tolist(), "std_error": mean_se.tolist(),
                          "ci_low": low.tolist(), "ci_high": high.tolist(), "eta": eta.tolist(),
                          "eta_std_error": eta_se.tolist(), "control_residual": residual.tolist()},
                         conditional_mean_only=True, interval_definition="asymptotic normal eta interval transformed through mean link",
                         full_gamma_beta_covariance=True, source_state_sha256=state["integrity_sha256"],
                         kind=state["kind"], alpha=alpha, missing=missing, input_positions=positions,
                         nobs_original=original_count, nobs=len(positions), excluded_rows=original_count-len(positions),
                         inference=bundle.inference, parameter_order=state["parameter_order"],
                         mean_jacobian=mean_jacobian.tolist(), eta_jacobian=jacobian.tolist(),
                         joint_parameter_covariance=covariance.tolist(),
                         covariance_representation="J C J' using recorded Jacobian and full joint parameter covariance",
                         work_estimate=replay_work+query_work, resource_plan=plan.record(),
                         structural_effects_identified=False, weak_instrument_robust=False,
                         finite_cluster_exact=False, stata_parity_validated=False)
    result_table.index = index.copy()
    return result_table


def _predict_compact(bundle, data, missing, alpha, max_work):
    """Retain the resident CF helper's columns, link CIs and full Jacobians."""
    from .stream_state import admit_stream_state
    state = bundle.extra["control_function_state"]
    n, kz, kx, replay_work, _ = admit_stream_state(state, max_work)
    count = _query_shape(data)
    width = kz + kx + 1
    query_work = 32 * count * (width * width + width + kz + kx)
    if query_work + replay_work > max_work:
        fail("Compact source replay and resident prediction exceed max_work.", "resource_limit")
    state = validate_state(bundle, max_work=max_work - query_work)
    spec = bundle.spec
    alpha = spec.alpha if alpha is None else float(alpha)
    source = _coerce_frame(data)
    if source.columns.has_duplicates:
        fail("Prediction data column names must be unique.", "invalid_data")
    endogenous = spec.columns["endogenous"]
    required = list(dict.fromkeys([endogenous, *spec.predictors, *spec.columns["instruments"]]))
    if any(name not in source for name in required):
        fail("New data need the observed endogenous, exogenous and excluded-instrument columns.", "missing_column")
    selected = source.loc[:, required]
    complete = ~selected.isna().any(axis=1)
    if missing == "raise" and not bool(complete.all()):
        fail("Prediction inputs contain missing endogenous/predictor/instrument values.", "missing_values")
    positions = [int(i) for i in complete.to_numpy().nonzero()[0]]
    if not positions:
        fail("No complete rows remain for conditional prediction.", "insufficient_data")
    selected = selected.iloc[positions]
    z = _design(selected, state["z_terms"], spec.intercept)
    x = _design(selected, state["x_terms"], spec.intercept)
    d = _numeric(selected[endogenous], endogenous)
    gamma, beta = _tensor(state["gamma"], "gamma"), _tensor(state["beta"], "beta")
    covariance = _tensor(state["joint_covariance"], "joint covariance")
    residual = d - z @ gamma
    q = torch.cat((x, residual[:, None]), 1)
    eta = q @ beta
    mean, derivative = kernel_call(mean_link, state["kind"], eta)
    jacobian = torch.cat((-beta[-1] * z, q), 1)
    variance = ((jacobian @ covariance) * jacobian).sum(1)
    zero_gradient = (jacobian == 0).all(1)
    if (not bool(torch.isfinite(residual).all()) or not bool(torch.isfinite(variance).all())
            or not bool((variance >= 0).all()) or bool(((variance == 0) & ~zero_gradient).any())):
        fail("Conditional mean uncertainty is nonfinite or unresolved; covariance is not repaired.", "numerical_domain")
    eta_se = variance.sqrt()
    mean_se = derivative * eta_se
    mean_jacobian = derivative[:, None] * jacobian
    critical = kernel_call(critical_value, alpha, None)
    low = kernel_call(mean_link, state["kind"], eta - critical * eta_se)[0]
    high = kernel_call(mean_link, state["kind"], eta + critical * eta_se)[0]
    if (any(not bool(torch.isfinite(v).all()) for v in (mean_se, mean_jacobian, jacobian, low, high))
            or bool(((mean_se == 0) & ~zero_gradient).any())):
        fail("Conditional mean uncertainty or confidence bounds leave finite link support.", "numerical_domain")
    plan = plan_workspace("compact CF resident conditional means", {
        "query designs and joint Jacobians": 256 * count * (2 * width + 16),
        "parameter covariance": 256 * width * width})
    output = table({"position": positions, "mean": mean.tolist(), "std_error": mean_se.tolist(),
                    "ci_low": low.tolist(), "ci_high": high.tolist(), "eta": eta.tolist(),
                    "eta_std_error": eta_se.tolist(), "control_residual": residual.tolist()},
                   conditional_mean_only=True,
                   interval_definition="asymptotic normal eta interval transformed through mean link",
                   full_gamma_beta_covariance=True, source_state_sha256=state["integrity_sha256"],
                   kind=state["kind"], alpha=alpha, missing=missing, input_positions=positions,
                   nobs_original=len(source), nobs=len(positions), excluded_rows=len(source) - len(positions),
                   inference=bundle.inference, parameter_order=state["parameter_order"],
                   mean_jacobian=mean_jacobian.tolist(), eta_jacobian=jacobian.tolist(),
                   joint_parameter_covariance=covariance.tolist(),
                   covariance_representation="J C J' using recorded Jacobian and full joint parameter covariance",
                   source_semantic_replay=True, work_estimate=replay_work + query_work,
                   resource_plan=plan.record(), structural_effects_identified=False,
                   weak_instrument_robust=False, finite_cluster_exact=False, stata_parity_validated=False)
    output.index = source.index.take(positions).copy()
    return output
