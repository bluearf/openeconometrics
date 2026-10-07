"""Postestimation by exact bounded replays of a streamed OLS sample."""
from __future__ import annotations

import math
from collections.abc import Mapping
from types import SimpleNamespace

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.frame import as_frame
from openecon.models import ModelSpec
from .postestimation import OLSPostestimation, _chi2_sf, _linear_values, _prediction_vector, _tensor, f_sf
from openecon.engines.streaming_ols import _CompensatedSum

_AUXILIARY_BYTES = 8 * 1024 * 1024


def _state(result):
    state = result._state
    if not state.get("samples_only") or "replay" not in state:
        raise AnalysisError("estimation_state_unavailable", "This operation needs the original replayable model source.")
    return state


def _block_values(state, block):
    """Global normalized weights and cancellation-safe residuals/leverage."""
    if hasattr(block, "fit_values"):
        return block.fit_values
    callback = state.get("residual_block")
    if callable(callback):
        _, weights, residual, _, leverage = callback(block)
        return weights, residual, leverage
    x = block.x[:, state["kept_indices"]]
    weights = block.weights * state.get("weight_scale", 1.)
    residual = block.y - _prediction_vector(state, "prediction_fitted", x, lambda: x @ state["params"])
    leverage = weights * _prediction_vector(state, "prediction_leverage", x, lambda: (x @ state["bread"] * x).sum(dim=1))
    return weights, residual, leverage


def predict_batch(result, frame, *, kind="xb", sample_only=False, **options):
    """Use the fitted global covariance/df on a single bounded prediction batch.

    Only source iterations may request estimation-sample-only statistics. Local
    row positions scatter back into this batch, never into a whole-source array.
    """
    state = getattr(result, "_state", {})
    if not callable(state.get("encode")) or "params" not in state or "bread" not in state:
        raise AnalysisError("estimation_state_unavailable", "Batch prediction needs the fitted model state.")
    if not isinstance(kind, str):
        raise AnalysisError("invalid_prediction", "Prediction kind must be a name.")
    frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    aliases = {"resid": "residuals", "residual": "residuals", "score": "residuals",
               "hat": "leverage", "cooksd": "cook", "dffits": "dfits", "dfbetas": "dfbeta"}
    selected_kind = aliases.get(kind, kind)
    fitted_only = selected_kind in {"dfbeta", "covratio", "dfits", "welsch"}
    if fitted_only and not sample_only:
        raise AnalysisError("prediction_sample_required", f"{selected_kind} is available only on the estimation sample; omit data=.")
    columns = list(state["predictor_columns"])
    design = state.get("design")
    lagged = bool(getattr(design, "lag_specs", None))
    current = list(state.get("current_predictor_columns", columns))
    if lagged:
        columns = list(dict.fromkeys([*columns, design.time]))
        current = list(dict.fromkeys([*current, design.time]))
    missing = set(columns) - set(frame)
    if missing:
        raise AnalysisError("missing_columns", f"Missing predictor columns: {', '.join(sorted(missing))}.")
    keep = ~frame.loc[:, current].isna().any(axis=1)
    if fitted_only:
        if "replay" in state:
            fit_columns = getattr(state["replay"], "sample_columns", state["replay"].columns)
        else:
            clusters = [result.spec.cluster] if isinstance(result.spec.cluster, str) else list(result.spec.cluster or [])
            fit_columns = list(dict.fromkeys([result.spec.outcome, *current, *clusters,
                                              *([result.spec.weights] if result.spec.weights else []),
                                              *([result.spec.time] if result.spec.time else [])]))
        missing = set(fit_columns) - set(frame)
        if missing:
            raise AnalysisError("missing_columns", f"Estimation-sample prediction needs: {', '.join(sorted(missing))}.")
        keep &= ~frame.loc[:, fit_columns].isna().any(axis=1)
        if result.spec.weights:
            keep &= frame[result.spec.weights] > 0
    positions = torch.tensor(keep.to_numpy().nonzero()[0], dtype=torch.int64)
    used = frame.iloc[positions.tolist()]
    if lagged:
        full = design.encode(frame, allow_missing=True)[:, state["kept_indices"]]
        x = full[positions]
    else:
        x = state["encode"](used) if len(used) else torch.empty((0, len(state["terms"])), dtype=torch.float64)
    finite = torch.isfinite(x).all(dim=1)
    positions, x = positions[finite], x[finite]
    used = frame.iloc[positions.tolist()]
    y = (torch.tensor(pd.to_numeric(used[result.spec.outcome], errors="raise").to_numpy(dtype="float64", na_value=float("nan")), dtype=torch.float64)
         if result.spec.outcome in used else torch.zeros(len(used), dtype=torch.float64))
    raw_weights = (torch.tensor(used[result.spec.weights].to_numpy(dtype="float64", na_value=float("nan")), dtype=torch.float64)
                   if result.spec.weights and result.spec.weights in used else torch.ones(len(used), dtype=torch.float64))
    weights = raw_weights * state.get("weight_scale", 1.)
    fitted = _prediction_vector(state, "prediction_fitted", x, lambda: x @ state["params"])
    local = result.model_copy()
    local._state = {**state, "frame": used, "x": x, "y": y, "weights": weights,
                    "resid": y - fitted, "fitted": fitted, "positions": positions,
                    "original_rows": len(frame), "original_index": frame.index,
                    "samples_only": False}
    local._state.pop("original_frame", None)
    if fitted_only:
        output = OLSPostestimation.predict(local, kind=kind, **options)
    else:
        output = OLSPostestimation.predict(local, data=frame, kind=kind, **options)
    if "physical_positions" in frame.attrs:
        output.attrs["physical_positions"] = list(frame.attrs["physical_positions"])
    return output


def vif(result, *, uncentered=False):
    state = _state(result)
    terms, k = state["terms"], len(state["terms"])
    centered = result.spec.intercept and not uncentered
    total_w, mean, m2 = 0., torch.zeros(k, dtype=torch.float64), torch.zeros(k, dtype=torch.float64)
    for block in state["replay"].batches():
        x = block.x[:, state["kept_indices"]]
        w = block.weights
        size = float(w.sum())
        if not size:
            continue
        new_mean = (w[:, None] * x).sum(dim=0) / size
        new_m2 = (w[:, None] * (x - new_mean).square()).sum(dim=0)
        delta = new_mean - mean
        combined = total_w + size
        m2 += new_m2 + delta.square() * (total_w * size / combined)
        mean += delta * (size / combined)
        total_w = combined
    # Bread uses normalized aw/pw/robust-iw weights; VIF is invariant to
    # that scalar. Recover the same scalar from the bounded reporting sample.
    scale = state.get("weight_scale", 1.)
    if result.spec.weights and result.spec.weight_type != "fweight" and not (
            result.spec.weight_type == "iweight" and result.spec.covariance == "nonrobust"):
        scale = state["streaming"]["physical_nobs"] / total_w
    numerator = m2 if centered else m2 + total_w * mean.square()
    values = numerator * scale * state["bread"].diagonal()
    rows = [{"term": term, "vif": float(values[i]), "tolerance": float(values[i].reciprocal()),
             "centered": bool(centered)} for i, term in enumerate(terms) if term != "Intercept"]
    return as_frame(pd.DataFrame(rows, columns=["term", "vif", "tolerance", "centered"]))


def margins(result, variables=None, at=None, method="ame"):
    state = _state(result)
    if method not in {"ame", "mem"}:
        raise AnalysisError("invalid_margins", "method must be 'ame' or 'mem'.")
    if getattr(state.get("design"), "lag_specs", None):
        raise AnalysisError("unsupported_margins_transform", "Marginal effects for lag/difference formulas need an explicit intervention over time. Use lincom on the fitted lag/difference coefficients.")
    if at is not None and (not isinstance(at, Mapping)
                           or any(name not in state["predictor_columns"] for name in at)):
        raise AnalysisError("invalid_margins", "at must map fitted predictor columns to values.")
    setting = dict(at) if at is not None else {}
    requested = (state["predictor_columns"] if variables is None else [variables]
                 if isinstance(variables, str) else list(variables))
    if not requested or any(not isinstance(name, str) for name in requested) or len(set(requested)) != len(requested) or any(name not in state["predictor_columns"] for name in requested):
        raise AnalysisError("invalid_margins", "Select distinct original predictor columns for marginal effects.")
    combinations = 1
    for name, value in setting.items():
        points = list(value) if isinstance(value, (list, tuple)) else [value]
        if not points:
            raise AnalysisError("invalid_margins", "Marginal-effect grids cannot be empty.")
        combinations *= len(points)
        if combinations > 1000:
            raise AnalysisError("margins_grid_limit", "The marginal-effect grid supports at most 1000 combinations.")
        for point in points:
            if name in state["categorical"]:
                if point not in state["categorical"][name]:
                    raise AnalysisError("unknown_category", f"Column '{name}' has an unfitted at= category level.")
            elif _tensor(point).ndim != 0:
                raise AnalysisError("invalid_margins", "Numeric evaluation grids require finite scalar values.")
    variables = requested
    if method == "mem":
        names = [name for name in state["predictor_columns"] if name not in state["categorical"]]
        total, weighted = 0., torch.zeros(len(names), dtype=torch.float64)
        for block in state["replay"].batches():
            w = block.weights
            total += float(w.sum())
            for i, name in enumerate(names):
                weighted[i] += torch.dot(w, torch.tensor(block.frame[name].to_numpy(dtype="float64"), dtype=torch.float64))
        setting = {**dict(zip(names, (weighted / total).tolist(), strict=True)), **setting}
    accumulated, total_weight, template = None, 0., None
    for block in state["replay"].batches():
        # The same derivative compiler serves a bounded slice. Only contrasts
        # are averaged; averaging standard errors would be statistically wrong.
        local = result.model_copy()
        local._state = {**state, "frame": block.frame, "x": block.x[:, state["kept_indices"]],
                        "y": block.y, "weights": block.weights,
                        "resid": block.y - block.x[:, state["kept_indices"]] @ state["params"],
                        "fitted": block.x[:, state["kept_indices"]] @ state["params"],
                        "samples_only": False}
        table = OLSPostestimation.margins(local, variables=variables, at=setting,
                                         method="ame", _return_gradients=True)
        gradients = torch.tensor(table.attrs["delta_gradients"], dtype=torch.float64)
        if gradients.numel() == 0:
            continue
        weight = float(block.weights.sum())
        if accumulated is None:
            template = table
            accumulated = torch.zeros_like(gradients)
        if accumulated.shape != gradients.shape:
            raise AnalysisError("source_changed", "Marginal-effect rows changed between source batches.")
        accumulated += gradients * weight
        total_weight += weight
    if accumulated is None:
        return as_frame(pd.DataFrame())
    gradients = accumulated / total_weight
    rows = []
    for (_, row), gradient in zip(template.iterrows(), gradients, strict=True):
        estimate = float(_linear_values(state, state["params"], gradient[None, :])[0])
        contrast = result._estimate_contrast(estimate, gradient)
        identity = {key: value for key, value in row.items() if key in {"variable", "method"} or key.startswith("at[")}
        identity["method"] = method
        rows.append({**identity, **{key: value for key, value in contrast.items() if key != "gradient"}})
    return as_frame(pd.DataFrame(rows))


def _auxiliary(result, make_columns, names, *, response="squared", covariance="nonrobust",
               intercept=True, make_columns_factory=None):
    from .streaming import fit_streaming
    state = _state(result)
    extra = [*([result.spec.weights] if result.spec.weights else []),
             *([result.spec.time] if result.spec.time else [])]
    clusters = [result.spec.cluster] if isinstance(result.spec.cluster, str) else list(result.spec.cluster or [])
    if covariance != "nonrobust":
        extra += clusters
    extra = list(dict.fromkeys(extra))
    outcome = "__oe_aux_outcome__"
    if outcome in extra or set(names).intersection(extra):
        raise AnalysisError("reserved_columns", "Auxiliary model names conflict with weight/time/cluster columns.")
    # Bound expansion before the child Dataset/TSQR planner sees the batch.
    # Eightfold headroom covers the column stack, frame, hash and reader copies.
    rows = max(1, min(65_536, _AUXILIARY_BYTES // (8 * max(1, len(names) + len(extra) + 1) * 8)))

    def factory():
        build = make_columns_factory() if make_columns_factory is not None else make_columns
        for block in state["replay"].batches():
            for start in range(0, len(block.frame), rows):
                span = slice(start, start + rows)
                part = SimpleNamespace(frame=block.frame.iloc[span], x=block.x[span], y=block.y[span],
                                       weights=block.weights[span], positions=block.positions[span])
                x = part.x[:, state["kept_indices"]]
                _, residual, _ = _block_values(state, part)
                fitted = part.y - residual
                values = residual.square() if response == "squared" else residual if response == "residual" else part.y
                design = build(x, fitted, residual)
                record = {outcome: values.numpy(), **{name: design[:, i].numpy() for i, name in enumerate(names)}}
                record.update({name: part.frame[name].to_numpy() for name in extra})
                yield pd.DataFrame(record)

    options = {key: value for key, value in result.spec.options.items()
               if key in {"lags", "kernel", "reps", "seed", "dfadjust", "hansen"}} if covariance != "nonrobust" else {}
    source = Dataset.from_batches(factory, [outcome, *names, *extra])
    spec = ModelSpec(estimator="ols", outcome=outcome, predictors=names, weights=result.spec.weights,
                     weight_type=result.spec.weight_type, covariance=covariance,
                     intercept=intercept,
                     cluster=result.spec.cluster if covariance != "nonrobust" else None,
                     time=result.spec.time, options=options, missing="raise", alpha=result.spec.alpha)
    return fit_streaming(spec, source)


def hettest(result, variables=None, *, rhs=False, method="normal"):
    state = _state(result)
    if result.spec.weight_type not in {None, "fweight"}:
        raise AnalysisError("unsupported_diagnostic_weight", "Heteroskedasticity tests require unweighted or frequency-weighted OLS.")
    if method not in {"normal", "iid", "fstat"}:
        raise AnalysisError("invalid_diagnostic", "Use method normal, iid or fstat.")
    if variables is not None:
        variables = [variables] if isinstance(variables, str) else list(variables)
        if not variables or any(term not in state["terms"] for term in variables):
            raise AnalysisError("unknown_term", "Select fitted design terms for the variance test.")
        indices = [state["terms"].index(term) for term in variables]
    elif rhs:
        indices = [i for i, term in enumerate(state["terms"]) if term != "Intercept"]
    else:
        indices = None
    width = len(indices) if indices is not None else 1
    names = [f"__oe_variance_{i}__" for i in range(width)]
    aux = _auxiliary(result, lambda x, fitted, resid: x[:, indices] if indices is not None else fitted[:, None], names)
    degrees = len(aux["terms"]) - 1
    if degrees < 1:
        raise AnalysisError("invalid_diagnostic", "No nonconstant variance predictors are available.")
    n = result.nobs
    if method == "normal":
        mse = state["metrics"]["ss_resid"] / n
        statistic = max(0., aux["metrics"]["ss_model"]) / (2 * mse * mse)
    elif method == "iid":
        statistic = n * aux["metrics"]["r_squared"]
    else:
        statistic = (aux["metrics"]["ss_model"] / degrees) / (aux["metrics"]["ss_resid"] / aux["df_resid"])
    if method == "fstat":
        return {"statistic": statistic, "distribution": "F", "df_num": degrees, "df_denom": aux["df_resid"],
                "p_value": f_sf(statistic, degrees, aux["df_resid"]), "method": method, "null": "constant variance"}
    return {"statistic": statistic, "distribution": "chi2", "df": degrees,
            "p_value": _chi2_sf(statistic, degrees), "method": method, "null": "constant variance"}


def white_test(result):
    state = _state(result)
    if result.spec.weight_type not in {None, "fweight"}:
        raise AnalysisError("unsupported_diagnostic_weight", "White's test requires unweighted or frequency-weighted OLS.")
    from itertools import combinations_with_replacement
    indices = [i for i, term in enumerate(state["terms"]) if term != "Intercept"]
    if not indices:
        raise AnalysisError("invalid_diagnostic", "White's test has no nonconstant auxiliary terms.")
    pairs = list(combinations_with_replacement(indices, 2))
    if len(indices) + len(pairs) + 1 > 384:
        raise AnalysisError("diagnostic_memory_limit", "White's auxiliary design exceeds 384 parameters.")
    names = [f"__oe_white_{i}__" for i in range(len(indices) + len(pairs))]
    aux = _auxiliary(result, lambda x, fitted, resid: torch.column_stack(
        [*(x[:, i] for i in indices), *(x[:, i] * x[:, j] for i, j in pairs)]), names)
    degrees = len(aux["terms"]) - 1
    if degrees < 1:
        raise AnalysisError("invalid_diagnostic", "White's auxiliary terms contain no variance predictors.")
    statistic = result.nobs * aux["metrics"]["r_squared"]
    return {"statistic": statistic, "distribution": "chi2", "df": degrees,
            "p_value": _chi2_sf(statistic, degrees), "null": "constant variance", "auxiliary_rank": len(aux["terms"])}


def reset_test(result, powers=(2, 3, 4)):
    state = _state(result)
    try:
        powers = [powers] if isinstance(powers, int) else list(powers)
    except TypeError as exc:
        raise AnalysisError("invalid_diagnostic", "RESET powers must be distinct integers of at least two.") from exc
    if not powers or any(type(power) is not int or power < 2 for power in powers) or len(set(powers)) != len(powers):
        raise AnalysisError("invalid_diagnostic", "RESET powers must be distinct integers of at least two.")
    if len(powers) > 32:
        raise AnalysisError("diagnostic_memory_limit", "RESET supports at most 32 fitted-value powers.")
    indices = [i for i, term in enumerate(state["terms"]) if term != "Intercept"]
    names = [*(f"__oe_base_{i}__" for i in indices), *(f"__oe_reset_{power}__" for power in powers)]
    low, high = math.inf, -math.inf
    for block in state["replay"].batches():
        _, residual, _ = _block_values(state, block)
        fitted = block.y - residual
        low, high = min(low, float(fitted.min())), max(high, float(fitted.max()))
    spread = high - low
    if not math.isfinite(spread) or spread <= 0:
        raise AnalysisError("invalid_diagnostic", "RESET requires varying finite fitted values.")
    aux = _auxiliary(result, lambda x, fitted, resid: torch.column_stack(
        [*(x[:, i] for i in indices), *(((fitted - low) / spread).pow(power) for power in powers)]),
        names, response="outcome", covariance=result.spec.covariance, intercept=result.spec.intercept)
    parameters, covariance = aux["params"], aux["covariance"]
    selected = [i for i, name in enumerate(aux["terms"]) if name.startswith("__oe_reset_")]
    if not selected:
        raise AnalysisError("invalid_diagnostic", "The fitted-value powers add no identified terms.")
    from .postestimation import _wald
    restrictions = torch.eye(len(parameters), dtype=torch.float64)[selected]
    inference_df = aux["df_inference"]
    if inference_df is not None and not math.isfinite(inference_df):
        inference_df = None
    test = _wald(parameters, covariance, restrictions, torch.zeros(len(selected), dtype=torch.float64), inference_df,
                 state=aux.get("state", aux))
    adjustment = aux.get("contrast_inference")
    if len(selected) == 1 and callable(adjustment):
        from openecon.engines.inference import student_t_two_sided
        control = adjustment(restrictions[0])
        statistic = test["t_statistic" if inference_df is not None else "z_statistic"] * control["scale"]
        test.update(statistic=statistic ** 2, distribution="F", df_num=1,
                    df_denom=control["df"], t_statistic=statistic,
                    p_value=student_t_two_sided(statistic, control["df"]), statistic_scale=control["scale"])
    elif result._adjusted_inference():
        test["df_adjustment_note"] = "Joint RESET Wald uses conventional augmented-model degrees of freedom."
    test.update(powers=powers, added_rank=len(selected), covariance=result.spec.covariance,
                null="no omitted fitted-value powers")
    return test


def breusch_godfrey(result, lags=1):
    """Exact LM/F auxiliary regression with a bounded residual-lag window.

    Ordering follows dense postestimation: retained physical observation order,
    including initial zero lag residuals. It does not reinterpret time gaps.
    """
    state = _state(result)
    if result.spec.weight_type is not None:
        raise AnalysisError("unsupported_diagnostic_weight", "Breusch-Godfrey requires an unweighted ordered observation sample.")
    k = len(state["terms"])
    n = state["streaming"]["physical_nobs"]
    if type(lags) is not int or lags < 1 or lags >= n - k - 1:
        raise AnalysisError("invalid_diagnostic", "Choose a positive lag count with sufficient residual degrees of freedom.")
    indices = [i for i, term in enumerate(state["terms"]) if term != "Intercept"]
    if len(indices) + lags + 1 > 384:
        raise AnalysisError("diagnostic_memory_limit", "The Breusch-Godfrey auxiliary design exceeds 384 parameters.")
    base_names = [f"__oe_bg_base_{i}__" for i in indices]
    names = [*base_names, *(f"__oe_bg_lag_{lag}__" for lag in range(1, lags + 1))]

    def make_builder():
        history = torch.zeros(lags, dtype=torch.float64)

        def build(x, fitted, residual):
            nonlocal history
            joined = torch.cat((history, residual))
            lagged = [joined[lags - lag:len(joined) - lag] for lag in range(1, lags + 1)]
            history = joined[-lags:].clone()
            return torch.column_stack([*(x[:, i] for i in indices), *lagged])
        return build

    auxiliary = _auxiliary(result, None, names, response="residual", make_columns_factory=make_builder)
    restricted = _auxiliary(result, lambda x, fitted, resid: x[:, indices], base_names, response="residual")
    degrees = len(auxiliary["terms"]) - len(restricted["terms"])
    if degrees < 1:
        raise AnalysisError("invalid_diagnostic", "Lagged residuals add no independent auxiliary columns.")
    statistic = n * auxiliary["metrics"]["r_squared"]
    denominator = auxiliary["df_resid"]
    rss = auxiliary["metrics"]["ss_resid"]
    f_value = max(0., (restricted["metrics"]["ss_resid"] - rss) / degrees / (rss / denominator))
    return {"statistic": statistic, "distribution": "chi2", "df": degrees,
            "p_value": _chi2_sf(statistic, degrees), "f_statistic": f_value,
            "f_p_value": f_sf(f_value, degrees, denominator), "df_denom": denominator,
            "lags": lags, "ordering": "retained estimation row order", "initial_residual_lags": "zero",
            "null": "no residual serial correlation"}


def _residual_distribution(result):
    """Two bounded passes compute centered moments and ordered differences."""
    state = _state(result)
    supported_jb = result.spec.weight_type in {None, "fweight"}
    supported_dw = result.spec.weight_type is None
    if not (supported_jb or supported_dw):
        return ({"available": False, "code": "unsupported_diagnostic_weight",
                 "message": "Jarque-Bera requires unweighted or frequency-weighted residual moments."},
                {"available": False, "code": "unsupported_diagnostic_weight",
                 "message": "Durbin-Watson requires an unweighted ordered observation sample."})
    magnitude, mean, mass = 0., 0., _CompensatedSum(())
    for block in state["replay"].batches():
        weights, residual, _ = _block_values(state, block)
        new_magnitude = max(magnitude, float(residual.abs().max()))
        if new_magnitude:
            mean *= magnitude / new_magnitude
            magnitude = new_magnitude
        size = float(weights.sum())
        previous_mass = float(mass.value)
        mass.add(weights.sum())
        local_mean = float(torch.dot(weights, residual / magnitude)) / size if magnitude else 0.
        mean += (local_mean - mean) * size / (previous_mass + size)
    moments = _CompensatedSum((3,))
    differences, squared = _CompensatedSum(()), _CompensatedSum(())
    previous = None
    if magnitude:
        for block in state["replay"].batches():
            weights, residual, _ = _block_values(state, block)
            scaled = residual / magnitude
            centered = scaled - mean
            moments.add(torch.stack([torch.dot(weights, centered.pow(order)) for order in (2, 3, 4)]))
            if supported_dw:
                squared.add(scaled.square().sum())
                differences.add(scaled.diff().square().sum())
                if previous is not None:
                    differences.add((scaled[0] - previous).square())
                previous = scaled[-1].clone()
    variance, third, fourth = (moments.value / mass.value).tolist()
    if variance > 0:
        skewness = third / variance ** 1.5
        kurtosis = fourth / variance ** 2
        statistic = result.nobs / 6 * (skewness ** 2 + (kurtosis - 3) ** 2 / 4)
        jb = {"statistic": statistic, "distribution": "chi2", "df": 2,
              "p_value": _chi2_sf(statistic, 2), "skewness": skewness, "kurtosis": kurtosis}
    else:
        jb = {"available": False, "code": "constant_residuals"}
    dw = ({"statistic": float(differences.value / squared.value), "ordering": "retained estimation row order"}
          if supported_dw and float(squared.value) > 0 else
          {"available": False, "code": "constant_residuals"} if supported_dw else
          {"available": False, "code": "unsupported_diagnostic_weight",
           "message": "Durbin-Watson requires an unweighted ordered observation sample."})
    return jb, dw


_INFLUENCE_COLUMNS = ("leverage", "rstandard", "rstudent", "cook", "covratio", "dfits", "welsch")


def iter_influence(result, *, batch_rows=65_536):
    """Yield all retained influence rows; output batches keep original labels.

    Physical source positions are bounded per-batch metadata. Global variance,
    degrees of freedom and bread come from the entire fitted sample.
    """
    state = getattr(result, "_state", {})
    state = _state(result) if state.get("samples_only") else result._dense_state()
    if type(batch_rows) is not int or not 1 <= batch_rows <= 65_536:
        raise AnalysisError("invalid_batch_size", "Use batch_rows between 1 and 65,536.")
    for kind in _INFLUENCE_COLUMNS:
        result._classical_prediction(kind)
    variance, degrees = result._sigma2(state), state["df_resid"]
    if degrees <= 1:
        raise AnalysisError("insufficient_observations", "Deleted-observation diagnostics require residual degrees of freedom greater than one.")
    def dense_blocks():
        for start in range(0, len(state["frame"]), batch_rows):
            rows = slice(start, start + batch_rows)
            yield SimpleNamespace(frame=state["frame"].iloc[rows], positions=state["positions"][rows],
                                  fit_values=(state["weights"][rows], state["resid"][rows], state["leverage"][rows]))

    blocks = state["replay"].batches() if state.get("samples_only") else dense_blocks()
    for block in blocks:
        weights, residual, leverage = _block_values(state, block)
        if result.spec.weight_type == "fweight":
            leverage = leverage / weights
        complement = 1 - leverage
        complement = torch.where(complement > 100 * torch.finfo(torch.float64).eps,
                                 complement, torch.full_like(complement, float("nan")))
        standardized = residual / (variance * complement).sqrt()
        deleted = (variance * degrees - residual.square() / complement) / (degrees - 1)
        deleted = torch.where(deleted > 0, deleted, torch.full_like(deleted, float("nan")))
        student = residual / (deleted * complement).sqrt()
        dfits = student * (leverage.clamp_min(0) / complement).sqrt()
        values = {"leverage": leverage, "rstandard": standardized, "rstudent": student,
                  "cook": standardized.square() * leverage / (len(state["terms"]) * complement),
                  "covratio": (deleted / variance).pow(len(state["terms"])) / complement,
                  "dfits": dfits, "welsch": dfits * ((state["nobs"] - 1) / complement).sqrt()}
        for start in range(0, len(block.frame), batch_rows):
            stop = start + batch_rows
            table = as_frame(pd.DataFrame({name: value[start:stop].tolist() for name, value in values.items()},
                                          index=block.frame.index[start:stop]))
            table.attrs.update(physical_positions=block.positions[start:stop].tolist(),
                               ordering="retained estimation row order", sample_only=True)
            yield table


def _influence_summary(result):
    totals = {name: _CompensatedSum(()) for name in _INFLUENCE_COLUMNS}
    magnitudes = dict.fromkeys(_INFLUENCE_COLUMNS, 0.)
    counts = dict.fromkeys(_INFLUENCE_COLUMNS, 0)
    minima = dict.fromkeys(_INFLUENCE_COLUMNS, math.inf)
    maxima = dict.fromkeys(_INFLUENCE_COLUMNS, -math.inf)
    for table in iter_influence(result):
        for name in _INFLUENCE_COLUMNS:
            values = torch.tensor(table[name].to_numpy(), dtype=torch.float64)
            values = values[torch.isfinite(values)]
            counts[name] += len(values)
            if len(values):
                magnitude = max(magnitudes[name], float(values.abs().max()))
                if magnitude:
                    totals[name].scale(torch.tensor(magnitudes[name] / magnitude, dtype=torch.float64))
                    totals[name].add((values / magnitude).sum())
                magnitudes[name] = magnitude
                minima[name] = min(minima[name], float(values.min()))
                maxima[name] = max(maxima[name], float(values.max()))
    return {"available": True, "mode": "summary", "rows": result._state["streaming"]["physical_nobs"],
            "columns": {name: {"count": counts[name], "minimum": minima[name] if counts[name] else None,
                               "maximum": maxima[name] if counts[name] else None,
                               "mean": float(totals[name].value) / counts[name] * magnitudes[name] if counts[name] else None}
                        for name in _INFLUENCE_COLUMNS},
            "ordering": "retained estimation row order", "full_rows": "iter_influence()"}


def diagnostics(result, *, lags=1):
    """Full-source tests and bounded influence summaries without row storage."""
    _state(result)
    output = {}
    for name, calculate in (("vif", lambda: vif(result)), ("breusch_pagan", lambda: hettest(result)),
                            ("white", lambda: white_test(result)), ("reset", lambda: reset_test(result)),
                            ("breusch_godfrey", lambda: breusch_godfrey(result, lags))):
        try:
            output[name] = calculate()
        except AnalysisError as exc:
            output[name] = {"available": False, "code": exc.code, "message": str(exc)}
    output["jarque_bera"], output["durbin_watson"] = _residual_distribution(result)
    try:
        output["influence"] = _influence_summary(result)
    except AnalysisError as exc:
        output["influence"] = {"available": False, "code": exc.code, "message": str(exc)}
    return output
