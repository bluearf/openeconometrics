"""OpenEconometrics's comprehensive native PyTorch OLS analysis family."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import math
import platform
from uuid import uuid4

from pydantic import PrivateAttr

from openecon.analysis_contracts import AnalysisError
from openecon.models import Coefficient, ModelSpec, ResultBundle


@lru_cache(maxsize=1)
def _result_type():
    from openecon.linear_ols.postestimation import OLSPostestimation

    class OLSResult(OLSPostestimation, ResultBundle):
        """A persistable result with private fit state for full-row operations."""
        _state: dict = PrivateAttr(default_factory=dict)

        def __eq__(self, other):
            return isinstance(other, ResultBundle) and self.model_dump() == other.model_dump()

        def margins(self, *args, **kwargs):
            if kwargs.get("data") is not None:
                from openecon.econometrics.postest.prediction import margins
                return margins(self, *args, **kwargs)
            if self._state.get("samples_only"):
                from .streaming_postestimation import margins
                return margins(self, *args, **kwargs)
            return super().margins(*args, **kwargs)

        def vif(self, *args, **kwargs):
            if self._state.get("samples_only"):
                from .streaming_postestimation import vif
                return vif(self, *args, **kwargs)
            return super().vif(*args, **kwargs)

        def hettest(self, *args, **kwargs):
            if self._state.get("samples_only"):
                from .streaming_postestimation import hettest
                return hettest(self, *args, **kwargs)
            return super().hettest(*args, **kwargs)

        def white_test(self, *args, **kwargs):
            if self._state.get("samples_only"):
                from .streaming_postestimation import white_test
                return white_test(self, *args, **kwargs)
            return super().white_test(*args, **kwargs)

        def reset_test(self, *args, **kwargs):
            if self._state.get("samples_only"):
                from .streaming_postestimation import reset_test
                return reset_test(self, *args, **kwargs)
            return super().reset_test(*args, **kwargs)

        def diagnostics(self, *args, **kwargs):
            if self._state.get("samples_only"):
                from .streaming_postestimation import diagnostics
                return diagnostics(self, *args, **kwargs)
            return super().diagnostics(*args, **kwargs)

        def breusch_godfrey(self, *args, **kwargs):
            if self._state.get("samples_only"):
                from .streaming_postestimation import breusch_godfrey
                return breusch_godfrey(self, *args, **kwargs)
            return super().breusch_godfrey(*args, **kwargs)

        def iter_influence(self, *, batch_rows=65_536):
            from .streaming_postestimation import iter_influence
            return iter_influence(self, batch_rows=batch_rows)

        def predict(self, data=None, kind="xb", alpha=None, *, interval=None, term=None, batch_rows=None):
            from openecon.dataset import Dataset
            if isinstance(data, Dataset):
                from openecon.econometrics.postest.streaming_prediction import materialize_predictions
                return materialize_predictions(
                    self.iter_predict(data, batch_rows=65536 if batch_rows is None else batch_rows,
                                      kind=kind, alpha=alpha, interval=interval, term=term),
                    {"estimator": "ols", "kind": kind, "precision": "float64",
                     "response_definition": "native fitted OLS prediction",
                     "full_source_collected": False})
            if batch_rows is not None:
                raise AnalysisError("invalid_batch_size", "batch_rows applies only to Dataset evaluation data.")
            if self._state.get("samples_only") and data is not None:
                from .streaming_postestimation import predict_batch
                return predict_batch(self, data, kind=kind, alpha=alpha, interval=interval, term=term)
            return super().predict(data=data, kind=kind, alpha=alpha, interval=interval, term=term)

        def iter_predict(self, data=None, *, batch_rows: int = 65_536, kind="xb", **options):
            """Yield prediction frames without an output allocation proportional to N."""
            from openecon.dataset import Dataset
            from .streaming_postestimation import predict_batch
            if isinstance(batch_rows, bool) or not isinstance(batch_rows, int) or not 1 <= batch_rows <= 65_536:
                raise AnalysisError("invalid_batch_size", "Use batch_rows between 1 and 65,536.")
            if not isinstance(kind, str):
                raise AnalysisError("invalid_prediction", "Prediction kind must be a name.")
            source = data if data is not None else self._state.get("source")
            if data is None and self._state.get("samples_only") and "replay" in self._state:
                replay = self._state["replay"]
                offset = 0
                with replay.frames() as (frames, record_positions):
                    for raw, retained, positions in frames:
                        encoded = self._state["design"].encode(retained)
                        import torch
                        record_positions(positions[torch.isfinite(encoded).all(dim=1)])
                        for start in range(0, len(raw), batch_rows):
                            frame = raw.iloc[start:start + batch_rows].copy(deep=False)
                            frame.attrs["physical_positions"] = list(range(offset + start, offset + start + len(frame)))
                            yield predict_batch(self, frame, kind=kind, sample_only=True, **options)
                        offset += len(raw)
                return
            if source is None:
                source = self._state.get("original_frame")
                if source is None:
                    raise AnalysisError("estimation_state_unavailable", "Prediction needs the fitted model state or new data.")
            columns = list(self._state.get("predictor_columns", self.spec.predictors))
            if self.spec.time:
                columns.append(self.spec.time)
            influence = {"rstandard", "rstudent", "cook", "cooksd", "dfits", "dffits",
                         "covratio", "welsch", "dfbeta", "dfbetas"}
            if kind in {"residuals", "resid", "residual", "score", *influence}:
                columns = list(dict.fromkeys([*columns, self.spec.outcome]))
            source_columns = source.columns if isinstance(source, Dataset) else getattr(source, "columns", [])
            if self.spec.weights and (data is None or self.spec.weights in source_columns
                                      or kind in {"leverage", "hat", *influence}):
                columns.append(self.spec.weights)
            if data is None and kind in influence:
                clusters = [self.spec.cluster] if isinstance(self.spec.cluster, str) else list(self.spec.cluster or [])
                columns.extend(clusters)
            columns = list(dict.fromkeys(columns))
            design = self._state.get("design")
            if not isinstance(source, Dataset):
                import pandas as pd
                frame = pd.DataFrame(source, copy=False)
                if getattr(design, "lag_specs", None):
                    from .lags import lag_values
                    frame = frame.copy(deep=False)
                    for (name, lag), column in design.lag_specs.items():
                        frame[column] = lag_values(frame, design.time, name, lag, allow_missing=True).numpy()
                    columns.extend(design.lag_columns)
                source = Dataset.from_frame(frame)
            elif getattr(design, "lag_specs", None):
                source = design.with_streaming_history(source, batch_rows=batch_rows, columns=columns)
                columns.extend(design.lag_columns)
            source.assert_unchanged()
            iterator = source.iter_batches(columns, batch_rows=batch_rows)
            offset = 0
            try:
                for frame in iterator:
                    frame.attrs["physical_positions"] = list(range(offset, offset + len(frame)))
                    offset += len(frame)
                    yield predict_batch(self, frame, kind=kind, sample_only=data is None, **options)
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
            source.assert_unchanged()

        def summary(self, format="text"):
            text = super().summary(format=format)
            if format == "text" and "rmse:" not in text and self.metrics.get("rmse") is not None:
                text += f"\nRoot MSE: {self.metrics['rmse']:.6g}  |  Residual df: {self.metrics['df_resid']:g}"
            if format == "text" and self.tests.get("model"):
                test = self.tests["model"]
                if test.get("statistic") is not None and "Model F test:" not in text:
                    if test.get("distribution") == "chi2":
                        text += f"\nModel Wald test: chi2({test['df']:g}) = {test['statistic']:.6g}, p = {test['p_value']:.6g}"
                    else:
                        text += f"\nModel F test: F({test['df']:g}, {test['df2']:g}) = {test['statistic']:.6g}, p = {test['p_value']:.6g}"
            return text

    return OLSResult


def ols(*, data, y=None, x=None, formula=None, covariance=None, categorical=None,
        intercept=True, cluster=None, weights=None, weight_type=None, missing="drop",
        alpha=0.05, time=None, lags=None, kernel=None, reps=None, seed=None,
        dfadjust=False, hansen=False, device="auto"):
    """Fit OLS/WLS; use either y/x or a safe Python formula, and explicit VCE options."""
    from openecon.linear_ols.design import OLSDesign, parse_formula
    if isinstance(x, (str, bytes)) or isinstance(categorical, (str, bytes)):
        raise AnalysisError("invalid_spec", "x must be a list of predictor names; categorical must be a sequence of column names.")
    if formula is not None:
        if y is not None or x is not None:
            raise AnalysisError("invalid_spec", "Use formula or y/x, not both.")
        y, terms, intercept = parse_formula(formula)
    else:
        if not isinstance(y, str) or not y.strip() or (x is not None and not isinstance(x, Sequence)):
            raise AnalysisError("invalid_spec", "OLS requires an outcome name and a predictor sequence, or formula=.")
        terms = list(x or [])
    if not all(isinstance(term, str) and term.strip() for term in terms):
        raise AnalysisError("invalid_spec", "Predictors must be nonempty formula expressions or column names.")
    design = OLSDesign(terms, list(categorical or []), intercept=intercept, time=time,
                       literal_names=terms if formula is None else ())
    options = {"terms": design.expressions}
    options.update({key: value for key, value in {"lags": lags, "kernel": kernel,
                   "reps": reps, "seed": seed}.items() if value is not None})
    if dfadjust:
        options["dfadjust"] = dfadjust
    if hansen:
        options["hansen"] = hansen
    try:
        spec = ModelSpec(estimator="ols", outcome=y, predictors=design.required,
                         categorical=design.categorical, intercept=intercept,
                         covariance=covariance, cluster=cluster, weights=weights,
                         weight_type=weight_type, time=time, options=options,
                         missing=missing, alpha=alpha)
    except ValueError as exc:
        raise AnalysisError("invalid_spec", str(exc)) from exc
    return fit_ols(spec, data=data, device=device, design=design)


def _device(name):
    import torch
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        selected = torch.device(name)
    except (TypeError, RuntimeError) as exc:
        raise AnalysisError("invalid_device", "Use auto, cpu or an available CUDA device.") from exc
    if selected.type == "cpu":
        return "cpu"
    if selected.type == "mps":
        from openecon.engines.execution import validate_device
        return validate_device(name)
    if selected.type == "cuda" and torch.cuda.is_available():
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        if index < torch.cuda.device_count():
            return str(selected)
    raise AnalysisError("device_unavailable", "Use cpu, an available CUDA GPU, or checked Metal acceleration.")


def fit_ols(spec, *, data, device="auto", design=None):
    import pandas as pd
    import torch
    from openecon.dataset import Dataset
    from openecon.linear_ols.design import OLSDesign
    from openecon.linear_ols.estimation import estimate
    from openecon.streaming_design import numeric_values

    from openecon.engines.execution import validate_device
    selected = validate_device(device) if isinstance(data, Dataset) else _device(device)
    design = design or OLSDesign(spec.options.get("terms", spec.predictors), spec.categorical,
                                 intercept=spec.intercept, time=spec.time, literal_names=spec.predictors)
    if (set(design.required) != set(spec.predictors)
            or set(design.categorical) != set(spec.categorical)
            or spec.outcome in design.required):
        raise AnalysisError("invalid_spec", "Formula terms must match the declared predictor and categorical columns.")
    cluster_names = [spec.cluster] if isinstance(spec.cluster, str) else list(spec.cluster or [])
    required = list(dict.fromkeys([spec.outcome, *design.required, *cluster_names,
                     *([spec.weights] if spec.weights else []), *([spec.time] if spec.time else [])]))
    if isinstance(data, Dataset):
        from openecon.linear_ols.streaming import fit_streaming
        from openecon.engines.contracts import KernelError
        try:
            estimated = fit_streaming(spec, data, design=design, device=selected)
        except KernelError as exc:
            raise AnalysisError(exc.code, str(exc)) from exc
        state = {key: value for key, value in estimated.items() if key != "state"}
        state.update(estimated.get("state", {}))
        state.update({"source": data, "design": design, "predictor_columns": design.required,
                      "current_predictor_columns": design.current_required,
                      "categorical": design.categories, "samples_only": True,
                      "encode": lambda frame: design.encode(frame)[:, estimated["kept_indices"]]})
        return _bundle(spec, estimated, state, selected)
    try:
        frame = pd.DataFrame(data, copy=False)
    except (TypeError, ValueError) as exc:
        raise AnalysisError("invalid_data", "Data must be a dataframe, a column mapping or row records.") from exc
    absent = set(required) - set(frame.columns)
    if absent:
        raise AnalysisError("missing_columns", f"Missing model columns: {', '.join(sorted(absent))}.")
    if len(frame.columns) != len(set(frame.columns)):
        raise AnalysisError("duplicate_columns", "Model data columns must be unique.")
    if not len(frame):
        raise AnalysisError("empty_sample", "The dataset is empty.")
    if selected == "mps":
        # Metal cannot store doubles. The bounded QR path refines its float32
        # preconditioners from original float64 observations before inference.
        return fit_ols(spec, data=Dataset.from_frame(frame), device=selected, design=design)
    # Preserve bounded processing for an existing large frame, too. Discover
    # formula category widths in batches before any full dummy expansion.
    from openecon import analysis
    budget = min(analysis._MAX_DESIGN_BYTES, 64 * 1024 * 1024)
    if len(frame) * max(1, len(design.required) + int(spec.intercept)) * 8 > budget:
        return fit_ols(spec, data=Dataset.from_frame(frame), device=device, design=design)
    original = frame.loc[:, required].copy(deep=True)
    design.prepare(original)
    if len(frame) * max(1, len(design.terms)) * 8 > budget:
        return fit_ols(spec, data=Dataset.from_frame(frame), device=device, design=design)
    sample_columns = list(dict.fromkeys([spec.outcome, *design.current_required, *cluster_names,
                          *([spec.weights] if spec.weights else []), *([spec.time] if spec.time else [])]))
    keep = ~original.loc[:, sample_columns].isna().any(axis=1)
    if spec.missing == "raise" and not keep.all():
        raise AnalysisError("missing_values", "The model inputs contain missing values; choose missing='drop'.")
    positions = torch.tensor(keep.to_numpy().nonzero()[0], dtype=torch.int64)
    used = original.iloc[positions.tolist()]
    raw_weights = numeric_values(used[spec.weights], spec.weights) if spec.weights else None
    if raw_weights is not None:
        if bool((raw_weights < 0).any()):
            raise AnalysisError("invalid_weights", "Weights must be nonnegative.")
        if spec.weight_type == "fweight" and bool((raw_weights != raw_weights.round()).any()):
            raise AnalysisError("invalid_weights", "Frequency weights must be integers.")
        positive = raw_weights > 0
        positions, raw_weights = positions[positive], raw_weights[positive]
        used = original.iloc[positions.tolist()]
    if not len(used):
        raise AnalysisError("empty_sample", "No usable observations with positive weights remain.")
    x = (design.encode(original, allow_missing=True)[positions]
         if getattr(design, "lag_specs", None) else design.encode(used))
    finite = torch.isfinite(x).all(dim=1)
    if not bool(finite.all()):
        if spec.missing == "raise":
            raise AnalysisError("missing_values", "A formula transform or lag produces missing/non-finite values.")
        positions, x = positions[finite], x[finite]
        if raw_weights is not None:
            raw_weights = raw_weights[finite]
        used = original.iloc[positions.tolist()]
    y = numeric_values(used[spec.outcome], spec.outcome)
    if len(y) and bool((y == y[0]).all()):
        raise AnalysisError("constant_outcome", "The outcome has no variation in the estimation sample.")
    time_values = numeric_values(used[spec.time], spec.time) if spec.time else None
    groups = [used[name].tolist() for name in cluster_names] or None
    options = dict(covariance=spec.covariance, weights=raw_weights.to(selected) if raw_weights is not None else None,
                   weight_type=spec.weight_type, clusters=groups,
                   time=time_values.to(selected) if time_values is not None else None,
                   lags=spec.options.get("lags"), kernel=spec.options.get("kernel", "bartlett"),
                   reps=spec.options.get("reps", 199), seed=spec.options.get("seed", 0))
    if spec.options.get("dfadjust"):
        options["dfadjust"] = True
    if spec.options.get("hansen"):
        options["hansen"] = True
    from openecon.engines.contracts import KernelError
    try:
        estimated = estimate(x.to(selected), y.to(selected), terms=design.terms,
                             intercept=spec.intercept, **options)
    except KernelError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
    kept = estimated["kept_indices"]
    state = {**{key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
               for key, value in estimated.items()}, "original_frame": original, "original_rows": len(original),
             "original_index": original.index,
             "frame": used, "positions": positions, "index_labels": used.index.tolist(),
             "design": design, "predictor_columns": design.required, "categorical": design.categories,
             "current_predictor_columns": design.current_required,
             "encode": lambda new_frame: design.encode(new_frame)[:, kept],
             "refit": lambda augmented_x: estimate(augmented_x.to(selected), y.to(selected),
                 terms=[*estimated["terms"], *[f"added_{i}" for i in range(augmented_x.shape[1] - len(kept))]],
                 intercept=spec.intercept, **options),
             "weight_scale": float(estimated["weights"][0] / raw_weights[0].to(selected)) if raw_weights is not None else 1.,
             "nparams": len(kept), "sigma2": estimated["metrics"]["ss_resid"] / estimated["df_resid"]}
    from openecon.analysis import _frame_hasher, _position_bytes
    hasher = _frame_hasher(original)
    sample = original.iloc[positions.tolist()].reset_index(drop=True) if len(original) != len(used) else original
    sample_hasher = hasher.copy() if sample is original else _frame_hasher(sample)
    sample_hasher.update(_position_bytes(positions.tolist()))
    estimated.update(nobs_original=len(original), sample_positions=positions.tolist(),
                     dropped_rows=len(original) - len(used),
                     data_hash=hasher.hexdigest(), sample_hash=sample_hasher.hexdigest())
    return _bundle(spec, estimated, state, selected)


def _bundle(spec, estimated, state, device):
    import torch
    from openecon import __version__
    from openecon.engines.inference import critical_value, two_sided_p_values, student_t_two_sided
    from openecon.linear_ols.postestimation import f_sf

    params, covariance = estimated["params"], estimated["covariance"]
    terms = estimated.get("terms", state.get("terms", []))
    se = torch.diagonal(covariance).clamp_min(0).sqrt()
    if not bool((se > 0).all()):
        raise AnalysisError("degenerate_inference", "One or more coefficient standard errors are zero; ordinary inference is undefined.")
    stats = params / se
    df = estimated["df_inference"]
    use_t = df is not None and math.isfinite(df)
    df = df if use_t else None
    pvalues = two_sided_p_values(stats, df)
    critical = critical_value(spec.alpha, df)
    coef_df = estimated.get("inference", {}).get("coefficient_df")
    coef_scale = estimated.get("inference", {}).get("coefficient_scale")
    lows, highs = [], []
    for i in range(len(terms)):
        individual_df = coef_df[i] if coef_df is not None else df
        individual_scale = coef_scale[i] if coef_scale is not None else 1.0
        if individual_df is not None:
            pvalues[i] = student_t_two_sided(float(stats[i]) * individual_scale, float(individual_df))
        margin = critical_value(spec.alpha, individual_df) * se[i] / individual_scale if coef_df is not None else critical * se[i]
        lows.append(float(params[i] - margin))
        highs.append(float(params[i] + margin))
    coefficients = [Coefficient(term=term, estimate=float(params[i]), std_error=float(se[i]),
                    statistic=float(stats[i]) * (coef_scale[i] if coef_scale is not None else 1.0), p_value=float(pvalues[i]),
                    ci_low=lows[i], ci_high=highs[i])
                    for i, term in enumerate(terms)]
    raw_metrics = estimated["metrics"]
    metrics = {"r_squared": raw_metrics.get("r_squared"), "pseudo_r_squared": None,
               "adjusted_r_squared": raw_metrics.get("adjusted_r_squared", raw_metrics.get("r_squared_adj")),
               "rmse": raw_metrics.get("rmse"), "df_model": len(terms) - int(spec.intercept and "Intercept" in terms),
               "df_resid": estimated["df_resid"], "ss_resid": raw_metrics.get("ss_resid"),
               "ss_model": raw_metrics.get("ss_model"), "ss_total": raw_metrics.get("ss_total"),
               "log_likelihood": raw_metrics.get("log_likelihood")}
    metrics["condition_number_scaled"] = raw_metrics.get("condition_number")
    if metrics["log_likelihood"] is not None:
        metrics.update(aic=-2 * metrics["log_likelihood"] + 2 * len(terms),
                       bic=-2 * metrics["log_likelihood"] + len(terms) * math.log(estimated["nobs"]))
    slopes = [i for i, term in enumerate(terms) if term != "Intercept"]
    model_test = None
    if slopes:
        v = covariance[slopes][:, slopes]
        scale = v.diagonal().clamp_min(torch.finfo(v.dtype).tiny).sqrt()
        correlation = v / scale[:, None] / scale[None, :]
        restriction_rank = int(torch.linalg.matrix_rank(correlation))
        if restriction_rank:
            standardized = params[slopes] / scale
            value = float(standardized @ torch.linalg.pinv(correlation, hermitian=True) @ standardized) / restriction_rank
            model_df = df
            adjustment = estimated.get("contrast_inference")
            if len(slopes) == 1 and callable(adjustment):
                gradient = torch.zeros_like(params)
                gradient[slopes[0]] = 1
                control = adjustment(gradient)
                value *= float(control["scale"]) ** 2
                model_df = float(control["df"])
            if model_df is None:
                from openecon.linear_ols.postestimation import _chi2_sf
                statistic = max(value, 0) * restriction_rank
                model_test = {"statistic": statistic, "df": restriction_rank, "df2": None,
                              "p_value": _chi2_sf(statistic, restriction_rank),
                              "distribution": "chi2", "label": "Model Wald test"}
            else:
                model_test = {"statistic": max(value, 0), "df": restriction_rank, "df2": model_df,
                              "p_value": f_sf(max(value, 0), restriction_rank, model_df),
                              "distribution": "F", "label": "Model F test"}
    details = _json_metadata(estimated.get("inference", {}))
    # Keep inferential choices stable across mathematically equivalent dense
    # and streamed solvers; coefficients/tests carry numerical statistics.
    numerical = {"standard_errors", "f_statistic", "f_df_num", "f_df_denom",
                 "resampling_mean", "bias_estimate", "df", "model_test", "correction"}
    inference = {key: value for key, value in details.items() if key not in numerical}
    inference.update(use_t=use_t, distribution="t" if use_t else "normal", df_inference=df,
                     confidence_level=1 - spec.alpha, correction=spec.covariance,
                     covariance=spec.covariance, weight_type=spec.weight_type,
                     dfadjust=bool(details.get("dfadjust", False)),
                     hansen=bool(details.get("hansen", False)),
                     cluster_count=details.get("cluster_count"))
    if spec.covariance in {"HC0", "HC1", "HC2", "HC3", "hac"}:
        inference["small_sample_correction"] = (estimated["nobs"] / estimated["df_resid"]
                                               if spec.covariance in {"HC1", "hac"} else 1.)
    elif spec.covariance == "cluster" and details.get("cluster_dimensions") == 1:
        count = details["cluster_count"]
        inference["small_sample_correction"] = count / (count - 1) * (estimated["nobs"] - 1) / estimated["df_resid"]
    if spec.covariance.startswith("cluster"):
        inference.setdefault("combination_counts", [])
        inference.setdefault("df_adjustment", "G-1")
    positions = estimated.get("sample_positions", list(range(min(400, len(state.get("y", []))))))
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().tolist()
    predictions = []
    for i, (observed, fitted) in enumerate(zip(estimated.get("y", [])[:400], estimated.get("fitted", [])[:400])):
        predictions.append({"row": positions[i], "observed": float(observed), "fitted": float(fitted),
                            "residual": float(observed - fitted)})
    import pandas as pd
    stream = estimated.get("streaming")
    source_data_hash = estimated.get("data_hash") or (stream or {}).get("data_hash")
    sample_hash = estimated.get("sample_hash") or (hashlib.sha256(
        (source_data_hash + stream["positions_hash"]).encode()).hexdigest() if stream else hashlib.sha256(str(positions).encode()).hexdigest())
    result = _result_type()(id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(),
        spec=spec, nobs=int(estimated["nobs"]), nobs_original=estimated.get("nobs_original", int(estimated["nobs"])),
        dropped_rows=estimated.get("dropped_rows", 0), coefficients=coefficients,
        covariance_matrix=covariance.detach().cpu().tolist(), metrics=metrics,
        warnings=estimated.get("warnings", []), predictions=predictions,
        sample_positions=[] if stream else positions, provenance={"engine": "openecon", "version": __version__,
            "torch_version": torch.__version__, "precision": "float64", "device": device,
            "solver": "torch_tsqr" if stream else "torch_qr", "categories": state.get("categorical", {}),
            "inference_details": details,
            "omitted_terms": estimated.get("omitted_terms", []),
            "schema_version": "5", "backend": "openecon.torch", "estimator": "ols",
            "versions": {"openecon": __version__, "python": platform.python_version(),
                         "pandas": pd.__version__, "torch": torch.__version__},
            "data_hash": source_data_hash, "sample_hash": sample_hash,
            "sample_positions_hash": (stream or {}).get("positions_hash"),
            "input_columns": state["replay"].columns if stream else list(state["original_frame"]),
            "hash_algorithm": "sha256 over schema and pandas row hashes (pandas version recorded)" if not stream else
                "SHA-256 over projected pandas row hashes; retained physical positions hashed separately; pandas version recorded",
            "design_terms": terms, "sample_position_base": 0,
            "prediction_sample": "first retained positional observations", "prediction_limit": 400,
            "sample_positions_omitted": bool(stream), "sample_position_count": (stream or {}).get("used", len(state.get("positions", []))),
            "categorical_encoding": {name: {"reference": levels[0] if spec.intercept else None,
                                      "levels": levels} for name, levels in state.get("categorical", {}).items()},
            "stata_parity_validated": False,
            **estimated.get("provenance", {}),
            **({"streaming": _json_metadata(estimated["streaming"])} if "streaming" in estimated else {})},
        inference=inference, title="OLS", tests={"model": model_test} if model_test else {})
    result._state = state
    return result


def _json_metadata(value):
    import torch
    if isinstance(value, torch.Tensor):
        return _json_metadata(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: _json_metadata(item) for key, item in value.items() if not callable(item)}
    if isinstance(value, (tuple, list)):
        return [_json_metadata(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def test(result, *args, **kwargs):
    return result.test(*args, **kwargs)


def testparm(result, *args, **kwargs):
    return result.testparm(*args, **kwargs)


def lincom(result, *args, **kwargs):
    return result.lincom(*args, **kwargs)


def nlcom(result, *args, **kwargs):
    return result.nlcom(*args, **kwargs)


def predict(result, *args, **kwargs):
    return result.predict(*args, **kwargs)


def margins(result, *args, **kwargs):
    return result.margins(*args, **kwargs)
