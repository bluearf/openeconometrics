"""OLS contrasts, predictions, marginal effects and specification diagnostics.

The fitted covariance drives coefficient and marginal-effect inference. Classical
deletion diagnostics follow the regress postestimation formulas and restrictions:
https://www.stata.com/manuals/rregresspostestimation.pdf. Auxiliary regressions
use PyTorch SVD, and scalar reference tails use OpenEconometrics's beta routine and an
independent incomplete-gamma implementation. No estimator package is imported.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import fnmatch
from itertools import product
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.inference import _regularized_beta, critical_value, student_t_two_sided
from openecon.frame import DataFrame, as_frame

_EPS = torch.finfo(torch.float64).eps
_AUX_BYTES = 128 * 1024 * 1024


def _error(code: str, message: str):
    raise AnalysisError(code, message)


def _prediction_vector(state, name, x, fallback):
    callback = state.get(name)
    if callable(callback):
        finite = torch.isfinite(x).all(dim=1)
        value = torch.full((len(x),), float("nan"), dtype=torch.float64, device=x.device)
        if bool(finite.any()):
            selected = torch.as_tensor(callback(x[finite]), dtype=torch.float64, device=x.device)
            if selected.shape != (int(finite.sum()),):
                _error("invalid_postestimation", "The fitted prediction basis returned an incompatible row statistic.")
            value[finite] = selected
    else:
        value = torch.as_tensor(fallback(), dtype=torch.float64, device=x.device)
    if value.shape != (len(x),):
        _error("invalid_postestimation", "The fitted prediction basis returned an incompatible row statistic.")
    return value


def _gamma_q(shape: float, value: float) -> float:
    """Regularized upper incomplete gamma: convergent series or Lentz fraction."""
    if not math.isfinite(shape) or shape <= 0 or math.isnan(value) or value < 0:
        _error("invalid_inference", "Chi-squared arguments must be positive and finite.")
    if value == 0:
        return 1.0
    if math.isinf(value):
        return 0.0
    factor = math.exp(shape * math.log(value) - value - math.lgamma(shape))
    if value < shape + 1:
        term = total = 1 / shape
        for iteration in range(1, 10001):
            term *= value / (shape + iteration)
            total += term
            if abs(term) <= abs(total) * 4e-15:
                return min(1.0, max(0.0, 1 - total * factor))
    else:
        tiny = 1e-300
        b = value + 1 - shape
        c, d = 1 / tiny, 1 / max(tiny, b)
        result = d
        for iteration in range(1, 10001):
            numerator = -iteration * (iteration - shape)
            b += 2
            d = numerator * d + b
            c = b + numerator / c
            if abs(d) < tiny:
                d = math.copysign(tiny, d)
            if abs(c) < tiny:
                c = math.copysign(tiny, c)
            d = 1 / d
            delta = d * c
            result *= delta
            if abs(delta - 1) <= 4e-15:
                return min(1.0, max(0.0, factor * result))
    _error("inference_nonconvergence", "The chi-squared tail did not converge.")


def _chi2_sf(statistic: float, df: int) -> float:
    return _gamma_q(df / 2, statistic / 2)


def _f_sf(statistic: float, numerator: int, denominator: float) -> float:
    if numerator <= 0 or denominator <= 0 or not math.isfinite(denominator) or math.isnan(statistic) or statistic < 0:
        _error("invalid_inference", "F statistics require nonnegative values and positive finite degrees of freedom.")
    if statistic == 0:
        return 1.0
    if math.isinf(statistic):
        return 0.0
    if numerator == 1:
        return student_t_two_sided(math.sqrt(statistic), denominator)
    ratio = numerator * statistic / denominator
    if math.isinf(ratio):
        return 0.0
    if ratio == 0:
        return 1.0
    if denominator >= 1_000_000 and float(numerator).is_integer():
        # Incomplete-beta recurrence in its second parameter adds positive
        # terms. Start at b=1 (even numerator) or Student-t b=1/2 (odd).
        # This avoids almost-equal log-gamma subtraction and rounding 1-x
        # near a hundred-billion-observation denominator. Finite df remains.
        # https://dlmf.nist.gov/8.17.iv; https://dlmf.nist.gov/5.11.i
        b = denominator / 2
        log_x = -math.log1p(ratio)
        if int(numerator) % 2 == 0:
            terms = [b * log_x]
            previous = terms[0]
            for j in range(1, int(numerator) // 2):
                factor = (numerator * statistic / 2 + (j - 1) * ratio) / ((1 + ratio) * j)
                previous += math.log(factor)
                terms.append(previous)
        else:
            base = student_t_two_sided(math.sqrt(numerator * statistic), denominator)
            terms = [math.log(base)] if base > 0 else []
            for j in range((int(numerator) - 1) // 2):
                a = j + .5
                # Stirling difference evaluated with log1p, not lgamma(b+a)-lgamma(b).
                inverse, shifted_inverse = 1 / b, 1 / (b + a)
                correction = shifted_inverse / 12 - shifted_inverse ** 3 / 360 - inverse / 12 + inverse ** 3 / 360
                gamma_ratio = a * math.log(b) + (b + a - .5) * math.log1p(a / b) - a + correction
                terms.append(b * log_x + a * (math.log(ratio) + log_x)
                             + gamma_ratio - math.lgamma(a + 1))
        maximum = max(terms, default=-math.inf)
        if maximum == -math.inf:
            return 0.0
        log_tail = maximum + math.log(math.fsum(math.exp(term - maximum) for term in terms))
        return min(1.0, math.exp(log_tail))
    return _regularized_beta(denominator / 2, numerator / 2, 1 / (1 + ratio))


def f_sf(value: float, df1: int, df2: float) -> float:
    """Upper F tail used by the OLS fit's joint model test."""
    return _f_sf(value, df1, df2)


def _tensor(value, *, ndim: int | None = None) -> torch.Tensor:
    try:
        if torch.as_tensor(value).is_complex():
            _error("invalid_postestimation", "Postestimation inputs must be real numbers.")
        result = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_postestimation", "Expected real numeric values.") from exc
    if (ndim is not None and result.ndim != ndim) or not bool(torch.isfinite(result).all()):
        _error("invalid_postestimation", "Postestimation inputs must have the required shape and finite values.")
    return result


def _alpha(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
        _error("invalid_inference", "alpha must lie strictly between zero and one.")
    return float(value)


def _contrast_covariance(state, covariance, matrix):
    callback = state.get("contrast_covariance")
    value = _tensor(callback(matrix) if callable(callback) else matrix @ covariance @ matrix.T, ndim=2)
    if value.shape != (len(matrix), len(matrix)):
        _error("invalid_postestimation", "The fitted contrast basis returned an incompatible covariance matrix.")
    return (value + value.T) / 2


def _linear_values(state, parameters, matrix):
    callback = state.get("prediction_fitted")
    value = _tensor(callback(matrix) if callable(callback) else matrix @ parameters, ndim=1)
    if value.shape != (len(matrix),):
        _error("invalid_postestimation", "The fitted contrast basis returned an incompatible estimate vector.")
    return value


def _wald(parameters, covariance, matrix, target, df, *, state=None) -> dict:
    state = state or {}
    q = len(matrix)
    if matrix.ndim != 2 or q < 1 or matrix.shape[1] != len(parameters):
        _error("invalid_restrictions", "Restrictions need one column per fitted coefficient.")
    basis = state.get("contrast_basis")
    rank_matrix = _tensor(basis(matrix), ndim=2) if callable(basis) else matrix
    if rank_matrix.shape != matrix.shape:
        _error("invalid_postestimation", "The fitted contrast basis returned incompatible restriction dimensions.")
    # Restrictions change units under an anchored design transform. Assess
    # their row space there, with each row scaled independently; raw-unit SVD
    # can call [Intercept + 1e8*x, x] dependent despite an identified model.
    row_scale = rank_matrix.abs().amax(dim=1).clamp_min(torch.finfo(torch.float64).tiny)
    if int(torch.linalg.matrix_rank(rank_matrix / row_scale[:, None])) != q:
        _error("invalid_restrictions", "Restriction rows must be linearly independent.")
    difference = _linear_values(state, parameters, matrix) - target
    variance = _contrast_covariance(state, covariance, matrix)
    diagonal = variance.diagonal()
    if bool((diagonal <= 0).any()):
        _error("nonestimable_restriction", "The restriction has no positive fitted sampling variance.")
    standard_error = diagonal.sqrt()
    standardized = variance / standard_error[:, None] / standard_error[None, :]
    eigenvalues = torch.linalg.eigvalsh(standardized)
    if (not bool(torch.isfinite(eigenvalues).all())
            or float(eigenvalues[0]) <= _EPS * q * float(eigenvalues[-1])):
        _error("nonestimable_restriction", "The restriction has no positive fitted sampling variance.")
    scaled_difference = difference / standard_error
    statistic = float(scaled_difference @ torch.linalg.solve(standardized, scaled_difference))
    if not math.isfinite(statistic) or statistic < 0:
        _error("invalid_inference", "The fitted covariance did not produce a finite Wald statistic.")
    if df is None:
        result = {"statistic": statistic, "distribution": "chi2", "df": q,
                  "p_value": _chi2_sf(statistic, q)}
    else:
        statistic /= q
        result = {"statistic": statistic, "distribution": "F", "df_num": q,
                  "df_denom": df, "p_value": _f_sf(statistic, q, df)}
    if q == 1:
        result["t_statistic" if df is not None else "z_statistic"] = float(scaled_difference[0])
    result["restriction_matrix"], result["null_values"] = matrix.tolist(), target.tolist()
    return result


def _auxiliary(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor | None = None) -> dict:
    """Rank-aware scaled SVD; auxiliary dummy cross-products may be redundant."""
    if 8 * x.numel() * 8 > _AUX_BYTES:
        _error("diagnostic_memory_limit", "The auxiliary diagnostic exceeds its working-memory budget; reduce its variables.")
    weights = torch.ones(len(y), dtype=torch.float64) if weights is None else weights
    scale = x.abs().amax(dim=0).clamp_min(torch.finfo(torch.float64).tiny)
    scaled = x / scale
    weighted = scaled * weights.sqrt()[:, None]
    u, singular, vh = torch.linalg.svd(weighted, full_matrices=False)
    if len(singular) == 0 or float(singular[0]) == 0:
        _error("invalid_diagnostic", "The auxiliary diagnostic contains no estimable columns.")
    rank = int((singular > _EPS * max(x.shape) * singular[0]).sum())
    normalized = vh[:rank].T @ ((u[:, :rank].T @ (y * weights.sqrt())) / singular[:rank])
    fitted = scaled @ normalized
    residual = y - fitted
    rss = float(torch.dot(weights, residual.square()))
    mean = float(torch.dot(weights, y) / weights.sum())
    total = float(torch.dot(weights, (y - mean).square()))
    inverse = (vh[:rank].T / singular[:rank]) / scale[:, None]
    return {"params": normalized / scale, "bread": inverse @ inverse.T, "rank": rank,
            "fitted": fitted, "resid": residual, "rss": rss, "total": total,
            "r_squared": max(0.0, min(1.0, 1 - rss / total)) if total > 0 else 0.0}


class OLSPostestimation:
    """Pure postestimation mixin for a persisted ResultBundle plus private state."""

    def _parameter_state(self):
        state = getattr(self, "_state", None) or {}
        if "params" in state:
            parameters = _tensor(state["params"], ndim=1)
            covariance = _tensor(state["covariance"], ndim=2)
            terms = list(state["terms"])
        else:
            parameters = _tensor([item.estimate for item in self.coefficients], ndim=1)
            covariance = _tensor(self.covariance_matrix, ndim=2)
            terms = [item.term for item in self.coefficients]
        if len(terms) != len(parameters) or covariance.shape != (len(parameters), len(parameters)):
            _error("invalid_postestimation", "Stored coefficient and covariance dimensions disagree.")
        inference = getattr(self, "inference", {})
        df = state.get("df_inference", inference.get("df_inference", inference.get("df_resid")))
        if not inference.get("use_t", True):
            df = None
        if df is not None and (not math.isfinite(df) or df <= 0):
            _error("invalid_inference", "The fitted inference degrees of freedom are unavailable.")
        return state, parameters, covariance, terms, df

    def _dense_state(self):
        state = self._parameter_state()[0]
        if any(name not in state for name in ("x", "y", "resid", "fitted", "bread", "frame")):
            _error("estimation_state_unavailable", "This operation needs the retained estimation data. Refit locally; use iter_predict for a streamed fit.")
        return state

    def _contrast(self, mapping, terms) -> torch.Tensor:
        if not isinstance(mapping, Mapping) or not mapping:
            _error("invalid_restrictions", "Specify a nonempty mapping of term names to contrast weights.")
        vector = torch.zeros(len(terms), dtype=torch.float64)
        for term, weight in mapping.items():
            if term not in terms:
                _error("unknown_term", f"Unknown or omitted coefficient: {term}.")
            value = _tensor(weight)
            if value.ndim != 0:
                _error("invalid_restrictions", "Contrast weights must be finite scalars.")
            vector[terms.index(term)] = value
        return vector

    def test(self, restrictions, value=0) -> dict:
        """Wald test; a mapping describes contrast weights, and value is the RHS."""
        state, parameters, covariance, terms, df = self._parameter_state()
        if isinstance(restrictions, str):
            restrictions = [restrictions]
        if isinstance(restrictions, Mapping):
            matrix = self._contrast(restrictions, terms)[None, :]
        elif (isinstance(restrictions, Sequence) and restrictions
              and all(isinstance(item, str) for item in restrictions)):
            if len(set(restrictions)) != len(restrictions):
                _error("invalid_restrictions", "Specify each tested coefficient only once.")
            matrix = torch.stack([self._contrast({term: 1}, terms) for term in restrictions])
        elif (isinstance(restrictions, Sequence) and restrictions
              and all(isinstance(item, Mapping) for item in restrictions)):
            matrix = torch.stack([self._contrast(item, terms) for item in restrictions])
        else:
            matrix = _tensor(restrictions)
            if matrix.ndim == 1:
                matrix = matrix[None, :]
        target = _tensor(value)
        if target.ndim == 0:
            target = target.expand(len(matrix))
        if target.shape != (len(matrix),):
            _error("invalid_restrictions", "Provide one null value per restriction row.")
        result = _wald(parameters, covariance, matrix, target, df, state=state)
        if len(matrix) == 1:
            adjusted_df, scale = self._contrast_controls(matrix[0], df)
            if adjusted_df != df or scale != 1:
                statistic = scale * result["t_statistic" if df is not None else "z_statistic"]
                result.update(statistic=statistic * statistic, distribution="F", df_num=1,
                              df_denom=adjusted_df, t_statistic=statistic,
                              p_value=student_t_two_sided(statistic, adjusted_df), statistic_scale=scale)
        elif self._adjusted_inference():
            result["df_adjustment_note"] = "Joint Wald uses conventional fitted degrees of freedom; contrast-specific adjustments apply to single restrictions."
        return result

    def testparm(self, patterns) -> dict:
        """Jointly test exact terms, wildcard patterns or a factor's dummy terms."""
        terms = self._parameter_state()[3]
        if not isinstance(patterns, (str, list, tuple)):
            _error("invalid_restrictions", "Provide term names or wildcard patterns.")
        patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        if not patterns or any(not isinstance(pattern, str) or not pattern for pattern in patterns):
            _error("invalid_restrictions", "Provide at least one term or wildcard pattern.")
        selected = []
        for pattern in patterns:
            matched = [term for term in terms if term == pattern or fnmatch.fnmatchcase(term, pattern)
                       or term.startswith(pattern + "[")]
            if not matched:
                _error("unknown_term", f"No fitted coefficients match {pattern}.")
            selected.extend(term for term in matched if term not in selected)
        result = self.test(selected)
        result["terms"] = selected
        return result

    def _adjusted_inference(self):
        state = getattr(self, "_state", None) or {}
        inference = getattr(self, "inference", {})
        return callable(state.get("contrast_inference")) or inference.get("dfadjust", False) or inference.get("hansen", False)

    def _contrast_controls(self, gradient, df):
        state = getattr(self, "_state", None) or {}
        control = state.get("contrast_inference")
        scale = 1.0
        if callable(control):
            adjusted = control(gradient)
            df, scale = adjusted["df"], adjusted["scale"]
        elif self._adjusted_inference():
            inference = self.inference
            indices = gradient.nonzero().flatten().tolist()
            if len(indices) != 1 or len(inference.get("coefficient_df", [])) != len(gradient):
                _error("estimation_state_unavailable", "This adjusted contrast requires the retained reference design; refit locally.")
            df = inference["coefficient_df"][indices[0]]
            scale = inference["coefficient_scale"][indices[0]]
        if (not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0
                or (df is not None and (not math.isfinite(df) or df <= 0))):
            _error("invalid_inference", "The contrast adjustment returned invalid degrees of freedom or scale.")
        return df, scale

    def _estimate_contrast(self, estimate: float, gradient: torch.Tensor, *, null: float = 0,
                           alpha: float | None = None) -> dict:
        state, _, covariance, terms, df = self._parameter_state()
        variance = float(_contrast_covariance(state, covariance, gradient[None, :])[0, 0])
        if not math.isfinite(variance) or variance <= 0 or not math.isfinite(estimate):
            _error("nonestimable_restriction", "The expression has no finite positive delta-method variance.")
        df, scale = self._contrast_controls(gradient, df)
        standard_error = math.sqrt(variance)
        statistic = scale * (estimate - null) / standard_error
        significance = _alpha(self.spec.alpha if alpha is None else alpha)
        critical = critical_value(significance, df)
        return {"estimate": estimate, "std_error": standard_error, "statistic": statistic,
                "distribution": "t" if df is not None else "normal", "df": df,
                "p_value": student_t_two_sided(statistic, df) if df is not None else math.erfc(abs(statistic) / math.sqrt(2)),
                "ci_low": estimate - critical * standard_error / scale, "ci_high": estimate + critical * standard_error / scale,
                "null": null, "alpha": significance,
                "statistic_scale": scale,
                "gradient": dict(zip(terms, gradient.tolist(), strict=True))}

    def lincom(self, mapping, constant=0) -> dict:
        """Estimate and test a linear combination plus a finite constant."""
        state, parameters, _, terms, _ = self._parameter_state()
        gradient = self._contrast(mapping, terms)
        constant = _tensor(constant)
        if constant.ndim != 0:
            _error("invalid_restrictions", "The linear-combination constant must be a finite scalar.")
        estimate = _linear_values(state, parameters, gradient[None, :])[0]
        return self._estimate_contrast(float(estimate + constant), gradient)

    def nlcom(self, function, null=0) -> dict:
        """Delta-method inference for a differentiable scalar Torch expression."""
        if not callable(function):
            _error("invalid_nlcom", "nlcom requires a callable receiving a term-to-Torch-scalar mapping.")
        _, parameters, _, terms, _ = self._parameter_state()
        null = _tensor(null)
        if null.ndim != 0:
            _error("invalid_nlcom", "The nonlinear null value must be a finite scalar.")
        try:
            with torch.inference_mode(False), torch.enable_grad():
                variable = parameters.detach().clone().requires_grad_(True)
                result = function(dict(zip(terms, variable.unbind(), strict=True)))
                if not isinstance(result, torch.Tensor) or result.ndim != 0 or not bool(torch.isfinite(result)):
                    _error("invalid_nlcom", "The nonlinear callable must return one finite differentiable Torch scalar.")
                gradient, = torch.autograd.grad(result, variable)
                if not bool(torch.isfinite(gradient).all()):
                    _error("invalid_nlcom", "The nonlinear derivative is not finite at the fitted coefficients.")
                estimate = float(result.detach())
                gradient = gradient.detach()
        except AnalysisError:
            raise
        except (RuntimeError, TypeError, ValueError, KeyError, ZeroDivisionError) as exc:
            raise AnalysisError("invalid_nlcom", "The nonlinear expression is undefined or does not retain Torch derivatives.") from exc
        return self._estimate_contrast(estimate, gradient, null=float(null))

    def _classical_prediction(self, kind: str):
        if self.spec.covariance != "nonrobust":
            _error("unsupported_prediction_vce", f"{kind} requires classical OLS covariance; use xb, residuals or stdp with this VCE.")
        weight_type = getattr(self.spec, "weight_type", None)
        if weight_type in {"aweight", "pweight", "iweight"} and kind != "leverage":
            _error("unsupported_prediction_weight", f"{kind} is not defined for {weight_type} estimation.")

    def _sigma2(self, state):
        variance = state.get("sigma2")
        if variance is None:
            weights = state.get("weights", torch.ones_like(state["resid"]))
            variance = float(torch.dot(weights, state["resid"].square()) / state["df_resid"])
        if not math.isfinite(variance) or variance <= 0:
            _error("invalid_postestimation", "Classical residual variance is unavailable.")
        return float(variance)

    def _prediction_input(self, data, state, parameters):
        if data is None:
            dense = self._dense_state()
            frame = dense.get("original_frame")
            if frame is None:
                return dense["frame"], dense["x"], torch.arange(len(dense["x"])), False
        else:
            frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        encoder = state.get("encode")
        if not callable(encoder):
            _error("estimation_state_unavailable", "New-data prediction needs the fitted design encoder; refit the model locally.")
        columns = state.get("predictor_columns", getattr(state.get("design"), "required", self.spec.predictors))
        design = state.get("design")
        lagged = bool(getattr(design, "lag_specs", None))
        current = list(state.get("current_predictor_columns", columns))
        if lagged:
            columns = list(dict.fromkeys([*columns, design.time]))
            current = list(dict.fromkeys([*current, design.time]))
        missing = set(columns) - set(frame.columns)
        if missing:
            _error("missing_columns", f"Missing predictor columns: {', '.join(sorted(missing))}.")
        keep = ~frame.loc[:, current].isna().any(axis=1)
        positions = torch.tensor([i for i, valid in enumerate(keep) if valid], dtype=torch.int64)
        if lagged:
            # A row excluded by its current inputs remains part of the raw
            # history for a later row's lag. Encode before selecting targets.
            full = design.encode(frame, allow_missing=True)[:, state.get("kept_indices", list(range(len(parameters))))]
            x = full[positions]
        else:
            x = encoder(frame.loc[keep]) if len(positions) else torch.empty((0, len(parameters)), dtype=torch.float64)
        if x.ndim != 2 or x.shape != (len(positions), len(parameters)):
            _error("invalid_postestimation", "The fitted encoder returned an incompatible prediction design.")
        return frame, x, positions, True

    def predict(self, data=None, kind="xb", alpha=None, *, interval=None, term=None) -> DataFrame:
        """Return full prediction rows; intervals are 'mean' or 'obs'.

        Classical influence diagnostics follow Stata's VCE/weight restrictions.
        DFBETA, COVRATIO, DFITS and Welsch are estimation-sample only. Exported
        results preserve coefficient tests but need a local refit for row methods.
        """
        state, parameters, covariance, terms, df = self._parameter_state()
        if not isinstance(kind, str) or (interval is not None and not isinstance(interval, str)):
            _error("invalid_prediction", "Prediction kind and interval must be names.")
        aliases = {"resid": "residuals", "residual": "residuals", "score": "residuals",
                   "hat": "leverage", "cooksd": "cook", "dffits": "dfits", "dfbetas": "dfbeta"}
        kind = aliases.get(kind, kind)
        if kind in {"mean", "obs", "observation"}:
            interval, kind = ("mean" if kind == "mean" else "obs"), "xb"
        if interval == "observation":
            interval = "obs"
        allowed = {"xb", "residuals", "stdp", "stdf", "stdr", "rstandard", "rstudent",
                   "leverage", "cook", "covratio", "dfits", "welsch", "dfbeta"}
        if kind not in allowed or interval not in {None, "mean", "obs"}:
            _error("invalid_prediction", "Unknown OLS prediction statistic or interval type.")
        if interval is not None and kind != "xb":
            _error("invalid_prediction", "Prediction intervals require kind='xb', 'mean' or 'obs'.")
        significance = _alpha(self.spec.alpha if alpha is None else alpha)
        if kind not in {"xb", "residuals", "stdp"}:
            self._classical_prediction(kind)
        if interval == "obs":
            self._classical_prediction("stdf")
        sample_only = kind in {"dfbeta", "covratio", "dfits", "welsch"}
        if sample_only and data is not None:
            _error("prediction_sample_required", f"{kind} is available only on the estimation sample; omit data=.")
        if sample_only:
            dense = self._dense_state()
            frame, x, positions, original = dense["frame"], dense["x"], torch.arange(len(dense["x"])), False
        else:
            frame, x, positions, original = self._prediction_input(data, state, parameters)
        fitted = _prediction_vector(state, "prediction_fitted", x, lambda: x @ parameters)
        standard = _prediction_vector(state, "prediction_variance", x, lambda: (x @ covariance * x).sum(dim=1))
        if bool((standard < -100 * _EPS * standard.abs().clamp_min(1)).any()):
            _error("invalid_covariance", "Prediction variance is negative.")
        stdp = standard.clamp_min(0).sqrt()
        columns = {}
        if kind == "xb":
            columns["xb"] = fitted
        elif kind == "stdp":
            columns[kind] = stdp
        else:
            if kind == "residuals":
                if self.spec.outcome not in frame:
                    _error("missing_outcome", "Residual prediction needs the outcome column.")
                outcome = torch.tensor(pd.to_numeric(frame.iloc[positions.tolist()][self.spec.outcome], errors="raise").to_numpy(dtype="float64", na_value=float("nan")), dtype=torch.float64)
                columns[kind] = outcome - fitted
            elif kind == "stdf":
                columns[kind] = (self._sigma2(state) + standard).sqrt()
            else:
                dense = self._dense_state()
                variance = self._sigma2(dense)
                h = _prediction_vector(dense, "prediction_leverage", x, lambda: (x @ dense["bread"] * x).sum(dim=1))
                if getattr(self.spec, "weight_type", None) == "aweight":
                    weights_name = getattr(self.spec, "weights", None)
                    if original or data is not None:
                        if weights_name not in frame:
                            _error("missing_weights", "Weighted leverage prediction needs the fitted weight column.")
                        raw_fit = torch.tensor(dense["frame"][weights_name].to_numpy(dtype="float64"), dtype=torch.float64)
                        scale = float((dense["weights"] / raw_fit).median())
                        h = h * torch.tensor(frame.iloc[positions.tolist()][weights_name].to_numpy(dtype="float64"), dtype=torch.float64) * scale
                    else:
                        h = h * dense["weights"]
                complement = 1 - h
                positive = complement > 100 * _EPS
                safe_complement = torch.where(positive, complement, torch.full_like(complement, float("nan")))
                if kind == "leverage":
                    columns[kind] = h
                elif kind == "stdr":
                    columns[kind] = (variance * safe_complement).sqrt()
                else:
                    if sample_only:
                        residual = dense["resid"]
                    else:
                        if self.spec.outcome not in frame:
                            _error("missing_outcome", f"{kind} prediction needs the outcome column.")
                        outcome = torch.tensor(frame.iloc[positions.tolist()][self.spec.outcome].to_numpy(dtype="float64", na_value=float("nan")), dtype=torch.float64)
                        residual = outcome - fitted
                    standardized = residual / (variance * safe_complement).sqrt()
                    if kind == "rstandard":
                        columns[kind] = standardized
                    elif kind == "cook":
                        columns[kind] = standardized.square() * h / (len(parameters) * safe_complement)
                    else:
                        residual_df = dense["df_resid"]
                        if residual_df <= 1:
                            _error("insufficient_observations", "Deleted-observation diagnostics require residual degrees of freedom greater than one.")
                        deleted = (variance * residual_df - residual.square() / safe_complement) / (residual_df - 1)
                        deleted = torch.where(deleted > 0, deleted, torch.full_like(deleted, float("nan")))
                        student = residual / (deleted * safe_complement).sqrt()
                        dfits = student * (h.clamp_min(0) / safe_complement).sqrt()
                        if kind == "rstudent":
                            columns[kind] = student
                        elif kind == "dfits":
                            columns[kind] = dfits
                        elif kind == "welsch":
                            columns[kind] = dfits * ((dense["nobs"] - 1) / safe_complement).sqrt()
                        elif kind == "covratio":
                            columns[kind] = (deleted / variance).pow(len(parameters)) / safe_complement
                        else:
                            differences = (x @ dense["bread"]) * (residual / safe_complement)[:, None]
                            dfbeta = differences / (deleted[:, None] * dense["bread"].diag()).sqrt()
                            selected = terms if term is None else [term]
                            if any(name not in terms for name in selected):
                                _error("unknown_term", "DFBETA term is unknown or omitted.")
                            columns.update({f"dfbeta[{name}]": dfbeta[:, terms.index(name)] for name in selected})
        if interval is not None:
            se = stdp
            if interval == "obs":
                se = (standard + self._sigma2(state)).sqrt()
            critical = critical_value(significance, df)
            columns.update({"stdp" if interval == "mean" else "stdf": se,
                            "ci_low": fitted - critical * se, "ci_high": fitted + critical * se})
        if data is None and not original:
            original_rows = state.get("original_rows", len(frame))
            output_positions = _tensor(state.get("positions", torch.arange(len(frame)))).to(torch.int64)
            index = state.get("original_index", pd.RangeIndex(original_rows))
        else:
            original_rows, output_positions, index = len(frame), positions, frame.index
        output = {}
        for name, values in columns.items():
            scattered = torch.full((original_rows,), float("nan"), dtype=torch.float64)
            scattered[output_positions] = values
            output[name] = scattered.tolist()
        result = as_frame(pd.DataFrame(output, index=index))
        result.attrs.update({"prediction": kind, "interval": interval, "alpha": significance,
                             "inference_distribution": "t" if df is not None else "normal"})
        return result

    def margins(self, variables=None, at=None, method="ame", *, _return_gradients=False) -> DataFrame:
        """Continuous derivatives and categorical reference-level contrasts.

        AME averages over the fitted covariates. MEM sets continuous covariates
        to their weighted means and averages the observed factor distribution;
        it never substitutes a category mode. Interactions are re-encoded.
        """
        state = self._dense_state()
        _, parameters, _, _, _ = self._parameter_state()
        if method not in {"ame", "mem"}:
            _error("invalid_margins", "method must be 'ame' or 'mem'.")
        if getattr(state.get("design"), "lag_specs", None):
            _error("unsupported_margins_transform", "Marginal effects for lag/difference formulas need an explicit intervention over time. Use lincom on the fitted lag/difference coefficients.")
        encoder = state.get("encode")
        if not callable(encoder):
            _error("estimation_state_unavailable", "Marginal effects need the fitted design encoder.")
        categories = getattr(state.get("design"), "categories", state.get("categorical", {}))
        if not isinstance(categories, Mapping):
            categories = {}
        required = state.get("predictor_columns", getattr(state.get("design"), "required", self.spec.predictors))
        variables = required if variables is None else [variables] if isinstance(variables, str) else list(variables)
        if not variables or len(set(variables)) != len(variables) or any(name not in required for name in variables):
            _error("invalid_margins", "Select distinct original predictor columns for marginal effects.")
        weights = state.get("weights", torch.ones(len(state["frame"]), dtype=torch.float64))
        normalized_weights = weights / weights.sum()
        base = state["frame"].copy(deep=True)
        if method == "mem":
            for name in required:
                if name not in categories and pd.api.types.is_numeric_dtype(base[name].dtype):
                    values = torch.tensor(base[name].to_numpy(dtype="float64"), dtype=torch.float64)
                    base[name] = float(torch.dot(normalized_weights, values))
        at = {} if at is None else at
        if not isinstance(at, Mapping) or any(name not in required for name in at):
            _error("invalid_margins", "at must map fitted predictor columns to scalar values or finite grids.")
        grids = []
        for name, value in at.items():
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            if not values:
                _error("invalid_margins", "Marginal-effect grids cannot be empty.")
            for point in values:
                if name in categories:
                    if point not in categories[name]:
                        _error("unknown_category", f"Column '{name}' has an unfitted at= category level.")
                else:
                    scalar = _tensor(point)
                    if scalar.ndim != 0:
                        _error("invalid_margins", "Numeric evaluation grids require finite scalar values.")
            grids.append((name, values))
        combinations = math.prod(len(values) for _, values in grids)
        if combinations > 1000:
            _error("margins_grid_limit", "The marginal-effect grid supports at most 1000 combinations.")
        rows, delta_gradients = [], []
        for values in product(*(values for _, values in grids)):
            scenario = base.copy(deep=True)
            setting = dict(zip((name for name, _ in grids), values, strict=True))
            for name, value in setting.items():
                scenario[name] = value
            for variable in variables:
                if variable in categories:
                    levels = list(categories[variable])
                    if len(levels) < 2:
                        continue
                    reference = scenario.copy(deep=True)
                    reference[variable] = levels[0]
                    baseline = encoder(reference)
                    gradients = []
                    for level in levels[1:]:
                        alternative = scenario.copy(deep=True)
                        alternative[variable] = level
                        gradient = normalized_weights @ (encoder(alternative) - baseline)
                        gradients.append((f"{variable}[{level}]", gradient))
                else:
                    raw = torch.tensor(pd.to_numeric(scenario[variable], errors="raise").to_numpy(dtype="float64"), dtype=torch.float64)
                    if not bool(torch.isfinite(raw).all()):
                        _error("invalid_margins", "Continuous evaluation values must be finite.")
                    step = _EPS ** (1 / 5) * torch.where(raw == 0, torch.ones_like(raw), raw.abs())
                    step = step.clamp_min(torch.finfo(torch.float64).tiny)
                    designs = []
                    for multiplier in (-2, -1, 1, 2):
                        shifted = scenario.copy(deep=True)
                        shifted[variable] = (raw + multiplier * step).tolist()
                        designs.append(encoder(shifted))
                    derivative = (designs[0] - 8 * designs[1] + 8 * designs[2] - designs[3]) / (12 * step[:, None])
                    gradients = [(variable, normalized_weights @ derivative)]
                for label, gradient in gradients:
                    if not bool(torch.isfinite(gradient).all()):
                        _error("invalid_margins", "The marginal-effect derivative is undefined at these covariates.")
                    estimate = float(_linear_values(state, parameters, gradient[None, :])[0])
                    contrast = self._estimate_contrast(estimate, gradient)
                    if _return_gradients:
                        delta_gradients.append(gradient.tolist())
                    rows.append({"variable": label, "method": method, **{f"at[{name}]": value for name, value in setting.items()},
                                 **{key: value for key, value in contrast.items() if key != "gradient"}})
        result = as_frame(pd.DataFrame(rows))
        if _return_gradients:
            result.attrs["delta_gradients"] = delta_gradients
        return result

    def _diagnostic_weights(self, state):
        if getattr(self.spec, "weight_type", None) not in {None, "fweight"}:
            _error("unsupported_diagnostic_weight", "This residual specification test requires unweighted or frequency-weighted OLS.")
        return state.get("weights", torch.ones(len(state["y"]), dtype=torch.float64))

    def vif(self, *, uncentered=False) -> DataFrame:
        state = self._dense_state()
        x, terms = state["x"], state["terms"]
        weights = state.get("weights", torch.ones(len(x), dtype=torch.float64))
        centered = self.spec.intercept and not uncentered
        rows = []
        for column, term in enumerate(terms):
            if term in {"Intercept", "_cons", "const"}:
                continue
            other = [i for i in range(x.shape[1]) if i != column]
            outcome = x[:, column]
            residual = _auxiliary(x[:, other], outcome, weights)["resid"] if other else outcome
            denominator = float(torch.dot(weights, residual.square()))
            centered_outcome = outcome - torch.dot(weights, outcome) / weights.sum() if centered else outcome
            numerator = float(torch.dot(weights, centered_outcome.square()))
            value = numerator / denominator if denominator > 0 else math.inf
            rows.append({"term": term, "vif": value, "tolerance": 1 / value if value > 0 else 0,
                         "centered": bool(centered)})
        return as_frame(pd.DataFrame(rows, columns=["term", "vif", "tolerance", "centered"]))

    def hettest(self, variables=None, *, rhs=False, method="normal") -> dict:
        """Breusch-Pagan/Cook-Weisberg; iid selects Koenker's N*R² form."""
        state = self._dense_state()
        weights = self._diagnostic_weights(state)
        if method not in {"normal", "iid", "fstat"}:
            _error("invalid_diagnostic", "Heteroskedasticity test method must be normal, iid or fstat.")
        x = state["x"]
        if variables is not None:
            variables = [variables] if isinstance(variables, str) else list(variables)
            terms = state["terms"]
            if not variables or any(term not in terms for term in variables):
                _error("unknown_term", "Heteroskedasticity test variables must be fitted design terms.")
            z = x[:, [terms.index(term) for term in variables]]
        elif rhs:
            z = x[:, [i for i, term in enumerate(state["terms"]) if term not in {"Intercept", "_cons", "const"}]]
        else:
            z = state["fitted"][:, None]
        n = state["nobs"]
        squared = state["resid"].square()
        response = squared / (torch.dot(weights, squared) / weights.sum())
        auxiliary = _auxiliary(torch.cat((torch.ones((len(x), 1), dtype=torch.float64), z), dim=1), response, weights)
        degrees = auxiliary["rank"] - 1
        if degrees < 1:
            _error("invalid_diagnostic", "No nonconstant variance predictors are available.")
        if method == "fstat":
            statistic = (auxiliary["total"] - auxiliary["rss"]) / degrees / (auxiliary["rss"] / (n - auxiliary["rank"]))
            return {"statistic": statistic, "distribution": "F", "df_num": degrees,
                    "df_denom": n - auxiliary["rank"], "p_value": _f_sf(statistic, degrees, n - auxiliary["rank"]),
                    "method": method, "null": "constant variance"}
        statistic = ((auxiliary["total"] - auxiliary["rss"]) / 2 if method == "normal"
                     else n * auxiliary["r_squared"])
        return {"statistic": max(0.0, statistic), "distribution": "chi2", "df": degrees,
                "p_value": _chi2_sf(max(0.0, statistic), degrees), "method": method, "null": "constant variance"}

    def white_test(self) -> dict:
        state = self._dense_state()
        weights = self._diagnostic_weights(state)
        x = state["x"]
        width = x.shape[1]
        columns = width * (width + 1) // 2 + width + 1
        if 8 * len(x) * columns * 8 > _AUX_BYTES:
            _error("diagnostic_memory_limit", "White's expanded auxiliary design exceeds its working-memory budget.")
        cross = [x[:, i] * x[:, j] for i in range(width) for j in range(i, width)]
        # Include original terms for models without an intercept as well.
        auxiliary_x = torch.column_stack([torch.ones(len(x), dtype=torch.float64), *x.unbind(dim=1), *cross])
        auxiliary = _auxiliary(auxiliary_x, state["resid"].square(), weights)
        degrees = auxiliary["rank"] - 1
        if degrees <= 0:
            _error("invalid_diagnostic", "White's test has no nonconstant auxiliary terms.")
        statistic = state["nobs"] * auxiliary["r_squared"]
        return {"statistic": statistic, "distribution": "chi2", "df": degrees,
                "p_value": _chi2_sf(statistic, degrees), "null": "constant variance",
                "auxiliary_rank": auxiliary["rank"]}

    def reset_test(self, powers=(2, 3, 4)) -> dict:
        """Ramsey RESET, with normalized fitted powers and the fitted VCE."""
        state = self._dense_state()
        powers = [powers] if isinstance(powers, int) else list(powers)
        if not powers or any(type(power) is not int or power < 2 for power in powers) or len(set(powers)) != len(powers):
            _error("invalid_diagnostic", "RESET powers must be distinct integers of at least two.")
        if len(powers) > 32:
            _error("diagnostic_memory_limit", "RESET supports at most 32 fitted-value powers.")
        x, y = state["x"], state["y"]
        weights = state.get("weights", torch.ones(len(y), dtype=torch.float64))
        fitted = state["fitted"]
        spread = fitted.max() - fitted.min()
        if float(spread) <= 0:
            _error("invalid_diagnostic", "RESET requires varying fitted values.")
        normalized = (fitted - fitted.min()) / spread
        extra = torch.column_stack([normalized.pow(power) for power in powers])
        # Remove redundant powers before refitting, including binary designs.
        residual_extra = torch.column_stack([_auxiliary(x, column, weights)["resid"] for column in extra.unbind(dim=1)])
        u, singular, _ = torch.linalg.svd(residual_extra * weights.sqrt()[:, None], full_matrices=False)
        threshold = _EPS * max(residual_extra.shape) * max(1.0, float(extra.norm()))
        rank = int((singular > threshold).sum())
        if rank == 0:
            _error("invalid_diagnostic", "The requested RESET powers add no independent design columns.")
        augmented_x = torch.cat((x, u[:, :rank] / weights.sqrt()[:, None]), dim=1)
        residual_df = state["nobs"] - x.shape[1] - rank
        if residual_df <= 0:
            _error("insufficient_observations", "RESET requires positive augmented residual degrees of freedom.")
        refit = state.get("refit")
        if callable(refit):
            augmented = refit(augmented_x)
            parameters = _tensor(augmented.get("params", augmented.get("parameters")), ndim=1)
            covariance = _tensor(augmented["covariance"], ndim=2)
            inference_df = augmented.get("df_inference")
            if inference_df is not None and not math.isfinite(inference_df):
                inference_df = None
            if inference_df is None and self._parameter_state()[4] is not None:
                inference_df = (self._parameter_state()[4] if self.spec.covariance == "cluster" else residual_df)
        else:
            if self.spec.covariance != "nonrobust":
                _error("estimation_state_unavailable", "RESET with this VCE needs the fitted covariance refit callback.")
            augmented = _auxiliary(augmented_x, y, weights)
            parameters = augmented["params"]
            covariance = augmented["bread"] * (augmented["rss"] / residual_df)
            inference_df = residual_df
        matrix = torch.zeros((rank, augmented_x.shape[1]), dtype=torch.float64)
        matrix[:, -rank:] = torch.eye(rank, dtype=torch.float64)
        result = _wald(parameters, covariance, matrix, torch.zeros(rank, dtype=torch.float64), inference_df,
                       state=augmented.get("state", augmented))
        adjustment = augmented.get("contrast_inference")
        if rank == 1 and callable(adjustment):
            control = adjustment(matrix[0])
            adjusted_df, scale = control["df"], control["scale"]
            statistic = result["t_statistic" if inference_df is not None else "z_statistic"] * scale
            result.update(statistic=statistic ** 2, distribution="F", df_num=1,
                          df_denom=adjusted_df, t_statistic=statistic,
                          p_value=student_t_two_sided(statistic, adjusted_df), statistic_scale=scale)
        elif self._adjusted_inference():
            result["df_adjustment_note"] = "Joint RESET Wald uses conventional augmented-model degrees of freedom."
        result.update({"powers": powers, "added_rank": rank, "covariance": self.spec.covariance,
                       "null": "no omitted fitted-value powers"})
        return result

    def breusch_godfrey(self, lags=1) -> dict:
        """Residual serial-correlation LM/F tests; initial lag residuals are zero."""
        state = self._dense_state()
        self._diagnostic_weights(state)
        if getattr(self.spec, "weight_type", None) is not None:
            _error("unsupported_diagnostic_weight", "Breusch-Godfrey requires an unweighted ordered observation sample.")
        residual, x = state["resid"], state["x"]
        if type(lags) is not int or lags < 1 or lags >= len(x) - x.shape[1] - 1:
            _error("invalid_diagnostic", "Choose a positive lag count with sufficient residual degrees of freedom.")
        columns = [torch.ones(len(x), dtype=torch.float64), *x.unbind(dim=1)]
        for lag in range(1, lags + 1):
            columns.append(torch.cat((torch.zeros(lag, dtype=torch.float64), residual[:-lag])))
        auxiliary = _auxiliary(torch.column_stack(columns), residual)
        degrees = auxiliary["rank"] - _auxiliary(torch.column_stack(columns[:-lags]), residual)["rank"]
        if degrees < 1:
            _error("invalid_diagnostic", "Lagged residuals add no independent auxiliary columns.")
        statistic = len(x) * auxiliary["r_squared"]
        unrestricted_df = len(x) - auxiliary["rank"]
        restricted = _auxiliary(torch.column_stack(columns[:-lags]), residual)
        f_statistic = max(0.0, (restricted["rss"] - auxiliary["rss"]) / degrees / (auxiliary["rss"] / unrestricted_df))
        return {"statistic": statistic, "distribution": "chi2", "df": degrees,
                "p_value": _chi2_sf(statistic, degrees), "f_statistic": f_statistic,
                "f_p_value": _f_sf(f_statistic, degrees, unrestricted_df), "df_denom": unrestricted_df,
                "lags": lags, "ordering": "retained estimation row order", "initial_residual_lags": "zero",
                "null": "no residual serial correlation"}

    def diagnostics(self, *, lags=1) -> dict:
        """Return diagnostics without altering the model, data or result history."""
        state = self._dense_state()
        result = {}
        for name, calculate in (("vif", self.vif), ("breusch_pagan", self.hettest),
                                ("white", self.white_test), ("reset", self.reset_test),
                                ("breusch_godfrey", lambda: self.breusch_godfrey(lags))):
            try:
                result[name] = calculate()
            except AnalysisError as exc:
                result[name] = {"available": False, "code": exc.code, "message": str(exc)}
        residual = state["resid"]
        weights = state.get("weights", torch.ones_like(residual))
        centered = residual - torch.dot(weights, residual) / weights.sum()
        variance = torch.dot(weights, centered.square()) / weights.sum()
        weight_type = getattr(self.spec, "weight_type", None)
        if weight_type not in {None, "fweight"}:
            result["jarque_bera"] = {"available": False, "code": "unsupported_diagnostic_weight",
                                     "message": "Jarque-Bera requires unweighted or frequency-weighted residual moments."}
        elif float(variance) > 0:
            skew = float(torch.dot(weights, centered.pow(3)) / weights.sum() / variance.pow(1.5))
            kurtosis = float(torch.dot(weights, centered.pow(4)) / weights.sum() / variance.square())
            statistic = state["nobs"] / 6 * (skew * skew + (kurtosis - 3) ** 2 / 4)
            result["jarque_bera"] = {"statistic": statistic, "distribution": "chi2", "df": 2,
                                     "p_value": _chi2_sf(statistic, 2), "skewness": skew, "kurtosis": kurtosis}
        else:
            result["jarque_bera"] = {"available": False, "code": "constant_residuals"}
        if weight_type is None:
            result["durbin_watson"] = {"statistic": float(residual.diff().square().sum() / residual.square().sum()),
                                       "ordering": "retained estimation row order"}
        else:
            result["durbin_watson"] = {"available": False, "code": "unsupported_diagnostic_weight",
                                       "message": "Durbin-Watson requires an unweighted ordered observation sample."}
        try:
            influence = self.predict(kind="leverage")
            for kind in ("rstandard", "rstudent", "cook", "covratio", "dfits", "welsch"):
                influence[kind] = self.predict(kind=kind)[kind]
            result["influence"] = influence
        except AnalysisError as exc:
            result["influence"] = {"available": False, "code": exc.code, "message": str(exc)}
        return result
