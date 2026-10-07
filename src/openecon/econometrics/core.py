"""Shared sample, design and result layer for registry estimators.

Every estimator family goes through the same three steps:

1. ``ModelFrame(spec, data)`` selects the model-input columns, applies the
   explicit missing-data policy, validates weights and records which original
   rows are used. Estimators may restrict it further (singletons, lags) or
   sort it by panel and time; positions and provenance follow automatically.
2. The family's tensor kernel estimates parameters and their covariance.
3. ``build_result`` turns tensors into the persistable ``ResultBundle`` with
   uniform inference, hashes and provenance.

pandas is used for tabular work only; all numerical work is float64 Torch.
"""

from __future__ import annotations

import math
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import uuid4

import pandas as pd
import torch
from pandas.api.types import (
    is_bool_dtype, is_datetime64_any_dtype, is_integer_dtype, is_numeric_dtype,
)
from pydantic import ValidationError
from torch import Tensor

from openecon.analysis import (
    _coerce_frame, _frame_hasher, _json_scalar, _numeric, _position_bytes, _safe_metric,
)
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.engines.contracts import KernelError
from openecon.engines.inference import critical_value, two_sided_p_values
from openecon.models import Coefficient, ModelSpec, ResultBundle
from openecon.resources import plan_workspace, tensor_bytes, workspace_budget_bytes

PREDICTION_LIMIT = 400


@dataclass
class Design:
    """A float64 design block and the provenance of its columns."""

    x: Tensor                      # [n, k]
    terms: list[str]
    categories: dict[str, Any]     # treatment-coding record per categorical predictor
    intercept: bool                # column 0 is the constant

    def select(self, kept: Sequence[int]) -> Design:
        kept = list(kept)
        return Design(self.x[:, kept], [self.terms[i] for i in kept], self.categories,
                      self.intercept and 0 in kept)


def column_list(value: Any, name: str) -> list[str]:
    """Validate a convenience-function column list (a bare string is an error)."""
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AnalysisError("invalid_spec", f"{name} must be a list of column names, for example {name}=['education'].")
    return list(value)


def make_spec(estimator: str, **fields: Any) -> ModelSpec:
    """Build a ModelSpec for a convenience function, dropping unset extras.

    ``columns`` and ``options`` entries whose value is ``None`` (or an empty
    list) are omitted so that a spec records only what the caller chose.
    """
    for key in ("columns", "options"):
        fields[key] = {name: value for name, value in (fields.get(key) or {}).items()
                       if value is not None and not (isinstance(value, list) and not value and key == "columns")}
    if fields.get("weights") is None:
        fields.pop("weight_type", None)
    try:
        return ModelSpec(estimator=estimator, **fields)
    except ValidationError as exc:
        # Convenience functions fail like every other analysis error; building
        # a ModelSpec directly keeps pydantic's ValidationError.
        problems = "; ".join(str(error.get("msg", "")).removeprefix("Value error, ")
                             for error in exc.errors())
        raise AnalysisError("invalid_spec", problems or str(exc)) from exc


def kernel_call(function, *args: Any, **kwargs: Any) -> Any:
    """Run a tensor kernel, translating its failures into ``AnalysisError``."""
    try:
        return function(*args, **kwargs)
    except KernelError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
    except (FloatingPointError, OverflowError) as exc:
        raise AnalysisError("numerical_failure", "The numerical solver could not produce a valid model fit.") from exc


class ModelFrame:
    """The estimation sample of one specification.

    ``original`` holds the model-input columns of every supplied row (index
    reset). ``positions`` are the 0-based original rows in the estimation
    sample, in estimation order. ``allow_missing`` names columns in which a
    missing value is meaningful to the estimator (for example the outcome of
    unselected observations in a selection model, or an open interval bound)
    and therefore does not exclude the row.
    """

    def __init__(self, spec: ModelSpec, data: Any, *, allow_missing: Sequence[str] = (),
                 extra_columns: Sequence[str] = ()):
        from openecon.dataset import Dataset

        if not isinstance(spec, ModelSpec):
            raise AnalysisError("invalid_spec", "Spec must be a validated ModelSpec instance.")
        if isinstance(data, Dataset):
            raise AnalysisError("streaming_unsupported", f"{spec.estimator} needs an in-memory table; "
                                "this estimator has no Dataset execution route. OLS, logit and probit have separate replayable-dataset implementations; no automatic collection is performed.")
        data = _coerce_frame(data)
        if data.columns.has_duplicates:
            raise AnalysisError("duplicate_columns", "Data must have unique column names.")
        if not len(data):
            raise AnalysisError("empty_data", "The dataset contains no observations.")
        self.spec = spec
        self.info = registry.get(spec.estimator)
        columns = list(dict.fromkeys([*registry.spec_columns(spec), *extra_columns]))
        absent = [name for name in columns if name not in data.columns]
        if absent:
            raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
        self.resource_budget_bytes = workspace_budget_bytes()
        self.resource_plans: list[dict[str, Any]] = []
        # This happens before selecting/resetting the frame or constructing the
        # missing-value matrix and positional indices. Deep column sizes count
        # object payload conservatively; they do not measure allocator/RSS use.
        selected_bytes = sum(int(data[name].memory_usage(index=False, deep=True)) for name in columns)
        self.resource_input_bytes = 2 * selected_bytes + len(data) * (len(columns) + 25)
        self.workspace_plan("model input selection", {})
        self.original = data.loc[:, columns].reset_index(drop=True)
        self.warnings: list[str] = []
        self.notes: dict[str, Any] = {}
        self._sample: pd.DataFrame | None = None
        self._sorted_by: list[str] | None = None
        self._permuted = False
        required = [name for name in columns if name not in set(allow_missing)]
        missing = self.original[required].isna().any(axis=1)
        missing_count = int(missing.sum())
        if missing_count and spec.missing == "raise":
            raise AnalysisError(
                "missing_values",
                f"{missing_count} observation(s) contain missing model inputs. Choose missing='drop' explicitly to exclude them.",
            )
        self._rows = torch.as_tensor((~missing).to_numpy().nonzero()[0], dtype=torch.int64)
        self.dropped_missing = missing_count
        if missing_count:
            self.warnings.append(f"Excluded {missing_count} observation(s) with missing model inputs.")
        self._require_rows()
        if spec.weights is not None:
            self._screen_weights()

    # ---- sample bookkeeping -------------------------------------------------

    def _require_rows(self) -> None:
        if not len(self._rows):
            raise AnalysisError("empty_sample", "No complete observations remain for estimation.")

    def _screen_weights(self) -> None:
        weights = self.numeric(self.spec.weights)
        if self.spec.weight_type != "iweight" and bool((weights < 0).any()):
            raise AnalysisError("negative_weights", f"{self.spec.weight_type}s must be nonnegative.")
        if self.spec.weight_type == "fweight" and bool((weights != weights.round()).any()):
            raise AnalysisError("noninteger_frequency_weights", "Frequency weights must be integers.")
        zero = weights == 0
        if bool(zero.any()):
            # Stata excludes zero-weight observations from the estimation sample.
            self.restrict(~zero, f"Excluded {int(zero.sum())} observation(s) with zero weight.")

    @property
    def n(self) -> int:
        return len(self._rows)

    @property
    def positions(self) -> list[int]:
        return self._rows.tolist()

    @property
    def sample(self) -> pd.DataFrame:
        if self._sample is None:
            if len(self._rows) == len(self.original) and not self._permuted:
                self._sample = self.original
            else:
                self._sample = self.original.iloc[self._rows.numpy()].reset_index(drop=True)
        return self._sample

    def restrict(self, keep: Any, reason: str | None = None) -> None:
        """Keep only rows of the current sample where ``keep`` is true."""
        mask = torch.as_tensor(keep.to_numpy(copy=True) if hasattr(keep, "to_numpy") else keep, dtype=torch.bool)
        if mask.ndim != 1 or len(mask) != self.n:
            raise AnalysisError("invalid_sample_filter", "A sample filter needs one flag per estimation row.")
        if bool(mask.all()):
            return
        self._rows = self._rows[mask]
        self._sample = None
        if reason:
            self.warnings.append(reason)
        self._require_rows()

    def reorder(self, order: Any, by: Sequence[str]) -> None:
        """Permute the current sample; ``by`` documents the new row order."""
        order = torch.as_tensor(order.to_numpy(copy=True) if hasattr(order, "to_numpy") else order, dtype=torch.int64)
        if len(order) != self.n or len(torch.unique(order)) != self.n:
            raise AnalysisError("invalid_sample_order", "A sample order must be a permutation of the estimation rows.")
        self._sorted_by = list(by)
        if bool((order == torch.arange(self.n)).all()):
            return
        self._rows = self._rows[order]
        self._sample = None
        self._permuted = True

    def sort_panel(self) -> None:
        """Stable sort by panel then time (whichever are declared), as xtset/tsset do.

        A repeated time value within a panel (or within the single series) is
        an error, mirroring Stata's "repeated time values" check.
        """
        keys = [name for name in (self.spec.panel, self.spec.time) if name is not None]
        if not keys:
            raise AnalysisError("invalid_spec", "Sorting needs a panel or time column.")
        sample = self.sample
        if self.spec.time is not None and bool(sample.duplicated(keys).any()):
            raise AnalysisError("repeated_time_values", "Time values are repeated within a panel"
                                if self.spec.panel else "Time values are repeated.")
        table = pd.DataFrame({name: self._sortable(sample[name]) for name in keys})
        self.reorder(table.sort_values(keys, kind="stable").index.to_numpy(), keys)

    @staticmethod
    def _sortable(series: pd.Series) -> pd.Series:
        if is_numeric_dtype(series.dtype) or is_datetime64_any_dtype(series.dtype):
            return series.reset_index(drop=True)
        # Mixed or text labels: order by first appearance, which is deterministic.
        return pd.Series(pd.factorize(series, sort=False)[0])

    # ---- columns ---------------------------------------------------------------

    def series(self, name: str) -> pd.Series:
        return self.sample[name]

    def numeric(self, name: str, *, allow_missing: bool = False) -> Tensor:
        """A finite float64 column of the current sample.

        ``allow_missing=True`` is for columns declared in ``allow_missing``:
        missing entries come back as NaN (test with ``torch.isnan``); every
        other entry must still be finite.
        """
        series = self.sample[name]
        if not allow_missing:
            return _numeric(series, name)
        present = series.notna()
        values = torch.full((len(series),), float("nan"), dtype=torch.float64)
        if bool(present.any()):
            values[torch.as_tensor(present.to_numpy(copy=True))] = _numeric(series[present], name)
        return values

    def matrix(self, names: Sequence[str]) -> Tensor:
        """Numeric columns stacked as an [n, len(names)] tensor."""
        self.workspace_plan("numeric column matrix", {
            "numeric_columns_and_stacked_matrix": tensor_bytes((self.n, len(names)), itemsize=24),
        })
        if not names:
            return torch.empty((self.n, 0), dtype=torch.float64)
        return torch.stack([self.numeric(name) for name in names], dim=1)

    def codes(self, names: str | Sequence[str]) -> tuple[Tensor, int]:
        """Dense int64 group codes (0..G-1, by first appearance) and G.

        Several names give the codes of their interaction (distinct value
        combinations), which is what multiway-cluster intersections need.
        """
        names = [names] if isinstance(names, str) else list(names)
        sample = self.sample
        for name in names:
            if is_numeric_dtype(sample[name].dtype):
                _numeric(sample[name], name)
        try:
            if len(names) == 1:
                values, unique = pd.factorize(sample[names[0]], sort=False)
                count = len(unique)
            else:
                values = sample.groupby(names, sort=False, observed=True).ngroup().to_numpy()
                count = int(values.max()) + 1 if len(values) else 0
        except (TypeError, ValueError) as exc:
            raise AnalysisError("invalid_groups", f"Group labels in {', '.join(names)} must be scalar values.") from exc
        if (values < 0).any():
            raise AnalysisError("invalid_groups", f"Group labels in {', '.join(names)} must not be missing.")
        return torch.from_numpy(values.astype("int64")), count

    def levels(self, name: str) -> list[Any]:
        """Group labels of one column in the order of ``codes(name)``."""
        return [_json_scalar(value) for value in pd.factorize(self.sample[name], sort=False)[1]]

    def cluster_dimensions(self) -> list[tuple[Tensor, int]]:
        """One ``(codes, G)`` pair per declared cluster column."""
        dimensions = [self.codes(name) for name in registry.cluster_columns(self.spec)]
        for (_, count), name in zip(dimensions, registry.cluster_columns(self.spec), strict=True):
            if count < 2:
                raise AnalysisError("insufficient_clusters", "Cluster covariance requires at least two observed clusters.")
            if count < 30:
                self.warn(f"Only {count} clusters in '{name}'; cluster-robust inference may be unreliable.")
        return dimensions

    def weights(self) -> Tensor | None:
        return None if self.spec.weights is None else self.numeric(self.spec.weights)

    def option(self, name: str) -> Any:
        """An option's value, or the registry default when it was not given."""
        if name in self.spec.options:
            return self.spec.options[name]
        option = self.info.option(name)
        if option is None:
            raise KeyError(name)
        return option.default

    def role(self, name: str) -> list[str]:
        return registry.role_columns(self.spec, name)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def time_index(self, delta: int = 1) -> Tensor:
        """Integer period index of the time column (Stata's tsset timevar / delta).

        Integer-valued numeric time is used as is, so gaps are visible to lag
        and difference operators. Datetime columns are ranked by their distinct
        sorted values (consecutive periods; gaps cannot be inferred) with a
        recorded warning.
        """
        if self.spec.time is None:
            raise AnalysisError("invalid_spec", "This operation needs a time column.")
        series = self.sample[self.spec.time]
        if is_datetime64_any_dtype(series.dtype):
            self.warn(f"Datetime column '{self.spec.time}' is treated as consecutive periods in sorted order; "
                      "gaps between dates are not detected.")
            return torch.as_tensor(series.rank(method="dense").to_numpy(dtype="int64") - 1, dtype=torch.int64)
        if is_integer_dtype(series.dtype) and not is_bool_dtype(series.dtype):
            # Read integer periods exactly: a float64 round trip loses spacing beyond 2^53.
            index = torch.from_numpy(series.to_numpy(dtype="int64", copy=True))
        else:
            values = _numeric(series, self.spec.time)
            if bool((values != values.round()).any()):
                raise AnalysisError("invalid_time", f"Time column '{self.spec.time}' must contain integer periods or datetimes.")
            index = values.to(torch.int64)
        if delta != 1:
            if bool((index % delta != 0).any()) and bool(((index - index.min()) % delta != 0).any()):
                raise AnalysisError("invalid_time", f"Time values are not multiples of delta={delta}.")
            index = (index - index.min()) // delta
        return index

    # ---- design ----------------------------------------------------------------

    def workspace_plan(self, operation: str, buffers: dict[str, int]):
        """Check family buffers together with model-owned sample/index storage."""
        plan = plan_workspace(operation, {"model_input_and_sample_indices": self.resource_input_bytes,
                                          **buffers}, budget_bytes=self.resource_budget_bytes)
        record = plan.record()
        if record not in self.resource_plans:
            self.resource_plans.append(record)
        return plan

    def design_width(self, predictors: Sequence[str] | None = None, *, intercept: bool | None = None,
                     categorical: Sequence[str] | None = None) -> int:
        """Exact encoded width before constructing a design or sample copy."""
        return self._design_metadata(predictors, intercept=intercept, categorical=categorical)[-1]

    def _design_metadata(self, predictors, *, intercept=None, categorical=None):
        spec = self.spec
        predictors = list(spec.predictors if predictors is None else predictors)
        intercept = spec.intercept if intercept is None else intercept
        categorical = [name for name in predictors if name in set(spec.categorical if categorical is None else categorical)]
        level_sets: dict[str, tuple[list[Any], bool]] = {}
        columns = int(intercept) + len(predictors) - len(categorical)
        for name in categorical:
            original = self.original[name]
            if isinstance(original.dtype, pd.CategoricalDtype):
                levels, ordered = list(original.cat.categories), bool(original.cat.ordered)
            else:
                try:
                    levels, ordered = sorted(original.dropna().unique().tolist()), False
                except (TypeError, ValueError) as exc:
                    raise AnalysisError("ambiguous_categories", f"Column '{name}' has mixed or non-scalar category types; use a pandas Categorical with explicit levels.") from exc
            if len(levels) < 2:
                raise AnalysisError("constant_predictor", f"Categorical predictor '{name}' has fewer than two levels.")
            level_sets[name] = (levels, ordered)
            columns += len(levels) - 1
        return predictors, bool(intercept), level_sets, columns

    def design(self, predictors: Sequence[str] | None = None, *, intercept: bool | None = None,
               categorical: Sequence[str] | None = None, prefix: str = "") -> Design:
        """Build a design block with treatment-coded categorical predictors.

        Category levels and the reference level are taken from the original
        column before the missing-value filter (first level omitted), matching
        the core estimators. Constant or collinear columns are NOT rejected
        here: registry estimators drop them Stata-style via ``drop_collinear``.
        ``prefix`` namespaces the term names of auxiliary equations.
        """
        predictors, intercept, level_sets, columns = self._design_metadata(
            predictors, intercept=intercept, categorical=categorical)
        n = self.n
        self.workspace_plan("dense encoded design", {
            "encoded_columns_and_reference_categories": tensor_bytes((n, columns + len(level_sets))),
            "design_and_qr_copies": tensor_bytes((n, columns), itemsize=40),
            "parameter_factors": tensor_bytes((columns, columns), itemsize=64),
            "encoding_indices": tensor_bytes((n,), itemsize=24),
        })
        sample = self.sample
        pieces: list[Tensor] = []
        terms: list[str] = []
        categories: dict[str, Any] = {}
        if intercept:
            pieces.append(torch.ones((n, 1), dtype=torch.float64))
            terms.append(prefix + "Intercept")
        for name in predictors:
            if name in level_sets:
                levels, ordered = level_sets[name]
                safe_levels = [_json_scalar(value) for value in levels]
                codes = pd.Categorical(sample[name], categories=levels, ordered=ordered).codes
                block = torch.zeros((n, len(levels)), dtype=torch.float64)
                index = torch.from_numpy(codes.astype("int64"))
                if bool((index < 0).any()):
                    raise AnalysisError("missing_values", f"Categorical predictor '{name}' contains missing values.")
                block[torch.arange(n), index] = 1.0
                pieces.append(block[:, 1:])
                terms.extend(f"{prefix}{name}[{value}]" for value in safe_levels[1:])
                categories[name] = {
                    "levels": safe_levels, "reference": safe_levels[0], "ordered": ordered,
                    "coding": "treatment_drop_first", "levels_source": "original_model_inputs_before_missing_filter",
                }
            else:
                pieces.append(self.numeric(name)[:, None])
                terms.append(prefix + name)
        if len(terms) != len(set(terms)):
            raise AnalysisError("duplicate_terms", "Predictor names collide with generated design terms. Rename the affected columns.")
        x = torch.cat(pieces, dim=1) if pieces else torch.empty((n, 0), dtype=torch.float64)
        return Design(x, terms, categories, bool(intercept))

    def drop_collinear(self, design: Design, weights: Tensor | None = None, *, tol: float | None = None) -> Design:
        """Omit collinear design columns left to right, as Stata does, and record them.

        With a constant in the design the screen runs on mean-deviated columns
        (Stata sweeps the constant first), so a polynomial in calendar years or
        a regressor with a large offset is not mistaken for the constant.
        ``tol`` is the relative sum-of-squares threshold of
        ``engines.linalg.collinear_columns`` (its default when ``None``).
        """
        from openecon.engines.linalg import collinear_columns

        if design.x.shape[1] == 0:
            return design
        x = design.x
        if design.intercept and x.shape[1] > 1:
            x = x.clone()
            if weights is None:
                x[:, 1:] -= x[:, 1:].mean(dim=0)
            else:
                x[:, 1:] -= (weights[:, None] * x[:, 1:]).sum(dim=0) / weights.sum()
        options = {} if tol is None else {"tol": tol}
        kept, omitted = kernel_call(collinear_columns, x, weights, **options)
        return self._omit(design, kept, omitted, "collinearity")

    def drop_absorbed(self, design: Design, demeaned: Tensor, weights: Tensor | None = None, *,
                      tol: float = 1e-13) -> tuple[Design, Tensor]:
        """Omit regressors absorbed by fixed effects and screen the rest.

        ``demeaned`` holds the design after partialling out the fixed effects.
        A column whose remaining weighted sum of squares is at most ``tol``
        times its mean-deviated original sum of squares has no within
        variation (it is collinear with the absorbed effects) and is omitted;
        the survivors then pass through the ordinary left-to-right
        collinearity screen on the demeaned block. The default ``tol`` is a
        relative length of about 3e-7, far above the residual that converged
        demeaning leaves on a truly absorbed column and far below the within
        variation of a regressor with a large offset or between variation.
        Returns the reduced design and the matching demeaned columns.
        """
        if demeaned.shape != design.x.shape:
            raise AnalysisError("invalid_result", "The partialled-out design must match the original one.")
        if design.x.shape[1] == 0:
            return design, demeaned
        if weights is None:
            centered = design.x - design.x.mean(dim=0)
            before, after = centered.square().sum(dim=0), demeaned.square().sum(dim=0)
        else:
            centered = design.x - (weights[:, None] * design.x).sum(dim=0) / weights.sum()
            before = (weights[:, None] * centered.square()).sum(dim=0)
            after = (weights[:, None] * demeaned.square()).sum(dim=0)
        absorbed = (after <= tol * before).nonzero().flatten().tolist()
        kept = [i for i in range(design.x.shape[1]) if i not in set(absorbed)]
        design = self._omit(design, kept, absorbed, "collinearity with the absorbed fixed effects")
        demeaned = demeaned[:, kept]
        from openecon.engines.linalg import collinear_columns

        kept, omitted = kernel_call(collinear_columns, demeaned, weights)
        if omitted:
            design = self._omit(design, kept, omitted, "collinearity")
            demeaned = demeaned[:, kept]
        return design, demeaned

    def _omit(self, design: Design, kept: Sequence[int], omitted: Sequence[int], reason: str) -> Design:
        if not omitted:
            return design
        names = [design.terms[i] for i in omitted]
        self.notes.setdefault("omitted_terms", []).extend(names)
        self.warn(f"Omitted because of {reason}: {', '.join(names)}.")
        return design.select(kept)


# ---- covariance and tests shared by likelihood estimators ----------------------


def _note_psd(frame: ModelFrame, adjusted: bool) -> None:
    if adjusted:
        frame.warn("The multiway cluster covariance was not positive semidefinite; its negative "
                   "eigenvalues were set to zero (Cameron, Gelbach and Miller 2011).")


def _cluster_setup(frame: ModelFrame, clusters: Sequence[tuple[Tensor, int]] | None,
                   cluster_names: Sequence[str] | None) -> tuple[list[tuple[Tensor, int]], list[str]]:
    """Declared cluster dimensions, or an estimator's override (e.g. the panel variable)."""
    if clusters is None:
        return frame.cluster_dimensions(), registry.cluster_columns(frame.spec)
    names = list(cluster_names or [])
    if len(names) != len(clusters):
        raise AnalysisError("invalid_result", "One name is needed per cluster dimension.")
    if any(count < 2 for _, count in clusters):
        raise AnalysisError("insufficient_clusters", "Cluster covariance requires at least two observed clusters.")
    return list(clusters), names


def ml_covariance(frame: ModelFrame, *, hessian: Tensor, scores: Tensor | None = None,
                  nobs: int | None = None, frequency: Tensor | None = None, kind: str | None = None,
                  clusters: Sequence[tuple[Tensor, int]] | None = None,
                  cluster_names: Sequence[str] | None = None) -> tuple[Tensor, dict[str, Any]]:
    """Covariance of a maximum-likelihood estimate under Stata's -ml- conventions.

    ``hessian`` is the Hessian of the (weighted) log likelihood at the maximum;
    ``scores`` are per-observation score rows, already multiplied by any
    sampling/analytic weight. With frequency weights pass unweighted scores and
    ``frequency``: each row then counts ``f_i`` times and ``N = sum f_i``.
    The outer product of gradients estimates an information matrix that is
    linear in the weights, so for ``opg`` with analytic or importance weights
    pass the unweighted scores with ``frequency=w`` as well (``(sum w s s')^-1``).

    - ``nonrobust``: ``(-H)^-1`` (observed information, vce(oim)).
    - ``opg``: ``(S'S)^-1`` (vce(opg)).
    - ``robust``: ``N/(N-1) * (-H)^-1 S'S (-H)^-1`` (vce(robust)).
    - ``cluster``: ``G/(G-1) * (-H)^-1 (sum_g s_g s_g') (-H)^-1``; several
      cluster columns use Cameron-Gelbach-Miller inclusion-exclusion with the
      smallest dimension's ``G/(G-1)``.

    ``kind`` overrides ``spec.covariance`` and ``clusters``/``cluster_names``
    override the declared cluster columns, for estimators whose named
    covariance maps onto another one (for example panel estimators whose
    ``robust`` means clustering on the panel variable).
    """
    from openecon.engines import covariance as cov
    from openecon.engines.optimize import information_inverse

    kind = kind or frame.spec.covariance
    n = int(nobs if nobs is not None else frame.n)
    info: dict[str, Any] = {"covariance": kind}
    if kind not in {"nonrobust", "opg", "robust", "cluster"}:
        raise AnalysisError("unsupported_covariance", f"{kind} covariance is not a likelihood covariance estimator.")
    if kind != "nonrobust" and scores is None:
        raise AnalysisError("unsupported_covariance", f"{kind} covariance needs per-observation scores.")
    bread = None if kind == "opg" else kernel_call(information_inverse, -hessian)
    if kind == "nonrobust":
        info["correction"] = "observed information"
        return bread, info
    weighted = scores if frequency is None else scores * frequency[:, None]
    if kind == "opg":
        info["correction"] = "outer product of gradients"
        return kernel_call(information_inverse, weighted.T @ scores), info
    if kind == "robust":
        factor = n / (n - 1)
        info.update({"correction": "Huber-White sandwich: N/(N-1)", "small_sample_correction": factor})
        return kernel_call(cov.sandwich, bread, (weighted.T @ scores) * factor), info
    dimensions, columns = _cluster_setup(frame, clusters, cluster_names)
    if len(dimensions) == 1:
        codes, count = dimensions[0]
        factor = count / (count - 1)
        meat = kernel_call(cov.meat_cluster, weighted, codes, count) * factor
        info.update({"correction": "cluster sandwich: G/(G-1)", "cluster_count": count})
    else:
        multiway = kernel_call(cov.meat_multiway, weighted, dimensions, adjust="min", force_psd=True)
        meat, count = multiway.meat, multiway.min_groups
        factor = count / (count - 1)
        _note_psd(frame, multiway.psd_adjusted)
        info.update({"correction": "multiway cluster sandwich (inclusion-exclusion): G_min/(G_min-1)",
                     "cluster_count": count, "cluster_counts": multiway.group_counts,
                     "psd_adjusted": multiway.psd_adjusted})
    info.update({"small_sample_correction": factor, "cluster_columns": columns, "cluster_column": columns[0],
                 "cluster_df": count - 1})
    return kernel_call(cov.sandwich, bread, meat), info


def linear_covariance(
    frame: ModelFrame, *, x: Tensor, resid: Tensor, bread: Tensor, n: int, k: int, df_resid: float,
    weights: Tensor | None = None, score_x: Tensor | None = None, small: bool = True,
    group_factor: bool = True, ssr: float | None = None, kind: str | None = None,
    clusters: Sequence[tuple[Tensor, int]] | None = None, cluster_names: Sequence[str] | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Covariance of a linear least-squares or IV estimator, by ``spec.covariance``.

    ``bread`` is ``(X'WX)^-1`` (for IV: ``(Xhat'W Xhat)^-1``); the moment rows
    are ``score_x_i * w_i * u_i`` with ``score_x`` defaulting to ``x`` (pass the
    instrument-projected regressors for IV). ``n`` is the number of
    observations as the estimator counts them (the sum of frequency weights),
    ``k`` the number of parameters entering small-sample factors and
    ``df_resid`` the divisor of the classical error variance.

    ``small`` applies the degrees-of-freedom factors of Stata's regress;
    ``group_factor`` applies ``G/(G-1)`` to cluster estimators:

    - ``nonrobust``: ``s2 * bread`` with ``s2 = sum w u^2 / df_resid`` (``/ n``
      when ``small`` is false).
    - ``HC0``/``HC1``/``HC2``/``HC3``: White sandwich; HC1 multiplies by
      ``n/(n-k)`` when ``small``; HC2/HC3 rescale residuals by leverage.
    - ``cluster``: ``G/(G-1) * (n-1)/(n-k)``; with several cluster columns the
      Cameron-Gelbach-Miller inclusion-exclusion meat with the smallest
      dimension's ``G/(G-1)``. Inference uses ``G_min - 1`` degrees of freedom.
    - ``hac``: Newey-West with ``options['lags']`` and ``options['kernel']``
      (Bartlett by default), within panels when a panel column is declared,
      times ``n/(n-k)`` when ``small`` (Stata's newey).

    Frequency weights replicate observations: White meats weight squared
    scores by ``f_i`` while cluster sums weight scores by ``f_i``. Other
    weight types multiply scores by ``w_i``.

    ``kind`` overrides ``spec.covariance`` (``robust`` is accepted as HC1) and
    ``clusters``/``cluster_names`` override the declared cluster columns, e.g.
    for panel estimators whose ``robust`` means clustering on the panel.

    Returns the covariance and an inference record (correction text, factor,
    cluster counts, ``df_inference``) to pass on to ``build_result``.
    """
    from openecon.engines import covariance as cov

    spec = frame.spec
    kind = kind or spec.covariance
    info: dict[str, Any] = {"covariance": kind, "df_inference": df_resid}
    kind = "HC1" if kind == "robust" else kind
    score_x = x if score_x is None else score_x
    frequency = weights is not None and spec.weight_type == "fweight"
    if kind == "nonrobust":
        if ssr is None:
            ssr = float((resid.square() * weights).sum() if weights is not None else resid.square().sum())
        divisor = df_resid if small else n
        info["correction"] = f"classical: SSR/{'(N-K)' if small else 'N'}"
        return bread * (ssr / divisor), info
    if kind in {"HC0", "HC1", "HC2", "HC3"}:
        leverage = None
        if kind in {"HC2", "HC3"}:
            leverage = ((x @ bread) * x).sum(dim=1)
            if weights is not None and not frequency:
                leverage = leverage * weights
        adjusted = kernel_call(cov.hc_residuals, resid, leverage, kind)
        if weights is not None:
            adjusted = adjusted * (weights.sqrt() if frequency else weights)
        factor = n / (n - k) if kind == "HC1" and small else 1.0
        meat = kernel_call(cov.meat_white, score_x * adjusted[:, None])
        info.update({"correction": {"HC0": "HC0", "HC1": "HC1: N/(N-K)" if small else "HC1 without N/(N-K)",
                                    "HC2": "HC2: residual / sqrt(1-h)", "HC3": "HC3: residual / (1-h)"}[kind],
                     "small_sample_correction": factor})
        return kernel_call(cov.sandwich, bread, meat * factor), info
    scores = score_x * (resid if weights is None else resid * weights)[:, None]
    if kind == "hac" and frequency:
        # A frequency weight replicates the observation's own squared score; the
        # lag-0 term then equals the White meat (cross-period products are not
        # defined for replicated rows and follow the same sqrt(f) scaling).
        scores = score_x * (resid * weights.sqrt())[:, None]
    if kind == "cluster":
        dimensions, columns = _cluster_setup(frame, clusters, cluster_names)
        dof = (n - 1) / (n - k) if small else 1.0
        if len(dimensions) == 1:
            codes, count = dimensions[0]
            groups = count / (count - 1) if group_factor else 1.0
            meat = kernel_call(cov.meat_cluster, scores, codes, count) * groups
            info["cluster_count"] = count
        else:
            multiway = kernel_call(cov.meat_multiway, scores, dimensions,
                                   adjust="min" if group_factor else "none", force_psd=True)
            meat, count = multiway.meat, multiway.min_groups
            groups = count / (count - 1) if group_factor else 1.0
            _note_psd(frame, multiway.psd_adjusted)
            info.update({"cluster_count": count, "cluster_counts": multiway.group_counts,
                         "psd_adjusted": multiway.psd_adjusted})
        text = "CR1: " + " * ".join(part for part, on in (("G/(G-1)", group_factor), ("(N-1)/(N-K)", small)) if on)
        info.update({"correction": text if group_factor or small else "cluster sandwich without finite-sample factors",
                     "small_sample_correction": groups * dof, "cluster_columns": columns,
                     "cluster_column": columns[0], "cluster_df": count - 1, "df_inference": count - 1})
        return kernel_call(cov.sandwich, bread, meat * dof), info
    if kind == "hac":
        lags = spec.options.get("lags")
        if lags is None:
            raise AnalysisError("invalid_spec", "HAC covariance requires the option lags (use 0 for White).")
        kernel = spec.options.get("kernel", "bartlett")
        time = frame.time_index() if spec.time is not None else None
        panel = frame.codes(spec.panel)[0] if spec.panel is not None else None
        if panel is not None and time is None:
            raise AnalysisError("invalid_spec", "HAC covariance on panel data needs a time column.")
        meat = kernel_call(cov.meat_hac, scores, int(lags), kernel, time=time, panel=panel)
        factor = n / (n - k) if small else 1.0
        info.update({"correction": f"Newey-West HAC ({kernel}, {lags} lags)" + (": N/(N-K)" if small else ""),
                     "small_sample_correction": factor, "lags": lags, "kernel": kernel})
        return kernel_call(cov.sandwich, bread, meat * factor), info
    raise AnalysisError("unsupported_covariance", f"{kind} covariance is not available for this linear estimator.")


def wald_test(params: Tensor, covariance: Tensor, indices: Sequence[int], *,
              df_resid: float | None = None, label: str | None = None) -> dict[str, Any]:
    """Joint Wald test that the selected coefficients are zero.

    Returns chi2(q) for normal-theory inference, or F(q, df_resid) = W/q when a
    residual degrees of freedom is supplied (Stata's ``test`` after a t-based
    estimator). A singular covariance block reduces q to its rank.
    """
    from openecon.engines.distributions import chi2_sf, f_sf
    from openecon.engines.linalg import wald_statistic

    indices = list(indices)
    if not indices:
        return {"statistic": None, "df": 0, "p_value": None, "distribution": "chi2", "label": label}
    block = covariance[indices][:, indices]
    try:
        statistic, rank = kernel_call(wald_statistic, params[indices], block)
    except AnalysisError as exc:
        if exc.code != "singular_covariance":
            raise
        return {"statistic": None, "df": 0, "p_value": None, "distribution": "chi2", "label": label,
                "note": "the covariance of the tested coefficients is singular"}
    if df_resid is None:
        result = {"statistic": statistic, "df": rank, "p_value": chi2_sf(statistic, rank), "distribution": "chi2"}
    else:
        result = {"statistic": statistic / rank, "df": rank, "df2": df_resid,
                  "p_value": f_sf(statistic / rank, rank, df_resid), "distribution": "F"}
    if label:
        result["label"] = label
    return result


def lr_test(log_likelihood: float, restricted_log_likelihood: float, df: int, *, label: str | None = None) -> dict[str, Any]:
    """Likelihood-ratio chi2 test: ``2 (ll - ll_restricted)`` with ``df`` restrictions."""
    from openecon.engines.distributions import chi2_sf

    statistic = max(0.0, 2 * (log_likelihood - restricted_log_likelihood))
    result = {"statistic": statistic, "df": df, "p_value": chi2_sf(statistic, df) if df > 0 else None,
              "distribution": "chi2"}
    if label:
        result["label"] = label
    return result


def information_criteria(log_likelihood: float, k: int, n: int) -> dict[str, float]:
    """Stata's estat ic: AIC = -2 ll + 2k and BIC = -2 ll + k ln(N)."""
    return {"log_likelihood": log_likelihood, "aic": -2 * log_likelihood + 2 * k,
            "bic": -2 * log_likelihood + math.log(n) * k}


def forecast(result: ResultBundle, steps: int, **options: Any) -> Any:
    """Out-of-sample forecasts from a fitted time-series model.

    Dispatches on ``result.spec.estimator`` to the forecast function its
    family registered (``FORECAST`` in the family manifest), so ``oe.forecast``
    works for every model that supports forecasting. ``steps`` is the horizon;
    further keyword options (``data``, ``exog``, ``alpha`` ...) are those of
    the family's function.
    """
    from importlib import import_module

    handlers = registry.forecasters()
    if not isinstance(result, ResultBundle):
        raise AnalysisError("invalid_result", "forecast needs the result returned by a time-series "
                            "estimator such as oe.arima.")
    name = result.spec.estimator
    if name not in handlers:
        available = ", ".join(sorted(handlers)) or "none"
        raise AnalysisError("unsupported_forecast", f"Forecasting is not available for {name!r} results "
                            f"(available for: {available}).")
    module, function = handlers[name]
    return getattr(import_module(module), function)(result, steps, **options)


# ---- tabular results of test procedures -------------------------------------------


def table(data: Any, *, columns: Sequence[str] | None = None, index: Sequence[Any] | None = None,
          **attrs: Any) -> pd.DataFrame:
    """Build a result table (``openecon.frame.DataFrame``) with scalar results in ``attrs``.

    Test procedures that are not model fits (t tests, ANOVA tables, unit-root
    tests, cross-tabulations) return tables: the console renders them as tables
    and they export to LaTeX. ``attrs`` values are made JSON-safe.
    """
    from openecon.frame import DataFrame

    frame = DataFrame(data, columns=list(columns) if columns is not None else None,
                      index=list(index) if index is not None else None)
    frame.attrs.update(_json_safe(attrs))
    return frame


class TableSet(dict):
    """Named result tables of a multi-table procedure.

    A mapping from table name to ``openecon.frame.DataFrame`` (for example
    ``{"anova": ..., "descriptives": ..., "posthoc": ...}``). ``attrs`` carries
    the scalar results (statistics, p-values, settings). ``str()`` renders
    every table as text and ``to_latex()`` concatenates their LaTeX source.
    """

    def __init__(self, tables: dict[str, pd.DataFrame] | None = None, *, title: str | None = None,
                 **attrs: Any):
        super().__init__(tables or {})
        self.title = title
        self.attrs: dict[str, Any] = _json_safe(attrs)

    def __str__(self) -> str:
        parts = [self.title] if self.title else []
        for name, frame in self.items():
            parts.append(f"[{name}]\n{frame.to_string()}")
        notes = [f"{key}: {value}" for key, value in self.attrs.items()
                 if isinstance(value, (str, int, float, bool))]
        # Free-text notes (small expected counts, simulation settings) are shown line by line.
        listed = self.attrs.get("notes")
        if isinstance(listed, list):
            notes.extend(f"Note: {item}" for item in listed if isinstance(item, str))
        if notes:
            parts.append("\n".join(notes))
        return "\n\n".join(parts)

    __repr__ = __str__

    def to_latex(self, **options: Any) -> str:
        return "\n\n".join(str(frame.to_latex(caption=name, **options)) for name, frame in self.items())


# ---- result assembly --------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Recursively convert tensors and scalars to finite JSON values (non-finite -> None)."""
    if isinstance(value, Tensor):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item) and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    return str(value)


def _versions() -> dict[str, str]:
    try:
        package_version = version("openecon")
    except PackageNotFoundError:
        package_version = "development"
    return {"openecon": package_version, "python": platform.python_version(),
            "pandas": pd.__version__, "torch": torch.__version__}


def build_result(
    frame: ModelFrame, *, terms: Sequence[str], params: Tensor, covariance: Tensor,
    title: str | None = None, equations: Sequence[str | None] | None = None,
    use_t: bool | None = None, df_inference: float | None = None, df_resid: float | None = None,
    metrics: dict[str, Any] | None = None, fitted: Tensor | None = None, observed: Tensor | None = None,
    solver: str = "", solver_diagnostics: dict[str, Any] | None = None, optimizer: dict[str, Any] | None = None,
    inference: dict[str, Any] | None = None, provenance: dict[str, Any] | None = None,
    tests: dict[str, Any] | None = None, extra: dict[str, Any] | None = None,
    categories: dict[str, Any] | None = None, nobs: int | None = None,
    warnings: Sequence[str] = (),
) -> ResultBundle:
    """Assemble the persistable result of a registry estimator.

    ``params`` [k] and ``covariance`` [k, k] are in reporting units and in the
    order of ``terms`` (unique names; ``equations`` optionally groups them).
    Coefficient tests use Student t with ``df_inference`` when ``use_t`` is
    true (default: the estimator's registry setting) and the standard normal
    otherwise. ``fitted`` (aligned with the frame's current sample) produces
    the bounded chart sample of observed/fitted/residual rows; ``observed``
    defaults to the outcome column. Metrics, tests and extra output are made
    JSON-safe: non-finite numbers become ``None``.
    """
    spec, info = frame.spec, frame.info
    terms = list(terms)
    k = len(terms)
    n = frame.n
    reported_n = int(nobs if nobs is not None else n)
    params = torch.as_tensor(params, dtype=torch.float64).reshape(-1)
    covariance = torch.as_tensor(covariance, dtype=torch.float64)
    if len(params) != k or covariance.shape != (k, k):
        raise AnalysisError("invalid_result", "Parameter, covariance and term dimensions do not agree.")
    if len(set(terms)) != k:
        raise AnalysisError("duplicate_terms", "Result terms must be unique. Rename the affected columns.")
    if equations is not None and len(equations) != k:
        raise AnalysisError("invalid_result", "One equation label is needed per term.")
    covariance = (covariance + covariance.T) / 2
    if not bool(torch.isfinite(params).all()) or not bool(torch.isfinite(covariance).all()):
        raise AnalysisError("non_finite_result", "The solver produced non-finite coefficients or covariance.")
    variances = covariance.diagonal()
    if bool((variances <= 0).any()):
        raise AnalysisError("invalid_covariance", "The fitted covariance does not provide strictly positive standard errors.")
    use_t = (info.inference == "t") if use_t is None else use_t
    if use_t and df_inference is None:
        raise AnalysisError("invalid_result", "Student-t inference needs its degrees of freedom.")
    reference_df = df_inference if use_t else None
    standard_errors = variances.sqrt()
    statistics = params / standard_errors
    try:
        p_values = two_sided_p_values(statistics, reference_df)
        critical = critical_value(spec.alpha, reference_df)
    except KernelError as exc:
        raise AnalysisError(exc.code, str(exc)) from exc
    low, high = params - critical * standard_errors, params + critical * standard_errors
    if not all(bool(torch.isfinite(v).all()) for v in (statistics, p_values, low, high)):
        raise AnalysisError("non_finite_result", "The fitted model did not provide finite statistical inference.")
    coefficients = [
        Coefficient(term=term, estimate=float(params[i]), std_error=float(standard_errors[i]),
                    statistic=float(statistics[i]), p_value=float(p_values[i]), ci_low=float(low[i]),
                    ci_high=float(high[i]), equation=None if equations is None else equations[i])
        for i, term in enumerate(terms)
    ]
    positions = frame.positions
    predictions: list[dict[str, int | float]] = []
    if fitted is not None:
        fitted = torch.as_tensor(fitted, dtype=torch.float64).reshape(-1)
        observed = frame.numeric(spec.outcome) if observed is None else torch.as_tensor(observed, dtype=torch.float64).reshape(-1)
        if len(fitted) != n or len(observed) != n:
            raise AnalysisError("invalid_result", "Fitted values must align with the estimation sample.")
        rows = torch.linspace(0, n - 1, steps=min(n, PREDICTION_LIMIT), dtype=torch.float64).to(torch.int64)
        chosen = torch.stack((observed[rows], fitted[rows]), dim=1)
        if not bool(torch.isfinite(chosen).all()):
            raise AnalysisError("non_finite_result", "The solver produced non-finite predictions.")
        predictions = [{"row": positions[i], "observed": o, "fitted": f, "residual": o - f}
                       for i, (o, f) in zip(rows.tolist(), chosen.tolist(), strict=True)]
    input_hasher = _frame_hasher(frame.original)
    data_hash = input_hasher.hexdigest()
    sample_hasher = input_hasher.copy() if frame.sample is frame.original else _frame_hasher(frame.sample)
    sample_hasher.update(_position_bytes(positions))
    clusters = registry.cluster_columns(spec)
    inference_record: dict[str, Any] = {
        "covariance": spec.covariance, "use_t": use_t, "distribution": "t" if use_t else "normal",
        "alpha": spec.alpha, "confidence_level": 1 - spec.alpha, "df_resid": df_resid,
        "df_inference": reference_df, "cluster_count": None, "cluster_df": None,
        "cluster_column": clusters[0] if clusters else None, "small_sample_correction": None,
        "correction": spec.covariance, "intercept": spec.intercept, "n_parameters": k,
        "residual_definition": "observed minus fitted response",
    }
    inference_record.update(inference or {})
    provenance_record: dict[str, Any] = {
        "schema_version": "3", "backend": "openecon.torch", "engine": "openecon", "device": "cpu",
        "estimator": spec.estimator, "family": info.family, "stata_equivalent": list(info.stata),
        "versions": _versions(), "data_hash": data_hash, "sample_hash": sample_hasher.hexdigest(),
        "hash_algorithm": "sha256 over schema and pandas row hashes (pandas version recorded)",
        "hash_scope": "model-input columns in positional row order; index labels excluded",
        "input_columns": list(frame.original.columns), "design_terms": terms,
        "categorical_encoding": categories or {}, "omitted_terms": frame.notes.get("omitted_terms", []),
        "precision": "float64", "sample_position_base": 0,
        "sample_order": "sorted by " + ", ".join(frame._sorted_by) if frame._sorted_by else "input row order",
        "prediction_sample": "evenly spaced estimation-sample observations" if predictions else None,
        "prediction_limit": PREDICTION_LIMIT, "stata_parity_validated": False, "solver": solver,
        "solver_diagnostics": solver_diagnostics or {}, "optimizer": optimizer,
        "resource_plans": frame.resource_plans,
        "weights": None if spec.weights is None else {"column": spec.weights, "type": spec.weight_type},
    }
    provenance_record.update(provenance or {})
    bundle = ResultBundle(
        id=str(uuid4()), created_at=datetime.now(timezone.utc).isoformat(), spec=spec,
        nobs=reported_n, nobs_original=len(frame.original), dropped_rows=len(frame.original) - n,
        coefficients=coefficients, covariance_matrix=covariance.tolist(),
        metrics={name: _safe_metric(value) for name, value in (metrics or {}).items()},
        warnings=[*frame.warnings, *(w for w in warnings if w not in frame.warnings)],
        predictions=predictions, sample_positions=positions,
        provenance=_json_safe(provenance_record), inference=_json_safe(inference_record),
        title=title or info.title, tests=_json_safe(tests or {}), extra=_json_safe(extra or {}),
    )
    from openecon.econometrics.postest.group_state import capture_resident
    return capture_resident(bundle, frame, fitted)
