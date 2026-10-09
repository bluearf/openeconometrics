"""Bounded saved-basis prediction and fixed-query margins, without refitting."""

from __future__ import annotations

import hashlib

import pandas as pd
import torch

from openecon.engines.streaming_ols import _CompensatedSum
from openecon.models import ResultBundle
from openecon.resources import plan_workspace, workspace_budget_bytes
from .common import DT, fail, finite
from .replay import validated_state

METHODS = frozenset({"bspline_regress", "rcs_regress", "fp_regress", "mfp_regress"})


class Query:
    def __init__(self, result, source, *, missing, max_work, extra_columns=()):
        if not isinstance(result, ResultBundle):
            fail("invalid_result", "Supply a saved smoothing ResultBundle.")
        if result.spec.estimator not in METHODS:
            fail(
                "streaming_unsupported",
                "Dataset smoothing evaluation supports saved B/RCS spline and FP/MFP bases only.",
            )
        if missing not in {"raise", "drop"} or type(max_work) is not int or max_work < 1:
            fail("invalid_option", "Use raise/drop missing and positive integer max_work.")
        self.state = validated_state(result)
        self.source, self.missing, self.max_work = source, missing, max_work
        self.columns = list(dict.fromkeys([*self.state["columns"], *extra_columns]))
        if set(self.columns) - set(source.columns):
            fail(
                "missing_columns",
                "Dataset evaluation requires every saved predictor and requested averaging-weight column.",
            )
        self.k = len(self.state["coefficients"])
        if not 1 <= self.k <= 1000:
            fail(
                "invalid_state",
                "Saved transformed basis dimensions exceed the bounded evaluation domain.",
            )
        budget = min(128 * 1024**2, workspace_budget_bytes())
        fixed = 128 * self.k**2
        per_row = 256 * (self.k + len(self.columns) + 4)
        plan_workspace(
            "saved transformed Dataset evaluation",
            {
                "full_covariance_and_reductions": fixed,
                "projected_rows_basis_and_gradients": per_row,
            },
            budget_bytes=budget,
        )
        self.rows = min(65536, max(1, (budget - fixed) // per_row))
        self.plan = plan_workspace(
            "saved transformed Dataset evaluation",
            {
                "full_covariance_and_reductions": fixed,
                "projected_rows_basis_and_gradients": self.rows * per_row,
            },
            budget_bytes=budget,
        )
        self.original = self.retained = self.maximum = 0
        self.digest = hashlib.sha256()

    def batches(self):
        self.source.assert_unchanged()
        iterator = self.source.iter_batches(self.columns, batch_rows=self.rows)
        try:
            for raw in iterator:
                n = len(raw)
                if (self.original + n) * (self.k**2 * 16 + len(self.columns) * 64) > self.max_work:
                    fail("work_limit", "Complete Dataset smoothing evaluation exceeds max_work.")
                self.maximum = max(self.maximum, n)
                self.digest.update(
                    pd.util.hash_pandas_object(raw, index=False, categorize=False)
                    .to_numpy(dtype="<u8")
                    .tobytes()
                )
                keep = ~raw.loc[:, self.columns].isna().any(axis=1)
                if not bool(keep.all()) and self.missing == "raise":
                    fail(
                        "missing_values",
                        "Query predictors/averaging weights contain missing values; specify missing='drop'.",
                    )
                positions = keep.to_numpy().nonzero()[0]
                selected = raw.iloc[positions].reset_index(drop=True)
                original = self.original
                self.original += n
                self.retained += len(selected)
                if len(selected):
                    yield selected, (positions + original).tolist()
            self.source.assert_unchanged()
            if not self.retained:
                fail("empty_sample", "No complete query rows remain.")
        finally:
            iterator.close()

    def metadata(self):
        return {
            "source": self.source.provenance,
            "batch_rows": self.rows,
            "maximum_batch_rows": self.maximum,
            "original_rows": self.original,
            "retained_rows": self.retained,
            "dropped_rows": self.original - self.retained,
            "evaluation_digest": self.digest.hexdigest(),
            "resource_plan": self.plan.record(),
            "full_source_collected": False,
            "conditioning": "saved transformation/model and fixed query population; selection uncertainty excluded",
        }


def predict_dataset(result, source, *, interval, alpha, missing, max_work):
    from .replay import smoothing_predict
    from openecon.econometrics.postest.streaming_prediction import materialize_predictions

    query = Query(result, source, missing=missing, max_work=max_work)

    def frames():
        for selected, positions in query.batches():
            frame = smoothing_predict(
                result,
                data=selected,
                interval=interval,
                alpha=alpha,
                missing="raise",
                max_work=max_work,
            )
            frame["row"] = positions
            frame.index = pd.Index(positions, dtype="int64")
            yield frame

    output = materialize_predictions(
        frames(),
        {
            "estimator": result.spec.estimator,
            "source_result_id": result.id,
            "state_digest": query.state["digest"],
            "precision": "float64",
            "inference": "conditional selected-model mean" if interval else "prediction only",
        },
    )
    output._metadata["analysis"]["streaming"] = query.metadata()
    return output


def margins_dataset(
    result, source, *, variable, averaging_weights, interval, alpha, missing, max_work
):
    from .postestimation import _functional, _finish
    from openecon.streaming_design import numeric_values

    if (
        type(interval) is not bool
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0 < alpha < 1
    ):
        fail("invalid_option", "Use bool interval and alpha in (0,1).")
    if averaging_weights is not None and (
        not isinstance(averaging_weights, str) or not averaging_weights
    ):
        fail(
            "invalid_weights",
            "Dataset averaging_weights must be a column name; resident whole-row lists are unsupported.",
        )
    query = Query(
        result,
        source,
        missing=missing,
        max_work=max_work,
        extra_columns=[] if averaging_weights is None else [averaging_weights],
    )
    aggregate = _CompensatedSum((query.k,))
    mass = _CompensatedSum(())
    scale = 0.0
    positive = 0
    for selected, positions in query.batches():
        _, basis = _functional(selected, query.state, variable)
        weights = (
            torch.ones(len(selected), dtype=DT)
            if averaging_weights is None
            else numeric_values(selected[averaging_weights], averaging_weights)
        )
        if bool((weights < 0).any()):
            fail("invalid_weights", "Fixed-query averaging weights must be finite and nonnegative.")
        maximum = float(weights.max())
        if maximum == 0:
            continue
        new_scale = max(scale, maximum)
        aggregate.scale(torch.tensor(scale / new_scale, dtype=DT))
        mass.scale(torch.tensor(scale / new_scale, dtype=DT))
        normalized = weights / new_scale
        aggregate.add(finite(normalized @ basis))
        mass.add(normalized.sum())
        scale = new_scale
        positive += int((weights > 0).sum())
    if float(mass.value) <= 0:
        fail("invalid_weights", "Fixed-query averaging weights require positive mass.")
    contrast = finite((aggregate.value / mass.value).reshape(1, -1))
    estimate = finite(contrast @ torch.tensor(query.state["coefficients"], dtype=DT))
    frame = _finish(
        result,
        query.state,
        estimate,
        contrast,
        [0],
        query.original,
        interval,
        alpha,
        variable=variable,
        target="fixed-covariate average partial"
        if variable
        else "fixed-covariate average response",
        retained_positions=None,
        sample_positions=[0],
        normalized_averaging_weights=None,
    )
    frame.attrs.update(
        original_rows=query.original,
        dropped_rows=query.original - query.retained,
        retained_rows=query.retained,
        positive_weight_rows=positive,
        averaging_weights=averaging_weights,
        sample_positions=None,
        streaming=query.metadata(),
        publication_notes=[
            "Conditional on the saved transformation/model and fixed query population; selection uncertainty is excluded."
        ],
    )
    return frame
