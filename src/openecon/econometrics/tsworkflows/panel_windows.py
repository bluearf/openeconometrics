"""Calendar windows over complete observed panel cross-sections."""

from __future__ import annotations

import torch
from pandas.api.types import is_bool_dtype, is_integer_dtype

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, TableSet, table
from openecon.econometrics.tsworkflows.workflows import _integer
from openecon.models import ModelSpec


def rolling_panel(
    spec, *, data, window, step=1, expanding=False, reverse=False, max_fits=100, on_error="raise"
):
    """Refit panel models over common integer-calendar windows, never row-count cuts.

    window/step count distinct consecutive periods. All observed unit rows in
    each window enter the model's own sample/lag/design contract, preserving
    unbalanced panels and original physical positions. Mean/control construction
    is rerun inside each window. Forward expanding windows stop at their origin;
    reverse windows are labeled retrospective. Returns coefficients, full V,
    diagnostics/spec/result IDs and explicit failures; no panel forecast adapter
    is synthesized. Resident input only, max_fits<=1000 and a 1e8 row-visit budget.
    """
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError("unsupported_dataset", "rolling_panel does not collect Dataset inputs.")
    if not isinstance(spec, ModelSpec) or spec.panel is None or spec.time is None:
        raise AnalysisError(
            "invalid_spec", "rolling_panel needs a panel ModelSpec with explicit time."
        )
    if any(not isinstance(flag, bool) for flag in (expanding, reverse)) or on_error not in {
        "raise",
        "record",
    }:
        raise AnalysisError(
            "invalid_option", "Use boolean window directions and on_error='raise'/'record'."
        )
    frame = ModelFrame(spec, data)
    if not is_integer_dtype(frame.series(spec.time).dtype) or is_bool_dtype(
        frame.series(spec.time).dtype
    ):
        raise AnalysisError(
            "invalid_time",
            "Panel windows need exact integer calendar periods; dates are not ranked silently.",
        )
    frame.sort_panel()
    time = frame.time_index()
    periods = torch.unique(time, sorted=True)
    if len(periods) < 2 or bool((torch.diff(periods) != 1).any()):
        raise AnalysisError(
            "time_gaps", "The shared panel calendar must contain consecutive periods."
        )
    window = _integer(window, "window", 2, len(periods))
    step = _integer(step, "step", 1, len(periods))
    max_fits = _integer(max_fits, "max_fits", 1, 1000)
    windows = (
        [
            (start, len(periods) if expanding else start + window)
            for start in range(len(periods) - window, -1, -step)
        ]
        if reverse
        else [
            (0 if expanding else stop - window, stop)
            for stop in range(window, len(periods) + 1, step)
        ]
    )
    if len(windows) > max_fits or frame.n * len(windows) > 100_000_000:
        raise AnalysisError(
            "work_budget_exceeded",
            "Panel windows exceed max_fits or the declared row-visit budget.",
        )
    frame.workspace_plan(
        "panel-window summaries", {"records": len(windows) * 16384, "row_masks": frame.n * 24}
    )
    history, coefficients = [], []
    for start, stop in windows:
        rows = torch.where((time >= periods[start]) & (time <= periods[stop - 1]))[0].tolist()
        positions = [frame.positions[i] for i in rows]
        record = {
            "start_period": int(periods[start]),
            "end_period": int(periods[stop - 1]),
            "origin_period": int(periods[start] if reverse else periods[stop - 1]),
            "input_positions": positions,
        }
        try:
            result = fit(spec, data=frame.sample.iloc[rows].copy())
            if result.extra.get("target") == "prediction":
                raise AnalysisError(
                    "unsupported_target",
                    "Panel coefficient windows require an inferential estimator.",
                )
            record.update(
                status="ok",
                nobs=result.nobs,
                result_id=result.id,
                sample_positions=[positions[i] for i in result.sample_positions],
                coefficients=[c.model_dump(mode="json") for c in result.coefficients],
                covariance_matrix=result.covariance_matrix,
                inference=result.inference,
                metrics=result.metrics,
                spec=result.spec.model_dump(mode="json"),
                optimizer=result.provenance.get("optimizer"),
            )
            coefficients.extend(
                {
                    "origin_period": record["origin_period"],
                    "term": c.term,
                    "estimate": c.estimate,
                    "std_error": c.std_error,
                    "ci_low": c.ci_low,
                    "ci_high": c.ci_high,
                }
                for c in result.coefficients
            )
        except AnalysisError as exc:
            if on_error == "raise":
                raise
            record.update(status="failed", error_code=exc.code, error=str(exc))
        history.append(record)
    return TableSet(
        {"coefficients": table(coefficients)},
        title="Panel calendar refits",
        origins=history,
        window_periods=window,
        step_periods=step,
        expanding=expanding,
        reverse=reverse,
        sample_positions=frame.positions,
        spec=spec.model_dump(mode="json"),
        lookahead="retrospective reverse windows"
        if reverse
        else "Each model sees only observed panel rows through its origin; preparation is rerun inside the window.",
    )
