"""Strict calendars and numerical contracts shared by the advanced families."""

from __future__ import annotations

import torch
from pandas.api.types import is_datetime64_any_dtype

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, kernel_call
from openecon.engines.linalg import least_squares


def regular_frame(spec, data):
    frame = ModelFrame(spec, data)
    if spec.time is not None:
        if is_datetime64_any_dtype(frame.series(spec.time).dtype):
            raise AnalysisError(
                "invalid_time",
                "Use an integer calendar with explicit equally spaced periods; dates are not silently ranked.",
            )
        frame.sort_panel()
        time = frame.time_index()
    else:
        time = torch.tensor(frame.positions, dtype=torch.int64)
    groups, count = (
        frame.codes(spec.panel) if spec.panel else (torch.zeros(frame.n, dtype=torch.int64), 1)
    )
    blocks = []
    for unit in range(count):
        rows = torch.where(groups == unit)[0]
        if len(rows) > 1 and bool((time[rows][1:] - time[rows][:-1] != 1).any()):
            raise AnalysisError(
                "time_gaps",
                "Each unit needs consecutive integer periods. Missing rows and calendar gaps are not compressed or imputed.",
            )
        blocks.append(rows)
    return frame, blocks


def ls(x, y):
    if len(y) <= x.shape[1]:
        raise AnalysisError(
            "insufficient_observations", "The effective sample must exceed the full design width."
        )
    result = kernel_call(least_squares, x, y, drop_collinear=False)
    if float(result.resid.square().sum()) <= torch.finfo(torch.float64).eps ** 2 * max(
        float(y.square().sum()), 1
    ):
        raise AnalysisError("perfect_fit", "An exact fit has no estimable residual uncertainty.")
    return result


def deterministic(n, trend, *, start=1):
    t = torch.arange(start, start + n, dtype=torch.float64)
    powers = {"n": [], "c": [0], "ct": [0, 1], "ctt": [0, 1, 2]}[trend]
    return (
        torch.stack([t**power for power in powers], dim=1)
        if powers
        else torch.empty((n, 0), dtype=torch.float64),
        ["Intercept", "trend", "trend_squared"][: len(powers)],
    )
