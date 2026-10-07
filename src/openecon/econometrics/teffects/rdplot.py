"""Regression-discontinuity plot data (community ``rdplot``; Calonico, Cattaneo, Titiunik 2015).

On each side of the cutoff the running variable is partitioned into J bins,
evenly spaced (``es``: equal width) or quantile spaced (``qs``: equal counts),
and the outcome is averaged within bins; a global polynomial of order ``p``
(default 4) is fitted separately on each side for the plotted curve.

The number of bins on side s with n_s observations is chosen from two
constants estimated from the side's data: the integrated variance V from
the spacings of the sorted data, V = (1/(2 l)) sum dx_i dy_i^2 for ``es``
(l the length of the side's support) or (1/(2 n_s)) sum dy_i^2 for ``qs``,
and the integrated squared bias B from the derivative mu'(x) of a global
quartic fit, B = (l^2/(12 n_s)) sum mu'(x_i)^2 for ``es`` or
(n_s/12) sum dx_i^2 mu'(xbar_i)^2 for ``qs``. Then

    IMSE-optimal (es, qs):          J = ceil((2 B n_s / V)^(1/3))
    mimicking variance (esmv, qsmv): J = ceil((var(y_s) / V) n_s / log(n_s)^2)

(the IMSE rule minimizes B/J^2 + V J/n_s). ``nbins`` overrides the choice.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, TableSet, kernel_call, make_spec, table
from openecon.engines.inference import critical_value
from openecon.engines.linalg import least_squares

_SELECTORS = ("es", "esmv", "qs", "qsmv")


def _polynomial(dx: Tensor, y: Tensor, order: int) -> Tensor:
    design = dx[:, None] ** torch.arange(order + 1, dtype=torch.float64)
    return kernel_call(least_squares, design, y, None, drop_collinear=False, tol=0.0).beta


def _evaluate(beta: Tensor, dx: Tensor) -> Tensor:
    return (dx[:, None] ** torch.arange(len(beta), dtype=torch.float64)) @ beta


def _derivative(beta: Tensor, dx: Tensor) -> Tensor:
    powers = torch.arange(1, len(beta), dtype=torch.float64)
    return (dx[:, None] ** (powers - 1)) @ (beta[1:] * powers)


def _bins(dx: Tensor, y: Tensor, length: float, selector: str) -> int:
    n = dx.numel()
    order = torch.argsort(dx, stable=True)
    sx, sy = dx[order], y[order]
    gaps, jumps = sx[1:] - sx[:-1], sy[1:] - sy[:-1]
    beta = _polynomial(dx, y, min(4, int(torch.unique(dx).numel()) - 1))
    if selector.startswith("es"):
        variance = float((gaps * jumps.square()).sum()) / (2 * length)
        bias = length ** 2 / (12 * n) * float(_derivative(beta, dx).square().sum())
    else:
        variance = float(jumps.square().sum()) / (2 * n)
        middle = (sx[1:] + sx[:-1]) / 2
        bias = n / 12 * float((gaps.square() * _derivative(beta, middle).square()).sum())
    if not variance > 0:
        return 1
    if selector.endswith("mv"):
        return max(1, math.ceil(float(y.var()) / variance * n / math.log(n) ** 2))
    if not bias > 0:
        return 1
    return max(1, math.ceil((2 * bias * n / variance) ** (1 / 3)))


def _edges(dx: Tensor, count: int, low: float, high: float, quantile: bool) -> Tensor:
    if quantile:
        probabilities = torch.linspace(0, 1, count + 1, dtype=torch.float64)
        edges = torch.quantile(dx, probabilities)
        edges[0], edges[-1] = low, high
        return edges
    return torch.linspace(low, high, count + 1, dtype=torch.float64)


def rdplot(*, data: Any, y: str, running: str, cutoff: float = 0.0, p: int = 4,
           nbins: int | Sequence[int] | None = None, binselect: str = "esmv",
           grid: int = 200, missing: str = "raise", alpha: float = 0.05) -> TableSet:
    """Data of a regression-discontinuity plot (community ``rdplot``).

    Returns a ``TableSet`` with two tables: ``bins`` (one row per bin: side,
    bin number, x_low, x_high, x_mid, x_mean, y_mean, n, y_se = sd/sqrt(n),
    ci_low, ci_high at level 1 - alpha) and ``poly`` (the global polynomial of
    order ``p`` fitted separately on each side, evaluated on ``grid`` points per
    side). ``attrs`` hold the numbers of bins, the selector, p and the cutoff.

    ``binselect``: ``'esmv'`` (default; evenly spaced, mimicking-variance
    number of bins), ``'es'`` (evenly spaced, IMSE-optimal), ``'qsmv'``,
    ``'qs'`` (quantile spaced); ``nbins`` (a number or ``[left, right]``)
    overrides the selection. Observations with running variable >= ``cutoff``
    form the right (treated) side. The selection rules are written out in
    ``docs/econometrics/teffects.md``.

    Example::

        import openecon as oe
        plot = oe.rdplot(data=df, y="vote", running="margin")
        plot["bins"], plot["poly"]
    """
    if binselect not in _SELECTORS:
        raise AnalysisError("invalid_spec", f"binselect must be one of {', '.join(_SELECTORS)}.")
    if not isinstance(p, int) or isinstance(p, bool) or not 0 <= p <= 8:
        raise AnalysisError("invalid_spec", "p must be an integer between 0 and 8.")
    if not isinstance(grid, int) or grid < 2:
        raise AnalysisError("invalid_spec", "grid must be an integer of at least 2.")
    spec = make_spec("rdrobust", outcome=y, columns={"running": running}, missing=missing,
                     alpha=alpha, options={"cutoff": float(cutoff)})
    frame = ModelFrame(spec, data)
    yv, x = frame.numeric(y), frame.numeric(running)
    c = float(cutoff)
    if isinstance(nbins, (list, tuple)):
        if len(nbins) != 2:
            raise AnalysisError("invalid_spec", "nbins must be a number or [left, right].")
        requested = [int(nbins[0]), int(nbins[1])]
    else:
        requested = None if nbins is None else [int(nbins), int(nbins)]
    if requested is not None and min(requested) < 1:
        raise AnalysisError("invalid_spec", "nbins must be positive.")
    critical = critical_value(alpha, None)
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    counts: list[int] = []
    for index, (name, mask) in enumerate((("left", x < c), ("right", x >= c))):
        dx, ys = x[mask] - c, yv[mask]
        distinct = int(torch.unique(dx).numel())
        if distinct <= max(p, 1):
            raise AnalysisError("insufficient_observations", f"The {name} side of the cutoff needs "
                                f"more than {max(p, 1)} distinct running-variable values.")
        low, high = (float(dx.min()), 0.0) if name == "left" else (0.0, float(dx.max()))
        length = high - low
        count = requested[index] if requested else _bins(dx, ys, length, binselect)
        count = min(count, dx.numel())
        counts.append(count)
        edges = _edges(dx, count, low, high, binselect.startswith("qs"))
        assignment = (torch.searchsorted(edges, dx, right=True) - 1).clamp(0, count - 1)
        n_bin = torch.bincount(assignment, minlength=count).to(torch.float64)
        sum_y = torch.zeros(count, dtype=torch.float64).index_add_(0, assignment, ys)
        sum_x = torch.zeros(count, dtype=torch.float64).index_add_(0, assignment, dx)
        sum_y2 = torch.zeros(count, dtype=torch.float64).index_add_(0, assignment, ys.square())
        for j in range(count):
            if n_bin[j] == 0:
                continue
            m = float(n_bin[j])
            mean_y = float(sum_y[j]) / m
            var = (float(sum_y2[j]) - m * mean_y ** 2) / (m - 1) if m > 1 else float("nan")
            se = math.sqrt(max(var, 0.0) / m) if m > 1 else None
            rows.append({
                "side": name, "bin": (j - count) if name == "left" else j + 1,
                "x_low": float(edges[j]) + c, "x_high": float(edges[j + 1]) + c,
                "x_mid": float(edges[j] + edges[j + 1]) / 2 + c,
                "x_mean": float(sum_x[j]) / m + c, "y_mean": mean_y, "n": int(m),
                "y_se": se, "ci_low": None if se is None else mean_y - critical * se,
                "ci_high": None if se is None else mean_y + critical * se,
            })
        beta = _polynomial(dx, ys, p)
        points = torch.linspace(low, high, grid, dtype=torch.float64)
        for value, fitted in zip(points.tolist(), _evaluate(beta, points).tolist(), strict=True):
            curves.append({"side": name, "x": value + c, "fit": fitted})
    columns = ["side", "bin", "x_low", "x_high", "x_mid", "x_mean", "y_mean", "n", "y_se",
               "ci_low", "ci_high"]
    bins = table(rows, columns=columns)
    poly = table(curves, columns=["side", "x", "fit"])
    return TableSet({"bins": bins, "poly": poly}, title="Regression-discontinuity plot data",
                    bins_left=counts[0], bins_right=counts[1], binselect=binselect
                    if requested is None else "manual", p=p, cutoff=c,
                    n_left=int((x < c).sum()), n_right=int((x >= c).sum()),
                    confidence_level=1 - alpha)
