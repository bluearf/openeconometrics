"""Pearson and partial correlations from bounded moment/TSQR reductions."""
from __future__ import annotations

import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.engines import distributions as dist
from openecon.engines.streaming_ols import _CompensatedSum
from openecon.resources import plan_workspace
from . import common as c
from .correlation import _t_p_value
from .replay import Moments, Replay


def correlate(data: Any, names: list[str], *, method: str, pairwise: bool, ci: bool,
              alpha: float) -> TableSet:
    if method != "pearson":
        from .streaming_rank import correlate as rank_correlate
        return rank_correlate(data, names, method=method, pairwise=pairwise, ci=ci, alpha=alpha)
    sample = Replay(data, names, names, "drop")
    p = len(names)
    geometry = plan_workspace("chunked pairwise Pearson correlations", {
        "pair_moment_matrices": 512 * p*p,
        "helper_source_buffers": sample.plan.estimated_bytes})
    states = [Moments(1) for _ in names]
    for frame in sample.batches(listwise=False):
        x = torch.stack([c.values(frame, name, allow_missing=True, check_scale=False) for name in names], 1)
        complete = ~torch.isnan(x).any(1)
        if not pairwise:
            x = x[complete]
        for position, state in enumerate(states):
            values = x[:, position]
            state.add(values[~torch.isnan(values), None])
    if (sample.raw_rows if pairwise else sample.raw_rows-sample.dropped) < 2:
        raise AnalysisError("insufficient_observations", "Correlations need at least two complete observations.")
    # Centre around anchored global means before pairwise reduction. The pairwise
    # correction is exactly the resident masked-cross-product definition; all
    # four live accumulators are p-by-p, independent of source row count.
    anchor = torch.tensor([float(state.anchor[0]) for state in states], dtype=torch.float64)
    location = torch.tensor([float(state.mean[0]) for state in states], dtype=torch.float64)
    accumulated = [_CompensatedSum((p, p)) for _ in range(4)]
    for frame in sample.batches(listwise=False):
        x = torch.stack([c.values(frame, name, allow_missing=True, check_scale=False) for name in names], 1)
        present = ~torch.isnan(x)
        if not pairwise:
            complete = present.all(1)
            x, present = x[complete], present[complete]
        mask = present.to(torch.float64)
        centred = torch.where(present, (x-anchor)-location, torch.zeros_like(x))
        accumulated[0].add(mask.T@mask)
        accumulated[1].add(centred.T@mask)
        accumulated[2].add(centred.square().T@mask)
        accumulated[3].add(centred.T@centred)
    n, sums, squares, cross = [value.value for value in accumulated]
    safe = n.clamp_min(1)
    spread = (squares-sums.square()/safe).clamp_min(0)
    cross = cross-sums*sums.T/safe
    product = spread*spread.T
    denominator = torch.where((product > 1e-290) & torch.isfinite(product), product.sqrt(),
                              spread.sqrt()*spread.sqrt().T)
    matrix = torch.where(denominator > 0, cross/denominator.clamp_min(1e-300),
                         torch.full_like(cross, float("nan"))).clamp(-1, 1)
    coefficients = [[None if math.isnan(value) else value for value in row] for row in matrix.tolist()]
    counts = [[int(value) for value in row] for row in n.tolist()]
    p_values = [[_t_p_value(coefficients[i][j], counts[i][j]) if i != j else None
                 for j in range(p)] for i in range(p)]
    for i, state in enumerate(states):
        counts[i][i] = state.n
        coefficients[i][i] = 1.0 if state.n > 1 and float(state.sscp[0,0]) > 0 else None
    tables = {"coefficients": c.frame(coefficients, columns=names, index=names),
              "p_values": c.frame(p_values, columns=names, index=names),
              "n": c.frame(counts, columns=names, index=names)}
    if ci:
        bound = dist.normal_isf(alpha*0.5)
        rows = []
        for i in range(p):
            for j in range(i+1, p):
                value, count = coefficients[i][j], counts[i][j]
                low = high = None
                if value is not None and count > 3:
                    if abs(value) == 1:
                        low = high = value
                    else:
                        half = bound / math.sqrt(count-3)
                        low, high = math.tanh(math.atanh(value)-half), math.tanh(math.atanh(value)+half)
                rows.append([names[i], names[j], value, low, high, count])
        tables["intervals"] = c.frame(rows, columns=["var_i", "var_j", "coefficient", "ci_low", "ci_high", "n"])
    return TableSet(tables, title="Pearson correlations", method=method,
                    missing="pairwise" if pairwise else "listwise", n=sample.raw_rows,
                    n_complete=sample.raw_rows-sample.dropped, alpha=alpha, distribution="t",
                    **sample.attrs(), correlation_resource_plan=geometry.record())


def pcorr(data: Any, y: str, names: list[str], controls: list[str], missing: str):
    from .streaming_anova import ReplayLinearModel
    columns = [y, *names, *controls]
    sample = Replay(data, columns, columns, missing)
    try:
        model = ReplayLinearModel(sample, y, [], [*names, *controls], "none", False)
    except AnalysisError as exc:
        if exc.code in {"collinear_design", "single_level"}:
            raise AnalysisError("collinear_design", "A variable is constant or collinear, so its partial correlation is undefined.") from exc
        raise
    ssr, total, df = float(model.error[0, 0]), model.corrected_total, model.df_error
    if not total > 0 or not ssr > 1e-28*total:
        raise AnalysisError("perfect_fit", "y is constant or fitted exactly, so partial correlations are undefined.")
    r2 = 1-ssr/total
    se = (ssr/df*model.full.xtx_inv.diagonal()).sqrt()
    rows = []
    for position in range(1, len(names)+1):
        statistic = float(model.beta[position, 0]/se[position])
        partial = statistic/math.sqrt(statistic*statistic+df)
        semi = math.copysign(math.sqrt(statistic*statistic*(1-r2)/df), statistic)
        rows.append([partial, semi, partial*partial, semi*semi, statistic, df, c.t_two_sided(statistic, df)])
    return c.frame(rows, columns=["partial_corr", "semipartial_corr", "partial_corr_sq", "semipartial_corr_sq",
                                  "statistic", "df", "p_value"], index=names, n=model.n,
                   n_missing=sample.dropped, r_squared=r2, controls=controls, outcome=y,
                   distribution="t", missing="listwise", **sample.attrs(),
                   factorial_resource_plan=model.resource_plan.record())
