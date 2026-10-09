"""Fixed degree-one spline CATREG with exact signed monotonic cones."""
from __future__ import annotations

import copy
import itertools
import math
from numbers import Real
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import summary_state
from .frequency import _admit
from .optimal import _integer, _seal
from .scaling import _save
from . import spline_core as c

SOURCES = ["https://arxiv.org/abs/1611.05433",
           "https://www.ibm.com/docs/en/spss-statistics/30.0.0?topic=catreg-examples-command"]
METHODS = ("catreg_spline", "catreg_mspline")


def _data(data, outcome, predictors, knots, missing, max_bytes, max_work, monotone=False):
    if not isinstance(predictors, list) or not 1 <= len(predictors) <= 6:
        c._error("Supply 1–6 spline predictors.")
    if (any(not isinstance(v, str) or not 1 <= len(v) <= 128 for v in predictors)
            or len(set(predictors)) != len(predictors)):
        c._error("Predictors must be distinct bounded column names.")
    if (not isinstance(data, pd.DataFrame) or not isinstance(outcome, str)
            or not 1 <= len(outcome) <= 128 or outcome in predictors
            or list(data.columns).count(outcome) != 1):
        c._error("Supply a distinct bounded numeric outcome column.")
    if missing not in ("drop", "raise"):
        c._error("missing must be drop or raise.")
    if not 4 <= len(data) <= 3000:
        c._error("Spline input admits 4–3000 physical rows.", "resource_limit")
    for name in predictors:
        if list(data.columns).count(name) != 1:
            c._error("Each required column must occur exactly once.")
    admitted = c._knots(knots, predictors)
    dimension = sum(len(v)-1 for v in admitted.values())
    patterns = 2**len(predictors) if monotone else 1
    work = 128*patterns*(len(data)*dimension*dimension + (40*dimension**4 if monotone else dimension**3))
    plan = _admit("fixed spline regression", 1024*len(data)*(dimension+len(predictors)+4),
                  work, max_bytes, max_work)
    selected = data.loc[:, [outcome]+predictors]
    absent = selected.isna().any(axis=1).tolist()
    if missing == "raise" and any(absent):
        c._error("Required response/predictors contain missing observations.", "missing_data")
    positions = [i for i, flag in enumerate(absent) if not flag]
    complete = selected.iloc[positions].reset_index(drop=True)
    prepared = c._prepare(complete, predictors, admitted, "raise", max_bytes, max_work,
                          "fixed spline regression")
    series = complete[outcome]
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        c._error("Response must be numeric.", "invalid_data")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
           or abs(v) > 1e100 for v in series.tolist()):
        c._error("Response must contain bounded finite real values.", "invalid_data")
    y = torch.tensor(series.tolist(), dtype=c.DT)
    if float((y-y.mean()).square().mean().sqrt()) <= 1e-12*max(1., float(y.abs().max())):
        c._error("Response is constant or numerically degenerate.", "degenerate_transform")
    prepared.update(y=y, positions=positions, input_nobs=len(data), plan=plan)
    return prepared


def _fit_coefficients(design, target, widths, monotone):
    c._rank(design)
    if not monotone:
        coefficient = torch.linalg.lstsq(design, target).solution
        if not c._stationary(design, target, coefficient):
            c._error("Spline least squares failed normalized residual acceptance.", "nonconvergence")
        return coefficient, [[0, ",".join(["free"]*len(widths)),
                              float((target-design@coefficient).square().sum())]], 0, [coefficient.tolist()]
    candidates, fits = [], []
    for index, signs in enumerate(itertools.product((-1., 1.), repeat=len(widths))):
        vector = torch.tensor([s for s, width in zip(signs, widths) for _ in range(width)], dtype=c.DT)
        positive = c._nnls(design*vector, target)
        coefficient = positive*vector
        loss = float((target-design@coefficient).square().sum())
        fits.append(coefficient)
        candidates.append([index, ",".join("+" if s > 0 else "-" for s in signs), loss])
    chosen = min(range(len(candidates)), key=lambda j: (candidates[j][2], j))
    return fits[chosen], candidates, chosen, [v.tolist() for v in fits]


def _output(state):
    raw = torch.tensor(state["raw"], dtype=c.DT)
    y = torch.tensor(state["y"], dtype=c.DT)
    variables = state["variables"]
    coefficient = torch.tensor(state["coefficient"], dtype=c.DT)
    transformed, maps, effects, centered, offset = [], [], [], [], 0
    for j, name in enumerate(variables):
        knots = state["knots"][name]
        basis = c._basis(raw[:, j], knots)
        mean = basis.mean(0)
        segment = coefficient[offset:offset+len(knots)-1]
        contribution = (basis-mean)@segment
        sd = contribution.square().mean().sqrt()
        if float(sd) <= 1e-12*max(1., float((y-y.mean()).square().mean().sqrt())):
            c._error("A spline effect has zero variance; its normalized transformation is unidentified.", "degenerate_transform")
        # q is increasing in the monotone domain; nominal q is oriented by its
        # largest absolute knot value. The regression effect retains its sign.
        knot_value = c._basis(torch.tensor(knots, dtype=c.DT), knots)@segment-mean@segment
        sign = (1. if float(segment.sum()) >= 0 else -1.) if state["monotone"] else (
            1. if float(knot_value[int(torch.argmax(knot_value.abs()))]) >= 0 else -1.)
        q = contribution/sd*sign
        beta = sd*sign
        transformed.append(q)
        centered.append(contribution)
        effects.append([name, float(beta)])
        maps.extend([[name, k, value, float(knot_value[k]/sd*sign)]
                     for k, value in enumerate(knots)])
        offset += len(knots)-1
    z = torch.stack(transformed, dim=1)
    fitted = y.mean()+torch.stack(centered, dim=1).sum(1)
    error = y-fitted
    sse = float(error.square().sum())
    n = len(y)
    result = TableSet({
        "fit": table([[n, sse, 1-sse/float((y-y.mean()).square().sum()), state["condition"]]],
                     columns=["n", "sse", "r_squared", "basis_condition"]),
        "effects": table(effects, columns=["variable", "coefficient"]),
        "spline_knots": table(maps, columns=["variable", "knot", "raw_value", "quantification"]),
        "transformed": table([[position]+row for position, row in zip(state["positions"], z.tolist())],
                             columns=["source_position"]+variables),
        "fitted": table([[position, observed, fit, residual]
                         for position, observed, fit, residual in zip(state["positions"], y.tolist(), fitted.tolist(), error.tolist())],
                        columns=["source_position", "observed", "fitted", "residual"]),
        "sign_patterns": table(state["patterns"], columns=["pattern", "directions", "sse"]),
    }, title="Declared degree-one "+("monotone" if state["monotone"] else "nonmonotone")+" spline CATREG",
        method=state["method"], variables=variables, outcome=state["outcome"],
        n=n, n_input=state["input_nobs"], sample_positions=state["positions"],
        n_missing=state["input_nobs"]-n, dtype="float64", device="cpu", weight_type="unweighted",
        converged=True, degree=1, monotone=state["monotone"], chosen_pattern=state["chosen"],
        resources=state["plan"], spline_state=state, state_sha256=_seal(state), sources=SOURCES,
        inference="descriptive fitted transformations and predictions; no adaptive coefficient Wald uncertainty",
        optimization=("all finite signed monotonic cones solved to KKT; minimum fixed-knot additive loss"
                      if state["monotone"] else "full-rank joint fixed-knot additive least squares"),
        outside_support="raise; no extrapolation", missing=state["missing"])
    return _save(result)


def _fit(data, outcome, predictors, knots, monotone, missing, max_bytes, max_work, device):
    if device != "cpu":
        c._error("Spline CATREG supports CPU float64 only.", "unsupported_option")
    prepared = _data(data, outcome, predictors, knots, missing, max_bytes, max_work, monotone)
    design = torch.cat(prepared["bases"], dim=1)
    condition = c._rank(design)
    widths = [b.shape[1] for b in prepared["bases"]]
    coefficient, patterns, chosen, candidate_coefficients = _fit_coefficients(design, prepared["y"]-prepared["y"].mean(), widths, monotone)
    state = dict(version=1, kind="fixed_spline_catreg", method=METHODS[int(monotone)],
                 variables=predictors, outcome=outcome, knots=prepared["knots"],
                 raw=prepared["raw"].tolist(), y=prepared["y"].tolist(),
                 positions=prepared["positions"], input_nobs=prepared["input_nobs"],
                 coefficient=coefficient.tolist(), condition=condition, patterns=patterns,
                 chosen=chosen, candidate_coefficients=candidate_coefficients,
                 monotone=monotone, missing=missing, plan=prepared["plan"])
    return _output(state)


@resident_cpu
def catreg_spline(data: Any, outcome: str, predictors: list[str], *, knots: dict,
                  missing: str = "drop", max_bytes: int = c.LIMIT_BYTES,
                  max_work: int = c.WORK, device: str = "cpu") -> TableSet:
    """Fit nonmonotone degree-one optimal predictor splines on declared knots.

    Numeric response/predictors, 1–6 variables, 2–8 knots including endpoints,
    at most 32 full-rank segment columns and 3000 rows. No extrapolation,
    automatic knots, weights or adaptive coefficient Wald inference.
    """
    return _fit(data, outcome, predictors, knots, False, missing, max_bytes, max_work, device)


@resident_cpu
def catreg_mspline(data: Any, outcome: str, predictors: list[str], *, knots: dict,
                   missing: str = "drop", max_bytes: int = c.LIMIT_BYTES,
                   max_work: int = c.WORK, device: str = "cpu") -> TableSet:
    """Fit increasing degree-one predictor maps with signed regression effects.

    Every one of 2**p signed cones is solved with native bounded NNLS and KKT
    acceptance, then the minimum fixed-knot loss is selected deterministically.
    Maps have centered unit variance; zero-effect maps are unidentified/refused.
    """
    return _fit(data, outcome, predictors, knots, True, missing, max_bytes, max_work, device)


def _checked(result, max_bytes, max_work, query_rows=0):
    if not isinstance(result, TableSet) or result.attrs.get("method") not in METHODS:
        c._error("Supply a complete fitted/restored spline CATREG.", "invalid_result")
    state = result.attrs.get("spline_state")
    fields = {"version", "kind", "method", "variables", "outcome", "knots", "raw", "y",
              "positions", "input_nobs", "coefficient", "condition", "patterns", "chosen", "candidate_coefficients", "monotone", "missing", "plan"}
    if not isinstance(state, dict) or set(state) != fields:
        c._error("Saved spline state is incomplete or unknown.", "invalid_state")
    try:
        estimate = c._metadata_bound(result.attrs)
        if len(result) != 7 or any(not isinstance(frame, pd.DataFrame) or len(frame) > 12000 or len(frame.columns) > 16 for frame in result.values()):
            raise ValueError("Tables")
        for frame in result.values():
            estimate += 64*(frame.size+len(frame.index)+len(frame.columns))
            for row in frame.itertuples(index=False, name=None):
                for value in row:
                    if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
                        value = value.item()
                    estimate += c._metadata_bound(value)
                    if estimate > 8*1024**2:
                        c._error("Saved spline result exceeds the bounded state domain.", "resource_limit")
        if estimate > 8*1024**2:
            c._error("Saved spline result exceeds the bounded state domain.", "resource_limit")
        variables, n = state["variables"], len(state["y"])
        if not 4 <= n <= 3000 or not isinstance(variables, list) or not 1 <= len(variables) <= 6:
            raise ValueError("Shape")
        dimension = sum(len(v)-1 for v in state["knots"].values())
        if not 1 <= dimension <= 32 or len(state["raw"]) != n or any(len(row) != len(variables) for row in state["raw"]):
            raise ValueError("Shape")
        _admit("saved spline validation and query", 2048*n*(dimension+len(variables)+4)+1024*query_rows*(len(variables)+32),
               128*2**len(variables)*(n*dimension**2+dimension**3)+128*query_rows*32**2, max_bytes, max_work)
        if len(summary_state(result).encode()) > 8*1024**2:
            c._error("Saved spline result exceeds 8 MiB.", "resource_limit")
        if (type(state["version"]) is not int or state["version"] != 1
                or type(state["chosen"]) is not int or state["kind"] != "fixed_spline_catreg"
                or type(state["monotone"]) is not bool or state["method"] != METHODS[int(state["monotone"])]):
            raise ValueError("Schema")
        if state["missing"] not in ("drop", "raise"):
            raise ValueError("Missing policy")
        n_input, positions = state["input_nobs"], state["positions"]
        if (type(n_input) is not int or not n <= n_input <= 3000 or len(positions) != n
                or positions != sorted(set(positions)) or any(type(v) is not int or not 0 <= v < n_input for v in positions)):
            raise ValueError("Positions")
        plan = state["plan"]
        if not isinstance(plan, dict) or set(plan) != {"operation", "estimated_workspace_bytes", "budget_bytes", "buffers", "scope"}:
            raise ValueError("Resource declaration")
        estimated = 1024*n_input*(dimension+len(variables)+4)
        if (plan["operation"] != "fixed spline regression" or plan["estimated_workspace_bytes"] != estimated
                or type(plan["budget_bytes"]) is not int or not estimated <= plan["budget_bytes"] <= c.LIMIT_BYTES
                or plan["buffers"] != {"physical_rows_and_numerical_workspace": estimated}):
            raise ValueError("Resource declaration")
        frame = pd.DataFrame(state["raw"], columns=variables)
        frame[state["outcome"]] = state["y"]
        prepared = _data(frame, state["outcome"], variables, state["knots"], "raise", max_bytes, max_work)
        design = torch.cat(prepared["bases"], dim=1)
        coefficient = torch.tensor(state["coefficient"], dtype=c.DT)
        target = prepared["y"]-prepared["y"].mean()
        widths = [b.shape[1] for b in prepared["bases"]]
        signs = list(itertools.product((-1., 1.), repeat=len(variables))) if state["monotone"] else [None]
        if len(state["candidate_coefficients"]) != len(signs):
            raise ValueError("Missing cone solutions")
        patterns = []
        for i, sign in enumerate(signs):
            candidate = torch.tensor(state["candidate_coefficients"][i], dtype=c.DT)
            if candidate.shape != coefficient.shape or len(candidate) != dimension or not bool(torch.isfinite(candidate).all()):
                raise ValueError("Candidate shape")
            vector = (torch.tensor([s for s, width in zip(sign, widths) for _ in range(width)], dtype=c.DT)
                      if sign else torch.ones(dimension, dtype=c.DT))
            positive = candidate*vector
            if not c._stationary(design*vector, target, positive, nonnegative=sign is not None):
                raise ValueError("Normalized feasible-direction stationarity")
            loss = float((target-design@candidate).square().sum())
            patterns.append([i, ",".join("+" if s > 0 else "-" for s in sign) if sign else ",".join(["free"]*len(variables)), loss])
        chosen = min(range(len(patterns)), key=lambda j: (patterns[j][2], j))
        c._close(coefficient, torch.tensor(state["candidate_coefficients"][chosen], dtype=c.DT), "selected full-basis optimum")
        c._close(torch.tensor([v[2] for v in state["patterns"]], dtype=c.DT),
                 torch.tensor([v[2] for v in patterns], dtype=c.DT), "all signed-cone losses")
        if (state["chosen"] != chosen or [v[:2] for v in state["patterns"]] != [v[:2] for v in patterns]
                or abs(state["condition"]-c._rank(design)) > 2e-8*max(1., state["condition"])):
            raise ValueError("Selection")
        if result.attrs.get("state_sha256") != _seal(state):
            raise ValueError("Checksum")
        expected_result = _output(copy.deepcopy(state))
        if summary_state(expected_result) != summary_state(result):
            raise ValueError("Full table or metadata mismatch")
    except AnalysisError as exc:
        if exc.code == "resource_limit":
            raise
        raise AnalysisError("invalid_state", "Saved spline regression numerical state is invalid.") from exc
    except (TypeError, ValueError, KeyError, IndexError, RuntimeError, AttributeError, OverflowError) as exc:
        raise AnalysisError("invalid_state", "Saved spline regression state or tables are inconsistent.") from exc
    return state


@resident_cpu
def catreg_spline_predict(result: TableSet, data: Any, *, missing: str = "raise",
                          max_bytes: int = c.LIMIT_BYTES, max_work: int = c.WORK,
                          device: str = "cpu") -> TableSet:
    """Project complete saved degree-one CATREG maps within their knot spans.

    Restored geometry, all signed-cone losses and display metadata are checked.
    Missing rows may be dropped with physical positions; outside support fails.
    """
    if device != "cpu":
        c._error("Spline prediction supports CPU float64 only.", "unsupported_option")
    if not isinstance(data, pd.DataFrame) or not 1 <= len(data) <= 10000:
        c._error("Prediction requires a DataFrame with 1–10000 rows.", "resource_limit")
    state = _checked(result, max_bytes, max_work, len(data))
    variables = state["variables"]
    if missing not in ("drop", "raise") or any(list(data.columns).count(v) != 1 for v in variables):
        c._error("Select complete required columns and missing drop/raise.")
    _integer(max_bytes, "max_bytes", 1, c.LIMIT_BYTES)
    plan = _admit("saved spline prediction", 1024*len(data)*(len(variables)+32)+64*len(state["raw"])*(len(variables)+32),
                  128*len(data)*32**2, max_bytes, max_work)
    projected = data.loc[:, variables]
    absent = projected.isna().any(axis=1).tolist()
    if missing == "raise" and any(absent):
        c._error("Prediction contains missing values.", "missing_data")
    positions = [i for i, flag in enumerate(absent) if not flag]
    if not positions:
        c._error("No complete prediction rows.", "insufficient_sample")
    projected = projected.iloc[positions]
    for name in variables:
        if (not pd.api.types.is_numeric_dtype(projected[name]) or pd.api.types.is_bool_dtype(projected[name])
                or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or abs(v) > 1e100 for v in projected[name].tolist())):
            c._error("Prediction variables must be bounded finite real numbers.", "invalid_data")
    raw = torch.tensor(projected.to_numpy(dtype=float).tolist(), dtype=c.DT)
    train = torch.tensor(state["raw"], dtype=c.DT)
    coefficient = torch.tensor(state["coefficient"], dtype=c.DT)
    fitted = torch.full((len(positions),), float(torch.tensor(state["y"], dtype=c.DT).mean()), dtype=c.DT)
    offset = 0
    for j, name in enumerate(variables):
        knots = state["knots"][name]
        basis = c._basis(raw[:, j], knots)
        mean = c._basis(train[:, j], knots).mean(0)
        width = len(knots)-1
        fitted += (basis-mean)@coefficient[offset:offset+width]
        offset += width
    return _save(TableSet({"predictions": table(list(zip(positions, fitted.tolist())),
                           columns=["source_position", "fitted"])}, title="Saved spline CATREG prediction",
        method="catreg_spline_predict", source_method=state["method"], source_sha256=result.attrs["state_sha256"],
        n=len(positions), n_input=len(data), sample_positions=positions, n_missing=len(data)-len(positions),
        dtype="float64", device="cpu", weight_type="unweighted", resources=plan,
        inference="fitted numeric-response means conditional on saved learned spline maps"))
