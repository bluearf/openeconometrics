"""Declared-knot degree-one single-vector spline categorical PCA."""
from __future__ import annotations

import math
from itertools import combinations

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from . import spline_core as core
from .frequency import _admit, _save, _seal
from .frequency_pca import _components
from .frequency_rotation import _same, _tables_same
from .optimal import _integer, _real

DT, BYTES, WORK = core.DT, core.LIMIT_BYTES, core.WORK
METHODS = ("catpca_spline", "catpca_mspline")
SOURCES = [
    "https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=command-level-keyword-catpca",
    "https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/catpca.pdf",
]


def _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device):
    if device != "cpu":
        core._error("Spline PCA supports resident CPU float64 only.", "unsupported_option")
    return dict(n_starts=_integer(n_starts, "n_starts", 1, 12),
                seed=_integer(seed, "seed", 0, 2**31-1),
                maxiter=_integer(maxiter, "maxiter", 1, 1000),
                tol=_real(tol, "tol", 1e-12, 1e-3),
                max_work=_integer(max_work, "max_work", 1, WORK),
                max_bytes=_integer(max_bytes, "max_bytes", 1, BYTES))


def _plan(n, widths, d, controls, monotone, *, reuse=False, query_rows=0, state_size=0):
    p, total = len(widths), sum(widths)
    roots = sum((2**k if monotone else 1)*k**3 for k in widths)
    sweep = 64*n*(p*p+total*d)+256*roots
    work = (2 if reuse else controls["n_starts"]*(controls["maxiter"]+2))*sweep+64*query_rows*(total+p*d)+8*state_size
    storage = 512*n*(p+d+total)+256*controls["n_starts"]*(controls["maxiter"]+1)*8
    size = storage+4*state_size+256*query_rows*(total+p+d)+4096*sum(2**k*k*k if monotone else k*k for k in widths)
    return {**_admit("degree-one spline PCA "+("validation/projection" if reuse else "fit"),
                    size, work, controls["max_bytes"], controls["max_work"]), "planned_work": work}


def _geometry(bases, monotone):
    output = []
    for basis in bases:
        gram = basis.T@basis/len(basis)
        k = basis.shape[1]
        supports = [tuple(range(k))] if not monotone else [s for length in range(1, k+1) for s in combinations(range(k), length)]
        blocks = []
        for support in supports:
            idx = list(support)
            sub = gram[idx][:, idx]
            try:
                lower = torch.linalg.cholesky(sub)
            except torch.linalg.LinAlgError as exc:
                raise AnalysisError("rank_deficient", "Spline Gram geometry is too ill-conditioned for stable cone roots.") from exc
            whitening = torch.linalg.solve_triangular(lower.T, torch.eye(len(idx), dtype=DT), upper=True)
            blocks.append((idx, whitening))
        output.append((gram, blocks))
    return output


def _block(basis, scores, geometry, monotone, previous=None):
    """Finite face/eigenvector search for the constrained rank-one block.

    A nonnegative Rayleigh maximizer lies on a face where it is a generalized
    eigenvector. Degenerate eigenspaces admit a boundary maximizer, included
    by the smaller faces. At most 127 supports and seven roots are inspected.
    """
    gram, blocks = geometry
    cross = basis.T@scores/len(basis)
    operator = cross@cross.T
    best_value, best = -math.inf, None
    for indices, whiten in blocks:
        sub = operator[indices][:, indices]
        values, vectors = torch.linalg.eigh(whiten.T@sub@whiten)
        roots = range(len(indices)-1, -1, -1) if monotone else [len(indices)-1]
        for root in roots:
            candidate = whiten@vectors[:, root]
            if monotone:
                tolerance = 1e-10*max(1., float(candidate.abs().max()))
                if bool((candidate <= tolerance).all()):
                    candidate = -candidate
                if bool((candidate < -tolerance).any()):
                    continue
                candidate = candidate.clamp_min(0)
            elif float(candidate[int(torch.argmax(candidate.abs()))]) < 0:
                candidate = -candidate
            coefficient = torch.zeros(basis.shape[1], dtype=DT)
            coefficient[indices] = candidate
            norm = torch.sqrt(coefficient@gram@coefficient)
            if not bool(torch.isfinite(norm)) or float(norm) <= 1e-12:
                continue
            coefficient /= norm
            value = float(coefficient@operator@coefficient)
            if value > best_value+1e-14:
                best_value, best = value, coefficient
    if best is None:
        core._error("Spline block has no nondegenerate component direction.", "degenerate_transform")
    # Tied roots leave a continuum of valid scalar maps. Retaining an already
    # optimal feasible map prevents simultaneous blocks collapsing a valid
    # retained component space solely because of arbitrary eigenvector bases.
    if previous is not None:
        previous_value = float(previous@operator@previous)
        if best_value-previous_value <= 1e-12*max(1., abs(best_value)):
            best, best_value = previous, previous_value
    return best, basis@best, best_value


def _stationarity(bases, scores, coefficients, geometry, monotone):
    gains = []
    for basis, coefficient, block_geometry in zip(bases, coefficients, geometry):
        _, _, optimum = _block(basis, scores, block_geometry, monotone)
        loading = scores.T@(basis@coefficient)/len(basis)
        gains.append(max(0., optimum-float(loading@loading)))
    return max(gains)


def _tables(state, raw, quantified, scores, loadings, axes, eigenvalues):
    variables, d, positions = state["variables"], state["components"], state["positions"]
    names = [f"component_{j+1}" for j in range(d)]
    rows = []
    for name, coefficient, means in zip(variables, state["coefficients"], state["basis_means"]):
        knots = state["knots"][name]
        for j, (left, right, value, mean) in enumerate(zip(knots, knots[1:], coefficient, means)):
            rows.append([name, j, left, right, value, mean])
    objective = float((quantified-scores@loadings.T).square().sum()/len(raw))
    # An early failed start has no evaluated objective. A text marker preserves
    # that fact without pandas promoting nulls to nonfinite numeric NaNs.
    starts = table([[row[0], row[1], row[2], "not evaluated" if row[3] is None else row[3], row[4]]
                    for row in state["starts"]], columns=["start", "converged", "iterations", "objective", "status"])
    return {
        "fit": table([[len(raw), d, objective, 1-objective/len(variables)]], columns=["n", "components", "reconstruction_loss", "variance_proportion"]),
        "raw": table([[pos]+row for pos, row in zip(positions, raw.tolist())], columns=["source_position"]+variables),
        "transformed": table([[pos]+row for pos, row in zip(positions, quantified.tolist())], columns=["source_position"]+variables),
        "scores": table([[pos]+row for pos, row in zip(positions, scores.tolist())], columns=["source_position"]+names),
        "loadings": table(loadings.tolist(), columns=names, index=variables),
        "eigenvectors": table(axes.tolist(), columns=names, index=variables),
        "eigenvalues": table([[j+1, float(v), float(v)/len(variables)] for j, v in enumerate(eigenvalues)], columns=["component", "eigenvalue", "proportion"]),
        "splines": table(rows, columns=["variable", "segment", "left", "right", "coefficient", "basis_mean"]),
        "iterations": table(state["histories"], columns=["start", "iteration", "objective", "improvement", "maximum_block_gain"]),
        "starts": starts,
    }


def _output(state, raw, quantified, scores, loadings, axes, eigenvalues, plan):
    output = TableSet(_tables(state, raw, quantified, scores, loadings, axes, eigenvalues),
        title="Monotone degree-one spline PCA" if state["monotone"] else "Degree-one spline PCA",
        method=state["kind"], spline_state=state, state_sha256=_seal(state),
        variables=state["variables"], components=state["components"], n=len(raw), n_input=state["input_nobs"],
        sample_positions=state["positions"], missing=state["missing"], settings=state["controls"],
        chosen_start=state["chosen_start"], iterations=state["iterations"], converged=True,
        degree=1, monotone=state["monotone"], resources=plan, sources=SOURCES,
        device="cpu", dtype="float64", weight_type="unweighted",
        normalization="centered population-variance-one continuous ramp maps; X'X/n=I",
        inference="descriptive local ALS solution; declared knots only; no uncertainty or vendor parity",
        support="closed caller-declared endpoints; query extrapolation refused")
    return _save(output)


def _fit(data, variables, knots, components, controls, missing, monotone):
    if (not isinstance(variables, list) or not 2 <= len(variables) <= 6
            or any(type(name) is not str or not 1 <= len(name) <= 128 for name in variables)
            or len(set(variables)) != len(variables)):
        core._error("Spline PCA requires 2–6 named variables.")
    d = _integer(components, "components", 1, len(variables)-1)
    admitted_knots = core._knots(knots, variables)
    if not isinstance(data, pd.DataFrame) or not 4 <= len(data) <= 3000:
        core._error("Spline PCA requires 4–3000 physical DataFrame rows.", "resource_limit")
    widths = [len(admitted_knots[name])-1 for name in variables]
    plan = _plan(len(data), widths, d, controls, monotone)
    prepared = core._prepare(data, variables, admitted_knots, missing, controls["max_bytes"], controls["max_work"], "spline PCA")
    raw, bases = prepared["raw"], prepared["bases"]
    n = len(raw)
    if n <= d+2:
        core._error("Complete rows must exceed components plus two.", "insufficient_sample")
    geometry = _geometry(bases, monotone)
    generator = torch.Generator(device="cpu").manual_seed(controls["seed"])
    weight = torch.ones(n, dtype=DT)
    histories, starts, candidates = [], [], []
    tolerance = max(1e-10, controls["tol"])
    for start in range(controls["n_starts"]):
        iteration, objective = 0, None
        try:
            coefficients = [torch.ones(b.shape[1], dtype=DT) if start == 0 else torch.randn(b.shape[1], dtype=DT, generator=generator) for b in bases]
            if monotone:
                coefficients = [c.abs()+.05 for c in coefficients]
            coefficients = [c/(b@c).square().mean().sqrt() for b, c in zip(bases, coefficients)]
            quantified = torch.stack([b@c for b, c in zip(bases, coefficients)], 1)
            scores, loadings, axes, eigenvalues, objective = _components(quantified, d, weight)
            gain = _stationarity(bases, scores, coefficients, geometry, monotone)
            histories.append([start, 0, objective, 0., gain])
            converged = False
            for iteration in range(1, controls["maxiter"]+1):
                previous = objective
                updates = [_block(b, scores, g, monotone, c) for b, g, c in zip(bases, geometry, coefficients)]
                coefficients = [item[0] for item in updates]
                quantified = torch.stack([item[1] for item in updates], 1)
                scores, loadings, axes, eigenvalues, objective = _components(quantified, d, weight)
                gain = _stationarity(bases, scores, coefficients, geometry, monotone)
                improvement = previous-objective
                histories.append([start, iteration, objective, improvement, gain])
                if improvement < -1e-9*max(1., previous):
                    core._error("Spline PCA objective increased beyond roundoff.", "numerical_failure")
                if abs(improvement) <= controls["tol"]*max(1., previous) and gain <= tolerance:
                    converged = True
                    break
            starts.append([start, converged, iteration, objective, "accepted" if converged else "nonconvergence"])
            if converged:
                candidates.append((objective, start, iteration, coefficients, quantified, scores, loadings, axes, eigenvalues))
        except AnalysisError as exc:
            starts.append([start, False, iteration, objective, exc.code])
    if not candidates:
        core._error("No spline PCA start reached objective and block-stationarity tolerance.", "nonconvergence")
    objective, chosen, iteration, coefficients, quantified, scores, loadings, axes, eigenvalues = min(candidates, key=lambda item: (item[0], item[1]))
    state = dict(version=1, kind=METHODS[int(monotone)], variables=variables, knots=prepared["knots"],
                 components=d, monotone=monotone, raw=raw.tolist(), positions=prepared["positions"],
                 input_nobs=len(data), coefficients=[c.tolist() for c in coefficients],
                 basis_means=[v.tolist() for v in prepared["basis_means"]], axes=axes.tolist(),
                 eigenvalues=eigenvalues.tolist(), histories=histories, starts=starts,
                 controls=controls, chosen_start=chosen, iterations=iteration, missing=missing)
    core._metadata_bound(state)
    return _output(state, raw, quantified, scores, loadings, axes, eigenvalues, plan)


@resident_cpu
def catpca_spline(data: pd.DataFrame, variables: list[str], *, knots: dict,
                  components: int = 2, n_starts: int = 3, seed: int = 0,
                  maxiter: int = 500, tol: float = 1e-8, missing: str = "drop",
                  max_work: int = WORK, max_bytes: int = BYTES,
                  device: str = "cpu") -> TableSet:
    """Single-vector CATPCA with freely signed continuous degree-one splines.

    Every numeric variable has 2–8 caller-declared support knots. The best
    converged local ALS start is retained; no automatic knots or extrapolation.
    """
    return _fit(data, variables, knots, components,
                _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device), missing, False)


@resident_cpu
def catpca_mspline(data: pd.DataFrame, variables: list[str], *, knots: dict,
                   components: int = 2, n_starts: int = 3, seed: int = 0,
                   maxiter: int = 500, tol: float = 1e-8, missing: str = "drop",
                   max_work: int = WORK, max_bytes: int = BYTES,
                   device: str = "cpu") -> TableSet:
    """Single-vector CATPCA with continuous nondecreasing degree-one maps.

    Nonnegative segment increments are solved on finite bounded cone faces.
    Each block is exact; the joint alternating solution remains local.
    """
    return _fit(data, variables, knots, components,
                _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device), missing, True)


def _trace(state, objective, gain):
    controls, starts, histories = state["controls"], state["starts"], state["histories"]
    if not isinstance(starts, list) or len(starts) != controls["n_starts"] or any(not isinstance(row, list) or len(row) != 5 for row in starts):
        core._error("Saved start ledger has invalid dimensions.", "invalid_state")
    if not isinstance(histories, list) or len(histories) > controls["n_starts"]*(controls["maxiter"]+1):
        core._error("Saved iteration ledger exceeds its bound.", "invalid_state")
    if any(not isinstance(row, list) or len(row) != 5 or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
           or type(row[0]) is not int or not 0 <= row[0] < controls["n_starts"] or type(row[1]) is not int for row in histories):
        core._error("Saved iteration ledger must be finite numeric rows.", "invalid_state")
    accepted = []
    for number, row in enumerate(starts):
        start, converged, iterations, value, status = row
        if type(start) is not int or start != number or type(converged) is not bool or type(iterations) is not int or not 0 <= iterations <= controls["maxiter"] or status not in ("accepted", "nonconvergence", "rank_deficient", "degenerate_transform", "numerical_failure"):
            core._error("Saved start controls are invalid.", "invalid_state")
        trace = [r for r in histories if r[0] == number]
        if trace and trace[0][3] != 0 or value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            core._error("Saved start objective is invalid.", "invalid_state")
        if any(r[1] != j or r[2] < -1e-10 or r[4] < 0 or j and (r[3] < -1e-9 or abs(r[3]-(trace[j-1][2]-r[2])) > 1e-8) for j, r in enumerate(trace)):
            core._error("Saved objective history is inconsistent.", "invalid_state")
        if converged:
            if status != "accepted" or len(trace) != iterations+1 or iterations < 1 or abs(value-trace[-1][2]) > 1e-8 or abs(trace[-1][3]) > controls["tol"]*max(1., trace[-2][2])+1e-10 or trace[-1][4] > max(1e-10, controls["tol"])+1e-10:
                core._error("Saved accepted start does not satisfy its stopping rule.", "invalid_state")
            accepted.append((value, number))
    if type(state["chosen_start"]) is not int or type(state["iterations"]) is not int or not accepted or state["chosen_start"] != min(accepted)[1]:
        core._error("Saved selected start is not the best accepted solution.", "invalid_state")
    chosen = starts[state["chosen_start"]]
    final = [r for r in histories if r[0] == state["chosen_start"]][-1]
    if state["iterations"] != chosen[2] or abs(objective-chosen[3]) > 1e-8 or abs(gain-final[4]) > 1e-8:
        core._error("Saved selected geometry and trace disagree.", "invalid_state")


def _checked(result, max_bytes, max_work, device, query_rows=0):
    if device != "cpu" or not isinstance(result, TableSet) or result.attrs.get("method") not in METHODS:
        core._error("Supply complete saved spline PCA on CPU.", "invalid_result")
    state = result.attrs.get("spline_state")
    fields = {"version", "kind", "variables", "knots", "components", "monotone", "raw", "positions", "input_nobs", "coefficients", "basis_means", "axes", "eigenvalues", "histories", "starts", "controls", "chosen_start", "iterations", "missing"}
    if not isinstance(state, dict) or set(state) != fields:
        core._error("Saved spline PCA schema is incomplete or unknown.", "invalid_state")
    estimate = core._metadata_bound(result.attrs)
    for frame in result.values():
        if not isinstance(frame, pd.DataFrame) or len(frame) > 12012 or len(frame.columns) > 10:
            core._error("Saved spline PCA tables exceed their bounds.", "resource_limit")
        estimate += 64*frame.size
    _admit("spline PCA bounded state parsing", 4*estimate, 8*estimate, max_bytes, max_work)
    try:
        if type(state["version"]) is not int or state["version"] != 1 or type(state["monotone"]) is not bool or state["kind"] != METHODS[int(state["monotone"])] or _seal(state) != result.attrs["state_sha256"]:
            raise ValueError("Saved identity/integrity")
        variables, n, p, d = state["variables"], len(state["positions"]), len(state["variables"]), state["components"]
        if not 2 <= p <= 6 or not 4 <= n <= 3000 or type(d) is not int or not 1 <= d < p:
            raise ValueError("Dimensions")
        if type(state["input_nobs"]) is not int or not n <= state["input_nobs"] <= 3000 or any(type(v) is not int or not 0 <= v < state["input_nobs"] for v in state["positions"]) or state["positions"] != sorted(set(state["positions"])):
            raise ValueError("Positions")
        if not isinstance(state["raw"], list) or len(state["raw"]) != n or any(not isinstance(row, list) or len(row) != p or any(type(v) not in (int, float) or not math.isfinite(v) for v in row) for row in state["raw"]):
            raise ValueError("Raw dimensions")
        controls = _controls(**state["controls"], device=device)
        widths = [len(state["knots"][name])-1 for name in variables]
        limits = {**controls, "max_bytes": max_bytes, "max_work": max_work}
        plan = _plan(n, widths, d, limits, state["monotone"], reuse=True, query_rows=query_rows, state_size=estimate)
        prepared = core._prepare(pd.DataFrame(state["raw"], columns=variables), variables, state["knots"], "raise", max_bytes, max_work, "saved spline PCA")
        raw, bases = prepared["raw"], prepared["bases"]
        if not isinstance(state["coefficients"], list) or len(state["coefficients"]) != p or any(not isinstance(c, list) or len(c) != k or any(type(v) not in (int, float) or not math.isfinite(v) for v in c) for c, k in zip(state["coefficients"], widths)):
            raise ValueError("Coefficient dimensions")
        coefficients = [torch.tensor(c, dtype=DT) for c in state["coefficients"]]
        if state["monotone"] and any(bool((c < 0).any()) for c in coefficients):
            raise ValueError("Negative monotone coefficient")
        _same(state["basis_means"], [v.tolist() for v in prepared["basis_means"]], "basis means")
        quantified = torch.stack([b@c for b, c in zip(bases, coefficients)], 1)
        core._close(quantified.mean(0), torch.zeros(p, dtype=DT), "centering")
        core._close(quantified.square().mean(0), torch.ones(p, dtype=DT), "normalization")
        if (not isinstance(state["axes"], list) or len(state["axes"]) != p
                or any(not isinstance(row, list) or len(row) != d
                       or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
                       for row in state["axes"])
                or not isinstance(state["eigenvalues"], list) or len(state["eigenvalues"]) != min(n, p)
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in state["eigenvalues"])
                or min(state["eigenvalues"][:d]) <= 1e-20):
            raise ValueError("Saved component geometry dimensions")
        # Preserve the fitted basis inside tied eigenspaces. A fresh SVD may
        # choose different valid axes; only the saved leading geometry governs
        # the fixed-map projection after restoration.
        axes = torch.tensor(state["axes"], dtype=DT)
        eigenvalues = torch.tensor(state["eigenvalues"], dtype=DT)
        covariance = quantified.T@quantified/n
        core._close(axes.T@axes, torch.eye(d, dtype=DT), "orthogonal axes")
        core._close(covariance@axes, axes*eigenvalues[:d][None, :], "saved eigensystem")
        core._close(eigenvalues, torch.linalg.eigvalsh(covariance).flip(0)[:min(n, p)], "leading spectrum")
        scores = quantified@axes/eigenvalues[:d].sqrt()[None, :]
        loadings = axes*eigenvalues[:d].sqrt()[None, :]
        core._close(scores.T@scores/n, torch.eye(d, dtype=DT), "score covariance")
        objective = float((quantified-scores@loadings.T).square().sum()/n)
        geometry = _geometry(bases, state["monotone"])
        gain = _stationarity(bases, scores, coefficients, geometry, state["monotone"])
        if gain > max(1e-10, controls["tol"])+1e-9:
            raise ValueError("Not block stationary")
        _trace(state, objective, gain)
        expected_plan = _plan(state["input_nobs"], widths, d, controls, state["monotone"])
        expected = _output(state, raw, quantified, scores, loadings, axes, eigenvalues, expected_plan)
        _same(dict(result.attrs), dict(expected.attrs), "spline attrs")
        _tables_same(result, expected)
        return state, plan
    except AnalysisError as exc:
        if exc.code in ("resource_limit", "workspace_limit", "invalid_state"):
            raise
        raise AnalysisError("invalid_state", "Saved spline PCA options or calibration are invalid.") from exc
    except (TypeError, KeyError, ValueError, OverflowError, torch.linalg.LinAlgError) as exc:
        raise AnalysisError("invalid_state", "Saved spline PCA state is inconsistent.") from exc


@resident_cpu
def catpca_spline_predict(result: TableSet, data: pd.DataFrame, *, missing: str = "raise",
                          max_work: int = WORK, max_bytes: int = BYTES,
                          device: str = "cpu") -> TableSet:
    """Project through saved continuous spline maps within declared knot spans.

    Numerical state, full sample and local block optimality are validated. The
    query does not refit coefficients, knots, centering, or component axes.
    """
    if not isinstance(data, pd.DataFrame) or not 0 <= len(data) <= 3000 or missing not in ("raise", "drop"):
        core._error("Queries require at most 3000 rows and missing drop/raise.")
    state, plan = _checked(result, max_bytes, max_work, device, len(data))
    variables = state["variables"]
    if any(list(data.columns).count(name) != 1 for name in variables):
        core._error("Every saved variable must occur exactly once.")
    frame = data[variables]
    complete = ~frame.isna().any(axis=1)
    if missing == "raise" and not bool(complete.all()):
        core._error("Spline query contains missing values.", "missing_data")
    positions = [i for i, keep in enumerate(complete.tolist()) if keep]
    frame = frame.iloc[positions]
    columns = []
    for j, name in enumerate(variables):
        values = frame[name]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values) or pd.api.types.is_complex_dtype(values):
            core._error("Spline queries require real numeric values.", "invalid_data")
        raw = torch.tensor(values.tolist(), dtype=DT)
        basis = core._basis(raw, state["knots"][name])
        columns.append((basis-torch.tensor(state["basis_means"][j], dtype=DT))@torch.tensor(state["coefficients"][j], dtype=DT))
    quantified = torch.stack(columns, dim=1)
    axes = torch.tensor(state["axes"], dtype=DT)
    roots = torch.tensor(state["eigenvalues"][:state["components"]], dtype=DT).sqrt()
    scores = quantified@axes/roots[None, :]
    reconstructed = scores@(axes*roots[None, :]).T
    names = [f"component_{j+1}" for j in range(state["components"])]
    return _save(TableSet({
        "scores": table([[pos]+row for pos, row in zip(positions, scores.tolist())], columns=["source_position"]+names),
        "transformed": table([[pos]+row for pos, row in zip(positions, quantified.tolist())], columns=["source_position"]+variables),
        "reconstruction": table([[pos]+row for pos, row in zip(positions, reconstructed.tolist())], columns=["source_position"]+variables),
    }, title="Saved spline PCA projection", method="catpca_spline_predict", n=len(positions),
       n_input=len(data), sample_positions=positions, missing=missing, resources=plan,
       source_state_sha256=result.attrs["state_sha256"], device="cpu", dtype="float64", weight_type="unweighted",
       inference="descriptive fixed-map projection within declared support; no adaptive uncertainty"))
