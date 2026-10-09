"""Frequency-weighted single-vector CATPCA and original-category centroids.

Frequency counts represent literal row replication, without expanding rows.
The scalar optimal-scaling maps are descriptive, locally optimized geometry.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import saved_summary
from .frequency import (
    BYTES, WORK, DT, _admit, _atom, _means, _normalize_categories,
    _prepare, _sample, _save, _seal, _weights,
)
from .optimal import _error, _integer, _label, _mapping_tables, _names, _pava, _real

SOURCES = [
    "https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/catpca.pdf",
    "https://www.jstatsoft.org/article/view/v031i04",
]
STATE_BYTES = 8 * 1024**2


def _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device):
    if device != "cpu":
        _error("Frequency CATPCA supports resident CPU float64 only.", "unsupported_option")
    return dict(n_starts=_integer(n_starts, "n_starts", 1, 12),
                seed=_integer(seed, "seed", 0, 2**31-1),
                maxiter=_integer(maxiter, "maxiter", 1, 1000),
                tol=_real(tol, "tol", 1e-12, 1e-3),
                max_work=_integer(max_work, "max_work", 1, WORK),
                max_bytes=_integer(max_bytes, "max_bytes", 1, BYTES))


def _components(quantified, dimensions, weight):
    total = weight.sum()
    _, singular, vh = torch.linalg.svd(quantified*(weight/total).sqrt()[:, None], full_matrices=False)
    if float(singular[dimensions-1]) <= 1e-10*float(singular[0]):
        _error("Requested weighted component space is rank-deficient.", "rank_deficient")
    axes = vh[:dimensions].T.clone()
    for k in range(dimensions):
        if float(axes[int(torch.argmax(axes[:, k].abs())), k]) < 0:
            axes[:, k] *= -1
    eigenvalues = singular.square()
    scores = quantified@axes/singular[:dimensions][None, :]
    loadings = axes*singular[:dimensions][None, :]
    objective = float(((quantified-scores@loadings.T).square()*weight[:, None]).sum()/total)
    return scores, loadings, axes, eigenvalues, objective


def _initial(descriptors, prepared, start, generator):
    columns = []
    for descriptor, values in zip(descriptors, prepared):
        if descriptor["scale"] == "numeric":
            columns.append(values.clone())
            continue
        count = torch.tensor(descriptor["counts"], dtype=DT)
        score = torch.arange(len(count), dtype=DT) if start == 0 else torch.randn(len(count), generator=generator, dtype=DT)
        if descriptor["scale"] == "ordinal":
            score = torch.sort(score).values
        score = _normalize_categories(score, count, nominal=descriptor["scale"] == "nominal")
        columns.append(score[values])
    return torch.stack(columns, dim=1)


def _update(scores, codes, descriptor, previous, weight):
    centroids, counts = _means(scores, codes, len(descriptor["levels"]), weight)
    if descriptor["scale"] == "nominal":
        left, singular, _ = torch.linalg.svd(centroids*counts.sqrt()[:, None], full_matrices=False)
        if float(singular[0]) <= 1e-12:
            _error("Weighted category centroids have no nondegenerate direction.", "degenerate_transform")
        return _normalize_categories(left[:, 0]/counts.sqrt(), counts, nominal=True)[codes]
    quantification, _ = _means(previous, codes, len(counts), weight)
    for _ in range(100):
        loading = scores.T@(weight*quantification[codes])/weight.sum()
        next_q = _normalize_categories(_pava(centroids@loading, counts), counts)
        difference = float((quantification-next_q).square().max())
        quantification = next_q
        if difference <= 1e-16:
            return quantification[codes]
    _error("Weighted ordinal inner quantification did not converge.", "nonconvergence")


@resident_cpu
def catpca_fweight(data: Any, variables: list[str], *, frequency: str,
                  components: int = 2, scales: dict | None = None,
                  orders: dict | None = None, n_starts: int = 3, seed: int = 0,
                  maxiter: int = 500, tol: float = 1e-8, missing: str = "drop",
                  max_work: int = WORK, max_bytes: int = BYTES,
                  device: str = "cpu") -> TableSet:
    """Single-vector CATPCA with exact integer replication frequencies.

    Counts and their total are at most 1e9; at most 3000 physical rows are
    admitted. Zero-count rows are excluded. Missing counts/variables obey
    missing='drop' or 'raise'. Scores satisfy X'WX/sum(w)=I. Retained typed
    memberships permit original-category centroids even when maps pool ties.
    """
    variables = _names(variables, 2)
    dimensions = _integer(components, "components", 1, len(variables)-1)
    controls = _controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device)
    n_starts, seed, maxiter, tol, max_work, max_bytes = (
        controls[name] for name in ("n_starts", "seed", "maxiter", "tol", "max_work", "max_bytes"))
    sample = _sample(data, variables, frequency, missing, max_bytes, max_work,
                     operation="frequency CATPCA input", min_rows=dimensions+3)
    frame, positions, counts = sample["frame"], sample["positions"], sample["counts"]
    n, p = len(frame), len(variables)
    # Includes weighted SVD, all ordinal inner loops, multi-start candidates,
    # histories, saved memberships and published table cells before tensors.
    scale_map = scales if isinstance(scales, dict) else {}
    level_sizes = [len({str(_label(_atom(x))) for x in frame[name].tolist()})
                   if scale_map.get(name, "nominal") != "numeric" else 0 for name in variables]
    work_per_iteration = 16*n*p*p+64*n*p
    for name, levels in zip(variables, level_sizes):
        if levels:
            work_per_iteration += 64*n*dimensions+64*levels*dimensions**2
        if scale_map.get(name) == "ordinal":
            work_per_iteration += 100*(4*n*dimensions+32*levels)
    work = n_starts*(maxiter+2)*work_per_iteration
    label_space = sum(1536+8*len(token) for name in variables
                      if scale_map.get(name, "nominal") != "numeric"
                      for token in {str(_label(_atom(x))) for x in frame[name].tolist()})
    # Mirror the conservative reusable-state cell/node domain before fitting.
    # Both raw membership/numeric values and all physical-row tables persist.
    portable_bound = 65536+64*n*(2*p+dimensions+10)+2048*n_starts*(maxiter+1)+label_space
    if portable_bound > STATE_BYTES:
        _error("Declared complete CATPCA state exceeds the 8 MiB reusable domain.", "resource_limit")
    plan = _admit("frequency CATPCA", 8*n*(32*p+20*dimensions+8)+
                  64*(n_starts+1)*(n*(3*p+2*dimensions)+maxiter*8)+4096*p*32,
                  work, max_bytes, max_work)
    weight = _weights(counts)
    descriptors, prepared = _prepare(frame, variables, scales, orders, "nominal", weight)
    for descriptor in descriptors:
        if descriptor["scale"] != "numeric":
            descriptor["levels"] = [_label(value) for value in descriptor["levels"]]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    histories, starts, candidates = [], [], []
    for start in range(n_starts):
        iteration, objective = 0, None
        try:
            quantified = _initial(descriptors, prepared, start, generator)
            scores, loadings, axes, eigenvalues, objective = _components(quantified, dimensions, weight)
            histories.append([start, 0, objective, 0.0])
            converged = False
            for iteration in range(1, maxiter+1):
                old = objective
                for j, (descriptor, values) in enumerate(zip(descriptors, prepared)):
                    if descriptor["scale"] != "numeric":
                        quantified[:, j] = _update(scores, values, descriptor, quantified[:, j], weight)
                scores, loadings, axes, eigenvalues, objective = _components(quantified, dimensions, weight)
                improvement = old-objective
                histories.append([start, iteration, objective, improvement])
                if improvement < -1e-9:
                    _error("Weighted CATPCA objective increased beyond roundoff.", "numerical_failure")
                if abs(improvement) <= tol*max(1.0, old):
                    converged = True
                    break
            if not converged:
                starts.append([start, False, iteration, objective, "nonconvergence"])
                continue
            starts.append([start, True, iteration, objective, "accepted"])
            candidates.append((objective, start, quantified.clone(), scores.clone(), loadings.clone(), axes.clone(), eigenvalues.clone(), iteration))
        except AnalysisError as exc:
            starts.append([start, False, iteration, objective if objective is not None else "unavailable", exc.code])
    if not candidates:
        _error("No weighted CATPCA start converged to nondegenerate geometry.", "nonconvergence")
    objective, chosen, quantified, scores, loadings, axes, eigenvalues, iterations = min(candidates, key=lambda candidate: candidate[:2])
    membership, numeric_values = {}, {}
    for j, (descriptor, values) in enumerate(zip(descriptors, prepared)):
        if descriptor["scale"] != "numeric":
            means, _ = _means(quantified[:, j], values, len(descriptor["levels"]), weight)
            descriptor["quantifications"] = means.tolist()
            descriptor["counts"] = [int(x) for x in descriptor["counts"]]
            membership[descriptor["name"]] = values.tolist()
        else:
            numeric_values[descriptor["name"]] = [float(x) for x in frame[descriptor["name"]].tolist()]
    state = dict(version=1, kind="catpca_fweight", variables=variables, descriptors=descriptors,
                 components=dimensions, axes=axes.tolist(), eigenvalues=eigenvalues[:dimensions].tolist(),
                 membership=membership, numeric_values=numeric_values, frequencies=counts, positions=positions,
                 frequency=frequency, frequency_total=sample["frequency_total"], input_nobs=sample["input_nobs"],
                 zero_positions=sample["zero_positions"], missing_positions=sample["missing_positions"])
    names = [f"component_{j+1}" for j in range(dimensions)]
    output = TableSet({
        "fit": table([[n, sample["frequency_total"], dimensions, objective, 1-objective/p]],
                     columns=["physical_n", "frequency_total", "components", "reconstruction_loss", "variance_proportion"]),
        "sample": table([[position, count] for position, count in zip(positions, counts)], columns=["source_position", "frequency"]),
        "scores": table([[position]+row for position, row in zip(positions, scores.tolist())], columns=["source_position"]+names),
        "transformed": table([[position]+row for position, row in zip(positions, quantified.tolist())], columns=["source_position"]+variables),
        "loadings": table(loadings.tolist(), columns=names, index=variables),
        "eigenvectors": table(axes.tolist(), columns=names, index=variables),
        "eigenvalues": table([[j+1, value, value/p] for j, value in enumerate(eigenvalues.tolist())], columns=["component", "eigenvalue", "proportion"]),
        "iterations": table(histories, columns=["start", "iteration", "objective", "improvement"]),
        "starts": table(starts, columns=["start", "converged", "iterations", "objective", "status"]),
        **_mapping_tables(descriptors),
    }, title="Frequency-weighted single-vector categorical PCA", method="catpca_fweight",
       frequency_state=state, state_sha256=_seal(state), n=n, n_input=sample["input_nobs"],
       frequency_total=sample["frequency_total"], frequency=frequency, sample_positions=positions,
       zero_positions=sample["zero_positions"], missing_positions=sample["missing_positions"],
       variables=variables, components=dimensions, chosen_start=chosen, iterations=iterations,
       converged=True, settings=controls, resources=plan, input_preparation=sample["workspace"],
       declared_work=work, missing=missing, sources=SOURCES, device="cpu", dtype="float64",
       weight_type="frequency", inference="descriptive reconstruction only; no adaptive uncertainty",
       solution="best converged declared deterministic start; local optimum only",
       normalization="weighted population-standardized columns; X'WX/sum(w)=I; single-vector maps")
    output = _save(output)
    # A successful fit must also satisfy its downstream bounded state schema.
    # This primitive-only pass allocates no new model tensor or serialization.
    _preflight(output, BYTES, WORK)
    return output


def _close(actual, expected, what, *, atol=2e-8):
    if actual.shape != expected.shape or not bool(torch.isfinite(actual).all()) or not torch.allclose(actual, expected, atol=atol, rtol=2e-8):
        _error(f"Saved frequency CATPCA {what} is inconsistent.", "invalid_state")


def _preflight(result, max_bytes, max_work):
    if not isinstance(result, TableSet) or result.attrs.get("method") != "catpca_fweight":
        _error("Supply a fitted/restored frequency CATPCA result retaining original memberships.", "invalid_result")
    expected_tables = {"fit", "sample", "scores", "transformed", "loadings", "eigenvectors", "eigenvalues", "iterations", "starts", "quantifications", "numeric_scaling", "settings"}
    if set(result) != expected_tables:
        _error("Saved frequency CATPCA table collection is incomplete or unknown.", "invalid_state")
    state = result.attrs.get("frequency_state")
    expected_state = {"version", "kind", "variables", "descriptors", "components", "axes", "eigenvalues",
                      "membership", "numeric_values", "frequencies", "positions", "frequency",
                      "frequency_total", "input_nobs", "zero_positions", "missing_positions"}
    if not isinstance(state, dict) or set(state) != expected_state:
        _error("Complete original membership state is absent or has unknown fields.", "invalid_state")
    try:
        n, p, d = len(state["positions"]), len(state["variables"]), state["components"]
        if not 4 <= n <= 3000 or not 2 <= p <= 12 or type(d) is not int or not 1 <= d < p or len(result) > 16:
            raise ValueError("Invalid shape")
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("invalid_state", "Saved frequency CATPCA dimensions are invalid.") from exc
    estimate, nodes = 0, 0
    pending = [(result.attrs, 0)]
    while pending:
        value, depth = pending.pop()
        nodes += 1
        estimate += 64
        if depth > 16 or nodes > 120000:
            _error("Saved metadata exceeds the bounded state domain.", "resource_limit")
        if isinstance(value, dict):
            if len(value) > 12012:
                _error("Saved metadata dimensions are excessive.", "resource_limit")
            pending.extend((x, depth+1) for pair in value.items() for x in pair)
        elif isinstance(value, (list, tuple)):
            if len(value) > 12012:
                _error("Saved metadata dimensions are excessive.", "resource_limit")
            pending.extend((x, depth+1) for x in value)
        elif value is not None and type(value) not in (str, int, float, bool):
            _error("Saved metadata must contain finite primitive JSON values.", "invalid_state")
        elif type(value) is int and abs(value) > 2**53-1:
            _error("Saved metadata integer is outside the exact JSON domain.", "invalid_state")
        elif type(value) is float and not math.isfinite(value):
            _error("Saved metadata contains nonfinite values.", "invalid_state")
        elif isinstance(value, str):
            if len(value) > 4096:
                _error("Saved metadata text is excessive.", "resource_limit")
            estimate += len(value)*4
        if estimate > STATE_BYTES:
            _error("Saved metadata exceeds 8 MiB.", "resource_limit")
    for frame in result.values():
        if not isinstance(frame, pd.DataFrame) or len(frame) > 12012 or len(frame.columns) > 16:
            _error("Saved table dimensions exceed the bounded state domain.", "resource_limit")
        for label in [*frame.index, *frame.columns]:
            if hasattr(label, "item") and type(label).__module__.startswith("numpy"):
                label = label.item()
            if type(label) not in (str, int, float, bool) or isinstance(label, str) and len(label) > 256 or type(label) is int and abs(label) > 2**53-1 or type(label) is float and not math.isfinite(label):
                _error("Saved table labels must be bounded finite primitives.", "invalid_state")
        estimate += 64*(frame.size+len(frame.index)+len(frame.columns))
        if estimate > STATE_BYTES:
            _error("Saved tables exceed the bounded state domain.", "resource_limit")
        for row in frame.itertuples(index=False, name=None):
            for cell in row:
                if hasattr(cell, "item") and type(cell).__module__.startswith("numpy"):
                    cell = cell.item()
                if cell is not None and type(cell) not in (str, int, float, bool):
                    _error("Saved table cells must contain finite primitive JSON values.", "invalid_state")
                if type(cell) is int and abs(cell) > 2**53-1 or type(cell) is float and not math.isfinite(cell):
                    _error("Saved table numbers exceed the finite exact JSON domain.", "invalid_state")
                if isinstance(cell, str) and len(cell) > 4096:
                    _error("Saved table text is excessive.", "resource_limit")
                if isinstance(cell, str):
                    estimate += len(cell)*4
                if estimate > STATE_BYTES:
                    _error("Saved table text exceeds the state domain.", "resource_limit")
    plan = _admit("frequency CATPCA state validation", 4*estimate+8*n*(12*p+12*d),
                  64*n*p*p+64*n*p*d, max_bytes, max_work)
    return state, n, p, d, plan


def _frame_tensor(result, name, columns, rows, *, index=None):
    frame = result.get(name)
    if frame is None or list(frame.columns) != columns or len(frame) != rows or (index is not None and list(frame.index) != index):
        _error(f"Saved {name} table has invalid dimensions or labels.", "invalid_state")
    try:
        values = torch.tensor(frame.to_numpy(dtype=float).tolist(), dtype=DT)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError("invalid_state", f"Saved {name} must be finite numeric geometry.") from exc
    if not bool(torch.isfinite(values).all()):
        _error(f"Saved {name} contains nonfinite geometry.", "invalid_state")
    return values


def _validate_trace(result, state, objective):
    """Validate declared starts, convergence and the published selected solution."""
    try:
        attrs = result.attrs
        settings = attrs["settings"]
        controls = _controls(settings["n_starts"], settings["seed"], settings["maxiter"],
                             settings["tol"], settings["max_work"], settings["max_bytes"], "cpu")
        if settings != controls:
            raise ValueError("Invalid controls")
        starts = result["starts"]
        history = result["iterations"]
        if list(starts.columns) != ["start", "converged", "iterations", "objective", "status"] or len(starts) != controls["n_starts"] or list(history.columns) != ["start", "iteration", "objective", "improvement"]:
            raise ValueError("Invalid trace tables")
        grouped = {start: [] for start in range(controls["n_starts"])}
        for start, iteration, value, improvement in history.to_numpy().tolist():
            if type(start) not in (int, float) or int(start) != start or start not in grouped or type(iteration) not in (int, float) or int(iteration) != iteration or not 0 <= iteration <= controls["maxiter"] or type(value) not in (int, float) or type(improvement) not in (int, float) or not math.isfinite(value) or not math.isfinite(improvement) or not -1e-8 <= value <= len(state["variables"])+1e-8:
                raise ValueError("Invalid iteration values")
            grouped[int(start)].append((int(iteration), value, improvement))
        accepted = []
        for expected, row in enumerate(starts.to_numpy().tolist()):
            start, converged, iterations, value, status = row
            if type(start) is not int or start != expected or type(converged) is not bool or type(iterations) is not int or not 0 <= iterations <= controls["maxiter"] or not isinstance(status, str):
                raise ValueError("Invalid start record")
            records = grouped[start]
            if [r[0] for r in records] != list(range(len(records))) or len(records) > iterations+1:
                raise ValueError("Noncontiguous trace")
            for j, (_, current, improvement) in enumerate(records):
                reference = 0 if j == 0 else records[j-1][1]-current
                if abs(improvement-reference) > 1e-10:
                    raise ValueError("Inconsistent improvement")
            if records:
                if type(value) not in (int, float) or abs(value-records[-1][1]) > 1e-10:
                    raise ValueError("Start objective mismatch")
            elif value != "unavailable":
                raise ValueError("Invalid unavailable objective")
            if converged:
                if status != "accepted" or iterations < 1 or len(records) != iterations+1 or any(r[2] < -1e-9 for r in records):
                    raise ValueError("Invalid converged trace")
                if abs(records[-1][2]) > controls["tol"]*max(1., records[-2][1])+1e-14:
                    raise ValueError("Convergence criterion was not reached")
                accepted.append((value, start, iterations))
            elif status == "accepted":
                raise ValueError("Failed start marked accepted")
        if not accepted:
            raise ValueError("No converged start")
        best = min(accepted)
        if attrs["chosen_start"] != best[1] or attrs["iterations"] != best[2] or abs(best[0]-objective) > 2e-8:
            raise ValueError("Selected start does not match fit")
        expected_settings = saved_summary(TableSet({}, **attrs))["settings"]
        actual_settings = result["settings"]
        if list(actual_settings.columns) != list(expected_settings.columns) or actual_settings.setting.duplicated().any() or dict(actual_settings.to_numpy().tolist()) != dict(expected_settings.to_numpy().tolist()):
            raise ValueError("Published settings disagree with metadata")
    except (KeyError, TypeError, ValueError, OverflowError, AnalysisError) as exc:
        raise AnalysisError("invalid_state", "Saved frequency CATPCA convergence trace is invalid.") from exc


def _base(result, max_bytes, max_work, device):
    if device != "cpu":
        _error("Saved frequency CATPCA supports resident CPU only.", "unsupported_option")
    state, n, p, d, plan = _preflight(result, max_bytes, max_work)
    try:
        if type(state["version"]) is not int or state["version"] != 1 or state["kind"] != "catpca_fweight" or _seal(state) != result.attrs["state_sha256"]:
            raise ValueError("Integrity mismatch")
        variables = _names(state["variables"], 2)
        if any(len(name) > 128 for name in variables):
            raise ValueError("Variable names exceed 128 characters")
        descriptors = state["descriptors"]
        if not isinstance(descriptors, list) or len(descriptors) != p or [x["name"] for x in descriptors] != variables:
            raise ValueError("Invalid descriptors")
        positions, frequencies = state["positions"], state["frequencies"]
        input_n = state["input_nobs"]
        if type(input_n) is not int or not n <= input_n <= 3000 or any(type(x) is not int or not 0 <= x < input_n for x in positions) or positions != sorted(set(positions)):
            raise ValueError("Invalid sample positions")
        if len(frequencies) != n or any(type(x) is not int or not 1 <= x <= 10**9 for x in frequencies) or not 1 <= sum(frequencies) <= 10**9 or state["frequency_total"] != sum(frequencies):
            raise ValueError("Invalid frequencies")
        zero, missing = state["zero_positions"], state["missing_positions"]
        for excluded in (zero, missing):
            if not isinstance(excluded, list) or excluded != sorted(set(excluded)) or any(type(x) is not int or not 0 <= x < input_n for x in excluded):
                raise ValueError("Invalid exclusions")
        if sorted(positions+zero+missing) != list(range(input_n)):
            raise ValueError("Sample partition mismatch")
        if not isinstance(state["frequency"], str) or not 1 <= len(state["frequency"]) <= 128 or state["frequency"] in variables:
            raise ValueError("Invalid frequency column")
        attrs = result.attrs
        if any(attrs.get(key) != state[key] for key in ("variables", "components", "frequency", "frequency_total", "zero_positions", "missing_positions")) or attrs.get("n") != n or attrs.get("n_input") != input_n or attrs.get("sample_positions") != positions or attrs.get("converged") is not True or attrs.get("device") != "cpu" or attrs.get("dtype") != "float64" or attrs.get("weight_type") != "frequency":
            raise ValueError("Inconsistent fit metadata")
        axes_data, eigen_data = state["axes"], state["eigenvalues"]
        if len(axes_data) != p or any(len(row) != d for row in axes_data) or len(eigen_data) != d:
            raise ValueError("Invalid eigensystem shapes")
        if any(type(x) not in (int, float) or not math.isfinite(x) for row in axes_data for x in row) or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in eigen_data):
            raise ValueError("Invalid eigensystem")
        numeric_values = state["numeric_values"]
        if not isinstance(numeric_values, dict) or set(numeric_values) != {x["name"] for x in descriptors if x["scale"] == "numeric"}:
            raise ValueError("Original numeric values missing")
        membership = state["membership"]
        if not isinstance(membership, dict) or set(membership) != {x["name"] for x in descriptors if x["scale"] != "numeric"}:
            raise ValueError("Original membership missing")
        for descriptor in descriptors:
            if descriptor["scale"] == "numeric":
                if type(descriptor["mean"]) not in (int, float) or type(descriptor["sd"]) not in (int, float) or not math.isfinite(descriptor["mean"]) or not math.isfinite(descriptor["sd"]) or descriptor["sd"] <= 0:
                    raise ValueError("Invalid standardization")
                raw = numeric_values[descriptor["name"]]
                if not isinstance(raw, list) or len(raw) != n or any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1e100 for x in raw):
                    raise ValueError("Invalid original numeric values")
                continue
            levels, values, totals = descriptor["levels"], descriptor["quantifications"], descriptor["counts"]
            if descriptor["scale"] not in ("nominal", "ordinal") or not 2 <= len(levels) <= 32 or len(values) != len(levels) or len(totals) != len(levels):
                raise ValueError("Invalid categorical mapping")
            if any(not isinstance(level, list) or len(level) != 2 or _label(_atom(level[1])) != level for level in levels) or len({str(x) for x in levels}) != len(levels):
                raise ValueError("Invalid typed labels")
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in values) or any(type(x) is not int or x < 1 for x in totals) or sum(totals) != sum(frequencies):
                raise ValueError("Invalid categorical frequencies")
            if abs(sum(c*q for c, q in zip(totals, values))/sum(totals)) > 1e-8 or abs(sum(c*q*q for c, q in zip(totals, values))/sum(totals)-1) > 1e-8:
                raise ValueError("Mapping is not standardized")
            if descriptor["scale"] == "ordinal" and any(b < a-1e-10 for a, b in zip(values, values[1:])):
                raise ValueError("Nonmonotone mapping")
            codes = membership[descriptor["name"]]
            if not isinstance(codes, list) or len(codes) != n or any(type(x) is not int or not 0 <= x < len(levels) for x in codes):
                raise ValueError("Invalid original memberships")
            if [sum(f for f, code in zip(frequencies, codes) if code == k) for k in range(len(levels))] != totals:
                raise ValueError("Membership frequencies mismatch")
    except (KeyError, TypeError, ValueError, OverflowError, AnalysisError) as exc:
        raise AnalysisError("invalid_state", "Saved frequency CATPCA state has invalid structure or integrity.") from exc
    names = [f"component_{j+1}" for j in range(d)]
    axes, eigenvalues, weight = torch.tensor(axes_data, dtype=DT), torch.tensor(eigen_data, dtype=DT), _weights(frequencies)
    scores = _frame_tensor(result, "scores", ["source_position"]+names, n)
    quantified = _frame_tensor(result, "transformed", ["source_position"]+variables, n)
    expected_positions = torch.tensor(positions, dtype=DT)
    _close(scores[:, 0], expected_positions, "score positions", atol=0)
    _close(quantified[:, 0], expected_positions, "transformed positions", atol=0)
    scores, quantified = scores[:, 1:], quantified[:, 1:]
    loadings = _frame_tensor(result, "loadings", names, p, index=variables)
    _close(_frame_tensor(result, "sample", ["source_position", "frequency"], n), torch.stack((expected_positions, weight), dim=1), "sample table", atol=0)
    _close(_frame_tensor(result, "eigenvectors", names, p, index=variables), axes, "eigenvector table")
    _close(axes.T@axes, torch.eye(d, dtype=DT), "component orthogonality")
    _close(loadings, axes*eigenvalues.sqrt()[None, :], "loadings")
    _close((quantified*weight[:, None]).sum(0)/weight.sum(), torch.zeros(p, dtype=DT), "weighted centering")
    _close((quantified.square()*weight[:, None]).sum(0)/weight.sum(), torch.ones(p, dtype=DT), "weighted scaling")
    _close(scores, quantified@axes/eigenvalues.sqrt()[None, :], "scores")
    _close(scores.T@(weight[:, None]*scores)/weight.sum(), torch.eye(d, dtype=DT), "weighted score covariance")
    covariance = quantified.T@(weight[:, None]*quantified)/weight.sum()
    _close(covariance@axes, axes*eigenvalues[None, :], "eigensystem")
    spectrum = torch.linalg.eigvalsh(covariance).flip(0)[:min(n, p)]
    _close(eigenvalues, spectrum[:d], "leading eigenvalues")
    eigentable = _frame_tensor(result, "eigenvalues", ["component", "eigenvalue", "proportion"], min(n, p))
    _close(eigentable, torch.stack((torch.arange(1, len(spectrum)+1, dtype=DT), spectrum, spectrum/p), dim=1), "full spectrum")
    objective = float(((quantified-scores@loadings.T).square()*weight[:, None]).sum()/weight.sum())
    fit = _frame_tensor(result, "fit", ["physical_n", "frequency_total", "components", "reconstruction_loss", "variance_proportion"], 1)
    _close(fit, torch.tensor([[n, sum(frequencies), d, objective, 1-objective/p]], dtype=DT), "objective")
    for j, descriptor in enumerate(descriptors):
        if descriptor["scale"] != "numeric":
            mapping = torch.tensor(descriptor["quantifications"], dtype=DT)
            codes = torch.tensor(membership[descriptor["name"]], dtype=torch.int64)
            _close(quantified[:, j], mapping[codes], "original category membership")
        else:
            raw = torch.tensor(numeric_values[descriptor["name"]], dtype=DT)
            mean = (raw*weight).sum()/weight.sum()
            sd = (((raw-mean).square()*weight).sum()/weight.sum()).sqrt()
            if float(sd) <= 1e-12*max(float(raw.abs().max()), 1.):
                _error("Saved raw numeric values have degenerate variance.", "invalid_state")
            _close(torch.tensor([descriptor["mean"], descriptor["sd"]], dtype=DT), torch.stack((mean, sd)), "original numeric scaling")
            _close(quantified[:, j], (raw-mean)/sd, "original numeric values")
    _validate_trace(result, state, objective)
    for name, expected in _mapping_tables(descriptors).items():
        actual = result.get(name)
        if actual is None or list(actual.columns) != list(expected.columns) or actual.to_numpy().tolist() != expected.to_numpy().tolist():
            _error("Saved mapping tables disagree with the fitted state.", "invalid_state")
    return state, scores, loadings, weight, plan


@resident_cpu
def catpca_fweight_predict(result: TableSet, data: Any, *, missing: str = "raise",
                          max_work: int = WORK, max_bytes: int = BYTES,
                          device: str = "cpu") -> TableSet:
    """Apply saved weighted CATPCA mappings to physical query rows without refit.

    Query rows need no frequency column. Unknown typed categories are refused.
    Fit membership and all weighted geometry are checked before projection.
    """
    state, _, _, _, validation = _base(result, max_bytes, max_work, device)
    variables, p, d = state["variables"], len(state["variables"]), state["components"]
    if not isinstance(data, pd.DataFrame) or not 1 <= len(data) <= 3000 or not data.columns.is_unique or any(name not in data for name in variables):
        _error("Projection requires a DataFrame with 1–3000 rows and unique fitted columns.")
    if missing not in ("drop", "raise"):
        _error("missing must be 'drop' or 'raise'.")
    validation_work = 64*len(state["positions"])*p*(p+d)
    plan = _admit("frequency CATPCA projection with source validation",
                  validation["estimated_workspace_bytes"]+256*len(data)*(p+d+1),
                  validation_work+32*len(data)*p*d, max_bytes, max_work)
    frame = data[variables]
    complete = ~frame.isna().any(axis=1)
    if missing == "raise" and not bool(complete.all()):
        _error("Projection variables contain missing values.", "missing_data")
    positions = [i for i, keep in enumerate(complete.tolist()) if keep]
    frame = frame.iloc[positions]
    columns = []
    for descriptor in state["descriptors"]:
        values = frame[descriptor["name"]]
        if descriptor["scale"] == "numeric":
            if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype) or pd.api.types.is_complex_dtype(values.dtype):
                _error("Numeric projection variables must have numeric dtype.")
            raw = values.tolist()
            if any(not math.isfinite(float(x)) or abs(float(x)) > 1e100 for x in raw):
                _error("Numeric projection values must be finite and bounded.")
            columns.append((torch.tensor(raw, dtype=DT)-descriptor["mean"])/descriptor["sd"])
        else:
            mapping = {str(level): q for level, q in zip(descriptor["levels"], descriptor["quantifications"])}
            try:
                columns.append(torch.tensor([mapping[str(_label(_atom(x)))] for x in values.tolist()], dtype=DT))
            except KeyError as exc:
                raise AnalysisError("unknown_category", "Unknown typed category; saved quantification maps are not refitted.") from exc
    quantified = torch.stack(columns, dim=1)
    scores = quantified@torch.tensor(state["axes"], dtype=DT)/torch.tensor(state["eigenvalues"], dtype=DT).sqrt()[None, :]
    if not bool(torch.isfinite(scores).all()):
        _error("Projection produced nonfinite scores.", "numerical_failure")
    names = [f"component_{j+1}" for j in range(d)]
    return _save(TableSet({
        "scores": table([[position]+row for position, row in zip(positions, scores.tolist())], columns=["source_position"]+names),
        "transformed": table([[position]+row for position, row in zip(positions, quantified.tolist())], columns=["source_position"]+variables),
    }, title="Saved frequency CATPCA projection", method="catpca_fweight_predict",
       n=len(positions), n_input=len(data), sample_positions=positions, missing=missing,
       source_state_sha256=result.attrs["state_sha256"], resources=plan, source_validation=validation,
       weight_type="frequency calibration; physical query rows", dtype="float64", device="cpu",
       inference="descriptive saved-map projection; no refit or adaptive uncertainty"))


@resident_cpu
def catpca_category_centroids(result: TableSet, *, transform: Any = None,
                              max_work: int = WORK, max_bytes: int = BYTES,
                              device: str = "cpu") -> TableSet:
    """Original typed-category weighted score means, with optional basis transport.

    Retained memberships distinguish categories with tied ordinal maps. A
    supplied finite square nonsingular T transports loadings A@T, scores and
    centroids X@inv(T).T, and metric inv(T)@inv(T).T. This accepts a declared
    basis, not an automatic varimax/promax optimization. Legacy map-only fits
    are refused because original membership cannot be recovered from ties.
    """
    if device != "cpu":
        _error("Saved frequency CATPCA supports resident CPU only.", "unsupported_option")
    _, n, p, d, validation = _preflight(result, max_bytes, max_work)
    validation_work = 64*n*p*(p+d)
    centroid_work = 64*n*p*d+64*n*d*d+64*d**3+32*p*d*d
    plan = _admit("frequency CATPCA centroids with source validation and basis transport",
                  validation["estimated_workspace_bytes"]+8*(8*n*d+16*d*d+8*p*d)+128*(n*d+64*p*d),
                  validation_work+centroid_work, max_bytes, max_work)
    state, scores, loadings, weight, _ = _base(result, max_bytes, max_work, device)
    if transform is None:
        transform = torch.eye(d, dtype=DT)
    else:
        # Shape/type checks precede tensor conversion: oversized array-likes are
        # not materialized merely to discover that the requested basis is bad.
        if isinstance(transform, torch.Tensor):
            if transform.device.type != "cpu" or tuple(transform.shape) != (d, d) or transform.dtype == torch.bool or transform.is_complex() or transform.layout != torch.strided or transform.is_quantized:
                _error("Basis transform must be a real CPU square component matrix.", "invalid_transform")
            transform = transform.to(dtype=DT)
        elif isinstance(transform, (list, tuple)):
            if len(transform) != d or any(not isinstance(row, (list, tuple)) or len(row) != d for row in transform) or any(type(x) not in (int, float) or not math.isfinite(x) for row in transform for x in row):
                _error("Basis transform must be a finite real square component matrix.", "invalid_transform")
            transform = torch.tensor(transform, dtype=DT)
        elif hasattr(transform, "shape") and tuple(transform.shape) == (d, d):
            if getattr(getattr(transform, "dtype", None), "kind", None) not in ("i", "u", "f"):
                _error("Basis transform must contain real non-boolean numbers.", "invalid_transform")
            try:
                raw = transform.tolist()
                if not isinstance(raw, list) or len(raw) != d or any(not isinstance(row, list) or len(row) != d for row in raw) or any(type(x) not in (int, float) or not math.isfinite(x) for row in raw for x in row):
                    _error("Basis transform must contain finite real non-boolean numbers.", "invalid_transform")
                transform = torch.tensor(raw, dtype=DT)
            except (TypeError, ValueError, OverflowError) as exc:
                raise AnalysisError("invalid_transform", "Basis transform must be finite real geometry.") from exc
        else:
            _error("Basis transform must match the component dimensions.", "invalid_transform")
    if not bool(torch.isfinite(transform).all()) or float(transform.abs().max()) > 1e6:
        _error("Basis transform must be finite and bounded by 1e6.", "invalid_transform")
    singular = torch.linalg.svdvals(transform)
    if float(singular[-1]) < 1e-6 or float(singular[0]/singular[-1]) > 1e6:
        _error("Basis transform is singular or exceeds the 1e6 condition bound.", "invalid_transform")
    inverse = torch.linalg.inv(transform)
    transported_scores, transported_loadings = scores@inverse.T, loadings@transform
    metric = inverse@inverse.T
    _close(transported_scores@transported_loadings.T, scores@loadings.T, "transported reconstruction")
    names = [f"component_{j+1}" for j in range(d)]
    rows, base_rows = [], []
    for descriptor in state["descriptors"]:
        if descriptor["scale"] == "numeric":
            continue
        codes = torch.tensor(state["membership"][descriptor["name"]], dtype=torch.int64)
        centroids, totals = _means(scores, codes, len(descriptor["levels"]), weight)
        transported = centroids@inverse.T
        for level, count, center, original in zip(descriptor["levels"], totals.tolist(), transported.tolist(), centroids.tolist()):
            prefix = [descriptor["name"], level[0], level[1], int(count)]
            rows.append(prefix+center)
            base_rows.append(prefix+original)
    if not rows:
        _error("Original-category centroids require at least one categorical variable.", "invalid_spec")
    columns = ["variable", "category_type", "category", "frequency_count"]+names
    return _save(TableSet({
        "centroids": table(rows, columns=columns),
        "base_centroids": table(base_rows, columns=columns),
        "scores": table([[position]+row for position, row in zip(state["positions"], transported_scores.tolist())], columns=["source_position"]+names),
        "loadings": table(transported_loadings.tolist(), columns=names, index=state["variables"]),
        "metric": table(metric.tolist(), columns=names, index=names),
        "transform": table(transform.tolist(), columns=names, index=names),
    }, title="Original-category frequency CATPCA centroids", method="catpca_category_centroids",
       n=len(weight), frequency_total=state["frequency_total"], source_state_sha256=result.attrs["state_sha256"],
       sample_positions=state["positions"], resources=plan, transform=transform.tolist(),
       basis="loadings A@T; scores and centroids X@inv(T).T; metric inv(T)@inv(T).T",
       membership="retained original typed categories; never inferred from quantified ties",
       inference="descriptive category means and declared basis transport; no automatic rotation or uncertainty",
       sources=SOURCES, device="cpu", dtype="float64", weight_type="frequency"))
