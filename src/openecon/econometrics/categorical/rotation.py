"""Saved single-vector CATPCA rotations; finite descriptive geometry only."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import plan_workspace, workspace_budget_bytes

from .optimal import (
    DEFAULT_BYTES, DEFAULT_WORK, DTYPE, _error, _integer, _mapping_tables,
    _prediction, _real, _saved, _seal, _state,
)

SOURCES = [
    "https://www.ibm.com/docs/en/spss-statistics/30.0.0?topic=catpca-rotation-subcommand-command",
    "https://www.ibm.com/docs/en/SS3RA7_18.4.0/pdf/AlgorithmsGuide.pdf",
]
METHODS = ("catpca_varimax", "catpca_promax")
STATE_LIMIT = 8 * 1024**2


def _close(actual, expected, what, *, atol=2e-8):
    if actual.shape != expected.shape or not bool(torch.isfinite(actual).all()) or not torch.allclose(actual, expected, atol=atol, rtol=2e-8):
        _error(f"Saved {what} is numerically inconsistent.", "invalid_state")


def _controls(normalize, max_iter, tol, max_work, max_bytes, device):
    if device != "cpu":
        _error("CATPCA rotations support resident CPU float64 only.", "unsupported_option")
    if not isinstance(normalize, bool):
        _error("normalize must be a boolean.", "invalid_option")
    return {
        "normalize": normalize,
        "max_iter": _integer(max_iter, "max_iter", 1, 1000),
        "tol": _real(tol, "tol", 1e-12, 1e-3),
        "max_work": _integer(max_work, "max_work", 1, 2**63-1),
        "max_bytes": _integer(max_bytes, "max_bytes", 1, 2**63-1),
    }


def _estimate_source(result):
    """Bound source dimensions and text before JSON serialization/tensor copies."""
    if not isinstance(result, TableSet) or result.attrs.get("method") != "catpca":
        _error("Supply a fitted or fully restored single-vector CATPCA result.", "invalid_result")
    attrs = result.attrs
    n = _integer(attrs.get("n"), "saved n", 4, 3000)
    p = len(attrs.get("variables", []))
    d = _integer(attrs.get("components"), "saved components", 2, min(6, p-1))
    if not 3 <= p <= 12 or len(result) > 16:
        _error("Rotations admit 3–12 variables and 2–min(6,p-1) components.", "resource_limit")
    estimate = 0
    for frame in result.values():
        if len(frame) > 12012 or len(frame.columns) > 16:
            _error("Saved source table dimensions exceed the bounded rotation domain.", "resource_limit")
        estimate += 64 * (frame.size + len(frame.index) + len(frame.columns))
        for row in frame.itertuples(index=False, name=None):
            for cell in row:
                if isinstance(cell, str):
                    if len(cell) > 4096:
                        _error("Saved source text exceeds 4096 characters per cell.", "resource_limit")
                    estimate += 4 * len(cell)
    pending = [(attrs, 0)]
    while pending:
        value, depth = pending.pop()
        if depth > 16:
            _error("Saved source metadata nesting is excessive.", "resource_limit")
        estimate += 64
        if isinstance(value, dict):
            if len(value) > 12012:
                _error("Saved source metadata exceeds the rotation domain.", "resource_limit")
            pending.extend((child, depth+1) for child in value.values())
            pending.extend((key, depth+1) for key in value)
        elif isinstance(value, (list, tuple)):
            if len(value) > 12012:
                _error("Saved source metadata exceeds the rotation domain.", "resource_limit")
            pending.extend((child, depth+1) for child in value)
        elif isinstance(value, str):
            if len(value) > 4096:
                _error("Saved metadata text exceeds 4096 characters.", "resource_limit")
            estimate += 4 * len(value)
        if estimate > STATE_LIMIT:
            _error("Saved CATPCA source exceeds the 8 MiB bounded state domain.", "resource_limit")
    return n, p, d, estimate


def _admit(result, controls, *, rotating):
    n, p, d, estimate = _estimate_source(result)
    iterations = controls["max_iter"] if rotating else 0
    work = 64*n*p*p + 64*iterations*p*d*d
    if work > controls["max_work"]:
        _error("CATPCA rotation/validation exceeds max_work before numerical copies.", "resource_limit")
    plan = plan_workspace("saved CATPCA rotation and semantic validation", {
        "base_and_rotated_geometry": 8*n*(8*p+12*d),
        "small_rotation_and_validation_matrices": 8*(20*p*d+32*d*d),
        "source_serialization_and_complete_state": 6*estimate,
        "published_table_cells": 64*n*(2*p+3*d),
    }, budget_bytes=min(controls["max_bytes"], workspace_budget_bytes()))
    return n, p, d, plan.record(), work


def _tensor_frame(result, name, columns, rows, *, index=None):
    frame = result.get(name)
    if frame is None or list(frame.columns) != columns or len(frame) != rows or (index is not None and list(frame.index) != index):
        _error(f"Saved CATPCA {name} has invalid dimensions or variable order.", "invalid_state")
    try:
        tensor = torch.tensor(frame.to_numpy(dtype=float).tolist(), dtype=DTYPE)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError("invalid_state", f"Saved CATPCA {name} must be finite numeric geometry.") from exc
    if not bool(torch.isfinite(tensor).all()):
        _error(f"Saved CATPCA {name} contains nonfinite geometry.", "invalid_state")
    return tensor


def _base(result, controls, *, rotating=False):
    n, p, d, plan, work = _admit(result, controls, rotating=rotating)
    state = _state(result, ("catpca",))
    variables = state["variables"]
    names = [f"component_{j+1}" for j in range(d)]
    attrs = result.attrs
    if attrs.get("converged") is not True or attrs.get("dtype") != "float64" or attrs.get("device") != "cpu" or attrs.get("weight_type") != "unweighted" or attrs.get("variables") != variables:
        _error("Saved CATPCA must be converged, unweighted CPU float64 with matching maps.", "invalid_state")
    n_input = attrs.get("n_input")
    positions = attrs.get("sample_positions")
    if isinstance(n_input, bool) or not isinstance(n_input, int) or not n <= n_input <= 3000 or not isinstance(positions, list) or len(positions) != n or any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < n_input for x in positions) or positions != sorted(set(positions)) or attrs.get("n_missing") != n_input-n:
        _error("Saved CATPCA sample positions are invalid.", "invalid_state")
    if state["components"] != d:
        _error("Saved CATPCA component metadata disagree.", "invalid_state")
    axes = torch.tensor(state["axes"], dtype=DTYPE)
    eigenvalues = torch.tensor(state["eigenvalues"], dtype=DTYPE)
    loadings = _tensor_frame(result, "loadings", names, p, index=variables)
    table_axes = _tensor_frame(result, "eigenvectors", names, p, index=variables)
    score_data = _tensor_frame(result, "scores", ["source_position"]+names, n)
    transformed_data = _tensor_frame(result, "transformed", ["source_position"]+variables, n)
    position_tensor = torch.tensor(positions, dtype=DTYPE)
    _close(score_data[:, 0], position_tensor, "score sample positions", atol=0)
    _close(transformed_data[:, 0], position_tensor, "quantified sample positions", atol=0)
    scores, quantified = score_data[:, 1:], transformed_data[:, 1:]
    _close(axes.T@axes, torch.eye(d, dtype=DTYPE), "component orthogonality")
    _close(table_axes, axes, "eigenvector table")
    _close(loadings, axes*eigenvalues.sqrt()[None, :], "base loadings")
    _close(quantified.mean(0), torch.zeros(p, dtype=DTYPE), "quantified centering")
    _close(quantified.square().mean(0), torch.ones(p, dtype=DTYPE), "quantified normalization")
    _close(scores, quantified@axes/eigenvalues.sqrt()[None, :], "normalized person scores")
    _close(scores.T@scores/n, torch.eye(d, dtype=DTYPE), "person covariance")
    _close((quantified.T@quantified/n)@axes, axes*eigenvalues[None, :], "saved eigensystem")
    spectrum = torch.linalg.eigvalsh(quantified.T@quantified/n).flip(0)
    _close(eigenvalues, spectrum[:d], "retained leading eigenvalues")
    eigenframe = result.get("eigenvalues")
    if eigenframe is None or list(eigenframe.columns) != ["component", "eigenvalue", "proportion"] or len(eigenframe) != min(n, p):
        _error("Saved CATPCA eigenvalue table is invalid.", "invalid_state")
    eigen_table = torch.tensor(eigenframe.to_numpy(dtype=float).tolist(), dtype=DTYPE)
    _close(eigen_table[:, 0], torch.arange(1, len(eigenframe)+1, dtype=DTYPE), "spectrum ordering")
    _close(eigen_table[:, 1], spectrum[:len(eigenframe)], "full spectrum")
    _close(eigen_table[:, 2], eigen_table[:, 1]/p, "spectrum proportions")
    objective = float((quantified-scores@loadings.T).square().sum()/n)
    fit = _tensor_frame(result, "fit", ["n", "components", "reconstruction_loss", "variance_proportion"], 1)
    _close(fit, torch.tensor([[n, d, objective, 1-objective/p]], dtype=DTYPE), "base reconstruction objective")
    # A quantification value identifies a class, not original membership if
    # distinct categories tie. Check the aggregate class counts, never invent it.
    for j, descriptor in enumerate(state["descriptors"]):
        if descriptor["scale"] == "numeric":
            continue
        if sum(descriptor["counts"]) != n:
            _error("Saved category frequencies disagree with the fit sample.", "invalid_state")
        values = descriptor["quantifications"]
        assigned = [False]*len(values)
        seen = torch.zeros(n, dtype=torch.bool)
        for k, value in enumerate(values):
            if assigned[k]:
                continue
            tied = [h for h, candidate in enumerate(values) if abs(candidate-value) <= 1e-9]
            matches = (quantified[:, j]-value).abs() <= 1e-9
            if bool((seen & matches).any()) or int(matches.sum()) != sum(descriptor["counts"][h] for h in tied):
                _error("Saved quantification classes do not match sample counts.", "invalid_state")
            seen |= matches
            for h in tied:
                assigned[h] = True
        if not bool(seen.all()):
            _error("Saved quantified values lie outside the category maps.", "invalid_state")
    for name, mapping_frame in _mapping_tables(state["descriptors"]).items():
        original = result.get(name)
        if original is None or list(original.columns) != list(mapping_frame.columns) or original.to_numpy().tolist() != mapping_frame.to_numpy().tolist():
            _error("Saved category/numeric map tables disagree with state.", "invalid_state")
    return state, quantified, scores, loadings, positions, objective, plan, work


def _criterion(z):
    return float((z.pow(4).sum(0)-z.square().sum(0).square()/len(z)).sum())


def _stationarity(z):
    gradient = z.pow(3)-z*z.square().sum(0)[None, :]/len(z)
    inner = z.T@gradient
    return float(torch.linalg.matrix_norm(inner-inner.T)/torch.linalg.matrix_norm(inner).clamp_min(1e-15))


def _varimax(loadings, controls):
    scale = loadings.square().sum(1).sqrt() if controls["normalize"] else torch.ones(len(loadings), dtype=DTYPE)
    if bool((scale <= 1e-12).any()):
        _error("Kaiser normalization requires nonzero retained communalities; set normalize=False.", "degenerate_rotation")
    base = loadings/scale[:, None]
    z = base.clone()
    d = z.shape[1]
    rotation = torch.eye(d, dtype=DTYPE)
    history = [[0, _criterion(z), _stationarity(z), 0.0]]
    for iteration in range(1, controls["max_iter"]+1):
        previous = _criterion(z)
        # Each pair is maximized exactly, including a stationary saddle at the
        # initial axes. A whole sweep precedes any convergence acceptance.
        for j in range(d-1):
            for k in range(j+1, d):
                x, y = z[:, j].clone(), z[:, k].clone()
                u, v = x.square()-y.square(), 2*x*y
                a, b = float(u.sum()), float(v.sum())
                c, e = float((u.square()-v.square()).sum()), 2*float((u*v).sum())
                angle = .25*math.atan2(e-2*a*b/len(z), c-(a*a-b*b)/len(z))
                cosine, sine = math.cos(angle), math.sin(angle)
                z[:, j], z[:, k] = cosine*x+sine*y, cosine*y-sine*x
                rj, rk = rotation[:, j].clone(), rotation[:, k].clone()
                rotation[:, j], rotation[:, k] = cosine*rj+sine*rk, cosine*rk-sine*rj
        value, measure = _criterion(z), _stationarity(z)
        history.append([iteration, value, measure, value-previous])
        if value < previous-1e-10*max(1, abs(previous)):
            _error("Varimax criterion decreased beyond roundoff.", "numerical_failure")
        if measure <= controls["tol"]:
            return rotation, history, scale
    _error("Varimax did not reach the requested stationary tolerance.", "nonconvergence")


def _identify(loadings, transform):
    pattern = loadings@transform
    order = sorted(range(pattern.shape[1]), key=lambda j: (-float(pattern[:, j].square().sum()), j))
    transform = transform[:, order].clone()
    pattern = loadings@transform
    for j in range(pattern.shape[1]):
        if float(pattern[int(torch.argmax(pattern[:, j].abs())), j]) < 0:
            transform[:, j] *= -1
    return transform


def _rotate(result, method, controls, power=None):
    state, quantified, original_scores, loadings, positions, objective, plan, work = _base(result, controls, rotating=True)
    transform, history, scale = _varimax(loadings, controls)
    varimax_transform = transform.clone()
    promax_residual = None
    if method == "catpca_promax":
        varimax = loadings@transform
        target_base = varimax/scale[:, None] if controls["normalize"] else varimax
        target = torch.sign(target_base)*target_base.abs().pow(power)
        fit = torch.linalg.lstsq(varimax, target).solution
        singular = torch.linalg.svdvals(fit)
        if not bool(torch.isfinite(fit).all()) or float(singular[-1]) <= 1e-10*float(singular[0]):
            _error("Promax target yields a singular or ill-conditioned transformation.", "degenerate_rotation")
        promax_residual = float((varimax@fit-target).square().sum())
        raw = transform@fit
        inverse = torch.linalg.inv(raw)
        phi = inverse@inverse.T
        transform = raw*phi.diagonal().sqrt()[None, :]
    transform = _identify(loadings, transform)
    inverse = torch.linalg.inv(transform)
    phi = inverse@inverse.T
    pattern = loadings@transform
    structure = pattern@phi
    scores = original_scores@inverse.T
    reconstructed = scores@pattern.T
    _close(reconstructed, original_scores@loadings.T, "rotation reconstruction")
    base_json = summary_state(result)
    if len(base_json.encode()) > STATE_LIMIT:
        _error("Complete saved CATPCA source exceeds the 8 MiB rotation state domain.", "resource_limit")
    names = [f"component_{j+1}" for j in range(state["components"])]
    rotation_state = {
        "version": 1, "kind": method, "base_summary": base_json,
        "base_summary_sha256": hashlib.sha256(base_json.encode()).hexdigest(),
        "transform": transform.tolist(), "pattern": pattern.tolist(),
        "varimax_transform": varimax_transform.tolist(),
        "structure": structure.tolist(), "phi": phi.tolist(),
        "scores": scores.tolist(), "reconstruction": reconstructed.tolist(),
        "settings": controls, "power": power, "trace": history,
        "promax_target_sse": promax_residual,
    }
    output = TableSet({
        "pattern": table(pattern.tolist(), columns=names, index=state["variables"]),
        "structure": table(structure.tolist(), columns=names, index=state["variables"]),
        "component_correlations": table(phi.tolist(), columns=names, index=names),
        "transformation": table(transform.tolist(), columns=names, index=names),
        "original_loadings": table(loadings.tolist(), columns=names, index=state["variables"]),
        "scores": table([[position]+row for position, row in zip(positions, scores.tolist())], columns=["source_position"]+names),
        "transformed": table([[position]+row for position, row in zip(positions, quantified.tolist())], columns=["source_position"]+state["variables"]),
        "reconstruction": table([[position]+row for position, row in zip(positions, reconstructed.tolist())], columns=["source_position"]+state["variables"]),
        "rotation_iterations": table(history, columns=["iteration", "varimax_criterion", "relative_stationarity", "improvement"]),
        "fit": table([[len(positions), state["components"], objective, 1-objective/len(state["variables"])]], columns=["n", "components", "reconstruction_loss", "variance_proportion"]),
        **_mapping_tables(state["descriptors"]),
    }, title="Saved CATPCA "+method.removeprefix("catpca_")+" rotation", method=method,
        variables=state["variables"], components=state["components"], n=len(positions),
        n_input=result.attrs["n_input"], n_missing=result.attrs["n_missing"], sample_positions=positions,
        settings=controls, power=power, resources=plan, declared_work=work,
        converged=True, iterations=len(history)-1, rotation_state=rotation_state,
        state_sha256=_seal(rotation_state), sources=SOURCES, device="cpu", dtype="float64", weight_type="unweighted",
        inference="descriptive geometry only; no SE, CI, p-value or global optimum/vendor-equivalence claim",
        category_coordinates="original scalar quantification maps unchanged; original-category person centroids are not reconstructible for tied maps and are omitted",
        normalization="X_rot = X T^{-T}; pattern = L T; Phi = T^{-1}T^{-T}; structure = pattern Phi; diag(Phi)=1",
        solution="cyclic planar local varimax maximum; promax is a powered-target least-squares transformation, without refitting")
    return _saved(output)


@resident_cpu
def catpca_varimax(result: TableSet, *, normalize: bool = True, max_iter: int = 1000,
                    tol: float = 1e-10, max_work: int = DEFAULT_WORK,
                    max_bytes: int = DEFAULT_BYTES, device: str = "cpu") -> TableSet:
    """Orthogonal varimax of a saved 2–6 component single-vector CATPCA fit.

    Kaiser row normalization is optional. Original scalar category maps stay
    fixed; pattern, structure, scores and reconstruction use one matrix T.
    Full oe.summary_state/restore_summary retains the source and projection.
    """
    controls = _controls(normalize, max_iter, tol, max_work, max_bytes, device)
    return _rotate(result, "catpca_varimax", controls)


@resident_cpu
def catpca_promax(result: TableSet, *, power: float = 4, normalize: bool = True,
                   max_iter: int = 1000, tol: float = 1e-10,
                   max_work: int = DEFAULT_WORK, max_bytes: int = DEFAULT_BYTES,
                   device: str = "cpu") -> TableSet:
    """Varimax followed by nonsingular powered-target oblique least squares.

    Power is bounded to [1,10]. Pattern, structure and Phi are distinct;
    rotated normalized score covariance equals Phi with unit diagonal.
    """
    controls = _controls(normalize, max_iter, tol, max_work, max_bytes, device)
    power = _real(power, "power", 1, 10)
    return _rotate(result, "catpca_promax", controls, power)


def _restored(result, controls):
    if not isinstance(result, TableSet) or result.attrs.get("method") not in METHODS:
        _error("Supply a fitted or fully restored CATPCA rotation.", "invalid_result")
    state = result.attrs.get("rotation_state")
    try:
        if not isinstance(state, dict) or state.get("version") != 1 or state.get("kind") != result.attrs["method"]:
            raise ValueError("Wrong rotation schema")
        if set(state) != {"version", "kind", "base_summary", "base_summary_sha256", "transform", "varimax_transform", "pattern", "structure", "phi", "scores", "reconstruction", "settings", "power", "trace", "promax_target_sse"}:
            raise ValueError("Unknown rotation state fields")
        base_json = state["base_summary"]
        if not isinstance(base_json, str) or len(base_json) > STATE_LIMIT or len(base_json.encode()) > STATE_LIMIT:
            raise ValueError("Source exceeds bounded complete state")
        # Admit JSON parsing and reconstructed tables before restore_summary.
        plan_workspace("saved CATPCA rotation source restoration", {
            "encoded_source_and_parser_tables": 12*len(base_json.encode()),
        }, budget_bytes=min(controls["max_bytes"], workspace_budget_bytes()))
        if hashlib.sha256(base_json.encode()).hexdigest() != state["base_summary_sha256"]:
            raise ValueError("Integrity mismatch")
        base = restore_summary(base_json)
        base_state, quantified, original_scores, loadings, positions, objective, plan, work = _base(base, controls)
        n, p, d = len(positions), len(base_state["variables"]), base_state["components"]
        for key, shape in (("transform", (d, d)), ("varimax_transform", (d, d)), ("pattern", (p, d)), ("structure", (p, d)), ("phi", (d, d)), ("scores", (n, d)), ("reconstruction", (n, p))):
            value = state[key]
            if not isinstance(value, list) or len(value) != shape[0] or any(not isinstance(row, list) or len(row) != shape[1] for row in value):
                raise ValueError("Invalid matrix shape")
            if any(isinstance(cell, bool) or not isinstance(cell, (int, float)) or not math.isfinite(cell) or abs(cell) > 1e100 for row in value for cell in row):
                raise ValueError("Invalid bounded numeric matrix cells")
        if state["kind"] == "catpca_varimax":
            if state["power"] is not None or state["promax_target_sse"] is not None:
                raise ValueError("Varimax has promax settings")
        else:
            _real(state["power"], "saved power", 1, 10)
            _real(state["promax_target_sse"], "saved target SSE", 0, 1e100)
        saved_controls = state["settings"]
        if not isinstance(saved_controls, dict) or set(saved_controls) != {"normalize", "max_iter", "tol", "max_work", "max_bytes"}:
            raise ValueError("Invalid saved controls")
        saved_controls = _controls(**saved_controls, device="cpu")
        trace = state["trace"]
        if not isinstance(trace, list) or not 2 <= len(trace) <= saved_controls["max_iter"]+1 or any(not isinstance(row, list) or len(row) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in row) for row in trace):
            raise ValueError("Invalid rotation trace")
        if any(row[0] != i or row[2] < 0 or (i > 0 and (abs(row[3]-(row[1]-trace[i-1][1])) > 1e-8 or row[3] < -1e-9)) for i, row in enumerate(trace)):
            raise ValueError("Invalid rotation trace progression")
        if _seal(state) != result.attrs["state_sha256"]:
            raise ValueError("Integrity mismatch")
        transform = torch.tensor(state["transform"], dtype=DTYPE)
        if not bool(torch.isfinite(transform).all()):
            raise ValueError("Invalid matrix")
        singular = torch.linalg.svdvals(transform)
        if float(singular[-1]) <= 1e-10*float(singular[0]):
            raise ValueError("Singular transformation")
        inverse = torch.linalg.inv(transform)
        expected_phi = inverse@inverse.T
        pattern = torch.tensor(state["pattern"], dtype=DTYPE)
        structure = torch.tensor(state["structure"], dtype=DTYPE)
        phi = torch.tensor(state["phi"], dtype=DTYPE)
        scores = torch.tensor(state["scores"], dtype=DTYPE)
        reconstructed = torch.tensor(state["reconstruction"], dtype=DTYPE)
        _close(phi, expected_phi, "component correlations")
        _close(phi.diagonal(), torch.ones(d, dtype=DTYPE), "component variance normalization")
        if state["kind"] == "catpca_varimax":
            _close(transform.T@transform, torch.eye(d, dtype=DTYPE), "orthogonal rotation")
        _close(pattern, loadings@transform, "rotated pattern")
        _close(structure, pattern@phi, "oblique structure")
        _close(scores, original_scores@inverse.T, "rotated person scores")
        _close(scores.T@scores/n, phi, "rotated person covariance")
        _close(reconstructed, scores@pattern.T, "rotated reconstruction")
        _close(reconstructed, original_scores@loadings.T, "preserved base reconstruction")
        varimax_transform = torch.tensor(state["varimax_transform"], dtype=DTYPE)
        _close(varimax_transform.T@varimax_transform, torch.eye(d, dtype=DTYPE), "saved initial varimax orthogonality")
        scale = loadings.square().sum(1).sqrt() if saved_controls["normalize"] else torch.ones(p, dtype=DTYPE)
        if bool((scale <= 1e-12).any()):
            raise ValueError("Degenerate Kaiser normalization")
        rotated_normalized = (loadings/scale[:, None])@varimax_transform
        measure = _stationarity(rotated_normalized)
        if measure > saved_controls["tol"]+1e-12 or abs(trace[-1][1]-_criterion(rotated_normalized)) > 1e-8 or abs(trace[-1][2]-measure) > 1e-8:
            raise ValueError("Saved varimax is not stationary at the requested tolerance")
        # A zero tangent gradient alone also describes a minimum. Verify every
        # planar pair is already at its criterion maximum, allowing roundoff.
        for j in range(d-1):
            for k in range(j+1, d):
                x, y = rotated_normalized[:, j], rotated_normalized[:, k]
                u, v = x.square()-y.square(), 2*x*y
                a, b = float(u.sum()), float(v.sum())
                c, e = float((u.square()-v.square()).sum()), 2*float((u*v).sum())
                angle = .25*math.atan2(e-2*a*b/p, c-(a*a-b*b)/p)
                trial = rotated_normalized.clone()
                trial[:, j] = math.cos(angle)*x+math.sin(angle)*y
                trial[:, k] = math.cos(angle)*y-math.sin(angle)*x
                if _criterion(trial)-_criterion(rotated_normalized) > 1e-8:
                    raise ValueError("Saved varimax admits a positive planar improvement")
        if state["kind"] == "catpca_varimax":
            if state["power"] is not None or state["promax_target_sse"] is not None:
                raise ValueError("Varimax has promax settings")
            expected_transform = _identify(loadings, varimax_transform)
        else:
            power = _real(state["power"], "saved power", 1, 10)
            varimax = loadings@varimax_transform
            target_base = varimax/scale[:, None] if saved_controls["normalize"] else varimax
            target = torch.sign(target_base)*target_base.abs().pow(power)
            least_squares = torch.linalg.lstsq(varimax, target).solution
            raw = varimax_transform@least_squares
            raw_inverse = torch.linalg.inv(raw)
            raw_phi = raw_inverse@raw_inverse.T
            expected_transform = _identify(loadings, raw*raw_phi.diagonal().sqrt()[None, :])
            target_sse = float((varimax@least_squares-target).square().sum())
            if not isinstance(state["promax_target_sse"], (int, float)) or not math.isfinite(state["promax_target_sse"]) or abs(target_sse-state["promax_target_sse"]) > 1e-8:
                raise ValueError("Saved powered-target loss is invalid")
        _close(transform, expected_transform, "identified method-specific transformation")
        names = [f"component_{j+1}" for j in range(d)]
        variables = base_state["variables"]
        for key, expected, rows in (("pattern", pattern, variables), ("structure", structure, variables), ("component_correlations", phi, names), ("transformation", transform, names), ("original_loadings", loadings, variables)):
            _close(_tensor_frame(result, key, names, len(rows), index=rows), expected, key+" table")
        position_tensor = torch.tensor(positions, dtype=DTYPE)[:, None]
        for key, expected, columns in (("scores", scores, names), ("transformed", quantified, variables), ("reconstruction", reconstructed, variables)):
            _close(_tensor_frame(result, key, ["source_position"]+columns, n), torch.cat([position_tensor, expected], dim=1), key+" table")
        fit = _tensor_frame(result, "fit", ["n", "components", "reconstruction_loss", "variance_proportion"], 1)
        _close(fit, torch.tensor([[n, d, objective, 1-objective/p]], dtype=DTYPE), "rotated objective")
        _close(_tensor_frame(result, "rotation_iterations", ["iteration", "varimax_criterion", "relative_stationarity", "improvement"], len(trace)), torch.tensor(trace, dtype=DTYPE), "rotation trace table")
        for name, mapping in _mapping_tables(base_state["descriptors"]).items():
            if result[name].to_numpy().tolist() != mapping.to_numpy().tolist() or list(result[name].columns) != list(mapping.columns):
                raise ValueError("Changed scalar mapping")
        if any(result.attrs.get(key) != base.attrs.get(key) for key in ("n", "n_input", "n_missing", "sample_positions", "variables", "components", "device", "dtype", "weight_type")) or result.attrs.get("converged") is not True:
            raise ValueError("Changed sample or device metadata")
        if result.attrs.get("settings") != state["settings"] or result.attrs.get("power") != state["power"]:
            raise ValueError("Changed settings metadata")
        if result.attrs.get("iterations") != len(trace)-1:
            raise ValueError("Changed iteration metadata")
        return base, transform, pattern, plan, work
    except AnalysisError as exc:
        if exc.code in ("workspace_limit", "resource_limit", "invalid_state"):
            raise
        raise AnalysisError("invalid_state", "Saved CATPCA rotation options or source state is invalid.") from exc
    except (KeyError, TypeError, ValueError, OverflowError, torch.linalg.LinAlgError) as exc:
        raise AnalysisError("invalid_state", "Saved CATPCA rotation integrity, geometry or metadata is invalid.") from exc


@resident_cpu
def catpca_rotated_predict(result: TableSet, data: Any, *, missing: str = "raise",
                            max_work: int = DEFAULT_WORK, max_bytes: int = DEFAULT_BYTES,
                            device: str = "cpu") -> TableSet:
    """Apply original saved maps and the coherent inverse-transpose rotation.

    Full source and rotated geometry are validated after restoration; unknown
    categories fail, missing-drop row positions are retained, and no fit runs.
    """
    controls = _controls(True, 1, 1e-10, max_work, max_bytes, device)
    base, transform, pattern, validation, work = _restored(result, controls)
    state, quantified, positions, n_input, plan = _prediction(base, data, ("catpca",), missing, max_work-work, max_bytes, device)
    axes = torch.tensor(state["axes"], dtype=DTYPE)
    eigenvalues = torch.tensor(state["eigenvalues"], dtype=DTYPE)
    scores = (quantified@axes/eigenvalues.sqrt()[None, :])@torch.linalg.inv(transform).T
    reconstructed = scores@pattern.T
    names = [f"component_{j+1}" for j in range(state["components"])]
    return _saved(TableSet({
        "scores": table([[position]+row for position, row in zip(positions, scores.tolist())], columns=["source_position"]+names),
        "transformed": table([[position]+row for position, row in zip(positions, quantified.tolist())], columns=["source_position"]+state["variables"]),
        "reconstruction": table([[position]+row for position, row in zip(positions, reconstructed.tolist())], columns=["source_position"]+state["variables"]),
    }, title="Saved rotated CATPCA projection", method="catpca_rotated_predict",
        n_input=n_input, n=len(positions), n_missing=n_input-len(positions), sample_positions=positions,
        state_sha256=result.attrs["state_sha256"], source_method=result.attrs["method"], resources=plan,
        source_validation=validation, declared_validation_work=work, inference="descriptive projection only; no SE or CI",
        device="cpu", dtype="float64"))
