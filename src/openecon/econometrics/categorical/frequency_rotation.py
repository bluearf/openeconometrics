"""Automatic loading-space rotations of complete frequency CATPCA results."""
from __future__ import annotations

import hashlib
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import restore_summary, summary_state
from . import frequency_pca as fp
from .frequency import BYTES, WORK, DT, _admit, _means, _save, _seal
from .optimal import _error, _integer, _mapping_tables, _real
from .rotation import SOURCES, _criterion, _identify, _stationarity, _varimax

METHODS = ("catpca_varimax_fweight", "catpca_promax_fweight")
STATE_BYTES = 24 * 1024**2


def _controls(normalize, max_iter, tol, max_work, max_bytes, device):
    if device != "cpu" or type(normalize) is not bool:
        _error("Rotations require CPU and a boolean normalize option.", "unsupported_option")
    return dict(normalize=normalize, max_iter=_integer(max_iter, "max_iter", 1, 1000),
                tol=_real(tol, "tol", 1e-12, 1e-3),
                max_work=_integer(max_work, "max_work", 1, WORK),
                max_bytes=_integer(max_bytes, "max_bytes", 1, BYTES))


def _primitive(value, *, source=False):
    """Inspect bounded primitive state before hashing, decoding or tensors."""
    pending, nodes, size = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        size += 32
        if nodes > 240000 or depth > 18 or size > STATE_BYTES:
            _error("Saved rotation exceeds the bounded primitive state.", "resource_limit")
        if isinstance(item, dict):
            if len(item) > 100 or len(pending)+2*len(item)+nodes > 240000:
                _error("Saved rotation dictionary is too large.", "resource_limit")
            if any(type(key) is not str for key in item):
                _error("Saved dictionary keys must be strings.", "invalid_state")
            pending.extend((child, depth+1) for pair in item.items() for child in pair)
        elif isinstance(item, (list, tuple)):
            if len(item) > 12012 or len(pending)+len(item)+nodes > 240000:
                _error("Saved rotation sequence is too large.", "resource_limit")
            pending.extend((child, depth+1) for child in item)
        elif type(item) is str:
            limit = fp.STATE_BYTES if source else 4096
            if len(item) > limit:
                _error("Saved rotation text exceeds its bound.", "resource_limit")
            size += 4*len(item)
        elif item is not None and type(item) not in (bool, int, float):
            _error("Saved rotation must contain JSON primitives.", "invalid_state")
        elif type(item) is int and abs(item) > 2**53-1 or type(item) is float and not math.isfinite(item):
            _error("Saved rotation numbers must be finite exact primitives.", "invalid_state")
    return size


def _same(actual, expected, label):
    """Canonical schema comparison, with roundoff only for floating geometry."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            _error(f"Saved {label} fields differ.", "invalid_state")
        for key in expected:
            _same(actual[key], expected[key], label+"."+key)
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            _error(f"Saved {label} dimensions differ.", "invalid_state")
        for a, b in zip(actual, expected):
            _same(a, b, label)
    elif type(expected) is float:
        if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=2e-8, abs_tol=2e-8):
            _error(f"Saved {label} geometry differs.", "invalid_state")
    elif type(actual) is not type(expected) or actual != expected:
        _error(f"Saved {label} metadata differs.", "invalid_state")


def _json_bound(raw):
    """Bound JSON nesting/token count without constructing the source tree."""
    depth, tokens, string_length = 0, 0, 0
    quoted, escaped = False, False
    for char in raw:
        if quoted:
            string_length += 1
            if string_length > 32768:
                _error("Embedded source scalar text exceeds its bound.", "resource_limit")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted, string_length = True, 0
        elif char in "[{":
            depth += 1
            tokens += 1
        elif char in "]}":
            depth -= 1
        elif char in ",:":
            tokens += 1
        if depth > 24 or tokens > 240000:
            _error("Embedded source JSON exceeds its bounded structure.", "resource_limit")


def _tables_same(actual, expected):
    if set(actual) != set(expected):
        _error("Saved table collection differs.", "invalid_state")
    for key, right in expected.items():
        left = actual[key]
        if not isinstance(left, pd.DataFrame) or list(left.columns) != list(right.columns) or list(left.index) != list(right.index):
            _error("Saved table labels differ.", "invalid_state")
        _same(left.to_numpy().tolist(), right.to_numpy().tolist(), key)


def _plan(source, controls, *, source_bytes=0, query_rows=0):
    _, n, p, d, source_plan = fp._preflight(source, controls["max_bytes"], controls["max_work"])
    if not 2 <= d <= min(6, p-1):
        _error("Weighted rotations admit 2–min(6, p-1) components.", "unsupported_option")
    # Two source validation passes cover downstream saved-map prediction too.
    work = (128*n*p*(p+d)+128*controls["max_iter"]*p*d*d+128*n*p*d+
            128*d**3+64*query_rows*p*d+64*source_bytes)
    # Before serialization the validated source's own conservative state
    # estimate bounds its JSON. Later reuse charges the actual encoded bytes.
    encoded_bound = source_bytes or source_plan["estimated_workspace_bytes"]//4
    size = 2*source_plan["estimated_workspace_bytes"]+20*encoded_bound+512*(n*(p+d)+p*d+32*p*d)+256*query_rows*(p+d+1)
    return {**_admit("frequency CATPCA rotation and complete reuse", size, work,
                    controls["max_bytes"], controls["max_work"]), "planned_work": work}


def _saved_rotation(saved, loadings, controls):
    d, p = loadings.shape[1], len(loadings)
    for name in ("transform", "varimax_transform"):
        matrix = saved[name]
        if not isinstance(matrix, list) or len(matrix) != d or any(not isinstance(row, list) or len(row) != d or any(type(x) not in (int, float) or not math.isfinite(x) for x in row) for row in matrix):
            _error("Saved rotation matrix has invalid shape or cells.", "invalid_state")
    initial = torch.tensor(saved["varimax_transform"], dtype=DT)
    fp._close(initial.T@initial, torch.eye(d, dtype=DT), "saved varimax orthogonality")
    scale = loadings.square().sum(1).sqrt() if controls["normalize"] else torch.ones(p, dtype=DT)
    if bool((scale <= 1e-12).any()):
        _error("Saved Kaiser normalization is degenerate.", "invalid_state")
    base, rotated = loadings/scale[:, None], (loadings/scale[:, None])@initial
    trace = saved["trace"]
    if not isinstance(trace, list) or not 2 <= len(trace) <= controls["max_iter"]+1 or any(not isinstance(row, list) or len(row) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in row) for row in trace):
        _error("Saved rotation trace has invalid dimensions/cells.", "invalid_state")
    if any(type(row[0]) is not int or row[0] != j or row[2] < 0 or j and (row[3] < -1e-9 or abs(row[3]-(row[1]-trace[j-1][1])) > 1e-8) for j, row in enumerate(trace)):
        _error("Saved rotation trace progression is inconsistent.", "invalid_state")
    fp._close(torch.tensor(trace[0][1:], dtype=DT), torch.tensor([_criterion(base), _stationarity(base), 0.], dtype=DT), "initial rotation trace")
    fp._close(torch.tensor(trace[-1][1:3], dtype=DT), torch.tensor([_criterion(rotated), _stationarity(rotated)], dtype=DT), "final rotation trace")
    if _stationarity(rotated) > controls["tol"]+1e-12:
        _error("Saved varimax misses its stationarity tolerance.", "invalid_state")
    for j in range(d-1):
        for k in range(j+1, d):
            x, y = rotated[:, j], rotated[:, k]
            u, v = x.square()-y.square(), 2*x*y
            a, b = float(u.sum()), float(v.sum())
            c, e = float((u.square()-v.square()).sum()), 2*float((u*v).sum())
            angle = .25*math.atan2(e-2*a*b/p, c-(a*a-b*b)/p)
            trial = rotated.clone()
            trial[:, j] = math.cos(angle)*x+math.sin(angle)*y
            trial[:, k] = math.cos(angle)*y-math.sin(angle)*x
            if _criterion(trial)-_criterion(rotated) > 1e-8:
                _error("Saved varimax admits a planar improvement.", "invalid_state")
    return initial, trace, scale


def _build(source, method, controls, power, *, source_json=None, limits=None, query_rows=0, saved=None):
    limits = controls if limits is None else {**controls, **limits}
    plan = _plan(source, limits, source_bytes=len(source_json.encode()) if source_json else 0, query_rows=query_rows)
    state, scores0, loadings, weight, _ = fp._base(source, limits["max_bytes"], limits["max_work"], "cpu")
    if source_json is None:
        source_json = summary_state(source)
        if len(source_json.encode()) > fp.STATE_BYTES:
            _error("Complete weighted CATPCA source exceeds 8 MiB.", "resource_limit")
        plan = _plan(source, limits, source_bytes=len(source_json.encode()), query_rows=query_rows)
    _json_bound(source_json)
    transform, trace, scale = (_varimax(loadings, controls) if saved is None else _saved_rotation(saved, loadings, controls))
    varimax = transform.clone()
    target_sse = None
    if method == METHODS[1]:
        rotated = loadings@transform
        target_base = rotated/scale[:, None] if controls["normalize"] else rotated
        target = torch.sign(target_base)*target_base.abs().pow(power)
        coefficient = torch.linalg.lstsq(rotated, target).solution
        singular = torch.linalg.svdvals(coefficient)
        if not bool(torch.isfinite(coefficient).all()) or float(singular[-1]) <= 1e-8*float(singular[0]):
            _error("Promax target transformation is singular or ill-conditioned.", "degenerate_rotation")
        target_sse = float((rotated@coefficient-target).square().sum())
        raw = transform@coefficient
        raw_inverse = torch.linalg.inv(raw)
        transform = raw*(raw_inverse@raw_inverse.T).diagonal().sqrt()[None, :]
    transform = _identify(loadings, transform)
    if saved is not None:
        saved_transform = torch.tensor(saved["transform"], dtype=DT)
        fp._close(saved_transform, transform, "identified method-specific rotation")
        _same(saved["target_sse"], target_sse, "powered-target residual")
        transform = saved_transform
        target_sse = saved["target_sse"]
    inverse = torch.linalg.inv(transform)
    if float(torch.linalg.cond(transform)) > 1e8 or float(transform.abs().max()) > 1e8 or float(inverse.abs().max()) > 1e8:
        _error("Rotation basis exceeds the bounded condition domain.", "degenerate_rotation")
    pattern, scores, phi = loadings@transform, scores0@inverse.T, inverse@inverse.T
    structure, reconstructed = pattern@phi, scores@pattern.T
    fp._close(scores.T@(weight[:, None]*scores)/weight.sum(), phi, "weighted rotated covariance")
    fp._close(reconstructed, scores0@loadings.T, "weighted reconstructed values")
    names = [f"component_{j+1}" for j in range(state["components"])]
    variables, positions = state["variables"], state["positions"]
    centers, original_centers = [], []
    for descriptor in state["descriptors"]:
        if descriptor["scale"] == "numeric":
            continue
        codes = torch.tensor(state["membership"][descriptor["name"]], dtype=torch.int64)
        means, counts = _means(scores0, codes, len(descriptor["levels"]), weight)
        for label, count, original, changed in zip(descriptor["levels"], counts.tolist(), means.tolist(), (means@inverse.T).tolist()):
            prefix = [descriptor["name"], label[0], label[1], int(count)]
            original_centers.append(prefix+original)
            centers.append(prefix+changed)
    centroid_columns = ["variable", "category_type", "category", "frequency_count"]+names
    rotation_state = dict(version=1, kind=method, source_summary=source_json,
                          source_sha256=hashlib.sha256(source_json.encode()).hexdigest(),
                          controls=controls, power=power, transform=transform.tolist(),
                          varimax_transform=varimax.tolist(), trace=trace, target_sse=target_sse)
    output = TableSet({
        "pattern": table(pattern.tolist(), columns=names, index=variables),
        "structure": table(structure.tolist(), columns=names, index=variables),
        "component_correlations": table(phi.tolist(), columns=names, index=names),
        "transformation": table(transform.tolist(), columns=names, index=names),
        "original_loadings": table(loadings.tolist(), columns=names, index=variables),
        "scores": table([[pos]+row for pos, row in zip(positions, scores.tolist())], columns=["source_position"]+names),
        "reconstruction": table([[pos]+row for pos, row in zip(positions, reconstructed.tolist())], columns=["source_position"]+variables),
        "category_centroids": table(centers, columns=centroid_columns),
        "original_category_centroids": table(original_centers, columns=centroid_columns),
        "rotation_iterations": table(trace, columns=["iteration", "varimax_criterion", "relative_stationarity", "improvement"]),
        "fit": source["fit"].copy(), "sample": source["sample"].copy(),
        "transformed": source["transformed"].copy(), **_mapping_tables(state["descriptors"]),
    }, title="Frequency CATPCA "+method.split("_")[1]+" rotation", method=method,
       variables=variables, components=state["components"], n=len(positions),
       n_input=state["input_nobs"], sample_positions=positions,
       frequency=state["frequency"], frequency_total=state["frequency_total"],
       zero_positions=state["zero_positions"], missing_positions=state["missing_positions"],
       rotation_state=rotation_state, state_sha256=_seal(rotation_state), settings=controls,
       power=power, iterations=len(trace)-1, converged=True, resources=plan,
       sources=SOURCES, dtype="float64", device="cpu", weight_type="frequency",
       normalization="pattern=L T; scores=X T^-T; Phi=T^-1 T^-T; structure=pattern Phi",
       inference="descriptive automatic rotation; original maps fixed; no adaptive uncertainty or global optimum",
       membership="original typed membership retained, including pooled ordinal quantifications")
    output = _save(output)
    _primitive(output.attrs, source=True)
    return output


@resident_cpu
def catpca_varimax_fweight(result: TableSet, *, normalize: bool = True,
                           max_iter: int = 1000, tol: float = 1e-10,
                           max_work: int = WORK, max_bytes: int = BYTES,
                           device: str = "cpu") -> TableSet:
    """Automatic varimax of weighted CATPCA, including original category means.

    Scores and centroids use integer-frequency geometry; scalar maps remain
    fixed. Kaiser normalization is optional. Complete source and trace persist.
    """
    controls = _controls(normalize, max_iter, tol, max_work, max_bytes, device)
    return _build(result, METHODS[0], controls, None)


@resident_cpu
def catpca_promax_fweight(result: TableSet, *, power: float = 4,
                          normalize: bool = True, max_iter: int = 1000,
                          tol: float = 1e-10, max_work: int = WORK,
                          max_bytes: int = BYTES, device: str = "cpu") -> TableSet:
    """Weighted CATPCA varimax followed by a powered-target oblique transform.

    Power is bounded to [1,10]. Pattern, structure and component correlation
    matrices are distinct; weighted score covariance equals the latter.
    """
    controls = _controls(normalize, max_iter, tol, max_work, max_bytes, device)
    return _build(result, METHODS[1], controls, _real(power, "power", 1, 10))


def _checked(result, max_bytes, max_work, device, query_rows=0):
    if device != "cpu" or not isinstance(result, TableSet) or result.attrs.get("method") not in METHODS:
        _error("Supply a complete saved frequency CATPCA rotation on CPU.", "invalid_result")
    state = result.attrs.get("rotation_state")
    fields = {"version", "kind", "source_summary", "source_sha256", "controls", "power", "transform", "varimax_transform", "trace", "target_sse"}
    if not isinstance(state, dict) or set(state) != fields:
        _error("Saved rotation schema is incomplete or unknown.", "invalid_state")
    estimate = _primitive(result.attrs, source=True)
    for frame in result.values():
        if not isinstance(frame, pd.DataFrame) or len(frame) > 12012 or len(frame.columns) > 20:
            _error("Saved rotation table is too large.", "resource_limit")
        estimate += 64*frame.size
        if estimate > STATE_BYTES:
            _error("Saved rotation tables exceed their bound.", "resource_limit")
        # Only bounded tables are materialized after their combined admission.
    _admit("saved weighted rotation parsing", 4*estimate, 8*estimate, max_bytes, max_work)
    try:
        if type(state["version"]) is not int or state["version"] != 1 or state["kind"] != result.attrs["method"]:
            raise ValueError("Wrong version or kind")
        raw = state["source_summary"]
        if type(raw) is not str or len(raw.encode()) > fp.STATE_BYTES or hashlib.sha256(raw.encode()).hexdigest() != state["source_sha256"] or _seal(state) != result.attrs["state_sha256"]:
            raise ValueError("Integrity mismatch")
        controls = _controls(**state["controls"], device=device)
        if state["kind"] == METHODS[0]:
            if state["power"] is not None:
                raise ValueError("Varimax power")
            power = None
        else:
            power = _real(state["power"], "power", 1, 10)
        _json_bound(raw)
        source = restore_summary(raw)
        reuse_plan = _plan(source, {**controls, "max_work": max_work, "max_bytes": max_bytes},
                           source_bytes=len(raw.encode()), query_rows=query_rows)
        expected = _build(source, state["kind"], controls, power, source_json=raw, saved=state)
        actual_attrs, expected_attrs = dict(result.attrs), dict(expected.attrs)
        for attrs in (actual_attrs, expected_attrs):
            # Resource ceiling records the fitting caller's budget; geometry
            # and the exact planned operations are independent of that ceiling.
            attrs["resources"] = {k: v for k, v in attrs["resources"].items() if k != "budget_bytes"}
        _same(actual_attrs, expected_attrs, "rotation attrs")
        _tables_same(result, expected)
        return source, torch.tensor(state["transform"], dtype=DT), reuse_plan
    except AnalysisError as exc:
        if exc.code in ("resource_limit", "workspace_limit", "invalid_state"):
            raise
        raise AnalysisError("invalid_state", "Saved weighted rotation source/options are invalid.") from exc
    except (TypeError, KeyError, ValueError, OverflowError, RecursionError, torch.linalg.LinAlgError) as exc:
        raise AnalysisError("invalid_state", "Saved weighted rotation is inconsistent.") from exc


@resident_cpu
def catpca_rotated_predict_fweight(result: TableSet, data: pd.DataFrame, *,
                                  missing: str = "raise", max_work: int = WORK,
                                  max_bytes: int = BYTES, device: str = "cpu") -> TableSet:
    """Apply complete saved weighted maps and rotation to physical query rows.

    Queries require no frequency column. Restoration checks the saved geometry
    without rerunning either optimizer; unknown categories are refused.
    """
    if not isinstance(data, pd.DataFrame) or not 1 <= len(data) <= 3000:
        _error("Projection requires a DataFrame of 1–3000 physical rows.", "resource_limit")
    source, transform, plan = _checked(result, max_bytes, max_work, device, len(data))
    projected = fp.catpca_fweight_predict(source, data, missing=missing, max_work=max_work, max_bytes=max_bytes, device=device)
    original = torch.tensor(projected["scores"].iloc[:, 1:].to_numpy().tolist(), dtype=DT).reshape(-1, transform.shape[0])
    scores = original@torch.linalg.inv(transform).T
    pattern = torch.tensor(result["pattern"].to_numpy().tolist(), dtype=DT)
    names = list(result["pattern"].columns)
    positions = projected.attrs["sample_positions"]
    return _save(TableSet({
        "scores": table([[pos]+row for pos, row in zip(positions, scores.tolist())], columns=["source_position"]+names),
        "transformed": projected["transformed"].copy(),
        "reconstruction": table([[pos]+row for pos, row in zip(positions, (scores@pattern.T).tolist())], columns=["source_position"]+source.attrs["variables"]),
    }, title="Saved weighted CATPCA rotated projection", method="catpca_rotated_predict_fweight",
       n=len(positions), n_input=len(data), sample_positions=positions, missing=missing,
       source_state_sha256=result.attrs["state_sha256"], resources=plan,
       dtype="float64", device="cpu", weight_type="frequency calibration; physical query rows",
       inference="descriptive saved-map projection; no refit or adaptive uncertainty"))
