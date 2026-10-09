"""Count-compressed MCA and multiset optimal scaling on resident CPU float64.

Integer frequencies mean literal replication. Geometry, category means and
normalization use these counts; numerical arrays retain only physical rows.
"""
from __future__ import annotations

import json
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from ..core import TableSet, table
from ..resident_cpu import resident_cpu
from ..summary_state import saved_summary, summary_state
from . import frequency as f
from .scaling import _axes, _integer, _key, _orient

DT, BYTES, WORK = f.DT, f.BYTES, f.WORK
STATE_BYTES = 8 * 1024**2
SOURCES = ["https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/catpca.pdf",
           "https://www.jstatsoft.org/article/view/v031i04"]
_COMMON = {"version", "kind", "variables", "frequency", "frequencies", "frequency_total",
           "positions", "input_nobs", "zero_positions", "missing_positions", "row_labels",
           "levels", "codes", "n_components"}
_MCA = {"category_mass", "category_standard", "eigenvalues", "rank", "scores"}
_OVERALS = {"sets", "scales", "orders", "quantifications", "loadings", "scores", "set_scores",
            "objective", "settings", "trace", "starts", "selected_start"}


def _device(device):
    if type(device) is not str or device != "cpu":
        f._error("Frequency categorical geometry supports resident CPU float64 only.", "unsupported_option")


def _variables(variables):
    if (not isinstance(variables, (list, tuple)) or not 2 <= len(variables) <= 12
            or any(type(name) is not str or not 1 <= len(name) <= 128 for name in variables)
            or len(set(variables)) != len(variables)):
        f._error("Select 2–12 distinct bounded variable names.")
    return list(variables)


def _levels(frame, variables, scales=None, orders=None):
    levels, codes = [], []
    for name in variables:
        observed = [f._atom(value) for value in frame[name].tolist()]
        mapping, unique = {}, []
        for value in observed:
            if _key(value) not in mapping:
                mapping[_key(value)] = len(unique)
                unique.append(value)
        numeric = scales is not None and scales[name] == "numeric"
        if not 2 <= len(unique) <= (3000 if numeric else 32):
            f._error("Each variable needs 2–32 retained categorical levels or nonconstant numeric values.", "degenerate_transform")
        if numeric and any(type(v) not in (int, float) or abs(v) > 1e100 for v in observed):
            f._error("Numeric scaling requires finite real values of magnitude at most 1e100.", "invalid_data")
        if scales is not None and scales[name] == "ordinal":
            declared = orders[name]
            if not isinstance(declared, (list, tuple)) or len(declared) != len(unique):
                f._error("Every ordinal order must list exactly its retained typed categories.")
            unique = [f._atom(v) for v in declared]
            if len(set(map(_key, unique))) != len(unique) or set(map(_key, unique)) != set(mapping):
                f._error("Ordinal order differs from the retained typed categories.")
            mapping = {_key(v): i for i, v in enumerate(unique)}
        levels.append(unique)
        codes.append([mapping[_key(v)] for v in observed])
    return levels, codes


def _state(sample, data, variables, frequency, levels, codes, dimensions, kind):
    return dict(version=1, kind=kind, variables=variables, frequency=frequency,
                frequencies=sample["counts"], frequency_total=sample["frequency_total"],
                positions=sample["positions"], input_nobs=sample["input_nobs"],
                zero_positions=sample["zero_positions"], missing_positions=sample["missing_positions"],
                row_labels=[f._atom(data.index[i]) for i in sample["positions"]],
                levels=levels, codes=codes, n_components=dimensions)


def _text(state):
    return sum(len(json.dumps(value, ensure_ascii=True)) for group in state["levels"] for value in group) \
        + sum(len(json.dumps(value, ensure_ascii=True)) for value in state["row_labels"])


def _plan(state, max_bytes, max_work, *, query_n=0, estimate=0):
    n, p, d = len(state["positions"]), len(state["variables"]), state["n_components"]
    k = sum(map(len, state["levels"]))
    if state["kind"] == "mca_fweight":
        work = 4*n*k*min(n, k)+4*k**3+16*k*k*d
        size = 8*(16*n*k+16*k*k+12*n*d)+512*n*(p+2)+12*_text(state)
    else:
        starts, iterations = state["settings"]["n_starts"], state["settings"]["max_iter"]
        sets = len(state["sets"])
        work = starts*(iterations+2)*(64*n*p*d*d+64*k*d*d+32*n*d*d)
        size = 8*(32*n*p*d+16*k*d+12*starts*n*d*(sets+1)) \
            + 1024*starts*(iterations+1)+512*n*(p+sets+2)+12*_text(state)
    if query_n:
        work += 16*query_n*k*d+8*query_n*p
        size += 256*query_n*(p+2)+8*(6*query_n*k+8*query_n*d)
    size += 4*estimate
    # Full summaries and the downstream verification domain have an explicit
    # conservative size gate, independent of the numerical workspace budget.
    portable = 32768+128*n*(p+d+4)+128*k*(d+3)+12*_text(state)
    if state["kind"] == "overals_fweight":
        portable += 128*n*d*len(state["sets"])+512*state["settings"]["n_starts"]*(state["settings"]["max_iter"]+1)
    if portable > STATE_BYTES:
        f._error("Complete frequency geometry exceeds the 8 MiB reusable state domain.", "resource_limit")
    plan = f._admit("frequency categorical geometry and verification", size, work, max_bytes, max_work)
    plan["planned_work"] = work
    return plan


def _indicator(state):
    n, k = len(state["positions"]), sum(map(len, state["levels"]))
    z = torch.zeros((n, k), dtype=DT)
    offset = 0
    for codes, levels in zip(state["codes"], state["levels"]):
        z[torch.arange(n), torch.tensor(codes, dtype=torch.int64)+offset] = 1.
        offset += len(levels)
    return z


def _tables(state):
    n, d = len(state["positions"]), state["n_components"]
    axes, labels = _axes(d), state["row_labels"]
    sample = table(list(zip(state["positions"], state["frequencies"])), columns=["source_position", "frequency"])
    if state["kind"] == "mca_fweight":
        mass = torch.tensor(state["category_mass"], dtype=DT)
        standard = torch.tensor(state["category_standard"], dtype=DT)
        eigen = torch.tensor(state["eigenvalues"], dtype=DT)
        row = torch.tensor(state["scores"], dtype=DT)
        category = standard*eigen[:d].sqrt()
        rows, offset = [], 0
        for name, levels in zip(state["variables"], state["levels"]):
            for value in levels:
                rows.append([name, value, float(mass[offset]), *category[offset].tolist()])
                offset += 1
        return dict(sample=sample, row_coordinates=table(row.tolist(), columns=axes, index=labels),
                    category_coordinates=table(rows, columns=["variable", "category", "mass"]+axes),
                    category_standard=table(standard.tolist(), columns=axes),
                    inertia=table([[i+1, float(v), float(v/eigen.sum())] for i, v in enumerate(eigen)],
                                  columns=["dimension", "raw_inertia", "raw_share"]),
                    row_contributions=table((row.square()*torch.tensor(state["frequencies"], dtype=DT)[:, None]
                                             /state["frequency_total"]/eigen[:d]).tolist(), columns=axes, index=labels),
                    category_contributions=table((category.square()*mass[:, None]/eigen[:d]).tolist(), columns=axes))
    qrows, arows = [], []
    for name, levels, quant, loading in zip(state["variables"], state["levels"], state["quantifications"], state["loadings"]):
        for category, values in zip(levels, quant):
            qrows.extend([name, category, j+1, value] for j, value in enumerate(values))
        arows.extend([name, j+1, *values] for j, values in enumerate(loading))
    x, weight = torch.tensor(state["scores"], dtype=DT), f._weights(state["frequencies"])
    fitrows, scorerows = [], []
    for j, raw in enumerate(state["set_scores"]):
        fit = torch.tensor(raw, dtype=DT)
        fitrows.append([j, float(((x-fit).square()*weight[:, None]).sum()/weight.sum()),
                        *torch.diag(x.T@(weight[:, None]*fit)/weight.sum()).tolist()])
        scorerows.extend([j, state["positions"][i], labels[i], *raw[i]] for i in range(n))
    return dict(sample=sample, object_scores=table(state["scores"], columns=axes, index=labels),
                category_quantifications=table(qrows, columns=["variable", "category", "copy", "quantification"]),
                variable_loadings=table(arows, columns=["variable", "copy"]+axes),
                set_scores=table(scorerows, columns=["set", "source_position", "row_label"]+axes),
                set_fit=table(fitrows, columns=["set", "loss_per_person"]+axes),
                iterations=table(state["trace"], columns=["start", "iteration", "loss", "decrease", "start_converged"]),
                starts=table(state["starts"], columns=["start", "loss", "iterations", "converged", "status"]))


def _result(state, sample, plan, missing):
    attrs = dict(method=state["kind"], frequency_state=state, state_sha256=f._seal(state),
                 nobs=len(state["positions"]), input_rows=state["input_nobs"], variables=state["variables"],
                 sample_positions=state["positions"], row_labels=state["row_labels"],
                 zero_positions=state["zero_positions"], missing_positions=state["missing_positions"],
                 frequency=state["frequency"], frequency_total=state["frequency_total"],
                 n_components=state["n_components"], missing=missing, resources=plan,
                 input_preparation=sample["workspace"], declared_work=plan["planned_work"],
                 device="cpu", dtype="float64", weight_type="frequency", converged=True,
                 global_optimum=False, inference=False, stata_parity_validated=False, sources=SOURCES)
    if state["kind"] == "mca_fweight":
        attrs.update(raw_total_inertia=sum(state["eigenvalues"]), adjusted_inertia=False,
                     normalization="frequency row masses and raw complete-disjunctive inertia")
    else:
        attrs.update(objective=state["objective"], selected_start=state["selected_start"], settings=state["settings"],
                     normalization="weighted mean zero; X'WX/sum(w)=I; equal sets",
                     solution="best converged declared start; local stationary geometry")
    output = f._save(TableSet(_tables(state), title="Frequency-weighted "+("MCA" if state["kind"] == "mca_fweight" else "multiset optimal scaling"), **attrs))
    summary_state(output)
    return output


@resident_cpu
def mca_fweight(data, variables, n_components=2, *, frequency, missing="drop",
                max_bytes=BYTES, max_work=WORK, device="cpu"):
    """Raw MCA with nonnegative integer replication counts, without expansion.

    Complete active physical rows define typed categories. Saved calibration
    gives supplementary principal coordinates without query frequencies.
    """
    _device(device)
    variables = _variables(variables)
    d = _integer(n_components, "n_components", 1, 12)
    sample = f._sample(data, variables, frequency, missing, max_bytes, max_work, operation="frequency MCA")
    levels, codes = _levels(sample["frame"], variables)
    state = _state(sample, data, variables, frequency, levels, codes, d, "mca_fweight")
    n, p, k = len(state["positions"]), len(variables), sum(map(len, levels))
    if d > min(n-1, k-p):
        f._error("Requested MCA axes exceed the centered disjunctive rank.")
    plan = _plan(state, max_bytes, max_work)
    weight, z = f._weights(state["frequencies"]), _indicator(state)
    mass = (z*weight[:, None]).sum(0)/(weight.sum()*p)
    profiles = z/p
    standardized = (profiles-mass)/mass.sqrt()*(weight/weight.sum()).sqrt()[:, None]
    _, singular, vt = torch.linalg.svd(standardized, full_matrices=False)
    rank = int((singular > singular[0]*1e-12).sum())
    if d > rank:
        f._error("Requested MCA axes contain zero inertia.", "singular_design")
    standard = vt[:d].T/mass.sqrt()[:, None]
    scores, standard = _orient((profiles-mass)@standard, standard)
    state.update(category_mass=mass.tolist(), category_standard=standard.tolist(),
                 eigenvalues=singular[:rank].square().tolist(), rank=rank, scores=scores.tolist())
    return _result(state, sample, plan, missing)


def _primitive(value):
    if (hasattr(value, "item") and type(value).__module__.startswith("numpy")
            and getattr(value, "ndim", 0) == 0):
        value = value.item()
    if value is not None and type(value) not in (str, int, float, bool):
        f._error("Saved geometry requires bounded primitive cells and labels.", "invalid_state")
    if type(value) is int and abs(value) > 2**53-1 or type(value) is float and not math.isfinite(value):
        f._error("Saved geometry numbers must be finite and bounded.", "invalid_state")
    if type(value) is str and len(value) > 4096:
        f._error("Saved geometry text exceeds the bounded domain.", "resource_limit")
    return 64+(len(json.dumps(value, ensure_ascii=True)) if type(value) is str else 0)


def _preflight(result, kind, max_bytes, max_work, *, query_n=0):
    if not isinstance(result, TableSet) or result.attrs.get("method") != kind:
        f._error("Supply a complete fitted/restored frequency geometry.", "invalid_state")
    state = result.attrs.get("frequency_state")
    fields = _COMMON | (_MCA if kind == "mca_fweight" else _OVERALS)
    if not isinstance(state, dict) or set(state) != fields:
        f._error("Saved frequency geometry has an unknown or incomplete schema.", "invalid_state")
    expected = {"sample", "settings"} | ({"row_coordinates", "category_coordinates", "category_standard", "inertia", "row_contributions", "category_contributions"}
                                          if kind == "mca_fweight" else {"object_scores", "category_quantifications", "variable_loadings", "set_scores", "set_fit", "iterations", "starts"})
    if set(result) != expected:
        f._error("Saved frequency geometry tables are incomplete or unknown.", "invalid_state")
    estimate, nodes, pending = 0, 0, [(result.attrs, 0)]
    while pending:
        value, depth = pending.pop()
        nodes += 1
        estimate += 64
        if depth > 16 or nodes > 500000:
            f._error("Saved metadata exceeds its bounded shape.", "resource_limit")
        if type(value) is dict:
            if len(value) > 128 or any(type(key) is not str for key in value):
                f._error("Saved metadata keys exceed the closed primitive domain.", "invalid_state")
            pending.extend((child, depth+1) for pair in value.items() for child in pair)
        elif type(value) in (list, tuple):
            if len(value) > 12012:
                f._error("Saved metadata dimension is excessive.", "resource_limit")
            pending.extend((child, depth+1) for child in value)
        else:
            estimate += _primitive(value)
        if estimate > STATE_BYTES:
            f._error("Saved metadata exceeds 8 MiB.", "resource_limit")
    for frame in result.values():
        if not isinstance(frame, pd.DataFrame) or len(frame) > 36000 or len(frame.columns) > 16:
            f._error("Saved table dimensions are excessive.", "resource_limit")
        estimate += 64*(frame.size+len(frame.index)+len(frame.columns))
        if estimate > STATE_BYTES:
            f._error("Saved table cells exceed the state domain.", "resource_limit")
        for label in [*frame.index, *frame.columns]:
            estimate += _primitive(label)
        for row in frame.itertuples(index=False, name=None):
            for cell in row:
                estimate += _primitive(cell)
                if estimate > STATE_BYTES:
                    f._error("Saved table text exceeds the state domain.", "resource_limit")
    try:
        _variables(state["variables"])
        n, p = len(state["positions"]), len(state["variables"])
        d = state["n_components"]
        if (state["version"] != 1 or type(state["version"]) is not int or state["kind"] != kind
                or not 4 <= n <= 3000 or type(d) is not int or not 1 <= d <= 12
                or type(state["input_nobs"]) is not int or not n <= state["input_nobs"] <= 3000
                or type(state["frequency"]) is not str or not 1 <= len(state["frequency"]) <= 128
                or state["frequency"] in state["variables"]):
            raise ValueError()
        frequency = state["frequencies"]
        if (not isinstance(frequency, list) or len(frequency) != n
                or any(type(v) is not int or not 1 <= v <= f.MAX_TOTAL for v in frequency)
                or type(state["frequency_total"]) is not int or state["frequency_total"] != sum(frequency)
                or not 1 <= sum(frequency) <= f.MAX_TOTAL):
            raise ValueError()
        for name in ("positions", "zero_positions", "missing_positions"):
            values = state[name]
            if (not isinstance(values, list) or values != sorted(set(values))
                    or any(type(v) is not int or not 0 <= v < state["input_nobs"] for v in values)):
                raise ValueError()
        if sorted(state["positions"]+state["zero_positions"]+state["missing_positions"]) != list(range(state["input_nobs"])):
            raise ValueError()
        if not isinstance(state["row_labels"], list) or len(state["row_labels"]) != n:
            raise ValueError()
        for v in state["row_labels"]:
            if type(v) not in (str, int, float, bool) or f._atom(v) != v:
                raise ValueError()
        if not isinstance(state["levels"], list) or len(state["levels"]) != p or not isinstance(state["codes"], list) or len(state["codes"]) != p:
            raise ValueError()
        if kind == "overals_fweight":
            _overals_controls(state["sets"], state["scales"], state["orders"], d, **state["settings"])
            if [name for group in state["sets"] for name in group] != state["variables"]:
                raise ValueError()
        for j, (levels, codes) in enumerate(zip(state["levels"], state["codes"])):
            numeric = kind == "overals_fweight" and state["scales"][state["variables"][j]] == "numeric"
            if (not isinstance(levels, list) or not 2 <= len(levels) <= (3000 if numeric else 32)
                    or any(type(v) not in (str, int, float, bool) or f._atom(v) != v for v in levels)
                    or len(set(map(_key, levels))) != len(levels)
                    or not isinstance(codes, list) or len(codes) != n
                    or any(type(v) is not int or not 0 <= v < len(levels) for v in codes)
                    or set(codes) != set(range(len(levels)))):
                raise ValueError()
            if numeric and any(type(v) not in (int, float) or abs(v) > 1e100 for v in levels):
                raise ValueError()
            if kind == "overals_fweight" and state["scales"][state["variables"][j]] == "ordinal" and list(map(_key, state["orders"][state["variables"][j]])) != list(map(_key, levels)):
                raise ValueError()
        _matrix(state["scores"], n, d)
        k = sum(map(len, state["levels"]))
        if kind == "mca_fweight":
            _matrix(state["category_standard"], k, d)
            if (type(state["rank"]) is not int or not d <= state["rank"] <= min(n-1, k-p)
                    or not isinstance(state["eigenvalues"], list) or len(state["eigenvalues"]) != state["rank"]
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in state["eigenvalues"])
                    or not isinstance(state["category_mass"], list) or len(state["category_mass"]) != k
                    or any(type(v) not in (int, float) or not 0 < v < 1 for v in state["category_mass"])):
                raise ValueError()
        else:
            if len(state["quantifications"]) != p or len(state["loadings"]) != p or len(state["set_scores"]) != len(state["sets"]):
                raise ValueError()
            for name, levels, q, a in zip(state["variables"], state["levels"], state["quantifications"], state["loadings"]):
                width = d if state["scales"][name] == "multiple_nominal" else 1
                _matrix(q, len(levels), width)
                _matrix(a, width, d)
            for raw in state["set_scores"]:
                _matrix(raw, n, d)
            if type(state["objective"]) not in (int, float) or not math.isfinite(state["objective"]) or not -1e-9 <= state["objective"] <= d+1e-8:
                raise ValueError()
        attrs = result.attrs
        if (type(attrs.get("nobs")) is not int or attrs["nobs"] != n
                or type(attrs.get("input_rows")) is not int or attrs["input_rows"] != state["input_nobs"]
                or any(json.dumps(attrs.get(a), sort_keys=True) != json.dumps(state[b], sort_keys=True) for a, b in (("variables", "variables"), ("sample_positions", "positions"),
                    ("row_labels", "row_labels"), ("frequency", "frequency"), ("frequency_total", "frequency_total"),
                    ("n_components", "n_components"), ("zero_positions", "zero_positions"), ("missing_positions", "missing_positions")))
                or attrs.get("converged") is not True or attrs.get("device") != "cpu" or attrs.get("dtype") != "float64"
                or attrs.get("weight_type") != "frequency" or attrs.get("inference") is not False
                or attrs.get("stata_parity_validated") is not False or attrs.get("missing") not in ("drop", "raise")):
            raise ValueError()
        plan = _plan(state, max_bytes, max_work, query_n=query_n, estimate=estimate)
        if type(attrs.get("state_sha256")) is not str or f._seal(state) != attrs["state_sha256"]:
            raise ValueError()
        return state, plan
    except AnalysisError as exc:
        if exc.code in ("resource_limit", "workspace_limit", "invalid_resource_budget"):
            raise
        raise AnalysisError("invalid_state", "Saved frequency geometry structure is inconsistent.") from exc
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError("invalid_state", "Saved frequency geometry structure is inconsistent.") from exc


def _matrix(raw, rows, columns):
    if (not isinstance(raw, list) or len(raw) != rows or any(not isinstance(row, list) or len(row) != columns for row in raw)
            or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1e100 for row in raw for v in row)):
        raise ValueError("Invalid bounded matrix")


def _close(actual, expected, message, *, tolerance=2e-8):
    if actual.shape != expected.shape or not bool(torch.isfinite(actual).all()) or not torch.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        f._error("Saved frequency geometry disagrees with "+message+".", "invalid_state")


@resident_cpu
def _checked(result, kind, max_bytes=BYTES, max_work=WORK, *, query_n=0):
    """Bound primitive shape, admit aggregate work, then certify saved geometry."""
    state, plan = _preflight(result, kind, max_bytes, max_work, query_n=query_n)
    p, d = len(state["variables"]), state["n_components"]
    w, x = f._weights(state["frequencies"]), torch.tensor(state["scores"], dtype=DT)
    if kind == "mca_fweight":
        z = _indicator(state)
        mass = (z*w[:, None]).sum(0)/(w.sum()*p)
        standard = torch.tensor(state["category_standard"], dtype=DT)
        eigen = torch.tensor(state["eigenvalues"], dtype=DT)
        standardized = (z/p-mass)/mass.sqrt()*(w/w.sum()).sqrt()[:, None]
        singular = torch.linalg.svdvals(standardized)
        rank = int((singular > singular[0]*1e-12).sum())
        if rank != state["rank"]:
            f._error("Saved MCA rank disagrees with active memberships.", "invalid_state")
        _close(torch.tensor(state["category_mass"], dtype=DT), mass, "category masses")
        _close(eigen, singular[:rank].square(), "complete raw inertia")
        _close(standard.T@(mass[:, None]*standard), torch.eye(d, dtype=DT), "mass orthogonality")
        _close(mass@standard, torch.zeros(d, dtype=DT), "category centering")
        v = mass.sqrt()[:, None]*standard
        _close(standardized.T@standardized@v, v*eigen[:d], "leading eigenspace")
        _close(x, (z/p-mass)@standard, "training principal coordinates")
        if abs(result.attrs.get("raw_total_inertia", -1)-float(eigen.sum())) > 2e-8 or result.attrs.get("adjusted_inertia") is not False:
            f._error("Saved MCA inertia metadata disagree.", "invalid_state")
    else:
        _check_overals(state, result, w, x)
    expected = _tables(state)
    expected["settings"] = saved_summary(TableSet({}, **result.attrs))["settings"]
    for name, frame in expected.items():
        actual = result[name]
        if name == "settings":
            # Generic JSON sorts attribute keys, while the complete settings
            # table preserves its original row order. Certify the keyed cells.
            if (list(actual.columns) != ["setting", "json"] or actual.setting.duplicated().any()
                    or dict(actual.to_numpy().tolist()) != dict(frame.to_numpy().tolist())):
                f._error("Saved settings table disagrees with complete geometry.", "invalid_state")
            continue
        if (json.dumps(list(actual.columns)) != json.dumps(list(frame.columns))
                or json.dumps(list(actual.index)) != json.dumps(list(frame.index))
                or actual.index.names != frame.index.names or actual.columns.names != frame.columns.names
                or json.dumps(actual.to_numpy().tolist(), allow_nan=False) != json.dumps(frame.to_numpy().tolist(), allow_nan=False)):
            f._error("Saved "+name+" table disagrees with complete geometry.", "invalid_state")
    return state, plan


@resident_cpu
def mca_fweight_project(result, data, *, missing="raise", max_bytes=BYTES, max_work=WORK, device="cpu"):
    """Project supplementary rows using certified weighted MCA calibration."""
    _device(device)
    if not isinstance(data, pd.DataFrame) or not 1 <= len(data) <= 3000:
        f._error("Projection needs 1–3000 resident physical rows.", "resource_limit")
    state, plan = _checked(result, "mca_fweight", max_bytes, max_work, query_n=len(data))
    variables = state["variables"]
    if missing not in ("drop", "raise") or any(list(data.columns).count(name) != 1 for name in variables):
        f._error("Projection requires each trained variable exactly once and explicit missing policy.")
    selected = data.loc[:, variables]
    missing_mask = selected.isna().any(axis=1).tolist()
    if missing == "raise" and any(missing_mask):
        f._error("Projection variables contain missing values.", "missing_data")
    positions = [i for i, absent in enumerate(missing_mask) if not absent]
    if not positions:
        f._error("At least one complete projection row is required.", "insufficient_sample")
    labels = [f._atom(data.index[i]) for i in positions]
    values = [[f._atom(v) for v in row] for row in selected.iloc[positions].itertuples(index=False, name=None)]
    k, n = sum(map(len, state["levels"])), len(positions)
    z, offset = torch.zeros((n, k), dtype=DT), 0
    for j, levels in enumerate(state["levels"]):
        mapping = {_key(v): h for h, v in enumerate(levels)}
        try:
            codes = [mapping[_key(row[j])] for row in values]
        except KeyError as exc:
            raise AnalysisError("unknown_category", "Projection contains an untrained typed category.") from exc
        z[torch.arange(n), torch.tensor(codes, dtype=torch.int64)+offset] = 1.
        offset += len(levels)
    scores = (z/len(variables)-torch.tensor(state["category_mass"], dtype=DT))@torch.tensor(state["category_standard"], dtype=DT)
    output = f._save(TableSet({"row_coordinates": table(scores.tolist(), columns=_axes(state["n_components"]), index=labels)},
        title="Supplementary frequency MCA coordinates", method="mca_fweight_project", source_state_sha256=result.attrs["state_sha256"],
        nobs=n, input_rows=len(data), sample_positions=positions, row_labels=labels, input_categories=values,
        resources=plan, declared_work=plan["planned_work"], missing=missing, device="cpu", dtype="float64",
        inference=False, weight_type="saved frequency calibration; supplementary rows have no fitted mass"))
    summary_state(output)
    return output


def _overals_controls(sets, scales, orders, dimensions, n_starts, seed, max_iter, tol):
    if (not isinstance(sets, (list, tuple)) or not 2 <= len(sets) <= 12
            or any(not isinstance(group, (list, tuple)) or not group for group in sets)):
        f._error("Supply at least two nonempty disjoint variable sets.")
    variables = _variables([name for group in sets for name in group])
    if (not isinstance(scales, dict) or set(scales) != set(variables)
            or any(type(value) is not str or value not in ("multiple_nominal", "nominal", "ordinal", "numeric") for value in scales.values())):
        f._error("Explicitly scale every variable as multiple_nominal, nominal, ordinal or numeric.")
    ordinal = {name for name in variables if scales[name] == "ordinal"}
    orders = {} if orders is None else orders
    if not isinstance(orders, dict) or set(orders) != ordinal:
        f._error("orders must name exactly the ordinal variables.")
    for order in orders.values():
        if not isinstance(order, (list, tuple)) or not 2 <= len(order) <= 32:
            f._error("Each ordinal order needs 2–32 explicit categories.")
    _integer(dimensions, "n_components", 1, 6)
    _integer(n_starts, "n_starts", 1, 12)
    _integer(seed, "seed", 0, 2**31-1)
    _integer(max_iter, "max_iter", 1, 1000)
    if type(tol) not in (int, float) or not math.isfinite(tol) or not 1e-12 <= tol <= 1e-3:
        f._error("tol must be finite from 1e-12 to 1e-3.")
    return variables, orders


def _unit(target, w):
    centered = target-(target*w[:, None]).sum(0)/w.sum()
    u, singular, vt = torch.linalg.svd(centered*(w/w.sum()).sqrt()[:, None], full_matrices=False)
    if float(singular[-1]) <= 1e-12*max(float(singular[0]), 1.):
        f._error("Weighted compromise scores lose rank.", "singular_design")
    return (u@vt)/(w/w.sum()).sqrt()[:, None]


def _scalar(raw, codes, w):
    _, counts = f._means(torch.zeros(len(codes), dtype=DT), codes, len(raw), w)
    centered = raw-(raw*counts).sum()/w.sum()
    norm = torch.sqrt((centered.square()*counts).sum()/w.sum())
    if float(norm) < 1e-12:
        return None
    return centered/norm


def _start(state, start):
    n, d, variables = len(state["positions"]), state["n_components"], state["variables"]
    w, settings = f._weights(state["frequencies"]), state["settings"]
    codes = [torch.tensor(raw, dtype=torch.int64) for raw in state["codes"]]
    groups = [[variables.index(name) for name in group] for group in state["sets"]]
    generator = torch.Generator(device="cpu").manual_seed((settings["seed"]+start) % (2**31))
    qs, loads, contributions, initial = [], [], [], []
    for name, c, levels in zip(variables, codes, state["levels"]):
        raw = torch.tensor(levels if state["scales"][name] == "numeric" else list(range(len(levels))), dtype=DT)
        scalar = _scalar(raw, c, w)
        if scalar is None:
            f._error("Weighted numeric values have collapsed population variance.", "degenerate_transform")
        if state["scales"][name] == "numeric":
            # Keep the same relative/absolute nondegeneracy contract as the
            # other weighted optimal-scaling numerical transformations.
            f._standardize(raw[c], w)
        initial.append(scalar[c])
        if state["scales"][name] == "multiple_nominal":
            qs.append(torch.zeros((len(levels), d), dtype=DT))
            loads.append(torch.eye(d, dtype=DT))
        else:
            qs.append(scalar[:, None])
            loads.append(torch.zeros((1, d), dtype=DT))
        contributions.append(torch.zeros((n, d), dtype=DT))
    if start == 0:
        candidates = []
        for name, c, values in zip(variables, codes, initial):
            if state["scales"][name] == "multiple_nominal":
                candidates.extend((c == k).to(DT) for k in range(len(state["levels"][variables.index(name)])-1))
            else:
                candidates.append(values)
        basis = []
        for candidate in candidates:
            centered = candidate-(candidate*w).sum()/w.sum()
            trial = torch.stack(basis+[centered], dim=1)*(w/w.sum()).sqrt()[:, None]
            singular = torch.linalg.svdvals(trial)
            if float(singular[-1]) > 1e-10:
                basis.append(centered)
                if len(basis) == d:
                    break
        target = torch.stack(basis, dim=1) if len(basis) == d else torch.randn((n, d), generator=generator, dtype=DT)
    else:
        target = torch.randn((n, d), generator=generator, dtype=DT)
    x = _unit(target, w)
    previous, trace, converged = float(d), [], False
    for iteration in range(1, settings["max_iter"]+1):
        for group in groups:
            fitted = sum((contributions[i] for i in group), torch.zeros_like(x))
            for i in group:
                residual = x-fitted+contributions[i]
                scale = state["scales"][variables[i]]
                if scale == "multiple_nominal":
                    q, _ = f._means(residual, codes[i], len(state["levels"][i]), w)
                    a = torch.eye(d, dtype=DT)
                elif scale == "numeric":
                    q = qs[i]
                    a = q[codes[i]].T@(w[:, None]*residual)/w.sum()
                elif scale == "nominal":
                    centroid, counts = f._means(residual, codes[i], len(state["levels"][i]), w)
                    u, _, _ = torch.linalg.svd(centroid*(counts/w.sum()).sqrt()[:, None], full_matrices=False)
                    scalar = _scalar(u[:, 0]/counts.sqrt(), codes[i], w)
                    q = qs[i] if scalar is None else scalar[:, None]
                    a = q[codes[i]].T@(w[:, None]*residual)/w.sum()
                else:
                    q = qs[i]
                    a = q[codes[i]].T@(w[:, None]*residual)/w.sum()
                    if float(a.square().sum()) > 1e-18:
                        target = (residual@a.T/a.square().sum()).flatten()
                        means, counts = f._means(target, codes[i], len(state["levels"][i]), w)
                        scalar = _scalar(f._pava(means, counts), codes[i], w)
                        if scalar is not None:
                            q = scalar[:, None]
                            a = q[codes[i]].T@(w[:, None]*residual)/w.sum()
                new = q[codes[i]]@a
                fitted += new-contributions[i]
                contributions[i], qs[i], loads[i] = new, q, a
        fits = [sum((contributions[i] for i in group), torch.zeros_like(x)) for group in groups]
        x = _unit(sum(fits, torch.zeros_like(x))/len(groups), w)
        loss = float(sum(((x-fit).square()*w[:, None]).sum() for fit in fits)/(w.sum()*len(groups)))
        change = previous-loss
        if change < -1e-9*max(1., previous):
            f._error("Weighted OVERALS objective increased beyond roundoff.", "numerical_failure")
        trace.append([start, iteration, loss, change])
        if change <= settings["tol"]*max(1., previous):
            converged = True
            break
        previous = loss
    return dict(scores=x, fits=fits, quantifications=qs, loadings=loads, loss=loss, trace=trace, converged=converged)


@resident_cpu
def overals_fweight(data, sets, n_components=2, *, frequency, scales, orders=None, missing="drop",
                    n_starts=4, seed=0, max_iter=500, tol=1e-7, max_bytes=BYTES, max_work=WORK, device="cpu"):
    """Frequency-count multiset nonlinear canonical/homogeneity block ALS.

    Multiple-nominal maps have a category vector per dimension; other maps are
    scalar, with explicitly increasing ordinal maps. Accepted local solutions
    minimize equal-set weighted loss and provide descriptive geometry only.
    """
    _device(device)
    variables, orders = _overals_controls(sets, scales, orders, n_components, n_starts, seed, max_iter, tol)
    sample = f._sample(data, variables, frequency, missing, max_bytes, max_work, operation="frequency OVERALS")
    levels, codes = _levels(sample["frame"], variables, scales, orders)
    state = _state(sample, data, variables, frequency, levels, codes, n_components, "overals_fweight")
    state.update(sets=[list(group) for group in sets], scales=dict(scales),
                 orders={name: levels[variables.index(name)] for name in orders},
                 settings=dict(n_starts=n_starts, seed=seed, max_iter=max_iter, tol=tol))
    ranks = [sum(len(levels[variables.index(name)])-1 if scales[name] == "multiple_nominal" else 1 for name in group) for group in sets]
    allowed = min(ranks) if len(sets) == 2 and "multiple_nominal" not in scales.values() else sum(ranks)
    if n_components >= len(state["positions"]) or n_components > allowed:
        f._error("Requested compromise dimensions exceed the declared set/scaling rank.")
    plan = _plan(state, max_bytes, max_work)
    solutions, traces, starts = [], [], []
    for start in range(n_starts):
        try:
            solution = _start(state, start)
            traces.extend([*row, solution["converged"]] for row in solution["trace"])
            starts.append([start, solution["loss"], len(solution["trace"]), solution["converged"],
                           "accepted" if solution["converged"] else "nonconvergence"])
            if solution["converged"]:
                solutions.append((solution["loss"], start, solution))
        except AnalysisError as exc:
            starts.append([start, "unavailable", 0, False, exc.code])
    if not solutions:
        f._error("No weighted OVERALS start converged to nondegenerate geometry.", "nonconvergence")
    objective, selected, best = min(solutions, key=lambda item: item[:2])
    w, x, fits = f._weights(state["frequencies"]), best["scores"], best["fits"]
    compromise = sum(fits, torch.zeros_like(x))/len(sets)
    explained = x.T@(w[:, None]*compromise)/w.sum()
    eigen, rotation = torch.linalg.eigh((explained+explained.T)/2)
    rotation = rotation[:, torch.argsort(eigen, descending=True)]
    x, rotation = _orient(x@rotation, rotation)
    fits = [fit@rotation for fit in fits]
    qs, loads = [], []
    for name, q, a in zip(variables, best["quantifications"], best["loadings"]):
        a = a@rotation
        if scales[name] == "multiple_nominal":
            q, a = q@rotation, torch.eye(n_components, dtype=DT)
        elif scales[name] == "nominal" and float(q[q.abs().argmax(), 0]) < 0:
            q, a = -q, -a
        qs.append(q.tolist())
        loads.append(a.tolist())
    state.update(scores=x.tolist(), set_scores=[fit.tolist() for fit in fits], quantifications=qs, loadings=loads,
                 objective=objective, selected_start=selected, trace=traces, starts=starts)
    return _result(state, sample, plan, missing)


def _check_overals(state, result, w, x):
    d, variables, settings = state["n_components"], state["variables"], state["settings"]
    _close((x*w[:, None]).sum(0)/w.sum(), torch.zeros(d, dtype=DT), "weighted score centering")
    _close(x.T@(w[:, None]*x)/w.sum(), torch.eye(d, dtype=DT), "weighted score covariance")
    contributions = []
    for name, levels, codes, raw_q, raw_a in zip(variables, state["levels"], state["codes"], state["quantifications"], state["loadings"]):
        q, a, c = torch.tensor(raw_q, dtype=DT), torch.tensor(raw_a, dtype=DT), torch.tensor(codes, dtype=torch.int64)
        scale = state["scales"][name]
        if scale == "multiple_nominal":
            _close(a, torch.eye(d, dtype=DT), "multiple-nominal identity loading")
            _close((q[c]*w[:, None]).sum(0)/w.sum(), torch.zeros(d, dtype=DT), "category-vector centering")
        else:
            _close((q[c]*w[:, None]).sum(0)/w.sum(), torch.zeros(1, dtype=DT), "scalar centering")
            _close((q[c].square()*w[:, None]).sum(0)/w.sum(), torch.ones(1, dtype=DT), "scalar population norm")
            if scale == "ordinal" and bool((q[1:, 0] < q[:-1, 0]-1e-10).any()):
                f._error("Saved ordinal map is decreasing.", "invalid_state")
            if scale == "numeric":
                raw = torch.tensor(levels, dtype=DT)[c]
                standardized, _, _ = f._standardize(raw, w)
                _close(q[c, 0], standardized, "numeric scaling")
        contributions.append(q[c]@a)
    groups = [[variables.index(name) for name in group] for group in state["sets"]]
    fits = [sum((contributions[i] for i in group), torch.zeros_like(x)) for group in groups]
    for fitted, stored in zip(fits, state["set_scores"]):
        _close(torch.tensor(stored, dtype=DT), fitted, "set reconstruction")
    objective = float(sum(((x-fit).square()*w[:, None]).sum() for fit in fits)/(w.sum()*len(groups)))
    if abs(objective-state["objective"]) > 2e-8 or abs(result.attrs.get("objective", -1)-objective) > 2e-8 or result.attrs.get("selected_start") != state["selected_start"] or result.attrs.get("settings") != settings:
        f._error("Saved OVERALS objective or selected-start metadata disagree.", "invalid_state")
    stationarity = max(2e-5, 20*math.sqrt(settings["tol"]))
    _close(x, _unit(sum(fits, torch.zeros_like(x))/len(groups), w), "compromise polar stationarity", tolerance=stationarity)
    for group, fitted in zip(groups, fits):
        for i in group:
            name = variables[i]
            c = torch.tensor(state["codes"][i], dtype=torch.int64)
            q = torch.tensor(state["quantifications"][i], dtype=DT)
            a = torch.tensor(state["loadings"][i], dtype=DT)
            residual = x-fitted+contributions[i]
            centroids, counts = f._means(residual, c, len(state["levels"][i]), w)
            scale = state["scales"][name]
            if scale == "multiple_nominal":
                _close(q, centroids, "multiple-nominal centroid stationarity", tolerance=stationarity)
            else:
                expected_a = q[c].T@(w[:, None]*residual)/w.sum()
                _close(a, expected_a, "weighted variable normal equations", tolerance=stationarity)
                if scale == "nominal":
                    singular = torch.linalg.svdvals(centroids*(counts/w.sum()).sqrt()[:, None])
                    current = float(((centroids-q@a).square()*counts[:, None]).sum()/w.sum())
                    optimum = float(singular[1:].square().sum())
                    if current > optimum+stationarity:
                        f._error("Saved nominal block is not a rank-one category optimum.", "invalid_state")
                elif scale == "ordinal" and float(a.square().sum()) > 1e-18:
                    means = (centroids@a.T/a.square().sum()).flatten()
                    updated = _scalar(f._pava(means, counts), c, w)
                    if updated is not None:
                        # Map distance is assessed in the actual frequency
                        # norm, including rare categories without amplification.
                        difference = float(((q[:, 0]-updated).square()*counts).sum()/w.sum())
                        if difference > stationarity:
                            f._error("Saved ordinal block fails weighted PAVA stationarity.", "invalid_state")
    accepted, traces = [], state["trace"]
    if not isinstance(state["starts"], list) or len(state["starts"]) != settings["n_starts"] or not isinstance(traces, list):
        f._error("Saved OVERALS traces have invalid shape.", "invalid_state")
    grouped = {i: [] for i in range(settings["n_starts"])}
    for row in traces:
        if (not isinstance(row, list) or len(row) != 5 or type(row[0]) is not int or row[0] not in grouped
                or type(row[1]) is not int or not 1 <= row[1] <= settings["max_iter"]
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in row[2:4]) or type(row[4]) is not bool):
            f._error("Saved OVERALS iteration record is invalid.", "invalid_state")
        grouped[row[0]].append(row)
    for expected, record in enumerate(state["starts"]):
        if (not isinstance(record, list) or len(record) != 5 or type(record[0]) is not int or record[0] != expected
                or type(record[2]) is not int or not 0 <= record[2] <= settings["max_iter"] or type(record[3]) is not bool
                or type(record[4]) is not str):
            f._error("Saved OVERALS start record is invalid.", "invalid_state")
        trace = grouped[expected]
        if [row[1] for row in trace] != list(range(1, len(trace)+1)) or len(trace) != record[2]:
            f._error("Saved OVERALS trace is not contiguous.", "invalid_state")
        previous = float(d)
        for row in trace:
            if abs(row[3]-(previous-row[2])) > 1e-10 or row[3] < -1e-8 or row[4] != record[3]:
                f._error("Saved OVERALS loss decreases or convergence flags disagree.", "invalid_state")
            previous = row[2]
        if trace and (type(record[1]) not in (int, float) or abs(record[1]-trace[-1][2]) > 1e-10):
            f._error("Saved OVERALS final trace loss disagrees.", "invalid_state")
        if record[3]:
            if record[4] != "accepted" or not trace or trace[-1][3] > settings["tol"]*max(1., float(d) if len(trace) == 1 else trace[-2][2])+1e-12:
                f._error("Saved OVERALS accepted start did not converge.", "invalid_state")
            accepted.append((record[1], record[0]))
        elif record[4] == "accepted" or not trace and record[1] != "unavailable":
            f._error("Saved OVERALS failed-start record is inconsistent.", "invalid_state")
    if not accepted or min(accepted) != (state["objective"], state["selected_start"]):
        f._error("Saved OVERALS result is not the best declared converged start.", "invalid_state")


__all__ = ["mca_fweight", "mca_fweight_project", "overals_fweight"]
