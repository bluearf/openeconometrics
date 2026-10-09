"""Full-estimator fixed-query bootstrap in compressed integer-case geometry.

The sampling law is ordinary iid pairs resampling of the literal repeated
cases, represented by multinomial counts. Runtime never expands those cases.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.summary_state import restore_summary, summary_state
from . import frequency as f
from . import frequency_regression as r
from . import optimal as base

MAX_REPS = 199
MAX_QUERIES = 64
ARTIFACT_BYTES = 32 * 1024**2
TITLE = "Frequency CATREG full-refit fixed-query bootstrap"
METHODS = {"catreg_nominal_fweight_bootstrap", "catreg_ordinal_fweight_bootstrap"}
SOURCES = [
    "https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=bootstrap-sampling-subcommand-command",
    "https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=bootstrapping-",
    "https://docs.pytorch.org/docs/stable/generated/torch.binomial.html",
]
INFERENCE = ("iid literal-case multinomial full-estimator empirical joint prediction covariance and "
    "linear-interpolated percentile intervals only when every declared draw succeeds; "
    "finite Monte Carlo/coverage limits, no ordinary adaptive coefficient Wald, p-values or survey inference")
ALIGNMENT = "fixed complete raw queries in original numeric-response units; refitted map sign/scale cancels in prediction"
STATE_FIELDS = {"version", "kind", "method", "settings", "baseline", "query_raw", "point",
    "draw_counts", "draw_fits", "draw_predictions", "receipts", "failures"}
SETTINGS_FIELDS = {"reps", "confidence", "seed", "n_starts", "maxiter", "tol", "missing",
    "failure", "max_work", "max_bytes"}
ATTR_FIELDS = {"method", "bootstrap_state", "state_sha256", "resources", "declared_work",
    "declared_artifact_bound_bytes", "inference_available", "successful_draws", "dtype", "device",
    "weight_type", "sources", "inference", "alignment", "sampling_law"}
BASE_TABLES = {"baseline_predictions", "draw_predictions", "draw_status", "frequency_draws", "settings"}
INFERENCE_TABLES = {"prediction_covariance", "percentile_intervals"}


def _error(message, code="invalid_spec"):
    raise AnalysisError(code, message)


def _settings(reps, confidence, seed, n_starts, maxiter, tol, missing, failure, max_work, max_bytes, device):
    try:
        controls = r._controls(n_starts, seed, maxiter, tol, max_work, max_bytes, device)
        confidence = base._real(confidence, "confidence", .5, .999)
    except (OverflowError, TypeError, ValueError) as exc:
        raise AnalysisError("invalid_spec", "Bootstrap scalar controls must be bounded finite reals/integers.") from exc
    if missing not in ("raise", "drop") or failure not in ("raise", "record"):
        _error("missing and failure must be raise/drop and raise/record respectively.")
    return dict(controls, reps=base._integer(reps, "reps", 2, MAX_REPS),
        confidence=confidence, missing=missing, failure=failure)


def _fit_controls(settings, draw=None):
    controls = {key: settings[key] for key in ("n_starts", "seed", "maxiter", "tol", "max_work", "max_bytes")}
    if draw is not None:
        controls["seed"] = (settings["seed"]+104729*(draw+1)) % (2**31-1)
    return controls


def _draw_counts(counts, generator):
    """Conditional binomials are exactly Multinomial(F, counts/F).

    Both remainders are integers: sampled trials and ORIGINAL probability mass.
    No total_count-sized tensor or accumulated floating probability subtraction.
    """
    remaining, mass = sum(counts), sum(counts)
    sampled = []
    for count in counts[:-1]:
        if remaining:
            value = float(torch.binomial(
                torch.tensor(float(remaining), dtype=f.DT, device="cpu"),
                torch.tensor(count/mass, dtype=f.DT, device="cpu"), generator=generator))
            if not math.isfinite(value) or not 0 <= value <= remaining or value != math.floor(value):
                _error("Native binomial returned an invalid integer count.", "numerical_failure")
            value = int(value)
        else:
            value = 0
        sampled.append(value)
        remaining -= value
        mass -= count
    return sampled+[remaining]


def _prediction(state, query_raw):
    z = r._map(query_raw, state["descriptors"])
    values = state["intercept"]+z@torch.tensor(state["beta"], dtype=f.DT, device="cpu")
    if not torch.isfinite(values).all():
        _error("Fixed-query predictions are nonfinite.", "numerical_failure")
    return values.tolist()


def _schema_failure(raw, counts, descriptors):
    if sum(value > 0 for value in counts) < len(descriptors)+3:
        return "insufficient_sample"
    for j, descriptor in enumerate(descriptors, 1):
        if descriptor["scale"] != "numeric":
            present = {base._key(row[j]) for row, count in zip(raw, counts) if count}
            if present != {base._key(value) for value in descriptor["levels"]}:
                return "bootstrap_absent_category"
    return None


def _geometry(baseline, query_raw, settings):
    """Conservative cumulative fit work and retained-state/one-fit peak."""
    n, p, nq, reps = len(baseline["counts"]), len(baseline["variables"]), len(query_raw), settings["reps"]
    controls = _fit_controls(settings)
    bound = r._bound(baseline["raw"], n, p, controls)
    sizes = [len(d.get("levels", [])) for d in [baseline["response"]]+baseline["descriptors"]]
    one_fit_work = r._work(n, p, sizes, controls)+8*max(n, baseline["input_nobs"])*(p+2)+4*bound
    query_size = sum(len(json.dumps(row, ensure_ascii=True))+12 for row in query_raw)
    artifact = 65536+(reps+1)*bound+128*reps*n+8*query_size+512*(reps*nq+nq*nq)
    if artifact > ARTIFACT_BYTES:
        _error("Complete draw counts, fits and tables exceed the conservative 32 MiB portable domain.", "resource_limit")
    work = (reps+1)*one_fit_work+4*artifact+reps*32*n+64*(reps+1)*nq*p*p
    size = 3*artifact+4*bound+256*baseline["input_nobs"]*(p+3)+8*n*(64*p+128)+512*nq*(p+4)+1024*nq*nq
    return size, work, artifact


def _output(state, resources, declared_work, artifact):
    nq, reps = len(state["point"]), state["settings"]["reps"]
    labels = [f"query_{i}" for i in range(nq)]
    rows = [[draw]+values for draw, values in enumerate(state["draw_predictions"]) if values is not None]
    tables = {
        "baseline_predictions": table([[i, value] for i, value in enumerate(state["point"])], columns=["query_position", "fitted"]),
        "draw_predictions": table(rows, columns=["draw"]+labels),
        "draw_status": table(state["receipts"], columns=["draw", "success", "chosen_start", "iterations", "objective", "status"]),
        "frequency_draws": table([[draw, position, count] for draw, counts in enumerate(state["draw_counts"])
            for position, count in enumerate(counts)], columns=["draw", "active_physical_position", "frequency"]),
    }
    if not state["failures"]:
        values = torch.tensor(state["draw_predictions"], dtype=f.DT, device="cpu")
        centered = values-values.mean(0)
        covariance = centered.T@centered/(reps-1)
        alpha = (1-state["settings"]["confidence"])/2
        intervals = torch.quantile(values, torch.tensor([alpha, 1-alpha], dtype=f.DT, device="cpu"), dim=0)
        tables["prediction_covariance"] = table(covariance.tolist(), columns=labels)
        tables["percentile_intervals"] = table([[i, state["point"][i], float(values[:, i].mean()),
            float(covariance[i, i].sqrt()), float(intervals[0, i]), float(intervals[1, i])]
            for i in range(nq)], columns=["query_position", "point", "bootstrap_mean", "bootstrap_sd", "percentile_lower", "percentile_upper"])
    return f._save(TableSet(tables, title=TITLE, method=state["method"], bootstrap_state=state,
        state_sha256=f._seal(state), resources=resources, declared_work=declared_work,
        declared_artifact_bound_bytes=artifact, inference_available=not state["failures"],
        successful_draws=len(rows), dtype="float64", device="cpu", weight_type="frequency",
        sources=SOURCES, inference=INFERENCE, alignment=ALIGNMENT,
        sampling_law="Multinomial(F, f/F), conditional binomial physical-row counts; no repeated-row allocation"))


def _bootstrap(data, outcome, predictors, frequency, queries, scales, orders, default, settings):
    variables = r._names(predictors, outcome, frequency)
    all_scales, all_orders = r._scales(variables, outcome, "numeric", default, scales, orders, None)
    if (not isinstance(data, pd.DataFrame) or not isinstance(queries, pd.DataFrame)
            or not 1 <= len(queries) <= MAX_QUERIES or not len(variables)+3 <= len(data) <= f.MAX_ROWS
            or any(list(queries.columns).count(name) != 1 for name in variables)):
        _error("Data must have at most 3000 physical rows; fixed queries require 1–64 DataFrame rows and unique predictors.")
    p, n_input, nq = len(variables), len(data), len(queries)
    f._admit("frequency bootstrap named input selection", 256*(n_input*(p+3)+nq*(p+1)),
        16*(n_input*(p+2)+nq*p), settings["max_bytes"], settings["max_work"])
    sample = f._sample(data, [outcome]+variables, frequency, settings["missing"], settings["max_bytes"],
        settings["max_work"], operation="frequency bootstrap", min_rows=p+3)
    frame = sample["frame"]
    raw = r._raw(frame, [outcome]+variables, all_scales)
    selected_query = queries.loc[:, variables]
    if bool(selected_query.isna().any(axis=None)):
        _error("Fixed queries must be complete; no query deletion is performed.", "missing_data")
    query_raw = r._raw(selected_query, variables, all_scales)
    # Before the first tensor/fit: actual escaped labels and worst full fit
    # trace dimensions are admitted jointly over all declared replicates.
    scaffold = dict(counts=sample["counts"], variables=variables, raw=raw, input_nobs=n_input,
        response={"scale": "numeric"}, descriptors=[{"scale": all_scales[name],
        "levels": list({base._key(row[j+1]) for row in raw}) if all_scales[name] != "numeric" else []}
        for j, name in enumerate(variables)])
    if any(len(d["levels"]) > 32 for d in scaffold["descriptors"]):
        _error("At most 32 retained levels per categorical predictor.", "resource_limit")
    size, work, artifact = _geometry(scaffold, query_raw, settings)
    resources = f._admit("frequency CATREG bootstrap fits and complete state", size, work,
        settings["max_bytes"], settings["max_work"])
    expected = [{base._key(row[j+1]) for row in raw} if all_scales[name] != "numeric" else None
        for j, name in enumerate(variables)]
    for j, keys in enumerate(expected):
        if keys is not None and any(base._key(row[j]) not in keys for row in query_raw):
            _error("A fixed-query category is absent from the baseline sample.", "unknown_category")
    baseline = r._fit(data, outcome, variables, frequency, "numeric", default, scales, orders, None,
        _fit_controls(settings), settings["missing"])
    baseline_state = baseline.attrs["frequency_state"]
    point = _prediction(baseline_state, query_raw)
    state = dict(version=1, kind="catreg_fweight_bootstrap", method=f"catreg_{default}_fweight_bootstrap",
        settings=settings, baseline=summary_state(baseline), query_raw=query_raw, point=point,
        draw_counts=[], draw_fits=[], draw_predictions=[], receipts=[], failures=[])
    rng = torch.Generator(device="cpu").manual_seed(settings["seed"])
    draw_frame = frame.copy()
    for draw in range(settings["reps"]):
        counts = _draw_counts(sample["counts"], rng)
        state["draw_counts"].append(counts)
        try:
            absent = _schema_failure(raw, counts, baseline_state["descriptors"])
            if absent:
                _error("A declared draw lacks the original category/physical-row estimator domain.", absent)
            draw_frame[frequency] = counts
            fitted = r._fit(draw_frame, outcome, variables, frequency, "numeric", default, scales, orders, None,
                _fit_controls(settings, draw), "raise")
            draw_state = fitted.attrs["frequency_state"]
            predictions = _prediction(draw_state, query_raw)
            state["draw_fits"].append(summary_state(fitted))
            state["draw_predictions"].append(predictions)
            state["receipts"].append([draw, True, draw_state["chosen_start"], draw_state["iterations"], draw_state["objective"], "accepted"])
        except AnalysisError as exc:
            message = str(exc)[:512]
            state["failures"].append([draw, exc.code, message])
            state["draw_fits"].append(None)
            state["draw_predictions"].append(None)
            state["receipts"].append([draw, False, None, None, None, exc.code])
            if settings["failure"] == "raise":
                error = AnalysisError("bootstrap_failure", f"Draw {draw} failed ({exc.code}); no surviving-draw inference.")
                error.bootstrap_failures = state["failures"]
                error.bootstrap_counts = state["draw_counts"]
                raise error from exc
    result = _output(state, resources, work, artifact)
    summary_state(result)
    return result


@resident_cpu
def catreg_nominal_fweight_bootstrap(data: Any, outcome: str, predictors: list[str], *, frequency: str,
    queries: Any, scales: dict | None = None, reps: int = 19, confidence: float = .95, seed: int = 0,
    n_starts: int = 2, maxiter: int = 100, tol: float = 1e-8, missing: str = "drop", failure: str = "raise",
    max_work: int = f.WORK, max_bytes: int = f.BYTES, device: str = "cpu") -> TableSet:
    """Full-refit iid literal-case frequency CATREG joint fixed-query bootstrap.

    Numeric response; nominal/numeric predictors; exact multinomial count law.
    Every declared draw refits all starts/maps. Any recorded failure withholds
    covariance and percentile intervals. CPU float64; 2–199 draws, 1–64 queries.
    """
    settings = _settings(reps, confidence, seed, n_starts, maxiter, tol, missing, failure, max_work, max_bytes, device)
    return _bootstrap(data, outcome, predictors, frequency, queries, scales, None, "nominal", settings)


@resident_cpu
def catreg_ordinal_fweight_bootstrap(data: Any, outcome: str, predictors: list[str], *, frequency: str,
    queries: Any, orders: dict, scales: dict | None = None, reps: int = 19, confidence: float = .95,
    seed: int = 0, n_starts: int = 2, maxiter: int = 100, tol: float = 1e-8, missing: str = "drop",
    failure: str = "raise", max_work: int = f.WORK, max_bytes: int = f.BYTES, device: str = "cpu") -> TableSet:
    """Full-refit frequency CATREG bootstrap with explicitly ordered predictors.

    Numeric response and nominal/ordinal/numeric predictors; signed monotone
    refits at every draw. Same multinomial/failure/portable domains as nominal.
    """
    settings = _settings(reps, confidence, seed, n_starts, maxiter, tol, missing, failure, max_work, max_bytes, device)
    return _bootstrap(data, outcome, predictors, frequency, queries, scales, orders, "ordinal", settings)


def _state_shape(state):
    """Bound every nested primitive before checksum, copies or tensors."""
    try:
        if (not isinstance(state, dict) or len(state) != len(STATE_FIELDS) or set(state) != STATE_FIELDS or type(state["version"]) is not int
                or state["version"] != 1 or type(state["kind"]) is not str or state["kind"] != "catreg_fweight_bootstrap"
                or type(state["method"]) is not str or len(state["method"]) > 64 or state["method"] not in METHODS):
            raise ValueError("Closed state")
        settings = state["settings"]
        if not isinstance(settings, dict) or len(settings) != len(SETTINGS_FIELDS) or set(settings) != SETTINGS_FIELDS:
            raise ValueError("Settings")
        checked = _settings(**settings, device="cpu")
        if not r._equal_primitive(checked, settings):
            raise ValueError("Settings types")
        if not isinstance(state["baseline"], str) or not 1 <= len(state["baseline"]) <= ARTIFACT_BYTES:
            raise ValueError("Baseline JSON")
        raw, point = state["query_raw"], state["point"]
        if (not isinstance(raw, list) or not 1 <= len(raw) <= MAX_QUERIES or not isinstance(raw[0], list)
                or not 1 <= len(raw[0]) <= 11 or any(not isinstance(row, list) or len(row) != len(raw[0])
                or not all(r._label_valid(v) for v in row) for row in raw)
                or not isinstance(point, list) or len(point) != len(raw) or not all(r._finite(v) for v in point)):
            raise ValueError("Fixed query")
        reps = settings["reps"]
        fields = ("draw_counts", "draw_fits", "draw_predictions", "receipts")
        if any(not isinstance(state[k], list) or len(state[k]) != reps for k in fields):
            raise ValueError("Replicate lengths")
        counts = state["draw_counts"]
        n = len(counts[0]) if isinstance(counts[0], list) else 0
        if not 4 <= n <= f.MAX_ROWS or any(not isinstance(row, list) or len(row) != n
                or any(type(v) is not int or not 0 <= v <= f.MAX_TOTAL for v in row) for row in counts):
            raise ValueError("Counts")
        if any(value is not None and (not isinstance(value, str) or not 1 <= len(value) <= ARTIFACT_BYTES) for value in state["draw_fits"]):
            raise ValueError("Fit JSON")
        if sum(len(v) for v in [state["baseline"]]+state["draw_fits"] if v is not None) > ARTIFACT_BYTES:
            raise ValueError("Combined fit JSON")
        if any(value is not None and (not isinstance(value, list) or len(value) != len(raw)
                or not all(r._finite(v) for v in value)) for value in state["draw_predictions"]):
            raise ValueError("Predictions")
        for i, row in enumerate(state["receipts"]):
            if (not isinstance(row, list) or len(row) != 6 or type(row[0]) is not int or row[0] != i
                    or type(row[1]) is not bool or not isinstance(row[5], str) or not 1 <= len(row[5]) <= 128
                    or (row[1] and (type(row[2]) is not int or not 0 <= row[2] < settings["n_starts"]
                        or type(row[3]) is not int or not 1 <= row[3] <= settings["maxiter"] or not r._finite(row[4])))
                    or (not row[1] and row[2:5] != [None, None, None])):
                raise ValueError("Receipts")
        failures = state["failures"]
        if (not isinstance(failures, list) or len(failures) > reps or any(not isinstance(row, list) or len(row) != 3
                or type(row[0]) is not int or not 0 <= row[0] < reps or not isinstance(row[1], str)
                or not 1 <= len(row[1]) <= 128 or not isinstance(row[2], str) or len(row[2]) > 512 for row in failures)
                or [row[0] for row in failures] != sorted({row[0] for row in failures})
                or settings["failure"] == "raise" and failures):
            raise ValueError("Failures")
    except (TypeError, ValueError, KeyError, OverflowError, AnalysisError) as exc:
        raise AnalysisError("invalid_state", "Malformed bounded frequency-bootstrap primitive state.") from exc
    return state


def _attrs(attrs):
    if not isinstance(attrs, dict) or len(attrs) != len(ATTR_FIELDS) or set(attrs) != ATTR_FIELDS:
        _error("Unknown complete bootstrap metadata.", "invalid_state")
    state = _state_shape(attrs["bootstrap_state"])
    r._resource_shape(attrs["resources"])
    expected = dict(method=state["method"], inference_available=not state["failures"],
        successful_draws=sum(row[1] for row in state["receipts"]), dtype="float64", device="cpu",
        weight_type="frequency", sources=SOURCES, inference=INFERENCE, alignment=ALIGNMENT,
        sampling_law="Multinomial(F, f/F), conditional binomial physical-row counts; no repeated-row allocation")
    if any(not r._equal_primitive(attrs[key], value) for key, value in expected.items()):
        _error("Saved metadata disagrees with its bootstrap state.", "invalid_state")
    if (not isinstance(attrs["state_sha256"], str) or len(attrs["state_sha256"]) != 64
            or type(attrs["declared_work"]) is not int or not 1 <= attrs["declared_work"] <= f.WORK
            or type(attrs["declared_artifact_bound_bytes"]) is not int or not 1 <= attrs["declared_artifact_bound_bytes"] <= ARTIFACT_BYTES):
        _error("Malformed bootstrap checksum/resource bounds.", "invalid_state")
    return state


def _checked(result, max_bytes, max_work):
    f._admit("frequency bootstrap primitive admission", 4096, 1, max_bytes, max_work)
    decode = 0
    if isinstance(result, str):
        if not 1 <= len(result) <= ARTIFACT_BYTES:
            _error("Supply complete bootstrap summary JSON up to 32 MiB.", "invalid_state")
        decode = 4*len(result)
        f._admit("frequency bootstrap complete JSON decode", 12*len(result)+4096, decode, max_bytes, max_work)
        try:
            payload = json.loads(result)
            if (not isinstance(payload, dict) or set(payload) != {"schema", "title", "attrs", "tables"}
                    or payload["schema"] != "openecon.summary.v1" or payload["title"] != TITLE):
                raise ValueError("Envelope")
            state = _attrs(payload["attrs"])
            expected = BASE_TABLES | (INFERENCE_TABLES if not state["failures"] else set())
            if not isinstance(payload["tables"], dict) or set(payload["tables"]) != expected:
                raise ValueError("Tables")
            for frame in payload["tables"].values():
                if (not isinstance(frame, dict) or set(frame) != {"columns", "index", "data", "index_names", "column_names"}
                        or not isinstance(frame["data"], list) or not isinstance(frame["columns"], list)
                        or len(frame["columns"]) > 65 or any(not isinstance(v, str) or len(v) > 128 for v in frame["columns"])
                        or frame["index"] != list(range(len(frame["data"]))) or frame["index_names"] != [None]
                        or frame["column_names"] != [None] or len(frame["data"]) > f.MAX_ROWS*MAX_REPS):
                    raise ValueError("Table schema")
                r._table_cells(frame["data"], f.MAX_ROWS*MAX_REPS, 65)
                if any(len(row) != len(frame["columns"]) for row in frame["data"]):
                    raise ValueError("Table rows")
        except (TypeError, KeyError, ValueError, RecursionError) as exc:
            raise AnalysisError("invalid_state", "Malformed complete bootstrap JSON tables.") from exc
        result = restore_summary(result)
    if not isinstance(result, TableSet) or type(result.title) is not str or result.title != TITLE:
        _error("Supply a complete frequency-bootstrap TableSet or full summary JSON.", "invalid_state")
    state = _attrs(result.attrs)
    expected = BASE_TABLES | (INFERENCE_TABLES if not state["failures"] else set())
    if set(result) != expected:
        _error("Bootstrap tables disagree with the failure policy.", "invalid_state")
    n, nq, reps = len(state["draw_counts"][0]), len(state["query_raw"]), state["settings"]["reps"]
    row_limits = dict(baseline_predictions=nq, draw_predictions=reps, draw_status=reps,
        frequency_draws=reps*n, settings=len(ATTR_FIELDS), prediction_covariance=nq, percentile_intervals=nq)
    if any(not isinstance(frame, pd.DataFrame) or len(frame) > row_limits[name] or len(frame.columns) > 65
            for name, frame in result.items()):
        _error("Saved tables exceed their declared sample/replicate dimensions.", "invalid_state")
    cell_count = sum(len(frame)*len(frame.columns) for frame in result.values())
    retained_bytes = sum(len(v) for v in [state["baseline"]]+state["draw_fits"] if v is not None)
    f._admit("frequency bootstrap saved table indexes and primitive scan", 4*retained_bytes+64*cell_count+4096,
        decode+8*cell_count+4*retained_bytes, max_bytes, max_work)
    for frame in result.values():
        if (not isinstance(frame, pd.DataFrame) or len(frame) > f.MAX_ROWS*MAX_REPS or len(frame.columns) > 65
                or any(not isinstance(v, str) or len(v) > 128 for v in frame.columns)
                or list(frame.index) != list(range(len(frame))) or list(frame.index.names) != [None]
                or list(frame.columns.names) != [None]):
            _error("Malformed bounded bootstrap tables.", "invalid_state")
        r._table_cells(frame.itertuples(index=False, name=None), f.MAX_ROWS*MAX_REPS, 65)
    fits = [state["baseline"]]+[v for v in state["draw_fits"] if v is not None]
    retained = sum(len(value) for value in fits)
    nq, n, reps = len(state["query_raw"]), len(state["draw_counts"][0]), state["settings"]["reps"]
    # Numerical saved-model validations are sequential. Precharge their combined
    # JSON/model reconstruction work plus one live validation peak and all fits.
    # Parse the baseline primitive shape under an early JSON-size admission;
    # no pandas/tensor/model reconstruction precedes the complete loop plan.
    f._admit("frequency bootstrap retained calibration decode", 12*retained+4096,
        decode+4*retained, max_bytes, max_work)
    try:
        baseline_payload = json.loads(state["baseline"])
        baseline_shape = r._attrs_shape(baseline_payload["attrs"])
    except (TypeError, KeyError, ValueError, RecursionError) as exc:
        raise AnalysisError("invalid_state", "Malformed baseline calibration JSON.") from exc
    if len(baseline_shape["counts"]) != n or len(baseline_shape["variables"]) != len(state["query_raw"][0]):
        _error("Saved baseline/query/sample dimensions disagree.", "invalid_state")
    size, work, artifact = _geometry(baseline_shape, state["query_raw"], state["settings"])
    p = len(baseline_shape["variables"])
    bound = r._bound(baseline_shape["raw"], n, p, baseline_shape["controls"])
    largest = max([len(d.get("levels", [])) for d in [baseline_shape["response"]]+baseline_shape["descriptors"]])
    per_validation = 4*bound+64*n*p*p+64*largest**3+16*state["settings"]["n_starts"]*(state["settings"]["maxiter"]+1)
    validation_work = decode+4*retained+len(fits)*per_validation+4*artifact+32*reps*n+64*(reps+1)*nq*p*p
    validation_size = max(12*retained+4096, 3*artifact+4*bound+8*n*(24*p+64)+1024*nq*nq)
    f._admit("frequency bootstrap aggregate saved replay", validation_size, validation_work, max_bytes, max_work)
    baseline, _, baseline_work = r._checked(state["baseline"], max_bytes, max_work)
    if (baseline["response_scale"] != "numeric" or len(baseline["variables"]) != len(state["query_raw"][0])
            or len(baseline["counts"]) != n or baseline["controls"] != _fit_controls(state["settings"])
            or baseline["missing"] != state["settings"]["missing"]
            or state["method"] != baseline["method"]+"_bootstrap"):
        _error("Saved baseline disagrees with declared bootstrap fitting controls.", "invalid_state")
    resource = result.attrs["resources"]
    if (result.attrs["declared_work"] != work or result.attrs["declared_artifact_bound_bytes"] != artifact
            or resource["operation"] != "frequency CATREG bootstrap fits and complete state"
            or resource["estimated_workspace_bytes"] != size or resource["buffers"] != {"physical_rows_and_numerical_workspace": size}
            or resource["budget_bytes"] > state["settings"]["max_bytes"]):
        _error("Saved bootstrap admission disagrees with its complete fit dimensions.", "invalid_state")
    for j, descriptor in enumerate(baseline["descriptors"]):
        if descriptor["scale"] == "numeric" and any(row[j][0] != "float" or abs(row[j][1]) > 1e100 for row in state["query_raw"]):
            _error("Saved numeric queries exceed the fitted raw numerical domain.", "invalid_state")
    if not torch.allclose(torch.tensor(_prediction(baseline, state["query_raw"]), dtype=f.DT, device="cpu"),
            torch.tensor(state["point"], dtype=f.DT, device="cpu"), atol=1e-10, rtol=1e-10):
        _error("Saved baseline fixed-query prediction disagrees with its learned map.", "invalid_state")
    rng = torch.Generator(device="cpu").manual_seed(state["settings"]["seed"])
    actual_work = decode+baseline_work
    failures = {row[0]: row for row in state["failures"]}
    for draw, counts in enumerate(state["draw_counts"]):
        if counts != _draw_counts(baseline["counts"], rng):
            _error("Saved multinomial counts disagree with the private-seeded draw law.", "invalid_state")
        fitted_json, values, receipt = state["draw_fits"][draw], state["draw_predictions"][draw], state["receipts"][draw]
        absent = _schema_failure(baseline["raw"], counts, baseline["descriptors"])
        if draw in failures:
            if (fitted_json is not None or values is not None or receipt != [draw, False, None, None, None, failures[draw][1]]
                    or (absent is not None and failures[draw][1] != absent)):
                _error("Saved failed-draw ledger violates its estimator domain or withholding policy.", "invalid_state")
            continue
        if absent is not None or fitted_json is None or values is None:
            _error("Saved successful draw lacks the original estimator domain.", "invalid_state")
        fitted, _, fit_work = r._checked(fitted_json, max_bytes, max_work)
        actual_work += fit_work
        positions = [i for i, count in enumerate(counts) if count]
        if (fitted["method"] != baseline["method"] or fitted["variables"] != baseline["variables"]
                or fitted["outcome"] != baseline["outcome"] or fitted["frequency"] != baseline["frequency"]
                or fitted["response_scale"] != "numeric" or fitted["input_nobs"] != n or fitted["positions"] != positions
                or fitted["counts"] != [counts[i] for i in positions] or fitted["raw"] != [baseline["raw"][i] for i in positions]
                or fitted["controls"] != _fit_controls(state["settings"], draw) or fitted["missing"] != "raise"
                or fitted["zero_positions"] != [i for i, count in enumerate(counts) if not count] or fitted["missing_positions"]):
            _error("Saved draw calibration does not represent its exact sampled physical counts.", "invalid_state")
        for original, transformed in zip(baseline["descriptors"], fitted["descriptors"]):
            if (original["scale"] != transformed["scale"] or (original["scale"] == "ordinal"
                    and original["levels"] != transformed["levels"]) or (original["scale"] == "nominal"
                    and {base._key(v) for v in original["levels"]} != {base._key(v) for v in transformed["levels"]})):
                _error("Saved draw changes the original category/order geometry.", "invalid_state")
        if receipt != [draw, True, fitted["chosen_start"], fitted["iterations"], fitted["objective"], "accepted"]:
            _error("Saved successful draw receipt disagrees with its full refit trace.", "invalid_state")
        if not torch.allclose(torch.tensor(_prediction(fitted, state["query_raw"]), dtype=f.DT, device="cpu"),
                torch.tensor(values, dtype=f.DT, device="cpu"), atol=1e-10, rtol=1e-10):
            _error("Saved draw predictions disagree with the full learned calibration.", "invalid_state")
    # Never give each fit a separate copy of the declared computational budget.
    f._admit("frequency bootstrap aggregate verified replay and outputs", validation_size,
        actual_work+4*artifact+32*reps*n+64*(reps+1)*nq*len(baseline["variables"])**2,
        max_bytes, max_work)
    if result.attrs["state_sha256"] != f._seal(state):
        _error("Saved bootstrap checksum disagrees.", "invalid_state")
    expected = _output(state, resource, work, artifact)
    if summary_state(expected) != summary_state(result):
        _error("Saved complete bootstrap tables disagree with numeric replay.", "invalid_state")
    return result


@resident_cpu
def catreg_fweight_bootstrap_restore(result: TableSet | str, *, max_work: int = f.WORK,
    max_bytes: int = f.BYTES, device: str = "cpu") -> TableSet:
    """Validate complete frequency bootstrap counts, refits and joint predictions.

    Optimizer-free replay of every successful calibration and its original-unit
    fixed-query predictions. Failed optimization ledgers remain declarations;
    any recorded failure withholds covariance and intervals. Full JSON required.
    """
    if device != "cpu":
        _error("Saved frequency bootstrap replay supports CPU only.", "unsupported_option")
    return _checked(result, max_bytes, max_work)
