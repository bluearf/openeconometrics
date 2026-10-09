"""Predeclared physical control-function FIT acceptance and independent oracle.

``generate`` freezes the input files and numerical tolerances before measurement.
``run`` calls the public SDK only. ``check`` imports no SDK, Torch, SciPy or
Statsmodels: it independently fits both stages and reconstructs the full
nonsymmetric stacked score sandwich, then checks saved conditional targets.
The fixed primary scope is eight 100,000-row inputs, each fit with HC0 and CR0.
An optional one-million-row Gaussian CSV has a separate receipt and scope.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

KINDS = ("gaussian", "logit", "probit", "cloglog", "poisson", "gamma",
         "inverse_gaussian", "fractional_logit")
APIS = dict(zip(KINDS, ("cfregress", "cflogit", "cfprobit", "cfcloglog",
                       "cfpoisson", "cfgamma", "cfinvgauss", "cffraclogit"), strict=True))
SEED = 914271
PROTOCOL_SCHEMA = "openecon.control_stream_fit.acceptance.v1"
NORMAL_975 = 1.959963984540054


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")


def fixture(kind, rows, *, seed=SEED):
    """Independent seeded geometry; no production estimator generates outcomes."""
    import numpy as np
    import pandas as pd

    if kind not in KINDS or isinstance(rows, bool) or rows < 32:
        raise ValueError("Use a declared outcome and at least 32 original rows.")
    random = np.random.Generator(np.random.PCG64(seed + KINDS.index(kind)))
    x, z1, z2, u, noise = np.clip(random.normal(size=(5, rows)), -3, 3)
    d = 0.25 + 0.45*x + 0.65*z1 - 0.4*z2 + 0.8*u
    eta = 0.1 + 0.23*x + 0.28*d + 0.18*u
    if kind == "gaussian":
        y = eta + 0.7*noise
    elif kind in {"logit", "fractional_logit"}:
        probability = 1 / (1 + np.exp(-eta))
        if kind == "logit":
            y = random.binomial(1, probability).astype(float)
        else:
            y = random.beta(4*probability, 4*(1-probability))
            y[::631], y[1::631] = 0, 1
    elif kind in {"probit", "cloglog"}:
        probability = (np.array([0.5*math.erfc(-v/math.sqrt(2)) for v in eta])
                       if kind == "probit" else -np.expm1(-np.exp(eta)))
        y = random.binomial(1, probability).astype(float)
    elif kind == "poisson":
        y = random.poisson(np.exp(eta)).astype(float)
    elif kind == "gamma":
        y = np.exp(eta) * random.gamma(shape=2, scale=0.5, size=rows)
    else:
        y = random.wald(np.exp(eta), 2, size=rows)
    groups = min(320, rows//4)
    frame = pd.DataFrame({"y": y, "x": x, "d": d, "z1": z1, "z2": z2,
                          "cluster": [f"g-{i % groups:03d}" for i in range(rows)]})
    # Disjoint missing coordinates exercise common two-stage sample alignment.
    for offset, column in enumerate(("y", "d", "x", "z1", "z2", "cluster")):
        frame.loc[frame.index % 997 == offset, column] = (
            None if column == "cluster" else float("nan"))
    return frame


def sample_geometry(frame, covariance):
    columns = ["y", "x", "d", "z1", "z2"]
    if covariance == "cluster":
        columns.append("cluster")
    positions = frame.index[frame[columns].notna().all(axis=1)].to_numpy(dtype="int64")
    # Canonical position hash is independently fixed, little-endian signed i64.
    return {"nobs_original": len(frame), "nobs": len(positions),
            "dropped_rows": len(frame)-len(positions),
            "positions_sha256": hashlib.sha256(positions.astype("<i8").tobytes()).hexdigest(),
            "cluster_count": int(frame.loc[positions, "cluster"].nunique())
            if covariance == "cluster" else None}


def source_identity(frame, covariance):
    """Independent spelling of the public projected/typed receipt contract."""
    import pandas as pd

    columns = ["y", "x", *(["cluster"] if covariance == "cluster" else []), "d", "z1", "z2"]
    projected = frame.loc[:, columns]
    raw = hashlib.sha256()
    raw.update(json.dumps({"columns": columns, "missing": "drop", "allowed_missing": []},
                          sort_keys=True).encode())
    hashes = pd.util.hash_pandas_object(projected, index=False, categorize=True)
    raw.update(hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    geometry = sample_geometry(frame, covariance)
    sample_hash = hashlib.sha256((raw.hexdigest()+geometry["positions_sha256"]).encode()).hexdigest()
    index = hashlib.sha256()
    index.update(json.dumps({"class": "RangeIndex", "names": [{"type": "scalar", "value": None}],
                             "encoding": "signed-little-i8", "version": "openecon.cf.typed_source.v1"},
                            sort_keys=True, separators=(",", ":")).encode())
    index.update(frame.index.to_numpy(dtype="int64").astype("<i8").tobytes())
    clusters = None
    if covariance == "cluster":
        cluster_hash = hashlib.sha256(b"openecon.cf.typed_source.v1:length-prefixed-typed-json")
        for value in frame.cluster:
            label = {"type": "missing", "value": None} if value is pd.NA else (
                {"type": "scalar", "value": value})
            encoded = json.dumps(label, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
            cluster_hash.update(len(encoded).to_bytes(8, "little"))
            cluster_hash.update(encoded)
        clusters = cluster_hash.hexdigest()
    return {"data_hash": raw.hexdigest(), "sample_hash": sample_hash,
            "sample_positions_hash": geometry["positions_sha256"], "input_columns": columns,
            "source_binding": {"typed_index_hash": index.hexdigest(), "typed_cluster_hash": clusters}}


def generate(directory, *, source_pin, rows=100000, include_million=False):
    """Freeze physical inputs, retained-sample geometry and budgets before a run."""
    directory.mkdir(parents=True, exist_ok=False)
    plans = []
    for number, kind in enumerate(KINDS):
        frame = fixture(kind, rows)
        suffix = "csv" if number % 2 == 0 else "parquet"
        path = directory / f"{kind}-{rows}.{suffix}"
        if suffix == "csv":
            frame.to_csv(path, index=False, float_format="%.17g")
        else:
            frame.to_parquet(path, index=False, row_group_size=8192)
        # Geometry is fixed on the actual serialized input, not the generator frame.
        import pandas as pd
        actual = pd.read_csv(path) if suffix == "csv" else pd.read_parquet(path)
        plans.append({"id": f"{kind}-{rows}-{suffix}", "kind": kind,
                      "estimator": APIS[kind], "rows": rows, "format": suffix,
                      "file": str(path.resolve()), "sha256": digest(path),
                      "bytes": path.stat().st_size,
                      "samples": {cov: sample_geometry(actual, cov)
                                  for cov in ("robust", "cluster")}})
    if include_million:
        frame = fixture("gaussian", 1000000)
        path = directory / "gaussian-1000000.csv"
        frame.to_csv(path, index=False, float_format="%.17g")
        import pandas as pd
        actual = pd.read_csv(path)
        plans.append({"id": "gaussian-1000000-csv", "kind": "gaussian",
                      "estimator": APIS["gaussian"], "rows": 1000000, "format": "csv",
                      "file": str(path.resolve()), "sha256": digest(path),
                      "bytes": path.stat().st_size, "optional_scope": True,
                      "samples": {cov: sample_geometry(actual, cov)
                                  for cov in ("robust", "cluster")}})
    manifest = {"schema": PROTOCOL_SCHEMA, "protocol_source_pin": source_pin,
                "seed": SEED, "random_generator": "NumPy PCG64, independent fixture only",
                "plan": plans, "batch_rows": 8192, "max_iterations": 20,
                "tolerance": 1e-9, "max_work": 10000000000,
                "covariances": ["robust", "cluster"],
                "tolerances": {"atol": 2e-7, "rtol": 2e-6},
                "stationarity_tolerance": 2e-9,
                "max_saved_model_bytes": 262144,
                "fit_coordinates": {"y": "y", "endogenous": "d", "x": ["x"],
                                    "instruments": ["z1", "z2"], "intercept": True,
                                    "missing": "drop", "alpha": 0.05},
                "scope": "Streamed two-stage fits with full HC0/CR0 joint uncertainty; saved conditional queries",
                "source_cache": "OS cache uncontrolled; no cold-cache claim",
                "gpu": False, "vendor_parity": False,
                "oracle": "Independent NumPy/math Newton and stacked score derivative; no SDK/Torch/SciPy/Statsmodels in check"}
    write(directory / "manifest.json", manifest)
    return manifest


def quantities(kind, y, eta):
    """Independent outcome criterion, score, observed score derivative and mean."""
    import numpy as np

    if kind == "gaussian":
        residual = y-eta
        return -0.5*residual**2, residual, -np.ones(len(y)), eta
    if kind in {"logit", "fractional_logit"}:
        mean = 1 / (1+np.exp(-eta))
        return y*eta-np.logaddexp(0, eta), y-mean, -mean*(1-mean), mean
    if kind == "probit":
        mean = np.array([0.5*math.erfc(-v/math.sqrt(2)) for v in eta])
        complement = np.array([0.5*math.erfc(v/math.sqrt(2)) for v in eta])
        first = np.exp(-eta**2/2)/math.sqrt(2*math.pi)
        score = first*(y/mean-(1-y)/complement)
        derivative = -eta*score-first**2*(y/mean**2+(1-y)/complement**2)
        criterion = y*np.log(mean)+(1-y)*np.log(complement)
        return criterion, score, derivative, mean
    if kind == "cloglog":
        exponential = np.exp(eta)
        complement = np.exp(-exponential)
        mean = -np.expm1(-exponential)
        first = exponential*complement
        score = y*first/mean-(1-y)*exponential
        derivative = (1-exponential)*score-first**2*(y/mean**2+(1-y)/complement**2)
        return y*np.log(mean)-(1-y)*exponential, score, derivative, mean
    mean = np.exp(eta)
    if kind == "poisson":
        factorial = np.array([math.lgamma(v+1) for v in y])
        return y*eta-mean-factorial, y-mean, -mean, mean
    if kind == "gamma":
        ratio = y/mean
        return -ratio-eta, ratio-1, -ratio, mean
    if kind == "inverse_gaussian":
        term = y/mean**2
        return -0.5*term+1/mean, term-1/mean, -2*term+1/mean, mean
    raise ValueError(kind)


def fit_oracle(kind, frame, covariance):
    """Independent resident offline reference; not used by the runtime fitter."""
    import numpy as np

    required = ["y", "x", "d", "z1", "z2"]
    if covariance == "cluster":
        required.append("cluster")
    selected = frame.loc[frame[required].notna().all(axis=1)]
    n = len(selected)
    z = np.column_stack((np.ones(n), selected[["x", "z1", "z2"]].to_numpy(dtype=float)))
    x = np.column_stack((np.ones(n), selected[["x", "d"]].to_numpy(dtype=float)))
    d, y = selected.d.to_numpy(dtype=float), selected.y.to_numpy(dtype=float)
    gamma = np.linalg.lstsq(z, d, rcond=None)[0]
    residual = d-z@gamma
    q = np.column_stack((x, residual))
    if kind == "gaussian":
        beta = np.linalg.lstsq(q, y, rcond=None)[0]
    else:
        beta = np.zeros(q.shape[1])
        mean = float(y.mean())
        if kind in {"logit", "fractional_logit"}:
            beta[0] = math.log(mean/(1-mean))
        elif kind == "cloglog":
            beta[0] = math.log(-math.log1p(-mean))
        elif kind in {"poisson", "gamma", "inverse_gaussian"}:
            beta[0] = math.log(mean)
        for iteration in range(100):
            value, score, derivative, _ = quantities(kind, y, q@beta)
            gradient = q.T@score
            if np.max(np.abs(gradient))/n < 5e-14:
                break
            information = -(q*derivative[:, None]).T@q
            # Positive-mean observed information may be indefinite away from
            # the optimum. Independently damp that Newton system, without
            # changing the criterion or the final covariance derivative.
            lowest = float(np.linalg.eigvalsh(information/n)[0])
            if lowest <= 1e-8:
                information = information + n*(1e-8-lowest)*np.eye(q.shape[1])
            direction = np.linalg.solve(information, gradient)
            direction /= max(1.0, float(np.max(np.abs(direction)))/2)
            initial = float(value.sum())
            for halving in range(50):
                candidate = beta+direction*(0.5**halving)
                trial = quantities(kind, y, q@candidate)[0]
                if np.isfinite(trial).all() and float(trial.sum()) >= initial-1e-9:
                    beta = candidate
                    break
            else:
                raise AssertionError("Independent Newton line search failed.")
        else:
            raise AssertionError("Independent outcome Newton did not converge.")
    criterion, score, derivative, mean = quantities(kind, y, q@beta)
    kz, width = z.shape[1], z.shape[1]+q.shape[1]
    scores = np.column_stack((z*residual[:, None], q*score[:, None]))
    bread = np.zeros((width, width))
    bread[:kz, :kz] = z.T@z
    cross = beta[-1]*(q*derivative[:, None]).T@z
    cross[-1] += score@z
    bread[kz:, :kz] = cross
    bread[kz:, kz:] = -(q*derivative[:, None]).T@q
    if covariance == "cluster":
        _, codes = np.unique(selected.cluster.to_numpy(), return_inverse=True)
        units = np.zeros((int(codes.max())+1, width))
        np.add.at(units, codes, scores)
    else:
        units = scores
    meat = units.T@units
    inverse = np.linalg.inv(bread)
    joint = inverse@meat@inverse.T
    joint = (joint+joint.T)/2
    p = np.r_[gamma, beta]
    se = np.sqrt(joint.diagonal())
    statistic = p/se
    probabilities = np.array([math.erfc(abs(v)/math.sqrt(2)) for v in statistic])
    return {"gamma": gamma, "beta": beta, "bread": bread, "meat": meat,
            "joint_covariance": joint, "criterion": float(criterion.sum()),
            "score_sum": scores.sum(0), "score_abs_sum": np.abs(scores).sum(0),
            "coefficient_columns": np.column_stack((p, se, statistic, probabilities,
                                                     p-NORMAL_975*se, p+NORMAL_975*se)),
            "sample": sample_geometry(frame, covariance), "fitted": mean,
            "z_terms": ["Intercept", "x", "z1", "z2"],
            "x_terms": ["Intercept", "x", "d"]}


def frame_record(frame):
    import pandas as pd
    # Missing evaluation coordinates remain explicit JSON nulls, never NaN JSON.
    serial = frame.astype(object).where(pd.notna(frame), None)
    return {"rows": serial.to_dict("records"), "attrs": frame.attrs}


def run(manifest_path, output, *, source_pin):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import openecon as oe

    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == PROTOCOL_SCHEMA
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": PROTOCOL_SCHEMA, "status": "running", "cases": [],
               "manifest_sha256": digest(manifest_path), "source_pin": source_pin,
               "sdk_version": oe.__version__, "frozen": bool(getattr(sys, "frozen", False)),
               "torch_num_threads": torch.get_num_threads()}
    try:
        for plan in manifest["plan"]:
            assert digest(plan["file"]) == plan["sha256"]
            for covariance in manifest["covariances"]:
                case_id = plan["id"]+"-"+covariance
                item = {"id": case_id, "kind": plan["kind"], "covariance": covariance,
                        "rows": plan["rows"], "status": "running", "targets": {}}
                receipt["cases"].append(item)
                write(output / "run.json", receipt)
                source = oe.scan(plan["file"])
                options = {**manifest["fit_coordinates"], "covariance": covariance,
                           "cluster": "cluster" if covariance == "cluster" else None,
                           "max_iterations": manifest["max_iterations"],
                           "tolerance": manifest["tolerance"], "max_work": manifest["max_work"],
                           "batch_rows": manifest["batch_rows"]}
                started = time.monotonic()
                fitted = getattr(oe, plan["estimator"])(data=source, **options)
                item["fit_seconds"] = time.monotonic()-started
                model_path = output / (case_id+"-model.json")
                model_path.write_text(fitted.model_dump_json())
                assert model_path.stat().st_size <= manifest["max_saved_model_bytes"]
                # A fresh JSON parse must validate/rebind compact persisted state.
                started = time.monotonic()
                restored = oe.cf_restore(result=model_path.read_text(), data=source,
                                         batch_rows=manifest["batch_rows"],
                                         max_work=manifest["max_work"])
                item["restore_seconds"] = time.monotonic()-started
                query = source.head(41)
                item["small_predictions"] = {
                    target: frame_record(oe.predict(restored, query, kind=target,
                        term="d" if target == "derivative" else None,
                        interval=None if target == "stdp" else "mean"))
                    for target in ("response", "xb", "stdp", "derivative")}
                item["small_margins"] = {
                    method: frame_record(oe.margins(restored, ["x", "d", "z1"], data=query,
                        method=method, at={"z2": [-0.4, 0.5]}))
                    for method in ("ame", "mem")}
                started = time.monotonic()
                for target in ("response", "derivative"):
                    predictions = oe.predict(restored, source, kind=target,
                        term="d" if target == "derivative" else None,
                        interval="mean", batch_rows=manifest["batch_rows"])
                    path = output / (case_id+"-"+target+".parquet")
                    writer, count = None, 0
                    try:
                        for frame in predictions.iter_batches(batch_rows=manifest["batch_rows"]):
                            block = pa.Table.from_pandas(pd.DataFrame(frame), preserve_index=True)
                            if writer is None:
                                writer = pq.ParquetWriter(path, block.schema)
                            writer.write_table(block)
                            count += len(frame)
                    finally:
                        if writer is not None:
                            writer.close()
                    assert count == plan["rows"]
                    item["targets"][target] = {"file": str(path.resolve()),
                        "sha256": digest(path), "rows": count, "metadata": predictions.metadata}
                    scratch = Path(predictions._owned_prediction_output.name)
                    del predictions
                    gc.collect()
                    assert not scratch.exists()
                item["prediction_seconds"] = time.monotonic()-started
                started = time.monotonic()
                item["margins"] = {
                    method: frame_record(oe.margins(restored, ["x", "d", "z1"], data=source,
                        method=method, batch_rows=manifest["batch_rows"]))
                    for method in ("ame", "mem")}
                item["margins_seconds"] = time.monotonic()-started
                item.update(status="passed", model=str(model_path.resolve()),
                            model_sha256=digest(model_path), model_bytes=model_path.stat().st_size,
                            source_unchanged=digest(plan["file"]) == plan["sha256"],
                            scratch_cleaned=True, nobs=fitted.nobs,
                            nobs_original=fitted.nobs_original, dropped_rows=fitted.dropped_rows,
                            compact_state_schema=fitted.extra["control_function_state"]["schema"])
                assert item["source_unchanged"]
                case_path = output / (case_id+"-case.json")
                write(case_path, {**item, "result": fitted.model_dump(mode="json")})
                item["file"] = str(case_path.resolve())
                item["file_sha256"] = digest(case_path)
                del fitted, restored
                gc.collect()
                write(output / "run.json", receipt)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        receipt.update(status="passed",
            peak_process_bytes=int(peak if sys.platform == "darwin" else peak*1024),
            scipy_loaded=any(n == "scipy" or n.startswith("scipy.") for n in sys.modules),
            statsmodels_loaded=any(n == "statsmodels" or n.startswith("statsmodels.") for n in sys.modules))
        assert not receipt["scipy_loaded"] and not receipt["statsmodels_loaded"]
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(output / "run.json", receipt)
    return receipt


def target_oracle(kind, query, state, parameters, *, target="response", variable=None):
    import numpy as np

    def design(terms):
        return np.column_stack([np.ones(len(query)) if name == "Intercept"
                                else query[name].to_numpy(dtype=float) for name in terms])

    kz = len(state["gamma"])
    gamma, beta = parameters[:kz], parameters[kz:]
    z, x = design(state["z_terms"]), design(state["x_terms"])
    eta = x@beta[:-1]+beta[-1]*(query.d.to_numpy(dtype=float)-z@gamma)
    if variable is None and target in {"xb", "stdp"}:
        return eta
    if kind == "gaussian":
        mean, first = eta, np.ones(len(eta))
    elif kind in {"logit", "fractional_logit"}:
        mean = 1/(1+np.exp(-eta))
        first = mean*(1-mean)
    elif kind == "probit":
        mean = np.array([0.5*math.erfc(-v/math.sqrt(2)) for v in eta])
        first = np.exp(-eta**2/2)/math.sqrt(2*math.pi)
    elif kind == "cloglog":
        mean, first = -np.expm1(-np.exp(eta)), np.exp(eta-np.exp(eta))
    else:
        mean = first = np.exp(eta)
    if variable is None:
        return mean
    slope = beta[state["x_terms"].index(variable)] if variable in state["x_terms"] else 0.0
    partial = float(variable == "d")
    if variable in state["z_terms"]:
        partial -= gamma[state["z_terms"].index(variable)]
    return first*(slope+beta[-1]*partial)


def jacobian(function, parameters):
    import numpy as np
    columns = []
    for j, value in enumerate(parameters):
        step = 1e-5*max(1.0, abs(value))
        upper, lower = parameters.copy(), parameters.copy()
        upper[j] += step
        lower[j] -= step
        columns.append((function(upper)-function(lower))/(2*step))
    return np.asarray(columns).T


def check(manifest_path, output):
    """Independent scientific replay of every coefficient and persisted target."""
    import numpy as np
    import pandas as pd

    assert not any(n.split(".")[0] in {"openecon", "torch", "scipy", "statsmodels"}
                   for n in sys.modules), "Run the independent checker in its own process."
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads((output / "run.json").read_text())
    assert manifest["schema"] == PROTOCOL_SCHEMA and receipt["status"] == "passed"
    assert receipt["manifest_sha256"] == digest(manifest_path)
    expected_ids = {plan["id"]+"-"+cov for plan in manifest["plan"]
                    for cov in manifest["covariances"]}
    assert len(receipt["cases"]) == len(expected_ids)
    assert {item["id"] for item in receipt["cases"]} == expected_ids
    checked = []

    def close(actual, expected):
        np.testing.assert_allclose(actual, expected, **manifest["tolerances"], equal_nan=True)

    def intervals(fn, parameters, covariance, *, stdp=False):
        estimate = fn(parameters)
        gradient = jacobian(fn, parameters)
        se = np.sqrt(np.einsum("nk,kl,nl->n", gradient, covariance, gradient))
        return se[:, None] if stdp else np.column_stack((estimate, se,
            estimate-NORMAL_975*se, estimate+NORMAL_975*se))

    def check_margins(record, query, state, parameters, covariance, *, grid=False):
        for method in ("ame", "mem"):
            values = record[method]
            settings = [-0.4, 0.5] if grid else [None]
            for grid_index, setting in enumerate(settings):
                selected = query.assign(z2=setting) if setting is not None else query
                if method == "mem":
                    selected = selected[["x", "d", "z1", "z2"]].mean().to_frame().T
                for var_index, variable in enumerate(("x", "d", "z1")):
                    def fn(p):
                        return np.array([target_oracle(kind, selected, state, p,
                                                     variable=variable).mean()])
                    index = grid_index*3+var_index
                    expected = intervals(fn, parameters, covariance)[0]
                    row = values["rows"][index]
                    close([row["estimate"], row["std_error"], row["ci_low"], row["ci_high"]], expected)
                    close(values["attrs"]["delta_gradients"][index], jacobian(fn, parameters).ravel())

    for plan in manifest["plan"]:
        assert digest(plan["file"]) == plan["sha256"]
        source = (pd.read_csv(plan["file"], dtype_backend="numpy_nullable")
                  if plan["format"] == "csv" else
                  pd.read_parquet(plan["file"], dtype_backend="pyarrow"))
        for covariance in manifest["covariances"]:
            case_id = plan["id"]+"-"+covariance
            item = next(value for value in receipt["cases"] if value["id"] == case_id)
            assert item["status"] == "passed" and item["source_unchanged"] and item["scratch_cleaned"]
            model_path = Path(item["model"])
            assert digest(model_path) == item["model_sha256"]
            assert model_path.stat().st_size <= manifest["max_saved_model_bytes"]
            model = json.loads(model_path.read_text())
            state = model["extra"]["control_function_state"]
            assert state["schema"] == "openecon.control_function.stream.v2"
            assert not {"z", "x", "d", "y", "residual", "design", "fitted", "row_scores",
                        "source_values", "source_index", "sample_positions", "cluster_codes"}.intersection(state)
            assert len(model["sample_positions"]) <= 1000 and len(model["predictions"]) <= 1000
            kind = plan["kind"]
            independent = fit_oracle(kind, source, covariance)
            identity = source_identity(source, covariance)
            for key, expected_identity in identity.items():
                assert state[key] == expected_identity, key
            assert state["z_terms"] == independent["z_terms"] and state["x_terms"] == independent["x_terms"]
            canonical = json.dumps({k: v for k, v in state.items() if k != "integrity_sha256"},
                sort_keys=True, allow_nan=False, ensure_ascii=True, separators=(",", ":"))
            assert state["integrity_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
            for key in ("gamma", "beta", "bread", "meat", "joint_covariance", "criterion"):
                close(state[key], independent[key])
            close(model["covariance_matrix"], independent["joint_covariance"])
            coeffs = np.array([[row[name] for name in ("estimate", "std_error", "statistic",
                "p_value", "ci_low", "ci_high")] for row in model["coefficients"]])
            close(coeffs, independent["coefficient_columns"])
            assert independent["sample"] == plan["samples"][covariance]
            for key in ("nobs", "nobs_original", "dropped_rows"):
                assert model[key] == independent["sample"][key] == item[key]
            assert state["cluster_count"] == independent["sample"]["cluster_count"]
            streamed = model["provenance"]["streaming"]
            assert streamed["actual_numeric_peak_rows"] <= manifest["batch_rows"]
            assert streamed["dense_observation_matrix"] is False
            assert streamed["retained_reporting_rows"] <= 400
            parameters = np.r_[state["gamma"], state["beta"]]
            joint = np.asarray(state["joint_covariance"])
            keep = source[["x", "d", "z1", "z2"]].notna().all(axis=1)
            small = source.iloc[:41]
            small_keep = keep.iloc[:41]
            for target, record in item["small_predictions"].items():
                def fn(p):
                    return target_oracle(kind, small.loc[small_keep], state, p, target=target,
                        variable="d" if target == "derivative" else None)
                actual = pd.DataFrame(record["rows"]).to_numpy(dtype=float)
                assert len(actual) == len(small)
                assert np.isnan(actual[~small_keep]).all()
                close(actual[small_keep], intervals(fn, parameters, joint, stdp=target == "stdp"))
            check_margins(item["small_margins"], small.loc[small_keep], state, parameters, joint, grid=True)
            max_errors = {}
            for target in ("response", "derivative"):
                target_record = item["targets"][target]
                path = Path(target_record["file"])
                assert digest(path) == target_record["sha256"]
                actual = pd.read_parquet(path)
                assert len(actual) == len(source) and actual.index.equals(source.index)
                assert actual.loc[~keep].isna().all().all()
                error = 0.0
                for start in range(0, len(source), manifest["batch_rows"]):
                    query = source.iloc[start:start+manifest["batch_rows"]]
                    selected = query.loc[keep.iloc[start:start+manifest["batch_rows"]]]
                    def fn(p):
                        return target_oracle(kind, selected, state, p,
                                             variable="d" if target == "derivative" else None)
                    expected = intervals(fn, parameters, joint)
                    observed = actual.loc[selected.index].to_numpy(dtype=float)
                    close(observed, expected)
                    error = max(error, float(np.max(np.abs(observed-expected), initial=0)))
                max_errors[target] = error
            check_margins(item["margins"], source.loc[keep], state, parameters, joint)
            checked.append({"id": case_id, "status": "passed", "kind": kind,
                "fit_rows": len(source), "retained_rows": model["nobs"],
                "both_stage_coefficients": True, "full_nonsymmetric_bread": True,
                "full_hc0_cr0_score_meat": True, "joint_cross_covariance": True,
                "coefficient_se_p_ci": True, "sample_geometry": independent["sample"],
                "independent_typed_source_and_sample_hashes": True,
                "actual_numeric_peak_rows": streamed["actual_numeric_peak_rows"],
                "compact_model_bytes": model_path.stat().st_size,
                "saved_restore_all_conditional_targets": True,
                "global_ame_mem": True, "max_target_absolute_errors": max_errors})
    result = {"schema": PROTOCOL_SCHEMA, "status": "passed", "cases": checked,
              "manifest_sha256": digest(manifest_path), "run_sha256": digest(output / "run.json"),
              "physical_fit_rows": sum(plan["rows"]*len(manifest["covariances"]) for plan in manifest["plan"]),
              "peak_process_bytes": receipt["peak_process_bytes"], "source_pin": receipt["source_pin"],
              "oracle_imports": "NumPy/pandas/math only; no SDK/Torch/SciPy/Statsmodels"}
    write(output / "independent-check.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "run", "check"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-pin")
    parser.add_argument("--include-million", action="store_true")
    args = parser.parse_args()
    if args.action in {"generate", "run"} and not args.source_pin:
        parser.error("--source-pin is required to bind the protocol/run to a revision.")
    if args.action != "generate" and args.manifest is None:
        parser.error("--manifest is required for run/check.")
    result = (generate(args.directory.resolve(), source_pin=args.source_pin,
                       include_million=args.include_million)
              if args.action == "generate" else
              run(args.manifest.resolve(), args.directory.resolve(), source_pin=args.source_pin)
              if args.action == "run" else check(args.manifest.resolve(), args.directory.resolve()))
    print(json.dumps({"status": result.get("status", "generated"),
                      "cases": len(result.get("plan", result.get("cases", [])))}))


if __name__ == "__main__":
    main()
