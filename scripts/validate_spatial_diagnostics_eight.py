"""Development-only independent verification of eight saved OLS spatial tests.

The forty-cell protocol is immutable. NumPy/SciPy are independent arithmetic
references; runtime diagnostic code must not import either package.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/spatial-diagnostics-eight-2026-10-07"
PROTOCOL = EVIDENCE / "protocol.json"
PROTOCOL_SHA = "2f9cd99b24057b7203a48b0ac7d39e8f675352196764f26a7b326e126c8c3da6"
ERRATUM = EVIDENCE / "protocol-erratum-1.json"
ERRATUM_SHA = "7363c7fd1435905456c651721a9bd0c938df9aeb8eb22aaa9e8a6a0239094442"
METHODS = ["moran_normal", "moran_gaussian_mc", "lm_error", "lm_lag",
           "robust_lm_error", "robust_lm_lag", "lm_joint", "wx_f"]
SOURCE_FILES = [
    "src/openecon/econometrics/spatial/diagnostics.py",
    "src/openecon/econometrics/spatial/diagnostic_kernels.py",
    "src/openecon/econometrics/spatial/weights.py",
    "src/openecon/econometrics/postest/common.py",
    "src/openecon/econometrics/postest/index_codec.py",
    "src/openecon/econometrics/postest/inference.py",
    "src/openecon/econometrics/resident_cpu.py",
    "src/openecon/econometrics/summary_state.py",
    "src/openecon/econometrics/core.py",
    "src/openecon/linear_ols/__init__.py",
    "src/openecon/linear_ols/design.py",
    "src/openecon/linear_ols/estimation.py",
    "src/openecon/linear_ols/spec.py",
    "src/openecon/engines/linalg.py",
    "src/openecon/engines/covariance.py",
    "src/openecon/engines/inference.py",
    "src/openecon/engines/distributions.py",
    "src/openecon/engines/execution.py",
    "src/openecon/engines/contracts.py",
    "src/openecon/analysis.py",
    "src/openecon/analysis_contracts.py",
    "src/openecon/models.py",
    "src/openecon/resources.py",
    "src/openecon/frame.py",
    "src/openecon/dataset.py",
    "src/openecon/streaming_design.py",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_protocol():
    if digest(PROTOCOL) != PROTOCOL_SHA:
        raise RuntimeError("Frozen protocol hash changed; do not run altered gates/cases")
    protocol = json.loads(PROTOCOL.read_text())
    if protocol["planned_cell_count"] != 40 or protocol["methods"] != METHODS:
        raise RuntimeError("Frozen methods/cell count disagree")
    if digest(ERRATUM) != ERRATUM_SHA:
        raise RuntimeError("Pre-outcome theoretical erratum changed")
    erratum = json.loads(ERRATUM.read_text())
    if (erratum["original_protocol_sha256"] != PROTOCOL_SHA
            or erratum["invariant_methods"] != ["moran_normal", "moran_gaussian_mc", "lm_error", "wx_f"]):
        raise RuntimeError("Pre-outcome operative gates disagree")
    return protocol


def pin_sources():
    return {name: digest(ROOT / name) if (ROOT / name).is_file() else None
            for name in SOURCE_FILES}


def compare(actual, expected, *, score=False, probability=False):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError("Numerical reference/output must be finite")
    atol, rtol = ((5e-6, 5e-7) if score else (2e-9, 0.) if probability else (2e-10, 2e-8))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    return float(np.max(np.abs(actual - expected), initial=0.))


def restored(result):
    return ResultBundle.model_validate_json(result.model_dump_json())


def logical_graph(case, keys):
    n = len(keys)
    if case["id"] == "symmetric_ring":
        shifts = [(-1, 1.), (1, 1.), (-4, .35), (4, .35)]
    elif case["id"] == "directed_weighted":
        shifts = [(1, 1.), (3, 2.), (9, .2)]
    elif case["id"] == "unnormalized_isolates":
        shifts = [(2, 1.3), (5, .4)]
    elif case["id"] == "missing_induced_graph":
        shifts = [(1, 1.), (4, .7), (11, .2)]
    else:
        shifts = [(1, .3), (6, 1.4), (13, .6)]
    matrix, edges = np.zeros((n, n)), []
    for i in range(n):
        if case["id"] == "unnormalized_isolates" and i in {0, 7}:
            continue
        for shift, weight in shifts:
            j = (i + shift) % n
            matrix[i, j] += weight
            edges.append((keys[i], keys[j], weight))
    normalization = "none" if case["id"] == "unnormalized_isolates" else "row"
    if normalization == "row":
        sums = matrix.sum(axis=1)
        matrix[sums > 0] /= sums[sums > 0, None]
    payload_keys = keys
    if case["id"] == "affine_key_permutation":
        order = np.random.default_rng(73155).permutation(n)
        payload_keys = [keys[i] for i in order]
    weights = oe.spatial_weights(payload_keys, edges, normalization=normalization, isolates="zero")
    return matrix, weights, normalization


def fit_fixture(case):
    n, rng = case["n_original"], np.random.default_rng(case["fixture_seed"])
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n) + .45 * x1
    values = {"x1": x1, "x2": x2}
    if "x3" in case["predictors"]:
        values["x3"] = rng.normal(size=n) - .2 * x1
    y = 1.3 + .75 * x1 - .55 * x2 + .7 * rng.normal(size=n)
    if "x3" in values:
        y += .3 * values["x3"]
    keys = [f"r{case['id']}_{i}" for i in range(n)]
    source = pd.DataFrame({"key": keys, "y": y, **values})
    if case["id"] == "symmetric_ring":
        source.index = pd.Index([f"label-{i}" for i in range(n)], name="source_row")
    elif case["id"] == "directed_weighted":
        source.index = pd.Index([f"label-{i // 2}" for i in range(n)], name="source_row")
    elif case["id"] == "unnormalized_isolates":
        source.index = pd.Index(np.arange(n) // 2, name="source_row")
    elif case["id"] == "missing_induced_graph":
        source.index = pd.MultiIndex.from_tuples([(f"block-{i // 4}", i % 2) for i in range(n)],
                                                names=["block", "slot"])
    else:
        source.index = pd.DatetimeIndex(pd.Timestamp("2020-01-01")
                                       + pd.to_timedelta(np.arange(n) // 2, unit="D"), name="source_row")
    for name, positions in case.get("missing_roles", {}).items():
        source.iloc[positions, source.columns.get_loc(name)] = np.nan
    original_w, weights, normalization = logical_graph(case, keys)
    # The frozen protocol calls ordinary native OLS 'regress'; the published
    # convenience name is oe.ols, with estimator='ols' and the same contract.
    result = restored(oe.ols(data=source, y="y", x=case["predictors"],
                             covariance="nonrobust", missing="drop", device="cpu"))
    transformed = None
    if case["id"] == "affine_key_permutation":
        changed = source.copy()
        changed["y"] = 4 * source.y + 3 + 1.25 * source.x1 - .75 * source.x2
        changed["x1"], changed["x2"] = 32 * source.x1, source.x2 / 8
        transformed = (changed, restored(oe.ols(data=changed, y="y", x=case["predictors"],
                                                covariance="nonrobust", missing="drop", device="cpu")))
    return {"case": case, "data": source, "result": result, "original_w": original_w,
            "weights": weights, "normalization": normalization, "transformed": transformed}


def geometry(fixture):
    source, result = fixture["data"], fixture["result"]
    positions = np.array(result.sample_positions, dtype=int)
    independently_kept = np.flatnonzero(source[["y", *result.spec.predictors]].notna().all(axis=1))
    np.testing.assert_array_equal(positions, independently_kept)
    x = np.column_stack([np.ones(len(positions)) if coefficient.term == "Intercept"
                         else source.iloc[positions][coefficient.term].to_numpy(float)
                         for coefficient in result.coefficients])
    y = source.iloc[positions].y.to_numpy(float)
    beta = np.array([coefficient.estimate for coefficient in result.coefficients])
    fitted, residual = x @ beta, y - x @ beta
    scaled = x / np.sqrt(np.sum(x * x, axis=0))
    q = np.linalg.qr(scaled, mode="reduced")[0]
    m = np.eye(len(x)) - q @ q.T
    # Independent residual-subspace basis yields the same projection without
    # invoking any numerical helper from the implementation being checked.
    complement = scipy_linalg.null_space(x.T)
    projection_error = compare(m, complement @ complement.T)
    if np.linalg.matrix_rank(x) != len(beta):
        raise AssertionError("Frozen fixture unexpectedly lacks full design rank")
    w = fixture["original_w"][np.ix_(positions, positions)].copy()
    if fixture["normalization"] == "row":
        sums = w.sum(axis=1)
        w[sums > 0] /= sums[sums > 0, None]
    n, k = x.shape
    rss, sigma2 = float(residual @ residual), float(residual @ residual / n)
    fitted_covariance = rss / (n - k) * np.linalg.inv(x.T @ x)
    covariance_error = compare(result.covariance_matrix, fitted_covariance)
    return {"positions": positions, "x": x, "y": y, "beta": beta, "fitted": fitted,
            "residual": residual, "m": m, "w": w, "n": n, "k": k, "rss": rss,
            "sigma2": sigma2, "projection_error": projection_error,
            "fit_covariance_error": covariance_error}


def moran_statistic(residual, w):
    return float(len(residual) / w.sum() * (residual @ w @ residual) / (residual @ residual))


def sac_score_reference(g):
    n, w, y, x, beta, variance = [g[name] for name in ("n", "w", "y", "x", "beta", "sigma2")]

    def log_likelihood(rho, error):
        a, b = np.eye(n) - rho * w, np.eye(n) - error * w
        la, lb = np.linalg.slogdet(a), np.linalg.slogdet(b)
        if la[0] <= 0 or lb[0] <= 0:
            raise AssertionError("Prespecified local SAC score direction left the stable determinant domain")
        residual = b @ (a @ y - x @ beta)
        return la[1] + lb[1] - n / 2 * np.log(2 * np.pi * variance) - residual @ residual / (2 * variance)

    step = 1e-4
    scores = []
    for axis in range(2):
        def point(value):
            return log_likelihood(value, 0.) if axis == 0 else log_likelihood(0., value)
        scores.append((point(-2 * step) - 8 * point(-step) + 8 * point(step) - point(2 * step)) / (12 * step))
    return np.array(scores)


def references(fixture, g):
    case = fixture["case"]
    n, k, m, w, e, y = [g[name] for name in ("n", "k", "m", "w", "residual", "y")]
    d, c = n - k, n / w.sum()
    a = (w + w.T) / 2
    trace = float(np.trace(m @ a))
    square_trace = float(np.trace(m @ a @ m @ a))
    expected = c * trace / d
    variance = 2 * c * c * (square_trace - trace * trace / d) / (d * (d + 2))
    observed = moran_statistic(e, w)
    z = (observed - expected) / np.sqrt(variance)
    alternative = case["alternative"]
    normal_p = (2 * scipy_stats.norm.sf(abs(z)) if alternative == "two-sided" else
                scipy_stats.norm.sf(z) if alternative == "greater" else scipy_stats.norm.cdf(z))
    t = float(np.trace(w.T @ w) + np.trace(w @ w))
    v = m @ w @ g["fitted"]
    h = float(v @ v / g["sigma2"])
    info = np.array([[t + h, t], [t, t]])
    scores = np.array([e @ w @ y, e @ w @ e]) / g["sigma2"]
    finite_scores = sac_score_reference(g)
    score_fd_error = compare(scores, finite_scores, score=True)
    lag, error = scores
    values = {"lm_error": error * error / t, "lm_lag": lag * lag / (t + h),
              "robust_lm_lag": (lag - error) ** 2 / h,
              "robust_lm_error": (error - t / (t + h) * lag) ** 2 / (t * (1 - t / (t + h))),
              "lm_joint": float(scores @ np.linalg.solve(info, scores))}
    compare(values["lm_joint"], values["lm_error"] + values["robust_lm_lag"])
    compare(values["lm_joint"], values["lm_lag"] + values["robust_lm_error"])
    output = {name: {"statistic": float(value), "df_num": 2 if name == "lm_joint" else 1,
                     "p_value": float(scipy_stats.chi2.sf(value, 2 if name == "lm_joint" else 1))}
              for name, value in values.items()}
    output["moran_normal"] = {"statistic": observed, "expected": expected,
                               "variance": variance, "z": z, "p_value": float(normal_p)}
    # The RNG is deliberately replayed, while every projection and quadratic
    # evaluation is independent NumPy arithmetic, never a diagnostic kernel.
    generator = torch.Generator(device="cpu").manual_seed(case["mc_seed"])
    draws = []
    for _ in range(case["simulations"]):
        gaussian = torch.randn((n,), dtype=torch.float64, device="cpu", generator=generator).numpy()
        projected = m @ gaussian
        draws.append(moran_statistic(projected, w))
    draws = np.array(draws)
    complement = scipy_linalg.null_space(g["x"].T)
    residual_eigenvalues = np.linalg.eigvalsh(complement.T @ a @ complement) * c
    if (np.min(draws) < residual_eigenvalues[0] - 2e-10
            or np.max(draws) > residual_eigenvalues[-1] + 2e-10):
        raise AssertionError("Projected Gaussian Moran draw left its exact residual-subspace Rayleigh range")
    if alternative == "two-sided":
        count = int(np.count_nonzero(abs(draws - expected) >= abs(observed - expected)))
    elif alternative == "greater":
        count = int(np.count_nonzero(draws >= observed))
    else:
        count = int(np.count_nonzero(draws <= observed))
    output["moran_gaussian_mc"] = {"statistic": observed, "expected": expected,
                                    "variance": variance, "p_value": (count + 1) / (len(draws) + 1),
                                    "extreme_count": count, "draws": len(draws), "seed": case["mc_seed"]}
    z_wx = np.column_stack([w @ fixture["data"].iloc[g["positions"]][name].to_numpy(float)
                            for name in case["wx_predictors"]])
    augmented = np.column_stack([g["x"], z_wx])
    added = len(case["wx_predictors"])
    if np.linalg.matrix_rank(augmented) != k + added:
        raise AssertionError("Frozen requested WX block unexpectedly lacks full rank")
    beta_augmented = np.linalg.lstsq(augmented, y, rcond=None)[0]
    augmented_residual = y - augmented @ beta_augmented
    rss_augmented = float(augmented_residual @ augmented_residual)
    value = ((g["rss"] - rss_augmented) / added) / (rss_augmented / (n - k - added))
    output["wx_f"] = {"statistic": value, "df_num": added, "df_denom": n - k - added,
                       "p_value": float(scipy_stats.f.sf(value, added, n - k - added))}
    return {"tests": output, "scores": scores, "finite_difference_scores": finite_scores,
            "score_fd_error": score_fd_error, "information": info, "null_statistics": draws,
            "wx_augmented_beta": beta_augmented, "wx_augmented_rss": rss_augmented,
            "moran_projected_trace": trace, "moran_projected_square_trace": square_trace,
            "moran_residual_eigenvalue_bounds": residual_eigenvalues[[0, -1]]}


@contextmanager
def refuse_original_refits():
    analysis = importlib.import_module("openecon.analysis")
    linear = importlib.import_module("openecon.linear_ols")
    originals = [(analysis, "fit", analysis.fit), (linear, "fit_ols", linear.fit_ols)]

    def deny(*args, **kwargs):
        raise AssertionError("A saved diagnostic must not refit the original model")

    for module, name, _ in originals:
        setattr(module, name, deny)
    try:
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def call(fixture, *, tests=None, data=None, result=None, **options):
    case = fixture["case"]
    kwargs = {"data": fixture["data"] if data is None else data,
              "key": "key", "spatial_weights": fixture["weights"],
              "tests": METHODS if tests is None else tests,
              "wx_predictors": case["wx_predictors"] if tests is None or "wx_f" in tests else None,
              "simulations": case["simulations"], "seed": case["mc_seed"],
              "alternative": case["alternative"]}
    kwargs.update(options)
    with refuse_original_refits():
        return oe.spatial_diagnostics(fixture["result"] if result is None else result, **kwargs)


def persistence(output):
    required = sorted(["tests", "sample", "design", "weights", "scores", "information", "null_simulation", "settings"])
    if list(output) != required:
        raise AssertionError(f"Complete table set/order changed: {list(output)}")
    # Caller sidecar is explicit. Shared summary.v1 does not automatically
    # retain pandas Index.names and this verifier does not claim that it does.
    names = {key: list(frame.index.names) for key, frame in output.items()}
    output.attrs["scientific_table_index_names"] = names
    state = oe.summary_state(output)
    recovered = oe.restore_summary(state)
    for key, frame in recovered.items():
        frame.index.names = names[key]
    if oe.summary_state(recovered) != state:
        raise AssertionError("Full tables/metadata changed across finite JSON restoration")
    for key, frame in recovered.items():
        if list(frame.index.names) != names[key] or list(frame.columns) != list(output[key].columns):
            raise AssertionError("Column/index metadata sidecar differs after restoration")
    latex = recovered.to_latex()
    if "tabular" not in latex or not latex:
        raise AssertionError("Restored complete TableSet must render LaTeX")
    return {"summary_sha256": hashlib.sha256(state.encode()).hexdigest(),
            "summary_bytes": len(state.encode()), "latex_bytes": len(latex.encode()),
            "table_shapes": {key: list(frame.shape) for key, frame in output.items()},
            "table_index_names_explicit_sidecar": names}


def validate_output_state(fixture, g, ref, output):
    rows = output["tests"]
    if rows.test.tolist() != METHODS:
        raise AssertionError("Requested eight tests were omitted or reordered")
    sample = output["sample"]
    np.testing.assert_array_equal(sample.index.to_numpy(), g["positions"])
    compare(sample.observed, g["y"])
    compare(sample.fitted, g["fitted"])
    compare(sample.residual, g["residual"])
    np.testing.assert_array_equal(sample.key.to_numpy(), fixture["data"].iloc[g["positions"]].key.to_numpy())
    codec = importlib.import_module("openecon.econometrics.postest.index_codec")
    decoded = [codec.decode(value) for value in sample.original_index_code]
    if decoded != list(fixture["data"].index[g["positions"]]):
        raise AssertionError("Complete caller-associated typed index changed")
    compare(output["design"].to_numpy(), g["x"])
    if list(output["design"].columns) != [c.term for c in fixture["result"].coefficients]:
        raise AssertionError("Saved coefficient order changed in diagnostic design")
    compare(output["weights"].to_numpy(), g["w"])
    compare(output["scores"]["score"], ref["scores"])
    if list(output["scores"].index) != ["lag", "error"]:
        raise AssertionError("Score order must remain lag,error")
    compare(output["information"].to_numpy(), ref["information"])
    compare(output["null_simulation"]["statistic"], ref["null_statistics"])
    source = output.attrs.get("source_model")
    if isinstance(source, str):
        source = json.loads(source)
    if source != fixture["result"].model_dump(mode="json"):
        raise AssertionError("Complete immutable source model not retained")
    def canonical_hash(value):
        canonical = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode()).hexdigest()
    if output.attrs.get("source_model_sha256") != canonical_hash(source):
        raise AssertionError("Complete saved model hash differs from canonical native state")
    original_payload = output.attrs.get("original_spatial_weights")
    if original_payload["keys"] != fixture["data"].key.tolist():
        raise AssertionError("Complete original graph must align to original physical key order")
    if original_payload["normalization"] != fixture["normalization"]:
        raise AssertionError("Declared original graph normalization changed")
    original_matrix = np.zeros(fixture["original_w"].shape)
    for row, column, value in zip(original_payload["rows"], original_payload["cols"], original_payload["values"], strict=True):
        original_matrix[row, column] += value
    compare(original_matrix, fixture["original_w"])
    for prefix in ("original_spatial_weights", "effective_spatial_weights", "association"):
        payload = output.attrs.get(prefix)
        if output.attrs.get(prefix + "_sha256") != canonical_hash(payload):
            raise AssertionError(f"Missing or inconsistent complete identity: {prefix}")
    effective_payload = output.attrs["effective_spatial_weights"]
    retained_keys = fixture["data"].iloc[g["positions"]].key.tolist()
    if effective_payload["keys"] != retained_keys:
        raise AssertionError("Effective weight payload keys do not follow physical retained positions")
    reconstructed = np.zeros(g["w"].shape)
    for row, column, value in zip(effective_payload["rows"], effective_payload["cols"], effective_payload["values"], strict=True):
        reconstructed[row, column] += value
    compare(reconstructed, g["w"])
    association = output.attrs["association"]
    expected_association = {"key_column": "key", "original_positions": list(range(len(fixture["data"]))),
                            "original_keys": fixture["data"].key.tolist(),
                            "original_index_codes": [codec.encode(value) for value in fixture["data"].index],
                            "original_index_names": [codec.encode(value) for value in fixture["data"].index.names],
                            "sample_positions": g["positions"].tolist(), "retained_keys": retained_keys}
    for key, value in expected_association.items():
        if association.get(key) != value:
            raise AssertionError(f"Caller-declared complete association changed: {key}")
    if output.attrs.get("refitted") is not False:
        raise AssertionError("Saved diagnostic must explicitly retain refitted=False")
    return persistence(output)


def expect_refusal(label, operation):
    try:
        operation()
    except AnalysisError as exc:
        return {"boundary": label, "status": "passed", "code": exc.code, "message": str(exc)}
    except Exception as exc:
        return {"boundary": label, "status": "failed", "error": type(exc).__name__, "message": str(exc)}
    return {"boundary": label, "status": "failed", "message": "Unsupported input was accepted"}


def altered_result(result, mutation):
    value = result.model_dump()
    mutation(value)
    return ResultBundle.model_validate(value)


def boundaries(fixture):
    rows, source, result = [], fixture["data"], fixture["result"]
    for option, value in [("alternative", "invalid"), ("simulations", True), ("simulations", 0),
                          ("seed", True), ("seed", -1), ("max_n", len(source) - 1), ("max_work", 1)]:
        rows.append(expect_refusal(f"invalid_or_exhausted_{option}_{value}",
                                   lambda option=option, value=value: call(fixture, **{option: value})))
    for tests in [[], ["moran_normal", "moran_normal"], ["unknown"]]:
        rows.append(expect_refusal(f"invalid_tests_{tests}", lambda tests=tests: call(fixture, tests=tests)))
    for columns in [["Intercept"], ["unknown"], ["x1", "x1"]]:
        rows.append(expect_refusal(f"invalid_wx_{columns}",
                                   lambda columns=columns: call(fixture, tests=["wx_f"], wx_predictors=columns)))
    for name in ["y", "x1"]:
        changed = source.copy()
        changed.iloc[0, changed.columns.get_loc(name)] += .125
        rows.append(expect_refusal(f"changed_retained_{name}", lambda changed=changed: call(fixture, data=changed)))
    reordered = source.iloc[::-1].copy()
    rows.append(expect_refusal("reordered_original_rows_and_index", lambda: call(fixture, data=reordered)))
    duplicate = source.copy()
    duplicate.iloc[1, duplicate.columns.get_loc("key")] = duplicate.iloc[0]["key"]
    rows.append(expect_refusal("duplicate_original_key", lambda: call(fixture, data=duplicate)))
    for key_name in ["absent", "y"]:
        rows.append(expect_refusal(f"invalid_key_{key_name}", lambda key_name=key_name: call(fixture, key=key_name)))
    rows.append(expect_refusal("wrong_original_graph_domain", lambda: call(
        fixture, spatial_weights=oe.spatial_weights([f"wrong-{i}" for i in range(len(source))],
                                                    [(f"wrong-{i}", f"wrong-{(i+1)%len(source)}", 1.)
                                                     for i in range(len(source))]))))
    for field in ["coefficients", "covariance_matrix", "sample_positions"]:
        def mutate(value, field=field):
            if field == "coefficients":
                value[field][0]["estimate"] += .1
            elif field == "covariance_matrix":
                value[field][0][0] *= 1.5
            else:
                value[field] = value[field][::-1]
        bad = altered_result(result, mutate)
        rows.append(expect_refusal(f"mutated_saved_{field}", lambda bad=bad: call(fixture, result=bad)))
    for covariance in ["HC1", "cluster", "hac"]:
        def covariance_spec(value, covariance=covariance):
            value["spec"].update(covariance=covariance)
            if covariance == "cluster":
                value["spec"]["cluster"] = "key"
            elif covariance == "hac":
                value["spec"]["options"] = {"lags": 1}
        bad = altered_result(result, covariance_spec)
        rows.append(expect_refusal(f"unsupported_covariance_{covariance}", lambda bad=bad: call(fixture, result=bad)))
    bad = altered_result(result, lambda value: value["spec"].update(weights="x1", weight_type="aweight"))
    rows.append(expect_refusal("unsupported_weighted_fit", lambda: call(fixture, result=bad)))
    bad = altered_result(result, lambda value: value["spec"].update(categorical=["x1"]))
    rows.append(expect_refusal("unsupported_categorical_fit", lambda: call(fixture, result=bad)))
    bad = altered_result(result, lambda value: value["spec"].update(
        estimator="cnsreg", options={"constraints": [{"terms": {"x1": 1.}, "value": 0.}]}))
    rows.append(expect_refusal("unsupported_non_ols_fit", lambda: call(fixture, result=bad)))
    rows.append(expect_refusal("invalid_result_type", lambda: call(fixture, result={})))
    dataset = Dataset.from_frame(source)
    try:
        rows.append(expect_refusal("unsupported_dataset", lambda: call(fixture, data=dataset)))
    finally:
        dataset.close()
    for kind in ["self", "negative", "nonfinite"]:
        payload = fixture["weights"].to_payload()
        if kind == "self":
            payload["cols"][0] = payload["rows"][0]
        else:
            payload["values"][0] = -1. if kind == "negative" else float("nan")
        rows.append(expect_refusal(f"invalid_graph_{kind}", lambda payload=payload: call(fixture, spatial_weights=payload)))
    # A complete equal-weight graph acts as a scalar on the intercept-orthogonal
    # residual subspace: Moran variance, competing-direction information and
    # additional WX rank vanish for distinct scientific reasons.
    keys = source.key.tolist()
    complete_weights = oe.spatial_weights(keys, [(a, b, 1.) for a in keys for b in keys if a != b])
    for tests in [["moran_normal"], ["moran_gaussian_mc"], ["robust_lm_error"],
                  ["robust_lm_lag"], ["lm_joint"], ["wx_f"]]:
        rows.append(expect_refusal(f"requested_degenerate_direction_{tests[0]}",
                                   lambda tests=tests: call(fixture, tests=tests, spatial_weights=complete_weights)))
    try:
        subset = call(fixture, tests=["lm_error"], spatial_weights=complete_weights)
        if subset["tests"].test.tolist() != ["lm_error"]:
            raise AssertionError("Explicit valid subset cannot silently include an undefined target")
        rows.append({"boundary": "explicit_identified_subset_of_degenerate_graph", "status": "passed"})
    except Exception as exc:
        rows.append({"boundary": "explicit_identified_subset_of_degenerate_graph", "status": "failed",
                     "error": type(exc).__name__, "message": str(exc)})
    exact = altered_result(result, lambda value: value["metrics"].update(ss_resid=0., rmse=0.))
    rows.append(expect_refusal("zero_residual_saved_geometry", lambda: call(fixture, result=exact)))
    wrong_geometry = altered_result(result, lambda value: value.update(nobs=value["nobs"] - 1))
    rows.append(expect_refusal("malformed_saved_sample_count", lambda: call(fixture, result=wrong_geometry)))
    for dtype in ["unsafe_integer", "nonfinite"]:
        changed = source.copy()
        if dtype == "unsafe_integer":
            changed["x1"] = changed["x1"].astype(object)
            changed.iloc[0, changed.columns.get_loc("x1")] = 2**70
        else:
            changed.iloc[0, changed.columns.get_loc("x1")] = np.inf
        rows.append(expect_refusal(f"invalid_sample_numeric_{dtype}", lambda changed=changed: call(fixture, data=changed)))
    if np.finfo(np.longdouble).nmant > np.finfo(float).nmant:
        changed = source.copy()
        changed["x1"] = changed["x1"].to_numpy().astype(np.longdouble)
        rows.append(expect_refusal("wider_float_sample", lambda: call(fixture, data=changed)))
    else:
        rows.append({"boundary": "wider_float_sample", "status": "platform_inapplicable",
                     "reason": "longdouble has no wider mantissa than float64 on this platform"})
    # Rank/df failure is prespecified independently of any method outcome.
    abscissa = np.linspace(-1., 1., 12)
    columns = [f"v{j}" for j in range(1, 8)]
    small_source = pd.DataFrame({name: abscissa**j for j, name in enumerate(columns, 1)})
    small_source["y"] = .2 + sum(.1 * small_source[name] for name in columns) + np.sin(5 * abscissa)
    small_source["key"] = [f"small-{i}" for i in range(len(small_source))]
    small_result = restored(oe.ols(data=small_source, y="y", x=columns, device="cpu"))
    small_weights = oe.spatial_weights(small_source.key.tolist(),
                                       [(f"small-{i}", f"small-{(i+1)%12}", 1.) for i in range(12)])
    rows.append(expect_refusal("wx_denominator_df_exhaustion", lambda: call(
        fixture, data=small_source, result=small_result, tests=["wx_f"],
        spatial_weights=small_weights, wx_predictors=columns)))
    return rows


def run(args):
    protocol = read_protocol()
    start, started = pin_sources(), time.time()
    cells, boundary_rows, fixtures, artifacts = [], [], {}, {}
    for case in protocol["design_cells"]:
        try:
            fixture = fit_fixture(case)
            g = geometry(fixture)
            ref = references(fixture, g)
            output = call(fixture)
            state = validate_output_state(fixture, g, ref, output)
            fixtures[case["id"]] = fixture
            native_rows = output["tests"].set_index("test")
            artifact = {"source_model": fixture["result"].model_dump(mode="json"),
                        "original_weight_payload": fixture["weights"].to_payload(),
                        "effective_matrix": g["w"].tolist(), "design": g["x"].tolist(),
                        "retained_positions": g["positions"].tolist(), "independent_reference": {
                            key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in ref.items()}, "complete_output": json.loads(oe.summary_state(output)),
                        "projection_error": g["projection_error"], "saved_fit_covariance_error": g["fit_covariance_error"],
                        "persistence": state}
            artifacts[case["id"]] = artifact
            if fixture["transformed"] is not None:
                altered_data, altered_fit = fixture["transformed"]
                alternate_fixture = dict(fixture, data=altered_data, result=altered_fit, transformed=None)
                alternate_geometry = geometry(alternate_fixture)
                alternate_reference = references(alternate_fixture, alternate_geometry)
                alternate_output = call(fixture, data=altered_data, result=altered_fit)
                alternate_state = validate_output_state(alternate_fixture, alternate_geometry,
                                                        alternate_reference, alternate_output)
                alternate_rows = alternate_output["tests"].set_index("test")
                artifact["transformed_case_reference"] = {
                    "tests": alternate_reference["tests"],
                    "scores": alternate_reference["scores"].tolist(),
                    "information": alternate_reference["information"].tolist(),
                    "complete_output": json.loads(oe.summary_state(alternate_output)),
                    "persistence": alternate_state,
                    "operative_pre_outcome_erratum_sha256": ERRATUM_SHA,
                }
            for method in METHODS:
                cell = {"id": f"{method}:{case['id']}", "method": method, "fixture": case["id"]}
                try:
                    native, expected = native_rows.loc[method], ref["tests"][method]
                    errors = {}
                    for field, value in expected.items():
                        if field in {"extreme_count", "draws", "seed", "df_num", "df_denom"}:
                            if native[field] != value:
                                raise AssertionError(f"{field} mismatch: {native[field]} versus {value}")
                        else:
                            errors[field] = compare(native[field], value, probability=field == "p_value")
                    if method == "moran_gaussian_mc":
                        if native.p_value != expected["p_value"]:
                            raise AssertionError("Plus-one rank p-value must equal the exact retained integer ratio")
                    if fixture["transformed"] is not None:
                        for field, expected_value in alternate_reference["tests"][method].items():
                            compare(alternate_rows.loc[method][field], expected_value,
                                    probability=field == "p_value")
                        if method in {"moran_normal", "moran_gaussian_mc", "lm_error", "wx_f"}:
                            compare(native.statistic, alternate_rows.loc[method].statistic)
                            compare(native.p_value, alternate_rows.loc[method].p_value, probability=True)
                    cell.update(status="passed", max_errors=errors, native={field: float(native[field])
                                                                           for field in expected}, reference=expected,
                                complete_output_sha256=state["summary_sha256"],
                                finite_difference_score_max_error=ref["score_fd_error"])
                except Exception as exc:
                    cell.update(status="failed", error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
                cells.append(cell)
        except Exception as exc:
            for method in METHODS:
                cells.append({"id": f"{method}:{case['id']}", "method": method, "fixture": case["id"],
                              "status": "failed", "error": type(exc).__name__, "message": str(exc),
                              "traceback": traceback.format_exc()})
    if "symmetric_ring" in fixtures:
        try:
            boundary_rows = boundaries(fixtures["symmetric_ring"])
        except Exception as exc:
            boundary_rows.append({"boundary": "boundary_suite_execution", "status": "failed",
                                  "error": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    else:
        boundary_rows.append({"boundary": "boundary_suite_fixture", "status": "failed", "message": "Prespecified fixture failed"})
    end = pin_sources()
    passed = sum(cell["status"] == "passed" for cell in cells)
    boundary_passed = sum(cell["status"] == "passed" for cell in boundary_rows)
    boundary_inapplicable = sum(cell["status"] == "platform_inapplicable" for cell in boundary_rows)
    accepted = (passed == 40 and len(cells) == 40
                and boundary_passed + boundary_inapplicable == len(boundary_rows)
                and start == end and all(start.values()) and digest(PROTOCOL) == PROTOCOL_SHA
                and digest(ERRATUM) == ERRATUM_SHA)
    receipt = {"schema": "openecon.spatial-diagnostics-eight.scientific.v1", "accepted": accepted,
               "protocol_sha256": PROTOCOL_SHA, "validator_sha256": digest(__file__),
               "pre_outcome_erratum_sha256": ERRATUM_SHA,
               "operative_gate": json.loads(ERRATUM.read_text())["operative_gates"],
               "source_hashes_start": start, "source_hashes_end": end, "source_hashes_stable": start == end,
               "planned_cells": 40, "accepted_cells": passed, "failed_cells": 40 - passed,
               "cells": cells, "boundaries": boundary_rows, "boundary_passed": boundary_passed,
               "boundary_platform_inapplicable": boundary_inapplicable,
               "artifacts": artifacts, "elapsed_seconds": time.time() - started,
               "environment": {"python": sys.version, "platform": platform.platform(),
                               "numpy": np.__version__, "scipy": scipy.__version__,
                               "torch": torch.__version__, "pandas": pd.__version__},
               "scientific_claim": protocol["model_scope"], "primary_sources": protocol["primary_sources"],
               "pre_outcome_clarifications": [
                   "Protocol's regress shorthand executes the native oe.ols estimator=ols route.",
                   "Original OLS hashes exclude index labels and W/key. Their identities are associations supplied now; row reorder is refused by original model-input hash."]}
    output_path = EVIDENCE / args.output
    with output_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    review = {key: receipt[key] for key in ("accepted", "planned_cells", "accepted_cells", "failed_cells",
                                          "boundary_passed", "protocol_sha256", "pre_outcome_erratum_sha256",
                                          "validator_sha256", "source_hashes_stable")}
    review.update(receipt=args.output, receipt_sha256=digest(output_path),
                  failing_cell_ids=[cell["id"] for cell in cells if cell["status"] != "passed"],
                  failing_boundaries=[cell["boundary"] for cell in boundary_rows if cell["status"] == "failed"])
    with (EVIDENCE / args.review).open("x") as stream:
        json.dump(review, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(review, sort_keys=True))
    return 0 if accepted else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--output", default="scientific-validation.json")
    parser.add_argument("--review", default="review.json")
    args = parser.parse_args()
    if args.protocol_only:
        read_protocol()
        print(json.dumps({"protocol_sha256": PROTOCOL_SHA, "pre_outcome_erratum_sha256": ERRATUM_SHA}))
        return 0
    sys.path.insert(0, str(ROOT / "src"))
    global np, pd, torch, scipy, scipy_stats, scipy_linalg, oe, ResultBundle, AnalysisError, Dataset
    import numpy as np
    import pandas as pd
    import scipy
    from scipy import stats as scipy_stats, linalg as scipy_linalg
    import torch
    import openecon as oe
    from openecon.models import ResultBundle
    from openecon.analysis_contracts import AnalysisError
    from openecon.dataset import Dataset
    torch.set_num_threads(1)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
