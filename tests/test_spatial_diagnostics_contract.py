"""Saved OLS admission, graph/sample identity and complete diagnostic receipts."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.stats import chi2, f, norm
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.postest.index_codec import decode
from openecon.econometrics.spatial.diagnostics import TESTS, spatial_diagnostics
from openecon.econometrics.spatial.weights import SpatialWeights
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


@lru_cache
def _fixture(missing=False, intercept=True, omitted=False, n=36):
    rng = np.random.default_rng(4221)
    keys = [f"site-{i}" for i in range(n)]
    data = pd.DataFrame({"id": keys, "x": rng.normal(size=n), "z": rng.normal(size=n)},
                        index=pd.Index([f"record-{i // 2}" for i in range(n)], name="source_record"))
    data["y"] = 1.2 + 0.85 * data.x - 0.3 * data.z + rng.normal(size=n)
    predictors = ["x", "z"]
    if omitted:
        data["duplicate_x"] = data.x * 2
        predictors.append("duplicate_x")
    if missing:
        data.iloc[4, data.columns.get_loc("x")] = np.nan
        data.iloc[19, data.columns.get_loc("y")] = np.nan
    result = oe.ols(data=data, y="y", x=predictors, intercept=intercept,
                    missing="drop", device="cpu")
    weights = oe.spatial_weights(
        keys[::-1], [(keys[i], keys[(i + jump) % n], value) for i in range(n)
                    for jump, value in ((1, 1.0), (4, 0.7), (n - 1, 0.4))],
    )
    return data, result.model_dump_json(), weights.to_payload()


def _case(**options):
    data, state, weights = _fixture(**options)
    return data.copy(deep=True), ResultBundle.model_validate_json(state), json.loads(json.dumps(weights))


def _call(data, result, weights, **options):
    return spatial_diagnostics(result, data=data, key="id", spatial_weights=weights,
                               **({"simulations": 31, "seed": 2718} | options))


def _w_payload(payload, keys):
    """Independent keyed COO expansion and induced row normalization."""
    lookup = {key: i for i, key in enumerate(keys)}
    w = np.zeros((len(keys), len(keys)))
    for i, j, value in zip(payload["rows"], payload["cols"], payload["values"], strict=True):
        source, target = payload["keys"][i], payload["keys"][j]
        if source in lookup and target in lookup:
            w[lookup[source], lookup[target]] = value
    if payload["normalization"] == "row":
        sums = w.sum(axis=1)
        w[sums != 0] /= sums[sums != 0, None]
    return w


def _oracle(data, result, weights):
    positions = result.sample_positions
    used = data.iloc[positions]
    x = np.column_stack([np.ones(len(used)) if c.term == "Intercept" else used[c.term]
                         for c in result.coefficients])
    y = used.y.to_numpy()
    beta = np.array([c.estimate for c in result.coefficients])
    e = y - x @ beta
    n, p = x.shape
    m = np.eye(n) - x @ np.linalg.inv(x.T @ x) @ x.T
    w = _w_payload(weights, used.id.tolist())
    a = (w + w.T) / 2
    residual_df = n - p
    c = n / w.sum()
    expected = c * np.trace(m @ a) / residual_df
    variance = 2 * c**2 * (np.trace(m @ a @ m @ a) - np.trace(m @ a)**2 / residual_df) / (residual_df * (residual_df + 2))
    moran = c * (e @ a @ e) / (e @ e)
    z = (moran - expected) / np.sqrt(variance)
    sigma2 = (e @ e) / n
    t = np.trace(w.T @ w + w @ w)
    dl, de = e @ w @ y / sigma2, e @ w @ e / sigma2
    h = w @ (x @ beta)
    d = h @ m @ h / sigma2 + t
    stats = {"moran_normal": moran, "lm_error": de**2 / t, "lm_lag": dl**2 / d,
             "robust_lm_error": (de - t * dl / d)**2 / (t * (1 - t / d)),
             "robust_lm_lag": (dl - de)**2 / (d - t)}
    stats["lm_joint"] = stats["lm_error"] + stats["robust_lm_lag"]
    slopes = [i for i, c in enumerate(result.coefficients) if c.term != "Intercept"]
    z_added = m @ w @ x[:, slopes]
    q = len(slopes)
    projector = z_added @ np.linalg.inv(z_added.T @ z_added) @ z_added.T
    rss_full = e @ (np.eye(n) - projector) @ e
    stats["wx_f"] = ((e @ e - rss_full) / q) / (rss_full / (n - p - q))
    p_values = {name: chi2.sf(value, 2 if name == "lm_joint" else 1) for name, value in stats.items() if name.startswith("lm_") or name.startswith("robust_")}
    p_values["moran_normal"] = 2 * norm.sf(abs(z))
    p_values["wx_f"] = f.sf(stats["wx_f"], q, n - p - q)
    return x, y, e, w, stats, p_values, expected, variance, np.array([dl, de]), np.array([[d, t], [t, t]])


@pytest.mark.parametrize("options", [{}, {"missing": True}, {"intercept": False}, {"omitted": True}])
def test_saved_full_geometry_and_independent_reference(options):
    data, result, weights = _case(**options)
    output = _call(data, result, weights)
    x, y, e, w, stats, p_values, expected, variance, scores, information = _oracle(data, result, weights)
    rows = output["tests"].set_index("test")
    assert rows.index.tolist() == list(TESTS)
    for name, value in stats.items():
        assert_allclose(float(rows.loc[name, "statistic"]), value, rtol=2e-8, atol=2e-10)
        assert_allclose(float(rows.loc[name, "p_value"]), p_values[name], rtol=2e-8, atol=2e-9)
    assert_allclose([rows.loc["moran_normal", "expected"], rows.loc["moran_normal", "variance"]], [expected, variance], rtol=2e-8, atol=2e-10)
    assert_allclose(output["design"].to_numpy(dtype=float), x)
    assert_allclose(output["sample"].observed.to_numpy(dtype=float), y)
    assert_allclose(output["sample"].residual.to_numpy(dtype=float), e)
    assert_allclose(output["weights"].to_numpy(dtype=float), w)
    assert_allclose(output["scores"].to_numpy(dtype=float).ravel(), scores)
    assert_allclose(output["information"].to_numpy(dtype=float), information)
    assert output["sample"].index.tolist() == result.sample_positions
    assert [decode(v) for v in output["sample"].original_index_code] == data.index[result.sample_positions].tolist()
    assert [decode(v) for v in output["weights"].columns] == data.id.iloc[result.sample_positions].tolist()
    assert output.attrs["refitted"] is False
    assert output.attrs["source_model_sha256"] == hashlib.sha256(output.attrs["source_model"].encode()).hexdigest()
    assert json.loads(output.attrs["source_model"]) == result.model_dump(mode="json")
    assert output.attrs["source_verified_geometry"]["df_resid"] == result.nobs - len(result.coefficients)
    assert output.attrs["work_plan"]["total"] == output.attrs["work_plan"]["preparation"] + output.attrs["work_plan"]["kernel"]


def test_no_refit_private_state_or_global_rng_and_cpu_defaults(monkeypatch):
    data, result, weights = _case(missing=True)
    expected = _call(data, result, weights)
    def refused(*args, **kwargs):
        raise AssertionError("An estimator or native fitted-state route was called")
    import openecon.analysis
    import openecon.linear_ols
    monkeypatch.setattr(openecon.analysis, "fit", refused)
    monkeypatch.setattr(openecon.linear_ols, "fit_ols", refused)
    rng_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        with torch.device("meta"):
            actual = _call(data, result, weights)
    finally:
        torch.set_default_dtype(old_dtype)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert_allclose(actual["null_simulation"].to_numpy(dtype=float), expected["null_simulation"].to_numpy(dtype=float), rtol=0, atol=0)
    assert actual.attrs["device"] == "cpu" and actual.attrs["precision"] == "float64"


def test_complete_summary_roundtrip_and_latex():
    data, result, weights = _case(missing=True)
    output = _call(data, result, weights)
    restored = oe.restore_summary(oe.summary_state(output))
    assert list(restored) == list(output)
    assert restored.attrs == output.attrs
    assert len(restored["null_simulation"]) == 31
    for name, original in output.items():
        restored[name].index.names = restored.attrs["table_index_names"][name]
        assert list(restored[name].columns) == list(original.columns)
        assert restored[name].index.tolist() == original.index.tolist()
        assert restored[name].index.names == original.index.names
        assert oe.summary_state(TableSet({name: restored[name]})) == oe.summary_state(TableSet({name: original}))
    assert "tabular" in restored.to_latex()
    state = json.loads(oe.summary_state(output))
    mc = next(v for v in state["attrs"]["test_metadata"] if v["test"] == "moran_gaussian_mc")
    assert mc["p_value"] == (mc["extreme_count"] + 1) / (mc["draws"] + 1)
    assert mc["tie_tolerance"] >= 0 and mc["rng_policy"]


def test_index_labels_are_new_declared_receipt_and_key_order_is_independent():
    data, result, weights = _case()
    original = _call(data, result, weights, tests=["moran_normal"])
    data.index = pd.Index([f"new-label-{i}" for i in range(len(data))], name="declared_now")
    relabeled = _call(data, result, weights, tests=["moran_normal"])
    assert original.attrs["source_data_hash"] == relabeled.attrs["source_data_hash"]
    assert original.attrs["association_sha256"] != relabeled.attrs["association_sha256"]
    assert "caller-declared" in relabeled.attrs["association"]["source"]
    assert [decode(v) for v in relabeled["sample"].original_index_code] == data.index.tolist()
    assert_allclose(original["weights"].to_numpy(dtype=float), relabeled["weights"].to_numpy(dtype=float))


@pytest.mark.parametrize("positions", [[], [-1] + list(range(1, 36)), [True] + list(range(1, 36)), [0.0] + list(range(1, 36)), [0, 0] + list(range(2, 36)), list(range(35, -1, -1)), list(range(35))])
def test_invalid_positions_refused_before_shared_helper(monkeypatch, positions):
    data, result, weights = _case()
    result.sample_positions = positions
    import openecon.econometrics.spatial.diagnostics as module
    monkeypatch.setattr(module, "matched_frame", lambda *a, **k: pytest.fail("Unsafe positions reached shared admission"))
    with pytest.raises(AnalysisError, match="positions"):
        _call(data, result, weights)


@pytest.mark.parametrize("mutation", ["beta", "covariance", "off_diagonal", "se", "statistic", "p_value", "ci", "rss", "df", "df_inference", "provenance_df", "terms", "input_columns", "omissions", "categories", "sample_count", "sample_hash", "covariance_label"])
def test_inconsistent_saved_scientific_state_is_refused(mutation):
    data, result, weights = _case()
    if mutation == "beta":
        result.coefficients[1].estimate += .2
    elif mutation == "covariance":
        result.covariance_matrix = (np.asarray(result.covariance_matrix) * 1.3).tolist()
    elif mutation == "off_diagonal":
        result.covariance_matrix[0][1] += .001
        result.covariance_matrix[1][0] += .001
    elif mutation == "se":
        result.coefficients[0].std_error *= 1.1
    elif mutation == "statistic":
        result.coefficients[0].statistic += .1
    elif mutation == "p_value":
        result.coefficients[0].p_value = .8
    elif mutation == "ci":
        result.coefficients[0].ci_low -= .1
    elif mutation == "rss":
        result.metrics["ss_resid"] += 1
    elif mutation == "df":
        result.metrics["df_resid"] -= 1
    elif mutation == "df_inference":
        result.inference["df_inference"] -= 1
    elif mutation == "provenance_df":
        result.provenance["inference_details"]["df"] -= 1
    elif mutation == "terms":
        result.provenance["design_terms"] = ["Intercept", "z", "x"]
    elif mutation == "input_columns":
        result.provenance["input_columns"] = ["y", "z", "x"]
    elif mutation == "omissions":
        result.provenance["omitted_terms"] = ["x"]
    elif mutation == "categories":
        result.provenance["categories"] = {"x": [0, 1]}
    elif mutation == "sample_count":
        result.provenance["sample_position_count"] -= 1
    elif mutation == "sample_hash":
        result.provenance["sample_hash"] = "0" * 64
    else:
        result.inference["covariance"] = "HC1"
    with pytest.raises(AnalysisError):
        _call(data, result, weights)


@pytest.mark.parametrize("mutation", ["row_order", "outcome", "predictor", "removed_row", "duplicate_key", "missing_key", "duplicate_column", "boolean", "string", "unsafe_integer", "wider_float"])
def test_original_input_contract(mutation):
    data, result, weights = _case()
    if mutation == "row_order":
        data = data.iloc[::-1]
    elif mutation == "outcome":
        data.iloc[0, data.columns.get_loc("y")] += .4
    elif mutation == "predictor":
        data.iloc[0, data.columns.get_loc("x")] += .4
    elif mutation == "removed_row":
        data = data.iloc[:-1]
    elif mutation == "duplicate_key":
        data.iloc[1, data.columns.get_loc("id")] = data.id.iloc[0]
    elif mutation == "missing_key":
        data.iloc[1, data.columns.get_loc("id")] = None
    elif mutation == "duplicate_column":
        data = pd.concat([data, data[["x"]]], axis=1)
    elif mutation == "boolean":
        data["x"] = data.x > 0
    elif mutation == "string":
        data["x"] = data.x.astype(str)
    elif mutation == "unsafe_integer":
        data["x"] = np.arange(len(data), dtype=np.int64) + 2**53
    else:
        if np.dtype(np.longdouble).itemsize <= 8:
            pytest.skip("This platform does not expose a floating dtype wider than float64")
        data["x"] = data.x.to_numpy(dtype=np.longdouble)
    with pytest.raises(AnalysisError):
        _call(data, result, weights)


@pytest.mark.parametrize("mutation", ["missing", "excess", "self", "negative", "nonfinite", "frozen_object"])
def test_complete_original_weight_validation(mutation):
    data, result, weights = _case(missing=True)
    if mutation == "missing":
        weights["keys"] = weights["keys"][:-1]
    elif mutation == "excess":
        weights["keys"].append("extra")
    elif mutation == "self":
        weights["cols"][0] = weights["rows"][0]
    elif mutation == "negative":
        weights["values"][0] = -1
    elif mutation == "nonfinite":
        weights["values"][0] = float("inf")
    else:
        weights = SpatialWeights.from_payload(weights)
        object.__setattr__(weights, "values", (-1.0, *weights.values[1:]))
    with pytest.raises(AnalysisError):
        _call(data, result, weights)


@pytest.mark.parametrize("options", [{"tests": []}, {"tests": ["lm_error", "lm_error"]}, {"tests": ["unknown"]}, {"tests": "lm_error"}, {"alternative": "both"}, {"simulations": True}, {"simulations": 0}, {"seed": True}, {"seed": -1}, {"max_n": True}, {"max_n": 513}, {"max_work": True}, {"wx_predictors": ["Intercept"]}, {"wx_predictors": ["x", "x"]}, {"wx_predictors": ["unknown"]}])
def test_strict_requested_options(options):
    data, result, weights = _case()
    with pytest.raises(AnalysisError):
        _call(data, result, weights, **options)


@pytest.mark.parametrize("limit", ["n", "work", "workspace", "source_json", "physical"])
def test_resource_guards_precede_tensor_copy(monkeypatch, limit):
    data, result, weights = _case(n=150) if limit == "workspace" else _case()
    import openecon.econometrics.spatial.diagnostics as module
    options = {}
    if limit == "n":
        options["max_n"] = 12
    elif limit == "work":
        options["max_work"] = 1
    elif limit == "source_json":
        result.extra["oversize"] = "x" * (3 * 1024**2)
    elif limit == "physical":
        data["id"] = ["x" * (512 * 1024) + str(i) for i in range(len(data))]
    for name in ("as_tensor", "tensor", "ones", "stack"):
        monkeypatch.setattr(module.torch, name, lambda *a, **k: pytest.fail("A tensor was allocated before resource admission"))
    if limit == "workspace":
        with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
            _call(data, result, weights, **options)
        assert error.value.code == "workspace_limit"
    else:
        with pytest.raises(AnalysisError):
            _call(data, result, weights, **options)


@pytest.mark.parametrize("kind", ["robust", "weighted", "categorical", "formula", "streamed", "other", "malformed", "boolean_coefficient"])
def test_unsupported_or_malformed_saved_domains(kind):
    data, result, weights = _case()
    if kind == "robust":
        result.spec.covariance = "HC1"
    elif kind == "weighted":
        result.spec.weights, result.spec.weight_type = "weight", "aweight"
    elif kind == "categorical":
        result.spec.categorical = ["x"]
    elif kind == "formula":
        result.spec.options["terms"] = ["x", "z**2"]
    elif kind == "streamed":
        result.provenance["sample_positions_omitted"] = True
    elif kind == "other":
        result.spec.estimator = "logit"
    elif kind == "boolean_coefficient":
        result.coefficients[0].estimate = True
    else:
        result.spec.options = ["terms"]
    with pytest.raises(AnalysisError):
        _call(data, result, weights)


def test_dataset_and_mapping_refused():
    data, result, weights = _case()
    with pytest.raises(AnalysisError, match="resident"):
        _call(oe.Dataset.from_frame(data), result, weights)
    with pytest.raises(AnalysisError, match="DataFrame"):
        _call(data.to_dict(orient="list"), result, weights)


def test_missing_sample_graph_is_explicit_and_complete_case_family_cannot_change():
    data, result, weights = _case(missing=True)
    output = _call(data, result, weights)
    receipt = output.attrs["association"]
    assert len(receipt["original_keys"]) == result.nobs_original
    assert len(receipt["retained_keys"]) == result.nobs
    assert output.attrs["original_spatial_weights"]["keys"] == data.id.tolist()
    assert output.attrs["effective_spatial_weights"]["keys"] == data.id.iloc[result.sample_positions].tolist()
    assert_allclose(output["weights"].to_numpy(dtype=float).sum(axis=1), 1)
    result.sample_positions[-1] -= 1
    with pytest.raises(AnalysisError):
        _call(data, result, weights)


def test_requested_unidentified_information_requires_explicit_valid_subset():
    rng = np.random.default_rng(128)
    data = pd.DataFrame({"id": list(range(14)), "y": rng.normal(size=14)})
    result = ResultBundle.model_validate_json(oe.ols(data=data, y="y", x=[], device="cpu").model_dump_json())
    weights = oe.spatial_weights(list(range(14)), [(i, (i + 1) % 14, 1) for i in range(14)])
    output = _call(data, result, weights, tests=["lm_error", "lm_lag"])
    assert output["tests"].test.tolist() == ["lm_error", "lm_lag"]
    for tests in (["robust_lm_lag"], ["robust_lm_error"], ["lm_joint"], ["wx_f"]):
        with pytest.raises(AnalysisError):
            _call(data, result, weights, tests=tests)
