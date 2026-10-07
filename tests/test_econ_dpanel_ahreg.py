"""AH public contracts, sample boundaries and degeneracy handling."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient
from numpy.testing import assert_allclose
from pydantic import ValidationError
from test_econ_dpanel_ahreg_oracle import oracle, simulate

import openecon as oe
from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.models import ModelSpec, ResultBundle
from openecon.server import create_app


def kwargs(frame):
    return {"data": frame, "y": "y", "x": ["x"], "panel": "id", "time": "time"}


def estimates(result):
    return [coefficient.estimate for coefficient in result.coefficients]


def test_convenience_generic_fit_serialization_latex_and_catalogue():
    frame = simulate(groups=35)
    direct = oe.ahreg(**kwargs(frame))
    spec = ModelSpec(estimator="ahreg", outcome="y", predictors=["x"], panel="id", time="time",
                     cluster="id", intercept=False)
    generic = fit(spec, data=frame)
    assert direct.spec.cluster == generic.spec.cluster == "id"
    assert direct.spec.covariance == generic.spec.covariance == "cluster"
    assert_allclose(estimates(direct), estimates(generic), rtol=1e-12)
    assert_allclose(direct.covariance_matrix, generic.covariance_matrix, rtol=1e-12)
    assert ResultBundle.model_validate_json(direct.model_dump_json()) == direct
    json.dumps(json.loads(direct.model_dump_json()), allow_nan=False)
    assert direct.provenance["precision"] == "float64"
    assert direct.provenance["dense_observation_matrices"] is False
    assert direct.provenance["stata_parity_validated"] is False
    assert direct.inference["distribution"] == "t"
    assert direct.inference["df_inference"] == 34
    assert direct.tests["overidentification"]["df"] == 0
    latex = str(direct.to_latex())
    assert "begin{tabular}" in latex and "L1.y" in latex
    assert "Anderson" in direct.summary()
    assert "ahreg" in dir(oe)
    manifest = oe.capabilities()["estimators"]["ahreg"]
    assert manifest["default_covariance"] == "cluster"
    assert manifest["panel"] == manifest["time"] == "required"
    assert manifest["options"]["instrument"]["choices"] == ["levels", "differences"]


def test_default_cluster_robust_alias_and_explicit_panel_cluster_are_identical():
    frame = simulate(groups=35)
    default = oe.ahreg(**kwargs(frame))
    robust = oe.ahreg(**kwargs(frame), covariance="robust")
    explicit = oe.ahreg(**kwargs(frame), covariance="cluster", cluster="id")
    assert robust.spec.cluster is None
    assert robust.inference["covariance"] == "robust"
    for result in (robust, explicit):
        assert_allclose(estimates(result), estimates(default), rtol=1e-12)
        assert_allclose(result.covariance_matrix, default.covariance_matrix, rtol=1e-12)
        assert result.inference["cluster_column"] == "id"
        assert result.inference["effective_covariance"] == "panel_cluster"


@pytest.mark.parametrize("update", [
    {"intercept": True}, {"covariance": "HC1"}, {"covariance": "hac"},
    {"panel": None}, {"time": None}, {"weights": "x", "weight_type": "aweight"},
    {"categorical": ["x"]}, {"options": {"instrument": "all"}},
    {"options": {"instrument": 2}}, {"options": {"lags": 2}},
    {"cluster": ["id", "time"]}, {"covariance": "cluster", "cluster": None},
])
def test_registry_rejects_unsupported_specifications(update):
    base = {"estimator": "ahreg", "outcome": "y", "predictors": ["x"], "intercept": False,
            "panel": "id", "time": "time", "covariance": "robust"}
    with pytest.raises(ValidationError):
        ModelSpec(**(base | update))


def test_only_model_panel_can_be_clustered_and_x_must_be_a_list():
    frame = simulate(groups=10).assign(region=lambda df: df.id // 2)
    with pytest.raises(AnalysisError, match="model panel") as caught:
        oe.ahreg(**kwargs(frame), cluster="region")
    assert caught.value.code == "invalid_spec"
    with pytest.raises(AnalysisError, match="x must be a list"):
        oe.ahreg(**(kwargs(frame) | {"x": "x"}))


@pytest.mark.parametrize("instrument", ["levels", "differences"])
def test_no_cross_panel_lags_and_each_variant_loses_the_correct_initial_periods(instrument):
    frame = simulate(groups=40, periods=7)
    frame.loc[frame.id < 5, "time"] -= 30
    result = oe.ahreg(**kwargs(frame), instrument=instrument)
    lag = 2 if instrument == "levels" else 3
    assert result.nobs == 40 * (7 - lag)
    assert result.extra["excluded_lag_window_rows"] == 40 * lag
    assert result.metrics["group_min"] == result.metrics["group_max"] == 7 - lag
    assert result.extra["instruments"][0] == ("L2.y" if lag == 2 else "D.L2.y")
    assert result.sample_positions == oracle(frame, instrument)["positions"]


def test_shuffling_string_panels_and_irrelevant_columns_preserves_estimates():
    frame = simulate(groups=40, gaps=True)
    before = oe.ahreg(**kwargs(frame))
    shuffled = frame.sample(frac=1, random_state=44).reset_index(drop=True)
    shuffled["id"] = "panel_" + shuffled.id.astype(str)
    shuffled["unused"] = None
    after = oe.ahreg(**kwargs(shuffled))
    assert_allclose(estimates(after), estimates(before), rtol=1e-10)
    assert_allclose(after.covariance_matrix, before.covariance_matrix, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("offset", [2**53 + 101, 2**62])
def test_large_exact_integer_times_do_not_lose_consecutive_spacing(offset):
    frame = simulate(groups=35)
    before = oe.ahreg(**kwargs(frame))
    after = oe.ahreg(**kwargs(frame.assign(time=frame.time.astype("int64") + offset)))
    assert after.sample_positions == before.sample_positions
    assert_allclose(estimates(after), estimates(before), rtol=1e-12)
    assert_allclose(after.covariance_matrix, before.covariance_matrix, rtol=1e-12)


def test_huge_between_panel_time_span_needs_no_dense_grid(monkeypatch):
    from openecon.econometrics.dpanel import structure

    frame = simulate(groups=35)
    frame.loc[frame.id % 2 == 0, "time"] += 10**12
    expected = oracle(frame)

    def reject_dense_grid(*args, **kwargs):
        pytest.fail("AH must not allocate the dynamic-GMM dense group/time grid")

    monkeypatch.setattr(structure, "layout", reject_dense_grid)
    result = oe.ahreg(**kwargs(frame))
    assert_allclose(estimates(result), expected["beta"], rtol=1e-10)
    assert result.nobs == len(expected["positions"])


@pytest.mark.parametrize("change,code", [
    (lambda d: d.assign(time=pd.to_datetime(d.time, format="%Y")), "invalid_time"),
    (lambda d: d.assign(time=d.time + 0.25), "invalid_time"),
    (lambda d: d.assign(time=False), "repeated_time_values"),
    (lambda d: d.assign(time=d.time.astype("uint64") + 2**63), "invalid_time"),
    (lambda d: d.assign(time=d.time.astype(float) + 1e20), "repeated_time_values"),
    (lambda d: d.assign(time=d.time * 2), "insufficient_observations"),
    (lambda d: pd.concat([d, d.iloc[:1]], ignore_index=True), "repeated_time_values"),
    (lambda d: d.assign(x=np.nan), "missing_values"),
    (lambda d: d.assign(x=np.inf), "non_finite_values"),
    (lambda d: d.assign(x="text"), "non_numeric_column"),
    (lambda d: d.assign(x=d.id), "singular_design"),
    (lambda d: d.loc[d.id == 0], "insufficient_clusters"),
    (lambda d: d.drop(columns="x"), "missing_columns"),
])
def test_invalid_data_fails_explicitly(change, code):
    frame = change(simulate(groups=8))
    with pytest.raises(AnalysisError) as caught:
        oe.ahreg(**kwargs(frame))
    assert caught.value.code == code


def test_collinear_differenced_predictors_and_zero_instrument_fail():
    frame = simulate(groups=10).assign(copy=lambda d: 2 * d.x)
    with pytest.raises(AnalysisError, match="collinear") as caught:
        oe.ahreg(**(kwargs(frame) | {"x": ["x", "copy"]}))
    assert caught.value.code == "singular_design"
    frame.loc[frame.time <= 2006, "y"] = 0
    with pytest.raises(AnalysisError) as caught:
        oe.ahreg(**kwargs(frame), instrument="differences")
    assert caught.value.code == "underidentified"


def test_irrelevant_instrument_is_not_inverted_as_rounding_noise():
    z = np.array([-2.0, -1.0, 1.0, 2.0, 3.0, -3.0])
    signal = np.array([1.0, 2.0, -3.0, 1.0, -2.0, 4.0])
    signal -= z * (z @ signal) / (z @ z)
    rows = []
    for group, (lag2, change) in enumerate(zip(z, signal, strict=True)):
        for period, y in enumerate((lag2, lag2 + change, 2 * lag2 + 0.1 * change)):
            rows.append({"id": group, "time": period, "y": y})
    with pytest.raises(AnalysisError) as caught:
        oe.ahreg(data=pd.DataFrame(rows), y="y", panel="id", time="time")
    assert caught.value.code == "underidentified"


def test_exact_dynamic_fit_does_not_emit_noise_standard_errors():
    rows = []
    for group in range(8):
        prev = group + 0.3
        for period in range(8):
            y = 0.7 * prev + group + 0.1
            rows.append({"id": group, "time": period, "y": y})
            prev = y
    with pytest.raises(AnalysisError) as caught:
        oe.ahreg(data=pd.DataFrame(rows), y="y", panel="id", time="time")
    assert caught.value.code == "perfect_fit"


@pytest.mark.parametrize("instrument", ["levels", "differences"])
def test_too_few_periods_or_residual_degrees_of_freedom_fail(instrument):
    lag = 2 if instrument == "levels" else 3
    with pytest.raises(AnalysisError) as caught:
        oe.ahreg(**kwargs(simulate(groups=2, periods=lag)), instrument=instrument)
    assert caught.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as caught:
        oe.ahreg(**kwargs(simulate(groups=2, periods=lag + 1)), instrument=instrument)
    assert caught.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as caught:
        oe.ahreg(**kwargs(simulate(groups=1, periods=1)), instrument=instrument)
    assert caught.value.code == "insufficient_observations"


def test_nonrobust_differencing_meat_never_allocates_an_n_by_n_matrix(monkeypatch):
    frame = simulate(groups=42, periods=8)
    n = 42 * 6
    original = torch.eye

    def checked_eye(rows, *args, **options):
        assert rows != n, "An N×N H/projection matrix must not be allocated"
        return original(rows, *args, **options)

    monkeypatch.setattr(torch, "eye", checked_eye)
    result = oe.ahreg(**kwargs(frame), covariance="nonrobust")
    assert result.nobs == n
    assert_allclose(result.covariance_matrix, oracle(frame, covariance="nonrobust")["covariance"],
                    rtol=1e-9, atol=1e-12)


def test_authenticated_api_runs_and_persists_generic_ah_spec(tmp_path):
    frame = simulate(groups=35)
    spec = {"estimator": "ahreg", "outcome": "y", "predictors": ["x"], "panel": "id",
            "time": "time", "intercept": False, "covariance": "cluster", "cluster": "id",
            "options": {"instrument": "differences"}}
    expected = oracle(frame, instrument="differences")
    with TestClient(create_app(tmp_path)) as client:
        client.headers["X-OpenEcon-Token"] = client.get("/api/session").json()["token"]
        uploaded = client.post("/api/datasets/upload",
                               files={"file": ("ah.csv", frame.to_csv(index=False), "text/csv")})
        assert uploaded.status_code == 200, uploaded.text
        response = client.post("/api/analyses", json={"dataset_id": uploaded.json()["id"], "spec": spec})
        assert response.status_code == 200, response.text
        stored = response.json()
        assert_allclose([c["estimate"] for c in stored["coefficients"]], expected["beta"], rtol=1e-9)
        assert_allclose(stored["covariance_matrix"], expected["covariance"], rtol=1e-9, atol=1e-12)
        assert client.get(f"/api/results/{stored['id']}").json() == stored
        assert client.get("/api/capabilities").json()["estimators"]["ahreg"]["panel"] == "required"
        invalid = client.post("/api/analyses", json={"dataset_id": uploaded.json()["id"],
                                                   "spec": spec | {"options": {"instrument": "bad"}}})
        assert invalid.status_code == 422
        assert client.get(f"/api/results/{stored['id']}",
                          headers={"X-OpenEcon-Token": "wrong"}).status_code == 401
