"""Native full-source dynamic-panel replay and independent lag/IV covariance."""

import builtins
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import registry
from openecon.econometrics.streaming_dpanel import fit_streaming
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def fixture(groups=40, periods=9, seed=352):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(groups, periods))
    z = rng.normal(size=(groups, periods))
    y = rng.normal(size=(groups, periods))
    effects = rng.normal(size=groups)
    for t in range(1, periods):
        y[:, t] = (
            0.4 * y[:, t - 1] + 0.25 * x[:, t] - 0.15 * z[:, t] + effects + rng.normal(size=groups)
        )
    return pd.DataFrame(
        {
            "y": y.ravel(),
            "x": x.ravel(),
            "z": z.ravel(),
            "id": np.repeat(np.arange(groups), periods),
            "t": np.tile(np.arange(periods), groups),
        }
    )


def spec(instrument="levels", covariance="cluster", predictors=("x", "z")):
    return ModelSpec(
        estimator="ahreg",
        outcome="y",
        predictors=list(predictors),
        intercept=False,
        panel="id",
        time="t",
        covariance=covariance,
        cluster="id" if covariance == "cluster" else None,
        options={"instrument": instrument},
    )


def compare(current, frame, rows=17):
    expected = registry.load_entry(registry.get("ahreg"))(current, frame)
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=rows)
    assert [c.term for c in actual.coefficients] == [c.term for c in expected.coefficients]
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        atol=2e-10,
        rtol=2e-8,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, atol=2e-10, rtol=2e-8
    )
    for name, value in expected.metrics.items():
        assert actual.metrics[name] == pytest.approx(value, rel=2e-8, abs=2e-10)
    for name, value in expected.extra["first_stage"][0].items():
        if isinstance(value, (float, int)):
            assert actual.extra["first_stage"][0][name] == pytest.approx(value, rel=2e-8, abs=2e-10)
    assert actual.nobs == expected.nobs
    assert actual.dropped_rows == expected.dropped_rows
    assert actual.sample_positions == []
    assert actual.provenance["sample_position_count"] == actual.nobs
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= rows
    saved = ResultBundle.model_validate_json(actual.model_dump_json())
    assert saved.coefficients == actual.coefficients
    assert r"\begin{tabular}" in saved.to_latex()
    json.dumps(actual.model_dump(), allow_nan=False)
    return actual


@pytest.mark.parametrize("instrument", ["levels", "differences"])
@pytest.mark.parametrize("covariance", ["cluster", "robust", "nonrobust"])
@pytest.mark.parametrize("predictors", [(), ("x", "z")])
def test_full_equations_and_first_stage_dense_parity(instrument, covariance, predictors):
    compare(spec(instrument, covariance, predictors), fixture())


@pytest.mark.parametrize("instrument", ["levels", "differences"])
def test_unbalanced_gaps_missing_signed_integer_identity_and_split_row_boundaries(instrument):
    frame = fixture()
    frame = frame.loc[~((frame.id % 4 == 0) & (frame.t == 4))].copy()
    frame.loc[(frame.id % 5 == 0) & (frame.t == 3), "x"] = np.nan
    frame.t += 2**53 + 7
    frame.id = frame.id.map(lambda value: f"panel-{value}")
    frame = frame.sample(frac=1, random_state=43).reset_index(drop=True)
    compare(spec(instrument).model_copy(update={"missing": "drop"}), frame, rows=1)


def test_independent_numpy_2sls_cluster_and_ma1_equations():
    frame = fixture(groups=30, periods=8, seed=631)
    y = frame.y.to_numpy().reshape(30, 8)
    x = frame[["x", "z"]].to_numpy().reshape(30, 8, 2)
    dy = (y[:, 2:] - y[:, 1:-1]).ravel()
    dx = (x[:, 2:] - x[:, 1:-1]).reshape(-1, 2)
    endogenous = (y[:, 1:-1] - y[:, :-2]).ravel()
    excluded = y[:, :-2].ravel()
    z = np.column_stack((dx, excluded))
    structural = np.column_stack((dx, endogenous))
    score_x = z @ np.linalg.lstsq(z, structural, rcond=None)[0]
    bread = np.linalg.inv(score_x.T @ score_x)
    beta = bread @ score_x.T @ dy
    residual = dy - structural @ beta
    scores = (score_x * residual[:, None]).reshape(30, 6, 3).sum(1)
    n, k = len(dy), 3
    cluster = bread @ (scores.T @ scores * 30 / 29 * (n - 1) / (n - k)) @ bread
    rows = score_x.reshape(30, 6, 3)
    cross = np.einsum("gti,gtj->ij", rows[:, 1:], rows[:, :-1])
    ma1 = (
        bread
        @ (2 * score_x.T @ score_x - cross - cross.T)
        @ bread
        * (residual @ residual / (2 * (n - k)))
    )
    order = [2, 0, 1]
    for covariance, matrix in [("cluster", cluster), ("nonrobust", ma1)]:
        actual = fit_streaming(spec(covariance=covariance), Dataset.from_frame(frame), batch_rows=7)
        np.testing.assert_allclose(
            [c.estimate for c in actual.coefficients], beta[order], atol=3e-10
        )
        np.testing.assert_allclose(
            actual.covariance_matrix, matrix[np.ix_(order, order)], atol=3e-10, rtol=2e-8
        )


def test_large_single_panel_all_rows_native_meta_and_no_external_estimators(monkeypatch):
    frame = fixture(groups=2, periods=230, seed=443)
    frame.loc[450:, "y"] += 2
    importer = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"scipy", "statsmodels", "linearmodels", "sklearn"}:
            raise AssertionError("External estimator loaded: " + name)
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with torch.device("meta"):
        actual = fit_streaming(spec(), Dataset.from_frame(frame), batch_rows=5)
        assert torch.ones(1).device.type == "meta"
    expected = registry.load_entry(registry.get("ahreg"))(spec(), frame)
    assert actual.nobs == 456
    assert len(actual.predictions) == 400
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=2e-8, atol=1e-9
    )


@pytest.mark.parametrize(
    "case", ["dates", "bool", "fraction", "duplicate", "short", "singular", "onepanel"]
)
def test_domain_guards_and_owned_cleanup(case, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture()
    expected = {
        "dates": "invalid_time",
        "bool": "invalid_time",
        "fraction": "invalid_time",
        "duplicate": "repeated_time_values",
        "short": "insufficient_observations",
        "singular": "singular_design",
        "onepanel": "insufficient_clusters",
    }[case]
    if case == "dates":
        frame.t = pd.Timestamp("2001-01-01") + pd.to_timedelta(frame.t, unit="D")
    elif case == "bool":
        frame.t = frame.t > 3
    elif case == "fraction":
        frame.t = frame.t.astype(float) + 0.1
    elif case == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif case == "short":
        frame = frame.loc[frame.t < 2]
    elif case == "singular":
        frame.x = frame.id.astype(float)
    elif case == "onepanel":
        frame = frame.loc[frame.id == 0]
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec(), Dataset.from_frame(frame), batch_rows=11)
    assert error.value.code == expected
    assert not list(tmp_path.iterdir())


def test_workspace_disk_and_source_changed_guards(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture()
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        fit_streaming(spec(), Dataset.from_frame(frame), batch_rows=11)
    assert error.value.code == "workspace_limit"
    calls = 0

    def batches():
        nonlocal calls
        calls += 1
        changed = frame.copy()
        if calls > 4:
            changed.loc[355, "y"] += 1
        yield changed

    with pytest.raises(AnalysisError) as error:
        fit_streaming(
            spec(),
            Dataset.from_batches(batches, frame.columns.tolist(), row_count=len(frame)),
            batch_rows=11,
        )
    assert error.value.code == "source_changed"
    import openecon.econometrics.streaming_dpanel as module

    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 1})())
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec(), Dataset.from_frame(frame), batch_rows=11)
    assert error.value.code == "insufficient_scratch_space"
    assert not list(tmp_path.iterdir())
