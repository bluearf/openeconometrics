"""Full global dynamic GMM, Windmeijer/AR/subset tests and independent equations."""

import builtins
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import registry
from openecon.econometrics.streaming_dynamic_gmm import fit_streaming
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def fixture(groups=35, periods=7, seed=274):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(groups, periods))
    z = rng.normal(size=(groups, periods))
    y = rng.normal(size=(groups, periods))
    effects = rng.normal(size=groups)
    for t in range(1, periods):
        y[:, t] = (
            0.35 * y[:, t - 1] + 0.2 * x[:, t] - 0.1 * z[:, t] + effects + rng.normal(size=groups)
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


def spec(*, covariance="robust", constant=True, **options):
    return ModelSpec(
        estimator="xtdpd",
        outcome="y",
        predictors=["x", "z"],
        panel="id",
        time="t",
        intercept=constant,
        covariance=covariance,
        options=options,
    )


def numeric_compare(actual, expected, path=""):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key, value in expected.items():
            numeric_compare(actual[key], value, path + "/" + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            numeric_compare(a, b, path + f"/{index}")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert actual == pytest.approx(expected, rel=2e-6, abs=2e-8), path
    elif expected is None or isinstance(expected, bool):
        assert actual == expected, path


def compare(current, frame, rows=13):
    expected = registry.load_entry(registry.get("xtdpd"))(current, frame)
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=rows)
    assert [c.term for c in actual.coefficients] == [c.term for c in expected.coefficients]
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        rtol=2e-6,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=2e-6, atol=2e-8
    )
    numeric_compare(actual.metrics, expected.metrics)
    numeric_compare(actual.tests, expected.tests)
    numeric_compare(actual.extra, expected.extra)
    assert actual.nobs == expected.nobs
    assert actual.dropped_rows == expected.dropped_rows
    assert actual.sample_positions == []
    assert actual.provenance["sample_position_count"] == actual.nobs
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= max(
        rows, frame.groupby("id").size().max()
    )
    saved = ResultBundle.model_validate_json(actual.model_dump_json())
    assert saved.coefficients == actual.coefficients
    assert r"\begin{tabular}" in saved.to_latex()
    json.dumps(actual.model_dump(), allow_nan=False)
    return actual


@pytest.mark.parametrize("system", [False, True])
@pytest.mark.parametrize("orthogonal", [False, True])
@pytest.mark.parametrize("twostep", [False, True])
@pytest.mark.parametrize("covariance", ["robust", "nonrobust"])
@pytest.mark.parametrize("h", [1, 2, 3])
def test_full_source_steps_transform_h_and_all_diagnostics(
    system, orthogonal, twostep, covariance, h
):
    compare(
        spec(system=system, orthogonal=orthogonal, twostep=twostep, covariance=covariance, h=h),
        fixture(),
    )


@pytest.mark.parametrize("system", [False, True])
@pytest.mark.parametrize("covariance", ["robust", "nonrobust"])
@pytest.mark.parametrize("twostep", [False, True])
def test_collapsed_small_time_dummies_and_global_centering(system, covariance, twostep):
    compare(
        spec(
            system=system,
            covariance=covariance,
            twostep=twostep,
            collapse=True,
            small=True,
            time_dummies=True,
        ),
        fixture(seed=356),
    )


@pytest.mark.parametrize("orthogonal", [False, True])
@pytest.mark.parametrize("system", [False, True])
def test_unbalanced_gaps_missing_exact_large_integer_and_separate_instrument_roles(
    orthogonal, system
):
    frame = fixture(groups=42, periods=8, seed=438)
    frame["w"] = np.random.default_rng(861).normal(size=len(frame))
    frame = frame.loc[~((frame.id % 4 == 0) & (frame.t == 3))].copy()
    frame.loc[(frame.id % 5 == 0) & (frame.t == 5), "x"] = np.nan
    frame.id = frame.id.map(lambda value: f"panel-{value}")
    frame.t += 2**53 + 9
    frame = frame.sample(frac=1, random_state=963).reset_index(drop=True)
    groups = [
        {"columns": ["y"], "lags": [2, 4], "equation": "both", "collapse": True},
        {"columns": ["x"], "lags": [1, 3], "equation": "diff"},
    ]
    if system:
        groups.append({"columns": ["w"], "lags": [0, 1], "equation": "level", "collapse": True})
    current = spec(
        system=system,
        orthogonal=orthogonal,
        twostep=True,
        gmm=groups,
        iv=[{"columns": ["z"], "equation": "both" if system else "diff"}],
    ).model_copy(update={"missing": "drop"})
    compare(current, frame, rows=3)


@pytest.mark.parametrize("lags", [0, 2])
@pytest.mark.parametrize("system", [False, True])
def test_dynamic_order_no_constant_and_passthrough_instruments(lags, system):
    frame = fixture(groups=42, periods=9, seed=537)
    iv = [{"columns": ["x", "z"], "equation": "diff", "passthru": True}]
    if system:
        iv.append({"columns": ["x", "z"], "equation": "level"})
    compare(spec(lags=lags, system=system, constant=False, collapse=True, iv=iv), frame, rows=11)


def test_independent_numpy_difference_gmm_windmeijer_sargan_hansen_and_ar():
    groups, periods = 45, 8
    frame = fixture(groups, periods, seed=955)
    y = frame.y.to_numpy().reshape(groups, periods)
    v = frame[["x", "z"]].to_numpy().reshape(groups, periods, 2)
    dy = y[:, 2:] - y[:, 1:-1]
    x = np.concatenate(((y[:, 1:-1] - y[:, :-2])[:, :, None], v[:, 2:] - v[:, 1:-1]), 2)
    dates = np.arange(2, periods)
    lagged = [
        np.where((dates - lag >= 0)[None, :], y[:, np.maximum(dates - lag, 0)], 0)
        for lag in range(2, periods)
    ]
    z = np.concatenate((np.stack(lagged, 2), v[:, 2:] - v[:, 1:-1]), 2)
    width = z.shape[2]
    h = np.eye(periods - 2) * 2 - np.eye(periods - 2, k=1) - np.eye(periods - 2, k=-1)
    omega = sum(zi.T @ h @ zi for zi in z)
    a = np.einsum("gti,gtj->ij", z, x)
    b = np.einsum("gti,gt->i", z, dy)
    w1 = np.linalg.inv(omega)
    bread1 = np.linalg.inv(a.T @ w1 @ a)
    beta1 = bread1 @ a.T @ w1 @ b
    e1 = dy - x @ beta1
    g1 = np.einsum("gti,gt->gi", z, e1)
    w2 = np.linalg.inv(g1.T @ g1)
    bread2 = np.linalg.inv(a.T @ w2 @ a)
    beta2 = bread2 @ a.T @ w2 @ b
    p1 = bread1 @ a.T @ w1
    v1 = p1 @ (g1.T @ g1) @ p1.T
    q = w2 @ (b - a @ beta2)
    ai = np.einsum("gti,gtj->gij", z, x)
    m = np.einsum("gij,g->ij", ai, g1 @ q) + np.einsum("gi,gj->ij", g1, ai.transpose(0, 2, 1) @ q)
    d = bread2 @ a.T @ w2 @ m
    covariance = bread2 + d @ bread2 + bread2 @ d.T + d @ v1 @ d.T
    actual = fit_streaming(
        spec(twostep=True, collapse=True), Dataset.from_frame(frame), batch_rows=13
    )
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], beta2, atol=3e-10)
    np.testing.assert_allclose(actual.covariance_matrix, covariance, atol=3e-10, rtol=2e-7)
    sigma2 = np.sum(e1**2) / (2 * dy.size)
    sargan = (b - a @ beta1) @ w1 @ (b - a @ beta1) / sigma2
    hansen = (b - a @ beta2) @ w2 @ (b - a @ beta2)
    assert actual.tests["sargan"]["statistic"] == pytest.approx(sargan, rel=2e-8)
    assert actual.tests["hansen"]["statistic"] == pytest.approx(hansen, rel=2e-8)
    assert actual.metrics["n_instruments"] == width
    residual = dy - x @ beta2
    moments = np.einsum("gti,gt->gi", z, residual)
    p2 = bread2 @ a.T @ w2
    for lag in [1, 2]:
        w = np.pad(residual[:, :-lag], ((0, 0), (lag, 0)))
        ai = np.sum(w * residual, 1)
        xw = np.einsum("gti,gt->i", x, w)
        variance = ai @ ai - 2 * xw @ p2 @ (moments.T @ ai) + xw @ covariance @ xw
        value = ai.sum() / np.sqrt(variance)
        assert actual.tests[f"ar{lag}"]["statistic"] == pytest.approx(value, rel=2e-8)


def test_datetime_ranking_short_panels_collinearity_and_all_rows_meta(monkeypatch):
    frame = fixture(groups=80, periods=7, seed=529)
    frame["twice"] = 2 * frame.x
    frame = frame.loc[~((frame.id % 11 == 0) & (frame.t >= 2))].copy()
    frame.t = pd.Timestamp("2001-01-01", tz="Europe/Istanbul") + pd.to_timedelta(
        frame.t * 2, unit="D"
    )
    current = spec(system=True, twostep=True, collapse=True, time_dummies=True).model_copy(
        update={"predictors": ["x", "z", "twice"]}
    )
    importer = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"scipy", "statsmodels", "linearmodels", "sklearn"}:
            raise AssertionError("External estimator loaded: " + name)
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with torch.device("meta"):
        actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=17)
        assert torch.ones(1).device.type == "meta"
    expected = registry.load_entry(registry.get("xtdpd"))(current, frame)
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, atol=2e-8, rtol=2e-6
    )
    assert [c.term for c in actual.coefficients] == [c.term for c in expected.coefficients]
    assert actual.nobs > 400
    assert len(actual.predictions) == 400
    assert any("Datetime" in message for message in actual.warnings)


@pytest.mark.parametrize(
    "case", ["fraction", "duplicate", "short", "onepanel", "underidentified", "span"]
)
def test_explicit_fit_domains_cleanup(case, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture()
    current = spec()
    expected = {
        "fraction": "invalid_time",
        "duplicate": "repeated_time_values",
        "short": "insufficient_observations",
        "onepanel": "insufficient_clusters",
        "underidentified": "underidentified",
        "span": "time_span_too_large",
    }[case]
    if case == "fraction":
        frame.t = frame.t.astype(float) + 0.1
    elif case == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif case == "short":
        frame = frame.loc[frame.t < 2]
    elif case == "onepanel":
        frame = frame.loc[frame.id == 0]
    elif case == "underidentified":
        current = spec(gmm=[], iv=[])
    elif case == "span":
        frame.t *= 10_000_000
    with pytest.raises(AnalysisError) as error:
        fit_streaming(current, Dataset.from_frame(frame), batch_rows=11)
    assert error.value.code == expected
    assert not list(tmp_path.iterdir())


def test_memory_disk_and_final_source_replay_guards(tmp_path, monkeypatch):
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
            changed.loc[240, "y"] += 1
        yield changed

    with pytest.raises(AnalysisError) as error:
        fit_streaming(
            spec(),
            Dataset.from_batches(batches, frame.columns.tolist(), row_count=len(frame)),
            batch_rows=11,
        )
    assert error.value.code == "source_changed"
    import openecon.econometrics.streaming_dpanel as storage

    monkeypatch.setattr(storage.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 1})())
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec(), Dataset.from_frame(frame), batch_rows=11)
    assert error.value.code == "insufficient_scratch_space"
    assert not list(tmp_path.iterdir())


def test_complete_panel_reservations_and_peak_plan_are_truthful(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    current = spec(
        collapse=True,
        gmm=[{"columns": ["y"], "lags": [2, 2], "collapse": True}],
    )
    actual = fit_streaming(current, Dataset.from_frame(fixture(periods=11)), batch_rows=3)
    panel_plan = actual.provenance["solver_diagnostics"]["peak_complete_panel_resource_plan"]
    buffers = panel_plan["buffers"]
    assert buffers["panel_SQLite_cache"] == 4 * 1024**2
    assert buffers["global_GMM_factors_and_diagnostics"] > 0
    assert buffers["complete_panel_values_and_transformation"] > 0
    assert buffers["native_panel_time_grid"] > 0
    assert panel_plan["estimated_workspace_bytes"] == sum(buffers.values())
    assert (
        actual.provenance["resource_plans"][0]["estimated_workspace_bytes"]
        >= panel_plan["estimated_workspace_bytes"]
    )
    assert actual.provenance["streaming"]["maximum_batch_rows"] == 11
    assert not list(tmp_path.iterdir())

    # The source reader is tiny, but a full panel still has to fit the guarded
    # transformation block. Refuse before reading its complete SQL row list.
    with use_workspace_budget(8), pytest.raises(AnalysisError) as error:
        fit_streaming(current, Dataset.from_frame(fixture(groups=2, periods=2400)), batch_rows=3)
    assert error.value.code == "workspace_limit"
    assert error.value.resource_plan["operation"] == "one complete native dynamic-panel block"
    assert error.value.resource_plan["buffers"]["complete_panel_values_and_transformation"] > 0
    assert not list(tmp_path.iterdir())
