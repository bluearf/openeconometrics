"""Independent exact-diffuse recursions and full-source UCM contracts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.econometrics.core import make_spec
from openecon.econometrics.ordered_replay import OrderedReplay
from openecon.econometrics.streaming_ucm import _Problem, _Replay, _kernel, fit_streaming_ucm
from openecon.econometrics.tsmodels import statespace as ss
from openecon.econometrics.tsmodels.ucm import _Problem as DenseProblem, fit_ucm, ucm_forecast
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


def source(frame, rows=11):
    return Dataset.from_batches(
        lambda: (frame.iloc[j : j + rows].copy() for j in range(0, len(frame), rows)),
        list(frame),
        row_count=len(frame),
    )


def data(n=90):
    rng = np.random.default_rng(338192)
    x = rng.normal(size=n)
    level = np.cumsum(rng.normal(scale=0.8, size=n))
    return pd.DataFrame(
        {"y": level + 0.7 * x + rng.normal(scale=0.5, size=n), "x": x, "t": np.arange(n) + 2**54}
    )


def spec(model="llevel", **options):
    return make_spec(
        "ucm",
        outcome="y",
        predictors=["x"],
        time="t",
        intercept=False,
        options={"model": model, **options},
    )


def numpy_filter(values, z, t, q, h, ps, pi):
    a = np.zeros((values.shape[0], len(z)))
    vs = []
    fs = []
    flags = []
    logs = []
    for y in values.T:
        v = y - a @ z
        ms = ps @ z
        f = float(z @ ms + h)
        mi = pi @ z
        fi = float(z @ mi)
        if fi > 1e-10:
            k0 = t @ mi / fi
            k1 = t @ ms / fi - k0 * f / fi
            nextpi = t @ pi @ t.T - np.outer(t @ mi, k0)
            nextps = t @ ps @ t.T - np.outer(t @ ms, k0) - np.outer(t @ mi, k1) + np.diag(q)
            flags.append(True)
            logs.append(math.log(fi))
            vs.append(np.zeros_like(v))
            fs.append(1.0)
        else:
            k0 = t @ ms / f
            nextpi = t @ pi @ t.T
            nextps = t @ ps @ t.T - np.outer(t @ ms, k0) + np.diag(q)
            flags.append(False)
            logs.append(0.0)
            vs.append(v.copy())
            fs.append(f)
        a = a @ t.T + v[:, None] * k0
        ps = (nextps + nextps.T) / 2.0
        pi = (nextpi + nextpi.T) / 2.0
        if pi.size and np.max(np.abs(pi)) <= 1e-10:
            pi[:] = 0.0
    return np.array(vs).T, np.array(fs), np.array(flags), np.array(logs), a, ps


@pytest.mark.parametrize(
    "model",
    [
        "rwalk",
        "llevel",
        "lltrend",
        "strend",
        "rtrend",
        "rwdrift",
        "lldtrend",
        "dconstant",
        "dtrend",
        "ntrend",
    ],
)
def test_full_series_filter_profile_and_joint_cross_products_match_dense_unreduced(model):
    df = data(62)
    specification = spec(model)
    structure = ss.Structure(model)
    scale = 1.3
    start = {name: 0.4 for name in structure.parameters}
    theta = ss.unconstrained(structure, start, scale)[None]
    with OrderedReplay(specification, source(df)) as ordered:
        replay = _Replay(ordered)
        replay.sample.rows = 3
        problem = _Problem(replay, structure, scale)
        result = problem.pieces(theta)
        if model == "ntrend":
            cross = np.stack((df.y, df.x)) @ np.stack((df.y, df.x)).T / 0.4
            expected = {
                "syy": torch.tensor([cross[0, 0]]),
                "sxy": torch.tensor(cross[None, 1:, 0]),
                "sxx": torch.tensor(cross[None, 1:, 1:]),
                "logdet": torch.tensor([len(df) * math.log(0.4)], dtype=torch.float64),
            }
        else:
            expected = DenseProblem(
                structure, torch.tensor(df.y.to_numpy()), torch.tensor(df[["x"]].to_numpy()), scale
            ).pieces(theta, reduce=False)
        for name in ["syy", "sxy", "sxx", "logdet"]:
            torch.testing.assert_close(result[name], expected[name], rtol=3e-10, atol=3e-9)
        assert problem.n == 62


@pytest.mark.parametrize(
    "model,seasonal,cycle",
    [("lltrend", 0, False), ("strend", 4, False), ("llevel", 0, True), ("none", 3, True)],
)
def test_exact_diffuse_filter_against_independent_numpy_with_states_carried_across_blocks(
    model, seasonal, cycle
):
    structure = ss.Structure(model, seasonal, cycle)
    values = {
        name: torch.tensor([0.4 if name != "damping" else 0.81], dtype=torch.float64)
        for name in structure.parameters
    }
    t, q, h, p0 = ss.system(structure, values, 1)
    df = data(41)
    raw = np.stack((df.y, df.x))
    expected = numpy_filter(
        raw,
        structure.z.numpy(),
        t[0].numpy(),
        q[0].numpy(),
        float(h[0]),
        p0[0].numpy(),
        np.diag(structure.diffuse.numpy()),
    )
    a = torch.zeros((1, 2, structure.m), dtype=torch.float64)
    ps = p0
    pi = torch.diag(structure.diffuse)
    collected = []
    for j in range(0, len(df), 3):
        out = _kernel().forward(
            torch.tensor(raw[:, j : j + 3]), structure.z, t, q, h, a, ps, pi, False
        )
        collected.append(out[:4])
        a, ps, pi = out[4:7]
    for i in range(4):
        actual = torch.cat([item[i] for item in collected], -1)
        if i == 0:
            actual = actual[0]
        if i == 1:
            actual = actual[0]
        np.testing.assert_allclose(actual.numpy(), expected[i], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(a[0], expected[4], rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(ps[0], expected[5], rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "opg"])
def test_complete_fit_full_covariance_smoothed_predictions_and_saved_forecast(covariance):
    df = data(94)
    specification = spec("llevel").model_copy(update={"covariance": covariance})
    actual = fit_streaming_ucm(specification, source(df), batch_rows=5)
    expected = fit_ucm(specification, df)
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        rtol=3e-5,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=2e-4, atol=3e-6
    )
    assert actual.metrics["log_likelihood"] == pytest.approx(
        expected.metrics["log_likelihood"], abs=2e-7
    )
    np.testing.assert_allclose(
        [r["fitted"] for r in actual.predictions],
        [r["fitted"] for r in expected.predictions],
        rtol=3e-5,
        atol=2e-6,
    )
    restored = ResultBundle.model_validate_json(actual.model_dump_json())
    future = pd.DataFrame({"x": [0.2, 0.4, -0.3]})
    np.testing.assert_allclose(
        ucm_forecast(restored, 3, exog=future).iloc[:, 1:],
        ucm_forecast(expected, 3, exog=future).iloc[:, 1:],
        rtol=3e-5,
        atol=2e-6,
    )
    assert actual.nobs == len(df) and actual.sample_positions == []
    assert actual.provenance["filter_state_reset"] == "only once at full series start"
    assert actual.inference["small_sample_correction"] == (
        len(df) / (len(df) - 1) if covariance == "robust" else None
    )


def test_actual_parquet_missing_end_sorting_category_reference_and_source_integrity(tmp_path):
    df = data(65)
    df["g"] = pd.Categorical(np.where(np.arange(65) % 2, "b", "a"), categories=["a", "b", "unused"])
    df.loc[0, "y"] = np.nan
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    path = tmp_path / "data.parquet"
    df.to_parquet(path, index=False)
    specification = make_spec(
        "ucm",
        outcome="y",
        predictors=["x", "g"],
        categorical=["g"],
        time="t",
        intercept=False,
        missing="drop",
        options={"model": "llevel"},
    )
    actual = fit_streaming_ucm(specification, scan(path), batch_rows=4)
    assert actual.nobs == 64 and actual.dropped_rows == 1
    assert len(actual.provenance["original_data_hash"]) == 64
    assert actual.provenance["ordered_source_rows"] == 64
    assert actual.extra["last_state"]["last_period"] == 2**54 + 64


@pytest.mark.parametrize(
    "fault,code",
    [
        ("gap", "time_gaps"),
        ("duplicate", "duplicate_time"),
        ("constant", "constant_outcome"),
        ("model", "invalid_model"),
        ("collinear", "collinear_regressors"),
    ],
)
def test_explicit_time_precision_and_model_domains_cleanup(fault, code, tmp_path, monkeypatch):
    df = data(35)
    specification = spec()
    if fault == "gap":
        df.loc[12:, "t"] += 1
    if fault == "duplicate":
        df.loc[11, "t"] = df.loc[10, "t"]
    if fault == "constant":
        df["y"] = 1.0
    if fault == "model":
        specification = spec("none")
    if fault == "collinear":
        df["x"] = 1.0
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with pytest.raises(AnalysisError) as caught:
        fit_streaming_ucm(specification, source(df), batch_rows=4)
    assert caught.value.code == code
    assert not list(tmp_path.iterdir())


def test_global_native_cpu_meta_rng_and_mandatory_workspace(tmp_path, monkeypatch):
    df = data(38)
    specification = spec("rwalk")
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    before = torch.random.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    with torch.device("meta"):
        actual = fit_streaming_ucm(specification, source(df), batch_rows=3)
        assert torch.empty(0).device.type == "meta"
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    assert torch.get_default_dtype() == dtype and actual.nobs == 38
    with use_workspace_budget(8), pytest.raises(AnalysisError) as caught:
        fit_streaming_ucm(specification, source(df))
    assert caught.value.code == "workspace_limit"
    assert not list(tmp_path.iterdir())


def test_disk_backed_all_period_components_and_refilter_forecast(tmp_path, monkeypatch):
    from openecon.econometrics.streaming_ucm import ucm_components_streaming, ucm_forecast_streaming
    from openecon.econometrics.tsmodels.ucm import ucm_components
    import gc

    df = data(58)
    fit = fit_streaming_ucm(spec(), source(df), batch_rows=3)
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    output = ucm_components_streaming(fit, source(df, 4))
    assert isinstance(output, Dataset) and output.row_count == 58
    actual = pd.concat(list(output.iter_batches(batch_rows=5)), ignore_index=True)
    expected = ucm_components(fit, df)
    np.testing.assert_allclose(
        actual.iloc[:, 1:].to_numpy(dtype=float),
        expected.iloc[:, 1:].to_numpy(dtype=float),
        rtol=2e-12,
        atol=2e-12,
    )
    assert actual.period.tolist() == expected.period.tolist()
    assert len(list(tmp_path.iterdir())) == 1
    del output
    gc.collect()
    assert not list(tmp_path.iterdir())
    future = pd.DataFrame({"x": [0.1, -0.2]})
    np.testing.assert_allclose(
        ucm_forecast_streaming(fit, 2, data=source(df), exog=future),
        ucm_forecast(fit, 2, data=df, exog=future),
        rtol=2e-12,
        atol=2e-12,
    )


def test_source_changed_on_final_original_replay_refuses_result(tmp_path, monkeypatch):
    df = data(46)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        for j in range(0, len(df), 5):
            rows = df.iloc[j : j + 5].copy()
            if calls > 1:
                rows["y"] += 0.02
            yield rows

    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    changed = Dataset.from_batches(factory, list(df), row_count=len(df))
    with pytest.raises(AnalysisError) as caught:
        fit_streaming_ucm(spec("rwalk"), changed, batch_rows=3)
    assert caught.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


def test_longer_series_preview_bounded_full_n_and_tail_used_with_seasonal_cycle():
    n = 433
    df = data(n)
    specification = spec("llevel", seasonal=4, cycle=True, cycle_frequency=0.48)
    with OrderedReplay(specification, source(df, 17)) as ordered:
        replay = _Replay(ordered)
        replay.sample.rows = 7
        structure = ss.Structure("llevel", 4, True)
        theta = ss.unconstrained(
            structure,
            {
                "frequency": 0.48,
                "damping": 0.8,
                "var(level)": 0.4,
                "var(seasonal)": 0.07,
                "var(cycle)": 0.2,
                "var(e)": 0.5,
            },
            1.2,
        )[None]
        problem = _Problem(replay, structure, 1.2)
        actual = problem.profile(theta)
        expected = DenseProblem(
            structure, torch.tensor(df.y.to_numpy()), torch.tensor(df[["x"]].to_numpy()), 1.2
        ).profile(theta)
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-10)
        assert len(replay.report) == 400 and problem.n == n
        assert replay.sample.provenance()["streaming"]["maximum_batch_rows"] <= ordered.rows
        assert replay.sample.rows <= 7
