"""Bounded CSDID: full unit influences, independent Jacobians and cohort-share variance."""

import builtins
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import registry
from openecon.econometrics.streaming_csdid import fit_streaming
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def fixture(n=150, periods=5, seed=824):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n), periods)
    time = np.tile(np.arange(1, periods + 1), n)
    first = rng.choice([3.0, 4.0, np.nan], n)
    x = rng.normal(size=n)
    cat = rng.choice(["east", "north", "west"], n)
    on = np.isfinite(first[unit]) & (time >= np.nan_to_num(first[unit], nan=99))
    effect = on * (1 + 0.3 * (time - np.nan_to_num(first[unit], nan=0)))
    y = (
        rng.normal(size=n)[unit]
        + 0.3 * time
        + 0.2 * x[unit] * time
        + 0.15 * (cat[unit] == "east") * time
        + effect
        + rng.normal(size=n * periods)
    )
    return pd.DataFrame(
        {"y": y, "id": unit, "t": time, "first": first[unit], "x": x[unit], "c": cat[unit]}
    )


def spec(method="dr", control="never", base="varying", covariates=True, categorical=False):
    return ModelSpec(
        estimator="csdid",
        outcome="y",
        predictors=["x", *(["c"] if categorical else [])] if covariates else [],
        panel="id",
        time="t",
        columns={"treatment_time": "first"},
        covariance="robust",
        categorical=["c"] if categorical else [],
        options={"method": method, "control": control, "base": base},
    )


def compare(current, frame, batch_rows=37):
    expected = registry.load_entry(registry.get("csdid"))(current, frame)
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=batch_rows)
    assert [c.term for c in actual.coefficients] == [c.term for c in expected.coefficients]
    assert [c.equation for c in actual.coefficients] == [c.equation for c in expected.coefficients]
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        atol=2e-9,
        rtol=3e-8,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, atol=2e-9, rtol=3e-7
    )
    for name in ["simple", "dynamic_overall", "group_overall", "calendar_overall"]:
        for key, value in expected.extra[name].items():
            if key != "label":
                assert actual.extra[name][key] == pytest.approx(value, rel=3e-7, abs=2e-9)
    for name in ["dynamic", "group", "calendar"]:
        for row, erow in zip(actual.extra[name], expected.extra[name], strict=True):
            assert row["label"] == erow["label"]
            for key in ["estimate", "std_error", "ci_low", "ci_high"]:
                assert row[key] == pytest.approx(erow[key], rel=3e-7, abs=2e-9)
    assert actual.nobs == expected.nobs
    assert actual.dropped_rows == expected.dropped_rows
    assert actual.sample_positions == []
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= batch_rows
    saved = ResultBundle.model_validate_json(actual.model_dump_json())
    assert saved.coefficients == actual.coefficients
    assert r"\begin{tabular}" in saved.to_latex()
    json.dumps(actual.model_dump(), allow_nan=False)
    return actual


@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
@pytest.mark.parametrize("control", ["never", "notyet"])
@pytest.mark.parametrize("base", ["varying", "universal"])
@pytest.mark.parametrize("covariates", [False, True])
def test_full_source_default_comparison_and_all_aggregations(method, control, base, covariates):
    compare(spec(method, control, base, covariates), fixture())


@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
def test_categorical_base_period_covariates_missing_and_shuffled_unit_keys(method):
    frame = fixture(seed=934).sample(frac=1, random_state=55).reset_index(drop=True)
    frame["id"] = frame.id.map(lambda x: f"unit-{x}")
    frame.loc[frame.id == "unit-7", "y"] = np.nan
    current = spec(method, categorical=True).model_copy(update={"missing": "drop"})
    # Dropping a complete unit retains a balanced panel.
    compare(current, frame, batch_rows=11)


def independent_m_equation(frame, method, g=3, t=4, b=2):
    y = frame.pivot(index="id", columns="t", values="y")
    first = frame.groupby("id")["first"].first().to_numpy()
    x = frame.groupby("id")["x"].first().to_numpy()
    chosen = (first == g) | np.isnan(first)
    dy = (y[t] - y[b]).to_numpy()[chosen]
    x = np.column_stack((np.ones(chosen.sum()), x[chosen]))
    d = (first[chosen] == g).astype(float)
    k = x.shape[1]
    gamma = np.zeros(k)
    if method != "reg":
        for _ in range(100):
            p = 1 / (1 + np.exp(-(x @ gamma)))
            step = np.linalg.solve(x.T @ (x * (p * (1 - p))[:, None]), x.T @ (d - p))
            gamma += step
            if np.max(np.abs(step)) < 1e-12:
                break
    beta = np.linalg.lstsq(x[d == 0], dy[d == 0], rcond=None)[0] if method != "ipw" else np.zeros(k)
    resid = dy - x @ beta
    odds = np.exp(x @ gamma)
    eta1 = np.sum(d * resid) / d.sum()
    eta0 = np.sum((1 - d) * odds * resid) / np.sum((1 - d) * odds)
    theta = np.concatenate(
        (
            gamma if method != "reg" else [],
            beta if method != "ipw" else [],
            [eta1],
            [eta0] if method != "reg" else [],
        )
    )

    def scores(th):
        offset = 0
        g = th[:k] if method != "reg" else np.zeros(k)
        offset += k if method != "reg" else 0
        beta = th[offset : offset + k] if method != "ipw" else np.zeros(k)
        offset += k if method != "ipw" else 0
        h1 = th[offset]
        h0 = th[offset + 1] if method != "reg" else 0
        residual = dy - x @ beta
        cols = []
        if method != "reg":
            cols.append(x * (d - 1 / (1 + np.exp(-(x @ g))))[:, None])
        if method != "ipw":
            cols.append(x * ((1 - d) * residual)[:, None])
        cols.append((d * (residual - h1))[:, None])
        if method != "reg":
            cols.append(((1 - d) * np.exp(x @ g) * (residual - h0))[:, None])
        return np.column_stack(cols)

    eye = np.eye(len(theta)) * 1e-5
    jac = np.column_stack(
        [(scores(theta + v).sum(0) - scores(theta - v).sum(0)) / (2e-5) for v in eye]
    )
    contrast = np.zeros(len(theta))
    contrast[-1 if method == "reg" else -2] = 1
    if method != "reg":
        contrast[-1] = -1
    influence = -scores(theta) @ np.linalg.solve(jac.T, contrast)
    return eta1 - (eta0 if method != "reg" else 0), influence @ influence


@pytest.mark.parametrize("method", ["reg", "ipw", "dr"])
def test_independent_numerical_stacked_jacobian_includes_nuisance_uncertainty(method):
    frame = fixture(seed=835)
    actual = fit_streaming(spec(method), Dataset.from_frame(frame), batch_rows=29)
    index = [c.term for c in actual.coefficients].index("ATT(3,4)")
    estimate, variance = independent_m_equation(frame, method)
    assert actual.coefficients[index].estimate == pytest.approx(estimate, rel=2e-8, abs=2e-9)
    assert actual.covariance_matrix[index][index] == pytest.approx(variance, rel=2e-7, abs=2e-10)


def test_no_covariates_joint_influence_and_estimated_cohort_share_oracle():
    frame = fixture(seed=784)
    actual = fit_streaming(spec(covariates=False), Dataset.from_frame(frame), batch_rows=31)
    y = frame.pivot(index="id", columns="t", values="y")
    first = frame.groupby("id")["first"].first().to_numpy()
    phis = []
    atts = []
    gs = []
    post = []
    for c in actual.coefficients:
        g, t = map(int, c.term[4:-1].split(","))
        b = g - 1 if t >= g else t - 1
        dy = (y[t] - y[b]).to_numpy()
        d = first == g
        control = np.isnan(first)
        m1, m0 = dy[d].mean(), dy[control].mean()
        phis.append(d * (dy - m1) / d.sum() - control * (dy - m0) / control.sum())
        atts.append(m1 - m0)
        gs.append(g)
        post.append(t >= g)
    phi = np.column_stack(phis)
    att = np.array(atts)
    post = np.array(post)
    np.testing.assert_allclose(actual.covariance_matrix, phi.T @ phi, atol=2e-10)
    raw = np.array([(first == g).mean() for g in gs]) * post
    raw_phi = (
        np.column_stack(
            [((first == g).astype(float) - (first == g).mean()) / len(first) for g in gs]
        )
        * post
    )
    total = raw.sum()
    weights = raw / total
    weight_phi = (raw_phi * total - raw[None, :] * raw_phi.sum(1)[:, None]) / total**2
    influence = phi @ weights + weight_phi @ att
    assert actual.extra["simple"]["std_error"] == pytest.approx(np.linalg.norm(influence), rel=2e-9)


def test_early_late_units_and_global_collinear_covariate():
    frame = fixture(seed=377)
    frame.loc[frame.id < 4, "first"] = 1
    frame.loc[(frame.id >= 4) & (frame.id < 8), "first"] = 99
    frame["z"] = frame.x * 2
    current = spec().model_copy(update={"predictors": ["x", "z"]})
    actual = compare(current, frame, batch_rows=19)
    assert actual.dropped_rows == 20
    assert actual.provenance["sample_position_count"] == actual.nobs
    assert actual.extra["covariates"] == ["x", "z"]
    assert any("after the last period" in note for note in actual.warnings)


def test_source_factory_cpu_scope_and_runtime_has_no_external_estimator(monkeypatch):
    frame = fixture(n=120, seed=763)
    current = spec()

    def chunks():
        for start in range(0, len(frame), 17):
            yield frame.iloc[start : start + 17].copy()

    source = Dataset.from_batches(chunks, frame.columns.tolist(), row_count=len(frame))
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"statsmodels", "linearmodels", "sklearn", "scipy"}:
            raise AssertionError("External numerical estimator import attempted")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with torch.device("meta"):
        result = fit_streaming(current, source, batch_rows=13)
        assert torch.empty(0).device.type == "meta"
    assert result.nobs == len(frame)
    assert result.provenance["solver_diagnostics"]["maximum_pair_batch_rows"] <= 13


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "unbalanced",
        "varying_first",
        "bad_first",
        "no_never",
        "all_early",
        "float_period",
    ],
)
def test_global_panel_and_treatment_domain_guards(change):
    frame = fixture(n=80)
    current = spec(covariates=False)
    if change == "duplicate":
        frame = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
    elif change == "unbalanced":
        frame = frame.drop(index=7)
    elif change == "varying_first":
        frame.loc[2, "first"] = 99
    elif change == "bad_first":
        frame.loc[frame.id == 0, "first"] = 3.5
    elif change == "no_never":
        frame["first"] = frame["first"].fillna(3)
    elif change == "all_early":
        frame["first"] = 1
    else:
        frame["t"] = frame.t.astype(float) + 0.01
    with pytest.raises(AnalysisError) as error:
        fit_streaming(current, Dataset.from_frame(frame), batch_rows=23)
    assert error.value.code in {"unbalanced_panel", "invalid_treatment", "invalid_time"}


def test_source_mutation_workspace_and_disk_refusal_clean_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture(n=80)
    with use_workspace_budget(1), pytest.raises(Exception):
        fit_streaming(spec(), Dataset.from_frame(frame), batch_rows=23)
    calls = 0

    def chunks():
        nonlocal calls
        calls += 1
        changed = frame.copy()
        if calls >= 6:
            changed.loc[7, "y"] += 1
        yield changed

    with pytest.raises(AnalysisError) as error:
        fit_streaming(
            spec(),
            Dataset.from_batches(chunks, frame.columns.tolist(), row_count=len(frame)),
            batch_rows=23,
        )
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())
    import openecon.econometrics.streaming_csdid as module

    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 1})())
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec(), Dataset.from_frame(frame), batch_rows=23)
    assert error.value.code == "insufficient_scratch_space"
    assert not list(tmp_path.iterdir())


def test_joint_workspace_retains_live_panel_and_nuisance_reservations(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    seen = []
    original = fit_streaming.__globals__["ReplaySample"].plan_rows

    def capture(sample, operation, buffers, bytes_per_row):
        seen.append((operation, dict(buffers)))
        return original(sample, operation, buffers, bytes_per_row)

    monkeypatch.setattr(fit_streaming.__globals__["ReplaySample"], "plan_rows", capture)
    # Both plans used to fit individually: ~4.5 MiB nuisance state and ~4 MiB
    # joint covariance. They coexist while all comparisons are being fitted.
    with use_workspace_budget(7), pytest.raises(AnalysisError) as error:
        fit_streaming(spec(), Dataset.from_frame(fixture(n=60, periods=45)), batch_rows=23)
    assert error.value.code == "workspace_limit"
    assert seen[-1][0] == "CSDID full unit influence covariance"
    for name in ("panel_SQLite_cache", "comparison_derivatives", "comparison_nuisance_metadata"):
        assert seen[-1][1][name] == seen[-2][1][name]
    assert "joint_TSQR_and_covariance" in seen[-1][1]
    assert not list(tmp_path.iterdir())


def test_unit_rescaling_and_partition_permutation():
    frame = fixture(n=120, seed=833)
    current = spec()
    expected = fit_streaming(current, Dataset.from_frame(frame), batch_rows=23)
    changed = frame.sample(frac=1, random_state=857).reset_index(drop=True)
    changed["y"] *= 1e80
    changed["x"] *= 1e120
    actual = fit_streaming(current, Dataset.from_frame(changed), batch_rows=59)
    np.testing.assert_allclose(
        np.array([c.estimate for c in actual.coefficients]) / 1e80,
        [c.estimate for c in expected.coefficients],
        rtol=1e-7,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        np.asarray(actual.covariance_matrix) / 1e160,
        expected.covariance_matrix,
        rtol=5e-7,
        atol=2e-9,
    )


def test_datetime_ranked_panel_contract():
    frame = fixture(n=100, seed=813)
    frame["first"] = frame["first"] - 1
    frame["t"] = pd.to_datetime("2001-01-01") + pd.to_timedelta(frame["t"] * 2, unit="D")
    actual = compare(spec(), frame, batch_rows=23)
    assert any("Datetime" in warning for warning in actual.warnings)


@pytest.mark.parametrize("method", ["reg", "dr"])
def test_covariate_constant_only_in_controls_is_omitted_or_refused(method):
    frame = fixture(n=110, seed=327)
    frame["x"] = (frame["first"] == 3).astype(float)
    if method == "reg":
        compare(spec(method), frame, batch_rows=19)
    else:
        with pytest.raises(AnalysisError) as error:
            fit_streaming(spec(method), Dataset.from_frame(frame), batch_rows=19)
        assert error.value.code == "overlap_violation"


def test_exact_int64_period_and_treatment_keys_above_float_precision():
    frame = fixture(n=100, seed=872)
    current = spec(covariates=False)
    expected = fit_streaming(current, Dataset.from_frame(frame), batch_rows=23)
    shift = 2**53 + 128
    frame["t"] = frame["t"].astype("int64") + shift
    frame["first"] = frame["first"].astype("Int64") + shift
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=31)
    np.testing.assert_allclose(actual.covariance_matrix, expected.covariance_matrix, atol=2e-10)
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        atol=2e-10,
    )
    assert len({c.term for c in actual.coefficients}) == len(actual.coefficients)
    assert all(str(shift + 3) in c.term or str(shift + 4) in c.term for c in actual.coefficients)


@pytest.mark.parametrize('options', [dict(sample='unbalanced'), dict(sample='repeated_cross_section'),
    dict(anticipation=1), dict(bootstrap_reps=19, seed=8, uniform=True)])
def test_new_resident_options_fail_closed_in_replay(options):
    requested = spec().model_copy(update={'options': options})
    with pytest.raises(AnalysisError) as exc:
        fit_streaming(requested, Dataset.from_frame(fixture()), batch_rows=23)
    assert exc.value.code == 'streaming_options_unsupported'
