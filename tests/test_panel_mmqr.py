"""Panel MM-QR independent dense-dummy GMM and published-author fixture checks."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ModelSpec, ResultBundle


def sample(seed=12, groups=9, periods=45):
    rng = np.random.default_rng(seed)
    n = groups * periods
    ids = np.repeat(np.arange(groups), periods)
    x = rng.uniform(-1, 1, (n, 2))
    u = rng.normal(size=n) / np.sqrt(2 / np.pi)
    scale = 1 + ids * 0.04 + x @ np.array([0.12, -0.08])
    y = ids * 0.3 + x @ np.array([1.3, -0.8]) + scale * u
    return pd.DataFrame(
        {
            "id": ids,
            "time": np.tile(np.arange(periods), groups),
            "x1": x[:, 0],
            "x2": x[:, 1],
            "y": y,
            "region": ids % 3,
        }
    )


def dense_dummy_oracle(df, taus, bandwidth, cluster=None):
    """Full incidental-parameter system, not the implementation's profiled algebra."""
    df = df.sort_values(["id", "time"], kind="stable")
    codes = pd.factorize(df.id)[0]
    G = codes.max() + 1
    n = len(df)
    x = df[["x1", "x2"]].to_numpy()
    p = x.shape[1]
    L = np.c_[x, np.eye(G)[codes]]
    d = L.shape[1]
    y = df.y.to_numpy()
    loc = np.linalg.lstsq(L, y, rcond=None)[0]
    r = y - L @ loc
    lam = np.mean(r >= 0)
    h = 2 * ((r >= 0) - lam)
    a = h * r
    sc = np.linalg.lstsq(L, a, rcond=None)[0]
    s = L @ sc
    v = a - s
    u = r / s
    qs = np.quantile(u, taus, method="averaged_inverted_cdf")
    f = np.mean(norm.pdf((u[:, None] - qs) / bandwidth), axis=0) / bandwidth
    scores = np.c_[L * r[:, None], L * v[:, None], np.array(taus)[None, :] - (u[:, None] <= qs)]
    A = np.zeros((2 * d + len(taus), 2 * d + len(taus)))
    A[:d, :d] = L.T @ L
    A[d : 2 * d, :d] = L.T @ (h[:, None] * L)
    A[d : 2 * d, d : 2 * d] = L.T @ L
    for j, q in enumerate(qs):
        A[2 * d + j, :d] = f[j] * np.sum(L / s[:, None], axis=0)
        A[2 * d + j, d : 2 * d] = f[j] * q * np.sum(L / s[:, None], axis=0)
        A[2 * d + j, 2 * d + j] = n * f[j]
    phi = scores @ np.linalg.inv(A).T
    if cluster:
        cc = pd.factorize(df[cluster])[0]
        C = cc.max() + 1
        phi = np.array([phi[cc == i].sum(axis=0) for i in range(C)])
        V = phi.T @ phi * C / (C - 1) * (n - 1) / (n - G - p)
    else:
        V = phi.T @ phi
    selector = [*range(p), *range(d, d + p), *range(2 * d, 2 * d + len(taus))]
    joint = V[np.ix_(selector, selector)]
    trans = np.zeros((p * len(taus), len(selector)))
    for j, q in enumerate(qs):
        trans[j * p : (j + 1) * p, :p] = np.eye(p)
        trans[j * p : (j + 1) * p, p : 2 * p] = np.eye(p) * q
        trans[j * p : (j + 1) * p, 2 * p + j] = sc[:p]
    b = np.concatenate([loc[:p] + q * sc[:p] for q in qs])
    return b, trans @ joint @ trans.T, joint, loc, sc, qs


@pytest.mark.parametrize("kind,column", [("HC0", None), ("robust", "id"), ("cluster", "region")])
def test_full_joint_covariance_matches_independent_dense_dummy_moment_system(kind, column):
    df = sample()
    taus = [0.2, 0.5, 0.8]
    h = 0.3
    kwargs = {"cluster": column} if kind == "cluster" else {}
    result = oe.panel_mmqr(
        data=df,
        y="y",
        x=["x1", "x2"],
        panel="id",
        time="time",
        quantiles=taus,
        covariance=kind,
        density_bandwidth=h,
        **kwargs,
    )
    b, v, joint, loc, sc, qs = dense_dummy_oracle(df, taus, h, column)
    assert_allclose([c.estimate for c in result.coefficients], b, atol=1e-12)
    assert_allclose(result.covariance_matrix, v, rtol=1e-10, atol=1e-12)
    assert_allclose(result.extra["joint_moment_covariance"], joint, rtol=1e-10, atol=1e-12)
    assert_allclose(result.extra["location_coefficients"], loc[:2], atol=1e-12)
    assert_allclose(result.extra["scale_coefficients"], sc[:2], atol=1e-12)
    assert_allclose(result.extra["error_quantiles"], qs, atol=1e-12)
    assert abs(np.array(v)[0, 4]) > 1e-5  # covariance ACROSS quantiles is retained
    assert np.linalg.eigvalsh(v).min() > -1e-10
    assert result.inference["df_inference"] == (
        len(df) - 9 - 2 if column is None else df[column].nunique() - 1
    )


def test_sample_positions_effects_non_crossing_json_latex_and_rank():
    df = sample().sample(frac=1, random_state=4).reset_index(drop=True)
    df["fixed"] = df.id * 3
    df["dup"] = df.x1 * 2
    df.loc[7, "y"] = np.nan
    original = df.copy(deep=True)
    result = oe.panel_mmqr(
        data=df, y="y", x=["x1", "x2", "fixed", "dup"], panel="id", time="time", missing="drop"
    )
    expected = df.dropna().sort_values(["id", "time"], kind="stable").index.tolist()
    assert result.sample_positions == expected and result.dropped_rows == 1
    assert set(result.provenance["omitted_terms"]) == {"fixed", "dup"}
    assert result.extra["fitted_scale_sample_positions"] == expected
    assert min(result.extra["fitted_scale"]) > 0
    beta = np.array(result.extra["location_coefficients"])
    gamma = np.array(result.extra["scale_coefficients"])
    rows = df.iloc[expected]
    x = rows[["x1", "x2"]].to_numpy()
    effects = {e["panel"]: e for e in result.extra["individual_effects"]}
    alpha = np.array([effects[i]["location"] for i in rows.id])
    delta = np.array([effects[i]["scale"] for i in rows.id])
    predictions = (
        alpha[:, None]
        + (x @ beta)[:, None]
        + (delta + x @ gamma)[:, None] * np.array(result.extra["error_quantiles"])[None, :]
    )
    assert np.all(np.diff(predictions, axis=1) >= 0)
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    json.dumps(result.model_dump(mode="json"), allow_nan=False)
    assert "\\begin{table}" in result.to_latex()
    assert "q0.25" in result.to_latex()
    pd.testing.assert_frame_equal(original, df)


def test_published_author_nlswork_fixture_and_frozen_independent_reference():
    root = Path(__file__).parent / "fixtures" / "robust"
    reference = json.loads((root / "nlswork-mmqr-reference.json").read_text())
    source = root / "nlswork-mmqr.csv"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == reference["selected_csv_sha256"]
    df = pd.read_csv(source)
    result = oe.panel_mmqr(
        data=df,
        y="ln_wage",
        x=reference["predictors"],
        panel="idcode",
        time="year",
        quantiles=reference["quantiles"],
    )
    assert result.nobs == reference["nobs"]
    assert result.metrics["n_groups"] == reference["n_groups"]
    assert_allclose(result.extra["location_coefficients"], reference["location"], atol=2e-12)
    assert_allclose(result.extra["scale_coefficients"], reference["scale"], atol=2e-12)
    assert_allclose(result.extra["error_quantiles"], reference["error_quantiles"], atol=2e-11)
    assert_allclose(
        [c.estimate for c in result.coefficients], reference["coefficients"], atol=2e-12
    )
    assert_allclose(
        result.extra["joint_moment_covariance"],
        reference["joint_moment_covariance"],
        rtol=3e-8,
        atol=1e-13,
    )
    assert_allclose(result.covariance_matrix, reference["covariance"], rtol=3e-8, atol=1e-13)
    assert reference["actual_stata_execution"] is False
    assert result.provenance["stata_parity_validated"] is False


def test_strict_quantile_scale_panel_time_and_cluster_domains():
    df = sample()
    with pytest.raises(AnalysisError) as error:
        oe.panel_mmqr(data=df, y="y", x=["x1"], panel="id", quantiles=[0.001])
    assert error.value.code == "invalid_quantile"
    with pytest.raises(AnalysisError) as error:
        oe.panel_mmqr(data=df, y="y", x=["x1"], panel="id", quantiles=[0.5, 0.5])
    assert error.value.code == "invalid_quantile"
    df["bad_cluster"] = np.arange(len(df)) % 6
    with pytest.raises(AnalysisError) as error:
        oe.panel_mmqr(
            data=df, y="y", x=["x1"], panel="id", covariance="cluster", cluster="bad_cluster"
        )
    assert error.value.code == "cluster_not_nested"
    with pytest.raises(AnalysisError) as error:
        oe.panel_mmqr(data=pd.concat([df, df.iloc[:1]]), y="y", x=["x1"], panel="id", time="time")
    assert error.value.code == "repeated_time_values"
    with pytest.raises(AnalysisError) as error:
        oe.panel_mmqr(data=df.groupby("id").head(2), y="y", x=["x1"], panel="id")
    assert error.value.code == "insufficient_periods"
    # Construct an admissible location fit whose estimated LINEAR scale crosses zero.
    rng = np.random.default_rng(6)
    negative = sample(seed=6, groups=4, periods=80)
    negative["x1"] = np.tile(np.linspace(-1, 1, 80), 4)
    negative["y"] = rng.normal(size=len(negative)) * np.exp(3 * negative.x1)
    with pytest.raises(AnalysisError) as error:
        oe.panel_mmqr(data=negative, y="y", x=["x1"], panel="id")
    assert error.value.code == "nonpositive_scale"
    # Unsupported weights or unidentified global intercept rejected by ModelSpec.
    with pytest.raises(ValueError):
        ModelSpec(
            estimator="panel_mmqr", outcome="y", predictors=["x1"], panel="id", intercept=True
        )


def test_dataset_and_workspace_guard_and_categorical_within_design():
    from openecon.dataset import Dataset
    from openecon.resources import use_workspace_budget

    df = sample()
    with pytest.raises(AnalysisError) as error:
        oe.panel_mmqr(data=Dataset.from_frame(df), y="y", x=["x1"], panel="id")
    assert error.value.code == "streaming_unsupported"
    df["category"] = np.where(df.time % 3 == 0, "a", "b")
    categorical = oe.panel_mmqr(
        data=df, y="y", x=["x1", "category"], panel="id", categorical=["category"]
    )
    assert categorical.extra["location_terms"] == ["x1", "category[b]"]
    large = sample(groups=30, periods=80)
    for i in range(20):
        large[f"x{i + 3}"] = np.sin(large.time / (i + 2))
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        oe.panel_mmqr(data=large, y="y", x=[f"x{i}" for i in range(1, 23)], panel="id")
    assert error.value.code == "workspace_limit"
