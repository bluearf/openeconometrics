"""Full-source PPML disk projections, independent dummy equations and safeguards."""

import builtins
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import registry
from openecon.econometrics.streaming_ppml import fit_streaming
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def fixture(n=180, seed=469):
    r = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "x": r.normal(size=n),
            "z": r.normal(size=n),
            "a": np.arange(n) % 9,
            "b": np.arange(n) % 5,
            "c": r.integers(0, 13, n),
            "d": r.integers(0, 11, n),
            "cat": r.choice(["east", "north", "west"], n),
            "fw": r.integers(1, 4, n).astype(float),
            "aw": r.uniform(0.4, 2.0, n),
            "expo": r.uniform(0.5, 2.0, n),
        }
    )
    frame["off"] = np.log(frame.expo)
    frame["y"] = r.poisson(
        np.exp(0.2 + 0.3 * frame.x - 0.15 * frame.z + 0.04 * frame.a - 0.02 * frame.b) * frame.expo
    ).astype(float)
    return frame


def spec(
    dimensions=("a", "b"),
    covariance="robust",
    weight=None,
    *,
    categorical=False,
    columns=None,
    **options,
):
    return ModelSpec(
        estimator="ppmlhdfe",
        outcome="y",
        predictors=["x", "z", *(["cat"] if categorical else [])],
        intercept=False,
        covariance=covariance,
        cluster=["c", "d"] if covariance == "cluster" else None,
        weights=("fw" if weight == "fweight" else "aw") if weight else None,
        weight_type=weight,
        categorical=["cat"] if categorical else [],
        columns={"absorb": list(dimensions), **(columns or {})},
        options={"tolerance": 1e-11, **options},
    )


def compare(current, frame, rows=37):
    expected = registry.load_entry(registry.get("ppmlhdfe"))(current, frame)
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=rows)
    assert [c.term for c in actual.coefficients] == [c.term for c in expected.coefficients]
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        rtol=2e-7,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=8e-7, atol=2e-9
    )
    for key in [
        "log_likelihood",
        "deviance",
        "df_resid",
        "df_absorbed",
        "n_separated_dropped",
        "n_singletons_dropped",
        "pseudo_r_squared",
    ]:
        assert actual.metrics[key] == pytest.approx(expected.metrics[key], rel=2e-8, abs=2e-9)
    assert actual.extra["null_log_likelihood"] == pytest.approx(
        expected.extra["null_log_likelihood"], rel=2e-9
    )
    assert actual.nobs == expected.nobs
    assert actual.nobs_original == expected.nobs_original
    assert actual.dropped_rows == expected.dropped_rows
    assert actual.sample_positions == []
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= rows
    saved = ResultBundle.model_validate_json(actual.model_dump_json())
    assert saved.coefficients == actual.coefficients
    assert r"\begin{tabular}" in saved.to_latex()
    json.dumps(actual.model_dump(), allow_nan=False)
    return actual


@pytest.mark.parametrize("dimensions", [("a",), ("a", "b")])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
@pytest.mark.parametrize("weight", [None, "fweight", "aweight", "pweight"])
def test_default_weighted_and_two_way_covariance_parity(dimensions, covariance, weight):
    if weight == "pweight" and covariance == "nonrobust":
        pytest.skip("Native probability weights require robust or clustered covariance.")
    compare(spec(dimensions, covariance, weight), fixture())


@pytest.mark.parametrize("role", ["offset", "exposure"])
def test_categorical_offset_missing_absorbed_and_global_collinear_terms(role):
    frame = fixture(seed=487)
    frame["constant_group"] = frame.a.astype(float)
    frame["twice"] = 2 * frame.x
    frame.loc[[4, 8, 17], "z"] = np.nan
    current = spec(
        categorical=True, columns={role: "off" if role == "offset" else "expo"}
    ).model_copy(
        update={"predictors": ["x", "z", "cat", "constant_group", "twice"], "missing": "drop"}
    )
    actual = compare(current, frame, rows=11)
    assert "constant_group" not in [c.term for c in actual.coefficients]
    assert "twice" not in [c.term for c in actual.coefficients]


def dummy_oracle(frame, dimensions, *, weights=None, offset=None):
    """Independent NumPy IRLS over explicit FE dummies, with no estimator package."""
    x = frame[["x", "z"]].to_numpy(float)
    dummy = np.column_stack([pd.get_dummies(frame[name]).to_numpy(float) for name in dimensions])
    design = np.column_stack((x, dummy))
    y = frame.y.to_numpy(float)
    w = np.ones(len(y)) if weights is None else weights
    off = np.zeros(len(y)) if offset is None else offset
    eta = np.log((y + np.average(y, weights=w)) / 2) - off
    for _ in range(200):
        mu = np.exp(eta + off)
        z = eta + (y - mu) / mu
        root = np.sqrt(w * mu)
        beta = np.linalg.lstsq(design * root[:, None], z * root, rcond=None)[0]
        change = np.max(np.abs(design @ beta - eta))
        eta = design @ beta
        if change < 1e-12:
            break
    mu = np.exp(eta + off)
    bread = np.linalg.pinv(design.T @ (design * (w * mu)[:, None]))
    influence = bread[:2] @ design.T * (w * (y - mu))[None, :]
    return beta[:2], influence, bread[:2, :2], np.linalg.matrix_rank(dummy), mu


@pytest.mark.parametrize("dimensions", [("a",), ("a", "b")])
def test_independent_dummy_coefficients_information_and_robust_covariance(dimensions):
    frame = fixture(seed=754)
    beta, influence, bread, rank, _ = dummy_oracle(frame, dimensions, offset=frame.off.to_numpy())
    for covariance in ("nonrobust", "robust"):
        actual = fit_streaming(
            spec(dimensions, covariance, columns={"offset": "off"}),
            Dataset.from_frame(frame),
            batch_rows=29,
        )
        matrix = (
            bread
            if covariance == "nonrobust"
            else influence @ influence.T * len(frame) / (len(frame) - rank - 2)
        )
        np.testing.assert_allclose(
            [c.estimate for c in actual.coefficients], beta, rtol=2e-7, atol=2e-9
        )
        np.testing.assert_allclose(actual.covariance_matrix, matrix, rtol=8e-7, atol=2e-9)


def test_both_dimensions_nested_in_different_clusters_independent_dof_and_covariance():
    frame = fixture(n=270, seed=813)
    beta, _, bread, _, mu = dummy_oracle(frame, ("a", "b"))
    dummy = np.column_stack([pd.get_dummies(frame[name]).to_numpy(float) for name in ("a", "b")])
    x = frame[["x", "z"]].to_numpy(float)
    root = np.sqrt(mu)
    within = x - dummy @ np.linalg.lstsq(dummy * root[:, None], x * root[:, None], rcond=None)[0]
    raw_scores = within * (frame.y.to_numpy() - mu)[:, None]
    scores = []
    for names in (["a"], ["b"], ["a", "b"]):
        grouped = pd.DataFrame(raw_scores).groupby([frame[name] for name in names]).sum().to_numpy()
        scores.append(grouped.T @ grouped)
    matrix = scores[0] + scores[1] - scores[2]
    eigen, vectors = np.linalg.eigh(matrix)
    matrix = (vectors * np.maximum(eigen, 0)) @ vectors.T
    matrix = bread @ matrix @ bread
    n, g = len(frame), min(frame.a.nunique(), frame.b.nunique())
    # Each FE dimension is fully nested in a DIFFERENT cluster dimension.
    # Neither is charged twice; just the common constant plus the two slopes.
    matrix *= g / (g - 1) * (n - 1) / (n - 3)
    for columns in (["a", "b"], ["b", "a"]):
        current = spec(covariance="cluster").model_copy(update={"cluster": columns})
        actual = compare(current, frame, rows=23)
        assert actual.metrics["df_absorbed"] == 1
        assert all(row["nested"] for row in actual.extra["absorbed"])
        np.testing.assert_allclose([c.estimate for c in actual.coefficients], beta, atol=2e-9)
        np.testing.assert_allclose(actual.covariance_matrix, matrix, atol=2e-9, rtol=8e-7)


def test_iterated_zero_groups_singletons_and_frequency_replicated_singleton():
    frame = fixture(seed=724)
    zero = frame.iloc[:5].copy().assign(a=90, b=91, y=0.0)
    lone = frame.iloc[:1].copy().assign(a=94, b=95, y=2.0, fw=2.0)
    original = pd.concat([frame, zero, lone], ignore_index=True)
    actual = compare(spec(), original, rows=19)
    assert actual.metrics["n_separated_dropped"] == 5
    assert actual.metrics["n_singletons_dropped"] == 1
    replicated = compare(spec(weight="fweight"), original, rows=19)
    assert replicated.metrics["n_singletons_dropped"] == 0
    assert replicated.nobs == original.fw.sum() - zero.fw.sum()
    compare(spec(drop_singletons=False), original, rows=19)


def test_three_dimensions_and_partition_rescaling_invariance():
    frame = fixture(n=180, seed=815)
    frame["e"] = np.arange(len(frame)) % 4
    current = spec(("a", "b", "e"))
    actual = compare(current, frame, rows=31)
    changed = frame.sample(frac=1, random_state=942).reset_index(drop=True)
    changed["x"] *= 1e120
    other = fit_streaming(current, Dataset.from_frame(changed), batch_rows=13)
    scale = np.array([1e120, 1.0])
    np.testing.assert_allclose(
        np.array([c.estimate for c in other.coefficients]) * scale,
        [c.estimate for c in actual.coefficients],
        rtol=2e-6,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(other.covariance_matrix) * np.outer(scale, scale),
        actual.covariance_matrix,
        rtol=2e-6,
        atol=1e-9,
    )


def test_all_rows_beyond_report_sample_used_and_native_meta_scope_restored(monkeypatch):
    frame = fixture(n=470, seed=539)
    frame.loc[450:, "y"] += 13
    importer = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"scipy", "statsmodels", "linearmodels", "sklearn"}:
            raise AssertionError("External numerical estimator loaded: " + name)
        return importer(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with torch.device("meta"):
        actual = fit_streaming(spec(("a",)), Dataset.from_frame(frame), batch_rows=41)
        assert torch.ones(1).device.type == "meta"
    expected = registry.load_entry(registry.get("ppmlhdfe"))(spec(("a",)), frame)
    assert actual.nobs == len(frame)
    assert len(actual.predictions) == 400
    np.testing.assert_allclose(actual.covariance_matrix, expected.covariance_matrix, rtol=2e-6)


@pytest.mark.parametrize(
    "case", ["zero", "negative", "exposure", "absorbed", "onecluster", "iterations", "separation"]
)
def test_explicit_fit_failure_domains_and_scratch_cleanup(case, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture(n=180)
    current = spec()
    expected = {
        "zero": "constant_outcome",
        "negative": "invalid_count_outcome",
        "exposure": "invalid_exposure",
        "absorbed": "empty_design",
        "onecluster": "insufficient_clusters",
        "iterations": "nonconvergence",
        "separation": "separation_detected",
    }[case]
    if case == "zero":
        frame["y"] = 0
    elif case == "negative":
        frame.loc[170, "y"] = -1
    elif case == "exposure":
        frame.loc[170, "expo"] = 0
        current = spec(columns={"exposure": "expo"})
    elif case == "absorbed":
        frame["x"] = frame.a.astype(float)
        current = current.model_copy(update={"predictors": ["x"]})
    elif case == "onecluster":
        frame["c"] = 1
        current = spec(covariance="cluster").model_copy(update={"cluster": "c"})
    elif case == "iterations":
        current = spec(max_iterations=1)
    elif case == "separation":
        frame["indicator"] = (frame.index % 7 == 0).astype(float)
        frame.loc[frame.indicator == 1, "y"] = 0
        current = current.model_copy(update={"predictors": ["x", "indicator"]})
    with pytest.raises(AnalysisError) as error:
        fit_streaming(current, Dataset.from_frame(frame), batch_rows=29)
    assert error.value.code == expected
    assert not list(tmp_path.iterdir())


def test_workspace_disk_and_source_fingerprint_refusals(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture()
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        fit_streaming(spec(), Dataset.from_frame(frame), batch_rows=23)
    assert error.value.code == "workspace_limit"
    calls = 0

    def batches():
        nonlocal calls
        calls += 1
        changed = frame.copy()
        if calls > 8:
            changed.loc[170, "y"] += 3
        yield changed

    with pytest.raises(AnalysisError) as error:
        fit_streaming(
            spec(),
            Dataset.from_batches(batches, frame.columns.tolist(), row_count=len(frame)),
            batch_rows=23,
        )
    assert error.value.code == "source_changed"
    import openecon.econometrics.streaming_ppml as module

    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 1})())
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec(), Dataset.from_frame(frame), batch_rows=23)
    assert error.value.code == "fixed_effect_disk_limit"
    assert not list(tmp_path.iterdir())
