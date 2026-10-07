"""Independent all-candidate QR, HC leverage and native row-major bootstrap law."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.econometrics.core import make_spec
from openecon.econometrics.replay_sample import ReplaySample
from openecon.econometrics.streaming_linear import _Notes
from openecon.econometrics.streaming_threshold import _Grid, _disk, fit_streaming_threshold
from openecon.econometrics.tsmodels.threshold import fit_threshold
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    # Tiny oracle matrices should not exercise thread-pool startup costs. The
    # library itself does not mutate process-wide CPU settings.
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def data(n=87):
    rng = np.random.default_rng(39218)
    q = rng.normal(size=n)
    x = rng.normal(size=n)
    w = rng.normal(size=n)
    y = np.where(q < 0.2, 1.0 + 2.0 * x, -1.0 - 0.3 * x) + 0.4 * w + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x, "w": w, "q": q, "t": np.arange(n) * 3 + 2**54})


def spec(**options):
    return make_spec(
        "threshold",
        outcome="y",
        predictors=["x", "w"],
        columns={"threshold_var": "q"},
        options={"regions": ["Intercept", "x"], **options},
    )


def source(df, rows=11):
    return Dataset.from_batches(
        lambda: (df.iloc[j : j + rows].copy() for j in range(0, len(df), rows)),
        list(df),
        row_count=len(df),
    )


def prepare(specification, df, sql, path, rows=5):
    sample = ReplaySample(specification, source(df), batch_rows=rows)
    sample.add_design("mean")
    sample.prepare()
    return _Grid(sample, _Notes(specification), sql, path)


def numpy_exact(df, thresholds):
    region = np.searchsorted(thresholds, df.q.to_numpy(), side="left")
    columns = [df.w.to_numpy()]
    for r in range(len(thresholds) + 1):
        mask = region == r
        columns.extend([mask.astype(float), df.x.to_numpy() * mask])
    x = np.array(columns).T
    y = df.y.to_numpy()
    q, r = np.linalg.qr(x, mode="reduced")
    b = np.linalg.solve(r, q.T @ y)
    residual = y - x @ b
    inverse = np.linalg.solve(r, np.eye(len(b)))
    return x, b, inverse @ inverse.T, residual, float(residual @ residual)


@pytest.mark.parametrize("covariance", ["nonrobust", "HC1", "HC2", "HC3", "robust"])
def test_all_observed_candidates_and_full_reported_covariance_independent_numpy_qr(covariance):
    df = data(79)
    specification = spec().model_copy(update={"covariance": covariance})
    actual = fit_streaming_threshold(specification, source(df), batch_rows=4)
    values = np.sort(np.unique(df.q))
    minimum = max(math.ceil(0.1 * len(df)), 3)
    oracles = []
    for threshold in values[:-1]:
        nleft = np.count_nonzero(df.q.to_numpy() <= threshold)
        if minimum <= nleft <= len(df) - minimum:
            x, b, bread, residual, rss = numpy_exact(df, [threshold])
            oracles.append((rss, threshold, b, bread, residual, x))
    rss, threshold, b, bread, residual, x = min(oracles, key=lambda item: (item[0], item[1]))
    assert actual.extra["thresholds"] == [threshold]
    assert actual.extra["candidates"] == len(oracles)
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], b, rtol=3e-12, atol=3e-12)
    df_resid = len(df) - len(b)
    if covariance == "nonrobust":
        cov = bread * rss / df_resid
    else:
        leverage = np.einsum("ni,ij,nj->n", x, bread, x)
        adjusted = residual / (
            np.sqrt(1 - leverage)
            if covariance == "HC2"
            else 1 - leverage
            if covariance == "HC3"
            else 1.0
        )
        scores = x * adjusted[:, None]
        cov = bread @ (scores.T @ scores) @ bread
        if covariance in ["HC1", "robust"]:
            cov *= len(df) / df_resid
    np.testing.assert_allclose(actual.covariance_matrix, cov, rtol=4e-12, atol=3e-12)
    assert actual.metrics["ssr"] == pytest.approx(rss, rel=3e-13)
    restored = ResultBundle.model_validate_json(actual.model_dump_json())
    assert restored.nobs == len(df)
    assert all(
        c.equation == c.term.split(":")[0]
        for c in actual.coefficients
        if c.term.startswith("region")
    )


@pytest.mark.parametrize("thresholds", [1, 2, 3])
def test_sequential_global_refinement_complete_native_parity(thresholds):
    df = data(91)
    if thresholds > 1:
        df["y"] += np.where(df.q > 0.8, 2.0 * df.x, 0.0)
    specification = spec(nthresholds=thresholds)
    actual = fit_streaming_threshold(specification, source(df), batch_rows=5)
    expected = fit_threshold(specification, df)
    assert actual.extra["thresholds"] == expected.extra["thresholds"]
    assert actual.extra["region_sizes"] == expected.extra["region_sizes"]
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=2e-11, atol=2e-11
    )
    assert actual.metrics["ssr"] == pytest.approx(expected.metrics["ssr"], rel=2e-12)
    if thresholds == 1:
        assert (
            actual.extra["threshold_confidence_set"]["low"]
            == expected.extra["threshold_confidence_set"]["low"]
        )
        assert (
            actual.extra["threshold_confidence_set"]["high"]
            == expected.extra["threshold_confidence_set"]["high"]
        )


@pytest.mark.parametrize(
    "n,reps,rows", [(43, 1, 3), (43, 3, 7), (47, 7, 5), (41, 16, 3), (53, 30, 4), (45, 33, 19)]
)
def test_disk_native_normal_row_major_draws_exact_for_odd_n_reps_and_chunk_tail(n, reps, rows):
    df = data(n)
    with _disk() as (sql, path):
        grid = prepare(spec(), df, sql, path, rows)
        residual = torch.tensor(np.linspace(0.2, 1.3, n), dtype=torch.float64)
        # Explicit unit residuals isolate RNG order from any fitted model code.
        grid.gaussian_draws(lambda: (residual[j : j + 3] for j in range(0, n, 3)), reps, 382910)
        expected = (
            torch.randn(
                (n, reps), generator=torch.Generator().manual_seed(382910), dtype=torch.float64
            )
            * residual[:, None]
        )
        from openecon.econometrics.streaming_ucm import _tensor

        actual = torch.stack(
            [
                _tensor(row[0], (reps,))
                for row in sql.execute("SELECT value FROM draws ORDER BY pos")
            ]
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("reps", [7, 30])
def test_complete_hansen_bootstrap_same_candidate_grid_statistic_and_seeded_p_value(reps):
    df = data(61)
    specification = spec(bootstrap=reps, seed=32817)
    before = torch.random.get_rng_state().clone()
    actual = fit_streaming_threshold(specification, source(df), batch_rows=5)
    expected = fit_threshold(specification, df)
    actual_test = actual.tests["threshold_effect"]
    expected_test = expected.tests["threshold_effect"]
    assert actual_test["statistic"] == pytest.approx(expected_test["statistic"], rel=2e-12)
    assert actual_test["p_value"] == expected_test["p_value"]
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    assert actual_test["reps"] == reps and actual_test["seed"] == 32817


def test_tied_threshold_actual_parquet_end_and_internal_missing_time_gaps_categories(tmp_path):
    df = data(80)
    df["q"] = np.round(df.q, 1)
    df["g"] = pd.Categorical(np.where(df.x > 0, "b", "a"), categories=["a", "b", "unused"])
    df["y"] += 0.2 * (df.g == "b")
    df.loc[18, "x"] = np.nan
    df = df.sample(frac=1, random_state=93).reset_index(drop=True)
    path = tmp_path / "data.parquet"
    df.to_parquet(path, index=False)
    specification = make_spec(
        "threshold",
        outcome="y",
        predictors=["x", "w", "g"],
        categorical=["g"],
        time="t",
        columns={"threshold_var": "q"},
        missing="drop",
        options={"regions": ["Intercept", "x"]},
    )
    actual = fit_streaming_threshold(specification, scan(path), batch_rows=4)
    expected = fit_threshold(specification, df)
    assert actual.nobs == 79 and actual.dropped_rows == 1
    assert actual.extra["thresholds"] == expected.extra["thresholds"]
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        rtol=2e-11,
        atol=2e-11,
    )
    assert len(actual.provenance["numeric_spool_hash"]) == 64


@pytest.mark.parametrize(
    "fault,code",
    [
        ("qconstant", "constant_threshold_variable"),
        ("yconstant", "constant_outcome"),
        ("regions", "invalid_option"),
        ("trim", "invalid_option"),
        ("trimfew", "insufficient_observations"),
        ("duplicate", "duplicate_time"),
    ],
)
def test_domain_guards_owned_scratch_cleanup(fault, code, tmp_path, monkeypatch):
    df = data(48)
    specification = spec()
    if fault == "qconstant":
        df["q"] = 1.0
    if fault == "yconstant":
        df["y"] = 1.0
    if fault == "regions":
        specification = spec(regions=["missing"])
    if fault == "trim":
        specification = spec(trim=0.6)
    if fault == "trimfew":
        specification = spec(nthresholds=5, trim=0.2)
    if fault == "duplicate":
        df.loc[1, "t"] = df.loc[0, "t"]
        specification = specification.model_copy(update={"time": "t"})
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with pytest.raises(AnalysisError) as caught:
        fit_streaming_threshold(specification, source(df), batch_rows=3)
    assert caught.value.code == code
    assert not list(tmp_path.iterdir())


def test_mandatory_workspace_meta_cpu_and_changed_replay_refusal(tmp_path, monkeypatch):
    df = data(53)
    specification = spec()
    before = torch.random.get_rng_state().clone()
    with torch.device("meta"):
        result = fit_streaming_threshold(specification, source(df), batch_rows=3)
        assert torch.empty(0).device.type == "meta"
    assert result.nobs == 53
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with use_workspace_budget(8), pytest.raises(AnalysisError) as caught:
        fit_streaming_threshold(specification, source(df))
    assert caught.value.code == "workspace_limit"
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        for j in range(0, len(df), 6):
            rows = df.iloc[j : j + 6].copy()
            if calls > 3:
                rows["y"] += 0.04
            yield rows

    with pytest.raises(AnalysisError) as caught:
        fit_streaming_threshold(
            specification, Dataset.from_batches(factory, list(df), row_count=len(df)), batch_rows=3
        )
    assert caught.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


def test_more_than_preview_candidates_complete_path_hull_and_fixed_block_bound():
    df = data(431)
    with _disk() as (sql, path):
        grid = prepare(spec(), df, sql, path, 7)
        position, _, count = grid.best([])
        sql.execute("INSERT INTO first_path SELECT * FROM search")
        actual_count, _, curve = grid.path()
        assert actual_count == count and count > 200
        assert len(curve["ssr"]) == 200
        assert grid.n == 431 and max(len(y) for _, _, y, _, _ in grid.blocks()) <= 7
        assert grid.resource.estimated_bytes <= grid.resource.budget_bytes
        assert position >= grid.minimum and position <= grid.n - grid.minimum


def test_near_exact_valid_fit_uses_native_perfect_fit_precision_not_arbitrary_ssr_cutoff():
    df = data(67)
    df["y"] = (
        np.where(df.q < 0.2, 1.0 + 2.0 * df.x, -1.0 - 0.3 * df.x)
        + 0.4 * df.w
        + 1e-8 * np.random.default_rng(31).normal(size=len(df))
    )
    # Multiple thresholds do not report the one-threshold Hansen LR hull.
    # The full coefficient fit remains meaningful well below 1e-14 SSR/y'y.
    specification = spec(nthresholds=2)
    actual = fit_streaming_threshold(specification, source(df), batch_rows=4)
    _, _, _, _, oracle_ssr = numpy_exact(df, actual.extra["thresholds"])
    assert 0.0 < actual.metrics["ssr"] < 1e-14 * float(df.y @ df.y)
    assert actual.metrics["ssr"] == pytest.approx(oracle_ssr, rel=4e-7)


@pytest.mark.parametrize("replay", [False, True])
def test_hansen_lr_subtraction_precision_is_explicit_instead_of_empty_or_fabricated_hull(replay):
    df = data(67)
    df["y"] = (
        np.where(df.q < 0.2, 1.0 + 2.0 * df.x, -1.0 - 0.3 * df.x)
        + 0.4 * df.w
        + 1e-8 * np.random.default_rng(31).normal(size=len(df))
    )
    with pytest.raises(AnalysisError) as caught:
        if replay:
            fit_streaming_threshold(spec(), source(df), batch_rows=4)
        else:
            fit_threshold(spec(), df)
    assert caught.value.code == "threshold_precision"


def test_common_coefficients_count_toward_total_degrees_of_freedom_before_search(tmp_path, monkeypatch):
    df = data(16)
    for j in range(11):
        df[f"a{j}"] = np.random.default_rng(381 + j).normal(size=len(df))
    specification = make_spec(
        "threshold",
        outcome="y",
        predictors=["x", "w", *[f"a{j}" for j in range(11)]],
        columns={"threshold_var": "q"},
        options={"regions": ["Intercept", "x"]},
    )
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with pytest.raises(AnalysisError) as caught:
        fit_streaming_threshold(specification, source(df), batch_rows=3)
    assert caught.value.code == "insufficient_observations"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("bootstrap", [False, True])
def test_insufficient_disk_is_refused_before_raw_or_bootstrap_spill_and_cleans_scratch(
    bootstrap, tmp_path, monkeypatch
):
    import openecon.econometrics.streaming_threshold as implementation

    calls = 0

    def free_space(path):
        nonlocal calls
        calls += 1
        # Permit initial records in the bootstrap case, then refuse its draws.
        return SimpleNamespace(free=10**12 if bootstrap and calls == 1 else 0)

    monkeypatch.setattr(implementation.shutil, "disk_usage", free_space)
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    specification = spec(bootstrap=7 if bootstrap else 0)
    with pytest.raises(AnalysisError) as caught:
        fit_streaming_threshold(specification, source(data(53)), batch_rows=3)
    assert caught.value.code == "threshold_spill_failed"
    assert calls == (2 if bootstrap else 1)
    assert not list(tmp_path.iterdir())
