"""Independent exhaustive split-regression and Gaussian null oracles."""

from itertools import combinations
import json

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def exhaustive(y, x, m, h):
    """Independent original-units lstsq for every admissible partition, no DP."""
    n = len(y)
    models = []
    for k in range(m + 1):
        best = None
        for points in combinations(range(h, n - h + 1), k):
            bounds = (0, *points, n)
            if min(np.diff(bounds)) < h:
                continue
            ssr = 0.
            for a, b in zip(bounds[:-1], bounds[1:]):
                if np.linalg.matrix_rank(x[a:b]) != x.shape[1]:
                    ssr = np.inf
                    break
                beta = np.linalg.lstsq(x[a:b], y[a:b], rcond=None)[0]
                ssr += np.square(y[a:b] - x[a:b] @ beta).sum()
            if best is None or ssr < best[0]:
                best = (ssr, list(points))
        models.append(best)
    return models


def fixture(seed=21, n=22, shifted=True):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 4 + 1.3 * x + rng.normal(size=n) * .5
    if shifted:
        y[n // 3:2 * n // 3] += 5 - 3 * x[n // 3:2 * n // 3]
    return pd.DataFrame({"t": np.arange(n), "y": y, "x": x}, index=np.arange(n) // 2)


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("m", [1, 2, 3])
def test_global_optimum_ssr_f_and_bic_against_exhaustive_original_units(intercept, m):
    df = fixture()
    result = oe.bai_perron(df, "y", ["x"], intercept=intercept, max_breaks=m,
                          min_segment=5, trim=.1, replications=19)
    x = np.column_stack([np.ones(len(df)), df.x]) if intercept else df[["x"]].to_numpy()
    oracle = exhaustive(df.y.to_numpy(), x, m, 5)
    expected_ssr = [a[0] for a in oracle]
    np.testing.assert_allclose(result["models"].ssr, expected_ssr, rtol=1e-10)
    assert result["models"].break_indices.tolist() == [a[1] for a in oracle]
    p, n = x.shape[1], len(df)
    for k in range(1, m + 1):
        f = ((expected_ssr[0] - expected_ssr[k]) / (k * p)
             / (expected_ssr[k] / (n - (k + 1) * p)))
        assert result["tests"].statistic.iloc[k - 1] == pytest.approx(f, rel=1e-10)
    assert result["tests"].statistic.iloc[-1] == result["tests"].statistic.iloc[:-1].max()
    bic = [n * np.log(expected_ssr[k] / n) + ((k + 1) * p + k) * np.log(n)
           for k in range(m + 1)]
    np.testing.assert_allclose(result["models"].bic, bic, rtol=1e-10)
    assert result.attrs["selected_breaks"] == np.argmin(bic)
    # Every descriptive coefficient and covariance in original units.
    points = oracle[np.argmin(bic)][1]
    bounds = [0, *points, n]
    sigma2 = expected_ssr[np.argmin(bic)] / (n - len(bounds[:-1]) * p)
    for j, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        beta = np.linalg.lstsq(x[a:b], df.y.to_numpy()[a:b], rcond=None)[0]
        np.testing.assert_allclose(result["coefficients"].estimate.iloc[j*p:(j+1)*p], beta, rtol=1e-10)
        covariance = sigma2 * np.linalg.inv(x[a:b].T @ x[a:b])
        np.testing.assert_allclose(result["covariance"].iloc[j*p:(j+1)*p, j*p:(j+1)*p], covariance, rtol=1e-10)


def test_monte_carlo_p_quantiles_repeated_full_search_against_exhaustive_null():
    df = fixture(n=18)
    m, h, b, seed = 2, 4, 39, 41
    result = oe.bai_perron(df, "y", ["x"], max_breaks=m, min_segment=h,
                          trim=.1, replications=b, seed=seed)
    x = np.column_stack([np.ones(len(df)), df.x])
    generator = torch.Generator().manual_seed(seed)
    simulated = []
    for offset in range(0, b, 16):
        noise = torch.randn((min(16, b-offset), len(df)), dtype=torch.float64, generator=generator).numpy()
        for y in noise:
            fits = exhaustive(y, x, m, h)
            ssr = [v[0] for v in fits]
            stats = [(ssr[0]-ssr[k]) / (k*2) / (ssr[k]/(len(df)-(k+1)*2)) for k in (1, 2)]
            simulated.append([*stats, max(stats)])
    simulated = np.array(simulated)
    expected_count = (simulated >= result["tests"].statistic.to_numpy()).sum(0)
    np.testing.assert_array_equal(result["tests"].exceedances, expected_count)
    np.testing.assert_allclose(result["tests"].p_value, (expected_count+1)/(b+1))
    np.testing.assert_allclose(result["tests"].iloc[:, 4:].to_numpy(),
                               np.quantile(simulated, [.9, .95, .99], axis=0).T, rtol=1e-10)
    assert result.attrs["successful_replications"] == b
    assert result.attrs["failed_replications"] == 0
    assert result.attrs["monte_carlo_denominator"] == b+1


def test_seeded_replay_preserves_global_rng_sample_and_persistable_contract():
    df = fixture().sample(frac=1, random_state=42)
    state = torch.get_rng_state().clone()
    a = oe.bai_perron(df, "y", ["x"], time="t", max_breaks=2, replications=39, seed=2)
    assert torch.equal(state, torch.get_rng_state())
    b = oe.bai_perron(df, "y", ["x"], time="t", max_breaks=2, replications=39, seed=2)
    pd.testing.assert_frame_equal(a["tests"], b["tests"])
    assert a.attrs == b.attrs
    np.testing.assert_array_equal(a["sample"].original_position, np.argsort(df.t.to_numpy()))
    assert a["sample"].original_index.tolist() == df.sort_values("t", kind="stable").index.tolist()
    json.dumps(a.attrs, allow_nan=False)
    assert dict((row.setting, json.loads(row.value_json))
                for row in a["settings"].itertuples()) == a.attrs
    assert len(a["settings"]) <= 50
    assert "\\caption{models}" in a.to_latex()
    assert "p_value" not in a["coefficients"]
    assert "not selective" in a["covariance"].attrs["inference_scope"]
    pd.testing.assert_frame_equal(df, fixture().sample(frac=1, random_state=42))


def test_location_scale_regressor_units_and_omitted_null_nuisance_invariance():
    df = fixture()
    a = oe.bai_perron(df, "y", ["x"], max_breaks=2, replications=39, seed=7)
    changed = df.assign(y=8 + 3*df.x + 100*df.y, x=10000+500*df.x)
    b = oe.bai_perron(changed, "y", ["x"], max_breaks=2, replications=39, seed=7)
    np.testing.assert_allclose(a["tests"].statistic, b["tests"].statistic, rtol=1e-8)
    np.testing.assert_allclose(a["tests"].p_value, b["tests"].p_value)
    assert a["models"].break_indices.tolist() == b["models"].break_indices.tolist()


def test_rank_deficient_segments_have_fixed_design_exclusions():
    df = fixture(n=30)
    df.loc[df.t < 6, "x"] = 0
    result = oe.bai_perron(df, "y", ["x"], max_breaks=2, trim=.1, replications=19)
    assert result.attrs["excluded_rank_deficient_segments"] > 0
    oracle = exhaustive(df.y.to_numpy(), np.column_stack([np.ones(len(df)), df.x]), 2, 4)
    np.testing.assert_allclose(result["models"].ssr, [v[0] for v in oracle], rtol=1e-9)


@pytest.mark.parametrize("options,code", [
    ({"errors": "hac"}, "invalid_option"), ({"errors": "robust"}, "invalid_option"),
    ({"max_breaks": True}, "invalid_option"), ({"max_breaks": 6}, "invalid_option"),
    ({"replications": 18}, "invalid_option"), ({"seed": -1}, "invalid_option"),
    ({"trim": .5}, "invalid_option"), ({"min_segment": 20}, "insufficient_observations"),
])
def test_option_guards(options, code):
    with pytest.raises(AnalysisError) as caught:
        oe.bai_perron(fixture(), "y", ["x"], **options)
    assert caught.value.code == code


@pytest.mark.parametrize("change,code", [
    (lambda d: d.assign(x=d.y), "perfect_fit"),
    (lambda d: d.assign(y=2.), "perfect_fit"),
    (lambda d: d.assign(x=1.), "collinear_regressors"),
    (lambda d: d.assign(t=d.t*2), "time_gaps"),
    (lambda d: d.assign(t=1), "repeated_time_values"),
    (lambda d: d.assign(y=np.nan), "missing_values"),
    (lambda d: d.assign(y=np.inf), "non_finite_values"),
])
def test_sample_and_rank_guards(change, code):
    with pytest.raises(AnalysisError) as caught:
        oe.bai_perron(change(fixture()), "y", ["x"], time="t", max_breaks=2, replications=19)
    assert caught.value.code == code


def test_quadratic_resource_refusal_precedes_segment_allocation(monkeypatch):
    import importlib
    module = importlib.import_module("openecon.econometrics.unitroot.bai_perron")
    def allocation(*args, **kwargs):
        pytest.fail("segment arrays allocated before refusal")
    monkeypatch.setattr(module, "_segments", allocation)
    for n, b in [(1001, 19), (900, 9999), (500, 9999)]:
        with pytest.raises(AnalysisError) as caught:
            oe.bai_perron(fixture(n=n), "y", ["x"], replications=b)
        assert caught.value.code == "resource_limit"


def test_datetime_gap_rejection_and_mean_only_support():
    df = fixture(n=30)
    df["date"] = pd.date_range("2000-01-01", periods=len(df), freq="MS")
    a = oe.bai_perron(df, "y", [], time="date", max_breaks=2, replications=19)
    assert a.attrs["terms"] == ["Intercept"]
    broken = df.copy()
    broken.loc[broken.t == 10, "date"] = pd.Timestamp("2000-11-03")
    with pytest.raises(AnalysisError, match="regular frequency"):
        oe.bai_perron(broken, "y", [], time="date", max_breaks=2, replications=19)


def test_datetime_multiindex_and_nonfinite_index_labels_are_saved():
    df = fixture(n=30)
    for labels in (pd.date_range("2001-01-01", periods=30),
                   pd.MultiIndex.from_arrays([range(30), ["a"]*30]),
                   pd.Index([float("nan"), *range(29)])):
        result = oe.bai_perron(df.set_axis(labels), "y", [], max_breaks=2, replications=19)
        assert result["sample"].original_position.tolist() == list(range(30))
        json.dumps(result["sample"].to_dict("records"), allow_nan=False)


def test_no_admissible_partition_refuses_without_silent_break_count_change():
    df = fixture(n=20)
    df.loc[df.t < 10, "x"] = 0
    df.loc[df.t >= 10, "x"] = 1
    with pytest.raises(AnalysisError) as caught:
        oe.bai_perron(df, "y", ["x"], max_breaks=4, trim=.1, replications=19)
    assert caught.value.code == "singular_subsample"
