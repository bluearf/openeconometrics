"""Full-source TWFE replay against native fits and independent dummy designs."""

import json
import builtins

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import registry
from openecon.econometrics.streaming_did import fit_streaming
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget
from test_econ_teffects_did import make_panel


def spec(name, covariance="robust", **changes):
    arguments = {
        "estimator": name,
        "outcome": "y",
        "predictors": ["x"],
        "panel": "g",
        "time": "t",
        "columns": {"treatment": "d"} if name == "didregress" else {"treatment_time": "first"},
        "covariance": covariance,
    }
    return ModelSpec(**{**arguments, **changes})


@pytest.mark.parametrize("name", ["didregress", "eventstudy"])
@pytest.mark.parametrize("kind", ["nonrobust", "HC1", "robust", "cluster", "two-way"])
@pytest.mark.parametrize("weight", [None, "aweight", "fweight", "pweight"])
def test_global_weighted_projection_and_covariance_match_native(name, kind, weight):
    if weight == "pweight" and kind == "nonrobust":
        pytest.skip("pweights require robust covariance in the public linear contract")
    frame = make_panel(seed=43, groups=32, repeated=2, staggered=name == "eventstudy")
    covariance = "cluster" if kind == "two-way" else kind
    cluster = ["state", "t"] if kind == "two-way" else "state" if kind == "cluster" else None
    column = "f" if weight == "fweight" else "w" if weight else None
    current = spec(name, covariance, cluster=cluster, weights=column, weight_type=weight)
    expected = registry.load_entry(registry.get(name))(current, frame)
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=47)
    assert [c.term for c in actual.coefficients] == [c.term for c in expected.coefficients]
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        rtol=1e-8,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=2e-7, atol=2e-8
    )
    for key in ["r_squared", "r_squared_within", "rmse", "df_resid", "n_groups", "n_periods"]:
        assert actual.metrics[key] == pytest.approx(expected.metrics[key], rel=2e-7, abs=2e-8)
    for key, test in expected.tests.items():
        for field in ["statistic", "p_value", "df", "df2"]:
            if test.get(field) is not None:
                assert actual.tests[key][field] == pytest.approx(test[field], rel=2e-6, abs=2e-8)
    assert (
        actual.extra["absorbed_degrees_of_freedom"] == expected.extra["absorbed_degrees_of_freedom"]
    )
    assert (actual.nobs, actual.nobs_original, actual.dropped_rows) == (
        expected.nobs,
        expected.nobs_original,
        expected.dropped_rows,
    )
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= 47
    assert actual.provenance["singletons_dropped"] == 0
    assert actual.sample_positions == [] and len(actual.predictions) <= 400
    restored = ResultBundle.model_validate_json(actual.model_dump_json())
    assert restored.coefficients == actual.coefficients
    assert r"\begin{tabular}" in restored.to_latex()
    json.dumps(actual.model_dump(), allow_nan=False)


def test_multiway_nesting_and_constant_degree_of_freedom_independent_oracle():
    from openecon.engines.covariance import nearest_psd

    frame = make_panel(seed=185, groups=32, repeated=3)
    current = spec("didregress", "cluster", cluster=["t", "g"])
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=59)
    y, d, x = frame.y.to_numpy(), frame.d.to_numpy(), frame.x.to_numpy()
    groups = pd.get_dummies(frame.g, dtype=float).to_numpy()
    time = pd.get_dummies(frame.t, dtype=float).to_numpy()[:, 1:]
    fixed = np.column_stack((groups, time))
    values = np.column_stack((y, d, x))
    within = values - fixed @ np.linalg.lstsq(fixed, values, rcond=None)[0]
    yd, xd = within[:, 0], within[:, 1:]
    beta = np.linalg.lstsq(xd, yd, rcond=None)[0]
    residual = yd - xd @ beta
    bread = np.linalg.inv(xd.T @ xd)
    score = xd * residual[:, None]

    def cluster_meat(labels):
        _, codes = np.unique(labels, return_inverse=True)
        totals = np.zeros((codes.max() + 1, 2))
        np.add.at(totals, codes, score)
        return totals.T @ totals

    mixed = frame.g.to_numpy() * 10 + (frame.t.to_numpy() - 2000)
    middle = (
        cluster_meat(frame.t.to_numpy()) + cluster_meat(frame.g.to_numpy()) - cluster_meat(mixed)
    )
    # Both fixed-effect dimensions nest in one of the declared clusters;
    # only the absorbed constant counts in K in addition to the two slopes.
    k, n, count = 3, len(frame), frame.t.nunique()
    canonical, _ = nearest_psd(torch.tensor(middle, dtype=torch.float64))
    expected = bread @ canonical.numpy() @ bread * ((n - 1) / (n - k)) * (count / (count - 1))
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients], beta, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(actual.covariance_matrix, expected, rtol=1e-7, atol=1e-10)
    assert actual.extra["absorbed_degrees_of_freedom"] == 1
    assert actual.metrics["df_resid"] == n - k
    dense = registry.load_entry(registry.get("didregress"))(current, frame)
    np.testing.assert_allclose(dense.covariance_matrix, expected, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("name", ["didregress", "eventstudy"])
def test_categories_missing_rows_singleton_cells_and_permutation(name):
    frame = make_panel(seed=993, groups=33, repeated=2, staggered=True)
    frame["category"] = pd.Categorical(np.arange(len(frame)) % 3, categories=[0, 1, 2, 3])
    frame.loc[3, "x"] = np.nan
    frame.loc[11, "f"] = 0.0
    # One group-period cell has only one physical observation; it is kept.
    frame = frame.drop(index=1).reset_index(drop=True)
    current = spec(
        name,
        "robust",
        predictors=["x", "category"],
        categorical=["category"],
        weights="f",
        weight_type="fweight",
        missing="drop",
    )
    expected = registry.load_entry(registry.get(name))(current, frame)
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=31)
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        rtol=1e-8,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=2e-7, atol=2e-8
    )
    permuted = frame.sample(frac=1, random_state=592).reset_index(drop=True)
    other = fit_streaming(current, Dataset.from_frame(permuted), batch_rows=73)
    np.testing.assert_allclose(
        other.covariance_matrix, actual.covariance_matrix, rtol=2e-7, atol=2e-8
    )
    assert actual.dropped_rows == 2
    assert actual.provenance["singletons_dropped"] == 0
    assert actual.provenance["categorical_encoding"]["category"]["levels"] == [0, 1, 2, 3]


@pytest.mark.parametrize("name", ["didregress", "eventstudy"])
def test_unsorted_datetime_and_bounded_reader_block_without_external_estimator(name, monkeypatch):
    frame = make_panel(seed=321, groups=32, staggered=name == "eventstudy")
    # Dense datetime periods are sorted global ranks, also for an unsorted source.
    if name == "eventstudy":
        frame["first"] = frame["first"] - 2000
    frame["t"] = pd.to_datetime(frame.t.astype(str) + "-01-01")
    frame = frame.sample(frac=1, random_state=398).reset_index(drop=True)
    current = spec(name)
    expected = registry.load_entry(registry.get(name))(current, frame)

    def chunks():
        for start in range(0, len(frame), 29):
            yield frame.iloc[start : start + 29].copy()

    source = Dataset.from_batches(chunks, frame.columns.tolist(), row_count=len(frame))
    original = builtins.__import__

    def guarded(module, *args, **kwargs):
        if module.split(".")[0] in {"statsmodels", "linearmodels", "sklearn", "scipy"}:
            raise AssertionError("External estimator dependency requested")
        return original(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with torch.device("meta"):
        actual = fit_streaming(current, source, batch_rows=19)
        assert torch.empty(0).device.type == "meta"
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=2e-7, atol=2e-8
    )
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= 19


@pytest.mark.parametrize("name", ["didregress", "eventstudy"])
def test_workspace_and_source_mutation_refuse_and_scratch_is_cleaned(name, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = make_panel(seed=890, groups=32)
    current = spec(name)
    with use_workspace_budget(1), pytest.raises(Exception) as error:
        fit_streaming(current, Dataset.from_frame(frame), batch_rows=23)
    assert "budget" in str(error.value).lower()
    calls = 0

    def chunks():
        nonlocal calls
        calls += 1
        copy = frame.copy()
        if calls >= 9:
            copy.loc[23, "y"] += 1
        yield copy

    with pytest.raises(AnalysisError) as error:
        fit_streaming(
            current,
            Dataset.from_batches(chunks, frame.columns.tolist(), row_count=len(frame)),
            batch_rows=23,
        )
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "failure", ["noninteger", "nonconstant", "noever", "window", "integeroverflow"]
)
def test_event_time_and_design_domains_are_checked_over_all_rows(failure):
    frame = make_panel(seed=420, groups=32)
    current = spec("eventstudy")
    if failure == "noninteger":
        frame.loc[180, "first"] = 2004.5
    elif failure == "nonconstant":
        frame.loc[180, "first"] = 2003
    elif failure == "noever":
        frame["first"] = np.nan
    elif failure == "window":
        current = current.model_copy(update={"options": {"leads": 0, "lags": 1, "reference": -1}})
    else:
        frame["t"] = frame.t.astype("uint64") + 2**63
    with pytest.raises(AnalysisError) as error:
        fit_streaming(current, Dataset.from_frame(frame), batch_rows=23)
    assert error.value.code in {
        "invalid_time",
        "invalid_treatment",
        "invalid_spec",
        "no_within_variation",
    }


def test_integer_period_identity_above_float64_roundtrip_is_preserved():
    frame = make_panel(seed=298, groups=32, staggered=True)
    current = spec('eventstudy', options={'leads':3,'lags':2})
    expected = registry.load_entry(registry.get('eventstudy'))(current, frame)
    shifted = frame.copy()
    shift = 2**53+100
    shifted['t'] = shifted.t.astype('int64')+shift
    shifted['first'] = shifted['first'].astype('Int64')+shift
    actual = fit_streaming(current, Dataset.from_frame(shifted), batch_rows=31)
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], [c.estimate for c in expected.coefficients], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(actual.covariance_matrix, expected.covariance_matrix, atol=1e-8, rtol=1e-7)
    assert actual.extra['cohorts'] == [2004+shift,2006+shift]


def test_optional_granger_width_guard_does_not_discard_global_did_fit():
    frame = make_panel(seed=926, groups=8, periods=390, start=2385)
    actual = fit_streaming(spec('didregress'), Dataset.from_frame(frame), batch_rows=127)
    assert actual.coefficients[0].term == 'ATET:r1vs0.d'
    assert np.isfinite(actual.coefficients[0].estimate)
    assert actual.nobs == len(frame)
    assert 'parallel_trends' in actual.tests
    assert 'granger' not in actual.tests
    assert 'bounded' in actual.extra['granger_note']
    assert 'sampled substitute' in actual.extra['granger_note']


def test_nonabsorbing_treatment_records_inapplicable_adoption_tests():
    frame = make_panel(seed=554, groups=32)
    frame.loc[(frame.g < 5)&(frame.t == 2006),'d'] = 0
    current = spec('didregress')
    expected = registry.load_entry(registry.get('didregress'))(current,frame)
    actual = fit_streaming(current,Dataset.from_frame(frame),batch_rows=31)
    np.testing.assert_allclose(actual.covariance_matrix,expected.covariance_matrix,rtol=1e-7,atol=1e-9)
    assert actual.extra['adoption'] == 'not absorbing'
    assert 'parallel_trends' not in actual.tests and 'granger' not in actual.tests
