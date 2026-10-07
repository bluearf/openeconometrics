"""Control-free replay retains generated-effect units and full-source inference."""

import numpy as np
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import registry
from openecon.econometrics.streaming_did import fit_streaming
from openecon.models import ModelSpec, ResultBundle
from test_econ_teffects_did import make_panel


def _spec(name, covariance="robust", **changes):
    return ModelSpec(
        **{
            "estimator": name,
            "outcome": "y",
            "predictors": [],
            "panel": "g",
            "time": "t",
            "columns": {"treatment": "d"}
            if name == "didregress"
            else {"treatment_time": "first"},
            "covariance": covariance,
            **changes,
        }
    )


def _source(frame, rows=29):
    def chunks():
        for start in range(0, len(frame), rows):
            yield frame.iloc[start : start + rows].copy()

    return Dataset.from_batches(chunks, frame.columns.tolist(), row_count=len(frame))


def _assert_native_parity(actual, expected):
    assert [(c.term, c.equation) for c in actual.coefficients] == [
        (c.term, c.equation) for c in expected.coefficients
    ]
    fields = ["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"]
    np.testing.assert_allclose(
        [[getattr(c, field) for field in fields] for c in actual.coefficients],
        [[getattr(c, field) for field in fields] for c in expected.coefficients],
        rtol=2e-7,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=2e-7, atol=2e-8
    )
    for key, value in expected.metrics.items():
        if value is not None:
            assert actual.metrics[key] == pytest.approx(value, rel=2e-7, abs=2e-8)
    assert actual.tests.keys() == expected.tests.keys()
    for key, test in expected.tests.items():
        for field in ["statistic", "p_value", "df", "df2"]:
            if test.get(field) is not None:
                assert actual.tests[key][field] == pytest.approx(
                    test[field], rel=2e-6, abs=2e-8
                )
    for field in ["df_resid", "df_inference", "cluster_count", "k_small_sample"]:
        if field in expected.inference:
            assert actual.inference[field] == expected.inference[field]
    assert [entry["column"] for entry in actual.extra["absorbed"]] == ["g", "t"]
    assert [entry["levels"] for entry in actual.extra["absorbed"]] == [
        expected.metrics["n_groups"],
        expected.metrics["n_periods"],
    ]
    assert (
        actual.extra["absorbed_degrees_of_freedom"]
        == expected.extra["absorbed_degrees_of_freedom"]
    )
    assert (actual.nobs, actual.nobs_original, actual.dropped_rows) == (
        expected.nobs,
        expected.nobs_original,
        expected.dropped_rows,
    )


@pytest.mark.parametrize("name", ["didregress", "eventstudy"])
@pytest.mark.parametrize("kind", ["nonrobust", "HC1", "robust", "two-way"])
@pytest.mark.parametrize("weight", [None, "fweight"])
def test_control_free_multibatch_fit_and_all_native_inference(name, kind, weight):
    frame = make_panel(seed=814, groups=12, staggered=name == "eventstudy")
    # Unrequested controls must neither create scale geometry nor exclude rows.
    frame["x"] = np.nan
    frame.loc[19, "y"] = np.nan
    current = _spec(
        name,
        "cluster" if kind == "two-way" else kind,
        cluster=["t", "g"] if kind == "two-way" else None,
        weights="f" if weight else None,
        weight_type=weight,
        missing="drop",
    )
    expected = registry.load_entry(registry.get(name))(current, frame)
    actual = fit_streaming(current, _source(frame), batch_rows=17)
    _assert_native_parity(actual, expected)
    assert actual.dropped_rows == 1
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= 17
    assert actual.provenance["sample_position_count"] == len(frame) - 1
    assert actual.provenance["design_terms"] == [c.term for c in actual.coefficients]
    assert actual.provenance["singletons_dropped"] == 0
    assert actual.sample_positions == [] and len(actual.predictions) <= 400
    assert "x" not in actual.provenance["omitted_terms"]
    restored = ResultBundle.model_validate_json(actual.model_dump_json())
    assert restored.coefficients == actual.coefficients
    assert r"\begin{tabular}" in restored.to_latex()


@pytest.mark.parametrize("kind", ["nonrobust", "robust"])
def test_control_free_did_matches_independent_dummy_and_covariance_oracle(kind):
    frame = make_panel(seed=503, groups=12)
    group = np.eye(frame.g.nunique())[frame.g.to_numpy()]
    period = np.eye(frame.t.nunique())[frame.t.to_numpy() - 2000][:, 1:]
    fixed = np.column_stack((group, period))
    values = frame[["d", "y"]].to_numpy()
    within = values - fixed @ np.linalg.lstsq(fixed, values, rcond=None)[0]
    x, y = within[:, :1], within[:, 1]
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    bread = np.linalg.inv(x.T @ x)
    n, groups, periods = len(frame), frame.g.nunique(), frame.t.nunique()
    if kind == "nonrobust":
        k = groups + periods
        covariance = bread * (residual @ residual) / (n - k)
    else:
        k = 1 + periods
        scores = np.zeros((groups, 1))
        np.add.at(scores, frame.g.to_numpy(), x * residual[:, None])
        covariance = bread @ (scores.T @ scores) @ bread
        covariance *= groups / (groups - 1) * (n - 1) / (n - k)
    actual = fit_streaming(_spec("didregress", kind), _source(frame), batch_rows=17)
    np.testing.assert_allclose([c.estimate for c in actual.coefficients], beta, atol=1e-10)
    np.testing.assert_allclose(actual.covariance_matrix, covariance, rtol=1e-8, atol=1e-10)
    assert actual.metrics["df_resid"] == n - k
    assert actual.extra["absorbed_degrees_of_freedom"] == k - 1


@pytest.mark.parametrize("name", ["didregress", "eventstudy"])
def test_control_free_effect_units_are_preserved(name):
    frame = make_panel(seed=931, groups=12, staggered=name == "eventstudy")
    current = _spec(name)
    baseline = fit_streaming(current, _source(frame), batch_rows=17)
    scale = 1e6
    frame["y"] *= scale
    actual = fit_streaming(current, _source(frame, rows=31), batch_rows=23)
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        np.array([c.estimate for c in baseline.coefficients]) * scale,
        rtol=1e-8,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix,
        np.array(baseline.covariance_matrix) * scale**2,
        rtol=2e-7,
        atol=1e-3,
    )
    assert actual.metrics["df_resid"] == baseline.metrics["df_resid"]
    assert actual.extra["absorbed_degrees_of_freedom"] == baseline.extra["absorbed_degrees_of_freedom"]


@pytest.mark.parametrize("name", ["didregress", "eventstudy"])
def test_control_free_source_mutation_still_refused_and_scratch_cleaned(name, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = make_panel(seed=374, groups=12, staggered=name == "eventstudy")
    calls = 0

    def chunks():
        nonlocal calls
        calls += 1
        copy = frame.copy()
        if calls >= 9:
            copy.loc[23, "y"] += 1
        for start in range(0, len(copy), 29):
            yield copy.iloc[start : start + 29].copy()

    source = Dataset.from_batches(chunks, frame.columns.tolist(), row_count=len(frame))
    with pytest.raises(AnalysisError) as error:
        fit_streaming(_spec(name), source, batch_rows=17)
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())
