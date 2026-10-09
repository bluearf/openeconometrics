"""Independent complete-sample Gaussian algebra, basis and replay contracts."""

from itertools import combinations_with_replacement
import json

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import BSpline
from scipy.stats import f as f_distribution, t as t_distribution
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.streaming_smoothing import fit_streaming
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget

METHODS = ("bspline_regress", "rcs_regress", "fp_regress", "mfp_regress")
POWERS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)


@pytest.fixture(autouse=True, scope="module")
def small_native_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def fixture(n=193):
    x = np.linspace(0.25, 4.75, n)
    z = np.cos(np.arange(n) * 0.47)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "y": 2 + 1.3 * np.log(x) + 0.2 * z + 0.08 * np.sin(np.arange(n) * 0.61),
            "w": 1 + np.arange(n) % 7,
        }
    )


def batches(frame, size=23):
    return Dataset.from_batches(
        lambda: (frame.iloc[j : j + size].copy() for j in range(0, len(frame), size)),
        list(frame.columns),
    )


def specification(method, **options):
    return ModelSpec(
        estimator=method,
        outcome="y",
        predictors=["x", "z"] if "fp" in method else ["x"],
        options=options,
    )


def oracle_basis(frame, state):
    # No tested transformation routine is called by this independent oracle.
    parts = [np.ones((len(frame), 1))]
    for transform in state["transforms"]:
        x = frame[transform["column"]].to_numpy()
        if transform["kind"] == "bs":
            degree = transform["degree"]
            lo, hi = transform["boundary"]
            knots = [lo] * (degree + 1) + transform["knots"] + [hi] * (degree + 1)
            parts.append(BSpline.design_matrix(x, knots, degree).toarray()[:, 1:])
        elif transform["kind"] == "rcs":
            knots = np.asarray(transform["knots"])
            u = (x - knots[0]) / (knots[-1] - knots[0])
            t = (knots - knots[0]) / (knots[-1] - knots[0])
            nonlinear = [
                np.maximum(u - a, 0) ** 3
                - np.maximum(u - t[-2], 0) ** 3 * (t[-1] - a) / (t[-1] - t[-2])
                + np.maximum(u - t[-1], 0) ** 3 * (t[-2] - a) / (t[-1] - t[-2])
                for a in t[:-2]
            ]
            parts.append(np.column_stack([u, *nonlinear]))
        elif transform["kind"] == "fp":
            z = x / transform["scale"]
            columns = []
            for j, power in enumerate(transform["powers"]):
                value = np.log(z) if power == 0 else z**power
                columns.append(
                    value * np.log(z) if j and power == transform["powers"][j - 1] else value
                )
            if columns:
                parts.append(np.column_stack(columns))
        else:
            parts.append(x[:, None])
    return np.column_stack(parts)


def oracle_fit(x, y):
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    rss = np.sum((y - x @ b) ** 2)
    df = len(y) - len(b)
    covariance = rss / df * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(covariance))
    p = 2 * t_distribution.sf(np.abs(b / se), df)
    critical = t_distribution.ppf(0.975, df)
    return b, covariance, rss, df, p, b - critical * se, b + critical * se


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("size", [1, 29, 500])
def test_full_inference_agrees_with_independent_algebra_and_resident(method, size):
    frame = fixture(73)
    options = {"powers": [0.0, 0.0], "scale": 2.0} if method == "fp_regress" else {}
    spec = specification(method, **options)
    resident = oe.fit(spec, data=frame)
    result = fit_streaming(spec, batches(frame, size), batch_rows=min(size, 500))
    x = oracle_basis(frame, result.extra["smoothing_state"])
    b, covariance, rss, df, p, low, high = oracle_fit(x, frame.y.to_numpy())
    actual = pd.DataFrame([c.model_dump() for c in result.coefficients])
    np.testing.assert_allclose(actual.estimate, b, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(result.covariance_matrix, covariance, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(actual.p_value, p, atol=2e-9)
    np.testing.assert_allclose(actual.ci_low, low, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(actual.ci_high, high, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(
        result.covariance_matrix, resident.covariance_matrix, rtol=2e-8, atol=2e-9
    )
    assert result.metrics["rss"] == pytest.approx(rss, abs=2e-10)
    assert result.inference["df_inference"] == df
    assert result.nobs == result.nobs_original == len(frame)
    assert result.sample_positions == []
    assert len(result.predictions) <= 400
    assert result.provenance["streaming"]["maximum_batch_rows"] <= min(size, 500)
    assert result.provenance["solver_diagnostics"]["full_source_collected"] is False
    json.dumps(result.model_dump(mode="json"), allow_nan=False)


def test_mfp_all45_objectives_and_closed_tests_have_independent_references():
    frame = fixture()
    result = fit_streaming(specification("mfp_regress"), batches(frame, 13), batch_rows=11)
    state = result.extra["smoothing_state"]
    candidates = [
        [],
        *[[p] for p in POWERS],
        *[list(p) for p in combinations_with_replacement(POWERS, 2)],
    ]
    references = []
    for powers in candidates:
        fake = {
            "transforms": [
                {"kind": "fp", "column": "x", "powers": powers, "scale": 1.0},
                {"kind": "linear", "column": "z"},
            ]
        }
        reference = oracle_fit(oracle_basis(frame, fake), frame.y.to_numpy())
        references.append(reference)
    np.testing.assert_allclose(
        [c["rss"] for c in state["candidates"]], [r[2] for r in references], rtol=2e-9, atol=2e-10
    )
    for test in state["closed_tests"]:
        full, reduced = references[test["full_candidate"]], references[test["reduced_candidate"]]
        f = max(0.0, (reduced[2] - full[2]) / test["df_num"]) / (full[2] / full[3])
        assert test["statistic"] == pytest.approx(f, rel=1e-8)
        assert test["p_value"] == pytest.approx(
            f_distribution.sf(f, test["df_num"], full[3]), abs=1e-8
        )
    assert "selection uncertainty" in result.inference["conditioning"]


@pytest.mark.parametrize("method", METHODS)
def test_missing_sample_knots_persistence_prediction_and_weighted_average(method, monkeypatch):
    frame = fixture(641)
    frame.loc[[1, 103, 622], "y"] = np.nan
    frame.loc[[32, 315], "x"] = np.nan
    spec = specification(method)
    spec.missing = "drop"
    monkeypatch.setattr(
        Dataset, "head", lambda *a, **k: pytest.fail("Dataset head was materialized")
    )
    from openecon.econometrics.smoothing import common

    monkeypatch.setattr(
        common, "prepare", lambda *a, **k: pytest.fail("Resident smoothing fit was called")
    )
    result = fit_streaming(spec, batches(frame, 31), batch_rows=17)
    restored = ResultBundle.model_validate_json(
        json.dumps(result.model_dump(mode="json"), allow_nan=False)
    )
    complete = frame.dropna(subset=["y", "x", "z"])
    assert restored.nobs == len(complete) and restored.dropped_rows == 5
    assert restored.provenance["sample_position_count"] == len(complete)
    assert restored.extra["smoothing_state"]["sample_positions_scope"].startswith("first400")
    assert len(restored.extra["smoothing_state"]["sample_positions"]) == 400
    if method in {"bspline_regress", "rcs_regress"}:
        transform = restored.extra["smoothing_state"]["transforms"][0]
        q = (
            np.linspace(0, 1, 5)[1:-1]
            if method == "bspline_regress"
            else np.linspace(0.05, 0.95, 5)
        )
        np.testing.assert_allclose(
            transform["knots"], np.quantile(complete.x, q), rtol=0, atol=2e-15
        )
        assert restored.provenance["solver_diagnostics"]["order_statistics"]["disk_bytes"] > 4096
    query = fixture(133)
    query.loc[[0, 79], "x"] = np.nan
    resident_prediction = oe.smoothing_predict(restored, data=query, interval=True, missing="drop")
    dataset_prediction = oe.smoothing_predict(
        restored, data=batches(query, 7), interval=True, missing="drop"
    )
    output = pd.concat(list(dataset_prediction.iter_batches()), ignore_index=True)
    np.testing.assert_allclose(output, resident_prediction, rtol=1e-10, atol=1e-10)
    assert dataset_prediction._metadata["analysis"]["streaming"]["dropped_rows"] == 2
    expected = oe.smoothing_margins(
        restored,
        data=query,
        variable="x",
        interval=True,
        missing="drop",
        averaging_weights=query.w.tolist(),
    )
    actual = oe.smoothing_margins(
        restored,
        data=batches(query, 7),
        variable="x",
        interval=True,
        missing="drop",
        averaging_weights="w",
    )
    np.testing.assert_allclose(pd.DataFrame(actual), expected, rtol=1e-10, atol=1e-10)
    assert actual.attrs["retained_rows"] == len(query) - 2
    assert "selection uncertainty" in oe.to_latex(actual)
    assert "selection uncertainty" in oe.to_latex(restored)


@pytest.mark.parametrize("method", ["bspline_regress", "rcs_regress"])
def test_explicit_knots_do_not_create_disk_order_statistics(method):
    frame = fixture()
    options = (
        {"knots": {"x": [1.0, 2.0, 3.0]}, "boundary": {"x": [0.25, 4.75]}}
        if method == "bspline_regress"
        else {"knots": {"x": [0.25, 1.0, 2.0, 3.0, 4.75]}}
    )
    result = fit_streaming(specification(method, **options), batches(frame))
    assert result.provenance["solver_diagnostics"]["order_statistics"] is None


def test_resource_work_domain_rank_and_missing_guards():
    frame = fixture()
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        fit_streaming(specification("bspline_regress"), batches(frame))
    with pytest.raises(AnalysisError) as error:
        fit_streaming(specification("fp_regress", max_work=1), batches(frame))
    assert error.value.code == "work_limit"
    bad = frame.copy()
    bad.loc[40, "x"] = 0
    with pytest.raises(AnalysisError) as error:
        fit_streaming(specification("fp_regress"), batches(bad, 7))
    assert error.value.code == "invalid_fp_domain"
    with pytest.raises(AnalysisError) as error:
        fit_streaming(specification("fp_regress", powers=[1.0, 1.0]), batches(frame.assign(z=1.0)))
    assert error.value.code == "rank_deficient"
    with pytest.raises(AnalysisError) as error:
        fit_streaming(specification("bspline_regress", knots={"x": [1.0, 1.0]}), batches(frame))
    assert error.value.code == "invalid_knots"
    with pytest.raises(AnalysisError) as error:
        fit_streaming(specification("bspline_regress", knots={"x": [10**500]}), batches(frame))
    assert error.value.code == "invalid_knots"
    bad.loc[5, "y"] = np.nan
    with pytest.raises(AnalysisError) as error:
        fit_streaming(specification("rcs_regress"), batches(bad))
    assert error.value.code == "missing_values"


def test_global_replay_hash_detects_source_change_and_scratch_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture()
    count = 0

    def factory():
        nonlocal count
        count += 1
        copy = frame.copy()
        if count > 4:
            copy.loc[90, "x"] += 0.01
        yield copy

    source = Dataset.from_batches(factory, list(frame.columns))
    with pytest.raises(AnalysisError) as error:
        fit_streaming(specification("bspline_regress"), source)
    assert error.value.code == "source_changed"
    assert list(tmp_path.iterdir()) == []


def test_still_unsupported_methods_and_query_options_are_explicit():
    frame = fixture()
    with pytest.raises(AnalysisError) as error:
        oe.gam_gaussian(data=batches(frame), y="y", x=["x"])
    assert error.value.code == "streaming_unsupported"
    result = oe.fp_regress(data=batches(frame), y="y", x=["x"])
    with pytest.raises(AnalysisError) as error:
        oe.smoothing_margins(result, data=batches(frame), averaging_weights=[1.0] * len(frame))
    assert error.value.code == "invalid_weights"
    with pytest.raises(AnalysisError) as error:
        oe.smoothing_predict(result, data=batches(frame), max_work=1)
    assert error.value.code == "work_limit"
    with pytest.raises(AnalysisError) as error:
        oe.fp_derivative(result, data=batches(frame), variable="x")
    assert error.value.code == "streaming_unsupported"
