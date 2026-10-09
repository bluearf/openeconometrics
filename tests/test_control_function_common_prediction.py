"""Independent complete-joint targets, disk replay and unchanged legacy state."""

from copy import deepcopy
import gc
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ResultBundle

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads(
    gzip.decompress(
        (
            ROOT / "docs/evidence/control-functions-eight-2026-10-07/persisted-states.json.gz"
        ).read_bytes()
    )
)["cases"]
IDS = [item["case_id"] for item in CASES]


def legacy_case(identifier):
    return deepcopy(next(item for item in CASES if item["case_id"] == identifier))


def independent(case, frame, parameters=None, *, variable=None, kind="response"):
    saved = case["result"]["extra"]["control_function_state"]
    p = np.r_[saved["gamma"], saved["beta"]] if parameters is None else np.asarray(parameters)
    kz = len(saved["gamma"])

    def design(terms):
        return np.column_stack(
            [
                np.ones(len(frame)) if name == "Intercept" else np.asarray(frame[name], dtype=float)
                for name in terms
            ]
        )

    z, x = design(saved["z_terms"]), design(saved["x_terms"])
    gamma, beta = p[:kz], p[kz:]
    residual = np.asarray(frame[case["inputs"]["endogenous"]]) - z @ gamma
    eta = x @ beta[:-1] + residual * beta[-1]
    if kind in {"xb", "stdp"} and variable is None:
        return eta
    domain = case["kind"]
    if domain == "gaussian":
        mu, first = eta, np.ones(len(frame))
    elif domain in {"logit", "fractional_logit"}:
        mu = 1 / (1 + np.exp(-eta))
        first = mu * (1 - mu)
    elif domain == "probit":
        mu = np.array([math.erfc(-float(v) / math.sqrt(2)) / 2 for v in eta])
        first = np.exp(-(eta**2) / 2) / math.sqrt(2 * math.pi)
    elif domain == "cloglog":
        mu, first = -np.expm1(-np.exp(eta)), np.exp(eta - np.exp(eta))
    else:
        mu = first = np.exp(eta)
    if variable is None:
        return mu
    slope = beta[saved["x_terms"].index(variable)] if variable in saved["x_terms"] else 0.0
    partial = float(variable == case["inputs"]["endogenous"])
    if variable in saved["z_terms"]:
        partial -= gamma[saved["z_terms"].index(variable)]
    slope += beta[-1] * partial
    return np.ones(len(frame)) * slope if kind == "xb" else first * slope


def differenced(function, point):
    columns = []
    for j, value in enumerate(point):
        step = 1e-5 * max(1.0, abs(value))
        plus, minus = point.copy(), point.copy()
        plus[j] += step
        minus[j] -= step
        columns.append((function(plus) - function(minus)) / (2 * step))
    return np.array(columns).T


@pytest.fixture(scope="module")
def models():
    answer = {}
    for case in CASES:
        inputs = case["inputs"]
        frame = pd.DataFrame(inputs["data"])
        options = {key: value for key, value in inputs.items() if key != "data"}
        options["missing"] = "drop"
        answer[case["case_id"]] = (
            getattr(oe, case["result"]["spec"]["estimator"])(data=frame, **options),
            frame,
        )
    return answer


@pytest.mark.parametrize("identifier", IDS)
@pytest.mark.parametrize("kind", ["response", "xb", "stdp", "derivative"])
def test_joint_delta_predictions_independent_legacy_no_refit(identifier, kind, monkeypatch):
    case = legacy_case(identifier)
    result = ResultBundle.model_validate(case["result"])
    frame = pd.DataFrame(case["inputs"]["data"]).iloc[:19]
    frame.index = pd.Index(["same", 1, "same"] * 6 + [None], dtype=object, name="row")
    from openecon.econometrics.control_function import kernels

    monkeypatch.setattr(
        kernels, "fit_joint", lambda *a, **kw: pytest.fail("Saved evaluation refitted")
    )
    variable = "d" if kind == "derivative" else None
    output = oe.predict(
        result, frame, kind=kind, term=variable, interval=None if kind == "stdp" else "mean"
    )
    state = case["result"]["extra"]["control_function_state"]
    p, covariance = np.r_[state["gamma"], state["beta"]], np.asarray(state["joint_covariance"])

    def fun(p):
        return independent(case, frame, p, variable=variable, kind=kind)

    jac = differenced(fun, p)
    se = np.sqrt(np.einsum("nk,kl,nl->n", jac, covariance, jac))
    expected = se if kind == "stdp" else fun(p)
    np.testing.assert_allclose(output.iloc[:, 0], expected, atol=2e-9, rtol=2e-8)
    assert output.index.equals(frame.index)
    if kind != "stdp":
        np.testing.assert_allclose(output.std_error, se, atol=2e-9, rtol=2e-8)
        np.testing.assert_allclose(
            output.ci_low, expected - 1.959963984540054 * se, atol=3e-9, rtol=3e-8
        )
        restored = oe.predict(
            ResultBundle.model_validate_json(result.model_dump_json()),
            frame,
            kind=kind,
            term=variable,
            interval="mean",
        )
        pd.testing.assert_frame_equal(output, restored)
    assert result.model_dump(mode="json") == case["result"]


@pytest.mark.parametrize("identifier", IDS)
@pytest.mark.parametrize("method", ["ame", "mem"])
@pytest.mark.parametrize("variable", ["x", "d", "z1"])
def test_global_conditional_effect_full_parameter_gradient(identifier, method, variable):
    case = legacy_case(identifier)
    result = ResultBundle.model_validate(case["result"])
    frame = pd.DataFrame(case["inputs"]["data"]).iloc[:31]
    output = oe.margins(result, variable, data=frame, method=method, at={"z2": [-0.4, 0.5]})
    state = case["result"]["extra"]["control_function_state"]
    p, covariance = np.r_[state["gamma"], state["beta"]], np.asarray(state["joint_covariance"])
    for i, value in enumerate([-0.4, 0.5]):
        query = frame.copy()
        query["z2"] = value
        if method == "mem":
            query = query.mean(numeric_only=True).to_frame().T

        def fn(p):
            return np.array([independent(case, query, p, variable=variable).mean()])

        gradient = differenced(fn, p).ravel()
        expected = float(fn(p)[0])
        np.testing.assert_allclose(
            output.attrs["delta_gradients"][i], gradient, atol=2e-9, rtol=2e-8
        )
        np.testing.assert_allclose(output.iloc[i]["estimate"], expected, atol=2e-10, rtol=2e-9)
        np.testing.assert_allclose(
            output.iloc[i]["std_error"],
            math.sqrt(gradient @ covariance @ gradient),
            atol=2e-9,
            rtol=2e-8,
        )


def source(frame):
    return Dataset.from_batches(
        lambda: (frame.iloc[i : i + 13] for i in range(0, len(frame), 13)),
        list(frame.columns),
        row_count=len(frame),
    )


@pytest.mark.parametrize("identifier", IDS)
def test_complete_disk_predictions_missing_typed_index_semantic_replay_once(
    identifier, models, tmp_path, monkeypatch
):
    result, frame = models[identifier]
    query = frame.iloc[:75].copy()
    query.index = pd.Index(([1, "1", None, "dup", "dup"] * 15), dtype=object, name="row")
    query.iloc[:17, query.columns.get_loc("z1")] = np.nan
    from openecon.econometrics.postest import control_prediction

    original, calls = control_prediction.validate_state, []

    def validate(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(control_prediction, "validate_state", validate)
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    expected = oe.predict(result, query, interval="mean")
    calls.clear()
    output = oe.predict(result, source(query), interval="mean", batch_rows=7)
    assert len(calls) == 1
    actual = pd.concat(list(output.iter_batches(batch_rows=11)))
    pd.testing.assert_index_equal(actual.index, query.index)
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12, equal_nan=True)
    assert output.metadata["analysis"]["missing_prediction_rows"] == 17
    assert output.metadata["analysis"]["streaming"]["full_source_collected"] is False
    path = Path(output._owned_prediction_output.name)
    del output
    gc.collect()
    assert not path.exists()


@pytest.mark.parametrize("identifier", IDS)
@pytest.mark.parametrize("method", ["ame", "mem"])
def test_streamed_global_effects_not_average_batch_mem(identifier, method, models):
    result, frame = models[identifier]
    query = frame.iloc[:90].copy()
    query.iloc[:13, query.columns.get_loc("x")] = np.nan
    options = dict(variables=["x", "d", "z1"], method=method, at={"d": [-0.3, 0.2]})
    expected = oe.margins(result, data=query, **options)
    output = oe.margins(result, data=source(query), batch_rows=7, **options)
    np.testing.assert_allclose(
        output.select_dtypes("number"), expected.select_dtypes("number"), atol=5e-12, rtol=2e-10
    )
    np.testing.assert_allclose(
        output.attrs["delta_gradients"], expected.attrs["delta_gradients"], atol=5e-12, rtol=2e-10
    )
    assert output.attrs["evaluation_rows"] == 77
    assert output.attrs["streaming"]["maximum_batch_rows"] <= 7


@pytest.mark.parametrize("identifier", IDS[::2])
def test_invalid_state_and_targets_rejected_before_query(identifier):
    case = legacy_case(identifier)
    result = ResultBundle.model_validate(case["result"])
    frame = pd.DataFrame(case["inputs"]["data"]).iloc[:5]
    with pytest.raises(AnalysisError):
        oe.predict(result, frame.drop(columns="z1"))
    with pytest.raises(AnalysisError):
        oe.predict(result, frame, interval="observation")
    with pytest.raises(AnalysisError):
        oe.predict(result, frame, outcome="structural")
    with pytest.raises(AnalysisError):
        oe.predict(result, frame, target="population")
    damaged = result.model_copy(deep=True)
    damaged.extra["control_function_state"]["gamma"][0] += 0.2
    with pytest.raises(AnalysisError):
        oe.predict(damaged, frame)


def test_dataset_mutation_refused_and_scratch_cleaned(models, tmp_path, monkeypatch):
    result, frame = models[IDS[0]]
    count = [0]

    def factory():
        count[0] += 1
        altered = frame.iloc[:50].copy()
        if count[0] > 1:
            altered["x"] += 0.1
        yield altered

    query = Dataset.from_batches(factory, list(frame.columns), row_count=50)
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with pytest.raises(AnalysisError, match="changed"):
        oe.margins(result, "x", data=query, method="mem", at={"d": [0.0, 1.0]})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("identifier", IDS)
@pytest.mark.parametrize("predictors", [[], ["x"]])
def test_no_intercept_roles_full_delta(identifier, predictors, monkeypatch):
    case = legacy_case(identifier).copy()
    frame = pd.DataFrame(case["inputs"]["data"])
    options = {key: value for key, value in case["inputs"].items() if key != "data"}
    options.update(intercept=False, x=predictors)
    fitted = getattr(oe, case["result"]["spec"]["estimator"])(data=frame, **options)
    case["result"] = fitted.model_dump(mode="json")
    state = case["result"]["extra"]["control_function_state"]
    p = np.r_[state["gamma"], state["beta"]]
    cov = np.asarray(state["joint_covariance"])
    query = frame.iloc[:31]
    from openecon.econometrics.control_function import kernels

    monkeypatch.setattr(kernels, "fit_joint", lambda *a, **kw: pytest.fail("Query refitted"))
    output = oe.predict(
        fitted,
        source(query),
        kind="derivative",
        term="z1",
        interval="mean",
        alpha=0.1,
        batch_rows=7,
    )
    actual = pd.concat(output.iter_batches(batch_rows=11))

    def fun(p):
        return independent(case, query, p, variable="z1")

    j = differenced(fun, p)
    se = np.sqrt(np.einsum("nk,kl,nl->n", j, cov, j))
    np.testing.assert_allclose(
        actual.to_numpy(),
        np.column_stack(
            [fun(p), se, fun(p) - 1.6448536269514722 * se, fun(p) + 1.6448536269514722 * se]
        ),
        atol=2e-8,
        rtol=2e-7,
    )
    margins = oe.margins(fitted, "z1", data=source(query), method="ame", alpha=0.1, batch_rows=7)

    def average(p):
        return np.array([fun(p).mean()])

    gradient = differenced(average, p).ravel()
    estimate = float(average(p)[0])
    se = math.sqrt(gradient @ cov @ gradient)
    row = margins.iloc[0]
    z = estimate / se
    np.testing.assert_allclose(
        [row.estimate, row.std_error, row.statistic, row.p_value, row.ci_low, row.ci_high],
        [
            estimate,
            se,
            z,
            math.erfc(abs(z) / math.sqrt(2)),
            estimate - 1.6448536269514722 * se,
            estimate + 1.6448536269514722 * se,
        ],
        atol=2e-8,
        rtol=2e-7,
    )
