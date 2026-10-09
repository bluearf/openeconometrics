"""Failure domains, sample provenance and honest resource contracts."""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


@pytest.fixture
def data():
    rng = np.random.default_rng(731)
    x = rng.normal(size=(80, 4))
    d = x[:, 0] + rng.normal(size=80)
    y = 1.4 * d + x[:, 1] + rng.normal(size=80)
    return pd.DataFrame(
        {"a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "e": x[:, 3], "d": d, "y": y},
        index=np.arange(80)[::-1],
    )


def test_lazy_catalogue_imports_no_torch_and_all_eight_manifest_contracts():
    script = "import sys,openecon as oe; c=oe.capabilities()['estimators']; assert all(k in c for k in ['ridge','lasso','elasticnet','kernelreg','localreg','postdouble','partiallingout','dmlplr']); assert 'torch' not in sys.modules"
    subprocess.run(
        [sys.executable, "-c", script], check=True, env=dict(os.environ), capture_output=True
    )


@pytest.mark.parametrize(
    "name",
    [
        "ridge",
        "lasso",
        "elasticnet",
        "kernelreg",
        "localreg",
        "postdouble",
        "partiallingout",
        "dmlplr",
    ],
)
def test_every_method_rejects_dataset_without_reading_or_materializing(data, monkeypatch, name):
    source = Dataset.from_frame(data)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Dataset must never be iterated or collected for eager regularized estimators"
        )

    monkeypatch.setattr(source, "iter_batches", forbidden)
    inference = name in {"postdouble", "partiallingout", "dmlplr"}
    options = dict(data=source, y="y", x=["a", "b"])
    if inference:
        options["treatment"] = "d"
    with pytest.raises(AnalysisError) as caught:
        getattr(oe, name)(**options)
    assert caught.value.code == "streaming_unsupported"


@pytest.mark.parametrize("name", ["ridge", "localreg", "dmlplr"])
def test_missing_alignment_hashes_and_original_user_data(data, name):
    data.iloc[[2, 13], data.columns.get_loc("a")] = np.nan
    original = data.copy(deep=True)
    options = dict(data=data, y="y", x=["a", "b"], missing="drop", selection="fixed")
    if name == "localreg":
        options["bandwidth"] = 2.0
    else:
        options["penalty"] = 0.1
    if name == "dmlplr":
        options.update(treatment="d", nuisance="ridge", folds=3)
    result = getattr(oe, name)(**options)
    assert result.nobs == 78 and result.dropped_rows == 2 and result.nobs_original == 80
    assert result.sample_positions == [i for i in range(80) if i not in [2, 13]]
    assert {entry["row"] for entry in result.predictions} == set(result.sample_positions)
    assert result.provenance["data_hash"] and result.provenance["sample_hash"]
    pd.testing.assert_frame_equal(original, data)
    if name == "dmlplr":
        folds = result.extra["fold_records"]
        assert set(pos for fold in folds for pos in fold["test_positions"]) == set(
            result.sample_positions
        )
        assert all(not set(fold["train_positions"]) & set(fold["test_positions"]) for fold in folds)
    with pytest.raises(AnalysisError, match="missing"):
        getattr(oe, name)(**{**options, "missing": "raise"})


@pytest.mark.parametrize(
    "options,code",
    [
        ({"selection": "fixed"}, "invalid_penalty"),
        ({"selection": "fixed", "penalty": 0.2, "lambda_path": [0.1, 1.0]}, "invalid_penalty"),
        ({"selection": "cv", "lambda_path": []}, "invalid_penalty"),
        ({"selection": "cv", "lambda_path": [1.0, 1.0]}, "invalid_penalty"),
        ({"selection": "plugin", "penalty": 1.0}, "invalid_penalty"),
        ({"selection": "plugin", "plugin_iterations": 1}, "nonconvergence"),
        (
            {"selection": "fixed", "penalty": 0.001, "max_iterations": 1, "tolerance": 1e-14},
            "nonconvergence",
        ),
        ({"selection": "fixed", "penalty": 0.1, "max_work": 1}, "work_limit"),
    ],
)
def test_lasso_invalid_penalty_and_nonconvergence_are_never_silent(data, options, code):
    with pytest.raises(AnalysisError) as caught:
        oe.lasso(data=data, y="y", x=["a", "b", "c", "e"], **options)
    assert caught.value.code == code


@pytest.mark.parametrize("selection", ["fixed", "cv"])
def test_extreme_finite_outcomes_cannot_persist_nonfinite_penalty_objectives(data, selection):
    extreme = data.copy()
    extreme["y"] *= 1e200
    options = (
        {"selection": selection, "penalty": 0.1}
        if selection == "fixed"
        else {"selection": selection}
    )
    with pytest.raises(AnalysisError) as caught:
        oe.ridge(data=extreme, y="y", x=["a", "b"], **options)
    assert caught.value.code == "numerical_failure"


@pytest.mark.parametrize("bandwidth,query", [(1e-300, 1e100), (1.0, 1e200)])
def test_extreme_finite_local_distances_raise_explicit_numerical_failure(data, bandwidth, query):
    with pytest.raises(AnalysisError) as caught:
        oe.localreg(
            data=data,
            y="y",
            x=["a"],
            selection="fixed",
            bandwidth=bandwidth,
            query=[[query]],
            support="extrapolate",
            kernel="gaussian",
        )
    assert caught.value.code == "numerical_failure"


@pytest.mark.parametrize(
    "maximum,code",
    [
        (1, "work_limit"),
        (True, "invalid_work_limit"),
        (1.5, "invalid_work_limit"),
        (0, "invalid_work_limit"),
    ],
)
def test_saved_linear_prediction_obeys_work_limit_without_silent_coercion(data, maximum, code):
    result = oe.ridge(data=data, y="y", x=["a", "b"], selection="fixed", penalty=0.1)
    with pytest.raises(AnalysisError) as caught:
        oe.regularized_predict(result, data, max_work=maximum)
    assert caught.value.code == code


def test_saved_linear_prediction_rejects_finite_input_multiplication_overflow(data):
    result = oe.ridge(data=data, y="y", x=["a"], selection="fixed", penalty=0.1)
    state = result.extra["penalized_state"]
    assert state["coefficients"][0] > 1.0
    with pytest.raises(AnalysisError) as caught:
        oe.regularized_predict(result, pd.DataFrame({"a": [np.finfo(np.float64).max]}))
    assert caught.value.code == "numerical_failure"


def test_ridge_plugin_explicitly_unsupported_and_no_intercept_constant_kept(data):
    with pytest.raises(AnalysisError) as caught:
        oe.ridge(data=data, y="y", x=["a"], selection="plugin")
    assert caught.value.code == "unsupported_selection"
    constant = data.assign(constant=1.0)
    result = oe.ridge(
        data=constant, y="y", x=["constant", "a"], intercept=False, selection="fixed", penalty=0.1
    )
    assert (
        result.extra["constant"] == 0 and result.extra["predictive_coefficients"]["constant"] != 0
    )


def test_resource_guard_before_design_and_solver(data, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Guard must run before solver allocation")

    monkeypatch.setattr("openecon.econometrics.regularized.prediction.fit_penalized", forbidden)
    big = pd.concat([data] * 800, ignore_index=True)
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError) as caught:
            oe.ridge(data=big, y="y", x=["a", "b"], selection="fixed", penalty=0.1)
    assert caught.value.code == "workspace_limit"
    assert caught.value.resource_plan["estimated_workspace_bytes"] > 1024**2


@pytest.mark.parametrize(
    "name",
    [
        "ridge",
        "lasso",
        "elasticnet",
        "kernelreg",
        "localreg",
        "postdouble",
        "partiallingout",
        "dmlplr",
    ],
)
def test_unsupported_weights_covariance_categories_time_and_options_are_validated(name):
    fields = dict(
        estimator=name,
        outcome="y",
        predictors=["a"],
        columns={"treatment": "d"} if name in {"postdouble", "partiallingout", "dmlplr"} else {},
    )
    for extras in [
        dict(weights="w", weight_type="aweight"),
        dict(categorical=["a"]),
        dict(time="t"),
        dict(options={"typo": 1}),
        dict(covariance="cluster", cluster="id"),
    ]:
        if name == "dmlplr" and extras.get("covariance") == "cluster":
            continue  # Whole-cluster cross-fitting/inference is now an explicit supported extension.
        if name in {"ridge", "lasso", "elasticnet"} and (
            "weights" in extras or "categorical" in extras
        ):
            ModelSpec(
                **{**fields, **extras}
            )  # Empirical weighted/category fixed/CV prediction is now supported.
            continue
        with pytest.raises(ValidationError):
            ModelSpec(**{**fields, **extras})


@pytest.mark.parametrize(
    "options,code",
    [
        ({"selection": "fixed", "bandwidth": 0}, "invalid_bandwidth"),
        ({"selection": "fixed", "bandwidth": [1, 2]}, "invalid_bandwidth"),
        ({"selection": "fixed", "bandwidth": 0.2, "query": [[100.0]]}, "outside_support"),
        ({"selection": "fixed", "bandwidth": 0.2, "query": [[0.0]], "max_work": 1}, "work_limit"),
        ({"selection": "fixed", "bandwidth": 0.2, "query": [0.0]}, "invalid_query"),
        ({"selection": "cv", "bandwidth_path": []}, "invalid_bandwidth"),
        ({"selection": "plugin", "bandwidth": 0.2}, "invalid_bandwidth"),
    ],
)
def test_local_invalid_query_bandwidth_support_and_work(data, options, code):
    with pytest.raises(AnalysisError) as caught:
        oe.localreg(data=data, y="y", x=["a"], **options)
    assert caught.value.code == code


def test_compact_kernel_holes_singular_support_and_explicit_extrapolation():
    data = pd.DataFrame(
        {
            "a": [-2.0, -1.9, -1.8, 1.8, 1.9, 2.0],
            "b": [-2.0, -1.9, -1.8, 1.8, 1.9, 2.0],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    with pytest.raises(AnalysisError) as caught:
        oe.kernelreg(
            data=data,
            y="y",
            x=["a"],
            selection="fixed",
            bandwidth=0.5,
            kernel="uniform",
            query=[[0.0]],
        )
    assert caught.value.code == "empty_local_support"
    with pytest.raises(AnalysisError) as caught:
        oe.localreg(data=data, y="y", x=["a", "b"], selection="fixed", bandwidth=3.0)
    assert caught.value.code == "singular_local_design"
    df = pd.DataFrame({"a": np.linspace(0, 1, 25), "y": 2 + 3 * np.linspace(0, 1, 25)})
    result = oe.localreg(
        data=df,
        y="y",
        x=["a"],
        selection="fixed",
        bandwidth=0.3,
        query=[[1.2]],
        support="extrapolate",
    )
    assert result.extra["query_estimates"][0] == pytest.approx(5.6)
    assert result.extra["query_diagnostics"][0]["outside_bounding_box"] is True


def test_cv_reports_invalid_candidates_without_dropping_rows():
    df = pd.DataFrame({"a": np.linspace(0, 1, 25), "y": np.sin(np.linspace(0, 1, 25))})
    result = oe.localreg(
        data=df, y="y", x=["a"], selection="cv", bandwidth_path=[1e-6, 0.3], kernel="epanechnikov"
    )
    selection = result.extra["smoother_state"]["selection"]
    assert selection["cv_mse"][0] is None and selection["invalid_candidate_reasons"][0] is not None
    assert selection["selected_index"] == 1 and result.nobs == 25


def test_unidentified_treatment_and_bad_roles_never_get_inference(data):
    const = data.assign(d=2.0)
    with pytest.raises(AnalysisError) as caught:
        oe.dmlplr(
            data=const,
            y="y",
            treatment="d",
            x=["a"],
            nuisance="ridge",
            selection="fixed",
            penalty=0.1,
        )
    assert caught.value.code == "unidentified_treatment"
    with pytest.raises(AnalysisError) as caught:
        oe.postdouble(data=data, y="y", treatment="a", x=["a", "b"], selection="fixed", penalty=0.1)
    assert caught.value.code == "invalid_roles"
    with pytest.raises(AnalysisError) as caught:
        oe.postdouble(
            data=data,
            y="y",
            treatment="d",
            x=["a", "b"],
            nuisance="ridge",
            selection="fixed",
            penalty=0.1,
        )
    assert caught.value.code == "unsupported_nuisance"


def test_saved_prediction_rejects_missing_and_dataset(data):
    result = oe.ridge(data=data, y="y", x=["a"], selection="fixed", penalty=0.2)
    with pytest.raises(AnalysisError):
        oe.regularized_predict(result, pd.DataFrame({"a": [np.nan]}))
    with pytest.raises(AnalysisError) as caught:
        oe.regularized_predict(result, Dataset.from_frame(data))
    assert caught.value.code == "streaming_unsupported"
    saved = ResultBundle.model_validate_json(result.model_dump_json())
    assert saved.extra["target"] == "prediction" and saved.coefficients == []


def test_high_dimensional_lasso_and_constant_collinearity_are_predicted_without_ols_ci():
    rng = np.random.default_rng(77)
    x = rng.normal(size=(30, 45))
    y = 2 * x[:, 0] + rng.normal(size=30)
    df = pd.DataFrame(x, columns=[f"x{j}" for j in range(45)])
    df["y"], df["constant"], df["duplicate"] = y, 1.0, df.x1
    result = oe.lasso(
        data=df,
        y="y",
        x=[*[f"x{j}" for j in range(45)], "constant", "duplicate"],
        selection="fixed",
        penalty=0.1,
    )
    assert result.extra["predictive_coefficients"]["constant"] == 0
    assert result.extra["penalized_state"]["path_diagnostics"][0]["kkt_max"] < 1e-7
    assert not result.coefficients


def test_default_and_diagnostic_publication_include_real_saved_prediction_values(data, tmp_path):
    from openecon.latex import _predictive_rows

    result = oe.ridge(data=data, y="y", x=["a", "b"], selection="fixed", penalty=0.2)
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    headers, rows, alignment = _predictive_rows(restored)
    assert headers == [["Predictive term", "Estimate"]]
    assert rows[0] == ["Constant", restored.extra["constant"]]
    assert rows[1:] == [
        [term, coefficient]
        for term, coefficient in restored.extra["predictive_coefficients"].items()
    ]
    for style in ["publication", "diagnostic"]:
        source = str(restored.to_latex(style=style))
        assert "Predictive term" in source and "Estimate" in source
        assert "coefficient inference unavailable" in source and "normal z inference" not in source
        assert "conventional standard errors" not in source and "Std. error" not in source
    saved = tmp_path / "predictive.json"
    saved.write_text(restored.model_dump_json())
    code = "from pathlib import Path; import sys; from openecon.models import ResultBundle; r=ResultBundle.model_validate_json(Path(sys.argv[1]).read_text()); s=str(r.to_latex()); assert 'Predictive term' in s; assert 'torch' not in sys.modules"
    subprocess.run(
        [sys.executable, "-c", code, str(saved)],
        check=True,
        env=dict(os.environ),
        capture_output=True,
    )


def test_mixed_prediction_and_inference_publication_never_silently_drops_saved_predictive_estimates(
    data,
):
    from openecon.latex import regression_table

    prediction = oe.ridge(data=data, y="y", x=["a", "b"], selection="fixed", penalty=0.2)
    inference = oe.ols(data=data, y="y", x=["a", "b"], covariance="HC0")
    with pytest.raises(ValueError, match="predict|target"):
        regression_table([prediction, inference])
    with pytest.raises(ValueError, match="predict|target"):
        regression_table([prediction, prediction])


def test_predictive_capabilities_and_plain_summary_do_not_advertise_coefficient_inference(data):
    for name in ["ridge", "lasso", "elasticnet", "kernelreg", "localreg"]:
        assert "unavailable" in oe.capabilities()["estimators"][name]["inference"]
    result = oe.ridge(data=data, y="y", x=["a"], selection="fixed", penalty=0.2)
    text = result.summary()
    assert "prediction" in text.lower() and "inference unavailable" in text.lower()
    assert "Confidence:" not in text and "P>|stat|" not in text


def test_all_eight_runtime_methods_do_not_import_external_estimator_oracle_dependencies():
    code = """
import builtins
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'scipy', 'statsmodels', 'sklearn', 'linearmodels'}:
        raise AssertionError('external runtime estimator: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
import openecon as oe
import pandas as pd
import torch
g = torch.Generator().manual_seed(179)
x = torch.randn(80, 3, dtype=torch.float64, generator=g)
d = x[:, 0] + torch.randn(80, dtype=torch.float64, generator=g)
y = 1.3*d + x[:, 1] + torch.randn(80, dtype=torch.float64, generator=g)
df = pd.DataFrame(x.numpy(), columns=['a', 'b', 'c']); df['d'] = d; df['y'] = y
for method in ['ridge', 'lasso', 'elasticnet']:
    getattr(oe, method)(data=df, y='y', x=['a', 'b'], selection='fixed', penalty=.1)
for method in ['kernelreg', 'localreg']:
    getattr(oe, method)(data=df, y='y', x=['a', 'b'], selection='fixed', bandwidth=3., query=[[0., 0.]])
for method in ['postdouble', 'partiallingout', 'dmlplr']:
    getattr(oe, method)(data=df, y='y', treatment='d', x=['a', 'b'], selection='plugin')
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=dict(os.environ), capture_output=True
    )
