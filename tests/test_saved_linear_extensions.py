"""Saved robust, CUE and Gaussian mixed targets against independent algebra."""

import copy
import importlib

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.stats import norm, t
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


def restored(result):
    return ResultBundle.model_validate_json(result.model_dump_json())


@pytest.fixture(scope="module", params=["sreg", "mmreg", "ivcue", "mixedflex"])
def saved(request):
    name = request.param
    rng = np.random.default_rng(6419)
    n = 144
    frame = pd.DataFrame({"x": rng.normal(size=n), "g": np.resize(["a", "b", "c"], n),
                          "z1": rng.normal(size=n), "z2": rng.normal(size=n),
                          "z3": rng.normal(size=n)})
    v = rng.normal(size=(n, 2))
    frame["p1"] = frame.z1 + .3 * frame.z3 + .2 * frame.x + v[:, 0]
    frame["p2"] = frame.z2 + .2 * frame.z3 - .1 * frame.x + v[:, 1]
    frame["y"] = 1 + .4 * frame.x + .6 * (frame.g == "b") - .2 * (frame.g == "c") + rng.normal(size=n)
    if name in {"sreg", "mmreg"}:
        frame.loc[:9, "y"] += 20
        result = getattr(oe, name)(data=frame, y="y", x=["x", "g"], categorical=["g"],
                                   starts=60, seed=73, missing="drop")
    elif name == "ivcue":
        frame["y"] = 1 + .4 * frame.x + .8 * frame.p1 - .3 * frame.p2 + .5 * v[:, 0] + rng.normal(size=n)
        result = oe.ivcue(data=frame, y="y", x=["x"], endog=["p1", "p2"],
                          instruments=["z1", "z2", "z3"], missing="drop")
    else:
        # Positive, identified nested components, with reused lower-level names.
        rng = np.random.default_rng(18)
        top = np.repeat(np.arange(6), 16)
        mid = np.tile(np.repeat(np.arange(2), 8), 6)
        low = np.tile(np.repeat(np.arange(2), 4), 12)
        x = rng.normal(size=len(top))
        y = (2 + .4 * x + rng.normal(size=6)[top]
             + rng.normal(size=12)[top * 2 + mid]
             + rng.normal(size=24)[top * 4 + mid * 2 + low]
             + rng.normal(size=len(top)) * .4)
        frame = pd.DataFrame(dict(y=y, x=x, top=top, mid=mid, low=low))
        result = oe.mixedflex(data=frame, y="y", x=["x"], group=["top", "mid", "low"],
                              grouping="nested", missing="drop")
    return name, frame, restored(result)


def query():
    return pd.DataFrame({"x": [-1.2, .3, 1.7, -.8], "g": ["c", "a", "b", "a"],
                         "p1": [.4, -1., .8, -.2], "p2": [-.9, .3, 1.3, .7]},
                        index=pd.Index(["same", "same", "last", "last"], name="source_row"))


def design(result, frame):
    columns = []
    for coefficient in result.coefficients:
        term = coefficient.term
        if term == "Intercept":
            column = np.ones(len(frame))
        elif term in {"x", "p1", "p2"}:
            column = frame[term].to_numpy()
        elif term.startswith("g["):
            column = (frame.g == term[2:-1]).to_numpy(dtype=float)
        else:
            column = np.zeros(len(frame))
        columns.append(column)
    return np.column_stack(columns)


def test_saved_point_targets_full_delta_intervals_and_reporting_reference(saved, monkeypatch):
    name, _, result = saved
    modules = {"sreg": ("robust.smm", "fit_sreg"), "mmreg": ("robust.smm", "fit_mmreg"),
               "ivcue": ("iv.cue", "fit_ivcue"), "mixedflex": ("mixed.flexible_lmm", "fit_mixedflex")}
    module, function = modules[name]

    def refuse_fit(*args, **kwargs):
        raise AssertionError("A restored prediction must never refit")

    monkeypatch.setattr(importlib.import_module("openecon.econometrics." + module), function, refuse_fit)
    new = query()
    original = copy.deepcopy(result.model_dump())
    x = design(result, new)
    beta = np.array([c.estimate for c in result.coefficients])
    covariance = np.array(result.covariance_matrix)
    expected, se = x @ beta, np.sqrt(np.einsum("nk,kl,nl->n", x, covariance, x))
    diagonal_only = np.sqrt((x * x) @ np.diag(covariance))
    assert np.max(abs(se - diagonal_only)) > 1e-5
    actual = oe.predict(result, new, interval="mean", alpha=.1)
    critical = t.ppf(.95, result.inference["df_inference"]) if name in {"sreg", "mmreg"} else norm.ppf(.95)
    assert_allclose(actual.response, expected, rtol=2e-13, atol=2e-13)
    assert_allclose(actual.std_error, se, rtol=2e-12)
    assert_allclose(actual.ci_low, expected - critical * se, rtol=2e-10, atol=2e-12)
    assert_allclose(actual.ci_high, expected + critical * se, rtol=2e-10, atol=2e-12)
    assert_allclose(oe.predict(result, new, kind="xb").xb, expected)
    assert_allclose(oe.predict(result, new, kind="stdp").stdp, se)
    assert actual.index.equals(new.index)
    assert {"sreg": "robust fitted", "mmreg": "robust fitted", "ivcue": "structural equation",
            "mixedflex": "population response"}[name] in actual.attrs["response_definition"]
    assert result.model_dump() == original


@pytest.mark.parametrize("method", ["ame", "mem"])
def test_marginal_effects_have_independent_full_parameter_gradients(saved, method):
    name, _, result = saved
    variables = ["x", "g"] if name in {"sreg", "mmreg"} else ["x", "p1", "p2"] if name == "ivcue" else ["x"]
    terms = [c.term for c in result.coefficients]
    effects = ["x", "g[b]", "g[c]"] if name in {"sreg", "mmreg"} else variables
    indices = [terms.index(term) for term in effects]
    beta, covariance = np.array([c.estimate for c in result.coefficients]), np.array(result.covariance_matrix)
    actual = oe.margins(result, variables, data=query(), method=method, at={"x": [-.5, .8]})
    gradient = np.vstack([np.eye(len(terms))[indices]] * 2)
    assert_allclose(actual.estimate, np.tile(beta[indices], 2))
    assert_allclose(actual.attrs["delta_gradients"], gradient, rtol=2e-13, atol=2e-13)
    assert_allclose(actual.std_error, np.sqrt(np.einsum("nk,kl,nl->n", gradient, covariance, gradient)))
    assert actual.attrs["parameter_terms"] == terms
    if name == "mixedflex":
        nuisance = [i for i, term in enumerate(terms) if term.startswith("/")]
        assert_allclose(np.array(actual.attrs["delta_gradients"])[:, nuisance], 0, atol=0)
    derivative = oe.predict(result, query(), kind="derivative", term=variables[-1] if name == "ivcue" else "x", interval="mean")
    index = terms.index(variables[-1] if name == "ivcue" else "x")
    assert_allclose(derivative.iloc[:, 0], beta[index])
    assert_allclose(derivative.std_error, np.sqrt(covariance[index, index]))


def test_complete_saved_parameter_permutation_and_default_device_cpu(saved):
    _, _, result = saved
    expected = oe.predict(result, query(), interval="mean")
    permutation = np.arange(len(result.coefficients))[::-1]
    changed = result.model_copy(deep=True)
    changed.coefficients = [changed.coefficients[i] for i in permutation]
    changed.covariance_matrix = np.array(changed.covariance_matrix)[permutation][:, permutation].tolist()
    changed = restored(changed)
    with torch.device("meta"):
        actual = oe.predict(changed, query(), interval="mean")
        margins = oe.margins(changed, "x", data=query())
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
    assert_allclose(margins.estimate, next(c.estimate for c in result.coefficients if c.term == "x"))


def test_dataset_complete_replay_missing_index_and_global_margins(saved, tmp_path):
    name, _, result = saved
    new = pd.concat([query()] * 7)
    new.iloc[:4, new.columns.get_loc("x")] = np.nan
    if name in {"sreg", "mmreg"}:
        new.iloc[12, new.columns.get_loc("g")] = None
    if name == "ivcue":
        new.iloc[12, new.columns.get_loc("p1")] = np.nan
    path = tmp_path / "queries.parquet"
    new.to_parquet(path)
    expected = oe.predict(result, new, interval="mean")
    output = oe.predict(result, scan(path), interval="mean", batch_rows=3)
    assert isinstance(output, Dataset)
    actual = pd.concat(list(output.iter_batches(batch_rows=2)))
    assert actual.index.equals(new.index)
    assert_allclose(actual, expected, equal_nan=True, rtol=2e-12, atol=2e-13)
    assert output.metadata["analysis"]["response_definition"] == expected.attrs["response_definition"]
    for method in ["ame", "mem"]:
        eager = oe.margins(result, "x", data=new, method=method, at={"x": [-1., .5]})
        streamed = oe.margins(result, "x", data=scan(path), method=method, at={"x": [-1., .5]}, batch_rows=3)
        pd.testing.assert_frame_equal(pd.DataFrame(streamed), pd.DataFrame(eager), rtol=2e-12, atol=2e-13)
        assert_allclose(streamed.attrs["delta_gradients"], eager.attrs["delta_gradients"])
        assert streamed.attrs["complete_evaluation_rows"] == len(new) - len(expected.attrs["missing_row_positions"])


def test_invalid_state_and_unsupported_targets_fail_closed(saved):
    name, _, result = saved
    new = query()
    for kind in ["conditional", "latent", "residuals"]:
        with pytest.raises(AnalysisError):
            oe.predict(result, new, kind=kind)
    for options in [{"interval": "observation"}, {"outcome": "y"}, {"target": "conditional"},
                    {"random_effects": {"new": 0.}}]:
        with pytest.raises(AnalysisError):
            oe.predict(result, new, **options)
    with pytest.raises(AnalysisError, match="explicitly"):
        oe.predict(result)
    changed = result.model_copy(deep=True)
    changed.covariance_matrix[0][0] *= 2
    with pytest.raises(AnalysisError, match="standard errors and covariance disagree"):
        oe.predict(changed, new)
    changed = result.model_copy(deep=True)
    changed.provenance["estimator"] = "ols"
    with pytest.raises(AnalysisError, match="identity"):
        oe.predict(changed, new)
    changed = result.model_copy(deep=True)
    changed.provenance["design_terms"][0] = "stale_term"
    with pytest.raises(AnalysisError, match="fitted design/covariance records"):
        oe.predict(changed, new)
    changed = result.model_copy(deep=True)
    changed.spec.weights, changed.spec.weight_type = "w", "aweight"
    with pytest.raises(AnalysisError, match="specification is invalid"):
        oe.predict(changed, new.assign(w=1.))
    changed = result.model_copy(deep=True)
    changed.spec.missing = "raise"
    with pytest.raises(AnalysisError, match="missing values"):
        oe.predict(changed, new.assign(x=np.nan))
    if name in {"sreg", "mmreg"}:
        changed = result.model_copy(deep=True)
        changed.extra["scale_fixed_during_mm"] = name != "mmreg"
        with pytest.raises(AnalysisError, match="joint-score state"):
            oe.predict(changed, new)
        with pytest.raises(AnalysisError, match="category absent from the fitted levels"):
            oe.predict(result, new.assign(g="unknown"))
        changed = result.model_copy(deep=True)
        changed.inference["df_inference"] += 1
        with pytest.raises(AnalysisError, match="degrees of freedom disagree"):
            oe.predict(changed, new)
    elif name == "ivcue":
        with pytest.raises(AnalysisError, match="lack: p1"):
            oe.predict(result, new.drop(columns="p1"))
        changed = result.model_copy(deep=True)
        changed.extra["endogenous"] = ["p2", "p1"]
        with pytest.raises(AnalysisError, match="structural roles"):
            oe.predict(changed, new)
    else:
        changed = result.model_copy(deep=True)
        changed.extra["levels"][0]["columns"] = ["low"]
        with pytest.raises(AnalysisError, match="grouping levels"):
            oe.predict(changed, new)
        changed = result.model_copy(deep=True)
        next(c for c in changed.coefficients if c.term == "/var(Residual)").estimate = 0.
        with pytest.raises(AnalysisError, match="variance components"):
            oe.predict(changed, new)
    if name in {"ivcue", "mixedflex"}:
        changed = result.model_copy(deep=True)
        sd = np.array([c.std_error for c in changed.coefficients])
        changed.covariance_matrix = np.outer(sd, sd).tolist()
        with pytest.raises(AnalysisError, match="nonsingular positive definite"):
            oe.predict(changed, new)


def test_bounded_resources_refuse_before_design_allocation(saved, monkeypatch):
    name, _, result = saved
    def refuse_row_tensor(*args, **kwargs):
        raise AssertionError("Resource plan must precede row-position/design tensors")

    monkeypatch.setattr(torch, "tensor", refuse_row_tensor)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        oe.predict(result, pd.concat([query()] * 5000))
    assert error.value.code == "workspace_limit"
    assert error.value.resource_plan["operation"] == "saved linear target evaluation"
    changed = result.model_copy(deep=True)
    key = "joint_estimating_equation_order" if name in {"sreg", "mmreg"} else "instruments" if name == "ivcue" else "levels"
    changed.extra[key] = [None] * 100
    with pytest.raises(AnalysisError, match="exceed the fitted domain"):
        oe.predict(changed, query())


def test_mixedflex_unstructured_population_target_and_singular_random_covariance():
    rng = np.random.default_rng(93)
    group = np.repeat(np.arange(12), 6)
    x = rng.normal(size=len(group))
    effects = rng.multivariate_normal([0., 0.], [[1.4, .3], [.3, .7]], 12)
    y = 1 + .4 * x + effects[group, 0] + x * effects[group, 1] + rng.normal(size=len(group)) * .35
    frame = pd.DataFrame(dict(y=y, x=x, id=group))
    result = restored(oe.mixedflex(data=frame, y="y", x=["x"], group=["id"], random=["x"],
                                   covstructure="unstructured"))
    # Neither fitted nor unknown group labels belong to this integrated mean.
    new = pd.DataFrame({"x": [-1., 0., 1.], "id": ["new", "another", None]})
    covariance = np.array(result.covariance_matrix)
    gradient = design(result, new)
    expected = gradient @ np.array([c.estimate for c in result.coefficients])
    actual = oe.predict(result, new, interval="mean")
    assert_allclose(actual.response, expected)
    assert_allclose(actual.std_error, np.sqrt(np.einsum("nk,kl,nl->n", gradient, covariance, gradient)))
    assert np.max(abs(covariance[:2, 2:])) > 1e-5
    margin = oe.margins(result, "x", data=new, method="mem")
    assert_allclose(margin.attrs["delta_gradients"], [[0., 1., 0., 0., 0., 0.]], atol=0.)
    changed = result.model_copy(deep=True)
    values = {c.term: c.estimate for c in changed.coefficients}
    next(c for c in changed.coefficients if c.term == "/cov(_cons,x[id])").estimate = np.sqrt(values["/var(x[id])"] * values["/var(_cons[id])"])
    with pytest.raises(AnalysisError, match="random-effect covariance must be positive definite"):
        oe.predict(changed, new)


@pytest.mark.parametrize("name", ["sreg", "mmreg"])
@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_smm_saved_covariance_domains_and_actual_omitted_predictor(name, covariance):
    rng = np.random.default_rng(22)
    x = rng.normal(size=95)
    frame = pd.DataFrame({"x": x, "copy_x": 2 * x, "y": 1 + 2 * x + rng.normal(size=95) * .5,
                          "group": np.arange(95) % 9})
    result = restored(getattr(oe, name)(data=frame, y="y", x=["x", "copy_x"], starts=25,
                                        covariance=covariance, cluster="group" if covariance == "cluster" else None))
    omitted = result.provenance["omitted_terms"]
    assert len(omitted) == 1
    new = frame[["x", "copy_x"]].iloc[:7]
    gradient = np.zeros((len(new), len(result.coefficients)))
    for index, coefficient in enumerate(result.coefficients):
        gradient[:, index] = 1. if coefficient.term == "Intercept" else new[coefficient.term]
    beta = np.array([c.estimate for c in result.coefficients])
    actual = oe.predict(result, new, interval="mean")
    assert_allclose(actual.response, gradient @ beta)
    assert_allclose(actual.std_error, np.sqrt(np.einsum("nk,kl,nl->n", gradient, result.covariance_matrix, gradient)))
    margin = oe.margins(result, omitted[0], data=new)
    assert_allclose(margin.estimate, 0., atol=0.)
    assert_allclose(margin.std_error, 0., atol=0.)
    assert_allclose(margin.attrs["delta_gradients"], 0., atol=0.)
