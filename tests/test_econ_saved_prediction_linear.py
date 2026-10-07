"""Saved linear/population predictions against independent NumPy algebra.

Whole parameter covariances, equation selection, categorical contrasts and
missing-row alignment are checked after the fit has been serialized.  No
third-party econometric implementation is needed by these prediction tests.
"""

import copy

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.models import ResultBundle


def sample(seed=9083, groups=18, periods=8):
    rng = np.random.default_rng(seed)
    n = groups * periods
    group = np.repeat(np.arange(groups), periods)
    frame = pd.DataFrame({"id": group, "t": np.tile(np.arange(periods), groups),
                          "tick": np.arange(n), "x": rng.normal(size=n),
                          "z": rng.normal(size=n), "g": np.resize(["A", "B", "C"], n)})
    u = rng.normal(size=groups)[group]
    error = rng.normal(size=n)
    frame["p"] = .7 * frame.z + .3 * frame.x + .3 * error + rng.normal(size=n)
    frame["y"] = .7 + .8 * frame.x + .3 * (frame.g == "B") - .2 * (frame.g == "C") + .6 * u + error
    frame["iv_y"] = .4 + .5 * frame.x + .9 * frame.p + .7 * u + error
    frame["count"] = rng.poisson(np.exp(.7 + .3 * frame.x + .2 * (frame.g == "B") + .2 * u))
    probability = 1 / (1 + np.exp(-(.2 + .4 * frame.x + .2 * (frame.g == "B") + .1 * u)))
    frame["binary"] = rng.binomial(1, probability)
    frame["w"] = rng.uniform(.1, 3, n)
    frame["offset"] = rng.uniform(-.3, .3, n)
    return frame


def stored(result):
    return ResultBundle.model_validate_json(result.model_dump_json())


def independent_design(result, frame, equation=None):
    values = {"Intercept": np.ones(len(frame)), "x": frame.x.to_numpy(),
              "p": frame.p.to_numpy() if "p" in frame else np.zeros(len(frame)),
              "g[B]": (frame.g == "B").to_numpy(float), "g[C]": (frame.g == "C").to_numpy(float)}
    columns = []
    for coefficient in result.coefficients:
        term = coefficient.term
        if equation is not None:
            value = values.get(term.removeprefix(equation + ":"), np.zeros(len(frame))) if coefficient.equation == equation else np.zeros(len(frame))
        else:
            value = values.get(term, np.zeros(len(frame)))
        columns.append(value)
    return np.column_stack(columns)


@pytest.fixture(scope="module", params=["rreg", "qreg", "bsqreg", "iqreg", "sqreg",
                                           "xtreg_pooled", "xtreg_re", "xtivreg_re", "xtgls",
                                           "xtpcse", "xtfmb", "prais", "mixed"])
def linear_result(request):
    frame = sample()
    name = request.param
    kwargs = dict(data=frame, y="y", x=["x", "g"], categorical=["g"], missing="drop")
    outcome = None
    if name in {"bsqreg", "iqreg", "sqreg"}:
        result = getattr(oe, name)(**kwargs, reps=8, seed=41)
        outcome = "q25" if name == "sqreg" else None
    elif name.startswith("xtreg_"):
        result = oe.xtreg(**kwargs, panel="id", time="t", model=name.split("_")[1])
    elif name == "xtivreg_re":
        result = oe.xtivreg(**{**kwargs, "y": "iv_y"}, endog=["p"], instruments=["z"], panel="id", time="t", model="re")
    elif name in {"xtgls", "xtpcse", "xtfmb"}:
        result = getattr(oe, name)(**kwargs, panel="id", time="t")
    elif name == "prais":
        result = oe.prais(**kwargs, time="tick")
    elif name == "mixed":
        result = oe.mixed(**kwargs, group="id")
    else:
        result = getattr(oe, name)(**kwargs)
    return name, frame, stored(result), outcome


def test_scalar_saved_prediction_response_and_complete_covariance(linear_result):
    _, _, result, outcome = linear_result
    new = pd.DataFrame({"x": [-.9, .3, 1.4], "p": [.1, -.7, 1.2], "g": ["C", "A", "B"]}, index=["row", "row", "other"])
    before = copy.deepcopy(result.model_dump())
    x = independent_design(result, new, outcome)
    beta = np.array([coefficient.estimate for coefficient in result.coefficients])
    expected = x @ beta
    variance = np.einsum("nk,kl,nl->n", x, result.covariance_matrix, x)
    predicted = oe.predict(result, new, outcome=outcome, interval="mean")
    assert_allclose(predicted.response, expected, rtol=2e-13, atol=2e-13)
    assert_allclose(predicted.std_error, np.sqrt(variance), rtol=2e-12, atol=2e-13)
    assert predicted.index.tolist() == new.index.tolist()
    assert predicted.attrs["response_definition"]
    assert_allclose(oe.predict(result, new, outcome=outcome, kind="xb").xb, expected, rtol=2e-13)
    assert_allclose(oe.predict(result, new, outcome=outcome, kind="stdp").stdp, np.sqrt(variance), rtol=2e-12)
    assert result.model_dump() == before


@pytest.mark.parametrize("method", ["ame", "mem"])
def test_linear_saved_margins_have_full_delta_covariance_and_categorical_contrasts(linear_result, method):
    _, frame, result, outcome = linear_result
    prefix = outcome + ":" if outcome else ""
    terms = [coefficient.term for coefficient in result.coefficients]
    indices = [terms.index(prefix + term) for term in ["x", "g[B]", "g[C]"]]
    beta = np.array([coefficient.estimate for coefficient in result.coefficients])
    gradient = np.eye(len(beta))[indices]
    actual = oe.margins(result, ["x", "g"], data=frame, method=method, outcome=outcome,
                        at={"x": [-.5, .5]})
    assert actual.variable.tolist() == ["x", "g[B]", "g[C]"] * 2
    assert_allclose(actual.estimate, np.tile(beta[indices], 2), rtol=2e-12, atol=2e-13)
    assert_allclose(actual.std_error, np.tile(np.sqrt(np.diag(np.array(result.covariance_matrix))[indices]), 2), rtol=2e-12)
    assert_allclose(actual.attrs["delta_gradients"], np.vstack([gradient, gradient]), rtol=2e-12, atol=2e-13)
    derivative = oe.predict(result, frame.head(), kind="derivative", term="x", outcome=outcome, interval="mean")
    assert_allclose(derivative["dydx[x]"], beta[indices[0]])
    assert_allclose(derivative.std_error, np.sqrt(np.diag(np.array(result.covariance_matrix))[indices[0]]))


@pytest.mark.parametrize("name", ["areg", "reghdfe", "ivreghdfe", "xtreg", "xtivreg", "ppmlhdfe"])
def test_absorbed_prediction_never_zeros_unrecorded_group_effects(name):
    frame = sample()
    kwargs = dict(data=frame, y="count" if name == "ppmlhdfe" else "y", x=["x", "g"], categorical=["g"], missing="drop")
    if name in {"ivreghdfe", "xtivreg"}:
        kwargs.update(y="iv_y", endog=["p"], instruments=["z"])
    kwargs.update({"panel": "id", "time": "t"} if name in {"xtreg", "xtivreg"} else {"absorb": "id" if name == "areg" else ["id"]})
    result = stored(getattr(oe, name)(**kwargs))
    # A historical result has no fitted effect map. New fits carry that map
    # and are covered by the complete group-state prediction tests.
    result.extra.pop('group_state', None)
    new = frame[["x", "g", "p"]].head(7)
    x = independent_design(result, new)
    beta = np.array([coefficient.estimate for coefficient in result.coefficients])
    actual = oe.predict(result, new, kind="xb", interval="mean")
    assert_allclose(actual.xb, x @ beta, rtol=2e-12, atol=2e-13)
    assert "excluding fitted group effects" in actual.attrs["response_definition"]
    with pytest.raises(AnalysisError, match="absorbed-effect|group effects"):
        oe.predict(result, new)
    margin = oe.margins(result, "x", data=new, kind="xb")
    index = [coefficient.term for coefficient in result.coefficients].index("x")
    assert_allclose(margin.estimate, beta[index])
    assert_allclose(margin.std_error, np.sqrt(np.array(result.covariance_matrix)[index, index]))
    with pytest.raises(AnalysisError, match="absorbed-effect|group effects"):
        oe.margins(result, "x", data=new)


@pytest.mark.parametrize("name", ["xtgee", "xtlogit", "xtprobit", "xtpoisson"])
def test_population_gee_saved_link_margins_independent_finite_differences(name):
    frame = sample()
    kwargs = dict(data=frame, y="count" if name == "xtpoisson" else "binary", x=["x", "g"], categorical=["g"], panel="id", missing="drop")
    if name == "xtgee":
        kwargs.update(family="binomial", link="logit", corr="independent")
    else:
        kwargs.update(model="pa", corr="independent")
    result = stored(getattr(oe, name)(**kwargs))
    new = frame.head(13).drop(columns=[result.spec.outcome, "id", "t"])
    x = independent_design(result, new)
    beta = np.array([coefficient.estimate for coefficient in result.coefficients])
    terms = [coefficient.term for coefficient in result.coefficients]
    if name == "xtprobit":
        from math import erf, sqrt
        def inverse(eta):
            return np.array([.5 * (1 + erf(value / sqrt(2))) for value in eta])
    elif name == "xtpoisson":
        inverse = np.exp
    else:
        def inverse(eta):
            return 1 / (1 + np.exp(-eta))
    expected = inverse(x @ beta)
    predicted = oe.predict(result, new, interval="mean")
    assert_allclose(predicted.response, expected, rtol=2e-12, atol=2e-14)
    step = 1e-5
    jac = np.column_stack([(inverse(x @ (beta + step * np.eye(len(beta))[i]))
                           - inverse(x @ (beta - step * np.eye(len(beta))[i]))) / (2 * step) for i in range(len(beta))])
    assert_allclose(predicted.std_error, np.sqrt(np.einsum("nk,kl,nl->n", jac, result.covariance_matrix, jac)), rtol=2e-8)
    index = terms.index("x")

    def effect(coefficients):
        eta = x @ coefficients
        if name == "xtprobit":
            derivative = np.exp(-eta ** 2 / 2) / np.sqrt(2 * np.pi)
        elif name == "xtpoisson":
            derivative = np.exp(eta)
        else:
            p = inverse(eta)
            derivative = p * (1 - p)
        return np.mean(derivative * coefficients[index])

    gradient = np.array([(effect(beta + step * np.eye(len(beta))[i]) - effect(beta - step * np.eye(len(beta))[i])) / (2 * step) for i in range(len(beta))])
    margins = oe.margins(result, "x", data=new)
    assert_allclose(margins.estimate, effect(beta), rtol=2e-12)
    assert_allclose(margins.std_error, np.sqrt(gradient @ np.array(result.covariance_matrix) @ gradient), rtol=2e-8)
    assert "population-averaged" in predicted.attrs["response_definition"]


def test_sqreg_selection_and_full_saved_parameter_permutation(linear_result):
    name, frame, result, outcome = linear_result
    if name != "sqreg":
        return
    with pytest.raises(AnalysisError, match="Select one saved simultaneous quantile"):
        oe.predict(result, frame.head())
    with pytest.raises(AnalysisError, match="Select one of the saved simultaneous quantile"):
        oe.predict(result, frame.head(), outcome="q20")
    expected = oe.predict(result, frame.head(), outcome="q75", interval="mean")
    permutation = np.arange(len(result.coefficients))[::-1]
    changed = result.model_copy(deep=True)
    changed.coefficients = [changed.coefficients[index] for index in permutation]
    changed.covariance_matrix = np.array(changed.covariance_matrix)[permutation][:, permutation].tolist()
    assert_allclose(oe.predict(changed, frame.head(), outcome="q75", interval="mean"), expected, rtol=2e-12)


@pytest.mark.parametrize("command", ["xtreg", "xtivreg"])
def test_first_difference_levels_are_rejected(command):
    frame = sample()
    kwargs = dict(data=frame, y="y", x=["x"], panel="id", time="t", model="fd")
    if command == "xtivreg":
        kwargs.update(y="iv_y", endog=["p"], instruments=["z"])
    result = stored(getattr(oe, command)(**kwargs))
    with pytest.raises(AnalysisError, match="First-difference predictions require consecutive"):
        oe.predict(result, frame)


def test_corrupt_saved_encoding_and_panel_model_are_rejected():
    frame = sample()
    result = stored(oe.xtreg(data=frame, y="y", x=["x", "g"], categorical=["g"], panel="id", model="re"))
    changed = result.model_copy(deep=True)
    changed.provenance["categorical_encoding"]["g"]["reference"] = "unrecorded"
    with pytest.raises(AnalysisError, match="fitted levels/reference"):
        oe.predict(changed, frame)
    changed = result.model_copy(deep=True)
    changed.extra["model"] = "fe"
    with pytest.raises(AnalysisError, match="saved panel model and fitted model disagree"):
        oe.predict(changed, frame)
    changed = result.model_copy(deep=True)
    changed.coefficients[0].term = "unrecorded_transform"
    with pytest.raises(AnalysisError, match="not a supported scalar response feature"):
        oe.predict(changed, frame)
    with pytest.raises(AnalysisError, match="category"):
        oe.predict(result, pd.DataFrame({"x": [1.], "g": ["unrecorded"]}))


@pytest.mark.parametrize("estimator", ["areg", "reghdfe", "ivreghdfe", "xtreg", "xtivreg", "xtgee", "xtgls", "xtpcse", "xtfmb", "mixed"])
def test_missing_mandatory_estimator_roles_are_not_inferred_from_coefficients(estimator):
    frame = sample()
    result = stored(oe.qreg(data=frame, y="y", x=["x"], missing="drop"))
    result.spec.estimator = estimator
    result.provenance["estimator"] = estimator
    with pytest.raises(AnalysisError, match="required model roles/options"):
        oe.predict(result, frame, kind="xb")


def test_relabelled_count_fit_does_not_become_an_identity_response():
    frame = sample()
    result = stored(oe.poisson(data=frame, y="count", x=["x"]))
    result.spec.estimator = "qreg"
    with pytest.raises(AnalysisError, match="estimator identity"):
        oe.predict(result, frame)
    result.provenance["estimator"] = "qreg"
    with pytest.raises(AnalysisError, match="family/link disagree"):
        oe.predict(result, frame)


def test_sqreg_corrupt_equation_labels_rejected():
    frame = sample()
    result = stored(oe.sqreg(data=frame, y="y", x=["x"], quantiles=[.25, .75], reps=8, seed=43))
    changed = result.model_copy(deep=True)
    changed.coefficients[0].equation = "q75"
    with pytest.raises(AnalysisError, match="terms do not match their equations"):
        oe.predict(changed, frame, outcome="q25")
    changed = result.model_copy(deep=True)
    changed.extra["equations"] = ["q25", "q25"]
    with pytest.raises(AnalysisError, match="Saved simultaneous quantile equations are invalid"):
        oe.predict(changed, frame, outcome="q25")


def test_gee_corrupt_link_rejected_and_logit_tail_delta_gradient_is_retained():
    frame = sample()
    result = stored(oe.xtgee(data=frame, y="binary", x=["x"], panel="id", family="binomial", link="logit", corr="independent"))
    changed = result.model_copy(deep=True)
    changed.extra["link"] = "identity"
    with pytest.raises(AnalysisError, match="family/link disagree"):
        oe.predict(changed, frame)
    changed = result.model_copy(deep=True)
    terms = [coefficient.term for coefficient in changed.coefficients]
    for coefficient in changed.coefficients:
        coefficient.estimate = 40. if coefficient.term == "Intercept" else 1.
    new = pd.DataFrame({"x": [0., .5, 1.]})
    actual = oe.margins(changed, "x", data=new)
    eta = 40 + new.x.to_numpy()
    probability_complement = 1 / (1 + np.exp(eta))
    first = (1 - probability_complement) * probability_complement
    second = first * (2 * probability_complement - 1)
    expected_gradient = np.array([np.mean(second), np.mean(second * new.x.to_numpy() + first)])
    if terms != ["Intercept", "x"]:
        expected_gradient = expected_gradient[[["Intercept", "x"].index(term) for term in terms]]
    assert actual.estimate.iloc[0] > 0
    assert_allclose(actual.estimate, np.mean(first), rtol=2e-13)
    assert_allclose(actual.attrs["delta_gradients"][0], expected_gradient, rtol=2e-12, atol=0)
    prediction = oe.predict(changed, new, interval="mean")
    assert (prediction.std_error > 0).all()
    assert np.isfinite(prediction.to_numpy()).all()


def test_linear_dataset_prediction_all_scalar_kinds_preserves_missing_rows_and_index(linear_result, tmp_path):
    _, frame, result, outcome = linear_result
    new = frame.head(23).copy()
    new.index = pd.Index([f"row{index // 2}" for index in range(len(new))], name="duplicate_index")
    new.loc[new.index[0], "x"] = np.nan
    new.iloc[9, new.columns.get_loc("g")] = None
    path = tmp_path / "evaluation.parquet"
    new.to_parquet(path)
    for kind, options in [("response", {}), ("xb", {}), ("stdp", {}), ("derivative", {"term": "x"})]:
        interval = None if kind == "stdp" else "mean"
        expected = oe.predict(result, new, kind=kind, interval=interval, outcome=outcome, **options)
        output = oe.predict(result, scan(path), kind=kind, interval=interval, outcome=outcome,
                            batch_rows=3, **options)
        assert isinstance(output, Dataset)
        actual = pd.concat(list(output.iter_batches(batch_rows=2)))
        assert actual.index.tolist() == new.index.tolist()
        assert actual.index.names == new.index.names
        assert actual.columns.tolist() == expected.columns.tolist()
        assert_allclose(actual, expected, rtol=2e-12, atol=2e-13, equal_nan=True)
        metadata = output.metadata["analysis"]
        assert metadata["streaming"]["maximum_batch_rows"] <= 3
        assert metadata["streaming"]["full_source_collected"] is False
        assert metadata["missing_prediction_rows"] == len(expected.attrs["missing_row_positions"])
        assert metadata["response_definition"] == expected.attrs["response_definition"]


@pytest.mark.parametrize("method", ["ame", "mem"])
def test_linear_dataset_margins_reduce_full_covariance_with_global_interventions(linear_result, method):
    _, frame, result, outcome = linear_result
    new = frame.head(31).copy()
    new.iloc[:3, new.columns.get_loc("x")] = np.nan  # a complete empty evaluation block
    new.iloc[8, new.columns.get_loc("g")] = None
    expected = oe.margins(result, ["x", "g"], data=new, method=method, outcome=outcome,
                          at={"x": [-.5, .5], "g": ["A", "C"]})
    actual = oe.margins(result, ["x", "g"], data=Dataset.from_frame(new), method=method,
                        outcome=outcome, at={"x": [-.5, .5], "g": ["A", "C"]}, batch_rows=3)
    pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(expected), rtol=2e-12, atol=2e-13)
    assert_allclose(actual.attrs["delta_gradients"], expected.attrs["delta_gradients"], rtol=2e-12, atol=2e-13)
    assert actual.attrs["parameter_terms"] == expected.attrs["parameter_terms"]
    assert actual.attrs["streaming"]["maximum_batch_rows"] <= 3
    assert actual.attrs["streaming"]["full_source_collected"] is False
    assert actual.attrs["complete_evaluation_rows"] == expected.attrs["complete_evaluation_rows"]


@pytest.mark.parametrize("name", ["areg", "reghdfe", "ivreghdfe", "xtreg", "xtivreg", "ppmlhdfe"])
def test_absorbed_dataset_xb_margins_are_globally_identical(name):
    frame = sample()
    kwargs = dict(data=frame, y="count" if name == "ppmlhdfe" else "y", x=["x", "g"], categorical=["g"], missing="drop")
    if name in {"ivreghdfe", "xtivreg"}:
        kwargs.update(y="iv_y", endog=["p"], instruments=["z"])
    kwargs.update({"panel": "id", "time": "t"} if name in {"xtreg", "xtivreg"} else {"absorb": "id" if name == "areg" else ["id"]})
    result = stored(getattr(oe, name)(**kwargs))
    new = frame.head(23).copy()
    new.iloc[0, new.columns.get_loc("x")] = np.nan
    source = Dataset.from_frame(new)
    expected = oe.predict(result, new, kind="xb", interval="mean")
    output = oe.predict(result, source, kind="xb", interval="mean", batch_rows=3)
    assert_allclose(pd.concat(list(output.iter_batches(batch_rows=2))), expected, rtol=2e-12, atol=2e-13, equal_nan=True)
    for method in ["ame", "mem"]:
        expected = oe.margins(result, ["x", "g"], data=new, kind="xb", method=method, at={"x": [-1., 1.]})
        actual = oe.margins(result, ["x", "g"], data=source, kind="xb", method=method,
                            at={"x": [-1., 1.]}, batch_rows=3)
        pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(expected), rtol=2e-12, atol=2e-13)
        assert_allclose(actual.attrs["delta_gradients"], expected.attrs["delta_gradients"], rtol=2e-12, atol=2e-13)
    historical = result.model_copy(deep=True)
    historical.extra.pop('group_state', None)
    with pytest.raises(AnalysisError, match="group effects"):
        oe.predict(historical, source)


@pytest.mark.parametrize("name", ["xtgee", "xtlogit", "xtprobit", "xtpoisson"])
@pytest.mark.parametrize("method", ["ame", "mem"])
def test_gee_dataset_nonlinear_effects_use_global_means_and_global_delta_gradients(name, method):
    frame = sample()
    kwargs = dict(data=frame, y="count" if name == "xtpoisson" else "binary", x=["x", "g"], categorical=["g"], panel="id", missing="drop")
    kwargs.update(dict(family="binomial", link="logit", corr="independent") if name == "xtgee" else dict(model="pa", corr="independent"))
    result = stored(getattr(oe, name)(**kwargs))
    new = frame.head(29).copy()
    new.iloc[0, new.columns.get_loc("x")] = np.nan
    new.iloc[8, new.columns.get_loc("g")] = None
    source = Dataset.from_frame(new)
    expected = oe.predict(result, new, interval="mean")
    output = oe.predict(result, source, interval="mean", batch_rows=3)
    assert_allclose(pd.concat(list(output.iter_batches(batch_rows=2))), expected, rtol=2e-12, atol=2e-13, equal_nan=True)
    expected = oe.margins(result, ["x", "g"], data=new, method=method, at={"x": [-.7, 1.3]})
    actual = oe.margins(result, ["x", "g"], data=source, method=method, at={"x": [-.7, 1.3]}, batch_rows=3)
    pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(expected), rtol=3e-12, atol=3e-13)
    assert_allclose(actual.attrs["delta_gradients"], expected.attrs["delta_gradients"], rtol=3e-12, atol=3e-13)


@pytest.mark.parametrize("command", ["qreg", "xtreg"])
@pytest.mark.parametrize("method", ["ame", "mem"])
def test_weighted_linear_dataset_skips_zero_weight_unknown_categories_and_matches_dense(command, method):
    frame = sample()
    kwargs = dict(data=frame, y="y", x=["x", "g"], categorical=["g"], weights="w", weight_type="aweight", missing="drop")
    if command == "xtreg":
        kwargs.update(panel="id", model="pooled")
    result = stored(getattr(oe, command)(**kwargs))
    new = frame.head(27).copy()
    new.iloc[:3, new.columns.get_loc("w")] = 0.
    new.iloc[:3, new.columns.get_loc("g")] = "unknown_zero_weight"
    new.iloc[10, new.columns.get_loc("w")] = np.nan
    new.loc[new.w > 0, "w"] *= 1e250  # global normalization must avoid overflow
    expected = oe.margins(result, ["x", "g"], data=new, method=method, at={"x": [-1., 1.]})
    actual = oe.margins(result, ["x", "g"], data=Dataset.from_frame(new), method=method,
                        at={"x": [-1., 1.]}, batch_rows=3)
    pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(expected), rtol=3e-12, atol=3e-13)
    assert_allclose(actual.attrs["delta_gradients"], expected.attrs["delta_gradients"], rtol=3e-12, atol=3e-13)
    assert actual.attrs["zero_weight_rows_excluded"] == expected.attrs["zero_weight_rows_excluded"] == 3
