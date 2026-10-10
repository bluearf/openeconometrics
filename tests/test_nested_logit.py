"""Choice geometry, native inference, finite-interior and portable-state gates."""
from __future__ import annotations

import copy
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.torch_test_state import preserve_torch_default_device

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.discrete import nested_logit as nl
from openecon.resources import use_workspace_budget


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic(seed=129, cases=180):
    rng = np.random.default_rng(seed)
    rows = []
    lambdas, code = np.array([.45, .67]), np.array([0, 0, 0, 1, 1])
    for case in range(cases):
        X = rng.uniform(.2, 2, (5, 2))
        utility = X@np.array([-.8, .55])
        scaled = utility/lambdas[code]
        sums = np.array([np.exp(scaled[code == nest]).sum() for nest in (0, 1)])
        upper = sums**lambdas
        probability = np.exp(scaled)/sums[code]*upper[code]/upper.sum()
        chosen = rng.choice(5, p=probability)
        for j in range(5):
            rows.append([case, j, "A" if j < 3 else "B", int(j == chosen), *X[j], case//3])
    return pd.DataFrame(rows, columns=["case", "alternative", "nest", "chosen", "cost", "quality", "subject"])


def fit(data, vce="oim", **kwargs):
    return nl.nlogit(data, "chosen", ["cost", "quality"], case="case", alternative="alternative", nest="nest",
                     vce=vce, cluster="subject" if vce == "cr0" else None, **kwargs)


@pytest.fixture(scope="module")
def data():
    return synthetic()


@pytest.fixture(scope="module")
def models(data):
    return {vce: fit(data, vce) for vce in ("oim", "hc0", "cr0")}


def state(model):
    return model.attrs["nested_logit_state"]


def portable(model):
    return json.loads(json.dumps(model.attrs, ensure_ascii=False, allow_nan=False))


def reseal(attrs):
    value = attrs["nested_logit_state"]
    value["checksum"] = nl._checksum(value)
    attrs["settings"] = value["settings"]
    return attrs


def matrix(model, name):
    return model[name].iloc[:, 1:].to_numpy(dtype=float)


def test_joint_fit_interior_and_covariance_dimensions(models):
    for model in models.values():
        value = state(model)
        assert len(model) == 13
        assert value["catalogue"]["nests"] == ["A", "B"]
        assert value["fit"]["parameter_names"] == ["cost", "quality", "lambda[n1]", "lambda[n2]"]
        assert all(.001 < x < .999 for x in value["fit"]["lambdas"])
        assert np.linalg.eigvalsh(matrix(model, "information")).min() > 0
        assert value["fit"]["stationarity"]["normalized_parameter_step"] <= 2e-8
        assert np.asarray(value["fit"]["covariance"]).shape == (4, 4)
        assert value["fit"]["convergence"]["globally_certified"] is False
        assert len(value["fit"]["convergence"]["starts"]) == 3


def test_covariance_uses_complete_case_and_cluster_joint_scores(models):
    for vce, model in models.items():
        f = state(model)["fit"]
        bread = np.asarray(f["bread"])
        score = np.asarray(f["cluster_scores"] if vce == "cr0" else f["case_scores"])
        np.testing.assert_allclose(f["meat"], score.T@score, rtol=2e-12, atol=1e-11)
        expected = bread if vce == "oim" else bread@score.T@score@bread
        np.testing.assert_allclose(f["covariance"], expected, rtol=2e-12, atol=1e-12)
        np.testing.assert_allclose(bread@np.asarray(f["information"]), np.eye(4), rtol=2e-12, atol=1e-12)
        assert np.max(np.abs(np.asarray(f["covariance"])[:2, 2:])) > 1e-4
        if vce == "cr0":
            expected_scores = np.asarray(f["case_scores"]).reshape(60, 3, 4).sum(1)
            np.testing.assert_allclose(f["cluster_scores"], expected_scores, rtol=2e-12, atol=2e-12)


def test_dissimilarity_intervals_are_transformed_full_joint_se(models):
    model = models["oim"]
    for j, row in model["dissimilarities"].iterrows():
        estimate, se = row["estimate"], row["std_error"]
        delta = se/(estimate*(1-estimate))
        location = math.log(estimate/(1-estimate))
        lower, upper = [1/(1+math.exp(-v)) for v in (location-1.959963984540054*delta, location+1.959963984540054*delta)]
        assert row.ci_lower == pytest.approx(lower, abs=1e-13)
        assert row.ci_upper == pytest.approx(upper, abs=1e-13)
        assert model["parameters"].iloc[2+j].ci_lower == row.ci_lower
        assert pd.isna(model["parameters"].iloc[2+j].p_value)


@pytest.mark.parametrize("vce", ("oim", "hc0", "cr0"))
def test_json_replay_all_tables_and_attrs_without_optimizer(models, vce, monkeypatch):
    model = models[vce]
    def forbidden(*args, **kwargs):
        raise AssertionError("restore called optimizer")
    monkeypatch.setattr(nl, "_fit", forbidden)
    monkeypatch.setattr(nl, "maximize_bfgs", forbidden)
    restored = nl.nlogit_restore(portable(model))
    assert restored.attrs == model.attrs
    assert set(restored) == set(model)
    for key in model:
        pd.testing.assert_frame_equal(restored[key], model[key], check_exact=True)


def test_replay_confidence_override_changes_inference_only(models):
    restored = nl.nlogit_restore(models["cr0"], level=.9)
    assert state(restored)["level"] == .9
    assert matrix(restored, "covariance").tolist() == matrix(models["cr0"], "covariance").tolist()
    assert restored["parameters"].iloc[0].ci_upper < models["cr0"]["parameters"].iloc[0].ci_upper
    assert state(restored)["checksum"] != state(models["cr0"])["checksum"]


@pytest.mark.parametrize("key", ("information", "bread", "meat", "covariance", "case_scores", "cluster_scores", "probability_components"))
def test_tampered_full_scientific_matrix_rejected_even_resealed(models, key):
    attrs = portable(models["cr0"])
    attrs["nested_logit_state"]["fit"][key][0][-1] += .005
    with pytest.raises(AnalysisError):
        nl.nlogit_restore(reseal(attrs))


@pytest.mark.parametrize("key", ("params", "beta", "lambdas", "probabilities", "log_probabilities", "case_loglikelihood", "scale"))
def test_tampered_full_scientific_vector_rejected_even_resealed(models, key):
    attrs = portable(models["hc0"])
    attrs["nested_logit_state"]["fit"][key][0] += .03
    with pytest.raises(AnalysisError):
        nl.nlogit_restore(reseal(attrs))


@pytest.mark.parametrize("mutation", ("unknown_field", "false_type", "geometry", "start", "selected_start", "diagnostic", "work", "missing_table"))
def test_complete_schema_and_convergence_tampering_rejected(models, mutation):
    attrs = portable(models["oim"])
    s, f = attrs["nested_logit_state"], attrs["nested_logit_state"]["fit"]
    if mutation == "unknown_field":
        s["hidden_partial_state"] = True
    elif mutation == "false_type":
        f["n_nests"] = True
    elif mutation == "geometry":
        s["catalogue"]["alternatives"][0][1] = "B"
    elif mutation == "start":
        f["convergence"]["starts"][0]["start_dissimilarity"] = .3
    elif mutation == "selected_start":
        f["convergence"]["selected_start"] = 100
    elif mutation == "diagnostic":
        f["stationarity"]["physical_score_max"] = 0.9
    elif mutation == "work":
        f["convergence"]["work_used"] = True
    else:
        del f["information"]
    with pytest.raises(AnalysisError):
        nl.nlogit_restore(reseal(attrs))


def test_input_identity_and_undeclared_availability_canonicalization(models):
    attrs = portable(models["oim"])
    attrs["nested_logit_state"]["inputs"]["available"][0] = 0
    with pytest.raises(AnalysisError):
        nl.nlogit_restore(reseal(attrs))
    attrs = portable(models["oim"])
    attrs["nested_logit_state"]["inputs"]["chosen"][0] = bool(attrs["nested_logit_state"]["inputs"]["chosen"][0])
    with pytest.raises(AnalysisError):
        nl.nlogit_restore(reseal(attrs))


def test_tampered_result_table_rejected(models):
    altered = copy.deepcopy(models["oim"])
    altered["probabilities"].iloc[0, 5] += .01
    with pytest.raises(AnalysisError, match="tables"):
        nl.nlogit_restore(altered)


@pytest.mark.parametrize("option", (
    dict(device="cuda"), dict(weights="cost"), dict(vce="robust"), dict(vce="cr0"),
    dict(cluster="subject"), dict(max_iterations=True), dict(max_iterations=0),
    dict(tolerance=float("nan")), dict(tolerance=1e-13), dict(level=True),
    dict(level=1), dict(max_work=0), dict(max_work=True),
    dict(fixed_dissimilarity=[("A", .0001)]), dict(fixed_dissimilarity={"A": 1.1}),
    dict(fixed_dissimilarity=[("A", .5), ("A", .8)]), dict(fixed_dissimilarity={"unknown": .4}),
))
def test_unsupported_options_fail_before_optimization(data, option, monkeypatch):
    monkeypatch.setattr(nl, "_fit", lambda p: pytest.fail("invalid options reached optimizer"))
    with pytest.raises(AnalysisError):
        nl.nlogit(data, "chosen", ["cost", "quality"], case="case", alternative="alternative", nest="nest", **option)


@pytest.mark.parametrize("mutation", (
    "missing_numeric", "boolean_attribute", "fractional_chosen", "no_chosen", "two_chosen",
    "chosen_unavailable", "duplicate_alternative", "alternative_changes_nest", "split_respondent",
    "one_available", "constant_attribute", "duplicate_attributes", "bad_identifier", "one_cluster",
))
def test_invalid_choice_geometry_rejected(data, mutation, monkeypatch):
    altered, kwargs = data.copy(), {}
    if mutation == "missing_numeric":
        altered.loc[0, "cost"] = np.nan
    elif mutation == "boolean_attribute":
        altered["cost"] = True
    elif mutation == "fractional_chosen":
        altered["chosen"] = altered.chosen.astype(float)
        altered.loc[0, "chosen"] = .5
    elif mutation == "no_chosen":
        altered.loc[altered.case == 0, "chosen"] = 0
    elif mutation == "two_chosen":
        altered.loc[altered.case == 0, "chosen"] = 1
    elif mutation == "chosen_unavailable":
        altered["available"] = 1
        altered.loc[(altered.case == 0) & (altered.chosen == 1), "available"] = 0
        kwargs["available"] = "available"
    elif mutation == "duplicate_alternative":
        altered.loc[1, "alternative"] = 0
    elif mutation == "alternative_changes_nest":
        altered.loc[0, "nest"] = "B"
    elif mutation == "split_respondent":
        altered.loc[0, "subject"] = 999
        kwargs |= dict(vce="cr0", cluster="subject")
    elif mutation == "one_available":
        altered["available"] = 1
        altered.loc[(altered.case == 0) & (altered.alternative > 0), "available"] = 0
        kwargs["available"] = "available"
    elif mutation == "constant_attribute":
        altered["cost"] = 1.0
    elif mutation == "duplicate_attributes":
        altered["quality"] = altered.cost
    elif mutation == "bad_identifier":
        altered["alternative"] = altered.alternative.astype(float)
        altered.loc[0, "alternative"] = .5
    else:
        altered["subject"] = 1
        kwargs |= dict(vce="cr0", cluster="subject")
    monkeypatch.setattr(nl, "_fit", lambda p: pytest.fail("invalid geometry reached optimizer"))
    with pytest.raises(AnalysisError):
        nl.nlogit(altered, "chosen", ["cost", "quality"], case="case", alternative="alternative", nest="nest", **kwargs)


def test_unidentified_free_dissimilarity_rejected_by_full_physical_information(data):
    altered = data.copy()
    altered["nest"] = "only"
    with pytest.raises(AnalysisError, match="No multistart"):
        fit(altered)


def single_active_choices(seed=711, cases=320):
    rng, rows = np.random.default_rng(seed), []
    for case in range(cases):
        X = rng.uniform(.2, 2., (4, 2))
        active = [0, 1] if case % 2 == 0 else [2, 3]
        lam = 1. if case % 2 == 0 else .52
        utility = X[active]@np.array([-.7, .5])/lam
        probability = np.exp(utility-utility.max())
        chosen = active[rng.choice(2, p=probability/probability.sum())]
        rows.extend([[case, j, "A" if j < 2 else "B", int(j == chosen), *X[j], int(j in active)] for j in range(4)])
    return pd.DataFrame(rows, columns=["case", "alternative", "nest", "chosen", "cost", "quality", "available"])


def test_single_active_cases_fixed_anchor_identify_shared_beta_and_free_lambda():
    data = single_active_choices()
    model = fit(data, available="available", fixed_dissimilarity={"A": 1.})
    f = state(model)["fit"]
    assert len(f["params"]) == 3
    assert f["lambdas"][0] == 1.
    assert .001 < f["lambdas"][1] < .999
    assert np.linalg.eigvalsh(matrix(model, "information")).min() > 0
    assert all(group.nonempty.sum() == 1 for _, group in model["nest_probabilities"].groupby("case"))
    assert nl.nlogit_restore(portable(model)).attrs == model.attrs


def test_single_active_cases_all_free_scales_are_unidentified():
    with pytest.raises(AnalysisError, match="No multistart"):
        fit(single_active_choices(), available="available")


def test_separation_does_not_become_tiny_score_false_convergence():
    rows = []
    for case in range(20):
        rows.extend([[case, 0, "A", 1, 1.0, float(case)], [case, 1, "B", 0, 0.0, float(case)+1]])
    data = pd.DataFrame(rows, columns=["case", "alternative", "nest", "chosen", "cost", "quality"])
    with pytest.raises(AnalysisError):
        fit(data)


def test_tail_likelihood_preserves_small_losing_score():
    data = pd.DataFrame([[0, "a", "A", 1, 1.], [0, "b", "B", 0, 0.],
                         [1, "a", "A", 1, 1.], [1, "b", "B", 0, 0.]], columns=["case", "alternative", "nest", "chosen", "x"])
    p = nl._prepare(data, nl._columns("chosen", ["x"], "case", "alternative", "nest"), nl._options())
    moments = nl._moments(torch.tensor([40.], dtype=torch.float64, device="cpu"), p)
    assert 0 < moments["gradient"][0] < 1e-15
    step = nl._inverse(moments["information"])@moments["gradient"]
    assert step[0] == pytest.approx(1., rel=1e-12)
    assert nl._stationarity(torch.tensor([40.], dtype=torch.float64, device="cpu"), p, moments)[2] is False


def test_upper_nest_tail_preserves_physical_dissimilarity_entropy_score():
    rows = []
    for case in (0, 1):
        rows += [[case, "a", "A", 0, 1.], [case, "b", "A", 0, 0.], [case, "c", "B", 1, 0.]]
    data = pd.DataFrame(rows, columns=["case", "alternative", "nest", "chosen", "x"])
    p = nl._prepare(data, nl._columns("chosen", ["x"], "case", "alternative", "nest"), nl._options())
    params = torch.tensor([50., .5], dtype=torch.float64, device="cpu")
    moments = nl._moments(params, p)
    t = math.exp(-100)
    entropy = math.log1p(t) + 100*t/(1+t)
    upper = 50 + .5*math.log1p(t)
    nest_probability = 1/(1+math.exp(-upper))
    assert moments["gradient"][1] == pytest.approx(-2*nest_probability*entropy, rel=2e-13, abs=0)
    components = nl._components(params.clone().requires_grad_(), p, 0)
    # The represented upper-utility value rounds to 50; its physical lambda
    # derivative must nevertheless retain the nonzero residual entropy.
    params_grad = params.clone().requires_grad_()
    upper_gradient = torch.autograd.grad(nl._components(params_grad, p, 0)["upper_utility"][0], params_grad)[0]
    assert upper_gradient[1] == pytest.approx(entropy, rel=2e-13, abs=0)
    assert components["group_nests"] == [0, 1]


def test_boundary_physical_gate_ignores_saturated_transform(models):
    _, p = nl._validate_state(models["oim"])
    params = torch.tensor(state(models["oim"])["fit"]["params"], dtype=torch.float64, device="cpu")
    params[-1] = .999999999
    with pytest.raises(AnalysisError, match="interior"):
        nl._stationarity(params, p, nl._moments(params, p))


def test_work_and_memory_rejected_before_model_tensor(data, monkeypatch):
    columns, options = nl._columns("chosen", ["cost", "quality"], "case", "alternative", "nest"), nl._options(max_work=1)
    monkeypatch.setattr(torch, "tensor", lambda *a, **k: pytest.fail("resource failure allocated tensor"))
    with pytest.raises(AnalysisError, match="work"):
        nl._prepare(data, columns, options)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        nl._prepare(data, columns, nl._options())


def test_explicit_cpu_and_float64_ignore_default_device_dtype(data):
    dtype = torch.get_default_dtype()
    with preserve_torch_default_device():
        try:
            torch.set_default_dtype(torch.float32)
            torch.set_default_device("meta")
            model = fit(data, fixed_dissimilarity={"A": 1., "B": 1.})
            restored = nl.nlogit_restore(portable(model))
            assert restored.attrs == model.attrs
            assert state(model)["settings"]["device"] == "cpu"
            assert state(model)["settings"]["precision"] == "float64"
        finally:
            torch.set_default_dtype(dtype)


def test_typed_nests_fixed_declarations_and_cases_survive_json(data):
    altered = data.copy()
    altered["case"] = altered.case.astype(object)
    altered.loc[altered.case == 1, "case"] = "1"
    altered.loc[altered.case == 0, "case"] = 1
    altered["nest"] = altered.nest.map({"A": 1, "B": "1"})
    model = fit(altered, fixed_dissimilarity=[(1, .5), ("1", .7)])
    assert state(model)["catalogue"]["nests"] == [1, "1"]
    assert state(model)["fixed_dissimilarity"] == [[1, .5], ["1", .7]]
    assert len(state(model)["fit"]["params"]) == 2
    restored = nl.nlogit_restore(portable(model))
    assert restored.attrs == model.attrs


def test_query_catalogue_preserves_fitted_empty_nest_and_allows_new_alternative(models, data):
    s = state(models["oim"])
    query = data[data.case == 0].drop(columns=["chosen", "subject"]).copy()
    query["available"] = query.nest == "A"
    query.loc[query.alternative == 1, "alternative"] = 1000
    columns = dict(s["columns"], available="available", cluster=None)
    p = nl._prepare(query, columns, s["options"], fixed_dissimilarity=s["fixed_dissimilarity"],
                    catalogue=s["catalogue"], require_chosen=False, for_fit=False)
    assert p.nest_labels == ["A", "B"]
    assert p.free_nests == [0, 1]
    components = nl._components(torch.tensor(s["fit"]["params"], dtype=torch.float64, device="cpu"), p, 0)
    assert components["group_nests"] == [0]
    assert components["group_probability"].tolist() == [1.]
    assert sum(components["probability"].tolist()) == pytest.approx(1.)


def test_query_known_alternative_cannot_switch_nest(models, data):
    s = state(models["oim"])
    query = data[data.case == 0].drop(columns=["chosen", "subject"]).copy()
    query.loc[0, "nest"] = "B"
    with pytest.raises(AnalysisError, match="same nest"):
        nl._prepare(query, dict(s["columns"], cluster=None), s["options"], fixed_dissimilarity=s["fixed_dissimilarity"],
                    catalogue=s["catalogue"], require_chosen=False, for_fit=False)
