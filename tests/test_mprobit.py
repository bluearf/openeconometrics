"""Normalized Gaussian choice, complete physical inference and portable replay."""
import copy
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import erfcx, log_ndtr, ndtri

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.discrete import mprobit as core
from openecon.resources import use_workspace_budget


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample(cases=192, seed=270, binary_only=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(cases, 3, 2))
    covariance = np.array([[1., .2*1.25], [.2*1.25, 1.25**2]])
    errors = rng.multivariate_normal([0., 0.], covariance, cases)
    utility = X@np.array([.55, -.4])+np.column_stack([np.zeros(cases), errors])
    rows = []
    patterns = [[0, 1], [1, 2], [0, 2]] if binary_only else [[0, 1, 2], [0, 1], [1, 2], [0, 2]]
    for ci in range(cases):
        active = patterns[ci % len(patterns)]
        selected = active[int(utility[ci, active].argmax())]
        for j in range(3):
            rows.append([ci, "ABC"[j], int(j == selected), int(j in active), *X[ci, j], ci//4])
    return pd.DataFrame(rows, columns=["case", "alternative", "chosen", "available", "x", "z", "respondent"])


def fit(data, vce="oim", **kwargs):
    return core.mprobit(data, "chosen", ["x", "z"], case="case", alternative="alternative", alternatives=["A", "B", "C"],
                        available="available", vce=vce, cluster="respondent" if vce == "cr0" else None, **kwargs)


@pytest.fixture(scope="module")
def models():
    data = sample()
    return {kind: fit(data, kind) for kind in ("oim", "hc0", "cr0")}


def state(model):
    return model.attrs["mprobit_state"]


def portable(model):
    return json.loads(json.dumps(model.attrs, ensure_ascii=False, allow_nan=False, sort_keys=True))


def reseal(attrs):
    attrs["mprobit_state"]["checksum"] = core._checksum(attrs["mprobit_state"])
    attrs["settings"] = attrs["mprobit_state"]["settings"]
    return attrs


def exact_binary(cases=100):
    rows = []
    for ci in range(cases):
        selected = int(ci < .7*cases)
        for j in (0, 1):
            rows.append([ci, j, int(j == selected), float(j), ci//5])
    return pd.DataFrame(rows, columns=["case", "alternative", "chosen", "x", "respondent"])


def test_exact_binary_probit_reduction_full_oim():
    frame = exact_binary()
    result = core.mprobit(frame, "chosen", ["x"], case="case", alternative="alternative", alternatives=[0, 1])
    f = state(result)["fit"]
    beta = float(ndtri(.7))
    density = math.exp(-beta*beta/2)/math.sqrt(2*math.pi)
    information = 100*density*density/(.7*.3)
    assert f["parameter_names"] == ["x"]
    assert f["params"][0] == pytest.approx(beta, abs=2e-10)
    assert f["information"][0][0] == pytest.approx(information, rel=2e-11)
    assert f["covariance"][0][0] == pytest.approx(1/information, rel=2e-11)
    assert f["log_likelihood"] == pytest.approx(100*(.7*math.log(.7)+.3*math.log(.3)), abs=3e-12)
    assert state(result)["fixed_covariance"] == [[1.]]


def test_full_joint_geometry_and_complete_original_sample(models):
    for model in models.values():
        s, p = core._validated(model)
        f = s["fit"]
        assert len(model) == 14
        assert f["parameter_names"] == ["x", "z", "sd3", "rho3"]
        assert np.asarray(f["information"]).shape == (4, 4)
        assert np.linalg.norm(np.array(f["information"])[:2, 2:]) > 1
        np.testing.assert_allclose(np.sum(f["case_scores"], axis=0), f["gradient"], atol=2e-12)
        assert f["stationarity"]["normalized_parameter_step"] <= f["stationarity"]["step_limit"]
        assert f["convergence"]["globally_certified"] is False
        assert len(f["convergence"]["starts"]) == 3
        assert p.n == len(sample())
        assert len(model["inputs"]) == len(model["probabilities"]) == p.n
        for _, rows in model["probabilities"].groupby("case"):
            assert rows.probability.sum() == pytest.approx(1., abs=2e-13)
            assert (rows.loc[~rows.available, "probability"] == 0).all()
            assert rows.loc[~rows.available, "log_probability"].isna().all()


def test_whole_case_hc0_and_respondent_cr0_full_crossblocks(models):
    for kind, model in models.items():
        f = state(model)["fit"]
        score = np.asarray(f["cluster_scores"] if kind == "cr0" else f["case_scores"])
        bread = np.asarray(f["bread"])
        np.testing.assert_allclose(f["meat"], score.T@score, rtol=3e-13, atol=2e-12)
        expected = bread if kind == "oim" else bread@score.T@score@bread
        np.testing.assert_allclose(f["covariance"], expected, rtol=3e-12, atol=2e-14)
        np.testing.assert_allclose(bread@np.asarray(f["information"]), np.eye(4), rtol=3e-12, atol=2e-12)
        if kind == "cr0":
            scores = np.asarray(f["case_scores"])
            np.testing.assert_allclose(f["cluster_scores"], scores.reshape(-1, 4, 4).sum(1), rtol=2e-13, atol=2e-13)
        else:
            assert f["cluster_scores"] == []
    assert np.linalg.norm(np.array(state(models["hc0"])["fit"]["covariance"])[:2, 2:]) > 1e-4


def test_sorted_json_restores_every_table_csv_latex_without_optimizer(models, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Portable restore must not optimize")
    monkeypatch.setattr(core, "_fit", forbidden)
    monkeypatch.setattr(core, "maximize_bfgs", forbidden)
    for model in models.values():
        restored = core.mprobit_restore(portable(model))
        assert restored.attrs == model.attrs
        assert restored.to_latex() == model.to_latex()
        for name in model:
            assert restored[name].equals(model[name])
            assert restored[name].to_csv() == model[name].to_csv()


def test_changed_confidence_only_inference_preserves_moments(models):
    original = models["oim"]
    restored = core.mprobit_restore(portable(original), level=.8)
    assert state(restored)["fit"] == state(original)["fit"]
    assert restored["parameters"].ci_lower.iloc[0] > original["parameters"].ci_lower.iloc[0]
    assert restored["parameters"].ci_upper.iloc[0] < original["parameters"].ci_upper.iloc[0]
    assert restored["parameters"].z.iloc[2] != restored["parameters"].z.iloc[2]
    assert restored["parameters"].ci_lower.iloc[2] > 0
    assert restored["parameters"].ci_lower.iloc[3] > -1
    assert restored["parameters"].ci_upper.iloc[3] < 1


@pytest.mark.parametrize("name", ["params", "beta", "information", "bread", "meat", "covariance", "gradient", "case_scores", "cluster_scores", "case_loglikelihood", "normalized_covariance", "probabilities", "log_probabilities", "scale"])
def test_resealed_changed_complete_moments_rejected(models, name):
    attrs = portable(models["cr0"])
    value = attrs["mprobit_state"]["fit"][name]
    if isinstance(value[0], list):
        value[0][0] += .02
    else:
        value[0] += .02
    with pytest.raises(AnalysisError) as error:
        core.mprobit_restore(reseal(attrs))
    assert error.value.code == "invalid_result"


@pytest.mark.parametrize("where", ["checksum", "catalogue", "input", "settings", "schema", "solver", "start", "selected", "stationarity", "extra", "row_identity"])
def test_complete_semantic_saved_contract_rejects_tamper(models, where):
    attrs = portable(models["oim"])
    s = attrs["mprobit_state"]
    if where == "checksum":
        s["checksum"] = "0"*64
    elif where == "catalogue":
        s["catalogue"].reverse()
    elif where == "input":
        s["inputs"]["x"][0][0] += .2
    elif where == "settings":
        s["settings"]["difference_scale"] = 2.
    elif where == "schema":
        s["schema"] = "different"
    elif where == "solver":
        s["fit"]["solver"] = "unknown"
    elif where == "start":
        s["fit"]["convergence"]["starts"][0]["start"] = [1.5, .1]
    elif where == "selected":
        s["fit"]["convergence"]["selected_start"] = True
    elif where == "stationarity":
        s["fit"]["stationarity"]["physical_score_max"] += .01
    elif where == "extra":
        s["fit"]["unsupported"] = 1
    else:
        s["inputs"]["index"]["labels"][0] = core.encode("changed")
    if where != "checksum":
        reseal(attrs)
    with pytest.raises(AnalysisError) as error:
        core.mprobit_restore(attrs)
    assert error.value.code == "invalid_result"


def test_changed_result_table_is_not_trusted(models):
    changed = copy.deepcopy(models["oim"])
    changed["covariance"].iloc[0, 1] += .01
    with pytest.raises(AnalysisError, match="tables"):
        core.mprobit_restore(changed)


def test_original_multiindex_duplicate_and_mixed_typed_labels_preserved():
    data = exact_binary(40)
    data["alternative"] = data.alternative.map({0: 1, 1: "1"})
    data.index = pd.MultiIndex.from_tuples([(None if j % 3 else "dup", j % 5) for j in range(len(data))], names=["row", "part"])
    result = core.mprobit(data, "chosen", ["x"], case="case", alternative="alternative", alternatives=[1, "1"])
    restored = core.mprobit_restore(portable(result))
    assert state(result)["catalogue"] == [1, "1"]
    assert result["inputs"].index.equals(data.index)
    assert restored["inputs"].index.equals(data.index)
    assert restored.to_latex() == result.to_latex()


def test_utility_location_and_attribute_units_are_invariant(models):
    data = sample()
    shifted = data.copy()
    shifted["x"] += shifted.case*2.
    shifted["z"] -= shifted.case*.5
    shifted_result = fit(shifted)
    np.testing.assert_allclose(state(shifted_result)["fit"]["params"], state(models["oim"])["fit"]["params"], rtol=2e-8, atol=2e-9)
    scaled = data.copy()
    scaled.x *= 1e8
    scaled.z *= 1e-8
    result = fit(scaled)
    transformation = np.diag([1e8, 1e-8, 1., 1.])
    np.testing.assert_allclose(transformation@np.array(state(result)["fit"]["params"]), state(models["oim"])["fit"]["params"], rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(transformation@np.array(state(result)["fit"]["covariance"])@transformation, state(models["oim"])["fit"]["covariance"], rtol=2e-8, atol=2e-10)


def test_missing_reference_available_pair_uses_global_covariance(models):
    s, p = core._validated(models["oim"])
    theta = torch.tensor(s["fit"]["params"], dtype=core.DT, device="cpu")
    ci = next(j for j, rows in enumerate(p.case_rows) if {p.alt_codes[i] for i in rows if p.available[i]} == {1, 2})
    pieces = core._log_probabilities(theta, p, ci)
    rows = pieces["rows"]
    gamma = np.array(s["fit"]["normalized_covariance"])
    sd = math.sqrt(gamma[0, 0]+gamma[1, 1]-2*gamma[0, 1])
    delta = float((p.X[rows[0]]-p.X[rows[1]])@theta[:2])/sd
    np.testing.assert_allclose(pieces["log_probability"].detach(), [log_ndtr(delta), log_ndtr(-delta)], rtol=2e-13, atol=1e-14)


def test_fixed_covariance_and_all_binary_pair_identification():
    data = sample()
    fixed = fit(data, covariance=[[1., .25], [.25, 1.5625]])
    assert state(fixed)["fit"]["parameter_names"] == ["x", "z"]
    assert state(fixed)["fit"]["normalized_covariance"] == [[1., .25], [.25, 1.5625]]
    # Three separately observed binary pair types identify all physical
    # coordinates. Deterministic counts put the unconstrained optimum inside
    # the covariance domain, without selecting successful random samples.
    rows = []
    for group, (pair, proportion) in enumerate([("AB", .7), ("AC", .6), ("BC", .65)]):
        for ci in range(100):
            selected = int(ci < proportion*100)
            for j, label in enumerate(pair):
                rows.append([group*100+ci, label, int(j == selected), float(j)])
    binary = pd.DataFrame(rows, columns=["case", "alternative", "chosen", "x"])
    result = core.mprobit(binary, "chosen", ["x"], case="case", alternative="alternative", alternatives=["A", "B", "C"])
    beta, sd, rho = state(result)["fit"]["params"]
    expected_beta = float(ndtri(.7))
    expected_sd = expected_beta/float(ndtri(.6))
    expected_variance_bc = (expected_beta/float(ndtri(.65)))**2
    expected_rho = (1+expected_sd**2-expected_variance_bc)/(2*expected_sd)
    np.testing.assert_allclose([beta, sd, rho], [expected_beta, expected_sd, expected_rho], rtol=2e-8, atol=2e-9)
    assert np.linalg.eigvalsh(np.array(state(result)["fit"]["information"])).min() > 0


def test_only_one_pair_cannot_identify_three_alternative_free_covariance():
    data = exact_binary(40)
    data["alternative"] = data.alternative.map({0: "A", 1: "B"})
    with pytest.raises(AnalysisError) as error:
        core.mprobit(data, "chosen", ["x"], case="case", alternative="alternative", alternatives=["A", "B", "C"])
    assert error.value.code in {"no_finite_mle", "rank_deficient"}


@pytest.mark.parametrize("change", ["missing", "bool", "duplicate", "common", "rank", "two_chosen", "unavailable_chosen", "unknown", "split_cluster"])
def test_strict_sample_and_identification_refusals(change):
    data = sample(16)
    if change == "missing":
        data.loc[0, "x"] = np.nan
    elif change == "bool":
        data["x"] = True
    elif change == "duplicate":
        data.loc[1, "alternative"] = "A"
    elif change == "common":
        data["x"] = data.case.astype(float)
    elif change == "rank":
        data["z"] = data.x*2
    elif change == "two_chosen":
        data.loc[:2, "chosen"] = [1, 1, 0]
    elif change == "unavailable_chosen":
        data.loc[data.chosen.astype(bool), "available"] = 0
    elif change == "unknown":
        data.loc[0, "alternative"] = "D"
    else:
        data.loc[0, "respondent"] = 99
    with pytest.raises(AnalysisError):
        fit(data, "cr0" if change == "split_cluster" else "oim")


@pytest.mark.parametrize("kwargs", [dict(alternatives=["A", "A"]), dict(alternatives=["A"]), dict(alternatives=[True, "B"]), dict(covariance=[[2., 0.], [0., 1.]]), dict(covariance=[[1., .2], [.3, 1.]]), dict(covariance=[[1., 2.], [2., 1.]]), dict(covariance=[[1., 0.], [0., 0.]]), dict(covariance=[[1., 0.], [0., 401.]]), dict(covariance=[[1., 0.], [0., .001]]), dict(vce="HC0"), dict(weights="w"), dict(device="cuda"), dict(tolerance=np.nan), dict(max_iterations=True), dict(max_work=True), dict(max_bytes=0), dict(cluster="respondent")])
def test_invalid_model_options_rejected(kwargs):
    defaults = dict(case="case", alternative="alternative", alternatives=["A", "B", "C"], available="available")
    defaults.update(kwargs)
    with pytest.raises(AnalysisError):
        core.mprobit(sample(12), "chosen", ["x", "z"], **defaults)


def test_separation_no_finite_mle():
    data = exact_binary(40)
    data["chosen"] = (data.alternative == 1).astype(int)
    with pytest.raises(AnalysisError) as error:
        core.mprobit(data, "chosen", ["x"], case="case", alternative="alternative", alternatives=[0, 1])
    assert error.value.code in {"no_finite_mle", "rank_deficient", "boundary_solution"}


def test_budget_guards_before_derivative_construction(models, monkeypatch):
    monkeypatch.setattr(core, "_fit", lambda *args: pytest.fail("Budget refusal must precede fitting"))
    with pytest.raises(AnalysisError) as error:
        fit(sample(12), max_work=1)
    assert error.value.code == "work_budget"
    with use_workspace_budget(1):
        with pytest.raises(AnalysisError) as error:
            fit(sample())
    assert error.value.code == "workspace_limit"
    monkeypatch.setattr(core, "_moments", lambda *args: pytest.fail("Caller budget must precede numerical replay"))
    with pytest.raises(AnalysisError) as error:
        core._validated(models["oim"], budget_bytes=1)
    assert error.value.code == "workspace_limit"
    with pytest.raises(AnalysisError) as error:
        core._validated(models["oim"], max_work=1)
    assert error.value.code == "work_budget"


def test_native_explicit_cpu_float64_under_hostile_defaults(models):
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        result = fit(sample(), "hc0")
        restored = core.mprobit_restore(portable(result))
        np.testing.assert_allclose(state(result)["fit"]["params"], state(models["hc0"])["fit"]["params"], rtol=2e-11, atol=2e-11)
        assert restored.attrs == result.attrs
        _, p = core._validated(result)
        assert p.X.dtype == torch.float64 and p.X.device.type == "cpu"
    finally:
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)


def test_public_fit_and_complete_replay_under_inference_mode(models):
    with torch.inference_mode():
        result = fit(sample(), "hc0")
        restored = core.mprobit_restore(portable(models["hc0"]))
    np.testing.assert_allclose(state(result)["fit"]["params"], state(models["hc0"])["fit"]["params"], rtol=1e-12, atol=1e-12)
    assert restored.attrs == models["hc0"].attrs


def test_restore_honors_current_workspace_and_preserves_original_plan(models):
    with use_workspace_budget(64):
        restored = core.mprobit_restore(portable(models["oim"]))
    assert restored.attrs == models["oim"].attrs


@pytest.mark.parametrize("rho", [-.97, -.5, 0., .6, .97])
def test_cdf_zero_threshold_exact_value_and_three_correlation_derivatives(rho):
    r = torch.tensor(rho, dtype=core.DT, device="cpu", requires_grad=True)
    value = core._log_cdf(r*0, r*0, r)
    first = torch.autograd.grad(value, r, create_graph=True)[0]
    second = torch.autograd.grad(first, r, create_graph=True)[0]
    third = torch.autograd.grad(second, r)[0]
    probability = .25+math.asin(rho)/(2*math.pi)
    d1 = 1/(2*math.pi*math.sqrt(1-rho*rho))
    d2 = rho/(2*math.pi*(1-rho*rho)**1.5)
    d3 = (1+2*rho*rho)/(2*math.pi*(1-rho*rho)**2.5)
    expected = [math.log(probability), d1/probability, d2/probability-(d1/probability)**2,
                d3/probability-3*d2*d1/probability**2+2*(d1/probability)**3]
    np.testing.assert_allclose([float(v.detach()) for v in (value, first, second, third)], expected, rtol=2e-10, atol=2e-12)


@pytest.mark.parametrize("a,b,rho", [(.2, .5, .3), (-7., -5., -.5), (-20., -18., .9), (9., 10., -.6), (.3, -.8, .97)])
def test_cdf_second_and_third_chains_match_derivative_of_boundary_identity(a, b, rho):
    point = torch.tensor([a, b, rho], dtype=core.DT, device="cpu", requires_grad=True)
    def fn(value):
        return core._log_cdf(value[0], value[1], value[2])
    hessian = torch.autograd.functional.hessian(fn, point, create_graph=True)
    third = torch.stack([torch.stack([torch.autograd.grad(hessian[i, j], point, retain_graph=True)[0] for j in range(3)]) for i in range(3)])
    # Third chains must be symmetric even in difficult tail/high-rho geometry.
    torch.testing.assert_close(third, third.permute(2, 1, 0), rtol=3e-9, atol=3e-9)
    delta = 2e-5
    finite = []
    for j in range(3):
        step = point.detach().new_zeros(3)
        step[j] = delta
        finite.append((torch.autograd.functional.hessian(fn, point.detach()+step)-torch.autograd.functional.hessian(fn, point.detach()-step))/(2*delta))
    torch.testing.assert_close(third.detach(), torch.stack(finite, dim=2), rtol=8e-5, atol=2e-5)
    if rho == .3:
        assert torch.autograd.gradcheck(fn, (point,), eps=1e-5, atol=1e-6)
        assert torch.autograd.gradgradcheck(fn, (point,), eps=1e-5, atol=1e-6)


def test_cdf_scalar_correlation_broadcast_gradient_sums_all_events():
    a = torch.tensor([-.5, .2, 1.], dtype=core.DT, device="cpu", requires_grad=True)
    b = torch.tensor([.7, -.3, .9], dtype=core.DT, device="cpu", requires_grad=True)
    rho = torch.tensor(.25, dtype=core.DT, device="cpu", requires_grad=True)
    log = core._log_cdf(a, b, rho)
    actual = torch.autograd.grad(log.sum(), rho)[0]
    expected = sum(torch.autograd.grad(core._log_cdf(a[j], b[j], rho), rho, retain_graph=True)[0] for j in range(3))
    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=1e-14)


def test_cdf_positive_and_negative_tails_keep_finite_log_and_nonzero_gradient():
    point = torch.tensor([30., 31., .2], dtype=core.DT, device="cpu", requires_grad=True)
    gradient = torch.autograd.grad(core._log_cdf(*point), point)[0]
    assert bool((gradient[:2] > 0).all())
    assert float(gradient[0]) < 1e-190
    point = torch.tensor([-120., -100., -.3], dtype=core.DT, device="cpu", requires_grad=True)
    value = core._log_cdf(*point)
    gradient = torch.autograd.grad(value, point, create_graph=True)[0]
    assert float(value) < -10000
    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.isfinite(torch.autograd.functional.hessian(lambda x: core._log_cdf(*x), point)).all())


def test_public_binary_far_tail_refuses_probability_effect_elasticity_before_cdf(monkeypatch):
    from openecon.econometrics.discrete.mprobit_postestimation import mprobit_margins, mprobit_predict
    data = exact_binary()
    fitted = core.mprobit(data, "chosen", ["x"], case="case", alternative="alternative", alternatives=[0, 1])
    beta = state(fitted)["fit"]["params"][0]
    query = pd.DataFrame(dict(case=["q", "q"], alternative=[0, 1], x=[30000., 30000.-10000./beta]))
    native = torch.special.log_ndtr
    def bounded(value):
        assert bool((value.detach().abs() <= core.MAX_STANDARDIZED).all()), "Out-of-domain values reached the CDF"
        return native(value)
    monkeypatch.setattr(torch.special, "log_ndtr", bounded)
    for function in (lambda: mprobit_predict(fitted, query, kind="log_probability"),
                     lambda: mprobit_margins(fitted, query, targets=[dict(outcome=1, changed=1, attribute="x")]),
                     lambda: mprobit_margins(fitted, query, targets=[dict(outcome=1, changed=1, attribute="x")], elasticity=True)):
        with pytest.raises(AnalysisError) as error:
            function()
        assert error.value.code == "numerical_domain"


def test_binary_near_supported_edge_has_stable_erfcx_score_curvature_and_public_delta():
    from openecon.econometrics.discrete.mprobit_postestimation import mprobit_margins, mprobit_predict
    fitted = core.mprobit(exact_binary(), "chosen", ["x"], case="case", alternative="alternative", alternatives=[0, 1])
    s = state(fitted)
    beta = s["fit"]["params"][0]
    a = 15.99
    X = [300., 300.-a/beta]
    query = pd.DataFrame(dict(case=["q", "q"], alternative=[0, 1], x=X))
    p = core._prepare(query, s["columns"], s["options"], catalogue=s["catalogue"], fixed_covariance=s["fixed_covariance"], require_chosen=False, for_fit=False)
    theta = torch.tensor([beta], dtype=core.DT, device="cpu", requires_grad=True)
    delta = X[1]-X[0]
    log = core._log_probabilities(theta, p, 0)["log_probability"][1]
    gradient = torch.autograd.grad(log, theta, create_graph=True)[0]
    hessian = torch.autograd.grad(gradient[0], theta)[0]
    mills = math.sqrt(2/math.pi)/float(erfcx(a/math.sqrt(2)))
    curvature = -1+1/a**2-6/a**4+50/a**6-518/a**8+6354/a**10
    assert float(gradient.detach()[0]) == pytest.approx(delta*mills, rel=1e-11)
    assert float(hessian.detach()[0]) == pytest.approx(delta*delta*curvature, rel=2e-8)
    prediction = mprobit_predict(fitted, query, targets=[dict(case="q", alternative=1)], kind="log_probability")
    assert prediction["predictions"].estimate.iloc[0] == pytest.approx(float(log.detach()), abs=1e-10)
    elasticity = mprobit_margins(fitted, query, targets=[dict(outcome=1, changed=1, attribute="x")], elasticity=True)
    expected_jacobian = X[1]*(mills+beta*delta*curvature)
    actual_jacobian = float(elasticity["jacobian"].iloc[0, 1])
    assert actual_jacobian == pytest.approx(expected_jacobian, rel=2e-8)
    expected_variance = expected_jacobian**2*s["fit"]["covariance"][0][0]
    assert float(elasticity["covariance"].iloc[0, 1]) == pytest.approx(expected_variance, rel=4e-8)


def test_bivariate_near_singular_conditional_threshold_refuses_before_cdf(monkeypatch):
    from openecon.econometrics.discrete.mprobit_postestimation import mprobit_predict
    data = exact_binary()
    data["alternative"] = data.alternative.map({0: "A", 1: "B"})
    fitted = core.mprobit(data, "chosen", ["x"], case="case", alternative="alternative", alternatives=["A", "B", "C"], covariance=[[1., -.98], [-.98, 1.]])
    beta = state(fitted)["fit"]["params"][0]
    query = pd.DataFrame(dict(case=["q"]*3, alternative=["A", "B", "C"], x=[100., 100.+2/beta, 100.+2/beta]))
    def forbidden(*args):
        pytest.fail("Conditional out-of-domain values reached bivariate quadrature")
    monkeypatch.setattr(core, "_log_cdf", forbidden)
    with pytest.raises(AnalysisError) as error:
        mprobit_predict(fitted, query, targets=[dict(case="q", alternative="A")])
    assert error.value.code == "numerical_domain"
    assert "Conditional" in str(error.value)


def test_optimizer_backtracks_only_numerical_domain_trials(monkeypatch):
    data = exact_binary()
    p = core._prepare(data, core._columns("chosen", ["x"], "case", "alternative"), core._options(), catalogue=[0, 1])
    objective = core._Objective(p)
    point = torch.tensor([10000.], dtype=core.DT, device="cpu", requires_grad=True)
    value, gradient = objective(point)
    assert float(value) == -math.inf
    assert float(gradient[0]) == 0.
    def other_failure(*args):
        raise AnalysisError("rank_deficient", "A different scientific refusal")
    monkeypatch.setattr(core, "_case_logs", other_failure)
    with pytest.raises(AnalysisError) as error:
        objective.value(point)
    assert error.value.code == "rank_deficient"


def test_public_positive_tail_rescued_by_large_attribute_explicitly_refused():
    from openecon.econometrics.discrete.mprobit_postestimation import mprobit_margins
    fitted = core.mprobit(exact_binary(), "chosen", ["x"], case="case", alternative="alternative", alternatives=[0, 1])
    beta = state(fitted)["fit"]["params"][0]
    X = [1e12-100., 1e12-100.+38.7/beta]
    query = pd.DataFrame(dict(case=["q", "q"], alternative=[0, 1], x=X))
    actual_z = beta*(X[1]-X[0])
    # A separately evaluated phi(z) is zero, but multiplying in log space
    # shows that a large raw attribute would rescue a representable target.
    assert math.exp(-actual_z**2/2)/math.sqrt(2*math.pi) == 0.
    rescued = math.exp(math.log(X[1]*beta)-actual_z**2/2-.5*math.log(2*math.pi))
    assert rescued > 0.
    with pytest.raises(AnalysisError) as error:
        mprobit_margins(fitted, query, targets=[dict(outcome=1, changed=1, attribute="x")], elasticity=True)
    assert error.value.code == "numerical_domain"
