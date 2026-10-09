"""Native joint Gaussian fitting, physical curvature and strict replay contracts."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.systems import nonlinear_sur as core
from openecon.resources import use_workspace_budget


@pytest.fixture(scope="module", autouse=True)
def native_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sample(n=96, seed=71):
    rng = np.random.default_rng(seed)
    x, z = rng.uniform(-1.2, 1.4, size=(2, n))
    noise = rng.multivariate_normal([0., 0.], [[.45, .18], [.18, .7]], n)
    frame = pd.DataFrame(dict(x=x, z=z, y1=2+.4*x+1.1*np.exp(.25)*x*x+noise[:, 0],
                              y2=2+.4*z+1.1*np.exp(-.25)*z*z+noise[:, 1], respondent=np.arange(n)//4))
    equations = [dict(y="y1", name="first", formula="{b0=2}+{b1=.4}*x+{amp=1.1}*exp({rate=.25})*x^2"),
                 dict(y="y2", name="second", formula="{b0=2}+{b1=.4}*z+{amp=1.1}*exp(-{rate=.25})*z^2")]
    return frame, equations


@pytest.fixture(scope="module")
def fitted():
    frame, equations = sample()
    return {kind: core.nlsur(frame, equations, covariance=kind, cluster="respondent" if kind == "cr0" else None)
            for kind in ("oim", "hc0", "cr0")}


def test_joint_shared_nonlinear_ml_scores_full_information_and_profile(fitted):
    for kind, result in fitted.items():
        state, p = core._validated(result)
        f = state["fit"]
        theta = np.array(f["params"])
        residual = np.array(f["residuals"])
        sigma = np.array(f["sigma"])
        np.testing.assert_allclose(sigma, residual.T@residual/p.n, rtol=1e-12)
        expected_ll = -.5*p.n*(p.m*(np.log(2*np.pi)+1)+np.linalg.slogdet(sigma)[1])
        assert f["log_likelihood"] == pytest.approx(expected_ll, rel=2e-14)
        np.testing.assert_allclose(np.array(f["case_scores"]).sum(0), f["gradient"], atol=2e-11)
        info, bread = np.array(f["information"]), np.array(f["bread"])
        np.testing.assert_allclose(info@bread, np.eye(len(theta)), rtol=2e-12, atol=2e-12)
        assert np.linalg.norm(info[:p.q, p.q:]) > .01
        assert f["parameter_names"] == ["b0", "b1", "amp", "rate", "cov__first__first", "cov__second__first", "cov__second__second"]
        if kind == "oim":
            np.testing.assert_array_equal(f["covariance"], f["bread"])
        else:
            scores = np.array(f["cluster_scores"] if kind == "cr0" else f["case_scores"])
            np.testing.assert_allclose(f["meat"], scores.T@scores, atol=1e-12)
            np.testing.assert_allclose(f["covariance"], bread@scores.T@scores@bread, rtol=2e-12, atol=1e-13)
        assert f["convergence"]["globally_certified"] is False
        assert len(f["convergence"]["starts"]) == 3
        assert f["stationarity"]["normalized_parameter_step"] < f["stationarity"]["step_limit"]
        assert "tabular" in result.to_latex()


def test_true_curvature_includes_nonlinear_residual_term(fitted):
    state, p = core._validated(fitted["oim"])
    physical = torch.tensor(state["fit"]["params"], dtype=core.DT)
    # This model reparameterizes a linear quadratic system. At its stationary
    # optimum the residual curvature vanishes; a displaced point exposes it.
    physical[0] += .2
    physical[2] += .1
    jac = torch.autograd.functional.jacobian(lambda t: core._means(t, p), physical)
    inv = torch.linalg.inv(core._sigma(physical, p))
    gn = torch.einsum('nai,ab,nbj->ij', jac, inv, jac)
    full = core._moments(physical, p)["information"]
    assert float((full[:p.q, :p.q]-gn[:p.q, :p.q]).abs().max()) > .01


def test_separate_linear_equations_reduce_to_multivariate_ols():
    rng = np.random.default_rng(91)
    n = 90
    x = rng.normal(size=n)
    design = np.column_stack([np.ones(n), x])
    y = design@np.array([[1., 2., 3.], [.4, -.2, .6]])+rng.multivariate_normal(np.zeros(3), [[.5, .1, -.1], [.1, .7, .2], [-.1, .2, .8]], n)
    frame = pd.DataFrame(dict(x=x, y1=y[:, 0], y2=y[:, 1], y3=y[:, 2]))
    eqs = [dict(y=f"y{i+1}", formula=f"{{a{i}}}+{{b{i}}}*x") for i in range(3)]
    result = core.nlsur(frame, eqs)
    fit = result.attrs["nonlinear_sur_state"]["fit"]
    beta = np.linalg.solve(design.T@design, design.T@y)
    expected = beta.T.ravel()
    np.testing.assert_allclose(fit["params"][:6], expected, atol=1e-8)
    residual = y-design@beta
    sigma = residual.T@residual/n
    np.testing.assert_allclose(fit["sigma"], sigma, atol=1e-11)
    np.testing.assert_allclose(np.array(fit["covariance"])[:6, :6], np.kron(sigma, np.linalg.inv(design.T@design)), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.array(fit["information"])[:6, 6:], 0., atol=2e-6)


@pytest.mark.parametrize("kind", ["oim", "hc0", "cr0"])
def test_restore_portable_all_tables_no_optimization_level_override(fitted, monkeypatch, kind):
    result = fitted[kind]
    before = copy.deepcopy(result.attrs)
    monkeypatch.setattr(core, "_fit", lambda *a: pytest.fail("restore optimized"))
    monkeypatch.setattr(core, "maximize_bfgs", lambda *a, **k: pytest.fail("restore optimized"))
    restored = core.nlsur_restore(json.loads(json.dumps(result.attrs, sort_keys=True)))
    assert set(restored) == set(result)
    for name in result:
        pd.testing.assert_frame_equal(result[name], restored[name])
        assert result[name].to_json(orient="split") == restored[name].to_json(orient="split")
        assert result[name].to_csv() == restored[name].to_csv()
    assert result.to_latex() == restored.to_latex()
    narrower = core.nlsur_restore(result, level=.8)
    np.testing.assert_array_equal(narrower["parameters"].estimate, result["parameters"].estimate)
    assert (narrower["parameters"].ci_upper-narrower["parameters"].ci_lower < result["parameters"].ci_upper-result["parameters"].ci_lower).all()
    assert result.attrs == before


@pytest.mark.parametrize("field", ["params", "sigma", "information", "bread", "meat", "covariance", "gradient", "case_scores", "cluster_scores", "case_loglikelihood", "fitted", "residuals", "mean_scale", "log_likelihood", "parameter_names", "stationarity", "n", "covariance_type"])
def test_resealed_scientific_moment_tamper_rejected(fitted, field):
    attrs = copy.deepcopy(fitted["cr0"].attrs)
    state = attrs["nonlinear_sur_state"]
    value = state["fit"][field]
    if field == "parameter_names":
        value[0] = "imposter"
    elif field == "stationarity":
        value["score_decrement"] += .2
    elif isinstance(value, list):
        if isinstance(value[0], list):
            value[0][0] += .05
        else:
            value[0] += .05
    elif isinstance(value, str):
        state["fit"][field] = "oim"
    else:
        state["fit"][field] += .05
    state["checksum"] = core._checksum(state)
    with pytest.raises(AnalysisError, match="Saved") as got:
        core.nlsur_restore(attrs)
    assert got.value.code == "invalid_result"


@pytest.mark.parametrize("where", ["inputs", "tapes", "equations", "settings", "index", "convergence", "schema", "checksum", "table"])
def test_sealed_metadata_and_public_table_tamper_rejected(fitted, where):
    result = copy.deepcopy(fitted["oim"])
    state = result.attrs["nonlinear_sur_state"]
    if where == "table":
        result["parameters"].iloc[0, 2] += 1
    elif where == "inputs":
        state["inputs"]["x"][0] += .1
    elif where == "tapes":
        state["tapes"][0]["tape"][0] = ["const", 5.]
    elif where == "equations":
        state["equations"][0]["formula"] = "{b0}+{b1}*x"
    elif where == "settings":
        state["settings"]["precision"] = "float32"
    elif where == "index":
        state["index"]["labels"][0] = '["int","1"]'
    elif where == "convergence":
        state["fit"]["convergence"]["selected_start"] = 5
    elif where == "schema":
        state["schema"] = "future"
    else:
        state["checksum"] = "0"*64
    if where not in {"checksum", "table"}:
        state["checksum"] = core._checksum(state)
    with pytest.raises(AnalysisError) as got:
        core.nlsur_restore(result)
    assert got.value.code == "invalid_result"


@pytest.mark.parametrize("formula", ["{a}+abs(x)", "{a}+tan(x)", "{a}+normal(x)", "{a}+expit(x)", "{a}+x^{b}", "{a}+x^.5", "{a}+x^13", "__import__('os')", "{a}+x.__class__", "{a}+sum([x])", "{a}+x[0]"])
def test_unsafe_nonsmooth_and_unsupported_powers_refused(formula):
    frame, eqs = sample()
    eqs[0]["formula"] = formula
    with pytest.raises(AnalysisError):
        core.nlsur(frame, eqs)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, None, True, "1"])
def test_complete_numeric_sample_never_silently_deleted(bad):
    frame, eqs = sample()
    frame["x"] = frame["x"].astype(object)
    frame.loc[0, "x"] = bad
    with pytest.raises(AnalysisError) as got:
        core.nlsur(frame, eqs)
    assert got.value.code == "missing_values"


def test_shared_starts_conflicts_outcome_rhs_and_role_validation():
    frame, eqs = sample()
    conflict = copy.deepcopy(eqs)
    conflict[1]["formula"] = conflict[1]["formula"].replace("{amp=1.1}", "{amp=1.2}")
    with pytest.raises(AnalysisError) as got:
        core.nlsur(frame, conflict)
    assert got.value.code == "invalid_start"
    for eq in (dict(y="y1", formula="{a}+y2"), dict(y="y1", formula="{a}+y1")):
        with pytest.raises(AnalysisError) as got:
            core.nlsur(frame, [eq, eqs[1]])
        assert got.value.code == "endogenous_rhs"
    for start in ({"unknown": 0}, {"b0": True}, {"b0": np.inf}):
        with pytest.raises(AnalysisError):
            core.nlsur(frame, eqs, start=start)
    # Explicit caller start values override defaults consistently in every equation.
    result = core.nlsur(frame, eqs, start={"b0": 2.1})
    assert result.attrs["nonlinear_sur_state"]["starts"]["b0"] == 2.1


@pytest.mark.parametrize("bad_options", [dict(covariance="HC0"), dict(covariance=[]), dict(covariance="cr0"), dict(cluster="respondent"), dict(tolerance=0), dict(max_iterations=True), dict(max_work=0), dict(max_bytes=0), dict(device="meta"), dict(level=1)])
def test_invalid_options_fail_cleanly(bad_options):
    frame, eqs = sample()
    with pytest.raises(AnalysisError):
        core.nlsur(frame, eqs, **bad_options)


@pytest.mark.parametrize("budget", ["max_work", "max_bytes", "global"])
def test_resource_guards_precede_tensor_or_derivative_allocation(monkeypatch, budget):
    frame, eqs = sample()
    monkeypatch.setattr(core.torch, "tensor", lambda *a, **k: pytest.fail("allocated before resource guard"))
    with pytest.raises(AnalysisError) as got:
        if budget == "global":
            with use_workspace_budget(1):
                core.nlsur(frame, eqs)
        else:
            core.nlsur(frame, eqs, **{budget: 100})
    assert got.value.code in {"work_budget", "workspace_limit"}


def test_exact_fit_singular_residual_and_unidentified_means_rejected():
    x = np.arange(25, dtype=float)
    frame = pd.DataFrame(dict(x=x, y1=1+2*x, y2=3-2*x))
    eqs = [dict(y="y1", formula="{a=1}+{b=2}*x"), dict(y="y2", formula="{c=3}-{b=2}*x")]
    with pytest.raises(AnalysisError) as got:
        core.nlsur(frame, eqs)
    assert got.value.code in {"rank_deficient", "no_finite_mle"}
    frame, eqs = sample()
    for eq in eqs:
        eq["formula"] += "+{unused}*0"
    with pytest.raises(AnalysisError) as got:
        core.nlsur(frame, eqs)
    assert got.value.code == "rank_deficient"


def test_typed_duplicate_multiindex_and_mapping_series_alignment():
    frame, eqs = sample(40)
    frame.index = pd.Index([1, "1", ("t", 2), None]*10, dtype=object, name="identity")
    result = core.nlsur(frame, eqs)
    for key in ("inputs", "fitted", "residuals", "case_scores", "case_likelihood"):
        assert result[key].index.equals(frame.index)
        assert core.nlsur_restore(result)[key].index.equals(frame.index)
    series = frame.to_dict("series")
    series["z"] = series["z"].iloc[::-1]
    with pytest.raises(AnalysisError) as got:
        core.nlsur(series, eqs)
    assert got.value.code == "shape_mismatch"
    frame.index = pd.MultiIndex.from_arrays([np.arange(40)//2, np.arange(40)%2], names=["group", "time"])
    result = core.nlsur(frame, eqs)
    assert core.nlsur_restore(result)["fitted"].index.equals(frame.index)


def test_native_cpu_float64_under_adversarial_defaults_and_inference(fitted):
    frame, eqs = sample()
    dtype, device = torch.get_default_dtype(), torch.get_default_device()
    grad, threads = torch.is_grad_enabled(), torch.get_num_threads()
    before = frame.copy(deep=True)
    try:
        torch.set_default_dtype(torch.float32)
        torch.set_default_device("meta")
        with torch.inference_mode():
            result = core.nlsur(frame, eqs)
            restored = core.nlsur_restore(result)
        np.testing.assert_allclose(result["parameters"].estimate, fitted["oim"]["parameters"].estimate, rtol=1e-12)
        pd.testing.assert_frame_equal(restored["covariance"], result["covariance"])
        assert torch.get_default_device().type == "meta"
        assert torch.get_default_dtype() == torch.float32
    finally:
        torch.set_default_dtype(dtype)
        torch.set_default_device(device)
    assert torch.is_grad_enabled() == grad and torch.get_num_threads() == threads
    pd.testing.assert_frame_equal(frame, before)


def test_saved_resource_context_is_portable_but_active_budget_still_guards():
    frame, equations = sample(48)
    with use_workspace_budget(32):
        result = core.nlsur(frame, equations)
    assert result.attrs["settings"]["resources"]["budget_bytes"] == 32*1024**2
    pd.testing.assert_frame_equal(core.nlsur_restore(result)["inputs"], result["inputs"])
    with use_workspace_budget(1), pytest.raises(AnalysisError) as got:
        core.nlsur_restore(result)
    assert got.value.code == "invalid_result"
