"""Independent Gaussian conditioning and complete physical delta oracles."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.systems import nonlinear_sur_postestimation as post
from openecon.resources import use_workspace_budget


def finite_jacobian(function, theta, step=2e-6):
    return np.stack([(function(theta + np.eye(len(theta))[j] * step)
                      - function(theta - np.eye(len(theta))[j] * step)) / (2 * step)
                     for j in range(len(theta))], axis=1)


def matrix(frame):
    return frame.iloc[:, 1:].to_numpy(dtype=float)


@pytest.fixture
def query(monkeypatch):
    from openecon.econometrics.systems import nonlinear_sur as fit

    labels = ["first", "second", "third"]
    index = pd.Index([1, "1", 1, ("g", 2)], dtype=object, name="typed")
    frame = pd.DataFrame({"x": [.2, .6, 1.1, .85], "z": [.35, .55, .95, .4],
                          "w": [.1, .4, .8, .2], "y1": [2.4, 3.1, 3.6, 2.9],
                          "y2": [2.8, 2.7, 3.2, 3.0], "y3": [2.9, 3.2, 3.0, 3.1]}, index=index)
    theta = torch.tensor([2., .7, .25, .3, .8, .25, .6, -.12, .11, .7], dtype=torch.float64)
    rng = np.random.default_rng(78)
    factor = rng.normal(size=(10, 10)) * np.array([.03, .02, .01, .02, .04, .025,
                                                 .035, .025, .025, .04])[:, None]
    covariance = torch.tensor(factor @ factor.T, dtype=torch.float64)
    names = ["a", "b", "c", "d"] + [f"cov__{labels[i]}__{labels[j]}"
                                         for i in range(3) for j in range(i + 1)]
    p = SimpleNamespace(q=4, m=3, n=4, mean_names=names[:4], parameter_names=names,
                        equations=[dict(name=name, y=f"y{i + 1}", formula="{a}+{b}*exp({c}*x)")
                                   for i, name in enumerate(labels)], columns=["x", "z", "w"],
                        x={name: torch.tensor(frame[name].to_numpy(), dtype=torch.float64)
                           for name in ["x", "z", "w"]},
                        y=torch.tensor(frame[["y1", "y2", "y3"]].to_numpy(), dtype=torch.float64),
                        index=index, options={"level": .95})
    state = dict(checksum="oracle-fixture", options={"level": .95},
                 fit=dict(params=theta.tolist(), covariance=covariance.tolist()))

    def means(t, prepared, x=None):
        x = prepared.x if x is None else x
        a, b, c, d = t[:4]
        return torch.stack([a + b * torch.exp(c * x["x"]) + d * x["z"],
                            a + b * torch.exp(-c * x["z"]) + d * x["x"],
                            a + b * x["x"] * x["z"] + d * torch.exp(c * x["w"])], dim=1)

    def sigma(t, prepared):
        return torch.stack([torch.stack([t[4], t[5], t[7]]),
                            torch.stack([t[5], t[6], t[8]]),
                            torch.stack([t[7], t[8], t[9]])])

    monkeypatch.setattr(fit, "_means", means)
    monkeypatch.setattr(fit, "_sigma", sigma)
    monkeypatch.setattr(post, "_saved", lambda result, level: (
        state, p, theta, covariance, *post._confidence(level, state)))
    return frame, p, theta.numpy(), covariance.numpy(), state


def oracle_means(t, frame):
    a, b, c, d = t[:4]
    x, z, w = (frame[name].to_numpy() for name in ["x", "z", "w"])
    return np.column_stack([a + b * np.exp(c*x) + d*z,
                            a + b * np.exp(-c*z) + d*x,
                            a + b*x*z + d*np.exp(c*w)])


def oracle_sigma(t):
    return np.array([[t[4], t[5], t[7]], [t[5], t[6], t[8]], [t[7], t[8], t[9]]])


def oracle_conditional(t, frame, observed):
    remaining = [j for j in range(3) if j not in observed]
    mu, sigma = oracle_means(t, frame), oracle_sigma(t)
    block = sigma[np.ix_(remaining, remaining)]
    means = mu[:, remaining]
    if observed:
        load = sigma[np.ix_(remaining, observed)] @ np.linalg.inv(sigma[np.ix_(observed, observed)])
        y = frame[[f"y{i + 1}" for i in observed]].to_numpy()
        means = means + (y - mu[:, observed]) @ load.T
        block = block - load @ sigma[np.ix_(observed, remaining)]
    return means, block


def oracle_targets(t, frame, observed):
    means, sigma = oracle_conditional(t, frame, observed)
    return np.r_[means.ravel(), [sigma[i, j] for i in range(len(sigma)) for j in range(i + 1)]]


@pytest.mark.parametrize("observed", [[], [0], [1], [2], [0, 2], [2, 0]])
def test_conditional_means_residual_and_all_physical_cross_deltas(query, observed):
    frame, p, theta, covariance, _ = query
    given = [p.equations[i]["name"] for i in observed]
    got = post.nlsur_predict(None, frame, given=given)
    expected = oracle_targets(theta, frame, observed)
    jacobian = finite_jacobian(lambda t: oracle_targets(t, frame, observed), theta)
    count = len(frame) * (3 - len(observed))
    np.testing.assert_allclose(got["means"].estimate, expected[:count], rtol=2e-15)
    np.testing.assert_allclose(got["residual_covariance"].estimate, expected[count:], rtol=2e-15)
    np.testing.assert_allclose(matrix(got["parameter_jacobian"]), jacobian, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(matrix(got["target_covariance"]), jacobian @ covariance @ jacobian.T,
                               rtol=4e-8, atol=2e-10)
    np.testing.assert_allclose(matrix(got["residual_variation"]),
                               oracle_conditional(theta, frame, observed)[1], rtol=2e-15)
    assert got["means"].index.equals(frame.index.repeat(3 - len(observed)))
    assert got.attrs["settings"]["no_optimizer"] is True
    assert got.attrs["settings"]["refit"] is False
    assert got.attrs["settings"]["row_index_codes"][0] != got.attrs["settings"]["row_index_codes"][1]
    if observed:
        assert np.any(np.abs(jacobian[:count, 4:]) > .01)
        assert np.any(np.abs(matrix(got["target_covariance"])[:count, count:]) > 1e-5)


def test_variance_ci_uses_log_delta_and_separates_random_variation(query):
    frame, _, theta, covariance, _ = query
    got = post.nlsur_predict(None, frame, given=["first"], level=.9)
    jac = finite_jacobian(lambda t: oracle_targets(t, frame, [0]), theta)
    v = jac @ covariance @ jac.T
    z = 1.6448536269514722
    for i, row in got["residual_covariance"].iterrows():
        if row.equation == row.other_equation:
            se = np.sqrt(v[8 + i, 8 + i])
            assert row.ci_low == pytest.approx(row.estimate * np.exp(-z*se/row.estimate), rel=3e-9)
            assert row.ci_high == pytest.approx(row.estimate * np.exp(z*se/row.estimate), rel=3e-9)
            assert "log variance" in row.inference
    assert not np.allclose(np.diag(v)[:2], np.diag(matrix(got["residual_variation"])))


def oracle_effects(t, frame, observed, variables, scale):
    _, b, c, d = t[:4]
    x, z, w = (frame[name].to_numpy() for name in ["x", "z", "w"])
    blocks = dict(x=np.column_stack([b*c*np.exp(c*x), np.full(len(frame), d), b*z]),
                  z=np.column_stack([np.full(len(frame), d), -b*c*np.exp(-c*z), b*x]),
                  w=np.column_stack([np.zeros(len(frame)), np.zeros(len(frame)), d*c*np.exp(c*w)]))
    remaining = [j for j in range(3) if j not in observed]
    sigma = oracle_sigma(t)
    load = (sigma[np.ix_(remaining, observed)] @ np.linalg.inv(sigma[np.ix_(observed, observed)])
            if observed else None)
    means = oracle_conditional(t, frame, observed)[0]
    effects = []
    for name in variables:
        effect = blocks[name][:, remaining]
        if observed:
            effect = effect - blocks[name][:, observed] @ load.T
        if scale == "elasticity":
            effect = effect * frame[name].to_numpy()[:, None] / means
        effects.append(effect)
    return np.stack(effects, axis=2)


@pytest.mark.parametrize("observed", [[], [0], [1, 2]])
@pytest.mark.parametrize("scale", ["effect", "elasticity"])
def test_exact_mixed_effects_fixed_weight_averages_full_joint_covariance(query, observed, scale):
    frame, p, theta, covariance, _ = query
    variables = ["x", "z", "w"]
    weights = np.array([1., 0., 3., 2.])
    got = post.nlsur_margins(None, frame, x=variables, scale=scale,
                             given=[p.equations[i]["name"] for i in observed], weights=weights)

    def oracle(t):
        effects = oracle_effects(t, frame, observed, variables, scale)
        averages = np.einsum("i,ijk->jk", weights / weights.sum(), effects)
        return np.r_[effects.ravel(), averages.ravel()]

    expected = oracle(theta)
    jac = finite_jacobian(oracle, theta)
    rows = got["effects" if scale == "effect" else "elasticities"]
    count = len(rows)
    np.testing.assert_allclose(rows.estimate, expected[:count], rtol=3e-15, atol=2e-16)
    np.testing.assert_allclose(got["averages"].estimate, expected[count:], rtol=3e-15, atol=2e-16)
    np.testing.assert_allclose(matrix(got["parameter_jacobian"]), jac, rtol=3e-7, atol=1e-9)
    np.testing.assert_allclose(matrix(got["target_covariance"]), jac @ covariance @ jac.T,
                               rtol=4e-7, atol=2e-11)
    assert rows.index.equals(frame.index.repeat((3-len(observed))*len(variables)))
    assert (got["averages"].query_rows == 4).all()
    assert (got["averages"].positive_weight_rows == 3).all()
    assert len(rows) == 4 * (3-len(observed)) * 3


def test_weight_rescaling_column_and_aligned_series_are_identical(query):
    frame, _, _, _, _ = query
    weights = [1., 0., 3., 2.]
    positional = post.nlsur_margins(None, frame, x="x", weights=weights)
    for value in [np.asarray(weights)*1e300, pd.Series(weights, index=frame.index), "fixed"]:
        current = post.nlsur_margins(None, frame.assign(fixed=weights), x="x", weights=value)
        np.testing.assert_array_equal(matrix(current["parameter_jacobian"]),
                                      matrix(positional["parameter_jacobian"]))
        np.testing.assert_array_equal(matrix(current["target_covariance"]),
                                      matrix(positional["target_covariance"]))


def test_saved_data_and_new_data_predictions_equal_and_no_cross_row_dependence(query):
    frame, _, _, _, _ = query
    original = post.nlsur_predict(None, frame, given=["first"])
    saved = post.nlsur_predict(None, given=["first"])
    np.testing.assert_array_equal(matrix(original["parameter_jacobian"]), matrix(saved["parameter_jacobian"]))
    changed = frame.copy()
    changed.iloc[3, changed.columns.get_loc("y1")] += 100
    current = post.nlsur_predict(None, changed, given=["first"])
    np.testing.assert_array_equal(original["means"].estimate.iloc[:6], current["means"].estimate.iloc[:6])
    np.testing.assert_array_equal(matrix(original["parameter_jacobian"])[:6],
                                  matrix(current["parameter_jacobian"])[:6])


@pytest.mark.parametrize("bad", [[0, 0, 0, 0], [1, -1, 3, 2], [1, 2], [1, np.nan, 1, 1],
                                 [True, True, True, True], {1: 2}])
def test_bad_weights_and_alignment_refused(query, bad):
    frame, _, _, _, _ = query
    with pytest.raises(AnalysisError):
        post.nlsur_margins(None, frame, x="x", weights=bad)
    with pytest.raises(AnalysisError, match="index"):
        post.nlsur_margins(None, frame, x="x", weights=pd.Series([1., 2., 3., 4.]))


@pytest.mark.parametrize("given", [["first", "first"], ["absent"], "first", ["first", "second", "third"]])
def test_bad_given_refused(query, given):
    with pytest.raises(AnalysisError):
        post.nlsur_predict(None, query[0], given=given)


@pytest.mark.parametrize("name,value,code", [("x", np.nan, "missing_values"),
                                            ("x", np.inf, "non_finite_values"),
                                            ("x", True, "non_numeric_column"),
                                            ("y1", "3", "non_numeric_column")])
def test_strict_query_numeric_domain(query, name, value, code):
    frame = query[0].copy()
    frame[name] = value
    with pytest.raises(AnalysisError) as error:
        post.nlsur_predict(None, frame, given=["first"])
    assert error.value.code == code


def test_missing_columns_duplicates_empty_and_strict_elasticity_support(query):
    frame, _, _, _, _ = query
    for bad in [frame.drop(columns="x"), frame.drop(columns="y1"), frame.iloc[:0],
                pd.concat([frame, frame[["x"]]], axis=1)]:
        with pytest.raises(AnalysisError):
            post.nlsur_predict(None, bad, given=["first"])
    for bad in [frame.assign(x=0.), frame.assign(y1=-100.)]:
        with pytest.raises(AnalysisError) as error:
            post.nlsur_margins(None, bad, x="x", given=["first"], scale="elasticity", weights=[0, 1, 1, 1])
        assert error.value.code == "no_support"
    with pytest.raises(AnalysisError):
        post.nlsur_margins(None, frame, x=["x", "x"])


def test_budgets_precede_target_evaluation(query, monkeypatch):
    from openecon.econometrics.systems import nonlinear_sur as fit

    monkeypatch.setattr(fit, "_means", lambda *a, **kw: pytest.fail("budget must reject before target"))
    frame = query[0]
    calls = [lambda: post.nlsur_predict(None, frame, max_work=1),
             lambda: post.nlsur_predict(None, frame, max_bytes=1),
             lambda: post.nlsur_margins(None, frame, x="x", max_work=1),
             lambda: post.nlsur_margins(None, frame, x="x", max_bytes=1)]
    for call in calls:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code in {"resource_budget", "workspace_limit"}
    with use_workspace_budget(1):
        large = pd.concat([frame] * 100)
        with pytest.raises(AnalysisError):
            post.nlsur_predict(None, large, max_work=10**12, max_bytes=10**12)


def test_joint_nonlinear_mean_and_covariance_contrasts_wald(query):
    _, _, theta, covariance, _ = query
    expressions = {"slope": "{b}", "variance": "{cov__second__second}-{cov__second__first}^2/{cov__first__first}",
                   "curvature": "{b}*exp({c})"}
    null = {"slope": .4, "variance": .5, "curvature": .7}
    got = post.nlsur_contrast(None, expressions, null=null)

    def oracle(t):
        return np.array([t[1], t[6]-t[5]**2/t[4], t[1]*np.exp(t[2])])

    jac = finite_jacobian(oracle, theta)
    variance = jac @ covariance @ jac.T
    difference = oracle(theta) - np.array(list(null.values()))
    wald = difference @ np.linalg.solve(variance, difference)
    np.testing.assert_allclose(got["contrasts"].estimate, oracle(theta), rtol=2e-15)
    np.testing.assert_allclose(matrix(got["parameter_jacobian"]), jac, rtol=2e-8, atol=1e-9)
    np.testing.assert_allclose(matrix(got["target_covariance"]), variance, rtol=3e-8, atol=1e-10)
    assert got["wald"].iloc[0].statistic == pytest.approx(wald, rel=4e-8)
    assert got["wald"].iloc[0].df == 3
    assert got.attrs["settings"]["null_feasibility"] == "not established by the delta method"


@pytest.mark.parametrize("expressions", [{"bad": "__import__('os')"}, {"bad": "{unknown}"},
                                         {"bad": "{b}*x"}, {"bad": "{b=1}"},
                                         {"bad": "abs({b})"}, {"bad": "{b}^.5"},
                                         {"bad": "invlogit({b})"}, {"bad": "sqrt(-{b})"}])
def test_contrast_hostile_unknown_nonsmooth_and_out_of_domain(query, expressions):
    with pytest.raises(AnalysisError):
        post.nlsur_contrast(None, expressions)


def test_contrast_rank_deficient_and_variance_boundary_refused(query):
    for expressions in [{"b": "{b}", "twice": "2*{b}"}, {"zero": "{b}-{b}"}]:
        with pytest.raises(AnalysisError) as error:
            post.nlsur_contrast(None, expressions)
        assert error.value.code == "nonestimable_restriction"
    with pytest.raises(AnalysisError) as error:
        post.nlsur_contrast(None, {"variance": "{cov__first__first}"})
    assert error.value.code == "nonregular_hypothesis"
    got = post.nlsur_contrast(None, {"variance": "{cov__first__first}"}, null=.5)
    assert got["wald"].iloc[0].df == 1
    with pytest.raises(AnalysisError):
        post.nlsur_contrast(None, {"b": "{b}"}, null={"different": 0})


def test_small_representable_exponential_derivatives_and_mixed_gradients(query, monkeypatch):
    from openecon.econometrics.systems import nonlinear_sur as fit

    frame, p, theta, covariance, state = query
    theta = torch.tensor(theta, dtype=torch.float64)
    theta[2] = -100.
    monkeypatch.setattr(post, "_saved", lambda result, level: (
        state, p, theta, torch.tensor(covariance), *post._confidence(level, state)))

    def means(t, p, x=None):
        return torch.stack([2 + t[1]*torch.exp(t[2]) * x["x"],
                            2 + t[1]*torch.exp(t[2]) * x["z"],
                            2 + t[1]*torch.exp(t[2]) * x["w"]], dim=1)

    monkeypatch.setattr(fit, "_means", means)
    got = post.nlsur_margins(None, frame, x="x")
    gradient = matrix(got["parameter_jacobian"])[0]
    assert got["effects"].iloc[0].estimate == pytest.approx(.7*np.exp(-100), rel=1e-15, abs=0)
    assert gradient[1] == pytest.approx(np.exp(-100), rel=1e-15, abs=0)
    assert gradient[2] == pytest.approx(.7*np.exp(-100), rel=1e-15, abs=0)
    assert got["effects"].iloc[0].std_error > 0


def test_settings_inputs_and_native_torch_context_are_unchanged(query):
    frame, _, _, _, state = query
    previous = copy.deepcopy(state)
    frame_before = frame.copy(deep=True)
    threads, dtype, grad = torch.get_num_threads(), torch.get_default_dtype(), torch.is_grad_enabled()
    with torch.inference_mode():
        got = post.nlsur_margins(None, frame, x="x", given=["first"])
    assert got["effects"].shape[0] == 8
    assert torch.get_num_threads() == threads
    assert torch.get_default_dtype() == dtype
    assert torch.is_grad_enabled() == grad
    assert state == previous
    pd.testing.assert_frame_equal(frame, frame_before)


def test_joint_tables_are_publication_exportable(query):
    frame = query[0]
    for got in [post.nlsur_predict(None, frame, given=["first"]),
                post.nlsur_margins(None, frame, x="x"),
                post.nlsur_contrast(None, {"b": "{b}"})]:
        latex = got.to_latex()
        assert "tabular" in latex
        assert "parameter\\_jacobian" in latex


@pytest.fixture(scope="module", params=["oim", "hc0", "cr0"])
def fitted(request):
    from openecon.econometrics.systems import nonlinear_sur as fit

    rng = np.random.default_rng(9182)
    n = 64
    frame = pd.DataFrame({"x": rng.uniform(.05, 1.3, n), "z": rng.uniform(.1, 1.5, n),
                          "group": np.repeat(np.arange(16), 4)})
    frame.index = pd.Index(["same"] * 2 + list(range(2, n)), dtype=object, name="subject")
    sigma = np.array([[.16, .06], [.06, .22]])
    error = rng.multivariate_normal([0, 0], sigma, n)
    frame["y1"] = 2 + .6*frame.x + .4*np.exp(.25)*frame.x**2 + error[:, 0]
    frame["y2"] = 2 + .6*frame.z + .4*np.exp(-.25)*frame.z**2 + error[:, 1]
    equations = [dict(name="first", y="y1", formula="{a}+{b}*x+{c}*exp({d})*x^2"),
                 dict(name="second", y="y2", formula="{a}+{b}*z+{c}*exp(-{d})*z^2")]
    result = fit.nlsur(frame, equations, start=dict(a=2, b=.6, c=.4, d=.25),
                        covariance=request.param, cluster="group" if request.param == "cr0" else None,
                        level=.9, max_work=300_000_000)
    return frame, result


def test_actual_fit_full_replay_query_restore_no_optimizer(fitted, monkeypatch):
    from openecon.econometrics.systems import nonlinear_sur as fit

    frame, result = fitted
    checksum = result.attrs["nonlinear_sur_state"]["checksum"]
    monkeypatch.setattr(fit, "_fit", lambda *a, **kw: pytest.fail("saved query must not optimize"))
    restored = fit.nlsur_restore(json.loads(json.dumps(result.attrs)))
    query = frame.iloc[:3]
    calls = [lambda saved: post.nlsur_predict(saved, query, given=["first"]),
             lambda saved: post.nlsur_margins(saved, query, x=["x", "z"], given=["first"],
                                               weights=[1, 2, 3]),
             lambda saved: post.nlsur_contrast(saved, {"slope": "{b}", "asymmetry": "2*{d}"})]
    for call in calls:
        original, replayed = call(result), call(restored)
        assert original.attrs["settings"]["level"] == .9
        for name in original:
            pd.testing.assert_frame_equal(original[name], replayed[name])
    assert result.attrs["nonlinear_sur_state"]["checksum"] == checksum


@pytest.mark.parametrize("path", [("fit", "covariance"), ("inputs", "x"), ("tapes",), ("checksum",)])
def test_actual_saved_tampering_rejected_by_every_public_query(fitted, path):
    _, result = fitted
    payload = json.loads(json.dumps(result.attrs))
    state = payload["nonlinear_sur_state"]
    if path == ("checksum",):
        state["checksum"] = "0" * 64
    elif path == ("tapes",):
        state["tapes"][0]["tape"][0][0] = "attribute_access"
    elif path[0] == "inputs":
        state["inputs"]["x"][0] += .1
    else:
        state["fit"]["covariance"][0][0] += .1
    for call in [lambda: post.nlsur_predict(payload),
                 lambda: post.nlsur_margins(payload, x="x"),
                 lambda: post.nlsur_contrast(payload, {"b": "{b}"})]:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == "invalid_result"


def test_actual_saved_multindex_and_intercept_only_query_length(monkeypatch):
    from openecon.econometrics.systems import nonlinear_sur as fit

    rng = np.random.default_rng(15)
    errors = rng.multivariate_normal([0., 0.], [[.4, .15], [.15, .6]], 24)
    index = pd.MultiIndex.from_tuples([(i//3, str(i % 3)) for i in range(24)], names=["group", "visit"])
    frame = pd.DataFrame({"y1": 2 + errors[:, 0], "y2": 3 + errors[:, 1]}, index=index)
    result = fit.nlsur(frame, [dict(name="first", y="y1", formula="{a}"),
                             dict(name="second", y="y2", formula="{b}")], start=dict(a=2., b=3.))
    monkeypatch.setattr(fit, "_fit", lambda *a, **kw: pytest.fail("saved query must not optimize"))
    saved = post.nlsur_predict(result)
    assert saved["means"].index.equals(index.repeat(2))
    query = frame.iloc[:2]
    current = post.nlsur_predict(result, query, given=["first"])
    assert len(current["means"]) == 2
    assert current["means"].index.equals(query.index)
    assert current.attrs["settings"]["row_index_names"] == ["group", "visit"]


def test_actual_query_override_level_preserves_sealed_fit(fitted):
    frame, result = fitted
    state = copy.deepcopy(result.attrs["nonlinear_sur_state"])
    query = post.nlsur_predict(result, frame.iloc[:2], level=.8)
    assert query.attrs["settings"]["level"] == .8
    assert result.attrs["nonlinear_sur_state"] == state


def test_positive_variance_interval_uses_log_coordinates_before_exponentiation():
    # exp(radius) alone would overflow even though the final variance limit is
    # representable. Units must not change an otherwise valid delta interval.
    got = post._interval(1e-160, 6.4e-315, 1., positive=True)
    assert np.isfinite(got["ci_high"]) and got["ci_high"] > 1e187
    assert np.log(got["ci_high"]) == pytest.approx(800 - 160*np.log(10), abs=1e-6)
    assert got["log_ci_low"] == pytest.approx(-800 - 160*np.log(10), abs=1e-6)
    assert got["ci_low"] == 0  # Its finite log endpoint remains explicitly retained.
