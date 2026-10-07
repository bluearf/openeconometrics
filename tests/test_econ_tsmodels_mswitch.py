"""oe.mswitch against statsmodels MarkovRegression / MarkovAutoregression and a NumPy filter."""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.tsmodels import mswitch_kernels as mk
from openecon.engines.optimize import check_derivatives

warnings.filterwarnings("ignore", module="statsmodels")


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(3)
    n = 300
    s = np.zeros(n, int)
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < 0.92 else 1 - s[t - 1]
    x = rng.normal(size=n)
    w = rng.normal(size=n)
    e = rng.normal(size=n) * np.where(s == 0, 1.0, 1.5)
    y = np.zeros(n)
    mu = np.where(s == 0, 1.0, 4.0) + 0.3 * w
    for t in range(1, n):
        y[t] = mu[t] + 0.4 * (y[t - 1] - mu[t - 1]) + np.where(s[t] == 0, 0.5, -0.5) * x[t] + e[t]
    return pd.DataFrame({"y": y, "x": x, "w": w, "t": np.arange(n) + 1})


def tensors(df, switching, common):
    n = len(df)
    columns = {"Intercept": np.ones(n), **{c: df[c].to_numpy() for c in ("x", "w")}}
    xs = torch.tensor(np.column_stack([columns[c] for c in switching]) if switching
                      else np.zeros((n, 0)))
    xc = torch.tensor(np.column_stack([columns[c] for c in common]) if common else np.zeros((n, 0)))
    return torch.tensor(df.y.to_numpy()), xs, xc


LAYOUTS = [
    (mk.Layout(2, [], 2, 0, False, True), ["Intercept", "x"], []),
    (mk.Layout(3, [], 1, 2, False, False), ["Intercept"], ["x", "w"]),
    (mk.Layout(2, [1], 2, 1, True, True), ["Intercept", "x"], ["w"]),
    (mk.Layout(2, [1, 3], 1, 0, False, False), ["Intercept"], []),
]


@pytest.mark.parametrize("layout,switching,common", LAYOUTS)
def test_analytic_gradient(df, layout, switching, common):
    y, xs, xc = tensors(df, switching, common)
    rng = np.random.default_rng(1)
    theta = torch.tensor(rng.normal(scale=0.3, size=layout.size))
    theta[:layout.states * layout.switching] += torch.linspace(0.5, 3.5, layout.states * layout.switching)
    result = check_derivatives(lambda t: (mk.score(layout, t, y, xs, xc)["loglik"],
                                          mk.score(layout, t, y, xs, xc)["gradient"]), theta)
    assert result["gradient_max_rel_error"] < 1e-7


def numpy_hamilton(log_density, a, start):
    """Plain sequential Hamilton filter and Kim smoother."""
    n, K = log_density.shape
    pred, filt, ll = np.zeros((n, K)), np.zeros((n, K)), 0.0
    xi = start
    for t in range(n):
        pred[t] = a.T @ xi
        joint = np.exp(log_density[t]) * pred[t]
        ll += np.log(joint.sum())
        xi = joint / joint.sum()
        filt[t] = xi
    smooth = np.zeros((n, K))
    smooth[-1] = filt[-1]
    for t in range(n - 2, -1, -1):
        ratio = np.divide(smooth[t + 1], pred[t + 1], out=np.zeros(K), where=pred[t + 1] > 0)
        smooth[t] = filt[t] * (a @ ratio)
    return pred, filt, smooth, ll


@pytest.mark.parametrize("layout,switching,common", LAYOUTS)
def test_blocked_filter_and_smoother_match_sequential(df, layout, switching, common):
    y, xs, xc = tensors(df, switching, common)
    theta = torch.tensor(np.random.default_rng(2).normal(scale=0.3, size=layout.size))
    parts = layout.split(theta)
    p = mk.transition(parts["logit"])
    pi, _ = mk.ergodic(p)
    a, start = mk.expanded_chain(layout, p, pi)
    assert_allclose(a.sum(1), 1.0)
    assert_allclose((start @ a).numpy(), start.numpy(), atol=1e-14)        # stationary
    log_density = mk.densities(layout, theta, y, xs, xc).log_density
    run = mk.hamilton_filter(log_density, a, start)
    pred, filt, smooth, ll = numpy_hamilton(log_density.numpy(), a.numpy(), start.numpy())
    assert_allclose(run["predicted"], pred, atol=1e-12)
    assert_allclose(run["filtered"], filt, atol=1e-12)
    assert_allclose(float(run["loglik"].sum()), ll, rtol=1e-12)
    assert_allclose(mk.kim_smoother(run["filtered"], run["predicted"], a), smooth, atol=1e-11)


def sm_values(model, fit, exog=("x",)):
    """Our estimates in statsmodels' parameterization (transition probabilities, sigma^2)."""
    est = {c.term: c.estimate for c in fit.coefficients}
    k = int(fit.metrics["states"])
    p = np.array([[cell["probability"] for cell in row] for row in fit.extra["transition_matrix"]])
    out = []
    for name in model.param_names:
        if name.startswith("p["):
            i, j = map(int, name[2:-1].split("->"))
            out.append(p[i, j])
        elif name.startswith("sigma2"):
            key = "/lnsigma" if name == "sigma2" else f"/lnsigma{int(name[-2]) + 1}"
            out.append(np.exp(2 * est[key]))
        elif name.startswith("ar.L"):
            lag = name.split(".L")[1].split("[")[0]
            key = f"L{lag}.ar" if "[" not in name else f"state{int(name[-2]) + 1}:L{lag}.ar"
            out.append(est[key])
        else:
            base, _, state = name.partition("[")
            names = {"const": "Intercept", **{f"x{i + 1}": c for i, c in enumerate(exog)}}
            base = names.get(base, base)
            key = f"state{int(state[:-1]) + 1}:{base}" if state else base
            out.append(est[key] if key in est else est[base])
    assert len(out) == len(model.param_names) and k >= 2
    return np.array(out)


def test_dynamic_regression_matches_statsmodels(df):
    fit = oe.mswitch(data=df, y="y", x=["x"], switch=["Intercept", "x"], varswitch=True, time="t")
    model = sm.tsa.MarkovRegression(df.y.to_numpy(), 2, exog=df[["x"]].to_numpy(),
                                    switching_variance=True)
    reference = model.fit(search_reps=10, cov_type="approx")
    ours = sm_values(model, fit)
    assert_allclose(model.loglike(ours), fit.metrics["log_likelihood"], rtol=1e-10)
    assert fit.metrics["log_likelihood"] >= reference.llf - 1e-6
    assert_allclose(fit.metrics["log_likelihood"], reference.llf, rtol=1e-8)
    assert_allclose(ours, reference.params, rtol=1e-4, atol=1e-5)
    # Standard errors: identical parameterization for the regression coefficients,
    # delta method for the transition probabilities.
    ses = {c.term: c.std_error for c in fit.coefficients}
    names = model.param_names
    assert_allclose(ses["state1:Intercept"], reference.bse[names.index("const[0]")], rtol=5e-3)
    assert_allclose(ses["state2:x"], reference.bse[names.index("x1[1]")], rtol=5e-3)
    assert_allclose(fit.extra["transition_matrix"][0][0]["std_error"],
                    reference.bse[names.index("p[0->0]")], rtol=5e-3)
    sigma = fit.extra["sigma"][1]
    assert_allclose(2 * sigma["sigma"] * sigma["std_error"], reference.bse[names.index("sigma2[1]")],
                    rtol=5e-3)
    probs = oe.mswitch_probabilities(fit, df)
    at_ours = model.smooth(ours)
    assert_allclose(probs[["smoothed_state1", "smoothed_state2"]],
                    at_ours.smoothed_marginal_probabilities, atol=1e-9)
    assert_allclose(probs[["filtered_state1", "filtered_state2"]],
                    at_ours.filtered_marginal_probabilities, atol=1e-9)
    assert_allclose(probs.filter(like="smoothed").sum(1), 1.0)
    assert fit.coefficients[0].estimate < fit.coefficients[2].estimate           # state order
    assert_allclose(fit.metrics["duration_state1"],
                    1 / (1 - fit.extra["transition_matrix"][0][0]["probability"]))


def test_autoregression_matches_statsmodels(df):
    fit = oe.mswitch(data=df, y="y", x=["w"], ar=1, time="t")
    model = sm.tsa.MarkovAutoregression(df.y.to_numpy(), 2, 1, exog=df[["w"]].to_numpy(),
                                        switching_exog=False, switching_ar=False)
    ours = sm_values(model, fit, exog=("w",))
    assert_allclose(model.loglike(ours), fit.metrics["log_likelihood"], rtol=1e-10)
    reference = model.fit(search_reps=10)
    assert fit.metrics["log_likelihood"] >= reference.llf - 1e-6
    assert fit.nobs == len(df) - 1
    probs = oe.mswitch_probabilities(fit, df)
    assert len(probs) == len(df) - 1
    assert_allclose(probs[["smoothed_state1", "smoothed_state2"]],
                    model.smooth(ours).smoothed_marginal_probabilities, atol=1e-8)


def test_switching_ar_and_three_states_likelihoods(df):
    """Log likelihood at fixed parameters for richer layouts.

    statsmodels (0.14) indexes a switching variance by the regime of t-1 instead of t
    when the AR order is 2 or more (it reshapes the variance to (k, 1, 1) against
    residuals of shape (k,)*(order+1) + (n,)), so the AR(2) oracle uses a common variance.
    """
    rng = np.random.default_rng(4)
    model = sm.tsa.MarkovAutoregression(df.y.to_numpy(), 2, 2, switching_ar=True,
                                        switching_variance=False)
    params = model.start_params + rng.normal(scale=0.05, size=len(model.start_params))
    names = model.param_names
    layout = mk.Layout(2, [1, 2], 1, 0, True, False)
    p = np.array([[params[names.index("p[0->0]")], 1 - params[names.index("p[0->0]")]],
                  [params[names.index("p[1->0]")], 1 - params[names.index("p[1->0]")]]])
    theta = np.concatenate([
        [params[names.index("const[0]")], params[names.index("const[1]")]],
        [params[names.index("ar.L1[0]")], params[names.index("ar.L2[0]")],
         params[names.index("ar.L1[1]")], params[names.index("ar.L2[1]")]],
        [0.5 * np.log(params[names.index("sigma2")])],
        np.log(p[:, 0] / p[:, 1])])
    y, xs, xc = tensors(df, ["Intercept"], [])
    ours = float(mk.score(layout, torch.tensor(theta), y, xs, xc)["loglik"])
    assert_allclose(ours, model.loglike(params), rtol=1e-10)
    three = sm.tsa.MarkovRegression(df.y.to_numpy(), 3, exog=df[["x"]].to_numpy(),
                                    switching_exog=False)
    params = three.start_params
    names = three.param_names
    P = np.zeros((3, 3))
    for i in range(3):
        for j in range(2):
            P[i, j] = params[names.index(f"p[{i}->{j}]")]
        P[i, 2] = 1 - P[i, :2].sum()
    layout = mk.Layout(3, [], 1, 1, False, False)
    theta = np.concatenate([[params[names.index(f"const[{s}]")] for s in range(3)],
                            [params[names.index("x1[2]")]],
                            [0.5 * np.log(params[names.index("sigma2")])],
                            np.log(P[:, :2] / P[:, 2:]).reshape(-1)])
    y, xs, xc = tensors(df, ["Intercept"], ["x"])
    ours = float(mk.score(layout, torch.tensor(theta), y, xs, xc)["loglik"])
    assert_allclose(ours, three.loglike(params), rtol=1e-10)


def test_score_covariances(df):
    base = oe.mswitch(data=df, y="y", x=["x"], varswitch=True)
    model = sm.tsa.MarkovRegression(df.y.to_numpy(), 2, exog=df[["x"]].to_numpy(),
                                    switching_exog=False, switching_variance=True)
    ours = sm_values(model, base)
    names = model.param_names
    for cov, sm_cov, factor in [("opg", "opg", 1.0), ("robust", "robust", len(df) / (len(df) - 1))]:
        fit = oe.mswitch(data=df, y="y", x=["x"], varswitch=True, covariance=cov)
        reference = model.smooth(ours, cov_type=sm_cov)
        se = {c.term: c.std_error for c in fit.coefficients}
        assert_allclose(se["x"], reference.bse[names.index("x1[1]")] * np.sqrt(factor), rtol=5e-3)
        assert_allclose(se["state2:Intercept"], reference.bse[names.index("const[1]")]
                        * np.sqrt(factor), rtol=5e-3)


def test_errors_and_rendering(df):
    with pytest.raises(AnalysisError) as err:
        oe.mswitch(data=df, y="y", x=["x"], switch=[])
    assert err.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as err:
        oe.mswitch(data=df, y="y", switch=["z"])
    assert err.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as err:
        oe.mswitch(data=df, y="y", states=6, ar=[1, 2, 3, 4])
    assert err.value.code == "model_too_large"
    with pytest.raises(AnalysisError) as err:
        oe.mswitch(data=df.iloc[:12], y="y", x=["x"], switch=["Intercept", "x"], varswitch=True)
    assert err.value.code == "insufficient_observations"
    with pytest.raises(AnalysisError) as err:
        oe.mswitch(data=df, y="y", states=1)
    assert err.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as err:
        oe.mswitch_probabilities(oe.prais(data=df, y="y", x=["x"]), df)
    assert err.value.code == "invalid_result"
    fit = oe.mswitch(data=df, y="y", time="t")
    assert type(fit).model_validate_json(fit.model_dump_json()) == fit
    assert "/lgt(p11)" in fit.summary() and "state1" in fit.to_latex()
    assert [c.term for c in fit.coefficients] == ["state1:Intercept", "state2:Intercept",
                                                  "/lnsigma", "/lgt(p11)", "/lgt(p21)"]
    assert fit.provenance["stata_parity_validated"] is False
