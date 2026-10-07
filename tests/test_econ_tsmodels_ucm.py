"""oe.ucm against statsmodels UnobservedComponents (exact diffuse initialization).

statsmodels initializes a stochastic cycle as diffuse; Stata and OpenEcon use its
stationary distribution, so the statsmodels oracle is given that initialization
explicitly. Likelihoods at fixed parameters, maximum likelihood estimates,
OIM / robust / OPG covariances, smoothed states and forecasts are compared.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.structural import UnobservedComponents as UC

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.tsmodels import statespace as ss
from openecon.econometrics.tsmodels import ucm as ucm_module
from openecon.engines.optimize import numerical_gradient, numerical_hessian

warnings.filterwarnings("ignore", module="statsmodels")

SM_NAMES = {"sigma2.irregular": "var(e)", "sigma2.level": "var(level)",
            "sigma2.trend": "var(slope)", "sigma2.seasonal": "var(seasonal)",
            "sigma2.cycle": "var(cycle)", "frequency.cycle": "frequency",
            "damping.cycle": "damping"}


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(11)
    n = 180
    t = np.arange(n)
    level = np.cumsum(rng.normal(size=n) * 0.4 + 0.05)
    cycle = np.zeros(n)
    for i in range(1, n):
        cycle[i] = 0.9 * np.cos(0.35) * cycle[i - 1] + rng.normal() * 0.5
    x = rng.normal(size=n)
    y = 10 + level + 1.5 * np.sin(2 * np.pi * t / 12) + cycle + 0.7 * x + rng.normal(size=n) * 0.8
    return pd.DataFrame({"y": y, "x": x, "t": t + 1})


def sm_model(y, structure_kwargs, exog=None, cycle=False):
    model = UC(y, exog=exog, use_exact_diffuse=True, **structure_kwargs)
    if cycle:
        init = Initialization(model.k_states)
        start = model.k_states - 2
        init.set((0, start), "diffuse")
        init.set((start, model.k_states), "stationary")
        model.ssm.initialization = init
    return model


def sm_params(model, values, beta=()):
    out = []
    for name in model.param_names:
        out.append(values[SM_NAMES[name]] if name in SM_NAMES else beta[int(name[-1]) - 1])
    return np.array(out)


def exact_ll(structure, values, y):
    tensors = {k: torch.tensor([v], dtype=torch.float64) for k, v in values.items()}
    t, q, h, p0 = ss.system(structure, tensors, 1)
    run = ss.kalman(structure, t, q, h, p0, torch.tensor(y)[None, :])
    used = ~run.diffuse_inf
    f, v = run.f[0][used], run.v[0, 0][used]
    return float(-0.5 * (len(y) * np.log(2 * np.pi) + torch.log(f).sum() + run.log_f_inf.sum()
                         + (v * v / f).sum()))


CASES = [
    ("llevel", 0, False, {"var(e)": 0.9, "var(level)": 0.2}, {"level": "llevel"}),
    ("lltrend", 0, False, {"var(e)": 0.9, "var(level)": 0.2, "var(slope)": 0.01},
     {"level": "lltrend"}),
    ("rwalk", 0, False, {"var(level)": 0.7}, {"level": "rwalk"}),
    ("rwdrift", 0, False, {"var(level)": 0.7}, {"level": "rwdrift"}),
    ("strend", 0, False, {"var(e)": 1.0, "var(slope)": 0.02}, {"level": "strend"}),
    ("dconstant", 0, False, {"var(e)": 1.0}, {"level": "dconstant"}),
    ("lldtrend", 0, False, {"var(e)": 0.5, "var(level)": 0.3}, {"level": "lldtrend"}),
    ("llevel", 12, False, {"var(e)": 0.9, "var(level)": 0.2, "var(seasonal)": 0.05},
     {"level": "llevel", "seasonal": 12, "stochastic_seasonal": True}),
    ("llevel", 0, True, {"var(e)": 0.9, "var(level)": 0.2, "var(cycle)": 0.2, "frequency": 0.4,
                         "damping": 0.85},
     {"level": "llevel", "cycle": True, "stochastic_cycle": True, "damped_cycle": True}),
]


@pytest.mark.parametrize("model,seasonal,cycle,values,kwargs", CASES)
def test_exact_diffuse_likelihood_and_smoother(df, model, seasonal, cycle, values, kwargs):
    y = df.y.to_numpy()
    structure = ss.Structure(model, seasonal, cycle)
    reference = sm_model(y, kwargs, cycle=cycle)
    params = sm_params(reference, values)
    assert_allclose(exact_ll(structure, values, y), reference.loglike(params), rtol=1e-9)
    tensors = {k: torch.tensor([v], dtype=torch.float64) for k, v in values.items()}
    t, q, h, p0 = ss.system(structure, tensors, 1)
    smoothed = ss.smooth(structure, t, q, h, p0, torch.tensor(y)).numpy()
    assert_allclose(smoothed, reference.smooth(params).smoothed_state.T, atol=1e-7)


def test_steady_state_shortcut_is_exact():
    rng = np.random.default_rng(8)
    y = torch.tensor(np.cumsum(rng.normal(size=700)) + rng.normal(size=700))
    structure = ss.Structure("lltrend", 4, False)
    values = {"var(e)": torch.tensor([0.9]), "var(level)": torch.tensor([0.2]),
              "var(slope)": torch.tensor([0.01]), "var(seasonal)": torch.tensor([0.05])}
    values = {k: v.double() for k, v in values.items()}
    t, q, h, p0 = ss.system(structure, values, 1)
    fast = ss.kalman(structure, t, q, h, p0, y[None, :], store=True)
    assert fast.steady_from < len(y)
    saved = ss._STEADY_TOL
    try:
        ss._STEADY_TOL = -1.0                      # disable the shortcut
        slow = ss.kalman(structure, t, q, h, p0, y[None, :], store=True)
    finally:
        ss._STEADY_TOL = saved
    assert slow.steady_from == len(y)
    assert_allclose(fast.v.numpy(), slow.v.numpy(), atol=1e-10)
    assert_allclose(fast.f.numpy(), slow.f.numpy(), rtol=1e-12)
    assert_allclose(fast.predicted.numpy(), slow.predicted.numpy(), atol=1e-9)
    assert_allclose(fast.final_state.numpy(), slow.final_state.numpy(), atol=1e-9)


FITS = [
    ({"model": "llevel"}, {"level": "llevel"}, False),
    ({"model": "llevel", "x": ["x"]}, {"level": "llevel"}, True),
    ({"model": "rwdrift", "x": ["x"]}, {"level": "rwdrift"}, True),
    ({"model": "llevel", "seasonal": 12, "x": ["x"]},
     {"level": "llevel", "seasonal": 12, "stochastic_seasonal": True}, True),
]


@pytest.mark.parametrize("kwargs,sm_kwargs,with_x", FITS)
def test_maximum_likelihood_and_oim(df, kwargs, sm_kwargs, with_x):
    fit = oe.ucm(data=df, y="y", time="t", **kwargs)
    reference = sm_model(df.y.to_numpy(), sm_kwargs, exog=df[["x"]].to_numpy() if with_x else None)
    inverse = {value: key for key, value in SM_NAMES.items()}
    fixed = {inverse[name]: 0.0 for name in fit.extra["fixed_zero"]}   # boundary variances
    with reference.fix_params(fixed):
        result = reference.fit(disp=0, maxiter=1000, cov_type="approx")
    assert fit.metrics["log_likelihood"] >= result.llf - 1e-6
    assert_allclose(fit.metrics["log_likelihood"], result.llf, rtol=1e-7)
    ours = {c.term.strip("/"): c for c in fit.coefficients}
    for name, value, se in zip(reference.param_names, result.params, result.bse):
        if name in fixed:
            continue
        key = SM_NAMES.get(name, "x")
        assert_allclose(ours[key].estimate, value, rtol=2e-3, atol=1e-4)
        assert_allclose(ours[key].std_error, se, rtol=2e-2)
    k = len(fit.coefficients)
    assert_allclose(fit.metrics["aic"], -2 * fit.metrics["log_likelihood"] + 2 * k)
    assert fit.inference["use_t"] is False


def test_robust_and_opg_covariances(df):
    reference = sm_model(df.y.to_numpy(), {"level": "llevel"}, exog=df[["x"]].to_numpy())
    for cov, sm_cov, factor in [("opg", "opg", 1.0), ("robust", "robust_approx",
                                                      len(df) / (len(df) - 1))]:
        fit = oe.ucm(data=df, y="y", x=["x"], model="llevel", covariance=cov)
        values = {c.term.strip("/"): c.estimate for c in fit.coefficients}
        params = sm_params(reference, values, beta=[values["x"]])
        result = reference.smooth(params, cov_type=sm_cov)
        ours = {c.term.strip("/"): c.std_error for c in fit.coefficients}
        for name, se in zip(reference.param_names, result.bse):
            assert_allclose(ours[SM_NAMES.get(name, "x")], se * np.sqrt(factor), rtol=2e-3)


def test_cycle_model_matches_statsmodels_likelihood(df):
    with pytest.raises(AnalysisError) as err:          # the period-12 sine becomes a fixed cycle
        oe.ucm(data=df, y="y", model="llevel", cycle=True, cycle_frequency=0.5)
    assert err.value.code == "degenerate_cycle"
    rng = np.random.default_rng(21)
    n = 300
    cycle = np.zeros(n)
    for i in range(2, n):
        cycle[i] = 2 * 0.85 * np.cos(0.4) * cycle[i - 1] - 0.85 ** 2 * cycle[i - 2] + rng.normal()
    data = pd.DataFrame({"y": np.cumsum(rng.normal(size=n) * 0.3) + cycle + rng.normal(size=n)})
    fit = oe.ucm(data=data, y="y", model="llevel", cycle=True)
    values = {c.term.strip("/"): c.estimate for c in fit.coefficients}
    reference = sm_model(data.y.to_numpy(), CASES[-1][4], cycle=True)
    assert_allclose(reference.loglike(sm_params(reference, values)), fit.metrics["log_likelihood"],
                    rtol=1e-9)
    result = reference.fit(start_params=sm_params(reference, values), disp=0, maxiter=500)
    assert fit.metrics["log_likelihood"] >= result.llf - 1e-5
    assert 0 < values["frequency"] < np.pi and 0 < values["damping"] < 1
    cycle = fit.extra["cycle_parameters"]
    assert_allclose(cycle["period"], 2 * np.pi / values["frequency"])


def test_numerical_derivatives_against_engines(df):
    y = torch.tensor(df.y.to_numpy())
    x = torch.tensor(df[["x"]].to_numpy())
    structure = ss.Structure("llevel", 4, False)
    problem = ucm_module._Problem(structure, y, x, 1.0)
    theta = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.float64)
    value, gradient = ucm_module._gradient(problem, theta)
    profile = lambda point: problem.profile(point[None, :])[0]       # noqa: E731
    assert_allclose(gradient, numerical_gradient(profile, theta), rtol=1e-6)
    hessian = ucm_module._profile_hessian(problem, theta)
    reference = numerical_hessian(lambda point: ucm_module._gradient(problem, point)[1], theta)
    assert_allclose(hessian, reference, rtol=1e-4, atol=1e-4)


def test_components_and_forecast(df):
    fit = oe.ucm(data=df, y="y", x=["x"], model="lltrend", seasonal=12, time="t")
    parts = oe.ucm_components(fit, df)
    values = {c.term.strip("/"): c.estimate for c in fit.coefficients}
    for name in fit.extra["fixed_zero"]:
        values[name] = 0.0
    reference = sm_model(df.y.to_numpy(), {"level": "lltrend", "seasonal": 12,
                                           "stochastic_seasonal": True}, exog=df[["x"]].to_numpy())
    params = sm_params(reference, values, beta=[values["x"]])
    smoothed = reference.smooth(params).smoothed_state
    assert_allclose(parts.level, smoothed[0], atol=1e-6)
    assert_allclose(parts.slope, smoothed[1], atol=1e-6)
    assert_allclose(parts.seasonal, smoothed[2], atol=1e-6)
    assert_allclose(parts.fitted + parts.irregular, df.sort_values("t").y, rtol=1e-12)
    assert list(parts.columns) == ["period", "observed", "level", "slope", "seasonal",
                                   "regression", "fitted", "irregular"]
    future = pd.DataFrame({"x": np.linspace(-1, 1, 6)})
    forecast = oe.forecast(fit, steps=6, exog=future)
    expected = reference.smooth(params).get_forecast(6, exog=future[["x"]].to_numpy())
    assert_allclose(forecast.forecast, expected.predicted_mean, rtol=1e-8)
    assert_allclose(forecast.std_error, np.sqrt(expected.var_pred_mean), rtol=1e-8)
    assert list(forecast.period) == list(range(len(df) + 1, len(df) + 7))
    again = oe.ucm_forecast(fit, 6, data=df, exog=future)
    assert_allclose(again.forecast, forecast.forecast, rtol=1e-10)
    with pytest.raises(AnalysisError) as err:
        oe.forecast(fit, steps=3)
    assert err.value.code == "missing_exog"


def test_zero_variance_boundary_and_errors(df):
    fit = oe.ucm(data=df, y="y", x=["x"], model="lltrend", seasonal=12)
    assert "var(slope)" in fit.extra["fixed_zero"]
    assert "/var(slope)" not in [c.term for c in fit.coefficients]
    assert fit.extra["variances"]["var(slope)"] == 0.0
    assert any("zero boundary" in w for w in fit.warnings)
    with pytest.raises(AnalysisError) as err:
        oe.ucm(data=df.assign(y=1.0), y="y", model="llevel")
    assert err.value.code == "constant_outcome"
    with pytest.raises(AnalysisError) as err:
        oe.ucm(data=df.assign(c=1.0), y="y", x=["c"], model="llevel")
    assert err.value.code == "collinear_regressors"
    with pytest.raises(AnalysisError) as err:
        oe.ucm(data=df, y="y", model="bogus")
    assert err.value.code == "invalid_spec"
    with pytest.raises(AnalysisError) as err:
        oe.ucm(data=df, y="y", model="none")
    assert err.value.code == "invalid_model"
    with pytest.raises(AnalysisError) as err:
        oe.ucm(data=df, y="y", model="llevel", cycle=True, cycle_frequency=4.0)
    assert err.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as err:
        oe.ucm(data=df.assign(t=df.t * 2), y="y", time="t", model="llevel")
    assert err.value.code == "time_gaps"


def test_round_trip_and_rendering(df):
    fit = oe.ucm(data=df, y="y", x=["x"], model="llevel", time="t")
    assert type(fit).model_validate_json(fit.model_dump_json()) == fit
    text = fit.summary()
    assert "/var(level)" in text and "Ljung-Box" in text
    assert "var" in fit.to_latex()
    variance_rows = [c for c in fit.coefficients if c.term.startswith("/var")]
    assert all(c.ci_low >= 0 for c in variance_rows)
    assert fit.provenance["stata_parity_validated"] is False


@pytest.mark.parametrize("model,seasonal,fixed", [
    ("rwdrift", 0, ()), ("dtrend", 4, ()), ("dconstant", 0, ()), ("lldtrend", 0, ()),
    ("lltrend", 12, ("var(slope)", "var(seasonal)")), ("llevel", 4, ("var(level)",))])
def test_deterministic_states_are_concentrated_exactly(df, model, seasonal, fixed):
    """de Jong's diffuse likelihood with deterministic states as GLS regressors equals DK's."""
    y = torch.tensor(df.y.to_numpy())
    x = torch.tensor(df[["x"]].to_numpy())
    structure = ss.Structure(model, seasonal, False, frozenset(fixed))
    problem = ucm_module._Problem(structure, y, x, 1.0)
    assert problem.fixed
    theta = torch.tensor([[-0.3 - 0.5 * i for i in range(len(structure.parameters))],
                          [0.2 + 0.1 * i for i in range(len(structure.parameters))]],
                         dtype=torch.float64)
    reduced, full = problem.pieces(theta), problem.pieces(theta, reduce=False)
    for name in ("syy", "sxy", "sxx", "logdet"):
        assert_allclose(reduced[name], full[name], rtol=1e-9, atol=1e-9)
