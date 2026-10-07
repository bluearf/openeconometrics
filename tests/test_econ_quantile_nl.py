"""Independent oracles for nonlinear least squares (nl) and its formula language.

Estimates are compared with scipy.optimize (curve_fit / least_squares),
covariances with statsmodels OLS on the linearized model (the Gauss-Newton
regression of the residuals on the Jacobian), the analytic Jacobian with
numerical derivatives of an independent NumPy implementation of each formula,
and frequency weights with duplicated rows.
"""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import curve_fit, least_squares
from scipy.special import expit

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.quantile import formula as formulas
from openecon.econometrics.quantile import nl as kernel
from openecon.engines.contracts import KernelError
from openecon.engines.optimize import check_derivatives
from openecon.models import ModelSpec

DECAY = "{b0} + {b1} * exp(-{b2} * x) + {b3} * z"
DECAY_START = {"b0": 1.0, "b1": 1.0, "b2": 0.3}


def make_data(seed=7, n=220):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"x": rng.uniform(0.2, 5, size=n), "z": rng.normal(size=n)})
    noise = rng.normal(size=n) * 0.2 * (1 + 0.5 * frame.x)
    frame["y"] = 1.5 + 2.5 * np.exp(-0.8 * frame.x) + 0.3 * frame.z + noise
    frame["g"] = rng.integers(0, 24, size=n)
    frame["f"] = rng.integers(1, 4, size=n).astype(float)
    frame["w"] = rng.uniform(0.4, 2.5, size=n)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def decay(frame, b0, b1, b2, b3):
    return b0 + b1 * np.exp(-b2 * frame.x.to_numpy()) + b3 * frame.z.to_numpy()


def decay_jacobian(frame, b):
    e = np.exp(-b[2] * frame.x.to_numpy())
    return np.column_stack([np.ones(len(frame)), e, -b[1] * frame.x.to_numpy() * e,
                            frame.z.to_numpy()])


def params(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def covariance(result):
    return np.array(result.covariance_matrix)


def scipy_fit(frame, start=(1.0, 1.0, 0.3, 0.0), weights=None):
    scale = np.ones(len(frame)) if weights is None else np.sqrt(weights)
    solution = least_squares(lambda b: scale * (frame.y.to_numpy() - decay(frame, *b)), start,
                             xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return solution.x


# ---- the formula language ------------------------------------------------------------------------


def test_parser_reads_parameters_columns_and_starting_values():
    parsed = formulas.parse(" {b0=1.5} + {b1}*exp(-{rate = 2e-1}*x) \n + ln(z)^2/{b1} ")
    assert parsed.parameters == ("b0", "b1", "rate")
    assert parsed.columns == ("x", "z")
    assert parsed.start == {"b0": 1.5, "rate": 0.2}
    # A column may share its name with a function; parameters may share a column's name.
    shadow = formulas.parse("{x}*x + {b}*exp + log(log)")
    assert shadow.parameters == ("x", "b") and shadow.columns == ("x", "exp", "log")
    assert formulas.parse("{a}*x**2").tape == formulas.parse("{a}*x^2").tape
    # Common subexpressions are stored once.
    repeated = formulas.parse("exp({a}*x) / (1 + exp({a}*x))")
    assert sum(entry[0] == "fn" for entry in repeated.tape) == 1


@pytest.mark.parametrize("text", [
    "__import__('os').system('true') + {b}", "x.real * {b}", "x[0] * {b}", "(lambda: 1)() + {b}",
    "x if x else {b}", "exp(x, 2) * {b}", "exp() + {b}", "exp(*x) + {b}", "exp(x=1) + {b}",
    "foo(x) * {b}", "{b}^x^2", "{a}^-{b}^2", "y = {b} * x", "{b", "b}", "x + 'a' + {b}",
    "{b: x1 x2}", "{1b} * x", "x + 1", "{b=abc} * x", "{b=1} + {b=2} * x", "1e999 * {b}",
    "{b} * True", "{b} * 1j", "x @ {b}", "x < {b}", "x // {b}", "x % {b}", "{b}; x", "",
    "   ", "(" * 300 + "{b} * x" + ")" * 300, "{b} * x\x00", "[{b}]", "{b} + __oe_parameter_b",
    "not {b}", "~{b}", "{b} and x", "f'{b}'", "{b} * " + "x" * 5000, "{}", "{b} +", "pow(x, {b})",
])
def test_parser_rejects_everything_outside_the_grammar(text):
    with pytest.raises(AnalysisError) as caught:
        formulas.parse(text)
    assert caught.value.code == "invalid_formula"


def test_parser_rejects_non_strings():
    for value in (None, 3, ["{b}*x"], b"{b}*x"):
        with pytest.raises(AnalysisError) as caught:
            formulas.parse(value)
        assert caught.value.code == "invalid_formula"


CASES = [
    ("{a} + {b}*x - {c}*z", lambda x, z, a, b, c: a + b * x - c * z),
    ("-{a}*x^2 + (-{b})^2*z", lambda x, z, a, b: -a * x ** 2 + b ** 2 * z),
    ("{a}*x^{b}", lambda x, z, a, b: a * x ** b),
    ("{a}^x + x^-{b}", lambda x, z, a, b: a ** x + x ** (-b)),
    ("({a}*x)^({b}*z)", lambda x, z, a, b: (a * x) ** (b * z)),
    ("{a}/(1 + exp(-{b}*(x - {c})))", lambda x, z, a, b, c: a / (1 + np.exp(-b * (x - c)))),
    ("{a}*x/({b} + x)", lambda x, z, a, b: a * x / (b + x)),
    ("exp({a}*z)/x + ln({b}*x) - log(x + {c}^2)",
     lambda x, z, a, b, c: np.exp(a * z) / x + np.log(b * x) - np.log(x + c ** 2)),
    ("sqrt({a}*x) + abs({b}*z - 1)", lambda x, z, a, b: np.sqrt(a * x) + np.abs(b * z - 1)),
    ("sin({a}*x) + cos({b}*z)*tan({c}*z/10)",
     lambda x, z, a, b, c: np.sin(a * x) + np.cos(b * z) * np.tan(c * z / 10)),
    ("expit({a} + {b}*z) - invlogit({a}*x)",
     lambda x, z, a, b: expit(a + b * z) - expit(a * x)),
    ("normal({a} + {b}*z) + normalden({c}*x - {a})",
     lambda x, z, a, b, c: stats.norm.cdf(a + b * z) + stats.norm.pdf(c * x - a)),
    ("{a}*({b}*x^{c} + (1 - {b})*abs(z)^{c})^(1/{c})",
     lambda x, z, a, b, c: a * (b * x ** c + (1 - b) * np.abs(z) ** c) ** (1 / c)),
    ("{a} + 2.5e-1*{b}*x*x/3 + +{c}", lambda x, z, a, b, c: a + 0.25 * b * x * x / 3 + c),
    ("{a}*x^2 + {b}*x^1 + {c}*x^0 + x^0.5",
     lambda x, z, a, b, c: a * x ** 2 + b * x + c + x ** 0.5),
]


@pytest.mark.parametrize("text, function", CASES)
def test_values_and_analytic_jacobian_match_numerical_derivatives(text, function):
    rng = np.random.default_rng(len(text))
    n = 40
    x, z = rng.uniform(0.3, 3, size=n), rng.normal(size=n)
    parsed = formulas.parse(text)
    p = len(parsed.parameters)
    theta = rng.uniform(0.4, 1.6, size=p)
    columns = {"x": torch.tensor(x), "z": torch.tensor(z)}
    value, jacobian = formulas.evaluate(parsed, columns, torch.tensor(theta), n)
    assert_allclose(value.numpy(), function(x, z, *theta), rtol=1e-13, atol=1e-13)
    assert jacobian.shape == (n, p)
    # Central differences of the independent NumPy implementation.
    step = 1e-6
    shifts = step * np.eye(p)
    numerical = np.column_stack([
        (function(x, z, *(theta + shifts[j])) - function(x, z, *(theta - shifts[j]))) / (2 * step)
        for j in range(p)])
    assert_allclose(jacobian.numpy(), numerical, rtol=2e-7, atol=2e-8)
    # The shared derivative checker on a random linear functional of f.
    direction = torch.tensor(rng.normal(size=n))

    def functional(point):
        f, j = formulas.evaluate(parsed, columns, point, n)
        return direction @ f, j.T @ direction

    report = check_derivatives(functional, torch.tensor(theta))
    assert report["gradient_max_rel_error"] < 1e-8
    without, none = formulas.evaluate(parsed, columns, torch.tensor(theta), n, jacobian=False)
    assert none is None
    assert_allclose(without.numpy(), value.numpy(), rtol=0, atol=0)


def test_jacobian_edge_values():
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    theta = torch.tensor([2.0, 1.5], dtype=torch.float64)
    # x^b at x = 0: the value and both partial derivatives are finite (zero).
    value, jacobian = formulas.evaluate(formulas.parse("{a}*x^{b}"), {"x": x}, theta, 3)
    assert_allclose(value.numpy(), [0, 2, 2 * 2 ** 1.5])
    assert_allclose(jacobian.numpy()[0], [0, 0])
    assert_allclose(jacobian.numpy()[2], [2 ** 1.5, 2 * 2 ** 1.5 * np.log(2)])
    # A constant power of a parameter expression at zero.
    parsed = formulas.parse("({a} - 2)^2 + ({b}*x)^0")
    value, jacobian = formulas.evaluate(parsed, {"x": x}, theta, 3)
    assert_allclose(value.numpy(), [1, 1, 1])
    assert_allclose(jacobian.numpy(), np.zeros((3, 2)))
    # A formula without any data column is a constant function of the parameters.
    value, jacobian = formulas.evaluate(formulas.parse("{a}*{b}"), {}, theta, 4)
    assert_allclose(value.numpy(), [3, 3, 3, 3])
    assert_allclose(jacobian.numpy(), np.tile([1.5, 2.0], (4, 1)))


# ---- estimation ----------------------------------------------------------------------------


def test_nl_matches_scipy(data):
    result = oe.nl(data=data, y="y", formula=DECAY, start=DECAY_START)
    assert [c.term for c in result.coefficients] == ["b0", "b1", "b2", "b3"]
    estimates, pcov = curve_fit(lambda _, *b: decay(data, *b), None, data.y.to_numpy(),
                                p0=[1, 1, 0.3, 0], xtol=1e-14, ftol=1e-14, gtol=1e-14)
    assert_allclose(params(result), estimates, rtol=1e-7)
    assert_allclose(params(result), scipy_fit(data), rtol=1e-7)
    assert_allclose(covariance(result), pcov, rtol=1e-5)
    n, k = len(data), 4
    resid = data.y.to_numpy() - decay(data, *estimates)
    rss = resid @ resid
    tss = ((data.y - data.y.mean()) ** 2).sum()
    metrics = result.metrics
    assert_allclose(metrics["rss"], rss, rtol=1e-10)
    assert_allclose(metrics["tss"], tss, rtol=1e-12)
    assert_allclose(metrics["rmse"], np.sqrt(rss / (n - k)), rtol=1e-10)
    assert_allclose(metrics["r_squared"], 1 - rss / tss, rtol=1e-10)
    assert_allclose(metrics["adjusted_r_squared"], 1 - (rss / (n - k)) / (tss / (n - 1)),
                    rtol=1e-10)
    assert_allclose(metrics["log_likelihood"],
                    stats.norm.logpdf(resid, scale=np.sqrt(rss / n)).sum(), rtol=1e-10)
    assert metrics["df_model"] == 3 and metrics["df_resid"] == n - k
    assert result.extra["constant_term"] == "b0"
    assert result.extra["total_sum_of_squares"] == "centered"
    assert result.extra["start"] == {"b0": 1.0, "b1": 1.0, "b2": 0.3, "b3": 0.0}
    assert result.extra["columns"] == ["x", "z"] and result.spec.predictors == ["x", "z"]
    assert result.inference["use_t"] and result.inference["df_inference"] == n - k
    coefficient = result.coefficients[2]
    assert_allclose(coefficient.p_value, 2 * stats.t.sf(abs(coefficient.statistic), n - k),
                    rtol=1e-8)
    diagnostics = result.provenance["solver_diagnostics"]
    assert diagnostics["converged"] is True and diagnostics["iterations"] == metrics["iterations"]
    assert diagnostics["relative_predicted_reduction"] < 1e-12
    assert diagnostics["rss_history"][-1] == pytest.approx(rss, rel=1e-10)
    assert result.provenance["solver"] == "gauss_newton_levenberg_marquardt"
    assert result.provenance["optimizer"]["tolerance"] == 1e-8
    assert result.tests == {} and result.title == "Nonlinear least squares"
    assert any("b3" in warning and "started at 0" in warning for warning in result.warnings)


def test_nl_reproduces_nist_certified_values():
    # NIST StRD nonlinear regression, dataset Misra1a: y = b1 (1 - exp(-b2 x)) + e.
    frame = pd.DataFrame({
        "x": [77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4, 434.8, 477.3, 536.8,
              593.1, 689.1, 760.0],
        "y": [10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82, 50.76, 55.05, 61.01,
              66.40, 75.47, 81.78],
    })
    certified = np.array([2.3894212918e2, 5.5015643181e-4])
    certified_errors = np.array([2.7070075241, 7.2668688436e-6])
    for start in ({"b1": 500, "b2": 1e-4}, {"b1": 250, "b2": 5e-4}):     # NIST starts 1 and 2
        result = oe.nl(data=frame, y="y", formula="{b1}*(1 - exp(-{b2}*x))", start=start)
        assert_allclose(params(result), certified, rtol=1e-9)
        assert_allclose(errors(result), certified_errors, rtol=1e-8)
        assert_allclose(result.metrics["rss"], 1.2455138894e-1, rtol=1e-9)
        assert_allclose(result.metrics["rmse"], 1.0187876330e-1, rtol=1e-9)
        assert result.metrics["df_resid"] == 12


@pytest.mark.parametrize("text, start, truth, function", [
    ("{a}*x/({b} + x)", {"a": 1.0, "b": 1.0}, [3.0, 0.9], lambda x, z, a, b: a * x / (b + x)),
    ("{a}/(1 + exp(-{b}*(x - {c})))", {"a": 4.0, "b": 1.0, "c": 2.0}, [3.0, 0.9, 2.5],
     lambda x, z, a, b, c: a / (1 + np.exp(-b * (x - c)))),
    ("{a}*x^{b}", {"a": 1.0, "b": 1.0}, [3.0, 0.9], lambda x, z, a, b: a * x ** b),
    ("{a}*(1 - exp(-{b}*x))", {"a": 2.0, "b": 0.5}, [3.0, 0.9],
     lambda x, z, a, b: a * (1 - np.exp(-b * x))),
    ("normal({a} + {b}*z)*{c}", {"a": 0.1, "b": 0.5, "c": 2.0}, [0.3, 0.9, 2.5],
     lambda x, z, a, b, c: stats.norm.cdf(a + b * z) * c),
])
def test_nl_models_match_scipy_least_squares(text, start, truth, function):
    rng = np.random.default_rng(len(text))
    n = 150
    x, z = rng.uniform(0.2, 6, size=n), rng.normal(size=n)
    y = function(x, z, *truth) + rng.normal(size=n) * 0.15
    frame = pd.DataFrame({"x": x, "z": z, "y": y})
    result = oe.nl(data=frame, y="y", formula=text, start=start)
    solution = least_squares(lambda b: y - function(x, z, *b), list(start.values()),
                             xtol=1e-14, ftol=1e-14, gtol=1e-14)
    assert_allclose(params(result), solution.x, rtol=1e-6)
    k = len(start)
    sigma2 = (solution.fun @ solution.fun) / (n - k)
    expected = sigma2 * np.linalg.inv(solution.jac.T @ solution.jac)
    assert_allclose(covariance(result), expected, rtol=1e-4)
    assert result.extra["constant_term"] is None
    # No additive constant: the total sum of squares is uncentered.
    assert_allclose(result.metrics["tss"], y @ y, rtol=1e-12)
    assert_allclose(result.metrics["adjusted_r_squared"],
                    1 - (1 - result.metrics["r_squared"]) * n / (n - k), rtol=1e-12)
    assert result.metrics["df_model"] == k


@pytest.mark.parametrize("kind, options", [
    ("robust", {}), ("HC2", {}), ("HC3", {}), ("cluster", {"cluster": "g"}),
])
def test_nl_robust_covariances_match_the_linearized_regression(data, kind, options):
    result = oe.nl(data=data, y="y", formula=DECAY, start=DECAY_START,
                   covariance=None if kind == "cluster" else kind, **options)
    assert result.spec.covariance == kind
    estimates = scipy_fit(data)
    jacobian = decay_jacobian(data, estimates)
    resid = data.y.to_numpy() - decay(data, *estimates)
    # Gauss-Newton regression: OLS of J b + e on J reproduces b with residuals e.
    linearized = sm.OLS(jacobian @ estimates + resid, jacobian)
    if kind == "cluster":
        reference = linearized.fit(cov_type="cluster", cov_kwds={"groups": data.g.to_numpy()})
        assert result.inference["df_inference"] == data.g.nunique() - 1
        assert result.inference["cluster_count"] == data.g.nunique()
    else:
        reference = linearized.fit(cov_type="HC1" if kind == "robust" else kind)
        assert result.inference["df_inference"] == len(data) - 4
    assert_allclose(params(result), reference.params, rtol=1e-6)
    assert_allclose(covariance(result), reference.cov_params(), rtol=1e-5)


def test_nl_linear_model_equals_ols(data):
    frame = data.assign(x2=data.x ** 2)
    x = sm.add_constant(frame[["x", "x2", "z"]])
    ols = sm.OLS(frame.y, x).fit()
    result = oe.nl(data=frame, y="y", formula="{c} + {bx}*x + {bq}*x2 + {bz}*z")
    assert_allclose(params(result), ols.params, rtol=1e-9)
    assert_allclose(covariance(result), ols.cov_params(), rtol=1e-8)
    assert_allclose(result.metrics["r_squared"], ols.rsquared, rtol=1e-10)
    assert_allclose(result.metrics["adjusted_r_squared"], ols.rsquared_adj, rtol=1e-10)
    assert_allclose(result.metrics["log_likelihood"], ols.llf, rtol=1e-10)
    assert result.metrics["iterations"] <= 2
    # Without a constant Stata's (and statsmodels') R-squared is uncentered.
    origin = sm.OLS(frame.y, frame[["x", "z"]]).fit()
    through = oe.nl(data=frame, y="y", formula="{bx}*x + {bz}*z")
    assert_allclose(params(through), origin.params, rtol=1e-9)
    assert_allclose(through.metrics["r_squared"], origin.rsquared, rtol=1e-10)
    assert_allclose(through.metrics["adjusted_r_squared"], origin.rsquared_adj, rtol=1e-10)
    # The same quadratic written with a power: the Jacobian has the same columns.
    power = oe.nl(data=frame, y="y", formula="{c} + {bx}*x + {bq}*x^2 + {bz}*z")
    assert_allclose(params(power), ols.params, rtol=1e-9)
    # A constant-only model returns the mean.
    mean = oe.nl(data=frame, y="y", formula="{m}")
    assert_allclose(params(mean), [frame.y.mean()], rtol=1e-12)
    assert_allclose(mean.coefficients[0].std_error, frame.y.std(ddof=1) / np.sqrt(len(frame)),
                    rtol=1e-10)
    assert mean.metrics["r_squared"] == pytest.approx(0, abs=1e-12) and mean.spec.predictors == []


def test_nl_weights(data):
    expanded = data.loc[data.index.repeat(data.f.astype(int))].reset_index(drop=True)
    for kind in ("nonrobust", "robust", "cluster"):
        options = {"cluster": "g"} if kind == "cluster" else {"covariance": kind}
        weighted = oe.nl(data=data, y="y", formula=DECAY, start=DECAY_START, weights="f",
                         weight_type="fweight", **options)
        plain = oe.nl(data=expanded, y="y", formula=DECAY, start=DECAY_START, **options)
        assert weighted.nobs == len(expanded)
        assert_allclose(params(weighted), params(plain), rtol=1e-8)
        assert_allclose(covariance(weighted), covariance(plain), rtol=1e-6)
        for name in ("r_squared", "adjusted_r_squared", "rmse", "rss", "log_likelihood"):
            assert_allclose(weighted.metrics[name], plain.metrics[name], rtol=1e-8)
    # Analytic weights: weighted least squares with the weights rescaled to sum to N.
    w = data.w.to_numpy()
    aweight = oe.nl(data=data, y="y", formula=DECAY, start=DECAY_START, weights="w",
                    weight_type="aweight")
    estimates = scipy_fit(data, weights=w)
    assert_allclose(params(aweight), estimates, rtol=1e-6)
    jacobian = decay_jacobian(data, estimates)
    resid = data.y.to_numpy() - decay(data, *estimates)
    wls = sm.WLS(jacobian @ estimates + resid, jacobian, weights=w).fit()
    assert_allclose(covariance(aweight), wls.cov_params(), rtol=1e-5)
    assert "log_likelihood" not in aweight.metrics
    wn = w * len(w) / w.sum()
    assert_allclose(aweight.metrics["rss"], (wn * resid ** 2).sum(), rtol=1e-8)
    ybar = (wn * data.y).sum() / len(w)
    assert_allclose(aweight.metrics["tss"], (wn * (data.y - ybar) ** 2).sum(), rtol=1e-10)
    # pweights: the same point estimates with the robust sandwich by default.
    pweight = oe.nl(data=data, y="y", formula=DECAY, start=DECAY_START, weights="w",
                    weight_type="pweight")
    assert pweight.spec.covariance == "robust"
    assert_allclose(params(pweight), params(aweight), rtol=1e-10)
    assert_allclose(covariance(pweight), wls.get_robustcov_results("HC1").cov_params(), rtol=1e-5)
    with pytest.raises(AnalysisError) as caught:
        oe.nl(data=data, y="y", formula=DECAY, start=DECAY_START, weights="w",
              weight_type="pweight", covariance="nonrobust")
    assert caught.value.code == "unsupported_covariance"
    with pytest.raises(AnalysisError) as caught:
        oe.nl(data=data, y="y", formula=DECAY, weights="w", weight_type="iweight")
    assert caught.value.code == "invalid_spec"


def test_nl_starting_values(data):
    inline = oe.nl(data=data, y="y", formula="{b0=1} + {b1=1}*exp(-{b2=0.3}*x) + {b3=0}*z")
    assert inline.warnings == []
    reference = oe.nl(data=data, y="y", formula=DECAY, start={**DECAY_START, "b3": 0})
    assert_allclose(params(inline), params(reference), rtol=1e-9)
    # start overrides the values written in the formula.
    override = oe.nl(data=data, y="y", formula="{b0=1} + {b1=1}*exp(-{b2=40}*x) + {b3=0}*z",
                     start={"b2": 0.3})
    assert override.extra["start"]["b2"] == 0.3
    assert_allclose(params(override), params(reference), rtol=1e-7)
    # Far-away starting values still reach the same minimum through the damped steps.
    far = oe.nl(data=data, y="y", formula=DECAY, start={"b0": -5, "b1": 20, "b2": 3, "b3": 4})
    assert_allclose(params(far), params(reference), rtol=1e-6)
    zero = oe.nl(data=data, y="y", formula=DECAY)
    assert_allclose(params(zero), params(reference), rtol=1e-6)
    assert "b0, b1, b2, b3" in zero.warnings[0]
    cases = [
        ({"start": {"nope": 1}}, "invalid_start"),
        ({"start": {"b0": "one"}}, "invalid_start"),
        ({"start": {"b0": float("inf")}}, "invalid_start"),
        ({"start": [1, 2, 3]}, "invalid_start"),
        ({"formula": "{a} + ln({b})*x"}, "invalid_start"),
        ({"formula": "{a} + {b}/({c}*x)"}, "invalid_start"),
    ]
    for options, code in cases:
        arguments = {"data": data, "y": "y", "formula": DECAY, **options}
        with pytest.raises(AnalysisError) as caught:
            oe.nl(**arguments)
        assert caught.value.code == code, options


def test_nl_failures(data):
    cases = [
        ({"formula": "{a}*{b}*x", "start": {"a": 1, "b": 1}}, "not_identified"),
        ({"formula": "{a} + {b}*x + {c}*x", "start": {"a": 1}}, "not_identified"),
        ({"formula": DECAY, "start": DECAY_START, "max_iterations": 2}, "nonconvergence"),
        ({"formula": "{a} + {b}*q"}, "missing_columns"),
        ({"formula": "{a} + {b}*y"}, "invalid_spec"),
        ({"formula": "exp(x"}, "invalid_formula"),
        ({"formula": DECAY, "tolerance": 0}, "invalid_option"),
        ({"formula": DECAY, "max_iterations": 0}, "invalid_spec"),
        ({"formula": DECAY, "covariance": "hac"}, "invalid_spec"),
        ({"formula": DECAY, "covariance": "cluster"}, "invalid_spec"),
    ]
    for options, code in cases:
        with pytest.raises(AnalysisError) as caught:
            oe.nl(data=data, y="y", **options)
        assert caught.value.code == code, options
    with pytest.raises(AnalysisError) as caught:
        oe.nl(data=data, y="y", formula="{a}*{b}*x", start={"a": 1, "b": 1})
    assert "b" in str(caught.value) and "rank deficient" in str(caught.value)
    exact = data.assign(y=2 + 3 * np.exp(-0.5 * data.x))
    with pytest.raises(AnalysisError) as caught:
        oe.nl(data=exact, y="y", formula="{a} + {b}*exp(-{c}*x)", start={"a": 1, "b": 1, "c": 1})
    assert caught.value.code == "perfect_fit"
    with pytest.raises(AnalysisError) as caught:
        oe.nl(data=data.iloc[:4], y="y", formula=DECAY, start=DECAY_START)
    assert caught.value.code == "insufficient_observations"
    # Specs built directly: the formula is required and validated at fit time.
    with pytest.raises(Exception) as caught:
        ModelSpec(estimator="nl", outcome="y", intercept=False)
    assert "formula" in str(caught.value)
    with pytest.raises(Exception) as caught:
        ModelSpec(estimator="nl", outcome="y", options={"formula": DECAY})
    assert "intercept=False" in str(caught.value)
    spec = ModelSpec(estimator="nl", outcome="y", intercept=False, predictors=["x", "z", "w"],
                     options={"formula": DECAY})
    with pytest.raises(AnalysisError) as caught:
        oe.fit(spec, data=data)
    assert caught.value.code == "invalid_spec" and "w" in str(caught.value)
    spec = ModelSpec(estimator="nl", outcome="y", intercept=False,
                     options={"formula": "{a} + os.x"})
    with pytest.raises(AnalysisError) as caught:
        oe.fit(spec, data=data)
    assert caught.value.code == "invalid_formula"
    # The kernel reports a function that cannot be evaluated at its start.
    with pytest.raises(KernelError) as caught:
        kernel.nonlinear_least_squares(
            lambda theta, need: (torch.full((5,), float("nan"), dtype=torch.float64),
                                 torch.zeros((5, 1), dtype=torch.float64)),
            torch.zeros(1, dtype=torch.float64), torch.ones(5, dtype=torch.float64))
    assert caught.value.code == "invalid_start"


def test_nl_missing_data_policy(data):
    frame = data.copy()
    frame.loc[[4, 9], "x"] = np.nan
    frame.loc[30, "y"] = np.nan
    frame.loc[50, "w"] = np.nan            # not a model input without weights
    with pytest.raises(AnalysisError) as caught:
        oe.nl(data=frame, y="y", formula=DECAY, start=DECAY_START)
    assert caught.value.code == "missing_values"
    dropped = oe.nl(data=frame, y="y", formula=DECAY, start=DECAY_START, missing="drop")
    complete = frame.dropna(subset=["x", "y", "z"])
    assert dropped.nobs == len(complete) == len(frame) - 3
    assert dropped.sample_positions == complete.index.tolist()
    reference = oe.nl(data=complete.reset_index(drop=True), y="y", formula=DECAY, start=DECAY_START)
    assert_allclose(params(dropped), params(reference), rtol=1e-10)


def test_nl_round_trip_rendering_and_public_api(data):
    result = oe.nl(data=data, y="y", formula=DECAY, start=DECAY_START, covariance="robust")
    restored = type(result).model_validate_json(result.model_dump_json())
    assert restored == result
    json.dumps(result.model_dump())
    summary = result.summary()
    assert "Nonlinear least squares" in summary and "adjusted_r_squared" in summary
    assert "b2" in summary
    assert "tabular" in result.to_latex() and "b1" in result.to_latex()
    assert len(result.predictions) == len(data)
    spec = ModelSpec(estimator="nl", outcome="y", intercept=False, covariance="robust",
                     options={"formula": DECAY, "start": DECAY_START})
    again = oe.fit(spec, data=data)
    assert_allclose(params(again), params(result), rtol=0, atol=0)
    assert_allclose(covariance(again), covariance(result), rtol=0, atol=0)
    listing = oe.capabilities()["estimators"]["nl"]
    assert listing["family"] == "quantile" and listing["stata"] == ["nl"]
    assert listing["covariances"] == ["nonrobust", "robust", "HC2", "HC3", "cluster"]
    assert listing["options"]["formula"]["required"] is True
    assert listing["intercept_rule"] == "never" and listing["categorical_predictors"] is False
    assert result.provenance["stata_parity_validated"] is False
    assert callable(oe.nl) and "Gauss-Newton" in oe.nl.__doc__
