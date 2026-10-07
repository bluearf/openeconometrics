"""Independent oracles and Stata reference output for the quantile family.

Two kinds of evidence, both independent of the implementation:

* Stata output.  The examples of the Stata Base Reference Manual on auto.dta
  ([R] qreg: default vce(iid) at the .25/.5/.75 quantiles and vce(robust);
  [R] rreg: ``rreg mpg weight foreign`` with its iteration log) and the Stata
  results for ``qreg foodexp income, vce(iid, kernel(k) bw)`` on Engel's data
  that ship with statsmodels' test suite (e(sparsity), e(kbwidth), e(bwidth),
  standard errors for eight kernels and three bandwidth rules).
* Derivations written here from the formulas: the exact linear program of
  quantile regression solved by HiGHS, NumPy algebra for every covariance,
  SciPy least squares with complex-step Jacobians for nl, and invariances
  (frequency weights = duplicated rows, permutations, rescaling, equivariance).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from scipy import optimize, sparse, stats

import openecon as oe

# The 1978 automobile data of Stata's manuals (auto.dta): the columns the examples use.
AUTO = pd.DataFrame({
    "price": [4099, 4749, 3799, 4816, 7827, 5788, 4453, 5189, 10372, 4082, 11385, 14500, 15906,
              3299, 5705, 4504, 5104, 3667, 3955, 3984, 4010, 5886, 6342, 4389, 4187, 11497,
              13594, 13466, 3829, 5379, 6165, 4516, 6303, 3291, 8814, 5172, 4733, 4890, 4181,
              4195, 10371, 4647, 4425, 4482, 6486, 4060, 5798, 4934, 5222, 4723, 4424, 4172,
              9690, 6295, 9735, 6229, 4589, 5079, 8129, 4296, 5799, 4499, 3995, 12990, 3895,
              3798, 5899, 3748, 5719, 7140, 5397, 4697, 6850, 11995],
    "mpg": [22, 17, 22, 20, 15, 18, 26, 20, 16, 19, 14, 14, 21, 29, 16, 22, 22, 24, 19, 30, 18,
            16, 17, 28, 21, 12, 12, 14, 22, 14, 15, 18, 14, 20, 21, 19, 19, 18, 19, 24, 16, 28,
            34, 25, 26, 18, 18, 18, 19, 19, 19, 24, 17, 23, 25, 23, 35, 24, 21, 21, 25, 28, 30,
            14, 26, 35, 18, 31, 18, 23, 41, 25, 25, 17],
    "weight": [2930, 3350, 2640, 3250, 4080, 3670, 2230, 3280, 3880, 3400, 4330, 3900, 4290,
               2110, 3690, 3180, 3220, 2750, 3430, 2120, 3600, 3600, 3740, 1800, 2650, 4840,
               4720, 3830, 2580, 4060, 3720, 3370, 4130, 2830, 4060, 3310, 3300, 3690, 3370,
               2730, 4030, 3260, 1800, 2200, 2520, 3330, 3700, 3470, 3210, 3200, 3420, 2690,
               2830, 2070, 2650, 2370, 2020, 2280, 2750, 2130, 2240, 1760, 1980, 3420, 1830,
               2050, 2410, 2200, 2670, 2160, 2040, 1930, 1990, 3170],
    "length": [186, 173, 168, 196, 222, 218, 170, 200, 207, 200, 221, 204, 204, 163, 212, 193,
               200, 179, 197, 163, 206, 206, 220, 147, 179, 233, 230, 201, 169, 221, 212, 198,
               217, 195, 220, 198, 198, 218, 200, 180, 206, 170, 157, 165, 182, 201, 214, 198,
               201, 199, 203, 179, 189, 174, 177, 170, 165, 170, 184, 161, 172, 149, 154, 192,
               142, 164, 174, 165, 175, 172, 155, 155, 156, 193],
    "foreign": [0] * 52 + [1] * 22,
}).astype(float)
AUTO_X = ["weight", "length", "foreign"]


# ---- helpers written from the definitions -----------------------------------------------


def check_loss(u, tau, w=None):
    loss = np.where(u < 0, (tau - 1) * u, tau * u)
    return float(loss.sum() if w is None else (loss * w).sum())


def lp_fit(x, y, tau, w=None):
    """min sum_i w_i (tau u_i + (1 - tau) v_i)  s.t.  x b + u - v = y, u, v >= 0  (HiGHS)."""
    n, k = x.shape
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    cost = np.r_[np.zeros(k), tau * w, (1 - tau) * w]
    a_eq = sparse.hstack([sparse.csr_matrix(x), sparse.eye(n), -sparse.eye(n)]).tocsr()
    solution = optimize.linprog(cost, A_eq=a_eq, b_eq=y, method="highs",
                                bounds=[(None, None)] * k + [(0, None)] * (2 * n))
    assert solution.status == 0
    beta = solution.x[:k]
    return beta, check_loss(y - x @ beta, tau, w)


def bandwidth(tau, n, rule):
    z, za = stats.norm.ppf(tau), stats.norm.ppf(0.975)
    if rule == "hsheather":
        return n ** (-1 / 3) * za ** (2 / 3) * (1.5 * stats.norm.pdf(z) ** 2
                                                / (2 * z ** 2 + 1)) ** (1 / 3)
    if rule == "bofinger":
        return n ** (-1 / 5) * (4.5 * stats.norm.pdf(z) ** 4 / (2 * z ** 2 + 1) ** 2) ** (1 / 5)
    return za * np.sqrt(tau * (1 - tau) / n)


def pctile(values, p, w=None):
    """Stata's summarize, detail percentile with (frequency or analytic) weights."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    cum = np.cumsum(np.ones(len(v)) if w is None else w[order])
    target = p * cum[-1]
    i = int(np.searchsorted(cum, target * (1 + 1e-12), side="right"))
    i = min(i, len(v) - 1)
    if i > 0 and abs(cum[i - 1] - target) <= 1e-9 * cum[-1]:
        return (v[i - 1] + v[i]) / 2
    return v[i]


def raw_quantile(values, tau, f=None):
    """qreg's "about" value: order statistic int(tau (N + 1)) of the (expanded) outcome."""
    expanded = np.sort(values if f is None else np.repeat(values, f.astype(int)))
    n = len(expanded)
    return expanded[min(max(int(np.floor(tau * (n + 1) + 1e-9)), 1), n) - 1]


KERNELS = {
    "epanechnikov": lambda u: np.where(np.abs(u) < 5 ** 0.5, 0.75 * (1 - u * u / 5) / 5 ** 0.5, 0),
    "epan2": lambda u: np.where(np.abs(u) < 1, 0.75 * (1 - u * u), 0),
    "biweight": lambda u: np.where(np.abs(u) < 1, 15 / 16 * (1 - u * u) ** 2, 0),
    "cosine": lambda u: np.where(np.abs(u) < 0.5, 1 + np.cos(2 * np.pi * u), 0),
    "gaussian": lambda u: np.exp(-u * u / 2) / np.sqrt(2 * np.pi),
    "parzen": lambda u: np.where(np.abs(u) <= 0.5, 4 / 3 - 8 * u * u + 8 * np.abs(u) ** 3,
                                 np.where(np.abs(u) <= 1, 8 * (1 - np.abs(u)) ** 3 / 3, 0)),
    "rectangle": lambda u: np.where(np.abs(u) < 1, 0.5, 0),
    "triangle": lambda u: np.where(np.abs(u) < 1, 1 - np.abs(u), 0),
}


def qreg_oracle(x, y, tau, *, kind="nonrobust", density="fitted", rule="hsheather",
                kernel="epanechnikov", w=None, wtype=None, groups=None):
    """Coefficients (LP) and covariance of qreg from the formulas of [R] qreg."""
    rows, k = x.shape
    if w is None:
        wn, n = np.ones(rows), rows
    elif wtype == "fweight":
        wn, n = w.astype(float), int(round(w.sum()))
    else:
        wn, n = w * rows / w.sum(), rows
    beta, objective = lp_fit(x, y, tau, wn)
    r = y - x @ beta
    r[np.abs(r) < 1e-9 * max(1.0, np.abs(y).max())] = 0.0
    h = bandwidth(tau, n, rule)
    out = {"beta": beta, "objective": objective, "h": h, "n": n, "resid": r}
    f_i = None
    if density == "fitted":
        spread = lp_fit(x, y, tau + h, wn)[0] - lp_fit(x, y, tau - h, wn)[0]
        xbar = (wn[:, None] * x).sum(axis=0) / wn.sum()
        sparsity = xbar @ spread / (2 * h)
        spacing = x @ spread
        f_i = np.where(spacing > 0, 2 * h / np.where(spacing > 0, spacing, 1.0), 0.0)
    elif density == "residual":
        sparsity = (pctile(r, tau + h, wn) - pctile(r, tau - h, wn)) / (2 * h)
    else:
        mean = (wn * r).sum() / wn.sum()
        sd = np.sqrt((wn * (r - mean) ** 2).sum() / (n - 1))
        iqr = pctile(r, 0.75, wn) - pctile(r, 0.25, wn)
        c = min(sd, iqr / 1.34) * (stats.norm.ppf(tau + h) - stats.norm.ppf(tau - h))
        f_i = KERNELS[kernel](r / c) / c
        sparsity = n / (wn * f_i).sum()
        out["c"] = c
    out["sparsity"] = sparsity
    if kind == "nonrobust":
        out["cov"] = tau * (1 - tau) * sparsity ** 2 * np.linalg.inv((x * wn[:, None]).T @ x)
        return out
    d_inv = np.linalg.inv((x * (wn * f_i)[:, None]).T @ x)
    if kind == "robust":
        m = wn if wtype == "fweight" else wn ** 2
        meat = tau * (1 - tau) * (x * m[:, None]).T @ x
    else:
        psi = wn * (tau - (r <= 0))
        meat = np.zeros((k, k))
        for g in np.unique(groups):
            score = (x[groups == g] * psi[groups == g][:, None]).sum(axis=0)
            meat += np.outer(score, score)
    out["cov"] = d_inv @ meat @ d_inv
    return out


def est(result):
    return np.array([c.estimate for c in result.coefficients])


def se(result):
    return np.array([c.std_error for c in result.coefficients])


def cov(result):
    return np.array(result.covariance_matrix)


def by_term(result, attribute="estimate"):
    return {c.term: getattr(c, attribute) for c in result.coefficients}


def check_t_inference(result, df, alpha=0.05):
    b, s = est(result), se(result)
    assert_allclose(s, np.sqrt(np.diag(cov(result))), rtol=1e-12)
    assert_allclose([c.statistic for c in result.coefficients], b / s, rtol=1e-12)
    assert_allclose([c.p_value for c in result.coefficients], 2 * stats.t.sf(np.abs(b / s), df),
                    rtol=1e-8, atol=1e-300)
    crit = stats.t.ppf(1 - alpha / 2, df)
    assert_allclose([c.ci_low for c in result.coefficients], b - crit * s, rtol=1e-8, atol=1e-12)
    assert_allclose([c.ci_high for c in result.coefficients], b + crit * s, rtol=1e-8, atol=1e-12)
    assert result.inference["use_t"] and result.inference["df_inference"] == df


# ---- Stata reference output: [R] qreg on auto.dta -----------------------------------------

# quantile: (coefficients, standard errors, raw sum of deviations, "about", pseudo R2 as printed)
STATA_QREG = {
    0.5: ({"weight": 3.933588, "length": -41.25191, "foreign": 3377.771, "Intercept": 344.6489},
          {"weight": 1.328718, "length": 45.46469, "foreign": 885.4198, "Intercept": 5182.394},
          71102.5, 4934.0, 0.2347),
    0.25: ({"weight": 1.831789, "length": 2.84556, "foreign": 2209.925, "Intercept": -1879.775},
           {"weight": 0.6328903, "length": 21.65558, "foreign": 421.7401, "Intercept": 2468.46},
           41912.75, 4187.0, 0.1697),
    0.75: ({"weight": 9.22291, "length": -220.7833, "foreign": 3595.133, "Intercept": 20242.9},
           {"weight": 1.785767, "length": 61.10352, "foreign": 1189.984, "Intercept": 6965.02},
           79860.75, 6342.0, 0.3840),
}


@pytest.mark.parametrize("tau", [0.5, 0.25, 0.75])
def test_qreg_reproduces_the_stata_manual_examples(tau):
    coefficients, errors, raw_sum, about, pseudo = STATA_QREG[tau]
    result = oe.qreg(data=AUTO, y="price", x=AUTO_X, quantile=tau)
    ours, ours_se = by_term(result), by_term(result, "std_error")
    x = np.column_stack([np.ones(74), AUTO[AUTO_X].to_numpy()])
    y = AUTO.price.to_numpy()
    stata_beta = np.array([coefficients[t] for t in ["Intercept", *AUTO_X]])
    for term in coefficients:
        assert_allclose(ours_se[term], errors[term], rtol=6e-7)
        if tau != 0.5 or term != "foreign":
            assert_allclose(ours[term], coefficients[term], rtol=6e-7)
    if tau == 0.5:
        # "note: alternate solutions exist": the median regression has an edge of minimizers
        # along the coefficient of foreign; Stata prints one end, any point is optimal.
        assert 3377.770 <= ours["foreign"] <= 3421.879
        assert result.extra["unique_solution"] is False
        assert any("not unique" in warning for warning in result.warnings)
        assert_allclose(result.metrics["sum_adev"], 54411.29, rtol=1e-7)
    # Stata's coefficients (7 digits) attain our minimum up to their rounding.
    assert_allclose(check_loss(y - x @ stata_beta, tau), result.metrics["sum_adev"], rtol=2e-6)
    assert_allclose(result.metrics["sum_rdev"], raw_sum, rtol=1e-12)
    assert result.metrics["raw_quantile"] == about
    assert round(result.metrics["pseudo_r_squared"], 4) == pseudo
    assert result.nobs == 74 and result.metrics["df_resid"] == 70
    check_t_inference(result, 70)
    if tau == 0.5:
        weight = result.coefficients[1]
        assert_allclose([weight.ci_low, weight.ci_high], [1.283543, 6.583632], rtol=6e-7)
        assert round(weight.statistic, 2) == 2.96 and round(weight.p_value, 3) == 0.004


def test_qreg_robust_reproduces_the_stata_manual_example():
    # qreg price weight length foreign, vce(robust): Hendricks-Koenker fitted densities.
    result = oe.qreg(data=AUTO, y="price", x=AUTO_X, covariance="robust")
    stata = {"weight": 1.694477, "length": 51.73571, "foreign": 728.5115, "Intercept": 5096.528}
    ours = by_term(result, "std_error")
    for term, value in stata.items():
        assert_allclose(ours[term], value, rtol=6e-7)
    assert result.extra["density_method"] == "fitted"
    assert result.extra["zero_density_observations"] == 3
    assert "Hendricks-Koenker" in result.inference["correction"]
    weight = result.coefficients[1]
    assert round(weight.statistic, 2) == 2.32 and round(weight.p_value, 3) == 0.023
    check_t_inference(result, 70)


def test_iqreg_point_estimates_reproduce_the_stata_manual_example():
    # iqreg price weight length foreign, q(.25 .75): differences of the two fits above.
    result = oe.iqreg(data=AUTO, y="price", x=AUTO_X, reps=5, seed=1)
    stata = {"weight": 7.391121, "length": -223.6288, "foreign": 1385.208, "Intercept": 22122.68}
    ours = by_term(result)
    for term, value in stata.items():
        assert_allclose(ours[term], value, rtol=6e-7)
    assert round(result.metrics["pseudo_r_squared_low"], 4) == 0.1697
    assert round(result.metrics["pseudo_r_squared_high"], 4) == 0.3840


def test_rreg_reproduces_the_stata_manual_example():
    # rreg mpg weight foreign ([R] rreg): coefficients, standard errors, F and iteration log.
    result = oe.rreg(data=AUTO, y="mpg", x=["weight", "foreign"])
    ours, ours_se = by_term(result), by_term(result, "std_error")
    assert_allclose(ours["weight"], -0.0063976, rtol=6e-6)
    assert_allclose(ours["foreign"], -3.182639, rtol=6e-7)
    assert_allclose(ours["Intercept"], 40.64022, rtol=6e-7)
    assert_allclose(ours_se["weight"], 0.0003718, rtol=1.5e-4)
    assert_allclose(ours_se["foreign"], 0.627964, rtol=6e-7)
    assert_allclose(ours_se["Intercept"], 1.263841, rtol=6e-7)
    test = result.tests["model"]
    assert round(test["statistic"], 2) == 168.32 and (test["df"], test["df2"]) == (2, 71)
    log = result.extra["iteration_log"]
    assert [entry["stage"] for entry in log] == ["huber"] * 4 + ["biweight"] * 4
    assert_allclose([entry["max_weight_change"] for entry in log],
                    [0.80280176, 0.2915438, 0.08911171, 0.02697328,
                     0.29186818, 0.11988101, 0.03315872, 0.00721325], rtol=0, atol=6e-9)
    foreign, constant = result.coefficients[2], result.coefficients[0]
    assert_allclose([foreign.ci_low, foreign.ci_high], [-4.434763, -1.930514], rtol=6e-7)
    assert_allclose([constant.ci_low, constant.ci_high], [38.1202, 43.16025], rtol=6e-7)
    assert [round(c.statistic, 2) for c in result.coefficients] == [32.16, -17.21, -5.07]
    assert result.nobs == 74 and result.metrics["n_dropped_cooks"] == 0
    check_t_inference(result, 71)
    # rreg mpg weight foreign, genwt(w): the manual lists the seven cars with w < .467.
    from openecon.econometrics.quantile import rreg as kernel

    x = torch.tensor(np.column_stack([np.ones(74), AUTO.weight, AUTO.foreign]))
    weights = kernel.robust_regression(x, torch.tensor(AUTO.mpg.to_numpy())).weights.numpy()
    lowest = np.argsort(weights, kind="stable")[:7]
    assert sorted(zip(AUTO.mpg[lowest[:3]], AUTO.weight[lowest[:3]], strict=True)) == [
        (35, 2020), (35, 2050), (41, 2040)]             # Datsun 210, Subaru, VW Diesel
    assert_allclose(weights[lowest], [0, 0, 0, 0.04429567, 0.08241943, 0.10443129, 0.28141296],
                    rtol=0, atol=6e-9)
    assert list(AUTO.mpg[lowest[3:]]) == [28, 21, 31, 21]
    assert int((weights < 0.467).sum()) == 7 and result.extra["weights"]["n_zero"] == 3


def _engel_cases():
    module = pytest.importorskip("statsmodels.regression.tests.results.results_quantile_regression")
    names = [name for name in dir(module)
             if hasattr(getattr(module, name), "kbwidth") and not name.endswith("q75")]
    return module, sorted(names)


def test_qreg_kernel_iid_reproduces_stata_on_engel_data():
    # Stata: qreg foodexp income, vce(iid, kernel(<kernel>) <bandwidth>) for 8 kernels x 3 rules.
    import statsmodels.api as sm

    module, names = _engel_cases()
    data = sm.datasets.engel.load_pandas().data
    assert len(names) == 24
    for name in names:
        stata = getattr(module, name)
        kernel, rule = name.split("_")
        result = oe.qreg(data=data, y="foodexp", x=["income"], density="kernel", kernel=kernel,
                         bandwidth=rule)
        metrics = result.metrics
        assert_allclose(metrics["bandwidth"], stata.bwidth, rtol=1e-12, err_msg=name)
        # Stata holds the data in single precision: agreement to about 1e-7.
        assert_allclose(metrics["kernel_bandwidth"], stata.kbwidth, rtol=1e-6, err_msg=name)
        assert_allclose(metrics["sparsity"], stata.sparsity, rtol=5e-6, err_msg=name)
        assert_allclose(metrics["density"], stata.f_r, rtol=5e-6, err_msg=name)
        errors = by_term(result, "std_error")
        assert_allclose([errors["income"], errors["Intercept"]], stata.table[:, 1], rtol=6e-6,
                        err_msg=name)
        assert_allclose([by_term(result)["income"], by_term(result)["Intercept"]],
                        stata.table[:, 0], rtol=2e-6)
        assert_allclose(metrics["raw_quantile"], stata.q_v, rtol=1e-7)
        assert metrics["df_resid"] == stata.df_r and result.nobs == stata.N
        assert round(metrics["pseudo_r_squared"], 4) == stata.psrsquared
        # That Stata release summed 2 tau |r| and 2 (1 - tau) |r|; current ones print half.
        assert_allclose(2 * metrics["sum_adev"], stata.sum_adev, rtol=1e-6)
        assert_allclose(2 * metrics["sum_rdev"], stata.sum_rdev, rtol=1e-6)
    # The .75 quantile (an older release whose "epanechnikov" was the epan2 formula).
    stata = module.epanechnikov_hsheather_q75
    result = oe.qreg(data=data, y="foodexp", x=["income"], quantile=0.75, density="kernel",
                     kernel="epan2")
    assert_allclose(result.metrics["bandwidth"], stata.bwidth, rtol=1e-12)
    assert_allclose(result.metrics["sparsity"], stata.sparsity, rtol=5e-6)
    assert_allclose(result.metrics["raw_quantile"], stata.q_v, rtol=1e-7)
    assert_allclose(se(result)[::-1], stata.table[:, 1], rtol=6e-6)


def test_qreg_robust_fitted_reproduces_r_quantreg_nid_on_engel_data():
    # R: summary(rq(foodexp ~ income, tau = .5, data = engel), se = "nid") prints
    # (Intercept) 81.48225 19.25066 4.23270 and income 0.56018 0.02828 19.81032:
    # the Hendricks-Koenker sandwich with the Hall-Sheather bandwidth.
    import statsmodels.api as sm

    data = sm.datasets.engel.load_pandas().data
    result = oe.qreg(data=data, y="foodexp", x=["income"], covariance="robust")
    assert_allclose(est(result), [81.48225, 0.56018], rtol=0, atol=6e-6)
    assert_allclose(se(result), [19.25066, 0.02828], rtol=0, atol=6e-6)
    assert_allclose([c.statistic for c in result.coefficients], [4.23270, 19.81032], atol=6e-6)
    assert result.extra["zero_density_observations"] == 0 and result.extra["unique_solution"]


# ---- qreg: coefficients and every covariance against the formulas -----------------------------


def designs():
    rng = np.random.default_rng(2024)
    out = {}
    n = 180
    x = np.column_stack([np.ones(n), rng.normal(size=n), rng.exponential(size=n)])
    y = x @ [1.0, 0.7, -0.4] + rng.standard_t(4, size=n) * (0.6 + 0.5 * x[:, 2])
    out["heteroskedastic"] = (x, y)
    n = 61
    x = np.column_stack([np.ones(n), rng.uniform(-2, 2, size=n)])
    out["small"] = (x, x @ [0.5, 2.0] + rng.laplace(size=n))
    n = 150
    x = np.column_stack([np.ones(n), 1e4 * rng.normal(size=n), 1e-4 * rng.normal(size=n)])
    out["badly_scaled"] = (x, 1e6 + x @ [0.0, 3e-4, 2e4] + 5 * rng.normal(size=n))
    return out


DESIGNS = designs()
GROUPS = {name: np.random.default_rng(5).integers(0, 14, size=len(y))
          for name, (x, y) in DESIGNS.items()}


def frame_of(x, y, **extra):
    data = {f"x{j}": x[:, j] for j in range(1, x.shape[1])}
    return pd.DataFrame({"y": y, **data, **extra})


@pytest.mark.parametrize("name", list(DESIGNS))
@pytest.mark.parametrize("tau", [0.2, 0.5, 0.8])
def test_qreg_covariances_match_the_formulas(name, tau):
    x, y = DESIGNS[name]
    groups = GROUPS[name]
    frame = frame_of(x, y, g=groups)
    cols = [c for c in frame.columns if c.startswith("x")]
    n, k = x.shape
    cases = [("nonrobust", "fitted"), ("nonrobust", "residual"), ("nonrobust", "kernel"),
             ("robust", "fitted"), ("robust", "kernel"), ("cluster", "kernel"),
             ("cluster", "fitted")]
    for kind, density in cases:
        for rule in ("hsheather", "bofinger"):
            oracle = qreg_oracle(x, y, tau, kind=kind, density=density, rule=rule, groups=groups)
            result = oe.qreg(data=frame, y="y", x=cols, quantile=tau, density=density,
                             bandwidth=rule, cluster="g" if kind == "cluster" else None,
                             covariance=None if kind == "cluster" else kind)
            label = f"{name} {tau} {kind} {density} {rule}"
            scale = np.abs(oracle["beta"]).max()
            assert_allclose(est(result), oracle["beta"], rtol=1e-7, atol=1e-9 * scale,
                            err_msg=label)
            assert_allclose(cov(result), oracle["cov"], rtol=2e-6, err_msg=label)
            assert_allclose(result.metrics["sum_adev"], oracle["objective"], rtol=1e-9)
            assert_allclose(result.metrics["bandwidth"], oracle["h"], rtol=1e-12)
            assert_allclose(result.metrics["sparsity"], oracle["sparsity"], rtol=1e-6,
                            err_msg=label)
            if density == "kernel":
                assert_allclose(result.metrics["kernel_bandwidth"], oracle["c"], rtol=1e-7)
            check_t_inference(result, n - k)
            q = raw_quantile(y, tau)
            assert result.metrics["raw_quantile"] == q
            assert_allclose(result.metrics["sum_rdev"], check_loss(y - q, tau), rtol=1e-12)
            assert_allclose(result.metrics["pseudo_r_squared"],
                            1 - oracle["objective"] / check_loss(y - q, tau), rtol=1e-9)
            assert result.metrics["df_model"] == k - 1 and result.metrics["df_resid"] == n - k
            # exactly k zero residuals: the reported coefficients are a vertex of the LP
            fitted = x @ est(result)
            assert int((np.abs(y - fitted) <= 1e-9 * max(1, np.abs(y).max())).sum()) >= k


@pytest.mark.parametrize("kernel", list(KERNELS))
def test_qreg_every_kernel_and_chamberlain_bandwidth(kernel):
    x, y = DESIGNS["heteroskedastic"]
    frame = frame_of(x, y)
    for kind in ("nonrobust", "robust"):
        oracle = qreg_oracle(x, y, 0.4, kind=kind, density="kernel", rule="chamberlain",
                             kernel=kernel)
        result = oe.qreg(data=frame, y="y", x=["x1", "x2"], quantile=0.4, covariance=kind,
                         kernel=kernel, bandwidth="chamberlain")
        assert result.extra["density_method"] == "kernel" and result.extra["kernel"] == kernel
        assert_allclose(cov(result), oracle["cov"], rtol=2e-6)


@pytest.mark.parametrize("wtype", ["aweight", "pweight", "fweight"])
def test_qreg_weights_match_the_formulas(wtype):
    x, y = DESIGNS["heteroskedastic"]
    rng = np.random.default_rng(9)
    w = rng.integers(1, 5, size=len(y)).astype(float) if wtype == "fweight" \
        else rng.uniform(0.2, 3.0, size=len(y))
    groups = GROUPS["heteroskedastic"]
    frame = frame_of(x, y, w=w, g=groups)
    cases = [("robust", "fitted"), ("robust", "kernel"), ("cluster", "kernel")]
    if wtype != "pweight":
        cases += [("nonrobust", "fitted"), ("nonrobust", "residual"), ("nonrobust", "kernel")]
    for kind, density in cases:
        oracle = qreg_oracle(x, y, 0.35, kind=kind, density=density, w=w, wtype=wtype,
                             groups=groups)
        result = oe.qreg(data=frame, y="y", x=["x1", "x2"], quantile=0.35, weights="w",
                         weight_type=wtype, density=density,
                         cluster="g" if kind == "cluster" else None,
                         covariance=None if kind == "cluster" else kind)
        label = f"{wtype} {kind} {density}"
        assert_allclose(est(result), oracle["beta"], rtol=1e-7, err_msg=label)
        assert_allclose(cov(result), oracle["cov"], rtol=2e-6, err_msg=label)
        assert_allclose(result.metrics["sum_adev"], oracle["objective"], rtol=1e-9)
        assert result.nobs == oracle["n"]
        check_t_inference(result, oracle["n"] - 3)
    default = oe.qreg(data=frame, y="y", x=["x1", "x2"], weights="w", weight_type=wtype)
    assert default.spec.covariance == ("robust" if wtype == "pweight" else "nonrobust")


def test_qreg_frequency_weights_equal_duplicated_rows():
    x, y = DESIGNS["heteroskedastic"]
    rng = np.random.default_rng(3)
    f = rng.integers(1, 4, size=len(y))
    groups = GROUPS["heteroskedastic"]
    weighted = frame_of(x, y, f=f.astype(float), g=groups)
    expanded = weighted.loc[np.repeat(np.arange(len(y)), f)].reset_index(drop=True)
    for options in ({}, {"density": "residual"}, {"density": "kernel"},
                    {"covariance": "robust"}, {"covariance": "robust", "kernel": "gaussian"},
                    {"cluster": "g"}, {"quantile": 0.8}):
        a = oe.qreg(data=weighted, y="y", x=["x1", "x2"], weights="f", weight_type="fweight",
                    **options)
        b = oe.qreg(data=expanded, y="y", x=["x1", "x2"], **options)
        assert a.nobs == b.nobs == int(f.sum())
        assert_allclose(est(a), est(b), rtol=1e-9, err_msg=str(options))
        assert_allclose(cov(a), cov(b), rtol=1e-7, err_msg=str(options))
        for key in ("sum_adev", "sum_rdev", "raw_quantile", "pseudo_r_squared", "sparsity",
                    "bandwidth"):
            assert_allclose(a.metrics[key], b.metrics[key], rtol=1e-8, err_msg=key)
        assert_allclose([c.p_value for c in a.coefficients], [c.p_value for c in b.coefficients],
                        rtol=1e-6)


def test_qreg_invariances():
    x, y = DESIGNS["heteroskedastic"]
    frame = frame_of(x, y)
    base = oe.qreg(data=frame, y="y", x=["x1", "x2"], quantile=0.3, covariance="robust")
    # Row order is irrelevant.
    shuffled = frame.sample(frac=1.0, random_state=4).reset_index(drop=True)
    again = oe.qreg(data=shuffled, y="y", x=["x1", "x2"], quantile=0.3, covariance="robust")
    assert_allclose(est(again), est(base), rtol=1e-10)
    assert_allclose(cov(again), cov(base), rtol=1e-8)
    # Rescaling a regressor rescales its coefficient and leaves t statistics unchanged.
    scaled = oe.qreg(data=frame.assign(x1=frame.x1 * 250.0), y="y", x=["x1", "x2"],
                     quantile=0.3, covariance="robust")
    assert_allclose(est(scaled) * [1, 250.0, 1], est(base), rtol=1e-9)
    assert_allclose([c.statistic for c in scaled.coefficients],
                    [c.statistic for c in base.coefficients], rtol=1e-7)
    # Equivariance: a + c y shifts the constant and scales everything by c > 0 ...
    moved = oe.qreg(data=frame.assign(y=3.0 + 2.5 * frame.y), y="y", x=["x1", "x2"],
                    quantile=0.3, covariance="robust")
    assert_allclose(est(moved), 2.5 * est(base) + [3.0, 0, 0], rtol=1e-9)
    assert_allclose(cov(moved), 6.25 * cov(base), rtol=1e-7)
    # ... and -y turns the tau-th quantile into minus the (1 - tau)-th one.
    upper = oe.qreg(data=frame, y="y", x=["x1", "x2"], quantile=0.7)
    mirrored = oe.qreg(data=frame.assign(y=-frame.y), y="y", x=["x1", "x2"], quantile=0.3)
    assert_allclose(est(mirrored), -est(upper), rtol=1e-9)
    assert_allclose(cov(mirrored), cov(upper), rtol=1e-7)
    # The scale of analytic weights is irrelevant.
    w = np.random.default_rng(1).uniform(0.5, 2, size=len(y))
    one = oe.qreg(data=frame.assign(w=w), y="y", x=["x1", "x2"], weights="w",
                  weight_type="aweight")
    two = oe.qreg(data=frame.assign(w=w * 1e6), y="y", x=["x1", "x2"], weights="w",
                  weight_type="aweight")
    assert_allclose(est(two), est(one), rtol=1e-9)
    assert_allclose(cov(two), cov(one), rtol=1e-7)
    assert_allclose(two.metrics["sum_adev"], one.metrics["sum_adev"], rtol=1e-9)


def test_qreg_design_handling_against_the_linear_program():
    rng = np.random.default_rng(12)
    n = 140
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                          "c": rng.choice(["u", "v", "w"], size=n)})
    frame["dup"] = 3 * frame.x1 - 2
    frame["y"] = 1 + frame.x1 + (frame.c == "w") * 0.8 + rng.gumbel(size=n)
    frame.loc[[5, 17, 60], "x2"] = np.nan
    result = oe.qreg(data=frame, y="y", x=["x1", "dup", "x2", "c"], categorical=["c"],
                     quantile=0.6, missing="drop")
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "x2", "c[v]", "c[w]"]
    assert result.provenance["omitted_terms"] == ["dup"]
    kept = frame.dropna().reset_index(drop=True)
    x = np.column_stack([np.ones(len(kept)), kept.x1, kept.x2, kept.c == "v", kept.c == "w"]
                        ).astype(float)
    beta, objective = lp_fit(x, kept.y.to_numpy(), 0.6)
    assert_allclose(result.metrics["sum_adev"], objective, rtol=1e-10)
    if result.extra["unique_solution"]:
        assert_allclose(est(result), beta, rtol=1e-7, atol=1e-9)
    assert result.nobs == n - 3 and result.dropped_rows == 3
    assert result.sample_positions == [i for i in range(n) if i not in (5, 17, 60)]
    # The constant-only model returns an order statistic that minimizes the check loss.
    only = oe.qreg(data=frame, y="y", x=[], quantile=0.3)
    assert_allclose(only.metrics["sum_adev"],
                    min(check_loss(frame.y.to_numpy() - v, 0.3) for v in frame.y), rtol=1e-12)
    assert est(only)[0] in set(frame.y)


def test_qreg_agrees_with_statsmodels_quantreg():
    import statsmodels.api as sm

    x, y = DESIGNS["heteroskedastic"]
    frame = frame_of(x, y)
    for tau in (0.25, 0.5, 0.9):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = sm.QuantReg(y, x).fit(q=tau, vcov="iid", kernel="epa",
                                              bandwidth="hsheather", max_iter=5000,
                                              p_tol=1e-10)
        result = oe.qreg(data=frame, y="y", x=["x1", "x2"], quantile=tau, density="kernel",
                         kernel="epan2")
        # statsmodels iterates IRLS to a tolerance; ours is the exact vertex.
        assert_allclose(est(result), reference.params, rtol=2e-4, atol=2e-4)
        assert check_loss(y - x @ est(result), tau) <= check_loss(y - x @ reference.params, tau)
        # Same sparsity estimator up to its sd/IQR conventions (ddof, interpolation).
        assert_allclose(se(result), reference.bse, rtol=0.03)
        assert_allclose(result.metrics["pseudo_r_squared"], reference.prsquared, atol=2e-3)


# ---- bootstrap commands -----------------------------------------------------------------------


def bootstrap_oracle(x, y, taus, reps, seed, groups=None):
    """Redraw the resamples and solve every expanded replicate as an explicit LP."""
    generator = torch.Generator().manual_seed(seed)
    n = len(y)
    draws = []
    for _ in range(reps):
        if groups is None:
            rows = torch.randint(n, (n,), generator=generator).numpy()
        else:
            labels = pd.factorize(groups)[0]
            count = labels.max() + 1
            picked = torch.randint(int(count), (int(count),), generator=generator).numpy()
            rows = np.concatenate([np.flatnonzero(labels == g) for g in picked])
        draws.append(np.concatenate([lp_fit(x[rows], y[rows], tau)[0] for tau in taus]))
    draws = np.array(draws)
    centered = draws - draws.mean(axis=0)
    return draws, centered.T @ centered / (reps - 1)


def test_bootstrap_commands_match_redrawn_linear_programs():
    x, y = DESIGNS["heteroskedastic"]
    n, k = x.shape
    frame = frame_of(x, y, g=GROUPS["heteroskedastic"])
    cols = ["x1", "x2"]
    reps, seed = 12, 321
    draws, v = bootstrap_oracle(x, y, [0.25, 0.5, 0.75], reps, seed)
    sq = oe.sqreg(data=frame, y="y", x=cols, quantiles=[0.25, 0.5, 0.75], reps=reps, seed=seed)
    expected = np.concatenate([lp_fit(x, y, tau)[0] for tau in (0.25, 0.5, 0.75)])
    assert_allclose(est(sq), expected, rtol=1e-7)
    assert_allclose(cov(sq), v, rtol=1e-6, atol=1e-12)
    assert [c.term for c in sq.coefficients][:4] == ["q25:Intercept", "q25:x1", "q25:x2",
                                                      "q50:Intercept"]
    assert [c.equation for c in sq.coefficients] == ["q25"] * 3 + ["q50"] * 3 + ["q75"] * 3
    check_t_inference(sq, n - k)
    assert sq.metrics["reps"] == reps and sq.extra["bootstrap"]["seed"] == seed
    for tau, label in ((0.25, "q25"), (0.5, "q50"), (0.75, "q75")):
        q = raw_quantile(y, tau)
        assert_allclose(sq.metrics[f"pseudo_r_squared_{label}"],
                        1 - lp_fit(x, y, tau)[1] / check_loss(y - q, tau), rtol=1e-9)
    # bsqreg uses the same stream: its covariance is the median block of a one-quantile run.
    _, v_median = bootstrap_oracle(x, y, [0.5], reps, seed)
    bs = oe.bsqreg(data=frame, y="y", x=cols, reps=reps, seed=seed)
    assert_allclose(est(bs), expected[3:6], rtol=1e-7)
    assert_allclose(cov(bs), v_median, rtol=1e-6, atol=1e-12)
    assert_allclose(cov(bs), cov(sq)[3:6, 3:6], rtol=1e-6)
    check_t_inference(bs, n - k)
    # iqreg: difference of the two fits; variance of the replicate differences.
    iq = oe.iqreg(data=frame, y="y", x=cols, quantiles=[0.25, 0.75], reps=reps, seed=seed)
    two, _ = bootstrap_oracle(x, y, [0.25, 0.75], reps, seed)
    difference = two[:, 3:] - two[:, :3]
    centered = difference - difference.mean(axis=0)
    assert_allclose(est(iq), expected[6:] - expected[:3], rtol=1e-7)
    assert_allclose(cov(iq), centered.T @ centered / (reps - 1), rtol=1e-6, atol=1e-12)
    # ... which is the contrast of sqreg's joint covariance when the resamples coincide.
    pair = oe.sqreg(data=frame, y="y", x=cols, quantiles=[0.25, 0.75], reps=reps, seed=seed)
    contrast = np.hstack([-np.eye(3), np.eye(3)])
    assert_allclose(cov(iq), contrast @ cov(pair) @ contrast.T, rtol=1e-6, atol=1e-12)
    check_t_inference(iq, n - k)
    # Cluster bootstrap: whole clusters are resampled.
    groups = frame.g.to_numpy()
    _, v_cluster = bootstrap_oracle(x, y, [0.5], reps, seed, groups=groups)
    clustered = oe.bsqreg(data=frame, y="y", x=cols, reps=reps, seed=seed, cluster="g")
    assert_allclose(cov(clustered), v_cluster, rtol=1e-6, atol=1e-12)
    assert clustered.extra["bootstrap"]["resampling"] == "clusters"
    # Determinism and seed bookkeeping.
    assert_allclose(cov(oe.bsqreg(data=frame, y="y", x=cols, reps=reps, seed=seed)), cov(bs),
                    rtol=0, atol=0)
    assert not np.allclose(cov(oe.bsqreg(data=frame, y="y", x=cols, reps=reps, seed=seed + 1)),
                           cov(bs))
    unseeded = oe.bsqreg(data=frame, y="y", x=cols, reps=reps)
    replay = oe.bsqreg(data=frame, y="y", x=cols, reps=reps,
                       seed=unseeded.extra["bootstrap"]["seed"])
    assert_allclose(cov(replay), cov(unseeded), rtol=0, atol=0)


def test_bootstrap_standard_errors_approach_the_analytic_ones():
    rng = np.random.default_rng(77)
    n = 4000
    frame = pd.DataFrame({"x1": rng.normal(size=n)})
    frame["y"] = 1 + 0.5 * frame.x1 + rng.normal(size=n)
    analytic = oe.qreg(data=frame, y="y", x=["x1"])
    boot = oe.bsqreg(data=frame, y="y", x=["x1"], reps=150, seed=5)
    assert_allclose(se(boot), se(analytic), rtol=0.2)
    # Normal errors: the asymptotic standard error of the median slope is sqrt(pi/2)/sqrt(n).
    assert_allclose(se(analytic)[1], np.sqrt(np.pi / 2 / n), rtol=0.1)


# ---- rreg -----------------------------------------------------------------------------------


def test_rreg_equivariance_and_ols_limit():
    rng = np.random.default_rng(31)
    n = 220
    frame = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.uniform(size=n)})
    frame["y"] = 2 - frame.x1 + 3 * frame.x2 + rng.standard_t(3, size=n)
    base = oe.rreg(data=frame, y="y", x=["x1", "x2"])
    moved = oe.rreg(data=frame.assign(y=5 - 4 * frame.y), y="y", x=["x1", "x2"])
    assert_allclose(est(moved), -4 * est(base) + [5, 0, 0], rtol=1e-8)
    assert_allclose(cov(moved), 16 * cov(base), rtol=1e-8)
    assert_allclose(moved.tests["model"]["statistic"], base.tests["model"]["statistic"],
                    rtol=1e-8)
    shuffled = frame.sample(frac=1.0, random_state=2).reset_index(drop=True)
    assert_allclose(est(oe.rreg(data=shuffled, y="y", x=["x1", "x2"])), est(base), rtol=1e-9)
    rescaled = oe.rreg(data=frame.assign(x1=frame.x1 * 1e5), y="y", x=["x1", "x2"])
    assert_allclose(est(rescaled) * [1, 1e5, 1], est(base), rtol=1e-8)
    # With a huge tuning constant no observation is downweighted in the biweight stage, the
    # fit is OLS and the pseudovalue covariance tends to the classical one.
    x = np.column_stack([np.ones(n), frame.x1, frame.x2])
    y = frame.y.to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    classical = resid @ resid / (n - 3) * np.linalg.inv(x.T @ x)
    loose = oe.rreg(data=frame, y="y", x=["x1", "x2"], tune=1e7, tolerance=1e-9)
    assert_allclose(est(loose), beta, rtol=1e-6, atol=1e-8)
    assert_allclose(cov(loose), classical, rtol=1e-6)
    assert loose.extra["weights"]["min"] > 1 - 1e-6


# ---- nl -------------------------------------------------------------------------------------


def nl_frame():
    rng = np.random.default_rng(99)
    n = 160
    frame = pd.DataFrame({"x": rng.uniform(0.2, 6, size=n), "k": rng.lognormal(0.5, 0.4, size=n),
                          "l": rng.lognormal(0.2, 0.5, size=n)})
    frame["decay"] = 1.5 + 2.5 * np.exp(-0.8 * frame.x) + rng.normal(0, 0.15, n) * (1 + frame.x / 4)
    frame["mm"] = 4 * frame.x / (1.3 + frame.x) + rng.normal(0, 0.2, n)
    ces = 0.5 * frame.k ** -1.2 + 0.5 * frame.l ** -1.2
    frame["ces"] = 2.0 - np.log(ces) / 1.2 + rng.normal(0, 0.1, n)
    frame["growth"] = 5 / (1 + np.exp(-1.4 * (frame.x - 3))) + rng.normal(0, 0.2, n)
    frame["g"] = rng.integers(0, 18, size=n)
    frame["w"] = rng.uniform(0.3, 2.5, size=n)
    frame["f"] = rng.integers(1, 4, size=n).astype(float)
    return frame


NL = nl_frame()
# name: (outcome, formula, start, numpy model written independently of the parser)
NL_MODELS = {
    "decay": ("decay", "{b0} + {b1} * exp(-{b2} * x)", {"b0": 1, "b1": 2, "b2": 0.5},
              lambda t, d: t[0] + t[1] * np.exp(-t[2] * d["x"])),
    "michaelis": ("mm", "{vmax} * x / ({km} + x)", {"vmax": 3, "km": 1},
                  lambda t, d: t[0] * d["x"] / (t[1] + d["x"])),
    "ces": ("ces", "{b0} - 1/{rho=1} * ln({d=0.4} * k^(-{rho}) + (1 - {d}) * l^(-{rho}))",
            {"b0": 1}, lambda t, d: t[0] - np.log(t[2] * d["k"] ** (-t[1])
                                                  + (1 - t[2]) * d["l"] ** (-t[1])) / t[1]),
    "growth": ("growth", "{top} * invlogit({rate} * (x - {mid}))",
               {"top": 4, "rate": 1, "mid": 2.5},
               lambda t, d: t[0] / (1 + np.exp(-t[1] * (d["x"] - t[2])))),
}


def complex_step_jacobian(model, theta, columns):
    theta = np.asarray(theta, dtype=float)
    out = np.empty((len(next(iter(columns.values()))), len(theta)))
    for j in range(len(theta)):
        shifted = theta.astype(complex)
        shifted[j] += 1e-30j
        out[:, j] = np.imag(model(shifted, columns)) / 1e-30
    return out


def nl_oracle(name, *, w=None, wtype=None, kind="nonrobust", groups=None, frame=NL):
    outcome, _, _, model = NL_MODELS[name]
    columns = {c: frame[c].to_numpy() for c in ("x", "k", "l")}
    y = frame[outcome].to_numpy()
    rows = len(y)
    if w is None:
        wn, n = np.ones(rows), rows
    elif wtype == "fweight":
        wn, n = w, int(round(w.sum()))
    else:
        wn, n = w * rows / w.sum(), rows
    truth = {"decay": [1.5, 2.5, 0.8], "michaelis": [4, 1.3], "ces": [2.0, 1.2, 0.5],
             "growth": [5, 1.4, 3]}[name]
    solution = optimize.least_squares(lambda t: np.sqrt(wn) * (y - model(t, columns)), truth,
                                      jac="3-point", xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                      x_scale="jac")
    theta = solution.x
    k = len(theta)
    jac = complex_step_jacobian(model, theta, columns)
    r = y - model(theta, columns)
    rss = (wn * r * r).sum()
    bread = np.linalg.inv((jac * wn[:, None]).T @ jac)
    out = {"theta": theta, "rss": rss, "n": n, "k": k, "jac": jac, "resid": r}
    if kind == "nonrobust":
        out["cov"] = rss / (n - k) * bread
        return out
    leverage = wn * np.einsum("ij,jk,ik->i", jac, bread, jac)
    if kind == "cluster":
        scores = jac * (wn * r)[:, None]
        meat = np.zeros((k, k))
        for g in np.unique(groups):
            s = scores[groups == g].sum(axis=0)
            meat += np.outer(s, s)
        count = len(np.unique(groups))
        out["cov"] = count / (count - 1) * (n - 1) / (n - k) * bread @ meat @ bread
        out["df"] = count - 1
        return out
    adjusted = {"robust": r, "HC2": r / np.sqrt(1 - leverage), "HC3": r / (1 - leverage)}[kind]
    # A frequency weight replicates the observation's squared score; other weights scale it.
    weight = wn if wtype == "fweight" else wn ** 2
    meat = (jac * (weight * adjusted ** 2)[:, None]).T @ jac
    factor = n / (n - k) if kind == "robust" else 1.0
    out["cov"] = factor * bread @ meat @ bread
    return out


@pytest.mark.parametrize("name", list(NL_MODELS))
def test_nl_matches_scipy_and_the_covariance_formulas(name):
    outcome, formula, start, _ = NL_MODELS[name]
    groups = NL.g.to_numpy()
    for kind in ("nonrobust", "robust", "HC2", "HC3", "cluster"):
        oracle = nl_oracle(name, kind=kind, groups=groups)
        result = oe.nl(data=NL, y=outcome, formula=formula, start=start,
                       covariance=None if kind == "cluster" else kind,
                       cluster="g" if kind == "cluster" else None)
        n, k = oracle["n"], oracle["k"]
        assert_allclose(est(result), oracle["theta"], rtol=2e-7, err_msg=f"{name} {kind}")
        assert_allclose(cov(result), oracle["cov"], rtol=2e-6, err_msg=f"{name} {kind}")
        check_t_inference(result, oracle.get("df", n - k))
        assert_allclose(result.metrics["rss"], oracle["rss"], rtol=1e-10)
        assert_allclose(result.metrics["rmse"], np.sqrt(oracle["rss"] / (n - k)), rtol=1e-10)
        assert result.metrics["df_resid"] == n - k
        assert result.provenance["solver_diagnostics"]["converged"] is True
    y = NL[outcome].to_numpy()
    # R-squared: centered when a parameter is an additive constant (decay, ces), else not.
    constant = {"decay": "b0", "ces": "b0", "michaelis": None, "growth": None}[name]
    assert result.extra["constant_term"] == constant
    c = int(constant is not None)
    tss = ((y - y.mean()) ** 2).sum() if c else (y ** 2).sum()
    r2 = 1 - oracle["rss"] / tss
    assert_allclose(result.metrics["r_squared"], r2, rtol=1e-9)
    assert_allclose(result.metrics["adjusted_r_squared"], 1 - (1 - r2) * (n - c) / (n - k),
                    rtol=1e-9)
    assert result.metrics["df_model"] == k - c
    assert_allclose(result.metrics["log_likelihood"],
                    -0.5 * n * (np.log(2 * np.pi * oracle["rss"] / n) + 1), rtol=1e-10)


def test_nl_statsmodels_agrees_on_the_linearized_model():
    import statsmodels.api as sm

    oracle = nl_oracle("decay")
    jac, r = oracle["jac"], oracle["resid"]
    groups = NL.g.to_numpy()
    for kind, options in (("robust", {"cov_type": "HC1"}), ("HC2", {"cov_type": "HC2"}),
                          ("HC3", {"cov_type": "HC3"}),
                          ("cluster", {"cov_type": "cluster", "cov_kwds": {"groups": groups}})):
        # Gauss-Newton regression: OLS of the residuals on the Jacobian has the same sandwich.
        reference = sm.OLS(r, jac).fit(**options)
        result = oe.nl(data=NL, y="decay", formula=NL_MODELS["decay"][1],
                       start=NL_MODELS["decay"][2], covariance=None if kind == "cluster" else kind,
                       cluster="g" if kind == "cluster" else None)
        assert_allclose(cov(result), reference.cov_params(), rtol=2e-6, err_msg=kind)


@pytest.mark.parametrize("wtype", ["aweight", "pweight", "fweight"])
def test_nl_weights(wtype):
    outcome, formula, start, _ = NL_MODELS["decay"]
    w = (NL.f if wtype == "fweight" else NL.w).to_numpy()
    column = "f" if wtype == "fweight" else "w"
    groups = NL.g.to_numpy()
    kinds = ["robust", "cluster"] + ([] if wtype == "pweight" else ["nonrobust", "HC2", "HC3"])
    for kind in kinds:
        if wtype == "fweight" and kind in {"HC2", "HC3"}:
            continue        # checked through the duplicated-rows identity below
        oracle = nl_oracle("decay", w=w, wtype=wtype, kind=kind, groups=groups)
        result = oe.nl(data=NL, y=outcome, formula=formula, start=start, weights=column,
                       weight_type=wtype, covariance=None if kind == "cluster" else kind,
                       cluster="g" if kind == "cluster" else None)
        assert_allclose(est(result), oracle["theta"], rtol=2e-7, err_msg=f"{wtype} {kind}")
        assert_allclose(cov(result), oracle["cov"], rtol=2e-6, err_msg=f"{wtype} {kind}")
        assert result.nobs == oracle["n"]
    if wtype == "fweight":
        expanded = NL.loc[np.repeat(np.arange(len(NL)), NL.f.astype(int))].reset_index(drop=True)
        for options in ({}, {"covariance": "robust"}, {"covariance": "HC2"},
                        {"covariance": "HC3"}, {"cluster": "g"}):
            a = oe.nl(data=NL, y=outcome, formula=formula, start=start, weights="f",
                      weight_type="fweight", **options)
            b = oe.nl(data=expanded, y=outcome, formula=formula, start=start, **options)
            assert_allclose(est(a), est(b), rtol=1e-8)
            assert_allclose(cov(a), cov(b), rtol=1e-6, err_msg=str(options))
            for key in ("r_squared", "adjusted_r_squared", "rmse", "rss", "log_likelihood"):
                assert_allclose(a.metrics[key], b.metrics[key], rtol=1e-8, err_msg=key)
    if wtype == "pweight":
        assert oe.nl(data=NL, y=outcome, formula=formula, start=start, weights="w",
                     weight_type="pweight").spec.covariance == "robust"
    if wtype == "aweight":
        scaled = oe.nl(data=NL.assign(w=NL.w * 1e4), y=outcome, formula=formula, start=start,
                       weights="w", weight_type="aweight")
        base = oe.nl(data=NL, y=outcome, formula=formula, start=start, weights="w",
                     weight_type="aweight")
        assert_allclose(cov(scaled), cov(base), rtol=1e-7)


def test_nl_analytic_jacobian_matches_numerical_derivatives():
    from openecon.econometrics.quantile import formula as formulas

    rng = np.random.default_rng(4)
    n = 40
    columns = {"x": rng.uniform(0.5, 2.0, size=n), "z": rng.uniform(-1.0, 1.0, size=n)}
    tensors = {name: torch.tensor(values) for name, values in columns.items()}
    cases = [
        ("{a} + {b}*x - {c}/x + x/{c} + {a}*{b}*z", [0.7, -1.2, 1.9]),
        ("{a}*x^{b} + {b}^x + ({a}*x)^2 + x^3*{c} + (x + {c})^-1.5", [1.3, 0.8, 2.0]),
        ("exp({a}*z) + ln({b} + x) + log(x*{b}) + sqrt({a}^2 + x)", [0.6, 1.4]),
        ("sin({a}*x) + cos({b}*z + {a}) + tan({a}*z/3)", [0.9, 0.4]),
        ("abs({a}*z - {b}) + expit({a} + {b}*x) + invlogit(z*{b})", [0.8, 0.35]),
        ("normal({a} + {b}*z) * {c} + normalden({b}*x - {a}) + -{c}^2", [0.3, 0.7, 1.1]),
        ("{a}*exp(-{b}*x) / (1 + {c}*exp(-{b}*x)) ** 2", [2.0, 0.5, 0.3]),
        ("--{a} + +{b}*z - -x*{a} + 2e-1*{b} + {a=3}*1", [1.5, -0.5]),
    ]
    for text, theta in cases:
        parsed = formulas.parse(text)
        point = torch.tensor(theta, dtype=torch.float64)
        value, jacobian = formulas.evaluate(parsed, tensors, point, n)

        def function(t, parsed=parsed):
            return formulas.evaluate(parsed, tensors, torch.tensor(t), n, jacobian=False)[0].numpy()

        numeric = np.empty((n, len(theta)))
        for j in range(len(theta)):
            step = np.zeros(len(theta))
            step[j] = 1e-5 * max(1.0, abs(theta[j]))
            # Richardson-extrapolated central difference: error O(h^4).
            wide = (function(np.add(theta, 2 * step)) - function(np.subtract(theta, 2 * step)))
            near = (function(np.add(theta, step)) - function(np.subtract(theta, step)))
            numeric[:, j] = (8 * near - wide) / (12 * step[j])
        assert_allclose(jacobian.numpy(), numeric, rtol=2e-7, atol=2e-9, err_msg=text)
        assert jacobian.requires_grad is False and value.requires_grad is False
    # Values agree with the same expressions written in NumPy.
    x, z = columns["x"], columns["z"]
    parsed = formulas.parse("{a}*x^{b} + {b}^x + ({a}*x)^2 + x^3*{c} + (x + {c})^-1.5")
    theta = torch.tensor([1.3, 0.8, 2.0], dtype=torch.float64)
    value = formulas.evaluate(parsed, tensors, theta, n, jacobian=False)[0]
    assert_allclose(value.numpy(), 1.3 * x ** 0.8 + 0.8 ** x + (1.3 * x) ** 2 + x ** 3 * 2.0
                    + (x + 2.0) ** -1.5, rtol=1e-13)
    parsed = formulas.parse("normal({a} + {b}*z) * -x^2 + normalden(z) - 2^-(x^2)")
    theta = torch.tensor([0.3, 0.7], dtype=torch.float64)
    value = formulas.evaluate(parsed, tensors, theta, n, jacobian=False)[0]
    assert_allclose(value.numpy(), stats.norm.cdf(0.3 + 0.7 * z) * -(x ** 2) + stats.norm.pdf(z)
                    - 2.0 ** -(x ** 2), rtol=1e-13)


def test_nl_is_independent_of_starting_values_and_units():
    outcome, formula, start, _ = NL_MODELS["decay"]
    base = oe.nl(data=NL, y=outcome, formula=formula, start=start)
    other = oe.nl(data=NL, y=outcome, formula=formula, start={"b0": 4, "b1": -1, "b2": 2.0})
    assert_allclose(est(other), est(base), rtol=1e-7)
    assert_allclose(cov(other), cov(base), rtol=1e-6)
    # x measured in other units: b2 scales inversely, everything else is unchanged.
    rescaled = oe.nl(data=NL.assign(x=NL.x * 1000.0), y=outcome, formula=formula,
                     start={"b0": 1, "b1": 2, "b2": 0.0005})
    assert_allclose(est(rescaled) * [1, 1, 1000.0], est(base), rtol=1e-6)
    assert_allclose([c.statistic for c in rescaled.coefficients],
                    [c.statistic for c in base.coefficients], rtol=1e-5)
    # A linear model written as a formula is OLS, with regress's R-squared and standard errors.
    linear = oe.nl(data=NL, y="decay", formula="{c} + {s}*x")
    x = np.column_stack([np.ones(len(NL)), NL.x])
    y = NL.decay.to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ beta
    assert_allclose(est(linear), beta, rtol=1e-9)
    assert_allclose(cov(linear), resid @ resid / (len(y) - 2) * np.linalg.inv(x.T @ x), rtol=1e-8)
    assert_allclose(linear.metrics["r_squared"],
                    1 - resid @ resid / ((y - y.mean()) ** 2).sum(), rtol=1e-10)
    assert any("started at 0" in warning for warning in linear.warnings)
