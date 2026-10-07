"""Independent oracles for ivprobit and ivtobit (verification stage).

Maximum likelihood: the joint density ``f(y1 | y2, z) f(y2 | z)`` is written
directly in the covariance matrix ``Sigma`` of ``(u, v)`` (its free elements
are the parameters), not in the recursive working parameterization of the
implementation and not in its reported ``atanh`` / ``ln`` form; the reported
ancillary parameters and their covariance follow from the Jacobian of the map.
Two-step: Newey's (1987) minimum chi-squared formulas of Stata's Methods and
formulas on statsmodels OLS / Probit fits (and the tobit oracle of
``test_econ_limited_oracle``).
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from scipy import special, stats
from test_econ_limited_oracle import (
    KINDS, LOG_SQRT_2PI, check_table, check_test, covariance_of, dummies, estimates,
    jacobian_cs, likelihood_pieces, likelihood_weights, newton, options_for, stata_covariance,
    tobit_olsen, wald,
)

import openecon as oe
from openecon.analysis import AnalysisError

SOLVED = {}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(770)
    n = 460
    sizes = rng.integers(4, 14, size=64)
    firm = np.repeat(np.arange(len(sizes)), sizes)[:n]
    firm = np.concatenate([firm, np.full(n - len(firm), len(sizes))])
    frame = pd.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.uniform(-1, 1, size=n),
        "z1": rng.normal(size=n), "z2": rng.normal(size=n), "z3": rng.exponential(size=n),
        "firm": firm, "fw": rng.integers(1, 4, size=n).astype(float),
        "aw": rng.gamma(3.0, 0.5, size=n) + 0.2,
    })
    sigma = np.array([[1.0, 0.45, -0.3], [0.45, 1.4, 0.35], [-0.3, 0.35, 0.8]])
    shocks = rng.multivariate_normal(np.zeros(3), sigma, size=n)
    frame["e1"] = 0.3 + 0.5 * frame.x1 + 0.8 * frame.z1 - 0.5 * frame.z2 + shocks[:, 1]
    frame["e2"] = -0.2 + 0.3 * frame.x2 + 0.6 * frame.z2 + 0.7 * frame.z3 + shocks[:, 2]
    latent = 0.2 + 0.6 * frame.x1 - 0.4 * frame.x2 - 0.5 * frame.e1 + 0.4 * frame.e2 \
        + shocks[:, 0]
    frame["yb"] = (latent > 0) * 1.0
    frame["yt"] = (1.0 + 1.5 * latent).clip(0.0, 5.0)
    return frame


XS = ["x1", "x2"]
IV = ["z1", "z2", "z3"]


def blocks(frame, endog, exog=XS, instruments=IV):
    x1, x_names = dummies(frame, exog)
    z, z_names = dummies(frame, [*exog, *instruments])
    y2 = frame[list(endog)].to_numpy(dtype=float)
    return x1, x_names, z, z_names, y2


def joint_likelihood(x1, z, y2, outcome, *, tobit):
    """Per-observation log likelihood in ``(d, Pi, Sigma elements)`` and the reported map.

    ``Sigma = Var(u, v)``. Parameters after ``d`` and ``vec(Pi)``: ``Cov(v, u)`` [p], the
    lower triangle of ``Var(v)`` row by row [p (p + 1) / 2] and, for the tobit, ``Var(u)``.
    ``outcome(index, sd)`` returns the conditional log likelihood of ``y1``.
    """
    n, p = y2.shape
    k1, kz = x1.shape[1], z.shape[1]
    kw = k1 + p
    lower = [(i, j) for i in range(p) for j in range(i + 1)]

    def unpack(theta):
        delta = theta[:kw]
        pi = theta[kw:kw + p * kz].reshape(p, kz)
        rest = theta[kw + p * kz:]
        cov_vu = rest[:p]
        var_v = np.zeros((p, p), dtype=theta.dtype)
        for value, (i, j) in zip(rest[p:p + len(lower)], lower, strict=True):
            var_v[i, j] = var_v[j, i] = value
        var_u = rest[p + len(lower)] if tobit else 1.0
        return delta, pi, cov_vu, var_v, var_u

    def obs(theta):
        delta, pi, cov_vu, var_v, var_u = unpack(theta)
        v = y2 - z @ pi.T
        inverse = np.linalg.inv(var_v)
        a = inverse @ cov_vu
        omega = np.sqrt(var_u - cov_vu @ a)
        index = x1 @ delta[:k1] + y2 @ delta[k1:] + v @ a
        normal = -p * LOG_SQRT_2PI - 0.5 * np.log(np.linalg.det(var_v)) \
            - 0.5 * ((v @ inverse) * v).sum(axis=1)
        return outcome(index, omega) + normal

    def reported(theta):
        delta, pi, cov_vu, var_v, var_u = unpack(theta)
        sd = [np.sqrt(var_u), *(np.sqrt(var_v[i, i]) for i in range(p))]
        ancillary = []
        for i in range(1, p + 1):
            for j in range(i):
                cov = cov_vu[i - 1] if j == 0 else var_v[i - 1, j - 1]
                ancillary.append(np.arctanh(cov / (sd[i] * sd[j])))
        ancillary.extend(np.log(sd[i]) for i in range(0 if tobit else 1, p + 1))
        return np.concatenate([delta, pi.reshape(-1), np.array(ancillary)])

    names = [f"/athrho{i + 1}_{j + 1}" for i in range(1, p + 1) for j in range(i)]
    names += [f"/lnsigma{i + 1}" for i in range(0 if tobit else 1, p + 1)]
    return obs, reported, names, lower


def probit_outcome(y):
    sign = 2 * y - 1
    return lambda index, sd: np.log(special.ndtr(sign * index / sd))


def tobit_outcome(y, ll, ul):
    left = np.zeros(len(y), dtype=bool) if ll is None else y <= ll
    right = np.zeros(len(y), dtype=bool) if ul is None else y >= ul
    low, high = (0.0 if ll is None else ll), (0.0 if ul is None else ul)

    def outcome(index, sd):
        exact = -0.5 * ((y - index) / sd) ** 2 - np.log(sd) - LOG_SQRT_2PI
        return np.where(left, np.log(special.ndtr((low - index) / sd)),
                        np.where(right, np.log(special.ndtr((index - high) / sd)), exact))

    return outcome


def ml_oracle(frame, endog, kind, *, tobit, weights=None, weight_type=None, exog=XS,
              instruments=IV, limits=(0.0, 5.0)):
    x1, x_names, z, z_names, y2 = blocks(frame, endog, exog, instruments)
    n, p = y2.shape
    w, nobs = likelihood_weights(frame, weights, weight_type)
    y = frame["yt" if tobit else "yb"].to_numpy()
    outcome = tobit_outcome(y, *limits) if tobit else probit_outcome(y)
    obs, reported, ancillary, lower = joint_likelihood(x1, z, y2, outcome, tobit=tobit)
    key = (tobit, tuple(endog), tuple(exog), tuple(instruments), weights,
           "pweight" if weight_type == "iweight" else weight_type, limits)
    if key not in SOLVED:
        root = np.sqrt(w)[:, None]
        pi = np.linalg.lstsq(z * root, y2 * root, rcond=None)[0].T
        v = y2 - z @ pi.T
        var_v = (v * w[:, None]).T @ v / w.sum()
        regressors = np.column_stack([x1, y2])
        if tobit:
            delta = np.linalg.lstsq(regressors * root, y * root[:, 0], rcond=None)[0]
            var_u = [float(w @ (y - regressors @ delta) ** 2 / w.sum())]
        else:
            delta, var_u = np.zeros(regressors.shape[1]), []
        start = np.r_[delta, pi.reshape(-1), np.zeros(p), [var_v[i, j] for i, j in lower], var_u]
        theta, value = newton(obs, start, w)
        SOLVED[key] = theta, value, likelihood_pieces(obs, theta, w)
    theta, value, pieces = SOLVED[key]
    internal = stata_covariance(obs, theta, kind, w=w, weight_type=weight_type,
                                groups=frame.firm, pieces=pieces)
    jacobian = jacobian_cs(reported, theta)
    names = [*x_names, *endog, *(f"{name}:{term}" for name in endog for term in z_names),
             *ancillary]
    return {"names": names, "params": reported(theta), "theta": theta, "value": value,
            "covariance": jacobian @ internal @ jacobian.T, "internal": internal,
            "nobs": nobs, "kw": x1.shape[1] + p, "kz": z.shape[1], "p": p, "w": w,
            "size": len(theta)}


def check_ml(result, oracle, outcome, endog, *, rtol=2e-6):
    kw, kz, p = oracle["kw"], oracle["kz"], oracle["p"]
    params, covariance, value = oracle["params"], oracle["covariance"], oracle["value"]
    assert [c.term for c in result.coefficients] == oracle["names"]
    equations = [outcome] * kw + [name for name in endog for _ in range(kz)]
    assert [c.equation for c in result.coefficients] \
        == equations + [None] * (len(params) - len(equations))
    check_table(result, params, covariance, rtol=rtol)
    size, nobs = oracle["size"], oracle["nobs"]
    assert result.nobs == nobs
    assert_allclose(result.metrics["log_likelihood"], value, rtol=1e-10)
    assert_allclose(result.metrics["aic"], -2 * value + 2 * size, rtol=1e-10)
    assert_allclose(result.metrics["bic"], -2 * value + np.log(nobs) * size, rtol=1e-10)
    check_test(result.tests["model"], wald(params, covariance, range(1, kw)), kw - 1, rtol=1e-4)
    first = kw + p * kz
    exogeneity = [first + i * (i + 1) // 2 for i in range(p)]
    assert [oracle["names"][i] for i in exogeneity] == [f"/athrho{j + 2}_1" for j in range(p)]
    check_test(result.tests["exogeneity"], wald(params, covariance, exogeneity), p, rtol=1e-4)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("weighting", [None, "fweight", "aweight", "iweight", "pweight"])
def test_ivprobit_ml_one_endogenous_regressor(data, kind, weighting):
    call = {"data": data, "y": "yb", "x": XS, "endog": ["e1"], "instruments": IV}
    if weighting == "pweight" and kind in ("nonrobust", "opg"):
        with pytest.raises(AnalysisError) as caught:
            oe.ivprobit(**call, covariance=kind, weights="aw", weight_type="pweight")
        assert caught.value.code == "unsupported_covariance"
        return
    column = {None: None, "fweight": "fw"}.get(weighting, "aw")
    extra = {} if weighting is None else {"weights": column, "weight_type": weighting}
    result = oe.ivprobit(**call, **options_for(kind), **extra)
    oracle = ml_oracle(data, ["e1"], kind, tobit=False, weights=column, weight_type=weighting)
    check_ml(result, oracle, "yb", ["e1"])
    # Natural-scale records: corr(v, u) and sd(v) with their own standard errors.
    theta, internal = oracle["theta"], oracle["internal"]
    cov_vu, var_v = theta[-2], theta[-1]
    natural = jacobian_cs(lambda t: np.array([t[-2] / np.sqrt(t[-1]), np.sqrt(t[-1])]), theta)
    errors = np.sqrt(np.diag(natural @ internal @ natural.T))
    correlation = result.extra["correlations"]["/athrho2_1"]
    deviation = result.extra["standard_deviations"]["/lnsigma2"]
    assert_allclose([correlation["estimate"], deviation["estimate"]],
                    [cov_vu / np.sqrt(var_v), np.sqrt(var_v)], rtol=1e-6)
    assert_allclose([correlation["std_error"], deviation["std_error"]], errors, rtol=1e-4)


@pytest.mark.parametrize("kind", KINDS)
def test_ivprobit_ml_two_endogenous_regressors(data, kind):
    result = oe.ivprobit(data=data, y="yb", x=XS, endog=["e1", "e2"], instruments=IV,
                         **options_for(kind))
    oracle = ml_oracle(data, ["e1", "e2"], kind, tobit=False)
    assert oracle["names"][-5:] == ["/athrho2_1", "/athrho3_1", "/athrho3_2", "/lnsigma2",
                                    "/lnsigma3"]
    check_ml(result, oracle, "yb", ["e1", "e2"])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("weighting", [None, "fweight", "aweight", "pweight"])
def test_ivtobit_ml_one_endogenous_regressor(data, kind, weighting):
    if weighting == "pweight" and kind in ("nonrobust", "opg"):
        return
    column = {None: None, "fweight": "fw"}.get(weighting, "aw")
    extra = {} if weighting is None else {"weights": column, "weight_type": weighting}
    result = oe.ivtobit(data=data, y="yt", x=XS, endog=["e1"], instruments=IV, ll=0, ul=5,
                        **options_for(kind), **extra)
    oracle = ml_oracle(data, ["e1"], kind, tobit=True, weights=column, weight_type=weighting)
    assert oracle["names"][-3:] == ["/athrho2_1", "/lnsigma1", "/lnsigma2"]
    check_ml(result, oracle, "yt", ["e1"])
    y, w = data.yt.to_numpy(), oracle["w"]
    counts = w if weighting == "fweight" else np.ones(len(y))
    assert result.metrics["n_left_censored"] == int(counts[y <= 0].sum())
    assert result.metrics["n_right_censored"] == int(counts[y >= 5].sum())
    assert result.metrics["n_uncensored"] == oracle["nobs"] - int(counts[(y <= 0) | (y >= 5)].sum())
    assert_allclose(result.metrics["sigma"], np.sqrt(oracle["theta"][-1]), rtol=1e-6)
    assert result.inference["distribution"] == "normal"          # ivtobit reports z, not t


@pytest.mark.parametrize("kind", ["nonrobust", "cluster"])
def test_ivtobit_ml_two_endogenous_regressors_one_limit(data, kind):
    result = oe.ivtobit(data=data, y="yt", x=["x1"], endog=["e1", "e2"], instruments=IV, ll=0,
                        **options_for(kind))
    oracle = ml_oracle(data, ["e1", "e2"], kind, tobit=True, exog=["x1"], limits=(0.0, None))
    assert oracle["names"][-6:] == ["/athrho2_1", "/athrho3_1", "/athrho3_2", "/lnsigma1",
                                    "/lnsigma2", "/lnsigma3"]
    check_ml(result, oracle, "yt", ["e1", "e2"])
    assert result.metrics["n_right_censored"] == 0


@pytest.mark.parametrize("kind", KINDS)
def test_iv_frequency_weights_equal_replicated_rows(data, kind):
    replicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
    for function, extra in ((oe.ivprobit, {"y": "yb"}), (oe.ivtobit, {"y": "yt", "ll": 0})):
        call = {"x": XS, "endog": ["e1", "e2"], "instruments": IV, **extra, **options_for(kind)}
        weighted = function(data=data, weights="fw", weight_type="fweight", **call)
        expanded = function(data=replicated, **call)
        assert weighted.nobs == expanded.nobs == int(data.fw.sum())
        assert_allclose(estimates(weighted), estimates(expanded), rtol=1e-7, atol=1e-9)
        assert_allclose(covariance_of(weighted), covariance_of(expanded), rtol=5e-6, atol=1e-11)
        for name in ("model", "exogeneity"):
            assert_allclose(weighted.tests[name]["statistic"], expanded.tests[name]["statistic"],
                            rtol=1e-5)
        for name in ("e1", "e2"):
            assert weighted.extra["first_stage"][name] == pytest.approx(
                expanded.extra["first_stage"][name], rel=1e-8)


def test_first_stage_summary_matches_statsmodels(data):
    result = oe.ivprobit(data=data, y="yb", x=XS, endog=["e1", "e2"], instruments=IV)
    x1, _, z, z_names, y2 = blocks(data, ["e1", "e2"])
    assert result.extra["exogenous"] == ["Intercept", *XS]
    assert result.extra["instruments"] == IV and result.extra["endogenous"] == ["e1", "e2"]
    assert result.extra["n_instruments"] == 3 and result.extra["n_endogenous"] == 2
    for j, name in enumerate(["e1", "e2"]):
        fit = sm.OLS(y2[:, j], z).fit()
        restricted = sm.OLS(y2[:, j], x1).fit()
        test = fit.f_test(np.eye(z.shape[1])[x1.shape[1]:])
        record = result.extra["first_stage"][name]
        assert_allclose(record["r_squared"], fit.rsquared, rtol=1e-9)
        assert_allclose(record["adjusted_r_squared"], fit.rsquared_adj, rtol=1e-9)
        assert_allclose(record["partial_r_squared"], 1 - fit.ssr / restricted.ssr, rtol=1e-9)
        assert_allclose(record["f_statistic"], float(np.squeeze(test.fvalue)), rtol=1e-8)
        assert_allclose(record["f_p_value"], float(test.pvalue), rtol=1e-6, atol=1e-300)
        assert (record["f_df1"], record["f_df2"]) == (3, len(data) - z.shape[1])
        assert_allclose(record["rmse"], np.sqrt(fit.mse_resid), rtol=1e-9)
        # The reduced-form block of the ML fit is the first-stage least squares when the
        # model is exactly identified in that equation's coefficients only asymptotically;
        # here it must simply be close.
        position = [c.term for c in result.coefficients].index(f"{name}:Intercept")
        assert_allclose(estimates(result)[position:position + z.shape[1]], fit.params, atol=0.1)


# ---- Newey's two-step estimator ------------------------------------------------------


def probit_fit(y, x, w=None):
    if w is None:
        fit = sm.Probit(y, x).fit(disp=0, tol=1e-12, maxiter=200, method="newton")
        return np.asarray(fit.params), np.asarray(fit.cov_params())
    family = sm.families.Binomial(link=sm.families.links.Probit())
    fit = sm.GLM(y, x, family=family, freq_weights=w).fit(tol=1e-13, maxiter=200)
    index, sign = x @ fit.params, 2 * y - 1
    ratio = stats.norm.pdf(index) / special.ndtr(sign * index)
    curvature = ratio * (ratio + sign * index)
    return np.asarray(fit.params), np.linalg.inv((x * (w * curvature)[:, None]).T @ x)


def tobit_fit(y, x, ll, ul, w=None):
    w = np.ones(len(y)) if w is None else w
    obs, reported = tobit_olsen(x, y, ll, ul)
    root = np.sqrt(w)
    beta = np.linalg.lstsq(x * root[:, None], y * root, rcond=None)[0]
    sigma = np.sqrt(w @ (y - x @ beta) ** 2 / w.sum())
    theta, _ = newton(obs, np.r_[beta / sigma, 1 / sigma], w)
    jacobian = jacobian_cs(reported, theta)
    covariance = jacobian @ stata_covariance(obs, theta, "nonrobust", w=w) @ jacobian.T
    return reported(theta), covariance


def newey_oracle(frame, endog, model, *, fweights=None, exog=XS, instruments=IV):
    """Stata's Methods and formulas for ``ivprobit, twostep`` / ``ivtobit, twostep``."""
    x1, x_names, z, z_names, y2 = blocks(frame, endog, exog, instruments)
    n, p = y2.shape
    k1, kz = x1.shape[1], z.shape[1]
    w = None if fweights is None else frame[fweights].to_numpy(dtype=float)
    weights = np.ones(n) if w is None else w
    nobs = weights.sum()
    zw = z * weights[:, None]
    cross = np.linalg.inv(zw.T @ z)
    pi = cross @ zw.T @ y2                                          # [kz, p]
    v = y2 - z @ pi
    reduced, reduced_cov = model(np.column_stack([z, v]), w)
    alpha, lam = reduced[:kz], reduced[kz:kz + p]
    conditional, conditional_cov = model(np.column_stack([x1, y2, v]), w)
    beta = conditional[k1:k1 + p]
    target = y2 @ (lam - beta)
    coefficients = cross @ zw.T @ target
    s2 = weights @ (target - z @ coefficients) ** 2 / (nobs - kz)
    omega = reduced_cov[:kz, :kz] + s2 * cross
    d = np.column_stack([np.eye(kz)[:, :k1], pi])
    inverse = np.linalg.inv(omega)
    covariance = np.linalg.inv(d.T @ inverse @ d)
    delta = covariance @ d.T @ inverse @ alpha
    span = range(k1 + p, k1 + 2 * p)
    exogeneity = wald(conditional, conditional_cov, span)
    return {"names": [*x_names, *endog], "params": delta, "covariance": covariance,
            "exogeneity": exogeneity, "conditional": conditional, "nobs": int(nobs),
            "kw": k1 + p, "p": p}


@pytest.mark.parametrize("endog", [["e1"], ["e1", "e2"]])
@pytest.mark.parametrize("fweights", [None, "fw"])
def test_ivprobit_two_step_matches_newey(data, endog, fweights):
    extra = {} if fweights is None else {"weights": "fw", "weight_type": "fweight"}
    result = oe.ivprobit(data=data, y="yb", x=XS, endog=endog, instruments=IV,
                         method="twostep", **extra)
    y = data.yb.to_numpy()
    oracle = newey_oracle(data, endog, lambda design, w: probit_fit(y, design, w),
                          fweights=fweights)
    assert [c.term for c in result.coefficients] == oracle["names"]
    check_table(result, oracle["params"], oracle["covariance"], rtol=2e-6)
    assert result.nobs == oracle["nobs"]
    check_test(result.tests["model"],
               wald(oracle["params"], oracle["covariance"], range(1, oracle["kw"])),
               oracle["kw"] - 1, rtol=1e-5)
    check_test(result.tests["exogeneity"], oracle["exogeneity"], oracle["p"], rtol=1e-5)
    assert "log_likelihood" not in result.metrics
    if fweights:
        replicated = data.loc[data.index.repeat(data.fw.astype(int))].reset_index(drop=True)
        expanded = oe.ivprobit(data=replicated, y="yb", x=XS, endog=endog, instruments=IV,
                               method="twostep")
        assert_allclose(estimates(result), estimates(expanded), rtol=1e-8)
        assert_allclose(covariance_of(result), covariance_of(expanded), rtol=1e-7)


@pytest.mark.parametrize("endog", [["e1"], ["e1", "e2"]])
def test_ivtobit_two_step_matches_newey(data, endog):
    result = oe.ivtobit(data=data, y="yt", x=XS, endog=endog, instruments=IV, ll=0, ul=5,
                        method="twostep")
    y = data.yt.to_numpy()
    oracle = newey_oracle(data, endog, lambda design, w: tobit_fit(y, design, 0.0, 5.0, w))
    assert [c.term for c in result.coefficients] == oracle["names"]
    check_table(result, oracle["params"], oracle["covariance"], rtol=2e-6)
    check_test(result.tests["exogeneity"], oracle["exogeneity"], oracle["p"], rtol=1e-5)
    assert result.inference["distribution"] == "normal"


def test_just_identified_two_step_is_the_control_function_estimator(data):
    """With as many instruments as endogenous regressors D is square, so Newey's estimator
    reproduces the Rivers-Vuong (2SIV) coefficients of [x1, y2] exactly."""
    y = data.yb.to_numpy()
    result = oe.ivprobit(data=data, y="yb", x=XS, endog=["e1"], instruments=["z1"],
                         method="twostep")
    oracle = newey_oracle(data, ["e1"], lambda design, w: probit_fit(y, design, w),
                          instruments=["z1"])
    assert_allclose(estimates(result), oracle["conditional"][:oracle["kw"]], rtol=1e-7)
    # ... and the ML estimates are the same coefficients rescaled by sd(u | v).
    ml = oe.ivprobit(data=data, y="yb", x=XS, endog=["e1"], instruments=["z1"])
    rho = np.tanh(estimates(ml)[-2])
    assert_allclose(estimates(ml)[:oracle["kw"]] / np.sqrt(1 - rho ** 2), estimates(result),
                    rtol=1e-6)


def test_two_step_rejects_what_stata_rejects(data):
    for function, extra in ((oe.ivprobit, {"y": "yb"}), (oe.ivtobit, {"y": "yt", "ll": 0})):
        call = {"data": data, "x": XS, "endog": ["e1"], "instruments": IV, "method": "twostep",
                **extra}
        for options, code in (
                ({"covariance": "robust"}, "unsupported_covariance"),
                ({"cluster": "firm"}, "unsupported_covariance"),
                ({"weights": "aw", "weight_type": "pweight"}, "unsupported_weights"),
                ({"weights": "aw", "weight_type": "iweight"}, "unsupported_weights")):
            with pytest.raises(AnalysisError) as caught:
                function(**call, **options)
            assert caught.value.code == code, options


def test_exogenous_regressor_reproduces_probit_and_tobit(data):
    """Instruments equal to an exogenous 'endogenous' regressor's own determinants with an
    independent error: rho is free, but the structural coefficients must agree with the
    two-step ones up to the conditional scale, and the model without endogeneity
    (no x, one endogenous regressor) is well defined."""
    result = oe.ivprobit(data=data, y="yb", endog=["e1"], instruments=["z1", "z2"])
    oracle = ml_oracle(data, ["e1"], "nonrobust", tobit=False, exog=[],
                       instruments=["z1", "z2"])
    check_ml(result, oracle, "yb", ["e1"])
    assert result.extra["exogenous"] == ["Intercept"]


def test_iv_missing_values_collinear_and_categorical_instruments_and_units(data):
    holes = data.copy()
    holes.loc[[4, 50], "z2"] = np.nan
    holes.loc[[9], "e1"] = np.nan
    holes.loc[[50, 61], "yb"] = np.nan
    holes["band"] = np.where(holes.z3 > 1.0, "high", np.where(holes.z3 > 0.4, "mid", "low"))
    holes["mid"] = (holes.band == "mid") * 1.0
    holes["low"] = (holes.band == "low") * 1.0                # sorted levels: high, low, mid
    holes["twin"] = holes.z1 - 2 * holes.x1 + 3               # collinear with x1, z1, constant
    lost = [4, 9, 50, 61]
    clean = holes.drop(index=lost).reset_index(drop=True)
    for method in ("ml", "twostep"):
        with pytest.raises(AnalysisError) as caught:
            oe.ivprobit(data=holes, y="yb", x=XS, endog=["e1"], instruments=["z1", "z2"],
                        method=method)
        assert caught.value.code == "missing_values"
        result = oe.ivprobit(data=holes, y="yb", x=XS, endog=["e1"],
                             instruments=["z1", "twin", "z2", "band"], categorical=["band"],
                             missing="drop", method=method)
        expected = oe.ivprobit(data=clean, y="yb", x=XS, endog=["e1"],
                               instruments=["z1", "z2", "low", "mid"], method=method)
        assert_allclose(estimates(result), estimates(expected), rtol=1e-8, atol=1e-10)
        assert_allclose(covariance_of(result), covariance_of(expected), rtol=1e-7, atol=1e-12)
        assert result.provenance["omitted_terms"] == ["twin"]
        assert result.extra["instruments"] == ["z1", "z2", "band[low]", "band[mid]"]
        assert result.extra["n_instruments"] == 4
        assert result.nobs == len(clean)
        assert result.sample_positions == [i for i in range(len(holes)) if i not in lost]
        # Units: e1 -> 1e3 e1 divides its coefficient; its reduced form and sd scale by 1e3.
        scaled = oe.ivprobit(data=clean.assign(e1=clean.e1 * 1e3), y="yb", x=XS, endog=["e1"],
                             instruments=["z1", "z2", "low", "mid"], method=method)
        terms = [c.term for c in expected.coefficients]
        factor = np.array([1e-3 if t == "e1" else 1e3 if t.startswith("e1:") else 1.0
                           for t in terms])
        shift = np.array([np.log(1e3) if t == "/lnsigma2" else 0.0 for t in terms])
        assert_allclose(estimates(scaled), estimates(expected) * factor + shift, rtol=1e-6,
                        atol=1e-9)
        assert_allclose(covariance_of(scaled), covariance_of(expected) * np.outer(factor, factor),
                        rtol=1e-5, atol=1e-14)
        assert_allclose(scaled.tests["exogeneity"]["statistic"],
                        expected.tests["exogeneity"]["statistic"], rtol=1e-6)


def test_iv_row_order_does_not_matter(data):
    shuffled = data.sample(frac=1.0, random_state=3).reset_index(drop=True)
    for function, extra in ((oe.ivprobit, {"y": "yb"}), (oe.ivtobit, {"y": "yt", "ll": 0})):
        for options in ({"cluster": "firm"}, {"method": "twostep"}):
            call = {"x": XS, "endog": ["e1", "e2"], "instruments": IV, **extra, **options}
            left, right = function(data=data, **call), function(data=shuffled, **call)
            assert_allclose(estimates(left), estimates(right), rtol=1e-7, atol=1e-10)
            assert_allclose(covariance_of(left), covariance_of(right), rtol=1e-6, atol=1e-12)
