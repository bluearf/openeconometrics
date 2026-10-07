"""Independent oracles for teffects ra / ipw / ipwra / aipw.

The oracle is explicit NumPy M-estimation: the nuisance models are fitted by
statsmodels (OLS/WLS, Logit, Probit, Poisson GLM, MNLogit), the potential-
outcome means by their closed forms, and the covariance is the sandwich
A^-1 B A^-T of the stacked estimating equations written independently in
NumPy, with A obtained by NUMERICAL differentiation of the summed estimating
functions. OpenEcon assembles A analytically, so agreement checks the
analytic Jacobian, the scores and the bookkeeping of weights and clusters.
"""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy import stats
from statsmodels.tools.numdiff import approx_fprime

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.teffects import index
from openecon.engines.optimize import check_derivatives, numerical_gradient
from openecon.models import ResultBundle


def make_data(seed=3, n=600, levels=2):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n) + 2.0
    if levels == 2:
        p = 1 / (1 + np.exp(-(-0.2 + 0.6 * x1 - 0.3 * (x2 - 2))))
        d = (rng.uniform(size=n) < p).astype(int)
    else:
        eta = np.column_stack([np.zeros(n), 0.2 + 0.5 * x1, -0.3 + 0.4 * (x2 - 2)])
        prob = np.exp(eta) / np.exp(eta).sum(1, keepdims=True)
        d = (rng.uniform(size=n)[:, None] > prob.cumsum(1)).sum(1)
    y = 1 + 1.5 * d + x1 + 0.5 * x2 + 0.4 * d * x1 + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    frame["yb"] = (y > np.median(y)).astype(float)
    frame["yc"] = rng.poisson(np.exp(0.1 + 0.3 * x1 + 0.3 * (d > 0)))
    frame["f"] = rng.integers(1, 4, size=n).astype(float)
    frame["pw"] = rng.uniform(0.5, 2.0, size=n)
    frame["cl"] = rng.integers(0, 40, size=n)
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


# ---- independent oracle -----------------------------------------------------------


def mean_fn(model, eta):
    return {"linear": lambda e: e, "logit": lambda e: 1 / (1 + np.exp(-e)),
            "probit": stats.norm.cdf, "poisson": np.exp}[model](eta)


def ml_resid(model, y, eta):
    if model == "probit":
        return y * stats.norm.pdf(eta) / stats.norm.cdf(eta) \
            - (1 - y) * stats.norm.pdf(eta) / stats.norm.sf(eta)
    return y - mean_fn(model, eta)


def sm_fit(model, y, x, w):
    if model == "linear":
        return sm.WLS(y, x, weights=w).fit().params
    if model == "logit":
        return sm.GLM(y, x, family=sm.families.Binomial(), var_weights=w).fit(tol=1e-12).params
    if model == "probit":
        family = sm.families.Binomial(link=sm.families.links.Probit())
        return sm.GLM(y, x, family=family, var_weights=w).fit(tol=1e-12).params
    return sm.GLM(y, x, family=sm.families.Poisson(), var_weights=w).fit(tol=1e-12).params


def treatment_probs(tmodel, xt, gamma, levels):
    if levels == 2:
        p = mean_fn(tmodel, xt @ gamma[0])
        return np.column_stack([1 - p, p])
    eta = np.column_stack([np.zeros(len(xt)), xt @ gamma.T])
    return np.exp(eta) / np.exp(eta).sum(1, keepdims=True)


def oracle(frame, method, estimand="ate", omodel="linear", tmodel="logit", x=("x1", "x2"),
           tx=("x1", "x2"), y="y", wcol=None, wtype=None, cluster=None, target=1):
    d = frame["d"].to_numpy()
    levels = len(np.unique(d))
    yv = frame[y].to_numpy(float)
    X = sm.add_constant(frame[list(x)].to_numpy(float), has_constant="add")
    Xt = sm.add_constant(frame[list(tx)].to_numpy(float), has_constant="add")
    n = len(frame)
    w = np.ones(n) if wcol is None else frame[wcol].to_numpy(float)
    use_t = method in {"ipw", "ipwra", "aipw"}
    use_o = method in {"ra", "ipwra", "aipw"}
    kt, ko, S = Xt.shape[1], X.shape[1], levels - 1
    # nuisance estimates
    if use_t:
        if levels == 2:
            gamma = sm_fit(tmodel, (d == 1).astype(float), Xt, w)[None, :]
        else:
            gamma = sm.MNLogit(d, Xt).fit(disp=0, method="newton", tol=1e-12, maxiter=200) \
                if wcol is None else None
            gamma = np.asarray(gamma.params).T
    else:
        gamma = np.zeros((S, kt))

    def omega_of(p):
        if estimand == "atet":
            return p[:, [target]] / p
        return 1 / p

    p = treatment_probs(tmodel, Xt, gamma, levels) if use_t else np.ones((n, levels))
    om = omega_of(p) if use_t else np.ones((n, levels))
    betas = []
    if use_o:
        for level in range(levels):
            m = d == level
            ow = w * om[:, level] if method == "ipwra" else w
            betas.append(sm_fit(omodel, yv[m], X[m], ow[m]))
    sel = (d == target).astype(float) if estimand == "atet" else np.ones(n)
    tau = np.zeros(levels)
    for level in range(levels):
        ind = (d == level).astype(float)
        if method in {"ra", "ipwra"}:
            tau[level] = np.sum(w * sel * mean_fn(omodel, X @ betas[level])) / np.sum(w * sel)
        elif method == "ipw":
            tau[level] = np.sum(w * ind * om[:, level] * yv) / np.sum(w * ind * om[:, level])
        else:
            mu = mean_fn(omodel, X @ betas[level])
            tau[level] = np.sum(w * (ind * om[:, level] * (yv - mu) + sel * mu)) / np.sum(w * sel)
    theta = np.concatenate([gamma.ravel() if use_t else [], *betas, tau])

    def psi(th):
        off = 0
        cols = []
        if use_t:
            g = th[:S * kt].reshape(S, kt)
            off = S * kt
            if levels == 2:
                cols.append(Xt * ml_resid(tmodel, (d == 1).astype(float), Xt @ g[0])[:, None])
            else:
                pr = treatment_probs(tmodel, Xt, g, levels)
                for s in range(1, levels):
                    cols.append(Xt * ((d == s) - pr[:, s])[:, None])
            pp = treatment_probs(tmodel, Xt, g, levels)
            omg = omega_of(pp)
        else:
            omg = np.ones((n, levels))
        bs = []
        if use_o:
            for level in range(levels):
                b = th[off:off + ko]
                off += ko
                bs.append(b)
                ow = omg[:, level] if method == "ipwra" else 1.0
                cols.append(X * ((d == level) * ow * ml_resid(omodel, yv, X @ b))[:, None])
        for level in range(levels):
            t = th[off + level]
            ind = (d == level).astype(float)
            if method in {"ra", "ipwra"}:
                cols.append((sel * (mean_fn(omodel, X @ bs[level]) - t))[:, None])
            elif method == "ipw":
                cols.append((ind * omg[:, level] * (yv - t))[:, None])
            else:
                mu = mean_fn(omodel, X @ bs[level])
                cols.append((ind * omg[:, level] * (yv - mu) + sel * (mu - t))[:, None])
        return np.column_stack(cols)

    rows = psi(theta)
    assert np.abs((w[:, None] * rows).sum(0)).max() < 1e-6 * n
    jac = approx_fprime(theta, lambda th: (w[:, None] * psi(th)).sum(0), centered=True)
    if cluster is not None:
        codes = pd.factorize(frame[cluster])[0]
        totals = np.zeros((codes.max() + 1, len(theta)))
        np.add.at(totals, codes, w[:, None] * rows)
        meat = totals.T @ totals
    elif wtype == "fweight":
        meat = (rows * w[:, None]).T @ rows
    else:
        meat = (rows * w[:, None]).T @ (rows * w[:, None])
    inv = np.linalg.inv(jac)
    cov = inv @ meat @ inv.T
    pom = cov[-levels:, -levels:]
    if estimand == "pomeans":
        transform = np.eye(levels)
    else:
        transform = np.zeros((levels, levels))
        for level in range(1, levels):
            transform[level - 1, level], transform[level - 1, 0] = 1, -1
        transform[-1, 0] = 1
    est = transform @ tau
    se = np.sqrt(np.diag(transform @ pom @ transform.T))
    return est, se, theta, cov


def estimates(result):
    return (np.array([c.estimate for c in result.coefficients]),
            np.array([c.std_error for c in result.coefficients]))


CASES = [
    ("ra", "ate", "linear", "logit"), ("ra", "atet", "linear", "logit"),
    ("ra", "pomeans", "logit", "logit"), ("ra", "ate", "probit", "logit"),
    ("ra", "ate", "poisson", "logit"),
    ("ipw", "ate", "linear", "logit"), ("ipw", "atet", "linear", "probit"),
    ("ipw", "pomeans", "linear", "probit"),
    ("ipwra", "ate", "linear", "logit"), ("ipwra", "atet", "linear", "logit"),
    ("ipwra", "ate", "logit", "probit"), ("ipwra", "atet", "poisson", "probit"),
    ("aipw", "ate", "linear", "logit"), ("aipw", "pomeans", "probit", "probit"),
    ("aipw", "ate", "poisson", "logit"), ("aipw", "ate", "logit", "logit"),
]


@pytest.mark.parametrize("method,estimand,omodel,tmodel", CASES)
def test_stacked_estimators_match_numpy_m_estimation(data, method, estimand, omodel, tmodel):
    y = {"linear": "y", "logit": "yb", "probit": "yb", "poisson": "yc"}[omodel]
    kwargs = {}
    if method in {"ra", "ipwra", "aipw"}:
        kwargs["omodel"] = omodel
    if method in {"ipw", "ipwra", "aipw"}:
        kwargs["tmodel"] = tmodel
    result = oe.teffects(data=data, y=y, treatment="d", x=["x1", "x2"], method=method,
                         estimand=estimand, **kwargs)
    est, se, theta, cov = oracle(data, method, estimand, omodel, tmodel, y=y)
    got, got_se = estimates(result)
    assert_allclose(got, est, rtol=1e-7, atol=1e-9)
    assert_allclose(got_se, se, rtol=2e-5)
    assert result.inference["use_t"] is False
    assert result.provenance["stata_parity_validated"] is False
    if method != "ipw":
        # auxiliary outcome equation of the control level, reported uncentred
        ome0 = [row["estimate"] for row in result.extra["auxiliary_equations"]["OME0"]]
        offset = 3 if method in {"ipwra", "aipw"} else 0
        assert_allclose(ome0, theta[offset:offset + 3], rtol=1e-6, atol=1e-8)


def test_ra_linear_ate_equals_difference_of_predicted_means(data):
    result = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"])
    X = sm.add_constant(data[["x1", "x2"]].to_numpy())
    preds = []
    for level in (0, 1):
        m = data.d == level
        preds.append(X @ sm.OLS(data.y[m], X[m]).fit().params)
    assert_allclose(result.coefficients[0].estimate, np.mean(preds[1] - preds[0]), rtol=1e-10)
    assert result.coefficients[0].term == "ATE:r1vs0.d"
    assert result.coefficients[1].term == "POmean:0.d"
    assert [c.equation for c in result.coefficients] == ["ATE", "POmean"]


def test_outcome_model_without_constant_is_not_centred(data):
    result = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"], intercept=False,
                         estimand="pomeans")
    X = data[["x1", "x2"]].to_numpy()
    means = [np.mean(X @ sm.OLS(data.y[data.d == level], X[data.d == level]).fit().params)
             for level in (0, 1)]
    assert_allclose(estimates(result)[0], means, rtol=1e-10)
    terms = [row["term"] for row in result.extra["auxiliary_equations"]["OME0"]]
    assert terms == ["x1", "x2"]


def test_frequency_weights_equal_duplicated_rows(data):
    for method in ("ra", "ipw", "ipwra", "aipw"):
        weighted = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"], method=method,
                               weights="f", weight_type="fweight")
        expanded = data.loc[data.index.repeat(data.f.astype(int))].reset_index(drop=True)
        plain = oe.teffects(data=expanded, y="y", treatment="d", x=["x1", "x2"], method=method)
        assert_allclose(estimates(weighted)[0], estimates(plain)[0], rtol=1e-9)
        assert_allclose(estimates(weighted)[1], estimates(plain)[1], rtol=1e-8)
        assert weighted.nobs == int(data.f.sum())


@pytest.mark.parametrize("method", ["ra", "ipw", "ipwra", "aipw"])
def test_sampling_weights_and_cluster_covariance_match_oracle(data, method):
    weighted = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"], method=method,
                           weights="pw", weight_type="pweight")
    est, se, *_ = oracle(data, method, wcol="pw", wtype="pweight")
    assert_allclose(estimates(weighted)[0], est, rtol=1e-7)
    assert_allclose(estimates(weighted)[1], se, rtol=2e-5)
    clustered = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"], method=method,
                            covariance="cluster", cluster="cl")
    est, se, *_ = oracle(data, method, cluster="cl")
    assert_allclose(estimates(clustered)[0], est, rtol=1e-7)
    assert_allclose(estimates(clustered)[1], se, rtol=2e-5)
    assert clustered.inference["cluster_count"] == data.cl.nunique()
    # rescaling sampling weights changes nothing
    scaled = data.assign(pw=data.pw * 1e6)
    again = oe.teffects(data=scaled, y="y", treatment="d", x=["x1", "x2"], method=method,
                        weights="pw", weight_type="pweight")
    assert_allclose(estimates(again)[1], estimates(weighted)[1], rtol=1e-8)


@pytest.mark.parametrize("method", ["ra", "ipw", "ipwra", "aipw"])
def test_multivalued_treatment_matches_oracle(method):
    frame = make_data(seed=11, n=900, levels=3)
    result = oe.teffects(data=frame, y="y", treatment="d", x=["x1", "x2"], method=method)
    est, se, *_ = oracle(frame, method)
    assert_allclose(estimates(result)[0], est, rtol=1e-6)
    assert_allclose(estimates(result)[1], se, rtol=5e-5)
    assert [c.term for c in result.coefficients] == ["ATE:r1vs0.d", "ATE:r2vs0.d", "POmean:0.d"]
    pomeans = oe.teffects(data=frame, y="y", treatment="d", x=["x1", "x2"], method=method,
                          estimand="pomeans", control=2)
    assert [c.term for c in pomeans.coefficients] == ["POmeans:2.d", "POmeans:0.d",
                                                      "POmeans:1.d"]
    assert_allclose(sorted(estimates(pomeans)[0]),
                    sorted(result.extra["potential_outcome_means"].values()), rtol=1e-10)
    if method != "ra":
        with pytest.raises(AnalysisError) as caught:
            oe.teffects(data=frame, y="y", treatment="d", x=["x1", "x2"], method=method,
                        tmodel="probit")
        assert caught.value.code == "unsupported_model"
    atet = oe.teffects(data=frame, y="y", treatment="d", x=["x1", "x2"], method=method,
                      estimand="atet")
    atet_est, atet_se, *_ = oracle(frame, method, estimand="atet")
    assert_allclose(estimates(atet)[0], atet_est, rtol=1e-6)
    assert_allclose(estimates(atet)[1], atet_se, rtol=5e-5)


def test_index_model_derivatives_are_analytic(data):
    x = torch.tensor(sm.add_constant(data[["x1", "x2"]].to_numpy()), dtype=torch.float64)
    w = torch.tensor(data.pw.to_numpy())
    theta = torch.tensor([0.1, 0.3, -0.2], dtype=torch.float64)
    outcomes = {"linear": data.y, "logit": data.yb, "probit": data.yb, "poisson": data.yc}
    for model, column in outcomes.items():
        y = torch.tensor(column.to_numpy(float))
        report = check_derivatives(index.IndexObjective(model, x, y, w), theta)
        assert report["gradient_max_rel_error"] < 1e-7, model
        assert report["hessian_max_rel_error"] < 1e-7, model
    frac = torch.tensor(np.clip(data.yb.to_numpy() * 0.7 + 0.1, 0, 1))
    report = check_derivatives(index.IndexObjective("probit", x, frac, w), theta)
    assert report["hessian_max_rel_error"] < 1e-7
    # treatment pieces: slopes = d p / d eta, curvature = d residual / d eta
    codes = torch.tensor(data.d.to_numpy(), dtype=torch.int64)
    for model in ("logit", "probit"):
        fit = index.treatment_pieces(model, x, codes, 2, theta[None, :])
        for i in (0, 5, 17):
            eta = float(x[i] @ theta)
            p = lambda e, m=model: index.mean_pieces(m, torch.tensor([e[0]]))[0][0]  # noqa: E731
            assert_allclose(float(fit.slopes[i, 1, 0]),
                            float(numerical_gradient(p, torch.tensor([eta]))[0]), rtol=1e-8)
            r = lambda e, m=model, i=i: index.residual_pieces(  # noqa: E731
                m, codes[i:i + 1].double(), torch.tensor([e[0]]))[0][0]
            assert_allclose(float(fit.curvature[i, 0, 0]),
                            float(numerical_gradient(r, torch.tensor([eta]))[0]), rtol=1e-7)
    three = torch.tensor(make_data(seed=11, n=300, levels=3).d.to_numpy(), dtype=torch.int64)
    xs = x[:300]
    objective = index.treatment_objective("logit", xs, three, 3, torch.ones(300,
                                                                            dtype=torch.float64))
    report = check_derivatives(objective, torch.tensor([0.1, 0.2, -0.1, 0.3, 0.0, 0.2],
                                                       dtype=torch.float64))
    assert report["hessian_max_rel_error"] < 1e-7
    gamma = torch.tensor([[0.1, 0.2, -0.1], [0.3, 0.0, 0.2]], dtype=torch.float64)
    fit = index.treatment_pieces("logit", xs, three, 3, gamma)
    for level in range(3):
        for s in range(2):
            def prob(e, level=level, s=s):
                g = gamma.clone()
                g[s, 0] += e[0]
                return index.treatment_pieces("logit", xs[:1], three[:1], 3, g).probabilities[
                    0, level]
            numeric = numerical_gradient(prob, torch.zeros(1, dtype=torch.float64))[0]
            assert_allclose(float(fit.slopes[0, level, s]), float(numeric), rtol=1e-7, atol=1e-12)


def test_collinear_covariates_are_omitted_per_equation(data):
    frame = data.assign(x3=2 * data.x1 - data.x2)
    full = oe.teffects(data=frame, y="y", treatment="d", x=["x1", "x2", "x3"], method="aipw")
    base = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"], method="aipw")
    assert_allclose(estimates(full)[0], estimates(base)[0], rtol=1e-9)
    assert_allclose(estimates(full)[1], estimates(base)[1], rtol=1e-8)
    omitted = full.provenance["omitted_terms"]
    assert {"TME:x3", "OME0:x3", "OME1:x3"} <= set(omitted)
    assert any("Omitted because of collinearity" in w for w in full.warnings)
    # a covariate constant within the treated group is dropped from that equation only
    noise = np.random.default_rng(0).normal(size=len(data))
    frame = data.assign(z=np.where(data.d == 1, 1.0, noise))
    result = oe.teffects(data=frame, y="y", treatment="d", x=["x1", "z"])
    assert "OME1:z" in result.provenance["omitted_terms"]
    assert "OME0:z" not in result.provenance["omitted_terms"]


def test_failure_contract_and_missing_policy(data):
    cases = [
        ({"method": "ra", "tmodel": "probit"}, "invalid_spec"),
        ({"method": "ipw", "omodel": "logit"}, "invalid_spec"),
        ({"method": "ra", "neighbors": 2}, "invalid_spec"),
        ({"method": "ra", "tx": ["x1"]}, "invalid_spec"),
        ({"method": "ra", "omodel": "logit"}, "invalid_outcome"),
        ({"method": "ra", "omodel": "poisson", "y": "x1", "x": ["x2"]}, "invalid_outcome"),
        ({"method": "ra", "x": ["d"]}, "invalid_spec"),
        ({"method": "ra", "covariance": "nonrobust"}, "invalid_spec"),
        ({"method": "ra", "control": 5}, "invalid_treatment"),
        ({"method": "ra", "x": "x1"}, "invalid_spec"),
        ({"method": "ra", "weights": "f", "weight_type": "aweight"}, "invalid_spec"),
    ]
    for update, code in cases:
        arguments = {"data": data, "y": "y", "treatment": "d", "x": ["x1", "x2"], **update}
        with pytest.raises(AnalysisError) as caught:
            oe.teffects(**arguments)
        assert caught.value.code == code, update
    with pytest.raises(AnalysisError) as caught:
        oe.teffects(data=data[data.d == 1], y="y", treatment="d", x=["x1"])
    assert caught.value.code == "invalid_treatment"
    # perfect separation of treatment by a covariate violates overlap
    separated = data.assign(s=data.d + 0.01 * np.random.default_rng(1).normal(size=len(data)))
    with pytest.raises(AnalysisError) as caught:
        oe.teffects(data=separated, y="y", treatment="d", x=["s"], method="ipw")
    assert caught.value.code == "overlap_violation"
    # near-violations are caught by pstolerance
    strong = data.assign(s=3.0 * data.d + np.random.default_rng(2).normal(size=len(data)))
    with pytest.raises(AnalysisError) as caught:
        oe.teffects(data=strong, y="y", treatment="d", x=["s"], method="aipw", pstolerance=0.01)
    assert caught.value.code == "overlap_violation"
    missing = data.copy()
    missing.loc[[3, 8], "x1"] = np.nan
    with pytest.raises(AnalysisError) as caught:
        oe.teffects(data=missing, y="y", treatment="d", x=["x1", "x2"])
    assert caught.value.code == "missing_values"
    dropped = oe.teffects(data=missing, y="y", treatment="d", x=["x1", "x2"], missing="drop")
    assert dropped.nobs == len(data) - 2 and dropped.dropped_rows == 2
    reference = oe.teffects(data=missing.dropna(), y="y", treatment="d", x=["x1", "x2"])
    assert_allclose(estimates(dropped)[0], estimates(reference)[0], rtol=1e-12)


def test_string_treatment_labels_and_categorical_covariates(data):
    frame = data.assign(arm=np.where(data.d == 1, "drug", "placebo"),
                        grp=pd.Categorical(np.where(data.x1 > 0, "hi", "lo")))
    result = oe.teffects(data=frame, y="y", treatment="arm", x=["x2", "grp"],
                         categorical=["grp"], method="ipwra", control="placebo")
    assert result.coefficients[0].term == "ATE:rdrugvsplacebo.arm"
    numeric = oe.teffects(data=frame.assign(g=(frame.grp == "lo").astype(float)), y="y",
                          treatment="d", x=["x2", "g"], method="ipwra")
    assert_allclose(estimates(result)[0], estimates(numeric)[0], rtol=1e-9)


def test_serialization_rendering_and_exports(data):
    result = oe.teffects(data=data, y="y", treatment="d", x=["x1", "x2"], method="aipw")
    restored = ResultBundle.model_validate_json(result.model_dump_json())
    assert restored == result
    json.loads(result.model_dump_json())
    text = result.summary()
    assert "Treatment-effects estimation: augmented IPW — y" in text and "[POmean]" in text
    latex = str(result.to_latex())
    assert "begin{tabular}" in latex and "ATE" in latex
    assert callable(oe.teffects) and "teffects" in dir(oe)
    capability = oe.capabilities()["estimators"]["teffects"]
    assert capability["default_covariance"] == "robust"
    assert capability["options"]["method"]["default"] == "ra"
    assert result.extra["overlap"]["propensity"]["P(d=1)"]["1"]["n"] == int(data.d.sum())
