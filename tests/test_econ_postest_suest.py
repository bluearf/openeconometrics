"""Independent oracles for oe.suest (post-estimation family).

The joint covariance is rebuilt in NumPy from explicitly coded score
contributions (statsmodels' score_obs/hessian for logit and Poisson, the OLS
normal-likelihood formulas written out, and central-difference scores of an
independently written ordered-logit and multinomial-logit likelihood),
stacked over the union of the samples and sandwiched with N/(N-1) or G/(G-1).
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from numpy.testing import assert_allclose
from scipy import stats
from scipy.special import expit, log_expit

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics.postest.scores import (
    GaussianObjective, LogitObjective, MultinomialScores, PoissonObjective, ProbitObjective,
)
from openecon.engines.optimize import check_derivatives
from openecon.models import ResultBundle


def make_data(seed=5, n=500):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(2, n))
    frame = pd.DataFrame({"x1": x1, "x2": x2, "g": rng.integers(0, 40, size=n)})
    frame["y1"] = 1 + x1 + rng.normal(size=n) * (1 + np.abs(x1))
    frame["y2"] = 2 - x2 + 0.5 * x1 + rng.normal(size=n) + 0.5 * (frame["y1"] - 1 - x1)
    frame["b"] = (0.2 + x1 - 0.4 * x2 + rng.logistic(size=n) > 0).astype(int)
    frame["exposure"] = rng.uniform(0.5, 2.0, size=n)
    frame["count"] = rng.poisson(frame["exposure"] * np.exp(0.2 + 0.3 * x1))
    latent = 0.8 * x1 - 0.5 * x2 + rng.logistic(size=n)
    frame["ordered"] = np.digitize(latent, [-1.0, 0.3, 1.5])
    frame["choice"] = rng.choice([1, 2, 3], size=n, p=[0.3, 0.45, 0.25])
    return frame


@pytest.fixture(scope="module")
def data():
    return make_data()


def design(frame, columns):
    return np.column_stack([np.ones(len(frame)), frame[columns].to_numpy(float)])


def ols_pieces(frame, y, columns):
    x = design(frame, columns)
    target = frame[y].to_numpy(float)
    beta = np.linalg.lstsq(x, target, rcond=None)[0]
    u = target - x @ beta
    n = len(u)
    s2 = u @ u / n
    scores = np.column_stack([x * (u / s2)[:, None], 0.5 * (u**2 / s2 - 1)])
    k = x.shape[1]
    hessian = np.zeros((k + 1, k + 1))
    hessian[:k, :k] = -x.T @ x / s2
    hessian[k, k] = -n / 2
    return np.r_[beta, np.log(s2)], scores, hessian


def sandwich(blocks, rows, n_total, clusters=None):
    """V = D (q S'S) D with zero-padded score rows."""
    widths = [b[1].shape[1] for b in blocks]
    stacked = np.zeros((n_total, sum(widths)))
    start = 0
    for (params, scores, hessian), where in zip(blocks, rows, strict=True):
        stacked[where, start:start + scores.shape[1]] = scores
        start += scores.shape[1]
    d = np.zeros((sum(widths), sum(widths)))
    start = 0
    for params, scores, hessian in blocks:
        w = scores.shape[1]
        d[start:start + w, start:start + w] = np.linalg.inv(-hessian)
        start += w
    if clusters is None:
        meat = stacked.T @ stacked * n_total / (n_total - 1)
    else:
        codes, groups = np.unique(clusters, return_inverse=True)[1], len(np.unique(clusters))
        sums = np.zeros((groups, stacked.shape[1]))
        np.add.at(sums, codes, stacked)
        meat = sums.T @ sums * groups / (groups - 1)
    return d @ meat @ d


def test_two_ols_fits_reproduce_sur_robust_cross_covariance(data):
    m1 = oe.ols(data=data, y="y1", x=["x1"])
    m2 = oe.ols(data=data, y="y2", x=["x1", "x2"])
    joint = oe.suest(m1, m2, data=data, names=["a", "b"])
    pieces = [ols_pieces(data, "y1", ["x1"]), ols_pieces(data, "y2", ["x1", "x2"])]
    n = len(data)
    expected = sandwich(pieces, [np.arange(n)] * 2, n)
    assert_allclose(np.array(joint.covariance_matrix), expected, rtol=1e-9, atol=1e-15)
    assert_allclose([c.estimate for c in joint.coefficients],
                    np.r_[pieces[0][0], pieces[1][0]], rtol=1e-10)
    assert [c.term for c in joint.coefficients] == [
        "a:Intercept", "a:x1", "a:/lnvar", "b:Intercept", "b:x1", "b:x2", "b:/lnvar"]
    assert [c.equation for c in joint.coefficients][:3] == ["a_mean", "a_mean", "a_lnvar"]
    # The slope block of each model is HC0 times N/(N-1) (statsmodels HC0).
    hc0 = sm.OLS(data["y2"], design(data, ["x1", "x2"])).fit(cov_type="HC0").cov_params()
    assert_allclose(np.array(joint.covariance_matrix)[3:6, 3:6], hc0 * n / (n - 1), rtol=1e-9)
    # Explicit SUR-robust cross block (X1'X1)^-1 sum x1 x2' u1 u2 (X2'X2)^-1 N/(N-1).
    x1, x2 = design(data, ["x1"]), design(data, ["x1", "x2"])
    u1 = data["y1"] - x1 @ pieces[0][0][:2]
    u2 = data["y2"] - x2 @ pieces[1][0][:3]
    cross = (np.linalg.inv(x1.T @ x1) @ (x1 * u1.to_numpy()[:, None]).T
             @ (x2 * u2.to_numpy()[:, None]) @ np.linalg.inv(x2.T @ x2) * n / (n - 1))
    assert_allclose(np.array(joint.covariance_matrix)[:2, 3:6], cross, rtol=1e-9)
    se = np.sqrt(np.diag(expected))
    estimates = np.r_[pieces[0][0], pieces[1][0]]
    assert_allclose([c.p_value for c in joint.coefficients],
                    2 * stats.norm.sf(np.abs(estimates / se)), rtol=1e-8, atol=1e-300)
    assert joint.inference["covariance"] == "robust" and joint.inference["use_t"] is False
    assert_allclose(joint.inference["small_sample_correction"], n / (n - 1))
    assert joint.nobs == n and joint.provenance["postestimation"]["method"] == "suest"


def logit_pieces(frame, y, columns, rows=None):
    sub = frame if rows is None else frame.iloc[rows]
    model = sm.Logit(sub[y], design(sub, columns))
    params = model.fit(disp=0, tol=1e-12, maxiter=100).params.to_numpy()
    return params, model.score_obs(params), model.hessian(params)


def probit_pieces(frame, y, columns):
    model = sm.Probit(frame[y], design(frame, columns))
    params = model.fit(disp=0, tol=1e-12, maxiter=100).params.to_numpy()
    return params, model.score_obs(params), model.hessian(params)


def test_logit_and_probit(data):
    lg = oe.logit(data=data, y="b", x=["x1", "x2"])
    pr = oe.probit(data=data, y="b", x=["x1", "x2"])
    joint = oe.suest(lg, pr, data=data)
    n = len(data)
    pieces = [logit_pieces(data, "b", ["x1", "x2"]), probit_pieces(data, "b", ["x1", "x2"])]
    expected = sandwich(pieces, [np.arange(n)] * 2, n)
    assert_allclose(np.array(joint.covariance_matrix), expected, rtol=2e-6)
    assert [c.term for c in joint.coefficients][:3] == ["logit:Intercept", "logit:x1",
                                                        "logit:x2"]
    robust = sm.Logit(data["b"], design(data, ["x1", "x2"])).fit(disp=0, cov_type="HC0")
    assert_allclose(np.array(joint.covariance_matrix)[:3, :3],
                    robust.cov_params() * n / (n - 1), rtol=2e-6)
    # A cross-model Wald test of equal x1 effects (scaled by 1.6) uses the joint covariance.
    v = np.array(joint.covariance_matrix)
    contrast = np.zeros(6)
    contrast[1], contrast[4] = 1, -1.6
    b = np.array([c.estimate for c in joint.coefficients])
    statistic = (contrast @ b) ** 2 / (contrast @ v @ contrast)
    assert statistic >= 0 and np.isfinite(statistic)


def test_overlapping_samples_and_clusters(data):
    holes = data.copy()
    holes.loc[holes.index[:60], "x2"] = np.nan
    pois = oe.poisson(data=holes, y="count", x=["x1", "x2"], exposure="exposure",
                      missing="drop")
    lg = oe.logit(data=holes, y="b", x=["x1"])
    joint = oe.suest(pois, lg, data=holes, cluster="g")
    rows_p = np.arange(60, len(holes))
    sub = holes.iloc[rows_p]
    model = sm.Poisson(sub["count"], design(sub, ["x1", "x2"]), exposure=sub["exposure"])
    params = model.fit(disp=0, tol=1e-12, maxiter=100).params.to_numpy()
    pieces = [(params, model.score_obs(params), model.hessian(params)),
              logit_pieces(holes, "b", ["x1"])]
    expected = sandwich(pieces, [rows_p, np.arange(len(holes))], len(holes),
                        clusters=holes["g"].to_numpy())
    assert_allclose(np.array(joint.covariance_matrix), expected, rtol=2e-6)
    assert joint.nobs == len(holes)
    assert joint.inference["covariance"] == "cluster"
    assert joint.inference["cluster_count"] == holes["g"].nunique()
    assert any("different estimation samples" in w for w in joint.warnings)
    assert joint.extra["models"][0]["nobs"] == len(rows_p)


def ordered_score_rows(theta, x, codes, cdf, pdf):
    """Analytic scores of log(F(c_y - x'b) - F(c_(y-1) - x'b)), written independently."""
    k = x.shape[1]
    cuts = theta[k:]
    eta = x @ theta[:k]
    edges = np.r_[-np.inf, cuts, np.inf]
    upper, lower = edges[codes + 1] - eta, edges[codes] - eta
    prob = cdf(upper) - cdf(lower)
    f_u = np.where(np.isfinite(upper), pdf(np.where(np.isfinite(upper), upper, 0)), 0)
    f_l = np.where(np.isfinite(lower), pdf(np.where(np.isfinite(lower), lower, 0)), 0)
    scores = np.zeros((len(codes), len(theta)))
    scores[:, :k] = x * (-(f_u - f_l) / prob)[:, None]
    for j in range(len(cuts)):
        scores[:, k + j] = (np.where(codes == j, f_u, 0) - np.where(codes == j + 1, f_l, 0)) / prob
    return scores


def mlogit_score_rows(theta, x, codes, base, n_categories):
    k = x.shape[1]
    others = [j for j in range(n_categories) if j != base]
    eta = np.zeros((len(codes), n_categories))
    for slot, j in enumerate(others):
        eta[:, j] = x @ theta[slot * k:(slot + 1) * k]
    p = np.exp(eta - np.logaddexp.reduce(eta, axis=1, keepdims=True))
    return np.column_stack([x * ((codes == j) - p[:, j])[:, None] for j in others])


def numerical_pieces(score_fn, theta, step=1e-6):
    """Analytic score rows; Hessian by central differences of their sum."""
    p = len(theta)
    hessian = np.zeros((p, p))
    for j in range(p):
        e = np.zeros(p)
        e[j] = step
        hessian[:, j] = (score_fn(theta + e).sum(axis=0)
                         - score_fn(theta - e).sum(axis=0)) / (2 * step)
    return theta, score_fn(theta), (hessian + hessian.T) / 2


def logistic_pdf(t):
    return expit(t) * expit(-t)


def test_ordered_and_multinomial_logit(data):
    ol = oe.ologit(data=data, y="ordered", x=["x1", "x2"])
    ml = oe.mlogit(data=data, y="choice", x=["x1"])
    joint = oe.suest(ol, ml, data=data, names=["ord", "mnl"])
    x_ord = data[["x1", "x2"]].to_numpy()
    codes_ord = np.searchsorted(np.unique(data["ordered"]), data["ordered"])
    theta_ord = np.array([c.estimate for c in ol.coefficients])
    x_ml = design(data, ["x1"])
    categories = sorted(data["choice"].unique())
    codes_ml = np.searchsorted(categories, data["choice"])
    base = categories.index(ml.extra["base"])
    theta_ml = np.array([c.estimate for c in ml.coefficients])
    pieces = [
        numerical_pieces(lambda t: ordered_score_rows(t, x_ord, codes_ord, expit, logistic_pdf),
                         theta_ord),
        numerical_pieces(lambda t: mlogit_score_rows(t, x_ml, codes_ml, base, 3), theta_ml),
    ]
    n = len(data)
    expected = sandwich(pieces, [np.arange(n)] * 2, n)
    assert_allclose(np.array(joint.covariance_matrix), expected, rtol=1e-6)
    terms = [c.term for c in joint.coefficients]
    assert terms[:2] == ["ord:x1", "ord:x2"] and "ord:/cut3" in terms
    assert any(term.startswith("mnl:") and term.endswith(":x1") for term in terms)


def test_oprobit_scores(data):
    op = oe.oprobit(data=data, y="ordered", x=["x1"])
    m1 = oe.ols(data=data, y="y1", x=["x1"])
    joint = oe.suest(op, m1, data=data)
    x = data[["x1"]].to_numpy()
    codes = np.searchsorted(np.unique(data["ordered"]), data["ordered"])
    theta = np.array([c.estimate for c in op.coefficients])
    n = len(data)
    piece = numerical_pieces(lambda t: ordered_score_rows(t, x, codes, stats.norm.cdf,
                                                          stats.norm.pdf), theta)
    expected = sandwich([piece, ols_pieces(data, "y1", ["x1"])], [np.arange(n)] * 2, n)
    assert_allclose(np.array(joint.covariance_matrix), expected, rtol=1e-6)


def test_objective_derivatives_are_analytic(data):
    rng = np.random.default_rng(0)
    x = torch.as_tensor(design(data, ["x1", "x2"]))
    y = torch.as_tensor(data["y1"].to_numpy())
    b = torch.as_tensor(data["b"].to_numpy(float))
    count = torch.as_tensor(data["count"].to_numpy(float))
    offset = torch.as_tensor(np.log(data["exposure"].to_numpy()))
    cases = [
        (GaussianObjective(x, y), torch.tensor([0.9, 1.1, -0.2, 0.4], dtype=torch.float64)),
        (LogitObjective(x, b), torch.tensor([0.1, 0.8, -0.3], dtype=torch.float64)),
        (ProbitObjective(x, b), torch.tensor([0.1, 0.5, -0.2], dtype=torch.float64)),
        (PoissonObjective(x, count, offset), torch.tensor([0.1, 0.2, 0.1], dtype=torch.float64)),
    ]
    codes = torch.as_tensor(data["choice"].to_numpy() - 1)
    cases.append((MultinomialScores(x, codes, 3, 1),
                  torch.as_tensor(rng.normal(scale=0.3, size=6))))
    for objective, theta in cases:
        report = check_derivatives(objective, theta)
        assert report["gradient_max_rel_error"] < 1e-7, (type(objective).__name__, report)
        assert report["hessian_max_rel_error"] < 1e-6, (type(objective).__name__, report)
        rows = objective.score_rows(theta)
        assert_allclose(rows.sum(dim=0).numpy(), objective(theta)[1].numpy(), rtol=1e-10,
                        atol=1e-9)
    # Logit value against an explicit formula.
    theta = cases[1][1]
    eta = x.numpy() @ theta.numpy()
    assert_allclose(float(cases[1][0](theta)[0]),
                    np.sum(b.numpy() * log_expit(eta) + (1 - b.numpy()) * log_expit(-eta)))


def test_suest_errors(data):
    m1 = oe.ols(data=data, y="y1", x=["x1"])
    lg = oe.logit(data=data, y="b", x=["x1"])
    weighted = oe.poisson(data=data, y="count", x=["x1"], weights="exposure",
                          weight_type="aweight")
    tampered = lg.model_copy(update={"coefficients": [
        c.model_copy(update={"estimate": c.estimate + 0.3}) for c in lg.coefficients]})
    cases = [
        (lambda: oe.suest(m1, data=data), "invalid_spec"),
        (lambda: oe.suest(m1, weighted, data=data), "suest_unsupported"),
        (lambda: oe.suest(m1, lg, data=data.assign(y1=data["y1"] * 2)), "data_mismatch"),
        (lambda: oe.suest(m1, lg, data=data, names=["a"]), "invalid_spec"),
        (lambda: oe.suest(m1, lg, data=data, names=["a", "a"]), "invalid_spec"),
        (lambda: oe.suest(m1, lg, data=data, cluster="nope"), "missing_columns"),
        (lambda: oe.suest(m1, tampered, data=data), "suest_mismatch"),
        (lambda: oe.suest(m1, "x", data=data), "invalid_result"),
    ]
    for call, code in cases:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code, (code, str(error.value))
    joint = oe.suest(m1, lg, data=data)
    with pytest.raises(AnalysisError) as error:
        oe.suest(joint, m1, data=data)
    assert error.value.code == "suest_unsupported"
    for procedure in (oe.lrtest,):
        with pytest.raises(AnalysisError):
            procedure(joint, m1)
    with pytest.raises(AnalysisError) as error:
        oe.bootstrap(joint, data, reps=5)
    assert error.value.code == "bootstrap_unsupported"


def test_categorical_terms_and_missing_rows(data):
    frame = data.assign(sector=pd.Categorical(np.where(data["x2"] > 0.5, "c",
                                                       np.where(data["x2"] > -0.5, "b", "a"))))
    frame.loc[frame.index[:5], "x1"] = np.nan
    m1 = oe.ols(data=frame, y="y1", x=["x1", "sector"], categorical=["sector"], missing="drop")
    m2 = oe.logit(data=frame.dropna(subset=["x1"]).reset_index(drop=True), y="b", x=["x1"])
    with pytest.raises(AnalysisError) as error:
        oe.suest(m1, m2, data=frame)
    assert error.value.code == "data_mismatch"
    m3 = oe.poisson(data=frame, y="count", x=["x1", "sector"], categorical=["sector"],
                    missing="drop")
    joint = oe.suest(m1, m3, data=frame)
    assert "ols:sector[b]" in [c.term for c in joint.coefficients]
    assert joint.nobs == len(frame) - 5


def test_round_trip_summary_latex_and_access(data):
    m1 = oe.ols(data=data, y="y1", x=["x1"])
    lg = oe.logit(data=data, y="b", x=["x1"])
    joint = oe.suest(m1, lg, data=data)
    assert ResultBundle.model_validate_json(joint.model_dump_json()) == joint
    text = joint.summary()
    assert "Seemingly unrelated estimation" in text and "[ols_lnvar]" in text
    assert "ols:x1" in str(joint.to_latex())
    assert "suest" in dir(oe)


def test_collinear_terms_are_skipped(data):
    frame = data.assign(x3=data["x1"] - data["x2"])
    pois = oe.poisson(data=frame, y="count", x=["x1", "x2", "x3"], exposure="exposure")
    assert [c.term for c in pois.coefficients] == ["Intercept", "x1", "x2"]
    lg = oe.logit(data=frame, y="b", x=["x1"])
    joint = oe.suest(pois, lg, data=frame)
    assert [c.term for c in joint.coefficients][:3] == ["poisson:Intercept", "poisson:x1",
                                                        "poisson:x2"]
    model = sm.Poisson(frame["count"], design(frame, ["x1", "x2"]), exposure=frame["exposure"])
    params = model.fit(disp=0, tol=1e-12, maxiter=100).params.to_numpy()
    n = len(frame)
    expected = sandwich([(params, model.score_obs(params), model.hessian(params)),
                         logit_pieces(frame, "b", ["x1"])], [np.arange(n)] * 2, n)
    assert_allclose(np.array(joint.covariance_matrix), expected, rtol=2e-6)
