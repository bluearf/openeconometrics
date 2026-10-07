"""Independent oracles for oe.xtgee.

The documented Stata conventions (phi = sum r^2 / N, moment estimators of the
working correlation) are re-implemented in plain NumPy with explicit per-panel
correlation matrices and inverses; at the converged alpha, statsmodels GEE
with the correlation held fixed (update_dep=False) must reproduce the
coefficients, the model-based covariance and the robust sandwich (times
G/(G-1)).
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def make_panel(seed=0, panels=80, periods=5, drop=30):
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(panels), periods)
    t = np.tile(np.arange(periods), panels)
    n = panels * periods
    frame = pd.DataFrame({"id": ids, "t": t, "x": rng.normal(size=n),
                          "w": rng.normal(size=n) + 2.0})
    u = rng.normal(size=panels)[ids]
    frame["y"] = 1 + 0.5 * frame.x - 0.2 * frame.w + u + rng.normal(size=n)
    frame["b"] = (rng.random(n) < 1 / (1 + np.exp(-(0.2 + 0.6 * frame.x + u)))).astype(float)
    frame["e"] = rng.uniform(0.5, 2, size=n)
    frame["c"] = rng.poisson(frame.e * np.exp(0.1 + 0.4 * frame.x + 0.5 * u))
    frame["g"] = rng.gamma(3, np.exp(0.2 + 0.3 * frame.x) / 3)
    if drop:
        # Drop whole trailing periods so panels stay equally spaced (no gaps).
        cut = rng.integers(2, periods + 1, size=panels)
        frame = frame[frame.t < cut[frame.id]]
    return frame.reset_index(drop=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


LINKS = {
    "identity": (lambda e: e, lambda e: np.ones_like(e)),
    "logit": (lambda e: 1 / (1 + np.exp(-e)), lambda e: np.exp(-e) / (1 + np.exp(-e)) ** 2),
    "log": (np.exp, np.exp),
    "reciprocal": (lambda e: 1 / e, lambda e: -1 / e ** 2),
}
VARIANCE = {"gaussian": lambda m: np.ones_like(m), "binomial": lambda m: m * (1 - m),
            "poisson": lambda m: m, "gamma": lambda m: m ** 2}


def numpy_gee(frame, y, xcols, family, link, corr, order=1, offset=None):
    """Plain NumPy GEE with explicit R_i: returns b, alpha/R, phi, bread, panel scores."""
    lag = {"ar1": 1, "stationary": order, "nonstationary": order}.get(corr, 0)
    if lag:
        # Stata: only panels with n_i > g enter a lag-dependent structure.
        sizes = frame.groupby("id").id.transform("size")
        keep = (sizes > lag).to_numpy()
        frame = frame[keep].reset_index(drop=True)
        if offset is not None:
            offset = offset[keep]
    X = np.column_stack([np.ones(len(frame)), frame[xcols].to_numpy()])
    Y = frame[y].to_numpy(float)
    off = np.zeros(len(Y)) if offset is None else offset
    ids = frame.id.to_numpy()
    t = frame.groupby("id").cumcount().to_numpy()          # position within the panel
    panels = [np.flatnonzero(ids == i) for i in np.unique(ids)]
    T = t.max() + 1
    inverse, derivative = LINKS[link]
    n, p = X.shape
    beta = np.linalg.lstsq(X, Y if link == "identity" else np.full(n, 0.0), rcond=None)[0]
    if family == "gamma":
        beta = np.r_[1 / Y.mean(), np.zeros(p - 1)]
    if family in {"binomial", "poisson"}:
        beta = np.zeros(p)
        if family == "poisson":
            beta[0] = np.log(Y.mean())

    def pieces(beta):
        eta = X @ beta + off
        mu = inverse(eta)
        sd = np.sqrt(VARIANCE[family](mu))
        return (Y - mu) / sd, X * (derivative(eta) / sd)[:, None]

    def correlation(r, phi):
        if corr == "independent":
            return np.eye(T)
        if corr == "exchangeable":
            num = sum(r[i].sum() ** 2 - (r[i] ** 2).sum() for i in panels)
            den = sum(len(i) * (len(i) - 1) for i in panels)
            a = num / den / phi
            return (1 - a) * np.eye(T) + a * np.ones((T, T))
        cross = np.zeros((T, T))
        count = np.zeros((T, T))
        for i in panels:
            cross[np.ix_(t[i], t[i])] += np.outer(r[i], r[i])
            count[np.ix_(t[i], t[i])] += 1
        lag = np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
        R = np.eye(T)
        # Stata's xtgee Methods and formulas: per-panel moments divided by n_i.
        per_panel = sum((r[i] ** 2).sum() / len(i) for i in panels)
        if corr in {"ar1", "stationary"}:
            values = [sum((r[i][:-k] * r[i][k:]).sum() / len(i) for i in panels) / per_panel
                      for k in range(1, order + 1)]
            if corr == "ar1":
                return values[0] ** lag
            for k, v in enumerate(values, start=1):
                R[lag == k] = v
            return R
        keep = (lag > 0) & ((lag <= order) if corr == "nonstationary" else True)
        R[keep] = cross[keep] / count[keep] / (per_panel / len(panels))
        return R

    def assemble(beta, R):
        r, d = pieces(beta)
        bread = np.zeros((p, p))
        scores = []
        for i in panels:
            Ri = np.linalg.inv(R[np.ix_(t[i], t[i])])
            bread += d[i].T @ Ri @ d[i]
            scores.append(d[i].T @ Ri @ r[i])
        return bread, np.array(scores)

    for _ in range(100):                       # independence start
        bread, scores = assemble(beta, np.eye(T))
        step = np.linalg.solve(bread, scores.sum(0))
        beta = beta + step
        if np.abs(step).max() < 1e-12:
            break
    for _ in range(300):
        r, _ = pieces(beta)
        R = correlation(r, (r ** 2).sum() / n)
        bread, scores = assemble(beta, R)
        step = np.linalg.solve(bread, scores.sum(0))
        beta = beta + step
        if np.abs(step).max() < 1e-12:
            break
    r, _ = pieces(beta)
    R = correlation(r, (r ** 2).sum() / n)
    bread, scores = assemble(beta, R)
    return beta, R, (r ** 2).sum() / n, bread, scores


@pytest.mark.parametrize("family,link,y", [("gaussian", "identity", "y"),
                                           ("binomial", "logit", "b"),
                                           ("poisson", "log", "c")])
@pytest.mark.parametrize("corr", ["exchangeable", "independent", "ar1"])
def test_closed_form_structures_match_numpy(panel, family, link, y, corr):
    result = oe.xtgee(data=panel, y=y, x=["x", "w"], panel="id", time="t", family=family,
                      corr=corr)
    beta, R, phi, bread, scores = numpy_gee(panel, y, ["x", "w"], family, link, corr)
    assert_allclose([c.estimate for c in result.coefficients], beta, rtol=1e-8, atol=1e-10)
    scale = 1.0 if family != "gaussian" else phi
    assert_allclose(np.array(result.covariance_matrix), scale * np.linalg.inv(bread), rtol=1e-7)
    if corr in {"exchangeable", "ar1"}:
        assert_allclose(result.extra["alpha"][0], R[0, 1], rtol=1e-8)
    assert_allclose(result.metrics["scale"], scale, rtol=1e-10)
    robust = oe.xtgee(data=panel, y=y, x=["x", "w"], panel="id", time="t", family=family,
                      corr=corr, covariance="robust")
    groups = len(scores)
    sandwich = np.linalg.inv(bread) @ scores.T @ scores @ np.linalg.inv(bread)
    assert_allclose(np.array(robust.covariance_matrix), groups / (groups - 1) * sandwich,
                    rtol=1e-6)


@pytest.mark.parametrize("corr,order", [("unstructured", 1), ("stationary", 2),
                                        ("nonstationary", 2)])
def test_patterned_structures_match_numpy(corr, order):
    panel = make_panel(seed=1, panels=200)
    result = oe.xtgee(data=panel, y="y", x=["x"], panel="id", time="t", corr=corr,
                      corr_order=order if corr != "unstructured" else None)
    beta, R, phi, bread, _ = numpy_gee(panel, "y", ["x"], "gaussian", "identity", corr, order)
    assert_allclose([c.estimate for c in result.coefficients], beta, rtol=1e-8)
    assert_allclose(np.array(result.extra["working_correlation"]), R, rtol=1e-8, atol=1e-12)
    assert_allclose(np.array(result.covariance_matrix), phi * np.linalg.inv(bread), rtol=1e-7)


def test_fixed_alpha_matches_statsmodels(panel):
    for family, sm_family, y in (("binomial", sm.families.Binomial(), "b"),
                                 ("gaussian", sm.families.Gaussian(), "y")):
        result = oe.xtgee(data=panel, y=y, x=["x", "w"], panel="id", family=family)
        robust = oe.xtgee(data=panel, y=y, x=["x", "w"], panel="id", family=family,
                          covariance="robust")
        structure = sm.cov_struct.Exchangeable()
        model = sm.GEE(panel[y], sm.add_constant(panel[["x", "w"]]), groups=panel.id,
                       family=sm_family, cov_struct=structure, update_dep=False)
        structure.dep_params = result.extra["alpha"][0]
        ref = model.fit(ddof_scale=0)
        assert_allclose([c.estimate for c in result.coefficients], ref.params, rtol=1e-7)
        groups = panel.id.nunique()
        assert_allclose(np.array(robust.covariance_matrix), ref.cov_robust * groups / (groups - 1),
                        rtol=1e-6)
        assert_allclose(np.array(result.covariance_matrix), ref.cov_naive, rtol=1e-6)


def test_gamma_family_and_scale_options(panel):
    result = oe.xtgee(data=panel, y="g", x=["x"], panel="id", family="gamma", link="log")
    beta, R, phi, bread, _ = numpy_gee(panel, "g", ["x"], "gamma", "log", "exchangeable")
    assert_allclose([c.estimate for c in result.coefficients], beta, rtol=1e-8)
    assert_allclose(result.metrics["scale"], phi, rtol=1e-10)
    fixed = oe.xtgee(data=panel, y="g", x=["x"], panel="id", family="gamma", link="log",
                     scale=2.0)
    assert_allclose(np.array(fixed.covariance_matrix), 2.0 * np.linalg.inv(bread), rtol=1e-7)
    x2 = oe.xtgee(data=panel, y="b", x=["x"], panel="id", family="binomial", scale="x2")
    assert x2.metrics["scale"] == x2.extra["pearson_dispersion"] != 1.0
    nmp = oe.xtgee(data=panel, y="y", x=["x"], panel="id", nmp=True)
    assert_allclose(nmp.metrics["scale"], nmp.metrics["pearson_chi2"] / (len(panel) - 2))
    dev = oe.xtgee(data=panel, y="y", x=["x"], panel="id", scale="dev")
    assert_allclose(dev.metrics["scale"], dev.metrics["deviance"] / len(panel))


def test_exposure_equals_log_offset(panel):
    a = oe.xtgee(data=panel, y="c", x=["x"], panel="id", family="poisson", exposure="e")
    b = oe.xtgee(data=panel.assign(le=np.log(panel.e)), y="c", x=["x"], panel="id",
                 family="poisson", offset="le")
    beta, *_ = numpy_gee(panel, "c", ["x"], "poisson", "log", "exchangeable",
                         offset=np.log(panel.e.to_numpy()))
    assert_allclose([c.estimate for c in a.coefficients], beta, rtol=1e-8)
    assert_allclose([c.estimate for c in a.coefficients], [c.estimate for c in b.coefficients])


def test_non_positive_definite_unstructured_is_refused(panel):
    with pytest.raises(AnalysisError) as error:
        oe.xtgee(data=panel, y="y", x=["x"], panel="id", time="t", corr="unstructured")
    assert error.value.code == "working_correlation_not_pd"


def test_time_gaps_and_force():
    frame = make_panel(seed=2, drop=0)
    gapped = frame[~((frame.t == 2) & (frame.id % 3 == 0))].reset_index(drop=True)
    with pytest.raises(AnalysisError) as error:
        oe.xtgee(data=gapped, y="y", x=["x"], panel="id", time="t", corr="ar1")
    assert error.value.code == "unequal_spacing"
    forced = oe.xtgee(data=gapped, y="y", x=["x"], panel="id", time="t", corr="ar1", force=True)
    ranked = gapped.assign(t=gapped.groupby("id").cumcount())
    beta, *_ = numpy_gee(ranked, "y", ["x"], "gaussian", "identity", "ar1")
    assert_allclose([c.estimate for c in forced.coefficients], beta, rtol=1e-8)
    # Exchangeable needs no time ordering at all.
    oe.xtgee(data=gapped, y="y", x=["x"], panel="id", time="t")


def test_collinearity_missing_and_errors(panel):
    result = oe.xtgee(data=panel.assign(x2=2 * panel.x), y="y", x=["x", "x2"], panel="id")
    assert result.provenance["omitted_terms"] == ["x2"]
    frame = panel.copy()
    frame.loc[[1, 4], "x"] = np.nan
    with pytest.raises(AnalysisError) as error:
        oe.xtgee(data=frame, y="y", x=["x"], panel="id")
    assert error.value.code == "missing_values"
    assert oe.xtgee(data=frame, y="y", x=["x"], panel="id", missing="drop").nobs == len(panel) - 2

    def code(**kwargs):
        options = {"data": panel, "y": "y", "x": ["x"], "panel": "id"}
        options.update(kwargs)
        with pytest.raises(AnalysisError) as error:
            oe.xtgee(**options)
        return error.value.code

    assert code(corr="ar1") == "invalid_spec"                        # no time column
    assert code(family="poisson", link="logit") == "invalid_spec"
    assert code(family="binomial") == "invalid_outcome"
    assert code(corr="exchangeable", corr_order=2) == "invalid_spec"
    assert code(scale="phi") == "invalid_spec"
    assert code(covariance="cluster") == "invalid_spec"
    assert code(x=["id"]) == "invalid_spec"


def test_round_trip_and_rendering(panel):
    result = oe.xtgee(data=panel, y="b", x=["x"], panel="id", family="binomial",
                      covariance="robust")
    assert type(result).model_validate_json(result.model_dump_json()) == result
    assert "Wald chi2" in result.summary()
    assert "tabular" in result.to_latex()
    assert result.extra["family"] == "binomial" and result.extra["link"] == "logit"
    assert len(result.extra["working_correlation"]) == 5
    assert oe.capabilities()["estimators"]["xtgee"]["panel"] == "required"


def test_negative_binomial_and_probit_link(panel):
    k = 0.5
    VARIANCE["nbinomial"] = lambda m: m + k * m ** 2
    result = oe.xtgee(data=panel, y="c", x=["x"], panel="id", family="nbinomial", nbk=k)
    beta, R, phi, bread, _ = numpy_gee(panel, "c", ["x"], "nbinomial", "log", "exchangeable")
    assert_allclose([c.estimate for c in result.coefficients], beta, rtol=1e-8)
    assert_allclose(np.array(result.covariance_matrix), np.linalg.inv(bread), rtol=1e-7)
    probit = oe.xtgee(data=panel, y="b", x=["x"], panel="id", family="binomial", link="probit")
    structure = sm.cov_struct.Exchangeable()
    model = sm.GEE(panel.b, sm.add_constant(panel[["x"]]), groups=panel.id,
                   family=sm.families.Binomial(link=sm.families.links.Probit()),
                   cov_struct=structure, update_dep=False)
    structure.dep_params = probit.extra["alpha"][0]
    ref = model.fit()
    assert_allclose([c.estimate for c in probit.coefficients], ref.params, rtol=1e-7)
    assert_allclose(np.array(probit.covariance_matrix), ref.cov_naive, rtol=1e-6)
