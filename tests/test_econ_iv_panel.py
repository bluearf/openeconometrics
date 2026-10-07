"""xtivreg and ivreghdfe against dummy-variable and explicit NumPy two-stage least squares."""

import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from numpy.testing import assert_allclose
from pydantic import ValidationError
from scipy import stats

import openecon as oe
from openecon.analysis import AnalysisError, fit
from openecon.econometrics import registry
from openecon.models import ModelSpec, ResultBundle


def make_panel(seed=77, n_id=50, periods=6, drop=(3, 50, 51, 200, 201)):
    rng = np.random.default_rng(seed)
    n = n_id * periods
    frame = pd.DataFrame({"id": np.repeat(np.arange(n_id), periods),
                          "year": np.tile(np.arange(2000, 2000 + periods), n_id)})
    effect = np.repeat(rng.normal(size=n_id), periods)
    v = rng.normal(size=n)
    frame["g"] = np.repeat(rng.normal(size=n_id), periods)            # time invariant
    frame["region"] = frame.id % 10
    frame["x1"] = rng.normal(size=n) + 0.5 * effect
    frame["z1"] = rng.normal(size=n) + 0.3 * effect
    frame["z2"] = rng.normal(size=n)
    frame["p"] = frame.z1 + 0.5 * frame.z2 + 0.4 * frame.x1 + effect + v
    frame["y"] = (1 + 2 * frame.p - frame.x1 + 0.5 * frame.g + effect + 0.6 * v
                  + rng.normal(size=n) * (1 + 0.3 * np.abs(frame.z2)))
    return frame.drop(index=list(drop)).reset_index(drop=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


def estimates(result):
    return np.array([c.estimate for c in result.coefficients])


def errors(result):
    return np.array([c.std_error for c in result.coefficients])


def tsls(y, x, z, w=None):
    """2SLS by explicit projection: b, residuals, (Xhat'W Xhat)^-1, Xhat."""
    w = np.ones(len(y)) if w is None else w
    xhat = z @ np.linalg.lstsq(z * np.sqrt(w)[:, None], x * np.sqrt(w)[:, None], rcond=None)[0]
    bread = np.linalg.inv(xhat.T @ (xhat * w[:, None]))
    b = bread @ (xhat.T @ (w * y))
    return b, y - x @ b, bread, xhat


def cluster_meat(scores, codes):
    sums = pd.DataFrame(scores).groupby(np.asarray(codes), sort=False).sum().to_numpy()
    return sums.T @ sums


def group_mean(values, codes):
    return pd.DataFrame(values).groupby(np.asarray(codes)).transform("mean").to_numpy()


def within(values, codes):
    values = np.asarray(values, dtype=float)
    return values - group_mean(values, codes).reshape(values.shape)


def xt(frame, **options):
    options = {"x": ["x1"], "endog": ["p"], "instruments": ["z1", "z2"], "panel": "id",
               "time": "year", **options}
    return oe.xtivreg(data=frame, y="y", **options)


def corr2(a, b):
    return np.corrcoef(a, b)[0, 1] ** 2


# ---- xtivreg: fixed effects -----------------------------------------------------------------


def test_xtivreg_fe_matches_dummy_variable_two_stage_least_squares(panel):
    result = xt(panel, small=True)
    assert xt(panel).inference["use_t"] is False          # Stata's xtivreg default: z, chi2
    y, ids = panel.y.to_numpy(), panel.id.to_numpy()
    n, groups, k = len(y), panel.id.nunique(), 2
    dummies = pd.get_dummies(panel.id).to_numpy(float)
    x, z = panel[["x1", "p"]].to_numpy(), panel[["x1", "z1", "z2"]].to_numpy()
    b, u, bread, _ = tsls(y, np.column_stack([x, dummies]), np.column_stack([z, dummies]))
    df = n - groups - k
    sigma_e = np.sqrt(u @ u / df)
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "p"]
    assert_allclose(estimates(result)[1:], b[:k], rtol=1e-10)
    assert_allclose(errors(result)[1:], np.sqrt(np.diag(bread)[:k]) * sigma_e, rtol=1e-9)
    # Stata's _cons: grand means added back to the within-transformed system.
    constant = np.ones((n, 1))
    xs = np.column_stack([constant, within(x, ids) + x.mean(axis=0)])
    zs = np.column_stack([constant, within(z, ids) + z.mean(axis=0)])
    b_c, u_c, bread_c, _ = tsls(within(y, ids) + y.mean(), xs, zs)
    assert estimates(result)[0] == pytest.approx(y.mean() - x.mean(axis=0) @ b[:k], rel=1e-10)
    assert_allclose(np.asarray(result.covariance_matrix), bread_c * (u_c @ u_c) / df, rtol=1e-9)
    assert result.inference["use_t"] and result.inference["df_inference"] == df
    assert_allclose([c.p_value for c in result.coefficients],
                    2 * stats.t.sf(np.abs(estimates(result) / errors(result)), df), rtol=1e-8,
                    atol=1e-300)
    u_i = (pd.Series(y).groupby(ids).mean() - pd.DataFrame(x).groupby(ids).mean() @ b[:k])
    sigma_u = u_i.std(ddof=1)
    xb = x @ b[:k]
    metrics = result.metrics
    assert metrics["sigma_e"] == pytest.approx(sigma_e, rel=1e-10)
    assert metrics["sigma_u"] == pytest.approx(sigma_u, rel=1e-10)
    assert metrics["rho"] == pytest.approx(sigma_u ** 2 / (sigma_u ** 2 + sigma_e ** 2), rel=1e-10)
    # within: R-squared of the transformed IV regression; between/overall: squared correlations
    assert metrics["r_squared_within"] == pytest.approx(
        1 - u @ u / (within(y, ids) ** 2).sum(), rel=1e-9)
    assert metrics["r_squared_between"] == pytest.approx(
        corr2(pd.Series(xb).groupby(ids).mean(), pd.Series(y).groupby(ids).mean()), rel=1e-9)
    assert metrics["r_squared_overall"] == pytest.approx(corr2(xb, y), rel=1e-9)
    # Stata's e(corr): corr(u_i, x_it b) over the N observations
    assert metrics["corr_u_xb"] == pytest.approx(
        np.corrcoef(u_i.loc[ids].to_numpy(), xb)[0, 1], rel=1e-9)
    assert (metrics["n_groups"], metrics["t_min"], metrics["t_max"]) == (groups, 4, 6)
    assert metrics["df_resid"] == df and metrics["n_instruments"] == 2
    test = result.tests["model"]
    v = np.asarray(result.covariance_matrix)[1:, 1:]
    assert test["statistic"] == pytest.approx(b[:k] @ np.linalg.solve(v, b[:k]) / k, rel=1e-9)
    assert (test["df"], test["df2"], test["distribution"]) == (k, df, "F")
    pooled = tsls(y, np.column_stack([constant, x]), np.column_stack([constant, z]))[1]
    fixed = result.tests["fixed_effects"]
    assert fixed["statistic"] == pytest.approx(
        ((pooled @ pooled - u @ u) / (groups - 1)) / (u @ u / df), rel=1e-9)
    assert (fixed["df"], fixed["df2"]) == (groups - 1, df)
    assert result.title == "Fixed-effects (within) IV regression"
    assert result.provenance["sample_order"] == "sorted by id, year"


def test_xtivreg_fe_omits_time_invariant_columns_and_supports_z_statistics(panel):
    result = xt(panel, x=["x1", "g"], instruments=["z1", "z2", "region"])
    assert result.provenance["omitted_terms"] == ["g"]
    assert result.extra["omitted_instruments"] == ["region"]
    assert_allclose(estimates(result), estimates(xt(panel)), rtol=1e-10)
    normal = xt(panel)
    assert normal.inference["use_t"] is False and normal.tests["model"]["distribution"] == "chi2"
    assert_allclose(errors(normal), errors(xt(panel, small=True)), rtol=1e-12)
    assert_allclose([c.p_value for c in normal.coefficients],
                    2 * stats.norm.sf(np.abs(estimates(normal) / errors(normal))), rtol=1e-8,
                    atol=1e-300)
    with pytest.raises(AnalysisError) as error:
        xt(panel, instruments=["region", "g"])
    assert error.value.code == "underidentified"


@pytest.mark.parametrize("cluster", [None, "region"])
def test_xtivreg_fe_robust_is_clustered_on_the_panel(panel, cluster):
    options = {"cluster": cluster} if cluster else {"covariance": "robust"}
    result = xt(panel, small=True, **options)
    y, ids = panel.y.to_numpy(), panel.id.to_numpy()
    n = len(y)
    x, z = panel[["x1", "p"]].to_numpy(), panel[["x1", "z1", "z2"]].to_numpy()
    constant = np.ones((n, 1))
    xs = np.column_stack([constant, within(x, ids) + x.mean(axis=0)])
    zs = np.column_stack([constant, within(z, ids) + z.mean(axis=0)])
    _, u, bread, xhat = tsls(within(y, ids) + y.mean(), xs, zs)
    codes = panel[cluster].to_numpy() if cluster else ids
    g = len(np.unique(codes))
    # xtreg's convention: K = slopes + constant, the nested panel effects are not counted.
    factor = g / (g - 1) * (n - 1) / (n - 3)
    expected = bread @ cluster_meat(xhat * u[:, None], codes) @ bread * factor
    assert_allclose(np.asarray(result.covariance_matrix), expected, rtol=1e-9)
    assert result.inference["df_inference"] == g - 1 and result.inference["cluster_count"] == g
    assert result.inference["covariance"] == ("cluster" if cluster else "robust")
    assert "fixed_effects" not in result.tests and result.tests["model"]["df2"] == g - 1
    if cluster is None:
        assert result.inference["correction"].startswith("vce(robust) = vce(cluster panel)")


# ---- xtivreg: first differences, between -------------------------------------------------------


def test_xtivreg_fd_uses_consecutive_periods_with_a_constant(panel):
    result = xt(panel, model="fd", small=True)
    ordered = panel.sort_values(["id", "year"])
    diff = ordered[["y", "x1", "p", "z1", "z2"]].diff()
    keep = (ordered.id.diff() == 0) & (ordered.year.diff() == 1)
    d, ids = diff[keep].to_numpy(), ordered.id[keep].to_numpy()
    rows = len(d)
    constant = np.ones((rows, 1))
    x = np.column_stack([constant, d[:, 1:3]])
    b, u, bread, xhat = tsls(d[:, 0], x, np.column_stack([constant, d[:, 1], d[:, 3:]]))
    assert result.nobs == rows and [c.term for c in result.coefficients] == ["Intercept", "x1", "p"]
    assert_allclose(estimates(result), b, rtol=1e-10)
    assert_allclose(np.asarray(result.covariance_matrix), bread * (u @ u) / (rows - 3), rtol=1e-9)
    assert result.inference["df_inference"] == rows - 3 and result.inference["use_t"]
    tss = ((d[:, 0] - d[:, 0].mean()) ** 2).sum()
    assert result.metrics["r_squared"] == pytest.approx(1 - u @ u / tss, rel=1e-10)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(u @ u / (rows - 3)), rel=1e-10)
    assert result.extra["differenced_observations"] == rows
    assert result.extra["dropped_for_differencing"] == len(panel) - rows
    assert any("consecutive periods" in warning for warning in result.warnings)
    robust = xt(panel, model="fd", covariance="robust", small=True)
    g = len(np.unique(ids))
    expected = bread @ cluster_meat(xhat * u[:, None], ids) @ bread \
        * g / (g - 1) * (rows - 1) / (rows - 3)
    assert_allclose(np.asarray(robust.covariance_matrix), expected, rtol=1e-9)
    assert robust.inference["df_inference"] == g - 1
    with pytest.raises(AnalysisError) as error:
        xt(panel, model="fd", time=None)
    assert error.value.code == "invalid_spec"


def test_xtivreg_be_is_two_stage_least_squares_on_panel_means(panel):
    result = xt(panel, model="be", x=["x1", "g"], small=True)
    means = panel.groupby("id")[["y", "x1", "g", "p", "z1", "z2", "region"]].mean()
    groups = len(means)
    constant = np.ones((groups, 1))
    x = np.column_stack([constant, means[["x1", "g", "p"]]])
    z = np.column_stack([constant, means[["x1", "g", "z1", "z2"]]])
    b, u, bread, xhat = tsls(means.y.to_numpy(), x, z)
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "g", "p"]
    assert_allclose(estimates(result), b, rtol=1e-10)
    assert_allclose(np.asarray(result.covariance_matrix), bread * (u @ u) / (groups - 4), rtol=1e-9)
    assert result.inference["df_inference"] == groups - 4 and result.nobs == len(panel)
    level = panel[["x1", "g", "p"]].to_numpy() @ b[1:]
    assert result.metrics["r_squared_between"] == pytest.approx(
        1 - u @ u / ((means.y - means.y.mean()) ** 2).sum(), rel=1e-9)
    assert result.metrics["r_squared_overall"] == pytest.approx(corr2(level, panel.y), rel=1e-9)
    assert result.metrics["r_squared_within"] == pytest.approx(
        corr2(within(level, panel.id), within(panel.y, panel.id)), rel=1e-9)
    assert result.metrics["rmse"] == pytest.approx(np.sqrt(u @ u / (groups - 4)), rel=1e-10)
    scores = xhat * u[:, None]
    robust = xt(panel, model="be", x=["x1", "g"], covariance="robust")
    assert_allclose(np.asarray(robust.covariance_matrix),
                    bread @ (scores.T @ scores) @ bread * groups / (groups - 4), rtol=1e-9)
    clustered = xt(panel, model="be", x=["x1", "g"], cluster="region", small=True)
    expected = bread @ cluster_meat(scores, means.region.to_numpy()) @ bread \
        * 10 / 9 * (groups - 1) / (groups - 4)
    assert_allclose(np.asarray(clustered.covariance_matrix), expected, rtol=1e-9)
    assert clustered.inference["df_inference"] == 9
    with pytest.raises(AnalysisError) as error:
        xt(panel, model="be", cluster="year")
    assert error.value.code == "cluster_varies_within_panel"


# ---- xtivreg: random effects -------------------------------------------------------------------


def g2sls(frame, x_names, ec2sls=False):
    """G2SLS / EC2SLS step by step; returns (b, V_conventional, parts)."""
    y, ids = frame.y.to_numpy(), frame.id.to_numpy()
    n, groups = len(y), frame.id.nunique()
    constant = np.ones((n, 1))
    x1 = frame[x_names].to_numpy()
    x = np.column_stack([constant, x1, frame.p])
    z = np.column_stack([constant, x1, frame[["z1", "z2"]]])
    varying = [c for c in x_names if c != "g"]
    xw = within(frame[[*varying, "p"]].to_numpy(), ids)
    zw = within(frame[[*varying, "z1", "z2"]].to_numpy(), ids)
    u_w = tsls(within(y, ids), xw, zw)[1]
    sigma_e2 = u_w @ u_w / (n - groups - xw.shape[1])
    means = lambda a: pd.DataFrame(a).groupby(ids).mean().to_numpy()      # noqa: E731
    sizes = pd.Series(ids).groupby(ids).size().to_numpy()
    # between 2SLS in which each panel mean appears T_i times
    u_b = tsls(means(y)[:, 0], means(x), means(z), sizes.astype(float))[1]
    # Swamy-Arora adapted to unbalanced panels: T_i-weighted between residuals and the trace
    xb = means(x)
    trace = np.trace(np.linalg.solve(xb.T @ (xb * sizes[:, None]),
                                     xb.T @ (xb * (sizes ** 2)[:, None])))
    sigma_u2 = max(0.0, ((sizes * u_b ** 2).sum() - (groups - x.shape[1]) * sigma_e2)
                   / (n - trace))
    theta = 1 - np.sqrt(sigma_e2 / (sizes * sigma_u2 + sigma_e2))
    shrink = pd.Series(theta, index=np.unique(ids)).loc[ids].to_numpy()[:, None]
    ys = y - shrink[:, 0] * group_mean(y, ids)[:, 0]
    xs = x - shrink * group_mean(x, ids)
    if ec2sls:
        level = np.column_stack([x1, frame[["z1", "z2"]]])
        instruments = np.column_stack([constant, within(level, ids), group_mean(level, ids)])
        xhat = instruments @ np.linalg.pinv(instruments) @ xs
    else:
        instruments = z - shrink * group_mean(z, ids)
        xhat = instruments @ np.linalg.lstsq(instruments, xs, rcond=None)[0]
    bread = np.linalg.inv(xhat.T @ xhat)
    b = bread @ (xhat.T @ ys)
    resid = ys - xs @ b
    parts = {"sigma_e2": sigma_e2, "sigma_u2": sigma_u2, "theta": theta, "xhat": xhat,
             "resid": resid, "bread": bread, "trace": trace}
    # conventional VCE of the 2SLS regression on the transformed data: RSS*/(N-K)
    return b, bread * (resid @ resid) / (n - x.shape[1]), parts


@pytest.mark.parametrize("ec2sls", [False, True])
def test_xtivreg_re_matches_explicit_g2sls_and_ec2sls(panel, ec2sls):
    result = xt(panel, model="re", x=["x1", "g"], ec2sls=ec2sls)
    b, v, parts = g2sls(panel, ["x1", "g"], ec2sls)
    assert [c.term for c in result.coefficients] == ["Intercept", "x1", "g", "p"]
    assert_allclose(estimates(result), b, rtol=1e-9)
    assert_allclose(np.asarray(result.covariance_matrix), v, rtol=1e-8)
    metrics = result.metrics
    assert metrics["sigma_e"] == pytest.approx(np.sqrt(parts["sigma_e2"]), rel=1e-10)
    assert metrics["sigma_u"] == pytest.approx(np.sqrt(parts["sigma_u2"]), rel=1e-9)
    assert metrics["rho"] == pytest.approx(
        parts["sigma_u2"] / (parts["sigma_u2"] + parts["sigma_e2"]), rel=1e-9)
    assert metrics["theta"] is None                                   # unbalanced panel
    components = result.extra["variance_components"]
    assert components["trace"] == pytest.approx(parts["trace"], rel=1e-10)
    assert components["within_regressors"] == 2
    assert result.extra["theta"]["max"] == pytest.approx(parts["theta"].max(), rel=1e-9)
    assert result.extra["estimator"] == ("ec2sls" if ec2sls else "g2sls")
    # z inference and a Wald chi2 model test by default.
    assert result.inference["use_t"] is False
    test = result.tests["model"]
    assert test["distribution"] == "chi2" and test["df"] == 3
    assert test["statistic"] == pytest.approx(b[1:] @ np.linalg.solve(v[1:, 1:], b[1:]), rel=1e-8)
    xb = panel[["x1", "g", "p"]].to_numpy() @ b[1:]
    assert metrics["r_squared_overall"] == pytest.approx(corr2(xb, panel.y), rel=1e-9)
    robust = xt(panel, model="re", x=["x1", "g"], ec2sls=ec2sls, covariance="robust")
    n, groups = len(panel), panel.id.nunique()
    scores = parts["xhat"] * parts["resid"][:, None]
    expected = parts["bread"] @ cluster_meat(scores, panel.id) @ parts["bread"] \
        * groups / (groups - 1) * (n - 1) / (n - 4)
    assert_allclose(np.asarray(robust.covariance_matrix), expected, rtol=1e-8)
    small = xt(panel, model="re", x=["x1", "g"], ec2sls=ec2sls, small=True)
    assert small.inference["use_t"] and small.inference["df_inference"] == n - 4
    assert small.tests["model"]["distribution"] == "F"


def test_xtivreg_re_on_a_balanced_panel_reports_theta(panel):
    balanced = make_panel(seed=5, drop=())
    g2 = xt(balanced, model="re")
    ec = xt(balanced, model="re", ec2sls=True)
    b, v, parts = g2sls(balanced, ["x1"])
    assert_allclose(estimates(g2), b, rtol=1e-9)
    assert g2.metrics["theta"] == pytest.approx(parts["theta"][0], rel=1e-9)
    assert_allclose(estimates(ec), g2sls(balanced, ["x1"], ec2sls=True)[0], rtol=1e-9)
    assert not np.allclose(estimates(ec), estimates(g2), rtol=1e-6)
    assert ec.extra["instrument_columns_used"] == 7 and g2.extra["instrument_columns_used"] == 4
    assert ec.title == "EC2SLS random-effects IV regression"


def test_xtivreg_contract_errors_and_round_trip(panel):
    info = registry.get("xtivreg")
    assert info.panel == "required" and info.weights == () and info.intercept == "always"
    with pytest.raises(AnalysisError) as error:
        xt(panel, ec2sls=True)
    assert error.value.code == "invalid_spec"
    repeated = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(AnalysisError) as error:
        xt(repeated)
    assert error.value.code == "repeated_time_values"
    with pytest.raises((ValidationError, AnalysisError)):
        oe.xtivreg(data=panel, y="y", endog=["p"], instruments=["z1"], panel=None)
    with pytest.raises((ValidationError, AnalysisError)):
        fit(ModelSpec(estimator="xtivreg", outcome="y", panel="id", weights="x1",
                      weight_type="aweight", columns={"endogenous": ["p"],
                                                      "instruments": ["z1"]}), data=panel)
    with pytest.raises((ValidationError, AnalysisError)):
        xt(panel, model="mle")
    with pytest.raises((ValidationError, AnalysisError)):
        xt(panel, cluster=["id", "region"])
    missing = panel.copy()
    missing.loc[5, "z1"] = np.nan
    with pytest.raises(AnalysisError) as error:
        xt(missing)
    assert error.value.code == "missing_values"
    assert xt(missing, missing="drop").nobs == len(panel) - 1
    for model in ("fe", "fd", "be", "re"):
        result = xt(panel, model=model, x=[])
        assert ResultBundle.model_validate_json(result.model_dump_json()) == result
        assert result.provenance["model"] == model and "Intercept" in result.summary()
        assert "\\begin{tabular}" in str(result.to_latex())
        assert result.provenance["stata_parity_validated"] is False
    shuffled = panel.sample(frac=1.0, random_state=1)
    assert_allclose(estimates(xt(shuffled, model="re")), estimates(xt(panel, model="re")),
                    rtol=1e-10)


# ---- ivreghdfe -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def firms():
    rng = np.random.default_rng(404)
    n_firm, periods = 40, 8
    n = n_firm * periods
    frame = pd.DataFrame({"firm": np.repeat(np.arange(n_firm), periods),
                          "year": np.tile(np.arange(periods), n_firm)})
    v = rng.normal(size=n)
    frame["size"] = np.repeat(rng.normal(size=n_firm), periods)       # constant within firm
    frame["x1"] = rng.normal(size=n)
    frame["z1"] = rng.normal(size=n)
    frame["z2"] = rng.normal(size=n)
    frame["p"] = frame.z1 + 0.5 * frame.z2 + 0.1 * frame.firm + v
    frame["y"] = (1.5 * frame.p - frame.x1 + 0.3 * frame.year + 0.05 * frame.firm + 0.7 * v
                  + rng.normal(size=n) * (1 + 0.4 * np.abs(frame.z1)))
    frame["fw"] = rng.integers(1, 4, size=n).astype(float)
    frame["aw"] = rng.uniform(0.5, 2.0, size=n)
    return frame


def hd(frame, **options):
    options = {"x": ["x1"], "endog": ["p"], "instruments": ["z1", "z2"],
               "absorb": ["firm", "year"], **options}
    return oe.ivreghdfe(data=frame, y="y", **options)


def lsdv(frame, w=None, absorb=("firm", "year")):
    """Dummy-variable 2SLS: (b, residuals, bread, Xhat, rank of X) with slopes first."""
    dummies = [pd.get_dummies(frame[absorb[0]]).to_numpy(float)]
    dummies += [pd.get_dummies(frame[name]).to_numpy(float)[:, 1:] for name in absorb[1:]]
    d = np.column_stack(dummies)
    x = np.column_stack([frame.x1, frame.p, d])
    z = np.column_stack([frame.x1, frame.z1, frame.z2, d])
    return (*tsls(frame.y.to_numpy(), x, z, w), x.shape[1])


@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "cluster"])
def test_ivreghdfe_matches_dummy_variable_two_stage_least_squares(firms, covariance):
    options = {"cluster": "firm"} if covariance == "cluster" else {"covariance": covariance}
    result = hd(firms, **options)
    b, u, bread, xhat, rank = lsdv(firms)
    n, k = len(firms), 2
    assert [c.term for c in result.coefficients] == ["x1", "p"]
    assert_allclose(estimates(result), b[:k], rtol=1e-9)
    scores = xhat * u[:, None]
    if covariance == "nonrobust":
        expected, df = bread * (u @ u) / (n - rank), n - rank
        assert result.metrics["df_absorbed"] == rank - k == 47
    elif covariance == "robust":
        expected, df = bread @ (scores.T @ scores) @ bread * n / (n - rank), n - rank
    else:
        # The firm effects are nested in the clusters and cost no degrees of freedom.
        total = k + 8
        factor = 40 / 39 * (n - 1) / (n - total)
        expected = bread @ cluster_meat(scores, firms.firm) @ bread * factor
        df = 39
        assert result.metrics["df_absorbed"] == 8
        assert [row["nested"] for row in result.extra["absorbed"]] == [True, False]
    assert_allclose(np.asarray(result.covariance_matrix), expected[:k, :k], rtol=1e-8)
    assert result.inference["use_t"] and result.inference["df_inference"] == df
    tss = ((firms.y - firms.y.mean()) ** 2).sum()
    assert result.metrics["r_squared"] == pytest.approx(1 - u @ u / tss, rel=1e-10)
    assert result.metrics["n_singletons_dropped"] == 0 and result.metrics["df_model"] == k
    assert result.title == "IV (2SLS) regression with absorbed fixed effects"
    assert [row["column"] for row in result.extra["absorbed"]] == ["firm", "year"]
    assert result.tests["model"]["distribution"] == "F" and result.tests["model"]["df2"] == df


def test_ivreghdfe_diagnostics_use_the_absorbed_degrees_of_freedom(firms):
    result = hd(firms)
    b, u, bread, xhat, rank = lsdv(firms)
    n = len(firms)
    d = np.column_stack([pd.get_dummies(firms.firm).to_numpy(float),
                         pd.get_dummies(firms.year).to_numpy(float)[:, 1:]])
    z = np.column_stack([firms.x1, firms.z1, firms.z2, d])
    first = sm.OLS(firms.p.to_numpy(), z).fit()
    restriction = np.zeros((2, z.shape[1]))
    restriction[0, 1] = restriction[1, 2] = 1
    test = first.f_test(restriction)
    row = result.extra["first_stage"][0]
    assert row["f_statistic"] == pytest.approx(float(test.fvalue), rel=1e-8)
    assert (row["df"], row["df2"]) == (2, n - z.shape[1])
    assert result.tests["cragg_donald"]["statistic"] == pytest.approx(row["f_statistic"], rel=1e-9)
    explained = u @ z @ np.linalg.lstsq(z, u, rcond=None)[0]
    sargan = result.tests["overid_sargan"]
    assert sargan["statistic"] == pytest.approx(n * explained / (u @ u), rel=1e-7)
    x = np.column_stack([firms.x1, firms.p, d])
    augmented = sm.OLS(firms.y.to_numpy(), np.column_stack([x, first.resid])).fit()
    wu = result.tests["endog_wu_hausman"]
    assert wu["statistic"] == pytest.approx(augmented.tvalues[-1] ** 2, rel=1e-7)
    assert (wu["df"], wu["df2"]) == (1, n - rank - 1)


@pytest.mark.parametrize("method", ["liml", "gmm"])
def test_ivreghdfe_liml_and_gmm_run_on_the_residualized_data(firms, method):
    d = np.column_stack([pd.get_dummies(firms.firm).to_numpy(float),
                         pd.get_dummies(firms.year).to_numpy(float)[:, 1:]])
    purge = lambda a: a - d @ np.linalg.lstsq(d, a, rcond=None)[0]        # noqa: E731
    y, x1 = purge(firms.y.to_numpy()), purge(firms[["x1"]].to_numpy())
    x2, z2 = purge(firms[["p"]].to_numpy()), purge(firms[["z1", "z2"]].to_numpy())
    n, k_total = len(y), 2 + 47
    x, z = np.column_stack([x1, x2]), np.column_stack([x1, z2])
    if method == "liml":
        result = hd(firms, method="liml")
        big = np.column_stack([y, x2])
        m_z = np.eye(n) - z @ np.linalg.pinv(z)
        m_1 = np.eye(n) - x1 @ np.linalg.pinv(x1)
        kappa = np.linalg.eigvals(np.linalg.solve(big.T @ m_z @ big,
                                                  big.T @ m_1 @ big)).real.min()
        xk = (np.eye(n) - kappa * m_z) @ x
        bread = np.linalg.inv(xk.T @ x)
        b = bread @ (xk.T @ y)
        u = y - x @ b
        assert result.metrics["kappa"] == pytest.approx(kappa, rel=1e-9)
        assert_allclose(estimates(result), b, rtol=1e-8)
        assert_allclose(np.asarray(result.covariance_matrix), bread * (u @ u) / (n - k_total),
                        rtol=1e-8)
        basmann = result.tests["basmann_f"]
        assert basmann["statistic"] == pytest.approx((kappa - 1) * (n - 3 - 47), rel=1e-8)
        assert basmann["df2"] == n - 3 - 47
    else:
        result = hd(firms, method="gmm", covariance="robust")
        u1 = tsls(y, x, z)[1]
        s = (z * u1[:, None]).T @ (z * u1[:, None])
        a = z.T @ x
        bread = np.linalg.inv(a.T @ np.linalg.solve(s, a))
        b = bread @ (a.T @ np.linalg.solve(s, z.T @ y))
        g = z.T @ (y - x @ b)
        assert_allclose(estimates(result), b, rtol=1e-8)
        assert_allclose(np.asarray(result.covariance_matrix), bread * n / (n - k_total), rtol=1e-8)
        assert result.tests["hansen_j"]["statistic"] == pytest.approx(
            g @ np.linalg.solve(s, g), rel=1e-7)
        assert result.metrics["j"] == pytest.approx(result.tests["hansen_j"]["statistic"])
        unadjusted = hd(firms, method="gmm")
        assert_allclose(estimates(unadjusted), estimates(hd(firms)), rtol=1e-12)


def test_ivreghdfe_small_false_reports_asymptotic_statistics(firms):
    _, u, bread, xhat, _ = lsdv(firms)
    n = len(firms)
    scores = xhat * u[:, None]
    plain = hd(firms, small=False)
    assert_allclose(np.asarray(plain.covariance_matrix), (bread * (u @ u) / n)[:2, :2], rtol=1e-8)
    assert plain.inference["use_t"] is False and plain.tests["model"]["distribution"] == "chi2"
    assert plain.metrics["rmse"] == pytest.approx(np.sqrt(u @ u / n), rel=1e-10)
    robust = hd(firms, small=False, covariance="robust")
    assert_allclose(np.asarray(robust.covariance_matrix),
                    (bread @ (scores.T @ scores) @ bread)[:2, :2], rtol=1e-8)
    clustered = hd(firms, small=False, cluster="firm")
    # ivreg2 without small: no G/(G-1) and no (N-1)/(N-K).
    assert_allclose(np.asarray(clustered.covariance_matrix),
                    (bread @ cluster_meat(scores, firms.firm) @ bread)[:2, :2], rtol=1e-8)
    assert clustered.inference["df_inference"] is None
    two_way = hd(firms, cluster=["firm", "year"])
    assert two_way.inference["cluster_counts"] == [40, 8]
    # A dimension nested in ANY cluster column costs no degrees of freedom (as the linear
    # family's reghdfe): both are nested here, so only the constant is counted, whatever
    # the order of the cluster columns.
    assert two_way.inference["df_inference"] == 7 and two_way.metrics["df_absorbed"] == 1
    swapped = hd(firms, cluster=["year", "firm"])
    assert swapped.metrics["df_absorbed"] == 1
    assert_allclose(np.asarray(swapped.covariance_matrix),
                    np.asarray(two_way.covariance_matrix), rtol=1e-10)


def test_ivreghdfe_weights(firms):
    n = len(firms)
    w = firms.aw.to_numpy() * n / firms.aw.sum()
    b, u, bread, _, rank = lsdv(firms, w)
    weighted = hd(firms, weights="aw", weight_type="aweight")
    assert_allclose(estimates(weighted), b[:2], rtol=1e-8)
    assert_allclose(np.asarray(weighted.covariance_matrix),
                    (bread * (w * u * u).sum() / (n - rank))[:2, :2], rtol=1e-8)
    frequency = hd(firms, weights="fw", weight_type="fweight", cluster="firm")
    expanded = firms.loc[firms.index.repeat(firms.fw.astype(int))].reset_index(drop=True)
    duplicated = hd(expanded, cluster="firm")
    assert frequency.nobs == duplicated.nobs == int(firms.fw.sum())
    assert_allclose(estimates(frequency), estimates(duplicated), rtol=1e-8)
    assert_allclose(errors(frequency), errors(duplicated), rtol=1e-7)
    sampling = hd(firms, weights="aw", weight_type="pweight")
    assert sampling.spec.covariance == "robust"
    assert_allclose(estimates(sampling), estimates(weighted), rtol=1e-12)
    with pytest.raises(AnalysisError) as error:
        hd(firms, weights="aw", weight_type="pweight", covariance="nonrobust")
    assert error.value.code == "unsupported_covariance"


def test_ivreghdfe_one_way_clustered_equals_xtivreg_fe(firms):
    absorbed = hd(firms, absorb=["firm"], cluster="firm")
    panel_fe = oe.xtivreg(data=firms, y="y", x=["x1"], endog=["p"], instruments=["z1", "z2"],
                          panel="firm", time="year", covariance="robust", small=True)
    assert_allclose(estimates(absorbed), estimates(panel_fe)[1:], rtol=1e-10)
    assert_allclose(errors(absorbed), errors(panel_fe)[1:], rtol=1e-9)
    assert absorbed.metrics["df_absorbed"] == 1
    assert absorbed.extra["constant_degree_of_freedom_added"] is True
    assert absorbed.inference["df_inference"] == panel_fe.inference["df_inference"] == 39
    plain = hd(firms, absorb=["firm"])
    fe = oe.xtivreg(data=firms, y="y", x=["x1"], endog=["p"], instruments=["z1", "z2"],
                    panel="firm", time="year")
    assert_allclose(errors(plain), errors(fe)[1:], rtol=1e-9)


def test_ivreghdfe_singletons_absorbed_columns_and_errors(firms):
    extra = firms.iloc[[0]].assign(firm=999)
    frame = pd.concat([firms, extra], ignore_index=True)
    result = hd(frame.assign(size_copy=frame["size"] * 2), x=["x1", "size"],
                instruments=["z1", "z2", "size_copy"])
    assert result.metrics["n_singletons_dropped"] == 1 and result.nobs == len(firms)
    assert any("singleton" in warning for warning in result.warnings)
    assert result.provenance["omitted_terms"] == ["size"]
    assert result.extra["omitted_instruments"] == ["size_copy"]
    assert_allclose(estimates(result), estimates(hd(firms)), rtol=1e-9)
    kept = oe.ivreghdfe(data=frame, y="y", x=["x1"], endog=["p"], instruments=["z1", "z2"],
                        absorb=["firm", "year"], drop_singletons=False)
    assert kept.nobs == len(frame) and kept.metrics["n_singletons_dropped"] == 0
    assert_allclose(estimates(kept), estimates(hd(firms)), rtol=1e-8)
    with pytest.raises(AnalysisError) as error:
        hd(firms.assign(only=firms["size"]), instruments=["only"])
    assert error.value.code == "underidentified"
    with pytest.raises(AnalysisError) as error:
        hd(firms, tolerance=2.0)
    assert error.value.code == "invalid_option"
    with pytest.raises((ValidationError, AnalysisError)):
        oe.ivreghdfe(data=firms, y="y", endog=["p"], instruments=["z1"], absorb=[])
    with pytest.raises((ValidationError, AnalysisError)):
        fit(ModelSpec(estimator="ivreghdfe", outcome="y", intercept=True,
                      columns={"endogenous": ["p"], "instruments": ["z1"], "absorb": ["firm"]}),
            data=firms)
    with pytest.raises((ValidationError, AnalysisError)):
        hd(firms, covariance="hac")
    no_exogenous = hd(firms, x=None)
    assert [c.term for c in no_exogenous.coefficients] == ["p"]
    assert ResultBundle.model_validate_json(result.model_dump_json()) == result
    json.dumps(result.model_dump())
    text = hd(firms, cluster="firm").summary()
    assert "IV (2SLS) regression with absorbed fixed effects" in text and "Kleibergen-Paap" in text
    assert "\\begin{tabular}" in str(result.to_latex())
    assert registry.get("ivreghdfe").intercept == "never"
