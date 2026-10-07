"""Dynamic panel GMM (xtdpd, xtabond, xtdpdsys): cross-checks, wrappers and conventions.

The dense oracle of ``test_econ_dpanel_oracle`` checks every estimator and
statistic formula; this file checks what that oracle cannot: the Windmeijer
derivative term against a numerical Jacobian of the two-step estimator, the
identity with 2SLS / two-step IV-GMM on differenced data (the iv family), the
orthogonal-deviations transform against its explicit Helmert matrix, the
Stata-style wrappers, small-sample conventions and Monte Carlo behaviour of
the estimators and tests.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose
from test_econ_dpanel_oracle import coefs, oracle, ses, simulate

import openecon as oe
from openecon.econometrics.dpanel import structure


@pytest.fixture(scope="module")
def panel():
    return simulate(seed=3)


def test_windmeijer_term_matches_numerical_jacobian_of_two_step_estimator(panel):
    """D = d b2(W(b)) / d b' at b1, evaluated by central differences, reproduces V_c."""
    options = {"system": True, "x": ["x", "w"],
               "gmm": [{"columns": ["y"], "lags": [2, 3]}, {"columns": ["w"], "lags": [1, 2]}],
               "iv": [{"columns": ["x"], "equation": "diff"}]}
    orc = oracle(panel, **options)
    blocks, a, b = orc["blocks"], orc["a"], orc["b"]

    def two_step(beta):
        omega = sum(np.outer(blk["z"].T @ (blk["y"] - blk["x"] @ beta),
                             blk["z"].T @ (blk["y"] - blk["x"] @ beta)) for blk in blocks)
        w = np.linalg.inv(omega)
        return np.linalg.solve(a.T @ w @ a, a.T @ w @ b)

    beta1, k = orc["beta1"], len(orc["beta1"])
    jac = np.zeros((k, k))
    for j in range(k):
        step = 1e-5 * max(1.0, abs(beta1[j]))
        up, down = beta1.copy(), beta1.copy()
        up[j] += step
        down[j] -= step
        jac[:, j] = (two_step(up) - two_step(down)) / (2 * step)
    v2, v1r = orc["v2"], orc["v1r"]
    numerical = v2 + jac @ v2 + v2 @ jac.T + jac @ v1r @ jac.T
    result = oe.xtdpd(data=panel, y="y", panel="id", time="year", twostep=True, robust=True,
                      **options)
    assert_allclose(coefs(result), two_step(beta1), rtol=1e-8)
    assert_allclose(ses(result), np.sqrt(np.diag(numerical)), rtol=1e-6)
    assert np.abs(jac).max() > 1e-3                 # the correction is not trivially zero


def _iv_panel(seed=11, n=150, periods=5):
    rng = np.random.default_rng(seed)
    ids, year = np.repeat(np.arange(n), periods), np.tile(np.arange(periods), n)
    u = rng.normal(size=n)[ids]
    z, q = rng.normal(size=(2, n * periods))
    x = rng.normal(size=n * periods) + u
    v = rng.normal(size=n * periods)
    w = 0.6 * z - 0.5 * q + 0.5 * v + u
    y = 1.0 * x - 0.5 * w + u + v + rng.normal(size=n * periods)
    return pd.DataFrame({"id": ids, "year": year, "y": y, "x": x, "w": w, "z": z, "q": q})


def test_static_difference_gmm_equals_2sls_and_iv_gmm_on_differenced_data():
    frame = _iv_panel()
    diffed = frame.sort_values(["id", "year"]).groupby("id")[["y", "x", "w", "z", "q"]].diff()
    diffed["id"] = frame.sort_values(["id", "year"]).id
    diffed = diffed.dropna().add_prefix("d").rename(columns={"did": "id"})
    common = dict(data=frame, y="y", x=["x", "w"], panel="id", time="year", lags=0, gmm=[],
                  iv=[{"columns": ["x", "z", "q"]}], h=1)
    iv_args = dict(data=diffed, y="dy", x=["dx"], endog=["dw"], instruments=["dz", "dq"],
                   intercept=False)
    tsls = oe.ivregress(**iv_args)
    one = oe.xtdpd(**common)
    assert one.nobs == tsls.nobs == len(diffed)
    assert_allclose(coefs(one), coefs(tsls), rtol=1e-10)
    # h(1) means H = I, so xtabond2's sigma2 is e'e / N (no factor 2): exactly 2SLS.
    assert_allclose(ses(one), ses(tsls), rtol=1e-9)
    # ... and the Sargan statistic is the iv family's N R^2 Sargan test of 2SLS.
    assert_allclose(one.tests["sargan"]["statistic"], tsls.tests["overid_sargan"]["statistic"],
                    rtol=1e-9)
    assert one.tests["sargan"]["df"] == tsls.tests["overid_sargan"]["df"] == 1
    clustered = oe.ivregress(**iv_args, covariance="cluster", cluster="id")
    robust = oe.xtdpd(**common, robust=True)
    groups = robust.metrics["n_groups"]
    assert_allclose(ses(robust), ses(clustered) * np.sqrt((groups - 1) / groups), rtol=1e-9)
    gmm = oe.ivregress(**iv_args, method="gmm", wmatrix="cluster", cluster="id")
    two = oe.xtdpd(**common, twostep=True)
    assert_allclose(coefs(two), coefs(gmm), rtol=1e-8)


def test_orthogonal_deviations_are_the_explicit_helmert_transform():
    codes = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2], dtype=torch.int64)
    period = torch.tensor([0, 1, 3, 4, 2, 3, 5, 0], dtype=torch.int64)
    grid = structure.layout(codes, period, 3)
    obs = structure.levels(grid, 0)
    fod = structure.OrthogonalDeviations(obs)
    dense = np.zeros((8, 8))
    for start, size in ((0, 4), (4, 3), (7, 1)):
        for j in range(size - 1):
            later = size - 1 - j
            c = np.sqrt(later / (later + 1))
            dense[start + j, start + j] = c
            dense[start + j, start + j + 1:start + size] = -c / later
    rows = fod.source.numpy()
    dense = dense[rows]
    values = torch.randn(8, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    assert_allclose(fod.apply(values).numpy(), dense @ values.numpy(), rtol=1e-13, atol=1e-14)
    image = torch.randn(len(rows), 2, dtype=torch.float64)
    assert_allclose(fod.adjoint(image).numpy(), dense.T @ image.numpy(), rtol=1e-13, atol=1e-14)
    assert_allclose(dense @ dense.T, np.eye(len(rows)), atol=1e-14)    # H = I for FOD
    # xtabond2 dates the deviation of period t at t + 1.
    assert fod.period.tolist() == (period[fod.source] + 1).tolist()
    diff = structure.Differences(grid, obs)
    assert diff.cur.tolist() == [1, 3, 5] and diff.prev.tolist() == [0, 2, 4]


def test_xtabond_and_xtdpdsys_build_stata_instrument_sets(panel):
    base = dict(data=panel, y="y", panel="id", time="year")
    ab = oe.xtabond(**base, x=["x"], predetermined=["w"], maxldep=3, maxlags=2, twostep=True,
                    robust=True)
    explicit = oe.xtdpd(**base, x=["x", "w"], gmm=[
        {"columns": ["y"], "lags": [2, 4], "equation": "diff"},
        {"columns": ["w"], "lags": [1, 2], "equation": "diff"}],
        iv=[{"columns": ["x"], "equation": "diff"}], twostep=True, robust=True, constant=False)
    assert_allclose(coefs(ab), coefs(explicit), rtol=1e-12)
    assert [c.term for c in ab.coefficients] == ["L1.y", "x", "w"]
    assert ab.extra["equations"] == "difference" and ab.spec.estimator == "xtdpd"
    sys_ = oe.xtdpdsys(**base, x=["x"], endogenous=["w"], instruments=["z"], lags=2)
    explicit = oe.xtdpd(**base, x=["x", "w"], lags=2, system=True, gmm=[
        {"columns": ["y"], "lags": [2, None]}, {"columns": ["w"], "lags": [2, None]}],
        iv=[{"columns": ["x", "z"], "equation": "diff"}])
    assert_allclose(coefs(sys_), coefs(explicit), rtol=1e-12)
    assert [c.term for c in sys_.coefficients] == ["Intercept", "L1.y", "L2.y", "x", "w"]
    assert "diff_hansen_level" not in sys_.tests                  # one-step nonrobust: no Hansen
    robust = oe.xtdpdsys(**base, x=["x"], covariance="robust")
    assert robust.spec.covariance == "robust" and "diff_hansen_level" in robust.tests


def test_xtdpdsys_and_default_xtdpd_agree(panel):
    a = oe.xtdpdsys(data=panel, y="y", x=["x"], panel="id", time="year", twostep=True)
    b = oe.xtdpd(data=panel, y="y", x=["x"], panel="id", time="year", system=True, twostep=True)
    assert_allclose(coefs(a), coefs(b), rtol=1e-12)


def test_small_sample_conventions(panel):
    """xtabond2: one-step nonrobust N/(N-k) with N - k df; otherwise G/(G-1) (N-1)/(N-k)
    with G - (constant) df (difference GMM has no constant, system GMM one)."""
    base = dict(data=panel, y="y", x=["x"], panel="id", time="year")
    for system in (False, True):
        for extra, key in (({"robust": True}, "g"), ({"twostep": True, "robust": True}, "g"),
                           ({"twostep": True}, "g"), ({}, "one")):
            large = oe.xtdpd(**base, system=system, **extra)
            small = oe.xtdpd(**base, system=system, small=True, **extra)
            n, k, g = small.nobs, len(small.coefficients), small.metrics["n_groups"]
            nt = small.extra["n_obs_transformed"]
            factor = {"g": g / (g - 1) * (n - 1) / (n - k), "one": nt / (nt - k)}[key]
            assert_allclose(ses(small), ses(large) * np.sqrt(factor), rtol=1e-10)
            assert small.inference["use_t"] and not large.inference["use_t"]
            df = n - k if key == "one" else g - int(system)
            assert small.inference["df_inference"] == df == small.metrics["df_resid"]
            assert small.inference["df_resid"] == df
            assert small.tests["model"]["df2"] == df
        assert small.tests["model"]["distribution"] == "F"
        assert large.tests["model"]["distribution"] == "chi2"


def _simulate_ar1(seed, n, periods, rho=0.5, beta=1.0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)
    # Mean-stationary start (E[y | u] = (1 + 0.5 beta) u / (1 - rho)), so system GMM is valid.
    y = (1 + 0.5 * beta) * u / (1 - rho) + rng.normal(size=n) * np.sqrt(
        (1 + beta ** 2) / (1 - rho ** 2))
    rows = []
    for t in range(periods):
        x = rng.normal(size=n) + 0.5 * u
        y = rho * y + beta * x + u + rng.normal(size=n)
        rows.append(pd.DataFrame({"id": np.arange(n), "t": t, "y": y, "x": x}))
    return pd.concat(rows, ignore_index=True)


def test_large_samples_recover_the_parameters():
    data = _simulate_ar1(5, n=4000, periods=6)
    base = dict(data=data, y="y", x=["x"], panel="id", time="t", twostep=True, robust=True)
    diff = oe.xtdpd(**base)
    system = oe.xtdpd(**base, system=True)
    fod = oe.xtdpd(**base, orthogonal=True, collapse=True)
    for result in (diff, system, fod):
        table = {c.term: c.estimate for c in result.coefficients}
        assert abs(table["L1.y"] - 0.5) < 0.05 and abs(table["x"] - 1.0) < 0.05
        assert result.tests["ar1"]["p_value"] < 1e-6       # MA(1) differenced errors
        assert result.tests["ar2"]["p_value"] > 0.001


def test_serial_correlation_and_overidentification_tests_have_roughly_correct_size():
    rejections = {"ar2": 0, "hansen": 0, "sargan": 0}
    replications = 120
    ar2, t_rho = [], []
    for seed in range(replications):
        data = _simulate_ar1(100 + seed, n=150, periods=5)
        result = oe.xtdpd(data=data, y="y", x=["x"], panel="id", time="t", twostep=True,
                          robust=True, collapse=True)
        for name in rejections:
            rejections[name] += result.tests[name]["p_value"] < 0.05
        ar2.append(result.tests["ar2"]["statistic"])
        lagged = result.coefficients[0]
        t_rho.append((lagged.estimate - 0.5) / lagged.std_error)
    for name, count in rejections.items():
        assert count / replications < 0.15, (name, count)
    # Under the null the AR(2) z and the Windmeijer-corrected t of L1.y are roughly N(0, 1).
    assert 0.75 < np.std(ar2) < 1.25 and abs(np.mean(ar2)) < 0.3
    assert 0.75 < np.std(t_rho) < 1.3 and abs(np.mean(t_rho)) < 0.4


def test_collinear_regressors_are_omitted_and_recorded(panel):
    data = panel.assign(x2=2 * panel.x)
    result = oe.xtdpd(data=data, y="y", x=["x", "x2"], panel="id", time="year",
                      iv=[{"columns": ["x"]}])
    assert [c.term for c in result.coefficients] == ["L1.y", "x"]
    assert result.provenance["omitted_terms"] == ["x2"]
    assert any("x2" in warning for warning in result.warnings)
    # A regressor that is constant within panels differences out of difference GMM.
    steady = panel.assign(c=panel.id % 3)
    result = oe.xtdpd(data=steady, y="y", x=["x", "c"], panel="id", time="year",
                      iv=[{"columns": ["x"]}])
    assert "c" in result.provenance["omitted_terms"]


def test_time_dummies_are_named_after_periods(panel):
    result = oe.xtdpd(data=panel, y="y", x=["x"], panel="id", time="year", time_dummies=True,
                      system=True)
    terms = [c.term for c in result.coefficients]
    first_level = sorted(panel.year.unique())[1]          # lag 1: the first period drops
    assert f"year[{first_level}]" not in terms
    assert f"year[{first_level + 1}]" in terms and terms[0] == "Intercept"
    assert result.extra["instrument_groups"]["time"]["columns"] >= 1


def test_large_offsets_keep_precision_against_high_precision_arithmetic():
    """With y around 1e6 the instrument Gram matrices are nearly singular in float64.

    OpenEcon never forms them (QR of the moment factors), so it matches a
    60-digit mpmath evaluation of the same formulas, which a float64 implementation
    with explicit inverses does not.
    """
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    data = simulate(seed=5, n=60, periods=6)
    data = data.assign(y=data.y + 1e6, x=data.x * 1e4)
    orc = oracle(data, system=True)
    blocks, n_inst, k = orc["blocks"], orc["n_instruments"], orc["k"]

    def matrix(values):
        return mp.matrix(np.atleast_2d(values).tolist())

    a = sum((matrix(b["z"]).T * matrix(b["x"]) for b in blocks), mp.zeros(n_inst, k))
    rhs = sum((matrix(b["z"]).T * matrix(b["y"][:, None]) for b in blocks), mp.zeros(n_inst, 1))
    zhz = sum((matrix(b["z"]).T * matrix(b["h"]) * matrix(b["z"]) for b in blocks),
              mp.zeros(n_inst, n_inst))
    w1 = zhz ** -1
    beta1 = (a.T * w1 * a) ** -1 * (a.T * w1 * rhs)
    omega = mp.zeros(n_inst, n_inst)
    for b in blocks:
        g = matrix(b["z"]).T * (matrix(b["y"][:, None]) - matrix(b["x"]) * beta1)
        omega += g * g.T
    w2 = omega ** -1
    beta2 = (a.T * w2 * a) ** -1 * (a.T * w2 * rhs)
    exact1 = np.array([float(v) for v in beta1])
    exact2 = np.array([float(v) for v in beta2])
    base = dict(data=data, y="y", x=["x"], panel="id", time="year", system=True)
    assert_allclose(coefs(oe.xtdpd(**base)), exact1, rtol=1e-8)
    assert_allclose(coefs(oe.xtdpd(**base, twostep=True)), exact2, rtol=1e-8)
