"""Independent cointegration oracles, sample, covariance and persistence checks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.models import ResultBundle


def synthetic(seed=72, n=220):
    rng = np.random.default_rng(seed)
    dx = rng.normal(size=(n, 2))
    x = dx.cumsum(0)
    error = rng.normal(size=n) + 0.35 * dx[:, 0]
    for t in range(1, n):
        error[t] += 0.35 * error[t - 1]
    return pd.DataFrame(
        {"y": 2 + x @ [1.2, -0.7] + error, "x": x[:, 0], "z": x[:, 1], "t": np.arange(n)}
    )


@pytest.mark.parametrize("method", ["fmols", "ccr", "dols"])
@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize(
    "kernel,bw", [("bartlett", 0), ("bartlett", 5), ("parzen", 5), ("quadratic_spectral", 5)]
)
def test_coefficients_full_covariance_independent_arch(method, trend, kernel, bw):
    oracle = pytest.importorskip("arch.unitroot.cointegration")
    df = synthetic()
    result = getattr(oe, method)(
        data=df, y="y", x=["x", "z"], time="t", trend=trend, bandwidth=bw, kernel=kernel
    )
    cls = {
        "fmols": oracle.FullyModifiedOLS,
        "ccr": oracle.CanonicalCointegratingReg,
        "dols": oracle.DynamicOLS,
    }[method]
    args = {"leads": 0, "lags": 0} if method == "dols" else {}
    ref = cls(df.y, df[["x", "z"]], trend=trend, **args).fit(
        bandwidth=bw, kernel=kernel.replace("_", ""), df_adjust=False
    )
    count = len(ref.params)
    params, v = np.array(ref.params), np.array(ref.cov)
    if method in ("ccr", "dols") and trend in ("ct", "ctt"):
        # arch restarts the transformed time index at 1; we retain original
        # periods (2..T). Map its coefficients/covariance into our basis.
        change = np.eye(count)
        change[2, 3] = -1
        if trend == "ctt":
            change[2, 4], change[3, 4] = 1, -2
        params, v = change @ params, change @ v @ change.T
    np.testing.assert_allclose(
        [c.estimate for c in result.coefficients[:count]], params, rtol=2e-9, atol=2e-9
    )
    np.testing.assert_allclose(
        np.array(result.covariance_matrix)[:count, :count], v, rtol=2e-8, atol=2e-10
    )
    np.testing.assert_allclose(
        [c.std_error for c in result.coefficients[:count]], np.sqrt(v.diagonal()), rtol=2e-8
    )
    assert result.inference["distribution"] == "normal"
    assert result.sample_positions == list(range(1, len(df)))
    assert ResultBundle.model_validate_json(result.model_dump_json()).extra == result.extra
    assert "tabular" in result.to_latex()


@pytest.mark.parametrize("trend", ["n", "c", "ct", "ctt"])
@pytest.mark.parametrize("kernel", ["bartlett", "parzen", "quadratic_spectral"])
def test_dols_hac_augmentation_and_df(trend, kernel):
    oracle = pytest.importorskip("arch.unitroot.cointegration")
    df = synthetic()
    result = oe.dols(
        data=df,
        y="y",
        x=["x", "z"],
        leads=2,
        lags=3,
        bandwidth=6,
        kernel=kernel,
        trend=trend,
        covariance="hac",
        df_adjust=True,
    )
    ref = oracle.DynamicOLS(df.y, df[["x", "z"]], trend=trend, leads=2, lags=3).fit(
        cov_type="robust", kernel=kernel.replace("_", ""), bandwidth=6, df_adjust=True
    )
    change = np.eye(len(ref.full_params))
    if trend in ("ct", "ctt"):
        change[2, 3] = -4
        if trend == "ctt":
            change[2, 4], change[3, 4] = 16, -8
    np.testing.assert_allclose(
        [c.estimate for c in result.coefficients], change @ ref.full_params, atol=2e-9
    )
    np.testing.assert_allclose(
        result.covariance_matrix, change @ ref.full_cov @ change.T, rtol=2e-8, atol=2e-10
    )
    assert result.sample_positions == list(range(4, len(df) - 2))


@pytest.mark.parametrize("selection", ["aic", "bic", "hqic"])
def test_dols_ic_common_sample(selection):
    oracle = pytest.importorskip("arch.unitroot.cointegration")
    df = synthetic()
    result = oe.dols(data=df, y="y", x=["x", "z"], selection=selection, max_leads=2, max_lags=3)
    ref = oracle.DynamicOLS(df.y, df[["x", "z"]], max_lead=2, max_lag=3, method=selection).fit(
        bandwidth=4
    )
    state = result.extra["units"][0]
    assert (state["leads"], state["lags"]) == (ref.leads, ref.lags)
    np.testing.assert_allclose([c.estimate for c in result.coefficients[:3]], ref.params, atol=2e-9)


@pytest.mark.parametrize("method", ["fmols", "dols"])
def test_panel_estimands_unbalanced_samples_and_covariances(method):
    dfs = [synthetic(i, 100 + 10 * i).assign(unit=i) for i in range(5)]
    df = pd.concat(dfs, ignore_index=True)
    unit_results = [getattr(oe, method)(data=g, y="y", x=["x", "z"], time="t") for g in dfs]
    beta = np.array([[c.estimate for c in r.coefficients[:2]] for r in unit_results])
    mean = getattr(oe, "panel_" + method)(
        data=df, y="y", x=["x", "z"], time="t", panel="unit", pooling="mean_group"
    )
    np.testing.assert_allclose([c.estimate for c in mean.coefficients], beta.mean(0), atol=2e-10)
    np.testing.assert_allclose(mean.covariance_matrix, np.cov(beta.T, ddof=1) / 5, atol=2e-10)
    assert mean.inference["df_inference"] == 4
    pooled = getattr(oe, "panel_" + method)(data=df, y="y", x=["x", "z"], time="t", panel="unit")
    assert pooled.nobs == len(df) - 5
    assert len(pooled.extra["units"]) == 5
    if method == "dols":
        # Independent block-design OLS, including unit-specific differences.
        ys, zs = [], []
        for i, g in enumerate(dfs):
            xx = g[["x", "z"]].to_numpy()
            z = np.zeros((len(g) - 1, 2 + 5 * 3))
            z[:, :2] = xx[1:]
            z[:, 2 + 3 * i : 5 + 3 * i] = np.column_stack(
                [np.ones(len(g) - 1), np.diff(xx, axis=0)]
            )
            ys.append(g.y.to_numpy()[1:])
            zs.append(z)
        ref = np.linalg.lstsq(np.vstack(zs), np.concatenate(ys), rcond=None)[0]
        np.testing.assert_allclose([c.estimate for c in pooled.coefficients], ref, atol=2e-9)


@pytest.mark.parametrize(
    "bad,code",
    [
        ("gap", "time_gaps"),
        ("missing", "time_gaps"),
        ("rank", "singular_design"),
        ("dates", "invalid_time"),
    ],
)
def test_domain_errors_do_not_compress_or_impute(bad, code):
    df = synthetic()
    if bad == "gap":
        df = df.drop(index=10)
    if bad == "missing":
        df.loc[10, "x"] = np.nan
    if bad == "rank":
        df.z = df.x
    if bad == "dates":
        df.t = pd.date_range("2000-01-01", periods=len(df))
    with pytest.raises(AnalysisError) as exc:
        oe.fmols(data=df, y="y", x=["x", "z"], time="t", missing="drop")
    assert exc.value.code == code


def test_published_fred_crude_replication():
    oracle = pytest.importorskip("arch.unitroot.cointegration")
    df = pd.read_csv(Path(__file__).parent / "fixtures/longrun-crude.csv")
    for name, cls in [
        ("fmols", oracle.FullyModifiedOLS),
        ("ccr", oracle.CanonicalCointegratingReg),
        ("dols", oracle.DynamicOLS),
    ]:
        opts = {"leads": 1, "lags": 1} if name == "dols" else {}
        result = getattr(oe, name)(data=df, y="WTI", x=["Brent"], bandwidth=6, **opts)
        ref = cls(df.WTI, df[["Brent"]], **opts).fit(bandwidth=6)
        np.testing.assert_allclose(
            [c.estimate for c in result.coefficients[:2]], ref.params, rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(
            np.array(result.covariance_matrix)[:2, :2], ref.cov, rtol=1e-8, atol=1e-9
        )


@pytest.mark.parametrize(
    "method,covariance", [("fmols", "nonrobust"), ("dols", "nonrobust"), ("dols", "hac")]
)
def test_panel_pooled_full_covariance_independent_block_algebra(method, covariance):
    frames = [synthetic(i + 30, 90 + 5 * i).assign(unit=i) for i in range(4)]
    width = 2 + 4 * (3 if method == "dols" else 1)
    designs, targets, variances, biases = [], [], [], []
    for i, g in enumerate(frames):
        x, y = g[["x", "z"]].to_numpy(), g.y.to_numpy()
        n = len(y) - 1
        z = np.zeros((n, width))
        z[:, :2] = x[1:]
        if method == "dols":
            z[:, 2 + 3 * i : 5 + 3 * i] = np.column_stack([np.ones(n), np.diff(x, axis=0)])
            unit = np.column_stack([x[1:], np.ones(n), np.diff(x, axis=0)])
            residual = y[1:] - unit @ np.linalg.lstsq(unit, y[1:], rcond=None)[0]
            innovations = residual[:, None]
            target = y[1:]
            bias = np.zeros(2)
        else:
            unit = np.column_stack([x, np.ones(len(y))])
            residual = y - unit @ np.linalg.lstsq(unit, y, rcond=None)[0]
            innovations = np.column_stack([residual[1:], np.diff(x - x.mean(0), axis=0)])
            z[:, 2 + i] = 1
        short = innovations.T @ innovations / n
        delta = short.copy()
        for lag in range(1, 5):
            delta += (1 - lag / 5) * (innovations[lag:].T @ innovations[:-lag]) / n
        omega = delta + delta.T - short
        if method == "fmols":
            correction = omega[0, 1:] @ np.linalg.inv(omega[1:, 1:])
            target = y[1:] - innovations[:, 1:] @ correction
            bias = delta[0, 1:] - correction @ delta[1:, 1:]
            variance = omega[0, 0] - correction @ omega[1:, 0]
        else:
            variance = omega[0, 0]
        designs.append(z)
        targets.append(target)
        biases.append(n * bias)
        variances.append(variance)
    z = np.vstack(designs)
    target = np.concatenate(targets)
    bread = np.linalg.inv(z.T @ z)
    fullbias = np.zeros(width)
    fullbias[:2] = np.sum(biases, axis=0)
    beta = bread @ (z.T @ target - fullbias)
    meat = np.zeros((width, width))
    for zi, yi, variance in zip(designs, targets, variances, strict=True):
        if covariance == "hac":
            score = zi * (yi - zi @ beta)[:, None]
            mi = score.T @ score
            for lag in range(1, 5):
                gamma = score[lag:].T @ score[:-lag]
                mi += (1 - lag / 5) * (gamma + gamma.T)
        else:
            mi = variance * (zi.T @ zi)
        meat += mi
    result = getattr(oe, "panel_" + method)(
        data=pd.concat(frames, ignore_index=True),
        y="y",
        x=["x", "z"],
        panel="unit",
        time="t",
        **({"covariance": covariance} if method == "dols" else {}),
    )
    np.testing.assert_allclose(
        [c.estimate for c in result.coefficients], beta, rtol=2e-8, atol=2e-9
    )
    np.testing.assert_allclose(result.covariance_matrix, bread @ meat @ bread, rtol=2e-7, atol=2e-9)
