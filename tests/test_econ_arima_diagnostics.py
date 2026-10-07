"""Independent oracles for the series diagnostics of the arima family.

oe.corrgram, oe.wntestq, oe.jarque_bera and oe.archlm are compared with explicit
NumPy formulas (Box-Jenkins autocorrelations, Durbin-Levinson, Ljung-Box) and with
statsmodels (acf, pacf, acorr_ljungbox, jarque_bera, het_arch).
"""

import math

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.signal import lfilter
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera as sm_jarque_bera
from statsmodels.tsa.stattools import acf, pacf

import openecon as oe
from openecon.analysis import AnalysisError
from openecon.econometrics import registry
from openecon.frame import DataFrame


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(7)
    e = rng.standard_t(df=6, size=420)
    y = lfilter([1.0, 0.4], [1.0, -0.6, 0.2], e)[120:] + 3.0
    return pd.DataFrame({"y": y, "t": np.arange(1, len(y) + 1)})


def box_jenkins(y, lags):
    """r_k with divisor n, by the defining sums."""
    n, d = len(y), y - y.mean()
    return np.array([np.sum(d[k:] * d[:n - k]) / np.sum(d ** 2) for k in range(1, lags + 1)])


def durbin_levinson(r):
    """phi_kk from the recursion written out as in Box and Jenkins."""
    rho = np.r_[1.0, r]
    out, phi = [], np.zeros((len(r) + 1, len(r) + 1))
    for k in range(1, len(r) + 1):
        numerator = rho[k] - sum(phi[k - 1, j] * rho[k - j] for j in range(1, k))
        denominator = 1.0 - sum(phi[k - 1, j] * rho[j] for j in range(1, k))
        phi[k, k] = numerator / denominator
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        out.append(phi[k, k])
    return np.array(out)


def test_corrgram_matches_textbook_formulas_and_statsmodels(series):
    y, n, lags = series["y"].to_numpy(), len(series), 12
    table = oe.corrgram(series, "y", lags=lags)
    r = box_jenkins(y, lags)
    assert_allclose(table["acf"], r, rtol=1e-12)
    assert_allclose(table["acf"], acf(y, nlags=lags, adjusted=False, fft=False)[1:], rtol=1e-12)
    assert_allclose(table["pacf"], durbin_levinson(r), rtol=1e-10)
    assert_allclose(table["pacf"], pacf(y, nlags=lags, method="ldb")[1:], rtol=1e-10)
    q = n * (n + 2) * np.cumsum(r ** 2 / (n - np.arange(1, lags + 1)))
    assert_allclose(table["q"], q, rtol=1e-12)
    assert_allclose(table["p_value"], stats.chi2.sf(q, np.arange(1, lags + 1)), rtol=1e-9)
    reference = acorr_ljungbox(y, lags=lags)
    assert_allclose(table["q"], reference["lb_stat"], rtol=1e-10)
    assert_allclose(table["p_value"], reference["lb_pvalue"], rtol=1e-9)
    assert isinstance(table, DataFrame)
    assert list(table.columns) == ["lag", "acf", "pacf", "q", "p_value"]
    assert table["lag"].tolist() == list(range(1, lags + 1))
    assert table.attrs["nobs"] == n and table.attrs["lags"] == lags
    assert table.attrs["white_noise_band"] == pytest.approx(stats.norm.ppf(0.975) / math.sqrt(n))
    assert "Durbin-Levinson" in table.attrs["pacf_method"]
    assert "acf" in str(table.to_latex())


def test_corrgram_regression_pacf_default_lags_and_time_order(series):
    y, n = series["y"].to_numpy(), len(series)
    table = oe.corrgram(series, "y", lags=8, pacf="regression")
    # Stata's default: coefficient on the k-th lag in an OLS regression on a constant and k lags.
    expected = []
    for k in range(1, 9):
        design = np.column_stack([np.ones(n - k)] + [y[k - j:n - j] for j in range(1, k + 1)])
        expected.append(np.linalg.lstsq(design, y[k:], rcond=None)[0][-1])
    assert_allclose(table["pacf"], expected, rtol=1e-9)
    assert_allclose(table["pacf"], pacf(y, nlags=8, method="ols")[1:], rtol=1e-8)
    assert "OLS" in table.attrs["pacf_method"]
    default = oe.corrgram(series, "y")
    assert len(default) == min(n // 2 - 2, 40) == 40                     # Stata's default
    short = oe.corrgram(series.iloc[:30], "y")
    assert len(short) == 30 // 2 - 2
    shuffled = series.sample(frac=1.0, random_state=1)
    by_time = oe.corrgram(shuffled, "y", lags=8, time="t")
    assert_allclose(by_time["acf"], oe.corrgram(series, "y", lags=8)["acf"], rtol=1e-13)
    row_order = oe.corrgram(shuffled, "y", lags=8)
    assert abs(row_order["acf"].iloc[0] - by_time["acf"].iloc[0]) > 0.1
    dated = series.assign(date=pd.date_range("2000-01-01", periods=n, freq="D"))
    by_date = oe.corrgram(dated.sample(frac=1.0, random_state=2), "y", lags=8, time="date")
    assert_allclose(by_date["acf"], by_time["acf"], rtol=1e-13)
    as_mapping = oe.corrgram({"y": y.tolist()}, "y", lags=3)
    assert_allclose(as_mapping["acf"], by_time["acf"].iloc[:3], rtol=1e-13)


def test_wntestq_matches_ljung_box(series):
    y, n = series["y"].to_numpy(), len(series)
    table = oe.wntestq(series, "y", lags=10)
    reference = acorr_ljungbox(y, lags=[10])
    assert table.attrs["statistic"] == pytest.approx(reference["lb_stat"].iloc[0], rel=1e-10)
    assert table.attrs["p_value"] == pytest.approx(reference["lb_pvalue"].iloc[0], rel=1e-8)
    assert table.attrs["df"] == 10 and table.attrs["lags"] == 10 and table.attrs["nobs"] == n
    assert table.attrs["distribution"] == "chi2" and "white noise" in table.attrs["label"]
    assert list(table.columns) == ["lags", "statistic", "df", "p_value"] and len(table) == 1
    assert table["statistic"].iloc[0] == table.attrs["statistic"]
    default = oe.wntestq(series, "y")
    assert default.attrs["lags"] == 40                                    # min(n/2 - 2, 40)
    assert default.attrs["statistic"] == pytest.approx(
        acorr_ljungbox(y, lags=[40])["lb_stat"].iloc[0], rel=1e-10)
    white = pd.DataFrame({"e": np.random.default_rng(3).normal(size=500)})
    assert oe.wntestq(white, "e", lags=12).attrs["p_value"] > 0.05
    assert table.attrs["p_value"] < 1e-6                                  # the ARMA series is not
    # The residual test stored by oe.arima is the same statistic on the innovations.
    fit = oe.arima(data=series, y="y", order=(2, 0, 1), ljung_lags=10)
    assert fit.tests["ljung_box"]["df"] == 10 and fit.tests["ljung_box"]["df_adjusted"] == 7
    assert fit.tests["ljung_box"]["p_value_adjusted"] == pytest.approx(
        stats.chi2.sf(fit.tests["ljung_box"]["statistic"], 7), rel=1e-9)


def test_jarque_bera_matches_statsmodels_and_scipy(series):
    y, n = series["y"].to_numpy(), len(series)
    table = oe.jarque_bera(series, "y")
    statistic, p_value, skewness, kurtosis = sm_jarque_bera(y)
    assert table.attrs["statistic"] == pytest.approx(statistic, rel=1e-10)
    assert table.attrs["p_value"] == pytest.approx(p_value, rel=1e-8)
    assert table.attrs["skewness"] == pytest.approx(skewness, rel=1e-10)
    assert table.attrs["kurtosis"] == pytest.approx(kurtosis, rel=1e-10)
    assert table.attrs["statistic"] == pytest.approx(stats.jarque_bera(y).statistic, rel=1e-10)
    d = y - y.mean()                                                      # biased moments
    s, k = np.mean(d ** 3) / np.mean(d ** 2) ** 1.5, np.mean(d ** 4) / np.mean(d ** 2) ** 2
    assert table.attrs["statistic"] == pytest.approx(n / 6 * (s ** 2 + (k - 3) ** 2 / 4))
    assert table.attrs["df"] == 2 and table.attrs["nobs"] == n
    assert list(table.columns) == ["statistic", "df", "p_value", "skewness", "kurtosis"]
    normal = pd.DataFrame({"e": np.random.default_rng(5).normal(size=2000)})
    assert oe.jarque_bera(normal, "e").attrs["p_value"] > 0.05
    skewed = pd.DataFrame({"e": np.random.default_rng(6).exponential(size=300)})
    assert oe.jarque_bera(skewed, "e").attrs["p_value"] < 1e-6


def test_archlm_matches_het_arch(series):
    y = series["y"].to_numpy()
    for lags in (1, 4):
        table = oe.archlm(series, "y", lags=lags, demean=True)
        lm, lm_p, _, _ = het_arch(y - y.mean(), nlags=lags)
        assert table.attrs["statistic"] == pytest.approx(lm, rel=1e-9)
        assert table.attrs["p_value"] == pytest.approx(lm_p, rel=1e-8)
        assert table.attrs["df"] == lags and table.attrs["nobs"] == len(y) - lags
    raw = oe.archlm(series, "y", lags=2)                                  # the column as given
    assert raw.attrs["statistic"] == pytest.approx(het_arch(y, nlags=2)[0], rel=1e-9)
    assert raw.attrs["demeaned"] is False
    # Explicit auxiliary regression: T' R^2 of u_t^2 on a constant and its first lag.
    u2 = (y - y.mean()) ** 2
    design = np.column_stack([np.ones(len(u2) - 1), u2[:-1]])
    resid = u2[1:] - design @ np.linalg.lstsq(design, u2[1:], rcond=None)[0]
    r2 = 1 - resid @ resid / np.sum((u2[1:] - u2[1:].mean()) ** 2)
    single = oe.archlm(series, "y", demean=True)
    assert single.attrs["statistic"] == pytest.approx((len(u2) - 1) * r2, rel=1e-9)
    assert single.attrs["r_squared"] == pytest.approx(r2, rel=1e-9)
    several = oe.archlm(series, "y", lags=[1, 2, 4], demean=True)
    assert several["lags"].tolist() == [1, 2, 4] and several["df"].tolist() == [1, 2, 4]
    assert several["statistic"].iloc[2] == pytest.approx(het_arch(y - y.mean(), nlags=4)[0],
                                                         rel=1e-9)
    assert several.attrs["distribution"] == "chi2" and "statistic" not in several.attrs
    # A GARCH-like series is detected.
    rng = np.random.default_rng(9)
    e, h = np.zeros(1500), 1.0
    for t in range(1, 1500):
        h = 0.2 + 0.7 * e[t - 1] ** 2
        e[t] = math.sqrt(h) * rng.normal()
    assert oe.archlm(pd.DataFrame({"e": e}), "e", lags=1).attrs["p_value"] < 1e-6


def test_diagnostic_error_codes(series):
    holes = series.copy()
    holes.loc[10, "y"] = np.nan
    gaps = series.drop(index=[50])
    repeated = series.copy()
    repeated.loc[3, "t"] = repeated.loc[4, "t"]
    cases = [
        (lambda: oe.corrgram(series, "nope"), "missing_columns"),
        (lambda: oe.corrgram(series, ["y"]), "invalid_spec"),
        (lambda: oe.corrgram(holes, "y"), "missing_values"),
        (lambda: oe.corrgram(gaps, "y", time="t"), "time_gaps"),
        (lambda: oe.corrgram(repeated, "y", time="t"), "repeated_time_values"),
        (lambda: oe.corrgram(series.assign(t=series["t"] + 0.5), "y", time="t"), "invalid_time"),
        (lambda: oe.corrgram(series, "y", lags=0), "invalid_lags"),
        (lambda: oe.corrgram(series, "y", lags=len(series)), "invalid_lags"),
        (lambda: oe.corrgram(series, "y", lags=2.0), "invalid_lags"),
        (lambda: oe.corrgram(series, "y", pacf="burg"), "invalid_option"),
        (lambda: oe.corrgram(series.assign(y=1.0), "y"), "constant_series"),
        (lambda: oe.corrgram(series.iloc[:3], "y"), "insufficient_observations"),
        (lambda: oe.corrgram(series.iloc[:0], "y"), "empty_data"),
        (lambda: oe.corrgram(series.assign(y="a"), "y"), "non_numeric_column"),
        (lambda: oe.corrgram(pd.DataFrame({"y": [1.0, 2.0] * 20}), "y", lags=3,
                             pacf="regression"), "collinear_lags"),
        (lambda: oe.wntestq(series, "y", lags=-1), "invalid_lags"),
        (lambda: oe.wntestq(holes, "y"), "missing_values"),
        (lambda: oe.jarque_bera(series.assign(y=2.0), "y"), "constant_series"),
        (lambda: oe.jarque_bera(holes, "y"), "missing_values"),
        (lambda: oe.archlm(series, "y", lags=0), "invalid_lags"),
        (lambda: oe.archlm(series, "y", lags=[1, 1]), "invalid_lags"),
        (lambda: oe.archlm(series, "y", lags=[]), "invalid_lags"),
        (lambda: oe.archlm(series, "y", demean=1), "invalid_option"),
        (lambda: oe.archlm(series.iloc[:8], "y", lags=3), "insufficient_observations"),
        (lambda: oe.archlm(pd.DataFrame({"y": [1.0, -1.0] * 10}), "y"), "constant_series"),
        (lambda: oe.corrgram(3, "y"), "invalid_data"),
    ]
    for call, code in cases:
        with pytest.raises(AnalysisError) as error:
            call()
        assert error.value.code == code, code


def test_diagnostics_are_exported_and_fast():
    exports = registry.public_exports()
    for name in ("corrgram", "wntestq", "jarque_bera", "archlm"):
        assert exports[name][0] == "openecon.econometrics.arima.diagnostics"
        assert callable(getattr(oe, name)) and getattr(oe, name).__doc__
    big = pd.DataFrame({"y": np.random.default_rng(0).normal(size=1_000_000)})
    import time

    start = time.perf_counter()
    table = oe.corrgram(big, "y", lags=40)
    oe.wntestq(big, "y")
    oe.jarque_bera(big, "y")
    oe.archlm(big, "y", lags=5)
    assert time.perf_counter() - start < 10.0
    assert float(table["acf"].abs().max()) < 0.01
