"""Deterministic real fits for the MARKET-110 publication regression matrix.

Development evidence only. Formatting/readback tests use the saved results and
never need the source data, an estimator, or an independent runtime dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import openecon as oe


def fitted_models():
    rng = np.random.default_rng(110)
    n, groups, size = 360, 36, 10
    g = np.repeat(np.arange(groups), size)
    t = np.tile(np.arange(size), groups)
    x, z, v, e = rng.normal(size=(4, n))
    u = rng.normal(size=groups)[g]
    p = 0.8 * z + 0.3 * x + v
    y = 1 + 0.7 * x + 0.5 * p + 0.7 * u + 0.5 * v + e
    d = pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "p": p,
            "id": g,
            "t": t,
            "region": g // 6,
            "fw": rng.integers(1, 4, n),
        }
    )
    d["binary"] = rng.binomial(1, 1 / (1 + np.exp(-(0.3 + 0.5 * x))))
    d["category"] = rng.integers(0, 3, n)
    excess = rng.random(n) < 1 / (1 + np.exp(-(-0.8 + 0.4 * z)))
    mu = np.exp(0.5 + 0.3 * x)
    d["count"] = np.where(excess, 0, rng.negative_binomial(2, 2 / (2 + mu)))
    d["censored"] = np.maximum(0, 0.3 + 0.6 * x + e)
    duration = rng.weibull(1.5, n) * np.exp(-0.3 * x)
    censor = rng.exponential(2, n)
    d["duration"] = np.minimum(duration, censor) + 0.01
    d["failure"] = (duration <= censor).astype(int)
    d["treat"] = ((g < groups // 2) & (t >= 5)).astype(int)
    d["did_y"] = 1 + x + u + 0.2 * t + 0.8 * d.treat + e
    d["y2"] = -0.5 + 0.4 * z + 0.6 * e + rng.normal(size=n)
    missing = d.copy()
    missing.loc[[2, 19], "x"] = np.nan
    yield "ols", oe.ols(data=missing, y="y", x=["x"], covariance="HC3", missing="drop")
    yield "logit", oe.logit(data=d, y="binary", x=["x"], cluster="id")
    yield "areg", oe.areg(data=d, y="y", x=["x"], absorb="id")
    yield "xtreg_robust", oe.xtreg(data=d, y="y", x=["x"], panel="id", covariance="robust")
    yield (
        "xtreg_dk",
        oe.xtreg(data=d, y="y", x=["x"], panel="id", time="t", covariance="driscoll_kraay", lags=1),
    )
    yield (
        "ivregress",
        oe.ivregress(
            data=d,
            y="y",
            x=["x"],
            endog=["p"],
            instruments=["z"],
            cluster=["id", "region"],
            small=True,
        ),
    )
    yield "glm", oe.glm(data=d, y="count", x=["x"], family="poisson", covariance="robust")
    yield "mlogit", oe.mlogit(data=d, y="category", x=["x"])
    yield "ologit", oe.ologit(data=d, y="category", x=["x"])
    yield "tobit", oe.tobit(data=d, y="censored", x=["x"], ll=0)
    yield (
        "zinb",
        oe.zinb(
            data=d,
            y="count",
            x=["x"],
            inflate=["z"],
            cluster=["id", "region"],
            weights="fw",
            weight_type="fweight",
        ),
    )
    yield "qreg", oe.qreg(data=d, y="y", x=["x"])
    series = pd.DataFrame(
        {
            "time": np.arange(180),
            "x": rng.normal(size=180),
            "y": rng.normal(size=180),
            "y2": rng.normal(size=180),
        }
    )
    for i in range(1, len(series)):
        series.loc[i, "y"] += 0.4 * series.loc[i - 1, "y"]
    yield "arima", oe.arima(data=series, y="y", order=(1, 0, 0), time="time", covariance="opg")
    volatility = series.copy()
    noise = rng.normal(size=180)
    for i in range(1, 180):
        volatility.loc[i, "y"] = noise[i] * np.sqrt(0.5 + 0.5 * volatility.loc[i - 1, "y"] ** 2)
    yield "arch", oe.arch(data=volatility, y="y", time="time", arch=1, covariance="robust")
    yield "var", oe.var(data=series, y=["y", "y2"], time="time", lags=1, small=True)
    yield "nardl", oe.nardl(data=series, y="y", x=["x"], time="time", lags=[1, 1])
    yield (
        "streg",
        oe.streg(
            data=d,
            time="duration",
            failure="failure",
            x=["x"],
            distribution="weibull",
            ancillary=["z"],
            cluster="id",
        ),
    )
    yield (
        "stcox",
        oe.stcox(data=d, time="duration", failure="failure", x=["x"], covariance="robust"),
    )
    yield "ahreg", oe.ahreg(data=d, y="y", x=["x"], panel="id", time="t", covariance="robust")
    yield (
        "didregress",
        oe.didregress(data=d, y="did_y", treatment="treat", group="id", time="t", x=["x"]),
    )
    yield "mixed", oe.mixed(data=d, y="y", x=["x"], group="id")
    yield (
        "sureg",
        oe.sureg(data=d, equations=[{"y": "y", "x": ["x"]}, {"y": "y2", "x": ["z"]}], small=True),
    )
    yield "ridge", oe.ridge(data=d, y="y", x=["x", "z"], selection="fixed", penalty=0.2)
    yield "mmreg", oe.mmreg(data=d, y="y", x=["x", "z"], starts=100, seed=110)
    spatial = d.iloc[:40].copy()
    spatial["unit"] = list(range(40))
    weights = oe.spatial_weights(
        list(range(40)), [(i, j, 1.0) for i in range(40) for j in ((i - 1) % 40, (i + 1) % 40)]
    )
    yield "sar", oe.sar(spatial, "y", ["x", "z"], key="unit", spatial_weights=weights)
    yield "ets", oe.ets(data=series, y="y", fixed={"alpha": 0.3}, initial=[0.0])
    h = np.ones(2)
    q = np.array([[1.0, 0.35], [0.35, 1.0]])
    target = q.copy()
    values = []
    vrng = np.random.default_rng(97)
    for _ in range(200):
        correlation = q / np.sqrt(np.outer(np.diag(q), np.diag(q)))
        shock = vrng.multivariate_normal([0.0, 0.0], correlation) * np.sqrt(h)
        values.append(shock)
        standardized = shock / np.sqrt(h)
        q = 0.1 * target + 0.2 * np.outer(standardized, standardized) + 0.7 * q
        h = 0.1 + 0.25 * shock**2 + 0.65 * h
    yield (
        "mgarch_ccc",
        oe.mgarch_ccc(
            pd.DataFrame(values, columns=["a", "b"]), ["a", "b"], intercept=False, tolerance=1e-6
        ),
    )
    # Keep the full family-inventory assertion meaningful as native families grow.
    # Append fixtures so earlier deterministic examples retain their exact draws.
    integrated = series.copy()
    integrated["x"] = series.x.cumsum()
    integrated["y"] = 1.3 * integrated.x + series.y
    yield "dols", oe.dols(data=integrated, y="y", x=["x"], time="time", leads=1, lags=1)
    # Six longer panels keep this all-coefficients preview fixture compact.
    # Long/truncated previews have a separate explicit 90-coefficient case.
    dfe_data = d.assign(id=np.repeat(np.arange(6), 60), t=np.tile(np.arange(60), 6))
    yield "dfe", oe.dfe(data=dfe_data, y="y", x=["x"], panel="id", time="t")
    yield "lp", oe.lp(data=series, y="y", x=["x"], time="time", horizons=2, lags=1)
    yield "mediation", oe.mediation(data=d, y="y", x=["x"], mediators=["p"], controls=["z"])
    yield "arfima", oe.arfima(series, "y", d=0.2, terms=32, time="time")
    yield "bspline_regress", oe.bspline_regress(data=d, y="y", x=["x"], knots={"x": [-1., 0., 1.]})
    yield "dmlirm", oe.dmlirm(data=d, y="y", treatment="binary", x=["x", "z"])
    yield "cfregress", oe.cfregress(data=d, y="y", endogenous="p", instruments=["z"], x=["x"])
