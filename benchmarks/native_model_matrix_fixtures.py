"""Development-only seeded physical fixtures for the original 80 Dataset routes.

Adapted from the withdrawn 7 October parallel draft, reviewed and exercised in
this checkout. Test fixture generators are deliberately confined to this writer. The measured
worker and the frozen runtime need neither pytest nor statistical reference
libraries. Every receipt declares one geometry, not all supported options.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ROSTER = """ols logit probit areg reghdfe ppmlhdfe cnsreg sureg mvreg reg3 gmm
xtreg xtfmb xtgee xtgls xtpcse ahreg xtdpd teffects var vec ardl nardl prais arima
arch ucm mswitch threshold rdrobust didregress eventstudy csdid clogit mixed stcox
nl rreg ivregress xtivreg ivreghdfe melogit meprobit mepoisson xtlogit xtprobit
xtpoisson qreg bsqreg sqreg iqreg glm poisson cloglog fracreg nbreg betareg
hetprobit biprobit ologit oprobit mlogit tobit intreg truncreg heckman heckprobit
ivprobit ivtobit cpoisson cnbreg tpoisson tnbreg zip zinb gnbreg hurdle churdle
frontier streg""".split()


def module(name):
    if str(ROOT / "tests") not in sys.path:
        sys.path.insert(0, str(ROOT / "tests"))
    return importlib.import_module("test_" + name)


def fixture(name, n):
    import numpy as np
    import pandas as pd
    from openecon.models import ModelSpec
    from openecon.econometrics.streaming_registry import _LIKELIHOODS

    def spec(**kw):
        return ModelSpec(estimator=name, outcome=kw.pop("outcome", "y"),
                         predictors=kw.pop("predictors", ["x", "z"]), **kw)

    rng = np.random.default_rng(20261007)
    x, z, v, e = rng.normal(size=(4, n))
    f = pd.DataFrame({"x": x, "z": z, "v": v, "y": .7 + .5*x - .3*z + e,
                      "g": np.arange(n) % 97, "t": np.arange(n),
                      "w": rng.integers(1, 4, n), "cat": np.arange(n) % 3})
    if name in _LIKELIHOODS and name not in ("streg", "frontier"):
        frame, specification = module("econ_streaming_likelihood").fixture(name, n=n, mixed=True)
        if name in {"tpoisson", "tnbreg"}:
            # Genuine zero-truncated draws. Clipping zeros to one can produce
            # a logarithmic-series boundary rather than an interior NB fit.
            mu = np.exp(.4+.3*frame.x.fillna(0)+.12*frame.cat.astype(float))
            counts = np.zeros(n, dtype=float)
            pending = np.ones(n, dtype=bool)
            while pending.any():
                draws = (rng.poisson(mu[pending]) if name == "tpoisson" else
                         rng.negative_binomial(1.4, 1.4/(1.4+mu[pending])))
                counts[pending] = draws
                pending = counts == 0
            frame.y = counts
        return frame, specification
    if name in ("ols", "logit", "probit"):
        if name != "ols":
            f.y = (rng.random(n) < 1/(1+np.exp(-(.3 + .5*x - .2*z)))).astype(float)
        f.loc[5, "x"] = np.nan
        f.loc[11, "w"] = 0
        return f, spec(predictors=["x", "z", "cat"], categorical=["cat"],
                       covariance="cluster", cluster="g", missing="drop",
                       weights="w" if name == "ols" else None,
                       weight_type="fweight" if name == "ols" else None)
    if name in ("areg", "cnsreg"):
        m = module("econ_streaming_linear")
        return m.frame(n=n), m.spec_for(name, covariance="robust")
    if name == "reghdfe":
        m = module("econ_streaming_hdfe")
        return m.data(n=n), m.spec(covariance="robust")
    if name == "ppmlhdfe":
        m = module("econ_streaming_ppml")
        return m.fixture(n=n), m.spec()
    if name in ("sureg", "mvreg", "reg3"):
        f["y1"], f["y2"] = 1+.6*x+.3*v+e, 2-.2*z+.5*x+.3*e+rng.normal(size=n)
        equations = [{"y": "y1", "x": ["x", "v"]}, {"y": "y2", "x": ["z", "x"]}]
        columns = {"outcomes": ["y1", "y2"]} if name == "mvreg" else {"system": ["y1", "y2", "x", "z", "v"]}
        return f, spec(outcome="y1", predictors=["x", "z"] if name == "mvreg" else [],
                       columns=columns, options={"corr": True} if name == "mvreg" else {"equations": equations})
    if name == "gmm":
        f["endog"] = .8*z+.4*e+v
        f.y = 1+.7*x-.9*f.endog+e
        return f, spec(predictors=[], columns={"system": ["y", "x", "endog", "z", "v"]},
                       covariance="robust", options={"moments": ["y-{b0}-{b1}*x-{b2}*endog"], "instruments": ["x", "z", "v"]})
    if name == "xtreg":
        m = module("econ_streaming_re")
        return m.data(groups=max(15, (n+7)//8), periods=8), m.spec(covariance="robust")
    if name == "xtfmb":
        m = module("econ_streaming_fmb")
        return m.fmb_data(n=max(20, (n+9)//10), periods=10), m.fmb_spec(covariance="hac", lags=2)
    if name == "xtgee":
        m = module("econ_streaming_gee")
        # Fixed group count avoids falsely treating a giant ordered group as small.
        return m.data(n=n), m.spec(covariance="robust")
    if name in ("xtgls", "xtpcse"):
        m = module("econ_streaming_panelgls")
        return m.data(periods=max(20, (n+5)//6), groups=6), m.spec(name)
    if name in ("ahreg", "xtdpd"):
        m = module("econ_streaming_dpanel" if name == "ahreg" else "econ_streaming_dynamic_gmm")
        return m.fixture(groups=max(20, (n+7)//8), periods=8), m.spec()
    if name == "teffects":
        m = module("econ_streaming_teffects")
        frame = m.make_data(n=n)
        frame["cat"] = pd.Categorical(np.arange(n)%3, categories=[0, 1, 2, 3])
        frame.loc[7, "x1"] = np.nan
        frame.loc[14, "f"] = 0
        return frame, m.specification(mixed=True)
    if name in ("didregress", "eventstudy"):
        m = module("econ_streaming_did")
        return m.make_panel(groups=max(36, (n+7)//8), periods=8, staggered=name == "eventstudy"), m.spec(name)
    if name == "csdid":
        m = module("econ_streaming_csdid")
        return m.fixture(n=max(40, (n+4)//5), periods=5), m.spec()
    if name == "clogit":
        m = module("econ_streaming_conditional")
        return m.fixture(groups=max(20, (n+4)//5), size=5), m.spec()
    if name == "mixed":
        m = module("econ_streaming_mixed")
        return m.data(n=n), m.spec(covariance="robust")
    if name in ("melogit", "meprobit", "mepoisson", "xtlogit", "xtprobit", "xtpoisson"):
        m = module("econ_streaming_glmm")
        family = "poisson" if "poisson" in name else "probit" if "probit" in name else "logit"
        return m.data(family, n_groups=max(12, (n+7)//8), size=8), m.spec(name)
    if name in ("qreg", "bsqreg", "sqreg", "iqreg"):
        m = module("econ_streaming_quantile")
        return m.data(n=n), m.spec(name)
    if name == "stcox":
        m = module("econ_streaming_cox")
        return m.data(n=n), m.spec(kind="robust")
    if name == "streg":
        return module("econ_survival_streg").make_data(n=n), spec(outcome="t", columns={"failure": "d", "entry": "t0"}, options={"distribution": "weibull"}, covariance="robust")
    if name == "frontier":
        return module("econ_systems_frontier").make_frontier(n=n), spec(outcome="yh", predictors=["x1", "x2"], options={"distribution": "hnormal"})
    if name == "nl":
        m = module("econ_streaming_nonlinear")
        return m.nonlinear_data(n=n), m.nl_spec("HC3")
    if name == "rreg":
        m = module("econ_streaming_nonlinear")
        return m.robust_data(n=n), spec(predictors=["x1", "x2"])
    if name in ("xtivreg", "ivreghdfe"):
        m = module("econ_streaming_fe_iv")
        return m.data(n=n), m.spec(name, covariance="robust")
    if name == "ivregress":
        f["endog"] = .4*x+.8*z-.3*v+rng.normal(size=n)
        f.y = .7+.3*x+1.4*f.endog+e
        return f, spec(predictors=["x"], columns={"endogenous": ["endog"], "instruments": ["z", "v"]}, covariance="robust", options={"method": "2sls"})
    if name in ("ucm", "mswitch", "threshold", "rdrobust"):
        m = module("econ_streaming_" + ("rd" if name == "rdrobust" else name))
        if name == "rdrobust":
            return m.fixture(n=n), m.spec()
        return m.data(n=n), m.specification() if name == "mswitch" else m.spec()
    if name == "arch":
        f = module("econ_arch_oracle").simulate(n=n)
        return f, spec(predictors=["x"], time="t", options={"model": "garch"}, covariance="robust")
    if name in ("var", "vec"):
        if name == "var":
            a = np.zeros((n, 2))
            noise = rng.normal(size=(n, 2))
            for j in range(1, n):
                a[j] = a[j-1] @ np.array([[.5, .1], [-.2, .4]]) + noise[j]
        else:
            walk = rng.normal(size=n).cumsum()
            a = np.column_stack([walk+rng.normal(size=n), .7*walk+rng.normal(size=n)])
        f["a"], f["b"] = a.T
        return f, spec(outcome="a", predictors=["x"] if name == "var" else [], time="t", columns={"system": ["a", "b"]}, options={"lags": 2, **({"rank": 1} if name == "vec" else {})})
    if name in ("ardl", "nardl"):
        f.x = rng.normal(size=n).cumsum()
        y = np.zeros(n)
        for j in range(2, n):
            y[j] = .5*y[j-1]-.1*y[j-2]+.25*f.x.iloc[j]+.3*z[j]+e[j]
        f.y = y
        return f, spec(predictors=["x"], time="t", columns={"exog": ["z"]}, options={"lags": [2, 1]}, covariance="HC3")
    if name in ("prais", "arima"):
        for j in range(1, n):
            e[j] += .39*e[j-1]
        f.y = 2.4+.8*x+e
        return f, spec(predictors=["x"], time="t", options={"order": [1, 0, 0]} if name == "arima" else {}, covariance="robust")
    raise ValueError(name)
