"""Generate independent dev-only fixtures; never used by the runtime.

Run with arch==8.0.0, ivmodels==0.10.0, rdrobust==2.1.0, rddensity==3.0.
The statistical modules are loaded without optional plotting frontends.
"""

import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import types

import numpy as np
from scipy.stats import f
from arch.unitroot.cointegration import phillips_ouliaris
from ivmodels.tests.anderson_rubin import anderson_rubin_test
from ivmodels.tests.conditional_likelihood_ratio import conditional_likelihood_ratio_test
from ivmodels.models.kclass import KClass

for name in ("rdrobust", "rddensity"):
    spec = importlib.util.find_spec(name)
    module = types.ModuleType(name)
    module.__path__ = list(spec.submodule_search_locations)
    sys.modules[name] = module
rdrobust = importlib.import_module("rdrobust.rdrobust").rdrobust
rddensity = importlib.import_module("rddensity.rddensity").rddensity

output = {
    "versions": {
        name: importlib.metadata.version(name)
        for name in ("arch", "ivmodels", "rdrobust", "rddensity", "numpy", "scipy")
    },
    "sources": [
        "https://github.com/bashtage/arch",
        "https://github.com/ralf-pezold/ivmodels",
        "https://github.com/rdpackages/rdrobust",
        "https://github.com/rdpackages/rddensity",
    ],
}
r = np.random.default_rng(931)
n = 300
x = r.normal(size=(n, 2)).cumsum(0)
y = x @ np.array([0.7, -0.4]) + r.normal(size=n)
po = []
for trend in ("n", "c", "ct", "ctt"):
    for test in ("Za", "Zt"):
        for kernel in ("bartlett", "parzen", "quadratic_spectral"):
            fit = phillips_ouliaris(
                y, x, trend=trend, test_type=test, kernel=kernel, bandwidth=5, force_int=True
            )
            po.append(
                {
                    "trend": trend,
                    "test": test,
                    "kernel": kernel,
                    "statistic": float(fit.stat),
                    "p_value": float(fit.pvalue),
                    "critical": fit.critical_values.tolist(),
                }
            )
output["po"] = {
    "data": {"y": y.tolist(), "x0": x[:, 0].tolist(), "x1": x[:, 1].tolist()},
    "cases": po,
}
output["po"]["uncoupled"] = []
for dimensions in range(1, 6):
    r = np.random.default_rng(445 + dimensions)
    n = 350
    x = r.normal(size=(n, dimensions)).cumsum(0)
    y = r.normal(size=n).cumsum()
    cohort = {
        "data": {"y": y.tolist(), **{f"x{i}": x[:, i].tolist() for i in range(dimensions)}},
        "dimensions": dimensions,
        "cases": [],
    }
    for trend in ("n", "c", "ct", "ctt"):
        for test in ("Za", "Zt"):
            for kernel in ("bartlett", "parzen", "quadratic_spectral"):
                fit = phillips_ouliaris(
                    y, x, trend=trend, test_type=test, kernel=kernel, bandwidth=3, force_int=True
                )
                cohort["cases"].append(
                    {
                        "trend": trend,
                        "test": test,
                        "kernel": kernel,
                        "statistic": float(fit.stat),
                        "p_value": float(fit.pvalue),
                        "critical": fit.critical_values.tolist(),
                    }
                )
    output["po"]["uncoupled"].append(cohort)
weak = []
for seed, strength in ((841, 0.2), (991, 1.5), (937, 0.0)):
    r = np.random.default_rng(seed)
    n = 300
    z = r.normal(size=(n, 3))
    v = r.normal(size=n)
    d = strength * z[:, 0] + v
    y = 1.4 * d + 0.7 * v + r.normal(size=n)
    row = {
        "strength": strength,
        "data": {"y": y.tolist(), "d": d.tolist(), **{f"z{i}": z[:, i].tolist() for i in range(3)}},
    }
    row["ar"] = [
        float(t)
        for t in anderson_rubin_test(
            z, d[:, None], y, beta=np.array([1.4]), fit_intercept=True, critical_values="f"
        )
    ]
    row["clr"] = [
        float(t)
        for t in conditional_likelihood_ratio_test(
            z, d[:, None], y, beta=np.array([1.4]), fit_intercept=True
        )
    ]
    row["models"] = {}
    for method in ("2sls", "liml", "fuller(1)", 0.5):
        fit = KClass(kappa=method).fit(d[:, None], y, Z=z)
        row["models"][str(method)] = {
            "coefficients": [float(fit.intercept_), float(fit.coef_[0])],
            "kappa": float(fit.kappa_),
        }
    row["ar_critical"] = float(f.isf(0.05, 3, 296))
    weak.append(row)
output["weak"] = weak
r = np.random.default_rng(2514)
n = 1200
x = r.uniform(-1, 1, n)
z = r.normal(size=(n, 2))
w = r.uniform(0.5, 2, n)
t = (r.uniform(size=n) < (0.2 + 0.55 * (x >= 0))).astype(float)
y = 1 + 0.3 * x + 2 * (x >= 0) + 0.8 * z[:, 0] - 0.4 * z[:, 1] + r.normal(size=n)
data = {
    "x": x.tolist(),
    "y": y.tolist(),
    "t": t.tolist(),
    "z0": z[:, 0].tolist(),
    "z1": z[:, 1].tolist(),
    "w": w.tolist(),
}
cases = []
for fuzzy in (False, True):
    for auto in (False, True):
        for deriv in (0, 1):
            options = dict(
                covs=z, weights=w, deriv=deriv, p=deriv + 1, masspoints="adjust", stdvars=False
            )
            if not auto:
                options.update(h=[0.4, 0.5], b=[0.6, 0.7])
            if fuzzy:
                options["fuzzy"] = t
            fit = rdrobust(y, x, **options)
            cases.append(
                {
                    "fuzzy": fuzzy,
                    "auto": auto,
                    "deriv": deriv,
                    "coef": fit.coef.iloc[:, 0].tolist(),
                    "se": fit.se.iloc[:, 0].tolist(),
                    "h": fit.bws.iloc[0, :].tolist(),
                    "b": fit.bws.iloc[1, :].tolist(),
                    "N_h": [int(v) for v in fit.N_h],
                }
            )
density = rddensity(x, h=[0.4, 0.5], p=2, q=3, useall=True, bino_flag=False)
output["rd"] = {
    "data": data,
    "cases": cases,
    "density": {
        "hat": density.hat.to_dict(),
        "se": density.sd_jk.to_dict(),
        "stat": {"t_jk": float(density.test["t_jk"]), "p_jk": float(density.test["p_jk"])},
    },
}
mass = []
xm = np.round(x, 2)
for option in ("off", "check", "adjust"):
    fit = rdrobust(y, xm, weights=w, masspoints=option, stdvars=False)
    mass.append(
        {
            "masspoints": option,
            "coef": fit.coef.iloc[:, 0].tolist(),
            "se": fit.se.iloc[:, 0].tolist(),
            "h": fit.bws.iloc[0, :].tolist(),
            "b": fit.bws.iloc[1, :].tolist(),
            "N_h": [int(v) for v in fit.N_h],
        }
    )
output["rd"]["mass"] = mass
density_mass = rddensity(xm, h=[0.4, 0.5], p=2, q=3, useall=True, bino_flag=False)
output["rd"]["density_mass"] = {
    "hat": density_mass.hat.to_dict(),
    "se": density_mass.sd_jk.to_dict(),
    "stat": {"t_jk": float(density_mass.test["t_jk"]), "p_jk": float(density_mass.test["p_jk"])},
}
target = Path(sys.argv[1])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(output, allow_nan=False, separators=(",", ":")) + "\n")
print("Independent fixtures written:", target)
