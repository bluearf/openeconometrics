"""Reference values for the GARCH extensions from the Python ``arch`` package (Sheppard).

Development-only oracle: ``arch`` is NOT a runtime or test dependency of OpenEconometrics.
The JSON written here is committed under ``tests/fixtures/garch_extensions/`` and the tests
compare the native float64 Torch implementation with it.

What is recorded (every series is simulated here from a fixed NumPy seed and stored):

* ``skewt_density``: Hansen's skewed-t log density (``SkewStudent.loglikelihood`` with
  unit variance) at a grid of points and parameter pairs.
* ``skewt_likelihood``: per-observation log likelihoods of GARCH(1,1), GJR(1,1) and
  EGARCH(1,1) with skewed-t errors at fixed parameter values, computed from the public
  component API (``GARCH/EGARCH.compute_variance`` and ``SkewStudent.loglikelihood``)
  with the backcast set to the presample convention of ``oe.arch``: the mean squared
  residual at the given mean (``h0``), ``ln h0`` for EGARCH. ``arch`` primes the GJR
  threshold term with ``0.5 * backcast``, as ``oe.arch`` does.
* ``skewt_fits``: zero-mean maximum likelihood fits (``arch_model(..., mean="Zero")``,
  ``fit(backcast=mean(y^2), cov_type="classic")``). With a zero mean the presample value is
  fixed, so ``arch``'s likelihood is exactly the one ``oe.arch(constant=False)`` maximizes;
  estimates and "classic" (inverse numerical Hessian) standard errors differ only by the
  optimizers (SLSQP versus BFGS + Newton) and the numerical Hessians.
* ``figarch_likelihood`` / ``figarch_fits``: FIGARCH(1,d,1) with truncation 1000 (``arch``'s
  default) in the same way: ``FIGARCH.compute_variance`` with ``backcast = h0``.

Run with an environment that has numpy, scipy and arch 8.0:

    python scripts/generate_garch_extension_oracles.py

Provenance (versions, seeds, date) is written into the JSON.
"""

from __future__ import annotations

import datetime
import json
import math
import sys
from pathlib import Path

import arch
import numpy as np
import scipy
from arch import arch_model
from arch.univariate import EGARCH, FIGARCH, GARCH, Normal, SkewStudent, ZeroMean

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "garch_extensions"
SEED = 161_2026


def _skewt_draws(rng: np.random.Generator, n: int, eta: float, lam: float) -> np.ndarray:
    """Hansen skewed-t draws by the quantile function of arch's SkewStudent."""
    return SkewStudent().ppf(rng.uniform(size=n), np.array([eta, lam]))


def simulate(rng: np.random.Generator, n: int, model: str, shocks: np.ndarray,
             gjr: tuple[float, float, float, float] = (0.05, 0.05, 0.08, 0.85)) -> np.ndarray:
    """A GARCH-type series with the given standardized shocks (burn-in 300 dropped)."""
    omega, alpha, gamma, beta = gjr
    total = n + 300
    e, h = np.zeros(total), np.ones(total)
    for t in range(1, total):
        if model == "egarch":
            z = e[t - 1] / math.sqrt(h[t - 1])
            h[t] = math.exp(0.01 + 0.15 * (abs(z) - math.sqrt(2 / math.pi)) - 0.08 * z
                            + 0.95 * math.log(h[t - 1]))
        else:
            h[t] = omega + alpha * e[t - 1] ** 2 + gamma * e[t - 1] ** 2 * (e[t - 1] < 0) \
                + beta * h[t - 1]
        e[t] = math.sqrt(h[t]) * shocks[t]
    return e[300:]


def skewt_density() -> list[dict]:
    rows = []
    grid = np.linspace(-6.0, 6.0, 49)
    dist = SkewStudent()
    for eta, lam in [(5.0, 0.3), (8.0, -0.4), (3.5, 0.75), (30.0, 0.05)]:
        values = dist.loglikelihood(np.array([eta, lam]), grid, np.ones_like(grid),
                                    individual=True)
        rows.append({"eta": eta, "lambda": lam, "z": grid.tolist(),
                     "log_density": np.asarray(values).tolist()})
    return rows


def _variance(model: str, params: np.ndarray, resids: np.ndarray, h0: float) -> np.ndarray:
    n = resids.shape[0]
    sigma2 = np.zeros(n)
    bounds = np.array([[1e-300, 1e300]] * n)
    if model == "egarch":
        vol = EGARCH(p=1, o=1, q=1)
        return vol.compute_variance(params, resids, sigma2, math.log(h0), bounds)
    vol = GARCH(p=1, o=int(model == "gjr"), q=1)
    return vol.compute_variance(params, resids, sigma2, h0, bounds)


def skewt_likelihood(rng: np.random.Generator) -> list[dict]:
    rows = []
    cases = [
        ("garch", 0.0, [0.04, 0.07, 0.88], (6.0, 0.25)),
        ("gjr", 0.15, [0.05, 0.03, 0.10, 0.86], (5.0, -0.3)),
        ("egarch", -0.05, [0.02, 0.18, -0.07, 0.94], (7.0, 0.4)),
    ]
    for model, mu, variance_params, (eta, lam) in cases:
        shocks = _skewt_draws(rng, 800, 6.0, -0.2)
        y = mu + 0.3 + simulate(rng, 500, model, shocks)
        resids = y - mu
        h0 = float(np.mean(resids ** 2))
        sigma2 = _variance(model, np.array(variance_params), resids, h0)
        loglik = SkewStudent().loglikelihood(np.array([eta, lam]), resids, sigma2,
                                             individual=True)
        rows.append({"model": model, "mu": mu, "variance_params": variance_params,
                     "arch_parameter_names": {"garch": ["omega", "alpha", "beta"],
                                              "gjr": ["omega", "alpha", "gamma", "beta"],
                                              "egarch": ["omega", "alpha(|z|)", "gamma(z)",
                                                         "beta"]}[model],
                     "eta": eta, "lambda": lam, "y": y.tolist(), "h0": h0,
                     "loglik": np.asarray(loglik).tolist(),
                     "sigma2_head": sigma2[:5].tolist(), "sigma2_last": float(sigma2[-1])})
    return rows


def skewt_fits(rng: np.random.Generator) -> list[dict]:
    rows = []
    for model in ("garch", "gjr", "egarch"):
        shocks = _skewt_draws(rng, 1300, 7.0, -0.35)
        y = simulate(rng, 1000, model, shocks, (0.05, 0.10, 0.10, 0.80))
        backcast = float(np.mean(y ** 2))
        vol = "EGARCH" if model == "egarch" else "GARCH"
        o = 1 if model in {"gjr", "egarch"} else 0
        am = arch_model(y, mean="Zero", vol=vol, p=1, o=o, q=1, dist="skewt", rescale=False)
        res = am.fit(disp="off", backcast=backcast, cov_type="classic",
                     options={"ftol": 1e-14, "maxiter": 2000})
        rows.append({"model": model, "y": y.tolist(), "backcast": backcast,
                     "param_names": list(res.params.index), "params": res.params.tolist(),
                     "std_errors": res.std_err.tolist(), "loglik": float(res.loglikelihood),
                     "converged": int(res.convergence_flag) == 0,
                     "cov_type": "classic (inverse numerical Hessian of the log likelihood)"})
    return rows


def figarch_likelihood(rng: np.random.Generator) -> list[dict]:
    rows = []
    for mu, params, p, q in [(0.1, [0.06, 0.2, 0.45, 0.5], 1, 1),
                             (0.0, [0.05, 0.4, 0.3], 0, 1),
                             (-0.05, [0.08, 0.15, 0.6], 1, 0)]:
        shocks = rng.standard_normal(800)
        y = mu + 0.2 + simulate(rng, 500, "garch", shocks)
        resids = y - mu
        h0 = float(np.mean(resids ** 2))
        n = resids.shape[0]
        sigma2 = FIGARCH(p=p, q=q, truncation=1000).compute_variance(
            np.array(params), resids, np.zeros(n), h0, np.array([[1e-300, 1e300]] * n))
        loglik = -0.5 * (math.log(2 * math.pi) + np.log(sigma2) + resids ** 2 / sigma2)
        rows.append({"p": p, "q": q, "mu": mu, "params": params,
                     "arch_parameter_names": ["omega", *(["phi"] if p else []), "d",
                                              *(["beta"] if q else [])],
                     "truncation": 1000, "y": y.tolist(), "h0": h0, "loglik": loglik.tolist(),
                     "sigma2_head": sigma2[:5].tolist(), "sigma2_last": float(sigma2[-1])})
    return rows


def figarch_fits(rng: np.random.Generator) -> list[dict]:
    rows = []
    for number in range(2):
        if number == 0:
            # A long-memory series: FIGARCH(1, 0.4, 1) simulated by arch's own recursion.
            sim = ZeroMean(None, volatility=FIGARCH(p=1, q=1),
                           distribution=Normal(seed=np.random.default_rng(SEED + 1)))
            data = sim.simulate([0.1, 0.2, 0.4, 0.5], 1500, burn=1000)["data"].to_numpy()
        else:
            data = simulate(rng, 1500, "garch", rng.standard_normal(1800))
        y = np.asarray(data, dtype=float)
        backcast = float(np.mean(y ** 2))
        am = arch_model(y, mean="Zero", vol="FIGARCH", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", backcast=backcast, cov_type="classic",
                     options={"ftol": 1e-14, "maxiter": 2000})
        rows.append({"y": y.tolist(), "backcast": backcast,
                     "param_names": list(res.params.index), "params": res.params.tolist(),
                     "std_errors": res.std_err.tolist(), "loglik": float(res.loglikelihood),
                     "converged": int(res.convergence_flag) == 0, "truncation": 1000})
    return rows


def _write(name: str, sections: dict) -> None:
    payload = {
        "provenance": {
            "generator": "scripts/generate_garch_extension_oracles.py",
            "date": datetime.date.today().isoformat(),
            "seed": SEED,
            "python": sys.version.split()[0],
            "arch": arch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "note": "development-only oracle; arch is not an OpenEconometrics dependency",
        },
        **sections,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=None, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


def main() -> None:
    rng = np.random.default_rng(SEED)
    _write("arch_skewt.json", {"skewt_density": skewt_density(),
                               "skewt_likelihood": skewt_likelihood(rng),
                               "skewt_fits": skewt_fits(rng)})
    _write("arch_figarch.json", {"figarch_likelihood": figarch_likelihood(rng),
                                 "figarch_fits": figarch_fits(rng)})


if __name__ == "__main__":
    main()
