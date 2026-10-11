#!/usr/bin/env python3
"""Seeded data and independent NumPy/SciPy oracles for shared frailty and recurrent events.

No OpenEconometrics code is imported. Three steps, run from the repository root:

    python scripts/generate_survival_frailty_oracles.py data
    Rscript --vanilla scripts/generate_survival_frailty_oracles.R
    python scripts/generate_survival_frailty_oracles.py numpy

``data`` writes the seeded synthetic datasets (``tests/fixtures/survival_frailty/
data_*.json``); the R script reads them and writes ``r_reference.json`` (R survival
``coxph`` with ``frailty()``/``cluster()``/``strata()``, R coxme); ``numpy`` writes
``numpy_reference.json``: parametric shared-frailty marginal likelihoods by explicit
numerical integration over the frailty density (``scipy.integrate.quad``) maximized
with SciPy, and the survival treatment-effect estimators as explicit stacked
estimating equations (GMM) with finite-difference Jacobians.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "survival_frailty"


def _round(values, digits=6):
    return [round(float(v), digits) for v in values]


def _columns(frame: dict[str, np.ndarray]) -> dict[str, list]:
    out = {}
    for name, values in frame.items():
        values = np.asarray(values)
        if values.dtype.kind in "iu":
            out[name] = [int(v) for v in values]
        else:
            out[name] = _round(values)
    return out


# ---- Cox shared frailty ---------------------------------------------------------------

def cox_gamma() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(16301)
    q = 40
    sizes = rng.integers(3, 8, q)
    g = np.repeat(np.arange(1, q + 1), sizes)
    n = g.size
    z = rng.gamma(1 / 0.6, 0.6, q)[g - 1]
    w = rng.normal(size=q)[g - 1]
    x1 = rng.normal(size=n)
    x2 = (rng.random(n) < 0.4).astype(int)
    rate = z * np.exp(0.6 * x1 - 0.5 * x2 + 0.3 * w)
    t = rng.exponential(1 / rate)
    c = np.minimum(rng.exponential(4.0, n), 3.0)
    time = np.maximum(np.round(np.minimum(t, c), 2), 0.01)
    return {"t": time, "d": (t <= c).astype(int), "x1": x1, "x2": x2, "w": w, "g": g}


def cox_entry() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(16302)
    q = 30
    sizes = rng.integers(5, 11, q)
    g = np.repeat(np.arange(1, q + 1), sizes)
    n = g.size
    z = np.exp(rng.normal(0, math.sqrt(0.4), q))[g - 1]
    s = rng.integers(0, 2, n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    rate = z * np.exp(0.5 * x1 + 0.25 * x2) * np.where(s == 1, 1.5, 1.0)
    t = rng.exponential(1 / rate)
    c = np.minimum(rng.exponential(3.0, n), 2.5)
    time = np.maximum(np.round(np.minimum(t, c), 2), 0.02)
    late = rng.random(n) < 0.3
    entry = np.where(late, np.floor(time * rng.uniform(0, 0.6, n) * 100) / 100, 0.0)
    return {"t": time, "d": (t <= c).astype(int), "t0": entry, "x1": x1, "x2": x2, "s": s,
            "g": g}


def _cox_breslow(x, t, d):
    """Cox MLE by Newton (Breslow ties) and the Breslow cumulative hazard at each record."""
    beta = np.zeros(x.shape[1])
    for _ in range(50):
        r = np.exp(x @ beta)
        times = np.unique(t[d == 1])
        grad = np.zeros_like(beta)
        hess = np.zeros((beta.size, beta.size))
        for tau in times:
            risk = t >= tau
            fail = (t == tau) & (d == 1)
            s0 = r[risk].sum()
            s1 = (r[risk, None] * x[risk]).sum(0)
            s2 = (r[risk, None, None] * x[risk, :, None] * x[risk, None, :]).sum(0)
            m = fail.sum()
            grad += x[fail].sum(0) - m * s1 / s0
            hess -= m * (s2 / s0 - np.outer(s1, s1) / s0 ** 2)
        step = np.linalg.solve(-hess, grad)
        beta += step
        if np.abs(step).max() < 1e-12:
            break
    r = np.exp(x @ beta)
    times = np.unique(t[d == 1])
    jumps = np.array([((t == tau) & (d == 1)).sum() / r[t >= tau].sum() for tau in times])
    cumhaz = np.array([jumps[times <= ti].sum() for ti in t])
    return beta, r * cumhaz


def cox_null() -> dict[str, np.ndarray]:
    """No frailty; the first seed whose gamma-frailty score at theta = 0 is clearly negative.

    The score of ``theta`` at 0 is ``sum_g ((D_g - A_g)^2 - D_g) / 2`` at the Cox MLE
    (A_g: sum of the Breslow cumulative hazards of the group), computed here by explicit
    loops; a negative score puts the integrated-likelihood maximum at the boundary.
    """
    for seed in range(16303, 16403):
        rng = np.random.default_rng(seed)
        q, size = 25, 4
        g = np.repeat(np.arange(1, q + 1), size)
        n = g.size
        x1 = rng.normal(size=n)
        t = rng.exponential(1 / np.exp(0.4 * x1))
        c = rng.exponential(3.0, n)
        time = np.maximum(np.round(np.minimum(t, c), 3), 0.001)
        d = (t <= c).astype(int)
        _, cumhaz = _cox_breslow(x1[:, None], time, d)
        events = np.bincount(g - 1, d, q)
        expected = np.bincount(g - 1, cumhaz, q)
        score = (((events - expected) ** 2 - events) / 2).sum()
        if score < -2.0:
            return {"t": time, "d": d, "x1": x1, "g": g,
                    "meta": {"seed": seed, "gamma_score_at_zero": round(float(score), 6)}}
    raise RuntimeError("no boundary seed found")


# ---- recurrent events -----------------------------------------------------------------

def recurrent() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(16310)
    grid = 0.02
    rows = {k: [] for k in ("id", "start", "stop", "event", "treat", "age", "z")}
    for subject in range(1, 101):
        treat = int(rng.random() < 0.5)
        age = rng.normal()
        frailty = rng.gamma(2.0, 0.5)
        follow = round(rng.uniform(1.5, 3.5) / grid) * grid
        switch = round(rng.uniform(0.2, 3.0) / grid) * grid if rng.random() < 0.5 else math.inf
        base = 0.8 * frailty * math.exp(0.5 * treat - 0.3 * age)
        rates = (base, base * math.exp(0.4))
        events, cursor = [], 0.0
        while True:
            e = rng.exponential()
            if cursor < switch:
                span = rates[0] * (switch - cursor)
                if e < span:
                    cursor += e / rates[0]
                else:
                    cursor = switch + (e - span) / rates[1]
            else:
                cursor += e / rates[1]
            if cursor >= follow:
                break
            moment = max(round(cursor / grid) * grid, grid)
            if moment < follow and (not events or moment > events[-1] + 1e-9):
                events.append(moment)
        cuts = sorted(set(events) | ({switch} if switch < follow else set()) | {follow})
        previous = 0.0
        for point in cuts:
            rows["id"].append(subject)
            rows["start"].append(previous)
            rows["stop"].append(point)
            rows["event"].append(int(any(abs(point - e) < 1e-9 for e in events)))
            rows["treat"].append(treat)
            rows["age"].append(age)
            rows["z"].append(int(previous >= switch - 1e-9))
            previous = point
    return {k: np.asarray(v) for k, v in rows.items()}


# ---- parametric shared frailty ----------------------------------------------------------

def parametric(seed: int, distribution: str, theta: float) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    q = 60
    sizes = rng.integers(2, 7, q)
    g = np.repeat(np.arange(1, q + 1), sizes)
    n = g.size
    if distribution == "gamma":
        z = rng.gamma(1 / theta, theta, q)
    else:
        z = rng.wald(1.0, 1.0 / theta, q)
    x1 = rng.normal(size=n)
    x2 = (rng.random(n) < 0.5).astype(int)
    p = 1.3
    rate = z[g - 1] * np.exp(-0.5 + 0.6 * x1 - 0.4 * x2)
    t = (rng.exponential(size=n) / rate) ** (1 / p)
    c = rng.exponential(2.5, n)
    time = np.maximum(np.minimum(t, c), 1e-4)
    return {"t": time, "d": (t <= c).astype(int), "x1": x1, "x2": x2, "g": g}


# ---- survival treatment effects ---------------------------------------------------------

def treatment_effects() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(16320)
    n = 600
    x1 = rng.normal(size=n)
    x2 = (rng.random(n) < 0.5).astype(int)
    p_treat = 1 / (1 + np.exp(-(0.2 + 0.6 * x1 - 0.5 * x2)))
    a = (rng.random(n) < p_treat).astype(int)
    shape = np.where(a == 1, 1.5, 1.2)
    mu = np.where(a == 1, -0.6 + 0.4 * x1 - 0.3 * x2, -0.2 + 0.5 * x1 + 0.2 * x2)
    t = (rng.exponential(size=n) / np.exp(mu)) ** (1 / shape)
    c = (rng.exponential(size=n) / np.exp(-1.6 + 0.4 * x2)) ** (1 / 1.1)
    time = np.maximum(np.minimum(t, c), 1e-4)
    return {"t": time, "d": (t <= c).astype(int), "x1": x1, "x2": x2, "treat": a}


DATASETS = {
    "cox": {"cox_gamma": cox_gamma, "cox_entry": cox_entry, "cox_null": cox_null},
    "recurrent": {"recurrent": recurrent},
    "parametric": {"param_gamma": lambda: parametric(16330, "gamma", 0.6),
                   "param_invgauss": lambda: parametric(16331, "invgaussian", 0.8)},
    "stteffects": {"stteffects": treatment_effects},
}


def write_data() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for stage, builders in DATASETS.items():
        datasets, meta = {}, {}
        for name, build in builders.items():
            frame = build()
            if "meta" in frame:
                meta[name] = frame.pop("meta")
            datasets[name] = _columns(frame)
        payload = {"generator": "scripts/generate_survival_frailty_oracles.py data",
                   "meta": meta, "datasets": datasets}
        path = FIXTURES / f"data_{stage}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        print(path, path.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("data", "numpy"))
    args = parser.parse_args()
    if args.step == "data":
        write_data()
    else:
        from_numpy()


def _load(stage: str) -> dict[str, dict[str, np.ndarray]]:
    raw = json.loads((FIXTURES / f"data_{stage}.json").read_text())["datasets"]
    return {name: {k: np.asarray(v) for k, v in columns.items()} for name, columns in raw.items()}


# ---- parametric frailty by explicit numerical integration -----------------------------

def _log_density(frailty: str, theta: float):
    from scipy import special

    if frailty == "gamma":
        nu = 1 / theta
        const = nu * math.log(nu) - special.gammaln(nu)
        return lambda z: const + (nu - 1) * np.log(z) - nu * z
    return lambda z: (-0.5 * np.log(2 * math.pi * theta * z ** 3)
                      - (z - 1) ** 2 / (2 * theta * z))


def _log_integral(log_f, events: float, exposure: float) -> float:
    """ln int_0^inf z^D exp(-A z) f(z) dz by adaptive quadrature in u = ln z."""
    from scipy import integrate, optimize

    def log_integrand(u):
        z = math.exp(u)
        return events * u - exposure * z + log_f(z) + u

    peak = optimize.minimize_scalar(lambda u: -log_integrand(u), bounds=(-30, 10),
                                    method="bounded", options={"xatol": 1e-10})
    top, mode = -peak.fun, peak.x
    value = 0.0
    # The integrand decays at least like exp((nu + D) u) on the left and like
    # exp(-c exp(u)) on the right of its mode: [mode - 200, mode + 12] holds it all.
    for low, high in ((mode - 200.0, mode), (mode, mode + 12.0)):
        piece, _ = integrate.quad(lambda u: math.exp(log_integrand(u) - top), low, high,
                                  epsabs=0.0, epsrel=1e-13, limit=400)
        value += piece
    return top + math.log(value)


def _index(params, x, t, metric_ph=True, weibull=True):
    k = x.shape[1]
    mu = x @ params[:k]
    p = math.exp(params[k]) if weibull else 1.0
    log_h = (params[k] if weibull else 0.0) + mu + (p - 1) * np.log(t)
    big_h = np.exp(mu) * t ** p
    return log_h, big_h


def parametric_loglik(params, data, frailty, weibull):
    """Marginal log likelihood (time scale) integrating the frailty numerically per group."""
    x = np.column_stack([np.ones(data["t"].size), data["x1"], data["x2"]])
    log_h, big_h = _index(params, x, data["t"], weibull=weibull)
    total = float((data["d"] * log_h).sum())
    if frailty is None:
        return total - float(big_h.sum())
    log_f = _log_density(frailty, math.exp(params[-1]))
    for group in np.unique(data["g"]):
        rows = data["g"] == group
        total += _log_integral(log_f, float(data["d"][rows].sum()), float(big_h[rows].sum()))
    return total


def _central_gradient(f, x, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        grad[i] = (f(x + e) - f(x - e)) / (2 * h)
    return grad


def _central_hessian(f, x, h=1e-3):
    k = x.size
    hess = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h, h
            value = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
            hess[i, j] = hess[j, i] = value
    return hess


def parametric_oracles() -> dict:
    from scipy import optimize

    out = {}
    data = _load("parametric")
    plan = {"param_gamma": "gamma", "param_invgauss": "invgaussian"}
    for name, frailty in plan.items():
        frame = data[name]
        log_t = float((frame["d"] * np.log(frame["t"])).sum())
        for weibull in (True, False):
            label = f"{'weibull' if weibull else 'exponential'}_ph"
            start = np.array([-0.5, 0.6, -0.4] + ([math.log(1.3)] if weibull else []) + [math.log(0.6)])
            null_start = start[:-1]

            def negative(params, frailty=frailty, weibull=weibull, frame=frame):
                return -parametric_loglik(params, frame, frailty, weibull)

            def negative_null(params, weibull=weibull, frame=frame):
                return -parametric_loglik(params, frame, None, weibull)

            null = optimize.minimize(negative_null, null_start, method="BFGS",
                                     jac=lambda p, f=negative_null: _central_gradient(f, p),
                                     options={"gtol": 1e-9, "maxiter": 500})
            best = optimize.minimize(negative, start, method="BFGS",
                                     jac=lambda p, f=negative: _central_gradient(f, p),
                                     options={"gtol": 1e-7, "maxiter": 500})
            hessian = _central_hessian(lambda p, f=negative: -f(p), best.x)
            covariance = np.linalg.inv(-hessian)
            checks = [start, best.x + 0.05]
            out[f"{name}_{label}"] = {
                "frailty": frailty, "distribution": "weibull" if weibull else "exponential",
                "params": best.x.tolist(), "log_likelihood": -best.fun + log_t,
                "covariance": covariance.tolist(), "gradient_norm": float(np.abs(best.jac).max()),
                "null_params": null.x.tolist(), "null_log_likelihood": -null.fun + log_t,
                "evaluations": [{"params": c.tolist(),
                                 "log_likelihood": parametric_loglik(c, frame, frailty, weibull) + log_t}
                                for c in checks],
            }
            print(name, label, best.x, -best.fun + log_t, best.message)
    return out


# ---- survival treatment effects as explicit GMM ----------------------------------------

def _weibull_ml(x, t, event, w):
    """Weighted Weibull PH ML: S = exp(-exp(x'b) t^p); BFGS with the analytic gradient."""
    from scipy import optimize

    lt = np.log(t)

    def negative(theta):
        b, a = theta[:-1], theta[-1]
        p = math.exp(a)
        mu = x @ b
        big_h = np.exp(mu + p * lt)
        value = (w * (event * (a + mu + (p - 1) * lt) - big_h)).sum()
        s_mu = event - big_h
        s_a = event * (1 + p * lt) - big_h * p * lt
        grad = np.concatenate([x.T @ (w * s_mu), [(w * s_a).sum()]])
        return -value, -grad

    start = np.zeros(x.shape[1] + 1)
    start[0] = math.log((w * event).sum() / (w * t).sum())
    fit = optimize.minimize(negative, start, jac=True, method="BFGS",
                            options={"gtol": 1e-11, "maxiter": 2000})
    return fit.x


def _logit(z, a):
    import statsmodels.api as sm

    return sm.Logit(a, z).fit(disp=0, tol=1e-14, maxiter=200).params


def _ste_moments(theta, layout, data, method, estimand):
    """Stacked estimating functions psi_i(theta) [n, size] written from the definitions."""
    t, d, a, x, z, c = data
    lt = np.log(t)
    n = t.size
    pom = theta[:2]
    out = np.zeros((n, theta.size))
    arms = [(a == 0).astype(float), (a == 1).astype(float)]
    target = np.ones(n) if estimand != "atet" else arms[1]
    if "gamma" in layout:
        gamma = theta[layout["gamma"]]
        p1 = 1 / (1 + np.exp(-(z @ gamma)))
        out[:, layout["gamma"]] = (a - p1)[:, None] * z
        omega = [p1 / (1 - p1), np.ones(n)] if estimand == "atet" else [1 / (1 - p1), 1 / p1]
    for j in (0, 1):
        block = theta[layout[f"model{j}"]]
        design = c if method == "ipw" else x
        event = 1 - d if method == "ipw" else d
        b, alpha = block[:-1], block[-1]
        p = math.exp(alpha)
        mu = design @ b
        big_h = np.exp(mu + p * lt)
        weight = arms[j] * (omega[j] if method == "ipwra" else 1.0)
        s = np.column_stack([design * (event - big_h)[:, None],
                             event * (1 + p * lt) - big_h * p * lt])
        out[:, layout[f"model{j}"]] = weight[:, None] * s
        if method == "ipw":
            w = arms[j] * d * omega[j] * np.exp(big_h)
            out[:, j] = w * (t - pom[j])
        else:
            from scipy import special
            mean = special.gamma(1 + 1 / p) * np.exp(-mu / p)
            out[:, j] = target * (mean - pom[j])
    return out


def ste_case(data, method, estimand, cluster=None):
    t, d, a, x, z, c = data
    n = t.size
    arms = [(a == 0).astype(float), (a == 1).astype(float)]
    layout, pieces = {}, [np.zeros(2)]
    size = 2
    if method in ("ipw", "ipwra"):
        gamma = _logit(z, a)
        layout["gamma"] = slice(size, size + gamma.size)
        size += gamma.size
        pieces.append(gamma)
        p1 = 1 / (1 + np.exp(-(z @ gamma)))
        omega = [p1 / (1 - p1), np.ones(n)] if estimand == "atet" else [1 / (1 - p1), 1 / p1]
    for j in (0, 1):
        if method == "ipw":
            theta_j = _weibull_ml(c, t, 1 - d, arms[j])
        else:
            w = arms[j] * (omega[j] if method == "ipwra" else 1.0)
            theta_j = _weibull_ml(x, t, d, w)
        layout[f"model{j}"] = slice(size, size + theta_j.size)
        size += theta_j.size
        pieces.append(theta_j)
    theta = np.concatenate(pieces)
    # POMs solve their own (linear) moments given the nuisance parameters.
    for j in (0, 1):
        trial = theta.copy()
        trial[j] = 0.0
        m0 = _ste_moments(trial, layout, data, method, estimand)[:, j].sum()
        trial[j] = 1.0
        m1 = _ste_moments(trial, layout, data, method, estimand)[:, j].sum()
        theta[j] = -m0 / (m1 - m0)
    psi = _ste_moments(theta, layout, data, method, estimand)
    jac = np.zeros((size, size))
    for k in range(size):
        h = 1e-6 * max(1.0, abs(theta[k]))
        up, down = theta.copy(), theta.copy()
        up[k] += h
        down[k] -= h
        jac[:, k] = (_ste_moments(up, layout, data, method, estimand).sum(0)
                     - _ste_moments(down, layout, data, method, estimand).sum(0)) / (2 * h)
    if cluster is None:
        meat = psi.T @ psi
    else:
        totals = np.zeros((cluster.max() + 1, size))
        np.add.at(totals, cluster, psi)
        meat = totals.T @ totals
    inverse = np.linalg.inv(jac)
    cov = inverse @ meat @ inverse.T
    return {"pom": theta[:2].tolist(), "pom_covariance": cov[:2, :2].tolist(),
            "moment_norm": float(np.abs(psi.sum(0)).max()),
            "nuisance": theta[2:].tolist()}


def ste_oracles() -> dict:
    raw = _load("stteffects")["stteffects"]
    t, d, a = raw["t"], raw["d"].astype(float), raw["treat"]
    x = np.column_stack([np.ones(t.size), raw["x1"], raw["x2"]])
    data = (t, d, a, x, x, x)
    out = {}
    for method in ("ra", "ipw", "ipwra"):
        for estimand in ("ate", "atet"):
            out[f"{method}_{estimand}"] = ste_case(data, method, estimand)
            print(method, estimand, out[f"{method}_{estimand}"]["pom"])
    cluster = np.arange(t.size) // 15
    out["ipwra_ate_cluster15"] = ste_case(data, "ipwra", "ate", cluster)
    zc = np.column_stack([np.ones(t.size), raw["x2"]])
    out["ipw_ate_cx_x2"] = ste_case((t, d, a, x, x, zc), "ipw", "ate")
    return out


def from_numpy() -> None:
    out = {"generator": "scripts/generate_survival_frailty_oracles.py numpy",
           "method": "scipy.integrate.quad over the frailty density per group; BFGS with central "
                     "difference gradients; central-difference Hessian (h = 1e-3). Treatment "
                     "effects: statsmodels Logit, SciPy BFGS Weibull fits, moments written "
                     "from their definitions, central-difference Jacobian (h = 1e-6), "
                     "A^-1 B A^-T",
           "parametric": parametric_oracles(),
           "stteffects": ste_oracles()}
    path = FIXTURES / "numpy_reference.json"
    path.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(path, path.stat().st_size)


if __name__ == "__main__":
    main()
