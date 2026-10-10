"""Read actual saved finite BMA states against an independent observation-space law.

Development reference only: NumPy/SciPy/mpmath are not runtime dependencies.
No production estimator, posterior replay or distribution function is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import special, stats


def payload(value):
    return value["payload"] if set(value) == {"schema_version", "payload"} else value


def observations(state):
    source = state["source"]
    rows = source["positions"]
    values = np.array([[column[i] for column in source["values"]] for i in rows], dtype=float)
    if len(rows) > 500:
        raise ValueError("Independent observation-space audit is bounded to 500 physical rows.")
    return values[:, 0], np.c_[np.ones(len(rows)), values[:, 1:]]


def gamma_ratio(a, increment):
    if a < 100000:
        return float(special.gammaln(a + increment) - special.gammaln(a))
    import mpmath as mp

    with mp.workdps(60):
        return float(mp.loggamma(mp.mpf(a) + mp.mpf(increment)) - mp.loggamma(mp.mpf(a)))


def inverse_gamma_variance(scale, first_increment, second_increment):
    if scale < 1e-100 or second_increment < 1e-100:
        import mpmath as mp

        with mp.workdps(60):
            return float(
                mp.mpf(scale) ** 2 / (mp.mpf(first_increment) ** 2 * mp.mpf(second_increment))
            )
    return scale**2 / (first_increment**2 * second_increment)


def reference(state):
    """NIG updates from the n-by-n observation covariance, not coefficient precision."""
    y, full = observations(state)
    models = []
    evidence = []
    n, total = full.shape
    spec = state["spec"]
    forced = len(spec["forced"])
    optional = len(spec["optional"])
    for mask, prior in enumerate(state["priors"]):
        ix = [*range(1 + forced), *[1 + forced + j for j in range(optional) if mask & (1 << j)]]
        X = full[:, ix]
        m, V = np.array(prior["mean"]), np.array(prior["scale_matrix"])
        A = np.eye(n) + X @ V @ X.T
        residual = y - X @ m
        solve = np.linalg.solve(A, np.c_[residual, X @ V])
        mean = m + V @ X.T @ solve[:, 0]
        vn = V - V @ X.T @ solve[:, 1:]
        shape = prior["shape"] + n / 2
        update = residual @ solve[:, 0] / 2
        scale = prior["scale"] + update
        # log1p avoids multiplying a tiny relative scale update by a huge shape.
        increment = update / prior["scale"]
        log = (
            -n / 2 * math.log(2 * math.pi)
            - np.linalg.slogdet(A)[1] / 2
            + gamma_ratio(prior["shape"], n / 2)
            - n / 2 * math.log(prior["scale"])
            - shape * math.log1p(increment)
        )
        mean_full, scale_full = np.zeros(total), np.zeros((total, total))
        mean_full[ix] = mean
        scale_full[np.ix_(ix, ix)] = vn * (scale / shape)
        denom1 = math.fsum((prior["shape"], (n - 2) / 2))
        denom2 = math.fsum((prior["shape"], (n - 4) / 2))
        beta_mean = math.fsum((prior["shape"], (n - 1) / 2)) > 0
        covariance = None
        if denom1 > 0:
            covariance = np.zeros((total, total))
            covariance[np.ix_(ix, ix)] = vn * (scale / denom1)
        models.append(
            dict(
                indices=ix,
                location=mean_full,
                scale=scale_full,
                covariance=covariance,
                shape=shape,
                b=scale,
                beta_mean=beta_mean,
                sigma_mean=scale / denom1 if denom1 > 0 else None,
                sigma_variance=inverse_gamma_variance(scale, denom1, denom2)
                if denom2 > 0
                else None,
            )
        )
        evidence.append(log)
    prior_logs = np.log(np.array(state["prior_model_odds"]))
    prior_logs -= special.logsumexp(prior_logs)
    logs = np.array(evidence) + prior_logs
    weights = np.exp(logs - special.logsumexp(logs))
    incidence = np.array(
        [[bool(mask & (1 << j)) for j in range(optional)] for mask in range(len(models))],
        dtype=float,
    ).reshape(len(models), optional)
    pip = weights @ incidence
    return dict(
        models=models,
        weights=weights,
        evidence=np.array(evidence),
        log_prior=prior_logs,
        log_weights=logs - special.logsumexp(logs),
        mixture_evidence=float(special.logsumexp(logs)),
        inclusion=pip,
        inclusion_covariance=(incidence - pip).T @ ((incidence - pip) * weights[:, None]),
    )


def t_cdf(value, location, scale, shape, *, left=False):
    if scale == 0:
        return float(value > location if left else value >= location)
    return float(stats.t.cdf((value - location) / scale, 2 * shape))


def coefficient_cdf(ref, coordinate, value, *, left=False):
    return math.fsum(
        w
        * t_cdf(
            value,
            m["location"][coordinate],
            math.sqrt(m["scale"][coordinate, coordinate]),
            m["shape"],
            left=left,
        )
        for w, m in zip(ref["weights"], ref["models"])
    )


def variance_cdf(ref, value):
    return math.fsum(
        w * float(stats.invgamma.cdf(value, m["shape"], scale=m["b"]))
        for w, m in zip(ref["weights"], ref["models"])
    )


def query_laws(ref, Z):
    return [(Z @ m["location"], Z @ m["scale"] @ Z.T, m) for m in ref["models"]]


def query_cdf(ref, Z, coordinate, value, *, outcome=False):
    return math.fsum(
        w
        * t_cdf(
            value,
            mu[coordinate],
            math.sqrt(scale[coordinate, coordinate] + (m["b"] / m["shape"] if outcome else 0)),
            m["shape"],
        )
        for w, (mu, scale, m) in zip(ref["weights"], query_laws(ref, Z))
    )


def weighted_geometry(means, covariances, weights):
    means = np.array(means)
    center = weights @ means
    within = sum(w * C for w, C in zip(weights, covariances))
    between = sum(w * np.outer(m - center, m - center) for w, m in zip(weights, means))
    return center, within, between, within + between


def audit(state, *, atol=2e-9):
    state = payload(state)
    isquery = state["schema"] == "openecon.bayesian_bma_prediction.v1"
    parent = state["posterior"] if isquery else state
    ref = reference(parent)
    errors = {}

    def check(name, actual, expected, *, limit=atol):
        if actual is None or expected is None:
            if actual is not expected:
                raise AssertionError(name + ": availability differs")
            errors[name] = 0.0
            return
        actual, expected = np.array(actual, dtype=float), np.array(expected, dtype=float)
        if actual.shape != expected.shape:
            raise AssertionError(name + ": complete shape differs")
        delta = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        error = delta / max(1.0, float(np.max(np.abs(expected))) if expected.size else 0.0)
        if not np.all(np.isfinite(actual)) or error > limit:
            raise AssertionError(f"{name}: normalized error {error}")
        errors[name] = error

    r = parent["results"]
    for key, reference_key in [
        ("log_model_evidence", "evidence"),
        ("log_model_prior", "log_prior"),
        ("log_model_weights", "log_weights"),
        ("model_weights", "weights"),
        ("log_mixture_evidence", "mixture_evidence"),
        ("posterior_inclusion", "inclusion"),
        ("inclusion_covariance", "inclusion_covariance"),
    ]:
        check(key, r[key], ref[reference_key])
    models, w = ref["models"], ref["weights"]
    if all(m["covariance"] is not None for m in models):
        mean, within, between, C = weighted_geometry(
            [m["location"] for m in models], [m["covariance"] for m in models], w
        )
        for name, value in [
            ("coefficient_mean", mean),
            ("within_coefficient_covariance", within),
            ("between_coefficient_covariance", between),
            ("coefficient_covariance", C),
        ]:
            check(name, r[name], value)
        if all(m["sigma_variance"] is not None for m in models):
            means = [np.r_[m["location"], m["sigma_mean"]] for m in models]
            Cs = []
            for m in models:
                block = np.zeros((len(mean) + 1, len(mean) + 1))
                block[:-1, :-1] = m["covariance"]
                block[-1, -1] = m["sigma_variance"]
                Cs.append(block)
            joint, jw, jb, JC = weighted_geometry(means, Cs, w)
            for name, value in [
                ("joint_mean", joint),
                ("within_joint_covariance", jw),
                ("between_joint_covariance", jb),
                ("joint_covariance", JC),
            ]:
                check(name, r[name], value)
    for j, limits in enumerate(r["coefficient_quantiles"]):
        for p, value in zip([parent["spec"]["alpha"] / 2, 1 - parent["spec"]["alpha"] / 2], limits):
            left = coefficient_cdf(ref, j, value, left=True)
            right = coefficient_cdf(ref, j, value)
            gap = max(0.0, left - p, p - right)
            if gap > 2e-10:
                raise AssertionError(f"coefficient[{j}] true mixture quantile: {gap}")
            errors[f"beta[{j}]quantile[{p}]"] = gap
    for p, value in zip(
        [parent["spec"]["alpha"] / 2, 1 - parent["spec"]["alpha"] / 2], r["sigma2_quantiles"]
    ):
        check(f"sigma2quantile[{p}]", variance_cdf(ref, value), p, limit=2e-10)
    if isquery:
        source = state["source"]
        Z = np.c_[
            np.ones(len(source["positions"])),
            np.array([[column[i] for column in source["values"]] for i in source["positions"]]),
        ]
        rr = state["results"]
        if all(m["covariance"] is not None for m in models):
            means = [Z @ m["location"] for m in models]
            center, within, between, C = weighted_geometry(
                means, [Z @ m["covariance"] @ Z.T for m in models], w
            )
            noise = math.fsum(ww * m["sigma_mean"] for ww, m in zip(w, models))
            for name, value in [
                ("mean", center),
                ("within_mean_covariance", within),
                ("between_mean_covariance", between),
                ("mean_covariance", C),
                ("outcome_covariance", C + noise * np.eye(len(Z))),
            ]:
                check(name, rr[name], value)
            if rr["parameter_query_covariance"] is not None:
                pm = np.array([np.r_[m["location"], m["sigma_mean"]] for m in models])
                pc = w @ pm
                within_cross = sum(
                    ww * np.vstack((m["covariance"] @ Z.T, np.zeros(len(Z))))
                    for ww, m in zip(w, models)
                )
                between_cross = sum(
                    ww * np.outer(mm - pc, mu - center) for ww, mm, mu in zip(w, pm, means)
                )
                check(
                    "parameter_query_covariance",
                    rr["parameter_query_covariance"],
                    within_cross + between_cross,
                )
        for j in range(len(Z)):
            for target in ("mean", "outcome"):
                for p, value in zip(
                    [state["alpha"] / 2, 1 - state["alpha"] / 2], rr[target + "_quantiles"][j]
                ):
                    check(
                        f"query[{j}]{target}[{p}]",
                        query_cdf(ref, Z, j, value, outcome=target == "outcome"),
                        p,
                        limit=2e-10,
                    )
    return dict(
        passed=True,
        reference="independent NumPy observation-space proper NIG; actual finite Student-t/IG/point-atom mixture; full within/between original-unit covariance",
        errors=errors,
        models=len(models),
        checks=len(errors),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = {"passed": True, "states": []}
    for path in args.state:
        raw = path.read_bytes()
        result = audit(json.loads(raw))
        receipt["states"].append(
            {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), **result}
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "states": len(receipt["states"]),
                "checks": sum(v["checks"] for v in receipt["states"]),
            }
        )
    )


if __name__ == "__main__":
    main()
