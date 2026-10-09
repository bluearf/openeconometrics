"""Independently verify eight recorded missing-data results with NumPy/SciPy.

This development oracle never imports openecon or Torch. Native provenance,
actual Run, source parity and restart are established by the separate native
acceptance receipt. This script compares its full saved inputs/states and raw
recorded table payloads; it does not certify finite-chain convergence or vendor
execution equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, f, norm, t

METHODS = ("em", "mcar", "mvn", "monotone", "normal", "pmm", "logit", "d1")
SOURCES = (
    "https://doi.org/10.1080/01621459.1988.10478722",
    "https://doi.org/10.1111/j.2517-6161.1977.tb01600.x",
    "https://www2.stat.duke.edu/~jerry/Papers/Bmtka07.pdf",
)


def canonical_hash(value, *, ascii=True):
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=ascii, allow_nan=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_json(value):
    """Use the declared null representation for infinite denominator limits."""
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Checks:
    """Retain every numerical comparison, including actual and reference values."""

    def __init__(self):
        self.checks = []

    def truth(self, name, passed, *, actual=None, reference=None):
        self.checks.append(
            {"name": name, "pass": bool(passed), "actual": actual, "reference": reference}
        )

    def equal(self, name, actual, reference):
        self.truth(name, actual == reference, actual=actual, reference=reference)

    def numeric(self, name, actual, reference, *, atol=1e-10, rtol=1e-8):
        a, b = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
        same_shape = a.shape == b.shape
        passed = same_shape and bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False))
        if same_shape:
            finite = np.isfinite(a) & np.isfinite(b)
            error = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
        else:
            error = None
        self.checks.append(
            {
                "name": name,
                "pass": bool(passed),
                "max_abs_error": error,
                "atol": atol,
                "rtol": rtol,
                "actual": a.tolist(),
                "reference": b.tolist(),
            }
        )

    def finish(self, **values):
        return {
            "pass": all(c["pass"] for c in self.checks),
            "check_count": len(self.checks),
            "max_abs_error": max((c.get("max_abs_error") or 0 for c in self.checks), default=0),
            "checks": self.checks,
            **values,
        }


def checksum(checks, state, name, *, ascii=True):
    payload = {k: v for k, v in state.items() if k != "integrity_sha256"}
    checks.equal(name, state.get("integrity_sha256"), canonical_hash(payload, ascii=ascii))


def matrix(inputs, key, columns):
    values = inputs[key]
    return pd.DataFrame({name: values[name] for name in columns}).to_numpy(dtype=float)


def display_frames(execution):
    outputs = execution["outputs"]
    if execution.get("status") != "ok" or len(outputs) != len(METHODS):
        raise ValueError("Require one completed execution with exactly eight table payloads.")
    result = {}
    for method, output in zip(METHODS, outputs):
        if output.get("type") != "table" or not output.get("latex"):
            raise ValueError(f"{method}: a recorded rendered/exportable table is required.")
        data = output["data"]
        columns = data["columns"]
        rows = data["rows"]
        if len(set(columns)) != len(columns) or any(len(row) != len(columns) for row in rows):
            raise ValueError(f"{method}: malformed recorded table.")
        if data.get("total_rows") != len(rows) or data.get("total_columns") != len(columns):
            raise ValueError(
                f"{method}: a truncated native table cannot establish full acceptance."
            )
        result[method] = pd.DataFrame(rows, columns=columns)
    return result


def gaussian_objective(data):
    """Observed marginal normal negative likelihood and its complete gradient.

    Parameterize Sigma=L L' with log diagonal Cholesky coordinates. This is a
    direct likelihood optimizer, without conditional filling or an EM update.
    """
    n, p = data.shape
    del n
    observed = ~np.isnan(data)
    groups = []
    for pattern in np.unique(observed, axis=0):
        if pattern.any():
            rows = np.all(observed == pattern, axis=1)
            groups.append((np.flatnonzero(pattern), data[rows][:, pattern]))
    lower = np.tril_indices(p)

    def unpack(z):
        factor = np.zeros((p, p))
        factor[lower] = z[p:]
        factor[np.diag_indices(p)] = np.exp(factor.diagonal())
        return z[:p], factor, factor @ factor.T

    def pack(mean, covariance):
        factor = np.linalg.cholesky(covariance)
        factor[np.diag_indices(p)] = np.log(factor.diagonal())
        return np.r_[mean, factor[lower]]

    def objective(z):
        mean, factor, covariance = unpack(z)
        score_mean, score_covariance = np.zeros(p), np.zeros((p, p))
        likelihood = 0.0
        for ix, values in groups:
            sub = covariance[np.ix_(ix, ix)]
            sign, determinant = np.linalg.slogdet(sub)
            if sign <= 0:
                raise ValueError("Oracle covariance is not positive definite.")
            inverse = np.linalg.inv(sub)
            centered = values - mean[ix]
            scatter = centered.T @ centered
            likelihood += 0.5 * (
                len(values) * (len(ix) * math.log(2 * math.pi) + determinant)
                + np.trace(inverse @ scatter)
            )
            score_mean[ix] -= inverse @ centered.sum(0)
            score_covariance[np.ix_(ix, ix)] += 0.5 * (
                len(values) * inverse - inverse @ scatter @ inverse
            )
        gradient_factor = 2 * score_covariance @ factor
        gradient_factor[np.diag_indices(p)] *= factor.diagonal()
        gradient = np.r_[score_mean, gradient_factor[lower]]
        return float(likelihood), gradient

    return objective, unpack, pack


def gaussian_fit(data):
    objective, unpack, pack = gaussian_objective(data)
    p = data.shape[1]
    initial = pack(np.nanmean(data, axis=0), np.eye(p))
    fitted = minimize(
        objective, initial, jac=True, method="BFGS", options={"gtol": 2e-9, "maxiter": 3000}
    )
    mean, _, covariance = unpack(fitted.x)
    likelihood, gradient = objective(fitted.x)
    if not np.isfinite(likelihood) or np.max(np.abs(gradient)) > 2e-5:
        raise ValueError(f"Independent observed-likelihood optimizer failed: {fitted.message}")
    return (
        {
            "mean": mean,
            "covariance": covariance,
            "loglikelihood": -likelihood,
            "gradient_max_abs": float(np.max(np.abs(gradient))),
            "optimizer_success": bool(fitted.success),
            "optimizer_message": str(fitted.message),
            "initialization": "observed-column mean and identity covariance; no native estimates used",
            "iterations": int(fitted.nit),
        },
        objective,
        pack,
    )


def diagnostic(method, state, inputs, shown, fitted, objective, pack):
    checks = Checks()
    checksum(checks, state, "diagnostic checksum", ascii=False)
    checks.equal("method", state["method"], "mvnorm_em" if method == "em" else "little_mcar")
    columns = state["columns"]
    data = matrix(inputs, "incomplete", columns)
    observed = ~np.isnan(data)
    informative = np.flatnonzero(observed.any(1)).tolist()
    sample = state["sample"]
    checks.equal("original rows", sample["n_total"], len(data))
    checks.equal("informative positions", sample["informative_positions"], informative)
    checks.equal(
        "all-missing positions",
        sample["all_missing_positions"],
        np.flatnonzero(~observed.any(1)).tolist(),
    )
    checks.equal(
        "observed coverage",
        sample["pair_counts"],
        (observed.astype(int).T @ observed.astype(int)).tolist(),
    )
    checks.equal("missing counts", sample["missing_counts"], (~observed).sum(0).tolist())
    labels = inputs["base"]["participant"]
    checks.equal(
        "duplicate row identity", sample["row_labels"], [f"builtins.str:{v!r}" for v in labels]
    )
    checks.numeric("saved means vs independent ML", state["estimates"], fitted["mean"], atol=2e-5)
    checks.numeric(
        "saved full population covariance vs independent ML",
        state["covariance"],
        fitted["covariance"],
        atol=2e-5,
    )
    checks.numeric(
        "saved full observed Gaussian log likelihood",
        state["observed_loglikelihood"],
        fitted["loglikelihood"],
        atol=2e-7,
        rtol=0,
    )
    native_value, native_gradient = objective(
        pack(np.asarray(state["estimates"]), np.asarray(state["covariance"]))
    )
    checks.numeric(
        "saved likelihood independently evaluated at native fit",
        state["observed_loglikelihood"],
        -native_value,
        atol=2e-8,
        rtol=0,
    )
    checks.truth(
        "native full likelihood score near zero",
        np.max(np.abs(native_gradient)) < 5e-5,
        actual=float(np.max(np.abs(native_gradient))),
        reference="<5e-5",
    )
    history = np.asarray(state["loglikelihood_history"])
    checks.truth(
        "observed likelihood monotonic to float64 roundoff",
        bool(np.all(np.diff(history) >= -1e-10 * np.maximum(1, np.abs(history[:-1])))),
        actual=history.tolist(),
    )
    checks.equal("EM history iteration geometry", len(history), state["iterations"] + 1)
    checks.truth(
        "reported parameter convergence",
        state["converged"] is True and state["final_parameter_change"] <= state["tolerance"],
    )
    if method == "em":
        checks.equal("displayed mean labels", shown["column"].tolist(), columns)
        checks.numeric(
            "actual displayed means vs independent ML",
            shown["mean"].to_numpy(),
            fitted["mean"],
            atol=2e-5,
        )
        checks.numeric(
            "actual displayed means vs saved means",
            shown["mean"].to_numpy(),
            state["estimates"],
            atol=1e-12,
            rtol=0,
        )
        return checks.finish(
            source_values={
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fitted.items()
            }
        )
    n = len(informative)
    correction = n / (n - 1)
    total, dimensional_sum, contributions = 0.0, 0, []
    by_names = {tuple(v["observed_columns"]): v for v in state["patterns"]}
    for pattern in np.unique(observed, axis=0):
        positions = np.flatnonzero(np.all(observed == pattern, axis=1))
        ix = np.flatnonzero(pattern)
        names = tuple(columns[i] for i in ix)
        dimensional_sum += len(ix)
        saved = by_names[names]
        checks.equal(f"pattern {names} physical positions", saved["positions"], positions.tolist())
        mean = data[positions][:, ix].mean(0) if len(ix) else np.empty(0)
        checks.numeric(
            f"pattern {names} observed means", saved["observed_means"], mean, atol=1e-12, rtol=0
        )
        value = 0.0
        if len(ix):
            delta = mean - fitted["mean"][ix]
            value = len(positions) * float(
                delta @ np.linalg.solve(fitted["covariance"][np.ix_(ix, ix)] * correction, delta)
            )
        checks.numeric(
            f"pattern {names} chi-square contribution",
            saved["statistic_contribution"],
            value,
            atol=3e-6,
        )
        total += value
        contributions.append(
            {
                "observed_columns": list(names),
                "positions": positions.tolist(),
                "mean": mean.tolist(),
                "statistic": value,
            }
        )
    df = dimensional_sum - len(columns)
    pv = float(chi2.sf(total, df))
    for name, value in (
        ("statistic", total),
        ("df", df),
        ("p_value", pv),
        ("statistic_covariance_scale", correction),
    ):
        checks.numeric(
            f"saved MCAR {name}", state[name], value, atol=3e-6 if name == "statistic" else 1e-7
        )
    for name, value in (
        ("statistic", total),
        ("df", df),
        ("p_value", pv),
        ("covariance_scale", correction),
    ):
        checks.numeric(
            f"actual displayed MCAR {name}",
            shown[name].to_numpy(),
            [value],
            atol=3e-6 if name == "statistic" else 1e-7,
        )
    return checks.finish(
        source_values={
            "statistic": total,
            "df": df,
            "p_value": pv,
            "covariance_scale": correction,
            "pattern_contributions": contributions,
        }
    )


def index_values(encoded):
    """Decode just the row labels needed to compare the declared native fixture."""
    if encoded["kind"] != "index" or encoded["dtype"] != "object":
        raise ValueError("Native fixture must retain its ordinary duplicate object index.")
    if any(v["type"] != "scalar" or not isinstance(v["value"], str) for v in encoded["values"]):
        raise ValueError("Native fixture index labels must remain strings.")
    return [v["value"] for v in encoded["values"]]


def generation(method, state, inputs, shown):
    checks = Checks()
    checksum(checks, state, "complete MI state checksum")
    expected_method = (
        "mi_mvn" if method == "mvn" else "mi_monotone" if method == "monotone" else "mi_chained"
    )
    checks.equal("method", state["method"], expected_method)
    columns = state["columns"]
    key = "monotone" if method == "monotone" else "binary" if method == "logit" else "incomplete"
    data = matrix(inputs, key, columns)
    original = np.asarray(state["original"], dtype=float)
    missing = np.isnan(data)
    checks.equal("original missing positions", np.isnan(original).tolist(), missing.tolist())
    checks.numeric(
        "original observed cells vs actual native input",
        original[~missing],
        data[~missing],
        atol=0,
        rtol=0,
    )
    completed = np.asarray(state["completed_matrices"], dtype=float)
    m, n, p = completed.shape
    checks.equal("complete matrix geometry", [n, p], list(data.shape))
    checks.truth("all completed cells finite", bool(np.isfinite(completed).all()))
    checks.numeric(
        "every observed cell preserved across all completed draws",
        completed[:, ~missing],
        np.repeat(data[~missing][None, :], m, axis=0),
        atol=0,
        rtol=0,
    )
    metadata = state["metadata"]
    checks.equal("metadata geometry", [metadata[k] for k in ("m", "n", "p")], [m, n, p])
    checks.equal("metadata mask", metadata["missing_mask"], missing.tolist())
    checks.equal("metadata missing counts", metadata["missing_counts"], missing.sum(0).tolist())
    checks.equal(
        "sample row preservation",
        metadata["sample"],
        {"nobs_original": n, "nobs": n, "dropped_rows": 0, "positions": list(range(n))},
    )
    checks.equal("ordered imputation IDs", metadata["imputation_ids"], list(range(1, m + 1)))
    checks.equal(
        "duplicate row index labels", index_values(metadata["index"]), inputs["base"]["participant"]
    )
    checks.equal(
        "row index name", metadata["index"]["name"], {"type": "scalar", "value": "participant"}
    )
    checks.truth(
        "no finite-chain or vendor-parity claim",
        metadata.get("converged") is False and metadata.get("stata_parity_validated") is False,
    )
    checks.equal("actual displayed variable order", shown["column"].tolist(), columns)
    for name, expected in (
        ("observed", n - missing.sum(0)),
        ("missing", missing.sum(0)),
        ("imputations", np.full(p, m)),
    ):
        checks.numeric(
            f"actual displayed {name} counts", shown[name].to_numpy(), expected, atol=0, rtol=0
        )
    if method in ("normal", "pmm", "logit"):
        expected = method
        checks.truth(
            "declared conditional models",
            set(metadata["methods"].values()) == {expected},
            actual=metadata["methods"],
            reference=expected,
        )
    if method == "monotone":
        order = [columns.index(name) for name in metadata["order"]]
        checks.truth(
            "nested observed sets in declared monotone order",
            all(not np.any(missing[:, a] & ~missing[:, b]) for a, b in zip(order, order[1:])),
        )
    if method == "pmm":
        for j, column in enumerate(columns):
            support = data[~missing[:, j], j]
            if missing[:, j].any():
                checks.truth(
                    f"{column} imputed values lie on actual observed support",
                    bool(np.isin(completed[:, missing[:, j], j], support).all()),
                    reference=np.unique(support).tolist(),
                )
                for i, chain in enumerate(metadata["chain_diagnostics"]):
                    donor_rows = chain["trace"][-1]["variables"][column]["donor_row_positions"]
                    checks.truth(
                        f"{column} imputation {i + 1} donors were originally observed",
                        len(donor_rows) == int(missing[:, j].sum())
                        and all(
                            type(v) is int and 0 <= v < n and not missing[v, j] for v in donor_rows
                        ),
                        actual=donor_rows,
                    )
                    checks.numeric(
                        f"{column} imputation {i + 1} saved final donors reproduce completed values",
                        completed[i, missing[:, j], j],
                        data[donor_rows, j],
                        atol=0,
                        rtol=0,
                    )
    if method == "logit":
        for j, column in enumerate(columns):
            if metadata["methods"].get(column) == "logit":
                checks.truth(
                    f"{column} completed values are binary",
                    bool(np.isin(completed[:, :, j], [0.0, 1.0]).all()),
                )
        checks.truth(
            "no MH stationarity claim", metadata["logit_sampler"]["stationarity_claim"] is False
        )
    return checks.finish(
        source_values={
            "rows": n,
            "columns": columns,
            "imputations": m,
            "missing_counts": missing.sum(0).tolist(),
            "observed_counts": (n - missing.sum(0)).tolist(),
            "completed_cells": int(completed.size),
        },
        scope="Observed-cell, saved-state, support and displayed-count checks; posterior-law accuracy is established by separate source oracles, not finite-chain convergence.",
    )


def pooling_values(pool):
    q, u = np.asarray(pool["estimates"]), np.asarray(pool["covariances"])
    m = len(q)
    mean, within = q.mean(0), u.mean(0)
    difference = q - mean
    between = difference.T @ difference / (m - 1)
    added = (1 + 1 / m) * between
    total = within + added
    lam = np.diag(added) / np.diag(total)
    riv = np.diag(added) / np.diag(within)
    with np.errstate(divide="ignore"):
        old_df = (m - 1) / lam**2
    complete = pool["complete_df"]
    df = (
        old_df
        if complete is None
        else 1 / (1 / old_df + 1 / (complete * (complete + 1) / (complete + 3) * (1 - lam)))
    )
    se, alpha = np.sqrt(np.diag(total)), pool["alpha"]
    statistic = mean / se
    probability = np.where(np.isinf(df), 2 * norm.sf(abs(statistic)), 2 * t.sf(abs(statistic), df))
    critical = np.where(np.isinf(df), norm.isf(alpha / 2), t.isf(alpha / 2, df))
    fmi = (riv + 2 / (df + 3)) / (1 + riv)
    return {
        "estimate": mean,
        "std_error": se,
        "statistic": statistic,
        "df": df,
        "p_value": probability,
        "ci_low": mean - critical * se,
        "ci_high": mean + critical * se,
        "lambda": lam,
        "fraction_missing_information": fmi,
        "relative_increase_variance": riv,
        "within_covariance": within,
        "between_covariance": between,
        "total_covariance": total,
    }


def d1_values(pool, restrictions, values, method):
    q, u = np.asarray(pool["estimates"]), np.asarray(pool["covariances"])
    r, null = np.asarray(restrictions), np.asarray(values)
    transformed = q @ r.T - null
    within = np.asarray([r @ value @ r.T for value in u]).mean(0)
    centered = transformed - transformed.mean(0)
    between = centered.T @ centered / (len(q) - 1)
    k, m = len(null), len(q)
    riv = (1 + 1 / m) * np.trace(np.linalg.solve(within, between)) / k
    delta = transformed.mean(0)
    statistic = float(delta @ np.linalg.solve(within, delta) / (k * (1 + riv)))
    degrees = k * (m - 1)
    if method == "li1991":
        denominator = (
            math.inf
            if riv == 0
            else (
                4 + (degrees - 4) * (1 + (1 - 2 / degrees) / riv) ** 2
                if degrees > 4
                else (m - 1) * (k + 1) * (1 + 1 / riv) ** 2 / 2
            )
        )
    elif method == "reiter2007":
        if degrees <= 4 or pool["complete_df"] is None:
            raise ValueError("Reiter finite-m domain is not satisfied.")
        a = riv * degrees / (degrees - 2)
        complete = pool["complete_df"]
        adjusted = complete * (complete + 1) / (complete + 3)
        c2, c4 = adjusted - 2 * (1 + a), adjusted - 4 * (1 + a)
        if c4 <= 0:
            raise ValueError("Reiter positive-moment domain is not satisfied.")
        # Reiter original eq2 expanded as independent scalar summands.
        pieces = [
            1 / c4,
            a * a * c2 / ((degrees - 4) * (1 + a) ** 2 * c4),
            8 * a * a * c2 / ((degrees - 4) * (1 + a) * c4**2),
            4 * a * a / ((degrees - 4) * (1 + a) * c4),
            4 * a * a / ((degrees - 4) * c4 * c2),
            16 * a * a * c2 / ((degrees - 4) * c4**3),
            8 * a * a / ((degrees - 4) * c4**2),
        ]
        denominator = 4 + 1 / sum(pieces)
    else:
        raise ValueError("Unknown saved D1 df convention.")
    pv = float(
        chi2.sf(k * statistic, k) if math.isinf(denominator) else f.sf(statistic, k, denominator)
    )
    return {
        "statistic": statistic,
        "df1": k,
        "df2": denominator,
        "p_value": pv,
        "relative_increase_variance": float(riv),
    }


def joint(state, states, inputs, shown):
    checks = Checks()
    checksum(checks, state, "joint saved-state checksum")
    pool = state["pool"]
    checksum(checks, pool, "full per-imputation pooling state checksum")
    mono = np.asarray(states["monotone"]["completed_matrices"], dtype=float)
    columns = states["monotone"]["columns"]
    xj, yj = columns.index("x"), columns.index("y")
    coefficients, covariances = [], []
    for completion in mono:
        design = np.column_stack((np.ones(len(completion)), completion[:, xj]))
        cross_inverse = np.linalg.inv(design.T @ design)
        coefficients.append(cross_inverse @ design.T @ completion[:, yj])
        residual = completion[:, yj] - design @ coefficients[-1]
        covariances.append(
            cross_inverse * (residual @ residual) / (len(completion) - design.shape[1])
        )
    checks.truth(
        "native pool term convention",
        len(pool["terms"]) == 2
        and pool["terms"][1] == "x"
        and pool["terms"][0] in ("_cons", "const", "Intercept", "intercept"),
        actual=pool["terms"],
    )
    checks.numeric(
        "all native per-imputation coefficients vs independent completed-data OLS",
        pool["estimates"],
        coefficients,
        atol=1e-10,
    )
    checks.numeric(
        "all native full per-imputation covariances vs independent completed-data OLS",
        pool["covariances"],
        covariances,
        atol=1e-10,
    )
    checks.numeric(
        "native complete-data df vs independent OLS df",
        pool["complete_df"],
        len(mono[0]) - 2,
        atol=0,
        rtol=0,
    )
    derived = pooling_values(pool)
    actual_tables = inputs["pool_tables"]
    for name in ("within_covariance", "between_covariance", "total_covariance"):
        checks.numeric(
            f"actual native full {name}", actual_tables[name]["rows"], derived[name], atol=1e-11
        )
    actual = pd.DataFrame(
        actual_tables["coefficients"]["rows"], columns=actual_tables["coefficients"]["columns"]
    )
    checks.equal("native marginal term order", actual["term"].tolist(), pool["terms"])
    for name, value in derived.items():
        if name.endswith("covariance"):
            continue
        vector = actual[name].to_numpy(dtype=float)
        if name == "df":
            vector = np.where(np.isnan(vector), np.inf, vector)
        checks.numeric(
            f"actual native Rubin/Barnard-Rubin marginal {name}", vector, value, atol=1e-9
        )
    expected = d1_values(pool, state["restrictions"], state["values"], state["df_method"])
    checks.equal("actual displayed joint method", shown["method"].tolist(), ["D1"])
    checks.equal(
        "actual displayed joint df convention", shown["df_method"].tolist(), [state["df_method"]]
    )
    for name, value in expected.items():
        actual_value = shown[name].to_numpy(dtype=float)
        if name == "df2":
            actual_value = np.where(np.isnan(actual_value), np.inf, actual_value)
        checks.numeric(
            f"actual displayed D1 {name} vs original-equation oracle",
            actual_value,
            [value],
            atol=1e-9,
        )
    return checks.finish(
        source_values={
            "pooled": {k: v.tolist() for k, v in derived.items()},
            "joint": {
                k: None if isinstance(v, float) and math.isinf(v) else v
                for k, v in expected.items()
            },
            "independent_OLS_coefficient_vectors": np.asarray(coefficients).tolist(),
            "independent_OLS_full_covariances": np.asarray(covariances).tolist(),
        },
        scope="Full per-imputation fitted OLS, Rubin/Barnard-Rubin marginal inference and original Reiter/Li D1 equations; no D2/D3 or vendor-equivalence claim.",
    )


def verify(states, inputs, execution):
    if set(states) != set(METHODS):
        raise ValueError("Exactly the eight full saved result states are required.")
    if not {"base", "incomplete", "monotone", "binary", "pool_tables"}.issubset(inputs):
        raise ValueError(
            "Actual native inputs including monotone and computed pool_tables are required."
        )
    frames = display_frames(execution)
    fitted, objective, pack = gaussian_fit(matrix(inputs, "incomplete", states["em"]["columns"]))
    results = {}
    for method in METHODS:
        try:
            if method in ("em", "mcar"):
                results[method] = diagnostic(
                    method, states[method], inputs, frames[method], fitted, objective, pack
                )
            elif method == "d1":
                results[method] = joint(states[method], states, inputs, frames[method])
            else:
                results[method] = generation(method, states[method], inputs, frames[method])
        except (KeyError, ValueError, TypeError, IndexError, np.linalg.LinAlgError) as exc:
            results[method] = {"pass": False, "error": str(exc), "error_type": type(exc).__name__}
    return {
        "status": "pass" if all(r["pass"] for r in results.values()) else "fail",
        "methods": results,
        "method_count": len(results),
        "source_urls": SOURCES,
        "infinite_df_representation": "null denotes the normal/chi-square complete-data limit",
        "scope": "Independent development numerical oracle for full saved states, actual native input readback and recorded eight display payloads. Native UI/source/restart provenance is established separately. Sampling posterior laws use separate source oracles; no arbitrary finite-chain convergence or licensed-vendor execution claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "states": args.states,
        "inputs": args.states.with_name("native-inputs.json"),
        "execution": args.states.with_name("native-execution.json"),
    }
    try:
        content = {name: json.loads(path.read_text()) for name, path in paths.items()}
        report = verify(content["states"], content["inputs"], content["execution"])
        report["artifact_hashes"] = {
            name: {
                "path": str(path.resolve()),
                "file_sha256": file_hash(path),
                "canonical_json_sha256": canonical_hash(content[name]),
            }
            for name, path in paths.items()
        }
        report["execution_id"] = content["execution"].get("id")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"status": "fail", "error": str(exc), "error_type": type(exc).__name__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(finite_json(report), indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "status": report["status"],
                "method_count": report.get("method_count", 0),
                "output": str(args.output.resolve()),
            }
        )
    )
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
