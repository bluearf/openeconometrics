"""Development-only independent full-affine SVD / difference-ECM oracle.

NumPy is solely an external numerical check, never a public runtime backend.
This replicates the nonlinear construction, not its inferential validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

NULLS = ("joint_levels", "adjustment", "explanatory_levels")


def signs(raw, anchors=None):
    raw = np.atleast_2d(raw)
    increments = np.diff(raw, axis=1, prepend=raw[:, :1])
    positive = np.cumsum(np.where(increments > 0, increments, 0), axis=1)
    negative = np.cumsum(np.where(increments < 0, increments, 0), axis=1)
    if anchors is not None:
        positive += anchors[:, :1]
        negative += anchors[:, 1:]
    return positive, negative


def design(y, positive, negative, lags):
    y, positive, negative = map(np.atleast_2d, (y, positive, negative))
    p, qp, qn = lags
    hold, total = max(lags), y.shape[1]
    columns = [np.ones_like(y[:, hold:])]
    columns += [y[:, hold - lag : total - lag] for lag in range(1, p + 1)]
    for variable, q in ((positive, qp), (negative, qn)):
        columns += [variable[:, hold - lag : total - lag] for lag in range(q + 1)]
    return np.stack(columns, axis=-1), y[:, hold:]


def restrictions(lags, null):
    p, qp, qn = lags
    matrix = np.zeros((3, p + qp + qn + 3))
    matrix[0, 1 : p + 1] = 1
    matrix[1, p + 1 : p + qp + 2] = 1
    matrix[2, p + qp + 2 :] = 1
    rows = {None: [], "joint_levels": [0, 1, 2], "adjustment": [0], "explanatory_levels": [1, 2]}[
        null
    ]
    return matrix[rows], np.array([1.0, 0.0, 0.0])[rows]


def fit(matrix, target, lags, null=None):
    """SVD of complete unprofiled parameter space, with full affine restrictions."""
    matrix = np.asarray(matrix)
    if matrix.ndim == 2:
        matrix = matrix[None, :, :]
    target = np.atleast_2d(target)
    r, value = restrictions(lags, null)
    if len(r):
        anchor = np.linalg.lstsq(r, value, rcond=None)[0]
        _, _, right = np.linalg.svd(r, full_matrices=True)
        basis = right[len(r) :].T
    else:
        anchor, basis = np.zeros(matrix.shape[-1]), np.eye(matrix.shape[-1])
    free = matrix @ basis
    left, singular, right = np.linalg.svd(free, full_matrices=False)
    tolerance = max(free.shape[-2:]) * np.finfo(float).eps * singular[:, :1]
    if not np.all(singular > tolerance):
        raise ValueError("Independent SVD found rank failure; no draws replaced.")
    coordinates = np.einsum(
        "bji,bj->bi", right, np.einsum("bji,bj->bi", left, target - matrix @ anchor) / singular
    )
    coefficients = anchor + coordinates @ basis.T
    residuals = target - np.einsum("btk,bk->bt", matrix, coefficients)
    df = target.shape[-1] - basis.shape[-1]
    factor = basis @ (right.transpose(0, 2, 1) / singular[:, None, :])
    covariance = (
        np.sum(residuals**2, axis=1)[:, None, None] / df * (factor @ factor.transpose(0, 2, 1))
    )
    return coefficients, covariance, residuals, df


def statistics(fitted, lags):
    coefficients, covariance, _, _ = fitted
    r, target = restrictions(lags, "joint_levels")
    delta, variance = coefficients @ r.T - target, r @ covariance @ r.T
    joint = np.einsum("bi,bi->b", delta, np.linalg.solve(variance, delta[..., None])[..., 0]) / 3
    adjustment = delta[:, 0] / np.sqrt(variance[:, 0, 0])
    explanatory = (
        np.einsum(
            "bi,bi->b",
            delta[:, 1:],
            np.linalg.solve(variance[:, 1:, 1:], delta[:, 1:, None])[..., 0],
        )
        / 2
    )
    return np.column_stack((joint, adjustment, explanatory))


def outcome(prefix, positive, negative, coefficients, lags, errors):
    """Recursive difference ECM, independent of production's levels AR filter."""
    p, qp, qn = lags
    hold, total = prefix.shape[1], positive.shape[1]
    coefficients = np.asarray(coefficients)
    phi = coefficients[1 : p + 1]
    rho = np.sum(phi) - 1
    result = np.empty_like(positive)
    result[:, :hold] = prefix
    terms, cursor = [], p + 1
    for variable, q in ((positive, qp), (negative, qn)):
        beta = coefficients[cursor : cursor + q + 1]
        terms.append((variable, beta, q))
        cursor += q + 1
    for t in range(hold, total):
        change = coefficients[0] + rho * result[:, t - 1] + errors[:, t - hold]
        for lag in range(1, p):
            change -= np.sum(phi[lag:]) * (result[:, t - lag] - result[:, t - lag - 1])
        for variable, beta, q in terms:
            change = (
                change
                + np.sum(beta) * variable[:, t - 1]
                + beta[0] * (variable[:, t] - variable[:, t - 1])
            )
            for lag in range(1, q):
                change -= np.sum(beta[lag + 1 :]) * (
                    variable[:, t - lag] - variable[:, t - lag - 1]
                )
        result[:, t] = result[:, t - 1] + change
    return result


def sample_from_shocks(cell, raw_shock, independent_shock, burn=64):
    """Independent nonlinear DGP written directly as an ECM in differences."""
    correlation = cell.structural_contemporaneous_correlation
    raw = np.r_[0.0, np.cumsum(cell.drift + raw_shock[1:])]
    positive, negative = signs(raw)
    p, qp, qn = cell.lags
    hold = max(cell.lags)
    y = np.zeros(len(raw))
    for t in range(hold, len(raw)):
        dy = 0.03 - correlation * cell.drift + cell.rho * y[t - 1]
        dy += cell.theta_positive * positive[0, t - 1] + cell.theta_negative * negative[0, t - 1]
        if p == 2:
            dy += cell.lagged_outcome_difference * (y[t - 1] - y[t - 2])
        for variable, q, delta in (
            (positive[0], qp, 0.25 + correlation),
            (negative[0], qn, 0.15 + correlation),
        ):
            dy += delta * (variable[t] - variable[t - 1])
            if q == 2:
                dy += 0.1 * (variable[t - 1] - variable[t - 2])
        dy += np.sqrt(1 - correlation**2) * independent_shock[t]
        y[t] = y[t - 1] + dy
    return y[burn:], raw[burn:]


def bootstrap(y, raw, lags, indices, starts, *, recenter, residual_scale):
    positive, negative = signs(raw)
    matrix, target = design(y, positive, negative, lags)
    observed = statistics(fit(matrix, target, lags), lags)[0]
    hold, rows = max(lags), len(raw) - max(lags)
    increments = raw[hold:] - raw[hold - 1 : -1]
    drift = increments.mean()
    records = {}
    for ni, null in enumerate(NULLS):
        coefficient, _, residual, df = fit(matrix, target, lags, null)
        residual = residual[0] - residual[0].mean()
        yscale, xscale = (
            (np.sqrt(rows / df), np.sqrt(rows / (rows - 1)))
            if residual_scale == "df"
            else (1.0, 1.0)
        )
        pool = np.column_stack((residual * yscale, (increments - drift) * xscale))
        paired = pool[indices]
        if recenter == "draw":
            paired = paired - paired.mean(axis=1, keepdims=True)
        positions = starts[:, None] + np.arange(hold)
        prefix = raw[positions]
        generated_raw = np.column_stack(
            (prefix, prefix[:, -1:] + np.cumsum(drift + paired[:, :, 1], axis=1))
        )
        anchors = np.column_stack((positive[0, starts], negative[0, starts]))
        xp, xn = signs(generated_raw, anchors)
        simulated = outcome(y[positions], xp, xn, coefficient[0], lags, paired[:, :, 0])
        draws = statistics(fit(*design(simulated, xp, xn, lags), lags), lags)
        extreme = int(
            np.sum(draws[:, ni] <= observed[ni] if ni == 1 else draws[:, ni] >= observed[ni])
        )
        records[null] = {
            "coefficients": coefficient[0],
            "draw_statistics": draws,
            "extreme_draws": extreme,
            "tail_probability": (1 + extreme) / (1 + len(indices)),
        }
    return {"observed_statistics": observed, "nulls": records}


def compare(replications=999):
    # Torch is confined to comparing the separate native implementation and
    # producing a shared random index stream; every oracle refit is NumPy SVD.
    import torch

    from benchmarks.research.nardl_bounds import research_bootstrap
    from benchmarks.research.nardl_bounds_grid import GRID, generate_sample

    torch.set_num_threads(2)
    start, rows = time.perf_counter(), []
    for ci in (0, 5, 7, 11):
        cell, seed = GRID[ci], 811763 + ci * 104729
        generator = torch.Generator(device="cpu").manual_seed(seed)
        total = cell.total + 64
        raw_shock = torch.randn(total, generator=generator, dtype=torch.float64)
        independent = torch.randn(total, generator=generator, dtype=torch.float64)
        y, x, _ = generate_sample(cell, seed)
        reference_y, reference_x = sample_from_shocks(cell, raw_shock.numpy(), independent.numpy())
        np.testing.assert_allclose(y.numpy(), reference_y, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(x.numpy(), reference_x, rtol=2e-12, atol=2e-12)
        for initial, recenter, scale in (
            ("fixed_prefix", "pool", "df"),
            ("original_block", "draw", "none"),
        ):
            inner_seed = seed + 524287
            result = research_bootstrap(
                y,
                x,
                p=cell.lags[0],
                q_positive=cell.lags[1],
                q_negative=cell.lags[2],
                replications=replications,
                seed=inner_seed,
                batch_size=512,
                memory_mb=128,
                initial=initial,
                recenter=recenter,
                residual_scale=scale,
            )
            generator = torch.Generator(device="cpu").manual_seed(inner_seed)
            hold = max(cell.lags)
            starts = (
                torch.randint(cell.total - hold + 1, (replications,), generator=generator)
                if initial == "original_block"
                else torch.zeros(replications, dtype=torch.long)
            )
            blocks = [
                torch.randint(
                    cell.total - hold,
                    (min(512, replications - first), cell.total - hold),
                    generator=generator,
                ).numpy()
                for first in range(0, replications, 512)
            ]
            oracle = bootstrap(
                y.numpy(),
                x.numpy(),
                cell.lags,
                np.vstack(blocks),
                starts.numpy(),
                recenter=recenter,
                residual_scale=scale,
            )
            np.testing.assert_allclose(
                result["observed_statistics"], oracle["observed_statistics"], rtol=2e-9, atol=2e-9
            )
            errors = {}
            for null in NULLS:
                actual = np.asarray(result["draw_statistics"][null])
                expected = oracle["nulls"][null]["draw_statistics"]
                np.testing.assert_allclose(actual, expected, rtol=2e-8, atol=2e-8)
                assert (
                    result["tests"][null]["extreme_draws"] == oracle["nulls"][null]["extreme_draws"]
                )
                errors[null] = float(np.max(np.abs(actual - expected)))
            rows.append(
                {
                    "cell": cell.name,
                    "initial": initial,
                    "recenter": recenter,
                    "scale": scale,
                    "inner_replications_per_null": replications,
                    "seed": seed,
                    "max_absolute_statistic_error": errors,
                    "all_tail_counts_identical": True,
                    "independent_difference_ecm_dgp_matches": True,
                }
            )
    root = Path(__file__).resolve().parents[2]
    return {
        "schema": "openecon.nardl-bounds-independent-numerical-oracle.v1",
        "private_research_only": True,
        "inferential_validity_established": False,
        "not_a_full_independent_size_replication": True,
        "construction": "Full affine unprofiled NumPy SVD with independent difference-ECM recursion and raw-sign reconstruction; shared index streams isolate arithmetic.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_source_sha256": hashlib.sha256(
            (root / "benchmarks/research/nardl_bounds.py").read_bytes()
        ).hexdigest(),
        "elapsed_seconds": time.perf_counter() - start,
        "records": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {"records": len(result["records"]), "elapsed_seconds": result["elapsed_seconds"]}
        )
    )


if __name__ == "__main__":
    main()
