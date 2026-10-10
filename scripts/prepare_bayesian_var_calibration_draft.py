"""Deterministically prepare prospective acceptance sets; never fit or sample."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from fractions import Fraction


def accepted(n, probability, level):
    masses = [
        Fraction(math.comb(n, k)) * probability**k * (1 - probability) ** (n - k)
        for k in range(n + 1)
    ]
    return [k for k, mass in enumerate(masses) if sum(v for v in masses if v <= mass) >= level]


def explicit_prior(m, p, version):
    k = m * p + 1
    mean = [[0.0] * m for _ in range(k)]
    for j in range(m):
        mean[1 + j][j] = 0.45
        if p >= 2:
            mean[1 + m + j][j] = 0.1
    diagonal = [0.2 if version == 0 else 0.3] + [
        (0.025 if version == 0 else 0.06) / lag**2 for lag in range(1, p + 1) for _ in range(m)
    ]
    row = [
        [
            (diagonal[i] if i == j else 0.0) + (0.005 / (1 + (i - j) ** 2) if version else 0.0)
            for j in range(k)
        ]
        for i in range(k)
    ]
    scale = [
        [
            (0.32 if i == j else 0.0) + 0.08 if version == 0 else (0.5 if i == j else 0.0) + 0.1
            for j in range(m)
        ]
        for i in range(m)
    ]
    return {
        "mean": mean,
        "row_scale": row,
        "innovation_scale": scale,
        "degrees_of_freedom": float(m + (5 if version == 0 else 8)),
    }


def draft():
    cells = []
    for version in range(2):
        for m, p, n in ((2, 1, 40), (3, 2, 80), (4, 4, 160), (2, 4, 20)):
            cell = len(cells)
            k = m * p + 1
            cells.append(
                {
                    "cell": cell,
                    "m": m,
                    "p": p,
                    "n_original": n,
                    "n_response": n - p,
                    "intercept": True,
                    "prior": explicit_prior(m, p, version),
                    "fixed_conditioning_rows": [
                        [0.1 * (row + 1) / (j + 1) for j in range(m)] for row in range(p)
                    ],
                    "rank_targets": k * m + m * (m + 1) // 2 + 3,
                    "exact_coverage_targets": k * m + m + 2,
                    "truth_seed_formula": f"202610090000 + {cell}*1024 + replicate",
                    "posterior_seed_formula": f"202610100000 + {cell}*1024 + replicate",
                }
            )
    rank_targets = sum(c["rank_targets"] for c in cells)
    rank_gates = 8 * rank_targets
    coverage_gates = sum(c["exact_coverage_targets"] for c in cells)
    family = rank_gates + coverage_gates
    alpha = Fraction(1, 100)
    threshold = alpha / family
    root = pathlib.Path(__file__).resolve().parents[1]
    paths = sorted(root.glob("src/openecon/econometrics/bayesian_var/*.py")) + [
        pathlib.Path(__file__).resolve()
    ]
    return {
        "schema": "openecon.bvar_sbc.prospective_draft.v1",
        "status": "DRAFT_REQUIRES_ROOT_REVIEW_NO_OUTCOMES",
        "cells": cells,
        "replicates_per_cell": 128,
        "total_fits": 1024,
        "posterior_draws_per_fit": 128,
        "bins": 8,
        "ranks": "all column-major B; lower-column-major vech Sigma; rank-one, nonrank-one and B/Sigma coupled functional",
        "extra_rank_functionals": {
            "rank_one": "l_i=(-1)^i/(i+1); r_j=(j+1)/m; l'B r",
            "nonrank_one": "B[0,0]+B[1,1] (contrast matrix rank two)",
            "coupled": "(B[1,0]-M0[1,0])**2/Sigma[0,0]",
        },
        "rank_transform": "rank = number strictly below truth, uniformly randomized over exact ties; independent U; (rank+U)/129; floor(8*u)",
        "rank_reference": "exact uniform continuous randomized ranks under prior predictive exchangeability; each bin probability 1/8",
        "data_gp": "independent SciPy IW prior truth Sigma; NumPy matrix-normal truth B conditional on that Sigma; fixed first p rows; recursive Gaussian VAR; no stability conditioning",
        "independent_truth_rng": "NumPy PCG64 + SciPy invwishart native Axen sampler; runtime/source pinned before launch",
        "production_rng": "native Torch gamma/Bartlett + matrix-normal, local generator; runtime/source pinned before launch",
        "saved_random_inputs": "truth Sigma, truth matrix-normal Z, each DGP innovation Z, held-out one-step Z, all rank/tie uniforms, all posterior Bartlett/Z primitives",
        "exact_coverage": "95% coefficient scalar t d=nuN-m+1; 95% diagonal Sigma inverse-gamma(shape=d/2,scale=SNii/2); 95% rank-one scalar t; 95% held-out one-step m*F(m,d) ellipsoid",
        "joint_predictive_ellipsoid": "held-out next y from same true B/Sigma after full conditioning data; scale (1+x'VNx)SN/d; Mahalanobis/m <= F_isf(.05,m,d)",
        "family": {
            "rank_targets": rank_targets,
            "rank_bin_gates": rank_gates,
            "exact_coverage_gates": coverage_gates,
            "total_gates": family,
            "control": "fixed Bonferroni family alpha .01 (prospective correction of adaptive-Holm draft to explicit fixed accepted sets)",
            "alpha": "1/100",
            "individual_exact_level": str(threshold),
        },
        "accepted_rank_bin_counts": accepted(128, Fraction(1, 8), threshold),
        "accepted_exact_coverage_counts": accepted(128, Fraction(19, 20), threshold),
        "binomial_test": "two-sided exact probability-ordering test, exact rational PMF; accept pvalue >= 1/(100*family)",
        "failure_rules": "zero computational failures required; all failures/unstable truth and posterior draws retained in denominators; no reseeding, resampling, clipping, early statistical stopping or gate relaxation",
        "negative_controls": {
            "point_mass_coefficients": "replace all 128 coefficient draws by MN; retain same prior truth/data/posterior Sigma draws; all continuous scalar B ranks fall in bin0/bin7, so one count >=64 is necessarily outside the fixed accepted set; require rejection for every cell, no tuning",
            "matrix_t_substitution": "fixed independent density oracle already detects generic elliptical multivariate-t substitution; no claim that 128 ranks identify every copula defect",
            "cross_equation_omission": "fixed full-covariance oracle detects zeroing cross-equation blocks; retain existing deterministic case rather than claim unregistered MC power",
            "wrong_df": "fixed exact rank-one t oracle detects d=nuN instead of nuN-m+1",
            "future_innovations_omitted": "fixed common-primitive full outcome path differs from conditional-mean path; independent recursion detects omission",
        },
        "state": "persist complete 1024 source/prior/posterior/draw/held-out/rank records; parent method acceptance is not implied",
        "resource_draft": {
            "native_workers": 1,
            "native_threads": 1,
            "per_fit_max_work": 100000000,
            "per_fit_max_bytes": 134217728,
            "max_total_work_upper_bound": 102400000000,
            "disk_raw_and_archives_reservation_bytes": 4294967296,
            "min_free_before_launch_bytes": 8589934592,
            "launch_needs_resource_and_disk_receipt": True,
        },
        "sources": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "author_execution": "original MATLAB source archived separately, not executed; no vendor acceptance",
    }


if __name__ == "__main__":
    destination = pathlib.Path("/tmp/market-734-prospective-sbc-draft-v2.json")
    with destination.open("x") as handle:
        json.dump(draft(), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(destination)
