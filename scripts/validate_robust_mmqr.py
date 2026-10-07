"""Development-only scientific oracle; NumPy/SciPy never enter estimator runtime.

Rebuilds the frozen published-author nlswork example from downloaded Stata
data, and computes covariance by a FULL sparse dummy-variable GMM Jacobian.
Run with --source /path/nlswork.dta --write to regenerate the fixture.
Without --write it verifies the checked-in CSV and frozen numeric results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "robust"
COLS = ["age", "ttl_exp", "tenure", "not_smsa", "south"]
TAUS = [0.1, 0.5, 0.9]


def independent_published_example(df):
    df = df.sort_values(["idcode", "year"], kind="stable")
    x = df[COLS].to_numpy(float)
    y = df.ln_wage.to_numpy(float)
    n, p = x.shape
    codes = pd.factorize(df.idcode)[0]
    G = codes.max() + 1
    counts = np.bincount(codes)
    D = sparse.csc_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, G))
    L = sparse.hstack([sparse.csc_matrix(x), D], format="csc")
    d = p + G
    # Sparse normal equations are a development oracle; runtime uses scaled QR.
    cross = (L.T @ L).tocsc()
    factor = splu(cross)
    loc = factor.solve(np.asarray(L.T @ y).ravel())
    r = y - L @ loc
    lam = np.mean(r >= 0)
    h = 2 * ((r >= 0) - lam)
    a = r * h
    sc = factor.solve(np.asarray(L.T @ a).ravel())
    s = L @ sc
    v = a - s
    u = r / s
    qs = np.quantile(u, TAUS, method="averaged_inverted_cdf")
    spread = min(
        np.std(u, ddof=1),
        np.subtract(*np.quantile(u, [0.75, 0.25], method="averaged_inverted_cdf")) / 1.349,
    )
    bandwidth = 0.9 * spread * n ** (-0.2)
    density = np.mean(norm.pdf((u[:, None] - qs) / bandwidth), axis=0) / bandwidth
    # Full moment derivative: all individual location/scale nuisance parameters.
    Qloc = np.array([f * np.asarray(L.T @ (1 / s)).ravel() for f in density])
    Qscale = Qloc * qs[:, None]
    A = sparse.bmat(
        [
            [cross, None, None],
            [L.T @ L.multiply(h[:, None]), cross, None],
            [sparse.csc_matrix(Qloc), sparse.csc_matrix(Qscale), sparse.diags(n * density)],
        ],
        format="csc",
    )
    scores = sparse.hstack(
        [
            L.multiply(r[:, None]),
            L.multiply(v[:, None]),
            sparse.csc_matrix(np.array(TAUS)[None, :] - (u[:, None] <= qs)),
        ],
        format="csc",
    )
    group_scores = (D.T @ scores).toarray()
    cluster_influence = splu(A).solve(group_scores.T).T
    selector = [*range(p), *range(d, d + p), *range(2 * d, 2 * d + len(TAUS))]
    reduced = cluster_influence[:, selector]
    joint = reduced.T @ reduced * G / (G - 1) * (n - 1) / (n - G - p)
    transform = np.zeros((len(TAUS) * p, 2 * p + len(TAUS)))
    for j, q in enumerate(qs):
        transform[j * p : (j + 1) * p, :p] = np.eye(p)
        transform[j * p : (j + 1) * p, p : 2 * p] = q * np.eye(p)
        transform[j * p : (j + 1) * p, 2 * p + j] = sc[:p]
    b = np.concatenate([loc[:p] + q * sc[:p] for q in qs])
    return {
        "nobs": n,
        "n_groups": int(G),
        "t_min": int(counts.min()),
        "t_max": int(counts.max()),
        "location": loc[:p].tolist(),
        "scale": sc[:p].tolist(),
        "error_quantiles": qs.tolist(),
        "coefficients": b.tolist(),
        "minimum_scale": float(s.min()),
        "density_bandwidth": float(bandwidth),
        "joint_moment_covariance": joint.tolist(),
        "covariance": (transform @ joint @ transform.T).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    csv = FIXTURES / "nlswork-mmqr.csv"
    reference = FIXTURES / "nlswork-mmqr-reference.json"
    if args.write:
        if args.source is None:
            parser.error("--write requires the original downloaded --source .dta")
        df = pd.read_stata(args.source, convert_categoricals=False)
        df["source_row"] = np.arange(len(df))
        df = df.loc[
            df.groupby("idcode").idcode.transform("count") >= 10,
            ["source_row", "idcode", "year", "ln_wage", *COLS],
        ].dropna()
        # Float64 decimal round trips preserve the original Stata float values.
        df.to_csv(csv, index=False, float_format="%.17g")
        # Read the exact frozen CSV to make regeneration independent of CSV formatting.
        values = independent_published_example(pd.read_csv(csv, float_precision="round_trip"))
        values.update(
            {
                "predictors": COLS,
                "quantiles": TAUS,
                "selected_csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
                "source_dta_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "source_dta_url": "https://www.stata-press.com/data/r17/nlswork.dta",
                "author_example_url": "https://jmcss.som.surrey.ac.uk/MM-QR-JK.do",
                "author_command_url": "http://fmwww.bc.edu/repec/bocode/x/xtqreg.ado",
                "author_example_selection": "count(idcode)>=10 before complete-case selection; ln_w abbreviation resolves ln_wage; age ttl_exp tenure not_smsa south; quantiles .1 .5 .9",
                "author_example_sha256": "2566274366e81df51d9ebcf97e84272f59233af431ef8005673cfa0bcf5ce528",
                "author_command_version": "1.5 (30 September 2021)",
                "author_command_sha256": "6bda9254042ea00595886288a49b12f5aafef92625fa4d9a8c76e6fb0ae23989",
                "reference_engine": "independent NumPy/SciPy full sparse dummy GMM",
                "actual_stata_execution": False,
                "stata_covariance_parity": False,
                "reference_limits": "Published-author data/command replication, not execution in Stata. Covariance uses a joint profile GMM system and Gaussian KDE, not xtqreg 1.5 covariance/density. Fixed-T bias not corrected. Stata sample data are demonstration data only.",
            }
        )
        reference.write_text(json.dumps(values, indent=2) + "\n")
    else:
        frozen = json.loads(reference.read_text())
        fresh = independent_published_example(pd.read_csv(csv, float_precision="round_trip"))
        assert hashlib.sha256(csv.read_bytes()).hexdigest() == frozen["selected_csv_sha256"]
        for field in [
            "location",
            "scale",
            "error_quantiles",
            "coefficients",
            "joint_moment_covariance",
            "covariance",
        ]:
            np.testing.assert_allclose(fresh[field], frozen[field], rtol=1e-10, atol=1e-12)
        print(
            json.dumps(
                {
                    "passed": True,
                    "nobs": fresh["nobs"],
                    "n_groups": fresh["n_groups"],
                    "reference": "published-author example, independent sparse dummy moments",
                    "actual_stata_execution": False,
                }
            )
        )


if __name__ == "__main__":
    main()
