#!/usr/bin/env python3
"""Independent scalar CLR geometry/probability audit; no Torch/OpenEcon imports.

NumPy least squares and generalized eigenvalues, SciPy adaptive quadrature and
Brent inversion implement the published equations independently. This is a
mathematical reference, not an original-author or licensed-vendor execution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.stats import chi2

SOURCES = {
    "inversion": "https://economics.mit.edu/sites/default/files/publications/thirdsubmission.pdf",
    "original_author_example": "https://economics.mit.edu/sites/default/files/publications/944.pdf",
    "original_author_software": "https://sites.google.com/site/moreiramarceloj/statistical-packages",
}


def tail(lr, qt, k):
    if lr <= 0:
        return 1.0
    if k == 1 or qt == 0:
        return float(chi2.sf(lr, k if qt == 0 else 1))
    a = (k - 1) / 2
    normalizer = 2 * math.exp(math.lgamma(k / 2) - math.lgamma(a) - math.lgamma(0.5))
    ratio = lr / (lr + qt)

    if lr < 0.01:
        # Integrate the lower-tail complement so an almost constant upper tail
        # cannot hide a narrow endpoint contribution. These two transition
        # points are independent of the native geometric panel construction.
        def complement(u):
            s, c = math.cos(u), math.sin(u)
            return normalizer * s ** (k - 2) * chi2.cdf(lr / (c * c + ratio * s * s), k)

        points = sorted(
            {min(math.pi / 2, math.sqrt(lr)), min(math.pi / 2, math.sqrt(ratio))}
            - {0.0, math.pi / 2}
        )
        return float(
            1
            - quad(
                complement, 0, math.pi / 2, epsabs=2e-14, epsrel=1e-11, points=points, limit=500
            )[0]
        )

    def integrand(theta):
        s, c = math.sin(theta), math.cos(theta)
        return normalizer * s ** (k - 2) * chi2.sf(lr / (c * c + ratio * s * s), k)

    return float(quad(integrand, 0, math.pi / 2, epsabs=1e-13, epsrel=1e-12, limit=300)[0])


def reference(state):
    cfg, spec = state["options"], state["spec"]
    rows = np.asarray([state["source"]["values"][i] for i in state["positions"]], dtype=float)
    n = len(rows)
    y = rows[:, :2]
    c = rows[:, 2 : 2 + len(spec["x"])]
    z = rows[:, 2 + len(spec["x"]) :]
    if cfg["intercept"]:
        c = np.column_stack((np.ones(n), c))
    # Independent least-squares projection, not the native QR path.
    if c.shape[1]:
        y = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
        z = z - c @ np.linalg.lstsq(c, z, rcond=None)[0]
    projected = z @ np.linalg.lstsq(z, y, rcond=None)[0]
    gram = projected.T @ projected
    omega = (
        np.asarray(cfg["omega"])
        if cfg["omega"] is not None
        else (y - projected).T @ (y - projected) / (n - c.shape[1] - z.shape[1])
    )
    eigenvalues, eigenvectors = eigh(gram, omega)
    maximum = float(eigenvalues[-1])
    gap = float(eigenvalues[-1] - eigenvalues[0])
    # Generalized eigenvectors satisfy V'ΩV=I and Ω^-1=VV'.
    directions = eigenvectors.T
    k, alpha = z.shape[1], 1 - cfg["confidence"]
    if maximum <= chi2.isf(alpha, k):
        lr, intervals, topology = None, [[None, None]], "all_real"
    else:
        lr = (
            float(chi2.isf(alpha, 1))
            if k == 1
            else float(
                brentq(
                    lambda v: tail(v, maximum - v, k) - alpha,
                    chi2.isf(alpha, 1),
                    chi2.isf(alpha, k),
                    xtol=2e-13,
                    rtol=1e-13,
                )
            )
        )
        if gap < lr:
            intervals, topology = [[None, None]], "all_real"
        else:
            h = directions.T @ np.diag([lr - gap, lr]) @ directions
            roots = sorted(np.roots([h[0, 0], 2 * h[0, 1], h[1, 1]]).tolist())
            topology = "two_unbounded_rays" if h[0, 0] > 0 else "bounded_interval"
            intervals = [[None, roots[0]], [roots[1], None]] if h[0, 0] > 0 else [roots]
    return dict(
        omega=omega,
        gram=gram,
        eigenvalues=eigenvalues,
        directions=directions,
        maximum=maximum,
        gap=gap,
        lr=lr,
        intervals=intervals,
        topology=topology,
        k=k,
    )


def null_reference(ref, null):
    d = ref["directions"] @ np.asarray([null, 1.0])
    d /= np.max(np.abs(d))
    lr = ref["gap"] * d[0] ** 2 / (d @ d)
    qt = max(0.0, ref["maximum"] - lr)
    return dict(
        statistic=float(lr),
        conditioning_statistic=float(qt),
        conditional_p_value=tail(float(lr), float(qt), ref["k"]),
    )


def relative_matrix_error(actual, expected):
    expected = np.asarray(expected)
    scale = np.sqrt(np.abs(np.diag(expected)))
    return float(
        np.max(
            np.abs(np.asarray(actual) - expected)
            / np.maximum(np.abs(expected), np.outer(scale, scale))
        )
    )


def audit(document):
    cases, failures = [], []
    for i, state in enumerate(document["states"]):
        ref = reference(state)
        g, s = state["geometry"], state["solution"]
        errors = {
            "omega_relative": relative_matrix_error(g["reduced_form_covariance"], ref["omega"]),
            "gram_relative": relative_matrix_error(g["projected_gram"], ref["gram"]),
            "maximum_relative": abs(g["maximum_eigenvalue"] - ref["maximum"])
            / max(1, ref["maximum"]),
            "gap_relative": abs(g["eigen_gap"] - ref["gap"]) / max(1, ref["gap"]),
        }
        if s["topology"] != ref["topology"]:
            failures.append(f"case {i}: topology")
        native_endpoints = [v for row in s["intervals"] for v in row if v is not None]
        ref_endpoints = [v for row in ref["intervals"] for v in row if v is not None]
        if len(native_endpoints) != len(ref_endpoints):
            failures.append(f"case {i}: endpoint count")
        errors["endpoint_normalized"] = max(
            (
                abs(a - b) / max(abs(b), g["coefficient_unit"])
                for a, b in zip(native_endpoints, ref_endpoints)
            ),
            default=0,
        )
        if max(errors.values()) > 2e-8:
            failures.append(f"case {i}: numerical reference")
        cases.append(dict(case=i, topology=ref["topology"], **errors))
    tests = []
    for i, state in enumerate(document.get("tests", [])):
        ref = reference(state["target"])
        expected = null_reference(ref, state["test"]["null"])
        errors = {k: abs(state["test"][k] - v) / max(1, abs(v)) for k, v in expected.items()}
        if max(errors.values()) > 2e-9:
            failures.append(f"test {i}: conditional probability")
        tests.append(dict(case=i, **errors))
    return dict(
        schema="openecon.scalar-clr-independent-audit.v1",
        passed=not failures,
        cases=cases,
        tests=tests,
        failures=failures,
        sources=SOURCES,
        reference="independent published-equation NumPy/SciPy whole-line and probability audit",
        original_author_code=dict(
            status="unavailable",
            reason="official st0033_2 software and hsng2 endpoints returned HTTP 403 in this environment; no original-author execution receipt",
        ),
        printed_author_table=dict(
            status="separate_open_gate",
            reason="printed rounded housing example is not a complete precision fixture",
        ),
        licensed_vendor=dict(status="not_run"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.input.read_text()
    # Native console output may precede the standalone JSON result line.
    document = None
    for line in reversed(text.splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and "states" in candidate:
            document = candidate
            break
    if document is None:
        document = json.loads(text)
    receipt = audit(document)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(
                passed=receipt["passed"],
                cases=len(receipt["cases"]),
                tests=len(receipt["tests"]),
                failures=receipt["failures"],
            )
        )
    )
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
