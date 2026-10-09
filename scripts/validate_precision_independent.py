"""Reproduce independent development-oracle checks for prospective precision.

Run from the repository with the development environment:
    .venv/bin/python scripts/validate_precision_independent.py

SciPy is a development reference only; these public methods use native kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy import stats
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.nonparametric.distribution import clopper_pearson
from openecon.econometrics.stats import planning_precision as methods
from openecon.engines.distributions import chi2_isf, chi2_ppf, f_ppf, t_isf

ROOT = Path(__file__).resolve().parents[1]


def cp_law(size, confidence):
    counts = np.arange(size + 1)
    low, high = np.zeros(size + 1), np.ones(size + 1)
    low[1:] = stats.beta.ppf((1 - confidence) / 2, counts[1:], size - counts[1:] + 1)
    high[:-1] = stats.beta.isf((1 - confidence) / 2, counts[:-1] + 1, size - counts[:-1])
    return counts, low, high


def minimum_check(kind, design):
    expected = None
    if kind == "binomial":
        p, confidence, assurance, width = design
        maximum = 80
        for size in range(1, maximum + 1):
            counts, low, high = cp_law(size, confidence)
            mass = stats.binom.pmf(counts, size, p)[high - low <= width].sum()
            if mass >= assurance:
                expected = size
                break
        def call():
            return methods.precision_binomial(
                p=p, width=width, confidence=confidence, assurance=assurance, max_n=maximum
            )
    elif kind == "poisson":
        rate, confidence, assurance, width, exposure = design
        maximum = 60
        counts = np.arange(500)
        for size in range(1, maximum + 1):
            total = size * exposure
            low = np.zeros(500)
            low[1:] = stats.chi2.ppf((1 - confidence) / 2, 2 * counts[1:]) / (2 * total)
            high = stats.chi2.isf((1 - confidence) / 2, 2 * (counts + 1)) / (2 * total)
            mass = stats.poisson.pmf(counts, rate * total)[high - low <= width].sum()
            if mass >= assurance:
                expected = size
                break
        def call():
            return methods.precision_poisson(
                rate=rate, width=width, exposure_per_unit=exposure,
                confidence=confidence, assurance=assurance, max_n=maximum,
            )
    else:
        ratio, confidence, assurance, width = design
        maximum = 300
        for size in range(2, maximum + 1):
            second = int((Decimal(str(ratio)) * size).to_integral_value(rounding=ROUND_CEILING))
            if not 2 <= second <= maximum:
                continue
            df = size + second - 2
            quantile = 2 * stats.t.isf((1 - confidence) / 2, df) * 1.3 * math.sqrt(
                (1 / size + 1 / second) * stats.chi2.ppf(assurance, df) / df
            )
            if quantile <= width:
                expected = size
                break
        def call():
            return methods.precision_twomeans_unknown(
                sd=1.3, width=width, ratio=ratio, confidence=confidence,
                assurance=assurance, max_n=maximum,
            )
    record = {"method": kind, "design": design, "expected_minimum_n": expected}
    try:
        actual = int(call()["plan"].iloc[0].n)
        record.update(actual_minimum_n=actual, passed=actual == expected)
    except AnalysisError as error:
        record.update(refusal=error.code, passed=expected is None and error.code == "resource_limit")
    return record


def validate():
    torch.set_num_threads(1)
    report = {
        "schema": "openecon.precision-independent-review.v1",
        "scope": "Declared sampling laws and bounded float64 domain; development SciPy oracles, not interval-arithmetic certification",
        "environment": {
            "python": platform.python_version(), "scipy": scipy.__version__,
            "numpy": np.__version__, "torch": torch.__version__,
        },
        "source_sha256": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "src/openecon/econometrics/stats/planning_precision.py",
                "src/openecon/econometrics/stats/planning.py",
                "src/openecon/engines/distributions.py",
                "src/openecon/econometrics/nonparametric/distribution.py",
                "src/openecon/econometrics/summary_state.py",
                "tests/test_planning_precision.py",
                "scripts/validate_precision_independent.py",
            )
        },
        "primitive_checks": {}, "complete_laws": [], "minimum_n_checks": [],
    }
    dfs = [2, 3, 8, 50, 200, 1000, 19998, 200000]
    for name, function, reference, probabilities in (
        ("chi2_ppf", chi2_ppf, stats.chi2.ppf, [5e-9, 1e-6, .01, .25, .499999999, .5, .9, .999999]),
        ("chi2_isf", chi2_isf, stats.chi2.isf, [5e-9, 1e-6, .01, .25, .499999999, .5, .9, .999999]),
        ("t_isf", t_isf, stats.t.isf, [5e-9, 1e-6, .01, .025, .1, .2, .24999995, .25]),
    ):
        worst, at = 0.0, None
        for df in dfs:
            for probability in probabilities:
                actual, expected = function(probability, df), reference(probability, df)
                error = abs(actual - expected) / abs(expected)
                if error > worst:
                    worst, at = error, [df, probability, actual, expected]
        report["primitive_checks"][name] = {
            "comparisons": len(dfs) * len(probabilities),
            "maximum_relative_error": float(worst), "worst_case": at,
            "passed": bool(worst < 2e-10),
        }
    worst, at, comparisons = 0.0, None, 0
    for size in [1, 2, 10, 101, 1000, 2000]:
        for confidence in [.5000001, .95, .99999999]:
            for count in sorted({0, 1, size // 4, size // 2}):
                low, high = clopper_pearson(count, size, 1 - confidence)
                if count:
                    f = f_ppf((1 - confidence) / 2, 2 * count, 2 * (size - count + 1))
                    low = count * f / (size - count + 1 + count * f)
                expected_low = 0 if count == 0 else stats.beta.ppf(
                    (1 - confidence) / 2, count, size - count + 1
                )
                expected_high = stats.beta.isf((1 - confidence) / 2, count + 1, size - count)
                error = max(abs(low - expected_low), abs(high - expected_high))
                if error > worst:
                    worst, at = error, [size, confidence, count]
                comparisons += 2
    report["primitive_checks"]["CP_endpoints"] = {
        "comparisons": comparisons, "maximum_absolute_error": float(worst),
        "worst_case": at, "passed": bool(worst < 3e-12),
    }
    for size, p, confidence, assurance in [
        (2000, .5, .99999999, .999999), (177, 1e-12, .5000001, 1e-6),
        (188, .99, .99999999, .5), (150, .001, .95, .999999),
    ]:
        result = methods.precision_binomial(
            p=p, n=size, max_n=size, confidence=confidence, assurance=assurance
        )
        law, row = result["width_distribution"], result["plan"].iloc[0]
        mass = stats.binom.pmf(np.arange(size + 1), size, p)
        expected = float(mass[law.qualifies.to_numpy(dtype=bool)].sum())
        error = float(np.max(np.abs(law.probability.to_numpy() - mass)))
        report["complete_laws"].append({
            "method": "binomial", "design": [size, p, confidence, assurance],
            "count_cells": len(law), "pmf_maximum_absolute_error": error,
            "reference_width_mass": expected, "reported_width_mass": row.width_probability,
            "probability_bounds": [row.probability_lower, row.probability_upper],
            "passed": bool(error < 3e-12 and row.probability_lower <= expected <= row.probability_upper),
        })
    for mean, confidence, assurance in [
        (1e-300, .99999999, .999999), (250, .99999999, .999999), (1100, .5000001, 1e-6),
    ]:
        result = methods.precision_poisson(
            rate=mean, n=1, max_n=1, confidence=confidence, assurance=assurance
        )
        law, row = result["width_distribution"], result["plan"].iloc[0]
        mass = stats.poisson.pmf(law["count"].to_numpy(dtype=int), mean)
        expected = float(mass[law.qualifies.to_numpy(dtype=bool)].sum())
        error = float(np.max(np.abs(law.probability.to_numpy() - mass)))
        tail_error = abs(stats.poisson.sf(int(row.count_cutoff), mean) - row.omitted_mass)
        report["complete_laws"].append({
            "method": "poisson", "design": [mean, confidence, assurance],
            "count_cells": len(law), "pmf_maximum_absolute_error": error,
            "tail_absolute_error": float(tail_error), "reference_width_mass": expected,
            "reported_width_mass": row.width_probability,
            "probability_bounds": [row.probability_lower, row.probability_upper],
            "passed": bool(error < 3e-12 and tail_error < 3e-14 and row.probability_lower <= expected <= row.probability_upper),
        })
    for method, designs in (
        ("binomial", [(0, .95, .9, .2), (1, .999999, .9, .4), (.5, .51, .5, .55),
                      (.01, .95, .95, .3), (.3, .95, .9, .4), (.5, .999999, .05, .9)]),
        ("poisson", [(0, .95, .9, .1, 1), (.02, .99999999, .9, 8, 1),
                     (.1, .95, .05, 1, 1), (2, .51, .95, 2, 1), (1, .95, .95, 2, .5)]),
        ("normal", [(.01, .95, .9, 3), (.99, .99999999, 1e-6, 2),
                    (3.71, .95, .99, 1), (.2, .51, 1e-6, .2), (100, .95, .9, 2)]),
    ):
        report["minimum_n_checks"].extend(minimum_check(method, design) for design in designs)
    records = [*report["primitive_checks"].values(), *report["complete_laws"], *report["minimum_n_checks"]]
    report["failures"] = [record for record in records if not record["passed"]]
    report["counts"] = {
        "primitive_comparisons": sum(row["comparisons"] for row in report["primitive_checks"].values()),
        "complete_laws": len(report["complete_laws"]),
        "complete_law_count_cells": sum(row["count_cells"] for row in report["complete_laws"]),
        "minimum_n_designs": len(report["minimum_n_checks"]), "failures": len(report["failures"]),
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/evidence/prospective-extensions-eight-2026-10-07/precision-independent-review.json")
    arguments = parser.parse_args()
    output = validate()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["counts"], sort_keys=True))
    raise SystemExit(bool(output["failures"]))
