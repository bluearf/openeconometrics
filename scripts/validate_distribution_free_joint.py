"""Frozen, independent finite-sample checks for eight joint inference methods.

SciPy/NumPy are development oracles only. Runtime methods use CPU float64
Torch. Exact conditional orbits, exact count-law coverage, native-call Monte
Carlo and lookup-calibrated coverage are reported as distinct evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/distribution-free-eight-2026-10-07"
PROTOCOL = EVIDENCE / "protocol.json"
METHODS = ["mean_sign_stepdown", "mean_permutation_stepdown", "simultaneous_dkw_band",
           "simultaneous_quantile_ci", "simultaneous_proportion_ci", "multinomial_region",
           "hoeffding_mean_ci", "empirical_bernstein_mean_ci"]
SOURCE_FILES = [
    "src/openecon/econometrics/postest/randomization_joint.py",
    "src/openecon/econometrics/postest/distribution_free_joint.py",
    "src/openecon/econometrics/postest/bounded_joint.py",
    "src/openecon/econometrics/postest/multiple.py",
    "src/openecon/econometrics/nonparametric/distribution.py",
    "src/openecon/econometrics/stats/planning.py",
    "src/openecon/engines/distributions.py",
    "src/openecon/econometrics/summary_state.py",
]
PRIMARY_SOURCES = {
    "randomization": "https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffd823d949/jasa.pdf",
    "dkw_massart": "https://doi.org/10.1214/aop/1176990746",
    "quantiles": "https://www.stat.berkeley.edu/~stark/Teach/S240/Notes/ch5.htm",
    "binomial_limits": "https://itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm",
    "bounded_means": "https://www.cs.mcgill.ca/~colt2009/papers/012.pdf",
}


def protocol_definition():
    return {
        "schema": 1,
        "methods": METHODS,
        "primary_sources": PRIMARY_SOURCES,
        "alphas": [.01, .05, .10],
        "tails": ["two-sided", "greater", "less"],
        "exact_orbits": {
            "sign_rows": 8, "permutation_rows": 8, "group_size": 4, "dimensions": 3,
            "null_configurations": ["full", "partial"],
            "partial_false_coordinate": 2, "false_shift": 12.,
            "base_entry": "1+((i+1)*(j+3)+i*i+2*j)%19+(j+1)*(i+1)/16",
            "native_calibration_alpha": .05,
            "independent_reference": "Pure-Python exhaustive transformations, invariant norms, inclusive maxT and monotone tie-group stepdown",
            "reference_equality_tolerance": 1e-12,
            "maximum_statistic_absolute_error": 2e-12,
            "maximum_p_absolute_error": 2e-12,
            "size_gate": "Exact uniform-orbit FWER <= alpha; no Monte Carlo tolerance",
            "scope": "Conditional fixed sign/assignment orbits; not arbitrary mean-equality validity",
        },
        "exact_counts": {
            "quantile_n": 8, "ordinal_support": [0., 1., 2.],
            "ordinal_probabilities": [.2, .5, .3],
            "quantile_probabilities": [.2, .5, .75], "lower_quantile_truth": [0., 1., 2.],
            "multinomial_n": 8, "multinomial_truths": [[.05, .30, .65], [0., .5, .5]],
            "bernoulli_n": 6, "bernoulli_truth": [.25, .5, .75],
            "row_patterns": [[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0]],
            "pattern_probabilities": [.25, .25, .25, .25],
            "weight_arithmetic": "Exact fractions and multinomial coefficients; probabilities supplied as decimal rational values",
            "size_gate": "Exact family noncoverage <= alpha, permitting at most 2e-12 accumulation/reporting tolerance",
            "maximum_endpoint_absolute_error": 2e-9,
        },
        "coverage_screens": {
            "families_per_cell": 1000, "dimensions": 3,
            "dependence": ["independent", "comonotonic", "mixed_opposite"],
            "dkw_n": 48, "bounded_n": 128,
            "supports": [[-2., 3.], [1., 1.5], [-100., 2.]],
            "bernoulli_truth": [.2, .5, .8],
            "partial_false_coordinate": 1, "partial_null_probability": .9,
            "dkw_partial_cdf_shift_in_support_units": .4,
            "seed_base": 260700000,
            "gate": "family_noncoverage <= alpha + max(.01,6*sqrt(alpha*(1-alpha)/1000)); failures must be zero",
            "uncertainty": "Wilson 95% observation Monte Carlo intervals; no nominal exact-coverage claim",
            "dkw_execution": "Every sampled family calls the native DKW API; true-CDF supremum independently checks left and right empirical limits",
            "bounded_execution": "Complete native calibration for all k=0..128, fixed 3-member family/supports; independent Bernoulli families lookup their calibrated native endpoints",
            "failure_policy": "All planned families remain in denominator, failed evaluations count as noncoverage; no retries or replacements",
            "power_policy": "False-null rejection rates descriptive only; no minimum-power gate",
        },
        "limits": {
            "evidence": "Source CPU only; frozen runtime and installed application require separate root acceptance",
            "external_libraries": "SciPy/NumPy only in this independent development validator",
            "statistical_scope": "Declared fixed family, fixed n, valid true-subset transformations or iid marginals and prespecified supports",
            "not_claimed": ["arbitrary mean-equality permutation", "unknown supports", "optional stopping", "weights", "Dataset", "GPU", "global Stata parity"],
            "outputs": "JSON stores all family outcome flags and failure indices; hashes and seeds permit replay",
        },
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compositions(n, parts):
    if parts == 1:
        yield (n,)
    else:
        for value in range(n + 1):
            for tail in compositions(n - value, parts - 1):
                yield (value, *tail)


def mass(counts, probabilities):
    n = sum(counts)
    coefficient = math.factorial(n)
    for count in counts:
        coefficient //= math.factorial(count)
    value = Fraction(coefficient)
    for count, probability in zip(counts, probabilities, strict=True):
        value *= Fraction(str(probability)) ** count
    return value


def sign_scores(data, signs):
    return [math.fsum(sign * row[j] for sign, row in zip(signs, data, strict=True))
            / math.sqrt(math.fsum(row[j] ** 2 for row in data)) for j in range(len(data[0]))]


def permutation_scores(data, group0):
    n, k = len(data), len(group0)
    selected = set(group0)
    result = []
    for j in range(len(data[0])):
        center = math.fsum(row[j] for row in data) / n
        rms = math.sqrt(math.fsum((row[j] - center) ** 2 for row in data) / n)
        a = math.fsum(data[i][j] for i in selected) / k
        b = math.fsum(data[i][j] for i in range(n) if i not in selected) / (n - k)
        result.append((a - b) / (rms * math.sqrt(1 / k + 1 / (n - k))))
    return result


def reference_stepdown(observed, joint, tail, tolerance):
    transform = abs if tail == "two-sided" else (lambda x: -x) if tail == "less" else (lambda x: x)
    observed = list(map(transform, observed))
    joint = [list(map(transform, row)) for row in joint]
    order = sorted(range(len(observed)), key=lambda j: -observed[j])
    adjusted = [None] * len(observed)
    raw = [sum(row[j] >= value - tolerance for row in joint) / len(joint)
           for j, value in enumerate(observed)]
    previous, start = 0., 0
    while start < len(order):
        end = start + 1
        while end < len(order) and observed[order[end]] == observed[order[start]]:
            end += 1
        cutoff = observed[order[start]]
        count = sum(max(row[j] for j in order[start:]) >= cutoff - tolerance for row in joint)
        previous = max(previous, count / len(joint))
        for j in order[start:end]:
            adjusted[j] = previous
        start = end
    return raw, adjusted


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0., center - half), min(1., center + half)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-only", action="store_true")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "scientific-validation.json")
    args = parser.parse_args()
    if args.protocol_only:
        args.protocol.parent.mkdir(parents=True, exist_ok=True)
        with args.protocol.open("x") as handle:
            json.dump(protocol_definition(), handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"protocol": str(args.protocol), "sha256": digest(args.protocol)}), flush=True)
        return 0
    if args.output.exists():
        parser.error("Choose a new output path; prior outcomes may not be overwritten.")
    protocol = json.loads(args.protocol.read_text())
    if protocol != protocol_definition():
        parser.error("Frozen protocol differs from this validator's declared procedure.")
    import numpy as np
    import pandas as pd
    from scipy import stats
    import torch

    from openecon.econometrics.postest.bounded_joint import empirical_bernstein_mean_ci, hoeffding_mean_ci
    from openecon.econometrics.postest.distribution_free_joint import (
        multinomial_region, simultaneous_dkw_band, simultaneous_proportion_ci, simultaneous_quantile_ci,
    )
    from openecon.econometrics.postest.randomization_joint import mean_permutation_stepdown, mean_sign_stepdown
    from openecon.econometrics.summary_state import restore_summary, summary_state

    torch.set_num_threads(1)
    started = time.monotonic()
    start_hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    counts = Counter()
    cells, failures, persistence = [], [], []
    names = ["a", "b", "c"]

    def invoke(name, function, *a, **kw):
        counts[name] += 1
        return function(*a, **kw)

    def save_example(name, result):
        state = summary_state(result)
        restored = restore_summary(state)
        assert restored.attrs == result.attrs
        assert set(restored) == set(result)
        for key in result:
            pd.testing.assert_frame_equal(restored[key], result[key], check_dtype=False)
            latex = restored[key].to_latex()
            assert "\\begin{tabular}" in latex or "\\begin{longtable}" in latex
        persistence.append({"method": name, "complete_state_sha256": hashlib.sha256(state.encode()).hexdigest(),
                            "state_bytes": len(state.encode()), "all_tables_rows_restored": True,
                            "table_rows": {key: len(value) for key, value in result.items()},
                            "latex": True})

    def exact_orbits():
        conf = protocol["exact_orbits"]
        for method in ("mean_sign_stepdown", "mean_permutation_stepdown"):
            n = conf["sign_rows"] if method == "mean_sign_stepdown" else conf["permutation_rows"]
            base = [[1 + ((i + 1) * (j + 3) + i * i + 2 * j) % 19 + (j + 1) * (i + 1) / 16
                     for j in range(3)] for i in range(n)]
            orbit = list(itertools.product((-1, 1), repeat=n)) if method == "mean_sign_stepdown" \
                else list(itertools.combinations(range(n), conf["group_size"]))
            for null_case, tail in itertools.product(conf["null_configurations"], protocol["tails"]):
                flags = {alpha: [] for alpha in protocol["alphas"]}
                powers = {alpha: [] for alpha in protocol["alphas"]}
                errors, max_stat_error, max_p_error = [], 0., 0.
                for index, assignment in enumerate(orbit):
                    try:
                        if method == "mean_sign_stepdown":
                            data = [[assignment[i] * value + (conf["false_shift"] if null_case == "partial" and j == 2 else 0.)
                                     for j, value in enumerate(base[i])] for i in range(n)]
                            observed = sign_scores(data, [1] * n)
                            joint = [sign_scores(data, signs) for signs in orbit]
                            output = invoke(method, mean_sign_stepdown, pd.DataFrame(data, columns=names), names,
                                symmetry_model="Independent centrally symmetric joint true-null row vectors",
                                calibration="enumerated", tail=tail, alpha=conf["native_calibration_alpha"])
                        else:
                            chosen = set(assignment)
                            data = [[value + (conf["false_shift"] if null_case == "partial" and j == 2 and i in chosen else 0.)
                                     for j, value in enumerate(base[i])] for i in range(n)]
                            observed = permutation_scores(data, assignment)
                            joint = [permutation_scores(data, members) for members in orbit]
                            # Fixed group identities and direction. Conditional assignment
                            # orbits are equivalent to permuting iid pooled outcomes into
                            # this fixed A/B sample; first observed label is always A.
                            ordered = list(assignment) + [i for i in range(n) if i not in chosen]
                            frame = pd.DataFrame([data[i] for i in ordered], columns=names)
                            frame["group"] = ["A"] * len(assignment) + ["B"] * (n - len(assignment))
                            output = invoke(method, mean_permutation_stepdown, frame, names, "group",
                                exchangeability_model="Joint true-null pooled outcomes are label-exchangeable; fixed counts",
                                calibration="enumerated", tail=tail, alpha=conf["native_calibration_alpha"])
                        raw, adjusted = reference_stepdown(observed, joint, tail, conf["reference_equality_tolerance"])
                        native = output["tests"]
                        stat_error = max(abs(float(actual) - expected) for actual, expected in zip(native.statistic, observed))
                        p_error = max(abs(float(actual) - expected) for actual, expected in zip(native.adjusted_p_value, adjusted))
                        p_error = max(p_error, max(abs(float(actual) - expected) for actual, expected in zip(native.p_value, raw)))
                        max_stat_error, max_p_error = max(max_stat_error, stat_error), max(max_p_error, p_error)
                        assert stat_error <= conf["maximum_statistic_absolute_error"]
                        assert p_error <= conf["maximum_p_absolute_error"]
                        assert native.reject.tolist() == (native.adjusted_p_value <= conf["native_calibration_alpha"]).tolist()
                        if index == 0 and null_case == "full" and tail == "two-sided":
                            save_example(method, output)
                        true = range(3) if null_case == "full" else range(2)
                        for alpha in protocol["alphas"]:
                            flags[alpha].append(int(any(float(native.adjusted_p_value[j]) <= alpha for j in true)))
                            powers[alpha].append(int(null_case == "partial" and float(native.adjusted_p_value[2]) <= alpha))
                    except Exception as error:
                        errors.append({"orbit_position": index, "type": type(error).__name__, "message": str(error)})
                        for alpha in protocol["alphas"]:
                            flags[alpha].append(1)
                            powers[alpha].append(0)
                for alpha in protocol["alphas"]:
                    rate = sum(flags[alpha]) / len(orbit)
                    cells.append({"method": method, "kind": "exact conditional orbit", "null_case": null_case,
                        "tail": tail, "alpha": alpha, "observed_orbits": len(orbit), "null_transformations": len(orbit),
                        "family_rejections": sum(flags[alpha]), "family_rejection_rate": rate,
                        "false_null_rejections": sum(powers[alpha]), "outcome_flags": "".join(map(str, flags[alpha])),
                        "power_flags": "".join(map(str, powers[alpha])), "failures": errors,
                        "max_statistic_error": max_stat_error, "max_p_error": max_p_error,
                        "passed": not errors and Fraction(sum(flags[alpha]), len(orbit)) <= Fraction(str(alpha))})
                print(json.dumps({"method": method, "null_case": null_case, "tail": tail,
                                  "native_cases": len(orbit), "failures": len(errors)}), flush=True)

    def count_coverage():
        conf = protocol["exact_counts"]
        for alpha in protocol["alphas"]:
            for truth in conf["multinomial_truths"]:
                rejected, total, errors, maximum = Fraction(0), Fraction(0), [], 0.
                outcomes = []
                for index, values in enumerate(compositions(conf["multinomial_n"], 3)):
                    weight = mass(values, truth)
                    total += weight
                    try:
                        output = invoke("multinomial_region", multinomial_region, values, sampling_model="iid_multinomial", alpha=alpha)
                        region = output["region"]
                        lows = [0. if k == 0 else stats.beta.ppf(alpha / 6, k, conf["multinomial_n"] - k + 1) for k in values]
                        highs = [1. if k == conf["multinomial_n"] else stats.beta.isf(alpha / 6, k + 1, conf["multinomial_n"] - k) for k in values]
                        expected_low = [max(lows[j], 1 - math.fsum(highs[i] for i in range(3) if i != j)) for j in range(3)]
                        expected_high = [min(highs[j], 1 - math.fsum(lows[i] for i in range(3) if i != j)) for j in range(3)]
                        maximum = max(maximum, max(abs(float(actual) - expected) for actual, expected in zip(region.ci_low, expected_low)),
                                      max(abs(float(actual) - expected) for actual, expected in zip(region.ci_high, expected_high)))
                        assert maximum <= conf["maximum_endpoint_absolute_error"]
                        miss = any(not float(region.ci_low[j]) <= truth[j] <= float(region.ci_high[j]) for j in range(3))
                        if index == 0 and truth == conf["multinomial_truths"][0] and alpha == .05:
                            save_example("multinomial_region", output)
                        rejected += weight * miss
                        outcomes.append(int(miss))
                    except Exception as error:
                        errors.append({"count_position": index, "type": type(error).__name__, "message": str(error)})
                        rejected += weight
                        outcomes.append(1)
                assert total == 1
                cells.append({"method": "multinomial_region", "kind": "exact multinomial count law", "alpha": alpha,
                    "truth": truth, "count_vectors": len(outcomes), "noncoverage_fraction": str(rejected),
                    "family_noncoverage_rate": float(rejected), "outcome_flags": "".join(map(str, outcomes)),
                    "maximum_endpoint_error": maximum, "failures": errors,
                    "passed": not errors and float(rejected) <= alpha + 2e-12})
            rejected, total, errors, flags = Fraction(0), Fraction(0), [], []
            n = conf["quantile_n"]
            for index, values in enumerate(compositions(n, 3)):
                weight = mass(values, conf["ordinal_probabilities"])
                total += weight
                data = [value for value, number in zip(conf["ordinal_support"], values) for _ in range(number)]
                try:
                    output = invoke("simultaneous_quantile_ci", simultaneous_quantile_ci, pd.DataFrame({"x": data}), ["x"],
                        conf["quantile_probabilities"], sampling_model="iid_marginals", alpha=alpha)
                    intervals = output["intervals"]
                    miss = False
                    for row, truth, q in zip(intervals.itertuples(), conf["lower_quantile_truth"], conf["quantile_probabilities"]):
                        lo = -math.inf if row.lower_unbounded else float(row.ci_low)
                        hi = math.inf if row.upper_unbounded else float(row.ci_high)
                        miss |= not lo <= truth <= hi
                        # Independent integer-rational inversion admits exact
                        # ranks. Native outward CDF guards may only widen them.
                        pmf = [Fraction(math.comb(n, k)) * Fraction(str(q)) ** k * (1 - Fraction(str(q))) ** (n - k) for k in range(n + 1)]
                        budget = Fraction(str(alpha)) / 6
                        rank_lo = max(k for k in range(n + 1) if sum(pmf[:k]) <= budget)
                        rank_hi = min(k for k in range(1, n + 2) if sum(pmf[k:]) <= budget)
                        assert row.lower_rank <= rank_lo and row.upper_rank >= rank_hi
                    if index == 0 and alpha == .05:
                        save_example("simultaneous_quantile_ci", output)
                    rejected += weight * miss
                    flags.append(int(miss))
                except Exception as error:
                    errors.append({"count_position": index, "type": type(error).__name__, "message": str(error)})
                    rejected += weight
                    flags.append(1)
            assert total == 1
            cells.append({"method": "simultaneous_quantile_ci", "kind": "exact ordinal count law with atoms/ties", "alpha": alpha,
                "count_vectors": len(flags), "noncoverage_fraction": str(rejected), "family_noncoverage_rate": float(rejected),
                "outcome_flags": "".join(map(str, flags)), "failures": errors,
                "passed": not errors and float(rejected) <= alpha + 2e-12})
            rejected, total, errors, flags, maximum = Fraction(0), Fraction(0), [], [], 0.
            n = conf["bernoulli_n"]
            for index, values in enumerate(compositions(n, 4)):
                weight = mass(values, conf["pattern_probabilities"])
                total += weight
                successes = [sum(number * pattern[j] for number, pattern in zip(values, conf["row_patterns"])) for j in range(3)]
                try:
                    output = invoke("simultaneous_proportion_ci", simultaneous_proportion_ci, successes, [n] * 3,
                        sampling_model="binomial_marginals", alpha=alpha)
                    interval = output["intervals"]
                    for j, k in enumerate(successes):
                        lo = 0. if k == 0 else stats.beta.ppf(alpha / 6, k, n - k + 1)
                        hi = 1. if k == n else stats.beta.isf(alpha / 6, k + 1, n - k)
                        maximum = max(maximum, abs(float(interval.ci_low[j]) - lo), abs(float(interval.ci_high[j]) - hi))
                    assert maximum <= conf["maximum_endpoint_absolute_error"]
                    miss = any(not float(interval.ci_low[j]) <= p <= float(interval.ci_high[j]) for j, p in enumerate(conf["bernoulli_truth"]))
                    if index == 0 and alpha == .05:
                        save_example("simultaneous_proportion_ci", output)
                    rejected += weight * miss
                    flags.append(int(miss))
                except Exception as error:
                    errors.append({"count_position": index, "type": type(error).__name__, "message": str(error)})
                    rejected += weight
                    flags.append(1)
            assert total == 1
            cells.append({"method": "simultaneous_proportion_ci", "kind": "exact dependent Bernoulli pattern-count law", "alpha": alpha,
                "count_vectors": len(flags), "noncoverage_fraction": str(rejected), "family_noncoverage_rate": float(rejected),
                "outcome_flags": "".join(map(str, flags)), "maximum_endpoint_error": maximum, "failures": errors,
                "passed": not errors and float(rejected) <= alpha + 2e-12})
            print(json.dumps({"method": "exact count-law coverage", "alpha": alpha}), flush=True)

    def random_uniforms(rng, n, dimensions, design):
        if design == "independent":
            return rng.uniform(size=(n, dimensions))
        u = rng.uniform(size=n)
        if design == "comonotonic":
            return np.repeat(u[:, None], dimensions, axis=1)
        return np.column_stack((u, rng.uniform(size=n), 1 - u))

    def screen_cell(method, alpha, design, seed, family_flags, subset_flags, power_flags, errors, **extra):
        conf = protocol["coverage_screens"]
        reps = conf["families_per_cell"]
        count = sum(family_flags)
        rate = count / reps
        gate = alpha + max(.01, 6 * math.sqrt(alpha * (1 - alpha) / reps))
        cells.append({"method": method, "kind": "independent observation Monte Carlo", "alpha": alpha,
            "dependence": design, "seed": seed, "planned_families": reps, "completed_families": reps - len(errors),
            "family_noncoverage": count, "family_noncoverage_rate": rate, "wilson_95": wilson(count, reps),
            "true_subset_partial_null_rejections": sum(subset_flags), "false_null_rejections": sum(power_flags),
            "full_family_flags": "".join(map(str, family_flags)), "true_subset_flags": "".join(map(str, subset_flags)),
            "false_null_flags": "".join(map(str, power_flags)), "failures": errors, "upper_gate": gate,
            "passed": not errors and rate <= gate, **extra})

    def coverage_screens():
        conf = protocol["coverage_screens"]
        supports = conf["supports"]
        lows, widths = np.array([a for a, _ in supports]), np.array([b - a for a, b in supports])
        truth = np.array(conf["bernoulli_truth"])
        reps = conf["families_per_cell"]
        for alpha_index, alpha in enumerate(protocol["alphas"]):
            lookup = {}
            n = conf["bounded_n"]
            for method, function in (("hoeffding_mean_ci", hoeffding_mean_ci),
                                     ("empirical_bernstein_mean_ci", empirical_bernstein_mean_ci)):
                low_lookup, high_lookup = [], []
                for k in range(n + 1):
                    normalized = np.r_[np.ones(k), np.zeros(n - k)]
                    frame = pd.DataFrame(lows + normalized[:, None] * widths, columns=names)
                    output = invoke(method, function, frame, names, bounds=supports, sampling_model="iid_bounded", alpha=alpha)
                    interval = output["intervals"]
                    low_lookup.append(interval.ci_low.to_numpy())
                    high_lookup.append(interval.ci_high.to_numpy())
                    if k == n // 2 and alpha == .05:
                        save_example(method, output)
                lookup[method] = np.array(low_lookup), np.array(high_lookup)
            for design_index, design in enumerate(conf["dependence"]):
                seed = conf["seed_base"] + 100 * alpha_index + design_index
                rng = np.random.default_rng(seed)
                counts_by_family = []
                input_digest = hashlib.sha256()
                for _ in range(reps):
                    sample = (random_uniforms(rng, n, 3, design) < truth).astype(np.int64)
                    input_digest.update(sample.tobytes())
                    counts_by_family.append(sample.sum(axis=0))
                ks = np.array(counts_by_family)
                population = lows + widths * truth
                partial = population.copy()
                partial[1] = lows[1] + widths[1] * conf["partial_null_probability"]
                for method, (lo, hi) in lookup.items():
                    lower = np.column_stack([lo[ks[:, j], j] for j in range(3)])
                    upper = np.column_stack([hi[ks[:, j], j] for j in range(3)])
                    missed = (lower > population) | (upper < population)
                    wrong = (lower > partial) | (upper < partial)
                    screen_cell(method, alpha, design, seed, missed.any(axis=1).astype(int).tolist(),
                                missed[:, [0, 2]].any(axis=1).astype(int).tolist(), wrong[:, 1].astype(int).tolist(), [],
                                n=n, evaluation="Lookup of exhaustive native k=0..n calibrated intervals; not fresh API calls per family",
                                native_calibration_cases=n + 1, input_sha256=input_digest.hexdigest(),
                                minimum_interval_width=float(np.min(upper - lower)),
                                maximum_interval_width=float(np.max(upper - lower)))
                # Continuous Uniform marginals with three cross-variable laws.
                # Full and partial-null CDF checks share observations; separate
                # from the Bernoulli/interval stream via a distinct fixed seed.
                dkw_seed = seed + 10000
                rng = np.random.default_rng(dkw_seed)
                flags, subset_flags, power_flags, errors = [], [], [], []
                data_hash, critical_values = hashlib.sha256(), set()
                for rep in range(reps):
                    normalized = random_uniforms(rng, conf["dkw_n"], 3, design)
                    frame = pd.DataFrame(lows + normalized * widths, columns=names)
                    data_hash.update(frame.to_numpy().tobytes())
                    try:
                        output = invoke("simultaneous_dkw_band", simultaneous_dkw_band, frame, names,
                                        sampling_model="iid_marginals", alpha=alpha)
                        band = output["bands"]
                        misses, wrongs = [], []
                        for j, name in enumerate(names):
                            block = band[band.variable == name]
                            f = np.clip((block.knot.to_numpy() - lows[j]) / widths[j], 0, 1)
                            miss = ((f < block.left_ci_low) | (f > block.left_ci_high)
                                    | (f < block.right_ci_low) | (f > block.right_ci_high)).any()
                            # Explicit outside-sample regions must also cover
                            # the true CDF between support and first/last knots.
                            epsilon = float(output.attrs["epsilon"])
                            miss |= f[0] > epsilon or 1 - f[-1] > epsilon
                            ordered = np.sort(normalized[:, j])
                            supremum = max(float(np.max(np.arange(1, len(ordered) + 1) / len(ordered) - ordered)),
                                           float(np.max(ordered - np.arange(len(ordered)) / len(ordered))))
                            assert bool(miss) == (supremum > epsilon)
                            misses.append(bool(miss))
                            false_cdf = np.clip(f - conf["dkw_partial_cdf_shift_in_support_units"], 0, 1) if j == 1 else f
                            wrongs.append(bool(((false_cdf < block.left_ci_low) | (false_cdf > block.left_ci_high)
                                               | (false_cdf < block.right_ci_low) | (false_cdf > block.right_ci_high)).any()))
                        if rep == 0 and alpha == .05 and design == "independent":
                            save_example("simultaneous_dkw_band", output)
                        flags.append(int(any(misses)))
                        subset_flags.append(int(misses[0] or misses[2]))
                        power_flags.append(int(wrongs[1]))
                        critical_values.add(float(output.attrs["epsilon"]))
                    except Exception as error:
                        errors.append({"family": rep, "type": type(error).__name__, "message": str(error)})
                        flags.append(1)
                        subset_flags.append(1)
                        power_flags.append(0)
                screen_cell("simultaneous_dkw_band", alpha, design, dkw_seed, flags, subset_flags, power_flags, errors,
                            n=conf["dkw_n"], evaluation="Fresh native API call for every planned family",
                            input_sha256=data_hash.hexdigest(), calibrated_epsilon=sorted(critical_values))
                print(json.dumps({"method": "coverage screens", "alpha": alpha, "dependence": design,
                                  "families_each_method": reps, "dkw_failures": len(errors)}), flush=True)

    for stage in (exact_orbits, count_coverage, coverage_screens):
        try:
            stage()
        except Exception as error:
            failures.append({"stage": stage.__name__, "type": type(error).__name__, "message": str(error)})
            print(json.dumps(failures[-1]), flush=True)
    end_hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    stable = start_hashes == end_hashes
    if not stable:
        failures.append({"stage": "source identity", "changed": [name for name in SOURCE_FILES if start_hashes[name] != end_hashes[name]]})
    planned_cells = (2 * len(protocol["exact_orbits"]["null_configurations"]) * len(protocol["tails"])
                     * len(protocol["alphas"]) + 4 * len(protocol["alphas"])
                     + 3 * len(protocol["coverage_screens"]["dependence"]) * len(protocol["alphas"]))
    if len(cells) != planned_cells:
        failures.append({"stage": "planned cell completeness", "planned": planned_cells, "observed": len(cells)})
    receipt = {
        "status": "passed" if not failures and cells and all(cell["passed"] for cell in cells) else "failed",
        "protocol": protocol, "protocol_sha256": digest(args.protocol), "validator_sha256": digest(__file__),
        "source_sha256_before": start_hashes, "source_sha256_after": end_hashes, "source_stable_during_run": stable,
        "native_calls": dict(counts), "planned_cells": planned_cells, "cells": cells, "persistence": persistence,
        "stage_failures": failures, "failures_replaced": False,
        "elapsed_seconds": time.monotonic() - started,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": __import__("scipy").__version__, "torch": torch.__version__},
        "scope": protocol["limits"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "cells": len(cells), "native_calls": dict(counts),
                      "output": str(args.output)}), flush=True)
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
