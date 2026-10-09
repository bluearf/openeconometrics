"""Exact Fisher inversion on a declared finite candidate grid.

The grid restriction is essential: these are confidence sets on the supplied
finite parameter space, not continuous intervals. Primary references are
Cattaneo, Idrobo and Titiunik (2024),
https://mdcattaneo.github.io/books/Cattaneo-Idrobo-Titiunik_2024_CUP.pdf,
and Luo et al. (2021), https://arxiv.org/abs/2004.08472. Non-monotone p-value
profiles are retained without a convex hull or interpolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, product
import math
from numbers import Integral

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import common as c
from openecon.resources import plan_workspace

from . import common as m
from .assignment_extensions import _label_preflight
from .observational_survival import _SUM_CAPACITY
from .paired import _pair_key


def _finite(value, role):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("numerical_failure", f"{role} exceeds finite float64 representation.")
    return value


def _require_gradual_underflow():
    # A read-only arithmetic probe; changing a caller's FTZ/DAZ mode would hide
    # a failed error-free-expansion assumption and affect unrelated analysis.
    smallest_normal = torch.tensor(torch.finfo(torch.float64).tiny,
                                   dtype=torch.float64, device="cpu")
    half = smallest_normal*.5
    if bool(half == 0) or bool(half*2 != smallest_normal):
        raise AnalysisError("numerical_failure", "Exact expansion arithmetic requires gradual float64 underflow; caller FTZ/DAZ mode is unsupported and is not changed.")


def _grid(candidates):
    if (not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes))
            or not 1 <= len(candidates) <= 64):
        raise AnalysisError("invalid_candidates", "Supply a finite sequence of 1 through 64 distinct numeric candidates.")
    values = [c.check_number(value, "candidate effect") for value in candidates]
    if any(isinstance(original, Integral) and int(value) != original
           for original, value in zip(candidates, values, strict=True)):
        raise AnalysisError("invalid_candidates", "Integer candidates must be represented exactly in float64.")
    if any(original != value for original, value in zip(candidates, values, strict=True)):
        raise AnalysisError("invalid_candidates", "Candidate effects must be represented exactly in float64; narrowing an original numeric value is unsupported.")
    if len(set(values)) != len(values):
        raise AnalysisError("invalid_candidates", "Candidate effects must be distinct; their original order is retained.")
    return values


def _prepare(data, y, treatment, identity, kind, missing, max_work):
    if missing != "raise":
        raise AnalysisError("unsupported_missing", "The original assignment universe requires missing='raise'; no rows or groups may be deleted.")
    names = [c.check_name(y, "y"), c.check_name(treatment, "treatment")]
    if identity is not None:
        names.append(c.check_name(identity, kind))
    if len(set(names)) != len(names):
        raise AnalysisError("invalid_spec", "Outcome, treatment and design identity must name distinct columns.")
    _label_preflight(data, names, max_work, cluster=identity)
    source = c.source(data)
    c.require_numeric(source, [y])
    if pd.api.types.is_bool_dtype(source[y].dtype):
        raise AnalysisError("invalid_outcome", "Outcomes must be numeric values, not Boolean labels.")
    if (pd.api.types.is_float_dtype(source[y].dtype)
            and source[y].dtype.itemsize > 8):
        raise AnalysisError("numerical_failure", "Floating outcomes wider than float64 are unsupported; narrowing an original outcome could alter exact comparisons.")
    if pd.api.types.is_integer_dtype(source[y].dtype):
        # Admission precedes every numeric sample tensor. A float64 sum may be
        # certified exactly only after the original integer values survive the
        # conversion; accepting rounded inputs would certify a different law.
        for value in source[y]:
            if not pd.isna(value) and int(float(value)) != int(value):
                raise AnalysisError("numerical_failure", "Original integer outcomes must be represented exactly in float64.")
    topology, _ = m.sample(data, names[1:], numeric=[treatment], missing="raise", max_work=max_work, cost=128)
    d = m.binary(topology, treatment)
    groups = []
    if identity is not None:
        found = {}
        for i, value in enumerate(topology[identity]):
            try:
                key, label = _pair_key(value)
            except AnalysisError as exc:
                raise AnalysisError("invalid_design_identifier", "Design identities must be finite typed numeric, Boolean or text scalars.") from exc
            found.setdefault(key, [label, []])[1].append(i)
        for key, (label, positions) in found.items():
            count = int(d[positions].sum())
            if kind == "pair" and (len(positions) != 2 or count != 1):
                raise AnalysisError("invalid_pair", "Each original pair must have exactly two rows, one treated and one control.")
            if kind == "cluster" and count not in (0, len(positions)):
                raise AnalysisError("invalid_cluster", "Every original cluster must have one common treatment assignment.")
            if kind == "pair":
                positions = sorted(positions, key=lambda position: -float(d[position]))
            groups.append(dict(label_type=key[0], label=label, positions=positions,
                               n=len(positions), n_treated=count))
    else:
        groups = [dict(label_type=None, label=None, positions=list(range(len(d))),
                       n=len(d), n_treated=int(d.sum()))]
    frame, metadata = m.sample(data, names, numeric=[y, treatment], missing="raise", max_work=max_work, cost=128)
    return m.tensor(frame, y), d, groups, metadata


def _support_count(units, treated, kind, max_work):
    if kind == "pair":
        # Bound the integer before constructing an enormous power or a tensor.
        if units >= max_work.bit_length():
            m.work(max_work + 1, max_work, "complete paired assignment support")
        count = 2**units
    else:
        count = 1
        for j in range(1, min(treated, units-treated) + 1):
            count = count*(units-j+1)//j
            if count > max_work:
                m.work(count, max_work, "complete fixed-count assignment support")
    return count


def _plan(n, groups, kind, grid_n, support, max_work):
    units = len(groups) if kind != "complete" else n
    treated = sum(group["n_treated"] != 0 for group in groups) if kind == "cluster" else int(groups[0]["n_treated"])
    coefficient = 1 if kind == "pair" else max(treated, units-treated)
    terms = 2*n*coefficient.bit_length()
    work = 128*n + 64*grid_n*n + 32*_SUM_CAPACITY*terms*(support+1)*grid_n
    work += 64*_SUM_CAPACITY**2*support*grid_n
    m.work(work, max_work, "all candidate nulls, exact assignments, expansion sums and comparisons")
    plan = plan_workspace("complete finite-grid Fisher profiles and full scientific state", {
        "typed_original_sample_topology_and_labels": 8192*n,
        "all_null_inputs_and_low_parts": 512*grid_n*n,
        "original_and_design_unit_assignment_bits": 64*support*(n+units),
        "live_expansion_arithmetic": 256*_SUM_CAPACITY*(support+1),
        "full_statistic_and_comparison_expansions_tables_and_state": 768*grid_n*(support+1)*_SUM_CAPACITY*2,
        "complete_candidate_assignment_tables": 4096*grid_n*support,
    })
    return work, plan.record(), coefficient


def _two_sum(a, b):
    high = _finite(a+b, "Null imputation")
    z = high-a
    low = _finite((a-(high-z))+(b-z), "Null imputation residual")
    return high, low


def _nulls(y, d, candidates, coefficient):
    # All grid imputations are checked before support-sized tensor allocation.
    nulls = []
    largest = 0.0
    for candidate in candidates:
        shift = d*candidate
        high, low = _two_sum(y, -shift)
        if candidate != 0 and bool(((d == 1) & (high == y)).any()):
            raise AnalysisError("numerical_failure", "A nonzero candidate null shift is absorbed at the outcome scale.")
        if bool(((y != 0) & (shift != 0) & (high == -shift)).any()):
            raise AnalysisError("numerical_failure", "A nonzero original outcome is absorbed by candidate null imputation.")
        largest = max(largest, float(high.abs().max()), float(low.abs().max()))
        nulls.append((high, low))
    exponent = math.frexp(largest)[1] if largest else 0
    power = -exponent-len(y).bit_length()-coefficient.bit_length()-2
    scaled = []
    for high, low in nulls:
        parts = []
        for value in (high, low):
            result = _finite(torch.ldexp(value, torch.full_like(value, power, dtype=torch.int32)), "Power-of-two null scaling")
            restored = torch.ldexp(result, torch.full_like(result, -power, dtype=torch.int32))
            if not bool((restored == value).all()):
                raise AnalysisError("numerical_failure", "Exact power-of-two scaling cannot retain every nonzero null component; restrict the dynamic range.")
            parts.append(result)
        scaled.append(parts)
    return nulls, scaled, power


def _expansion(values, shape):
    """FastTwoSum expansions, with no rounding-based equality tolerance."""
    parts = []
    for value in values:
        x = value
        keep = 0
        for y in parts:
            larger = x.abs() >= y.abs()
            a, b = torch.where(larger, x, y), torch.where(larger, y, x)
            high = _finite(a+b, "Assignment expansion sum")
            low = _finite(b-(high-a), "Assignment expansion residual")
            if bool((low != 0).any()):
                parts[keep] = low
                keep += 1
            x = high
        parts[keep:] = [x]
        if len(parts) > _SUM_CAPACITY:
            raise AnalysisError("numerical_failure", "The exact assignment sum exceeds its 64-part native expansion capacity.")
    if not parts:
        parts = [torch.zeros(shape, dtype=torch.float64, device="cpu")]
    sign = torch.zeros(shape, dtype=torch.int64, device="cpu")
    total = torch.zeros(shape, dtype=torch.float64, device="cpu")
    for value in reversed(parts):
        sign = torch.where((sign == 0) & (value != 0), value.sign().to(torch.int64), sign)
        total = _finite(total+value, "Rounded assignment sum")
    if bool(((sign != 0) & (total == 0)).any()) or bool(((sign != 0) & (total.sign() != sign)).any()):
        raise AnalysisError("numerical_failure", "A nonzero assignment comparison cannot be represented or certified; it is not an exact tie.")
    return parts, total, sign


def _assignments(n, d, groups, kind, count):
    units = n if kind == "complete" else len(groups)
    bits = torch.zeros((count, units), dtype=torch.bool, device="cpu")
    if kind == "pair":
        for row, code in enumerate(product((0, 1), repeat=units)):
            bits[row] = torch.tensor(code, dtype=torch.bool, device="cpu")
        unit_bits = torch.zeros((count, n), dtype=torch.bool, device="cpu")
        for group, topology in enumerate(groups):
            treated, control = topology["positions"]
            unit_bits[:, treated] = bits[:, group]
            unit_bits[:, control] = ~bits[:, group]
    else:
        treated = int(d.sum()) if kind == "complete" else sum(group["n_treated"] != 0 for group in groups)
        for row, chosen in enumerate(combinations(range(units), treated)):
            bits[row, list(chosen)] = True
        if kind == "complete":
            unit_bits = bits
        else:
            codes = torch.empty(n, dtype=torch.int64, device="cpu")
            for group, topology in enumerate(groups):
                codes[topology["positions"]] = group
            unit_bits = bits[:, codes]
    return bits, unit_bits


def _score(parts, bits, positive, negative):
    def terms():
        for row in range(bits.shape[1]):
            for value in parts:
                for count, assigned, direction in ((positive, bits[:, row], 1), (negative, ~bits[:, row], -1)):
                    for bit in range(count.bit_length()):
                        if count & (1 << bit):
                            shifted = torch.ldexp(value[row], torch.tensor(bit, dtype=torch.int32, device="cpu"))
                            yield torch.where(assigned, direction*shifted, torch.zeros_like(shifted))
    return _expansion(terms(), (len(bits),))


def _report(value, multiplier, power, sign):
    # Combine exponents before restoring original units. Multiplying a tiny
    # normalized numerator by its denominator first can invent a zero statistic.
    mantissa, exponent = torch.frexp(value)
    fraction, factor_exponent = math.frexp(multiplier)
    reported = _finite(torch.ldexp(mantissa*fraction, exponent+factor_exponent-power), "Original-unit assignment statistic")
    if bool(((sign != 0) & (reported == 0)).any()):
        raise AnalysisError("numerical_failure", "A nonzero original-unit assignment statistic underflows.")
    return reported


def _components(candidates, accepted):
    order = sorted(range(len(candidates)), key=lambda index: candidates[index])
    runs = []
    active = []
    for index in order:
        if accepted[index]:
            active.append(index)
        elif active:
            runs.append(active)
            active = []
    if active:
        runs.append(active)
    return order, runs


def _confidence(name, data, y, treatment, identity, *, candidates, design, required,
                kind, level, missing, device, weights, max_work):
    m.options(device, weights, max_work, level)
    _require_gradual_underflow()
    if not isinstance(design, str) or design != required:
        raise AnalysisError("unsupported_design", f"Explicitly declare design='{required}'.")
    candidates = _grid(candidates)
    level = c.check_number(level, "level", minimum=0, maximum=1, exclusive=True)
    level_numerator, level_denominator = level.as_integer_ratio()
    outcome, d, groups, metadata = _prepare(data, y, treatment, identity, kind, missing, max_work)
    n = len(d)
    units = n if kind == "complete" else len(groups)
    treated = int(d.sum()) if kind == "complete" else sum(group["n_treated"] != 0 for group in groups)
    count = _support_count(units, treated, kind, max_work)
    work, plan, coefficient = _plan(n, groups, kind, len(candidates), count, max_work)
    nulls, scaled_nulls, power = _nulls(outcome, d, candidates, coefficient)
    bits, unit_bits = _assignments(n, d, groups, kind, count)
    all_bits = torch.cat((d.bool()[None, :], unit_bits), dim=0)
    if kind == "pair":
        positive = negative = 1
        multiplier = 1/units
    else:
        positive, negative = units-treated, treated
        multiplier = 1/(positive*negative)
        if kind == "cluster":
            multiplier *= units/n
    encoded = ["".join("1" if value else "0" for value in row) for row in bits.tolist()]
    unit_encoded = ["".join("1" if value else "0" for value in row) for row in unit_bits.tolist()]
    profile, tests, records, accepted = [], [], [], []
    for candidate_index, (candidate, null, scaled) in enumerate(zip(candidates, nulls, scaled_nulls, strict=True)):
        parts, rounded, signs = _score(scaled, all_bits, positive, negative)
        reported = _report(rounded, multiplier, power, signs)
        comparison, gap, comparison_sign = _expansion(
            (term for value in parts for term in (value[1:]*signs[1:], -value[0]*signs[0]*torch.ones(count, dtype=torch.float64, device="cpu"))),
            (count,),
        )
        extreme = comparison_sign >= 0
        n_extreme = int(extreme.sum())
        p_value = n_extreme/count
        # The exact support fraction determines acceptance even if its printed
        # float64 p-value rounds onto the threshold.
        keep = n_extreme*level_denominator > (level_denominator-level_numerator)*count
        accepted.append(keep)
        profile.append([candidate_index, candidate, float(reported[0]), p_value, keep, n_extreme, count])
        tests.extend([candidate_index, draw, float(reported[draw+1]), float(rounded[draw+1]),
                      int(comparison_sign[draw]), bool(extreme[draw]), bool(comparison_sign[draw] == 0)]
                     for draw in range(count))
        records.append(dict(candidate_index=candidate_index, effect=candidate,
            null_imputed_outcomes=null[0].tolist(), null_imputation_low_parts=null[1].tolist(),
            normalized_null_high_parts=scaled[0].tolist(), normalized_null_low_parts=scaled[1].tolist(),
            normalized_numerator_expansion=[value.tolist() for value in parts],
            normalized_numerators=rounded.tolist(), numerator_signs=signs.tolist(),
            observed_statistic=float(reported[0]), assignment_statistics=reported[1:].tolist(),
            absolute_comparison_expansion=[value.tolist() for value in comparison],
            normalized_absolute_comparison_gaps=gap.tolist(), comparison_signs=comparison_sign.tolist(),
            exact_ties=(comparison_sign == 0).tolist(), extreme=extreme.tolist(),
            n_extreme=n_extreme, p_value=p_value, p_value_fraction=[n_extreme, count], accepted=keep))
    order, runs = _components(candidates, accepted)
    design_table = m.frame([[i, group["label_type"], group["label"], group["n"], group["n_treated"], group["positions"]]
                           for i, group in enumerate(groups)], columns=["group", "label_type", "label", "n", "n_treated", "positions"])
    design_table["label"] = pd.Series([group["label"] for group in groups], dtype=object)
    accepted_table = m.frame([[i, candidates[i]] for i in range(len(candidates)) if accepted[i]], columns=["candidate", "effect"])
    accepted_table["candidate"] = accepted_table["candidate"].astype("int64")
    accepted_table["effect"] = accepted_table["effect"].astype("float64")
    if accepted_table.empty:
        # The artifact reader reconstructs a genuinely empty generic index;
        # retain that exact index contract rather than an empty RangeIndex.
        accepted_table.index = pd.Index([], dtype=object)
    settings = dict(y=y, treatment=treatment, candidates=candidates, design=design,
        level=level, method="exact", alternative="two-sided", missing=missing,
        device=device, weights=None, max_work=max_work)
    if identity is not None:
        settings[kind] = identity
    state = dict(target="constant additive Fisher effect on a caller-declared finite parameter grid",
        parameter_space="supplied finite candidate grid only", outcomes=outcome.tolist(), treatment=d.tolist(),
        groups=groups, candidates=candidates, candidate_tests=records,
        assignment_bits=encoded, unit_assignment_bits=unit_encoded,
        assignment_probabilities=[1/count]*count, assignment_probability_fraction=[1, count],
        assignment_universe_size=count, n_candidates=len(candidates),
        accepted_candidate_indices=[i for i, value in enumerate(accepted) if value],
        sorted_candidate_indices=order, accepted_grid_components=runs,
        empty_set=not any(accepted), all_candidates_accepted=all(accepted),
        disconnected_on_sorted_grid=len(runs) > 1,
        normalization_power_of_two=power, numerator_coefficients=dict(treated=positive, control=-negative),
        gradual_float64_underflow_required=True,
        reported_statistic_multiplier=multiplier,
        comparison_rule="absolute exact original-row expansion numerators; sign>=0; no equality tolerance",
        acceptance_rule="exact two-sided p_value>1-level", confidence_interval=None,
        alpha_fraction=[level_denominator-level_numerator, level_denominator],
        covariance=None, standard_error=None, df=None,
        planned_work=work, computation_resource_plan=plan,
        coverage="design coverage at least level if the true constant effect belongs to the predeclared finite grid; no off-grid or whole-real-line coverage",
        assumptions=["caller fixed the finite parameter grid before examining outcomes", "the declared original assignment law is correct",
                     "one constant additive effect at every original unit", "complete original topology/outcomes and no interference"],
        assignment_encoding="unit bits use original row order; paired bits select the originally treated member in first-appearance pair order" if kind == "pair" else "original row order for complete; first-appearance cluster order for cluster",
        rng=None)
    return m.result(name, {
        "profile": m.frame(profile, columns=["candidate", "effect", "observed_statistic", "p_value", "accepted", "n_extreme", "n_assignments"]),
        "accepted_candidates": accepted_table,
        "design": design_table,
        "assignments": m.frame([[i, code, unit_encoded[i], 1/count] for i, code in enumerate(encoded)], columns=["assignment", "bits", "unit_bits", "probability"]),
        "tests": m.frame(tests, columns=["candidate", "assignment", "statistic", "normalized_numerator", "comparison_sign", "extreme", "exact_tie"]),
    }, metadata, settings, state, notes=[
        "Only the supplied finite candidate set is inverted; no hull, interpolation or continuous confidence interval is returned.",
        "The grid must be predeclared and contain the true constant effect for the stated design-coverage interpretation.",
        "Every original assignment and candidate statistic is retained; ambiguous nonzero arithmetic is refused, not treated as an exact tie.",
    ])


@m.procedure
def randomization_confidence_set(data, y, treatment, *, candidates, design, level=.95,
                                missing="raise", device="cpu", weights=None, max_work=100_000_000):
    """Exact two-sided Fisher inversion on a predeclared finite complete-design grid."""
    return _confidence("randomization_confidence_set", data, y, treatment, None,
        candidates=candidates, design=design, required="complete_randomized", kind="complete",
        level=level, missing=missing, device=device, weights=weights, max_work=max_work)


@m.procedure
def paired_randomization_confidence_set(data, y, treatment, pair, *, candidates, design, level=.95,
                                       missing="raise", device="cpu", weights=None, max_work=100_000_000):
    """Exact fair independent-pair Fisher inversion on a predeclared finite grid."""
    return _confidence("paired_randomization_confidence_set", data, y, treatment, pair,
        candidates=candidates, design=design, required="paired_randomized", kind="pair",
        level=level, missing=missing, device=device, weights=weights, max_work=max_work)


@m.procedure
def cluster_randomization_confidence_set(data, y, treatment, cluster, *, candidates, design, level=.95,
                                        missing="raise", device="cpu", weights=None, max_work=100_000_000):
    """Exact fixed-count whole-cluster HT Fisher inversion on a predeclared finite grid."""
    return _confidence("cluster_randomization_confidence_set", data, y, treatment, cluster,
        candidates=candidates, design=design, required="cluster_randomized", kind="cluster",
        level=level, missing=missing, device=device, weights=weights, max_work=max_work)
