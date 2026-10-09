"""Declared fixed-count assignment tests and paired signed-rank hidden-bias bounds.

Primary references: Cattaneo, Idrobo and Titiunik (2024),
https://doi.org/10.1017/9781009441896; Rosenbaum (1987),
https://doi.org/10.1093/biomet/74.1.13. Signed-rank DP also handles tied average
ranks via doubled integer scores; no vendor exact-ties equivalence is asserted.
"""

from __future__ import annotations

from itertools import combinations, product
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import common as c
from openecon.resources import plan_workspace

from . import common as m
from .paired import _pair_key, _pair_table, _sample as paired_sample


def _choice(design, required):
    if not isinstance(design, str) or design != required:
        raise AnalysisError("unsupported_design", f"Explicitly declare design='{required}'; labels do not establish random assignment.")


def _sample(data, y, treatment, strata, missing, max_work):
    if missing != "raise":
        raise AnalysisError("unsupported_missing", "Fixed-count randomization requires missing='raise'; deleting outcomes would change the declared assignment universe.")
    names = [c.check_name(y, "y"), c.check_name(treatment, "treatment")]
    if strata is not None:
        names.append(c.check_name(strata, "strata"))
    if len(set(names)) != len(names):
        raise AnalysisError("invalid_spec", "Outcome, treatment and strata must name distinct columns.")
    # Check original assignment/stratum topology before inspecting outcomes.
    design, design_metadata = m.sample(data, names[1:], numeric=[treatment], missing="raise", max_work=max_work, cost=8)
    assignment = m.binary(design, treatment)
    groups = []
    if strata is None:
        groups.append((None, None, list(range(len(design))), int(assignment.sum())))
    else:
        found = {}
        for i, value in enumerate(design[strata]):
            try:
                key, label = _pair_key(value)
            except AnalysisError as exc:
                raise AnalysisError("invalid_stratum", "Strata identifiers must be finite typed numeric, boolean or text scalars.") from exc
            found.setdefault(key, [label, []])[1].append(i)
        for key, (label, rows) in found.items():
            count = int(assignment[rows].sum())
            if count == 0 or count == len(rows):
                raise AnalysisError("invalid_stratum", "Every original stratum must contain both treatment arms.")
            groups.append((key[0], label, rows, count))
    selected, metadata = m.sample(data, names, numeric=[y, treatment], missing="raise", max_work=max_work, cost=8)
    outcome = m.tensor(selected, y)
    metadata.update(design_sample_sha256=design_metadata["sample_sha256"], n_strata=len(groups),
                    assignment_universe="all supplied complete observations; no outcome deletion")
    return outcome, assignment, groups, metadata


def _bounded_combinations(n, k, limit, max_work):
    """Reject huge supports without constructing their potentially giant count."""
    count = 1
    for j in range(1, min(k, n-k)+1):
        count = count * (n-j+1) // j
        if count > limit:
            m.work(max_work+1, max_work, "complete fixed-count assignment support")
    return count


def _assignments(groups, n, method, draws, seed, max_work, initial_work):
    max_count = max(0, (max_work-initial_work)//max(12*n, 1))
    count = 1
    component_counts = []
    if method == "exact":
        for _, _, rows, treated in groups:
            component = _bounded_combinations(len(rows), treated, max_count//count if count else 0, max_work)
            component_counts.append(component)
            count *= component
            if count > max_count:
                m.work(max_work+1, max_work, "complete stratified Cartesian assignment support")
    else:
        count = draws
    planned_work = initial_work + 12*count*n
    m.work(planned_work, max_work, "full fixed-count assignments and test statistics")
    plan = plan_workspace("complete randomization assignments and saved scientific state", {
        "selected_sample_and_design_state": 2048*n,
        "assignment_and_statistic_tensors": 32*count*n + 64*count,
        "complete_assignment_strings_tables_and_state": 160*count*n + 640*count,
        "stratum_combination_pools": 64*count*n if method == "exact" else 0,
    })
    bits = torch.zeros((count, n), dtype=torch.bool, device="cpu")
    if method == "exact":
        supports = [combinations(rows, treated) for _, _, rows, treated in groups]
        for i, pieces in enumerate(product(*supports)):
            for chosen in pieces:
                bits[i, list(chosen)] = True
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for i in range(count):
            for _, _, rows, treated in groups:
                permutation = torch.randperm(len(rows), generator=generator, device="cpu")[:treated]
                selected = torch.tensor(rows, dtype=torch.int64, device="cpu")[permutation]
                bits[i, selected] = True
    return bits, component_counts, planned_work, plan.record()


def _statistics(bits, adjusted, groups):
    # A common positive scaling leaves assignment p-values unchanged, while
    # avoiding overflow in means/sums. Guard any lost positive input precision.
    scale = float(adjusted.abs().max())
    scale = scale if scale > 0 else 1.0
    normalized = adjusted/scale
    if bool(((adjusted != 0) & (normalized == 0)).any()):
        raise AnalysisError("numerical_failure", "Null outcomes underflow after statistic scaling; rescale the outcome/null or restrict their dynamic range.")
    statistics = torch.zeros(len(bits), dtype=torch.float64, device="cpu")
    absolute_scale = 0.0
    for _, _, rows, treated in groups:
        values = normalized[rows]
        centered = values-values.mean()
        weight = len(rows)/len(adjusted)
        control = len(rows)-treated
        # Check contribution scale before it can turn distinct assignments into
        # artificial numerical ties. Balanced observed contrasts may still be0.
        for denominator in (treated, control):
            if bool(((centered != 0) & (centered*weight/denominator == 0)).any()):
                raise AnalysisError("numerical_failure", "A nonzero assignment-statistic contribution underflowed; rescale the outcome and null together.")
        total = centered.sum()
        selected_sum = bits[:, rows].to(torch.float64) @ centered
        statistics += weight*(selected_sum/treated - (total-selected_sum)/control)
        absolute_scale += weight*float(centered.abs().mean())
    tolerance = 64*torch.finfo(torch.float64).eps*absolute_scale
    reported = statistics*scale
    if (not bool(torch.isfinite(reported).all())
            or bool(((statistics != 0) & (reported == 0)).any())):
        raise AnalysisError("numerical_failure", "An assignment statistic exceeds finite nonzero float64 representation.")
    return statistics, reported, scale, tolerance


def _randomization(name, data, y, treatment, strata, *, design, required, null_effect,
                   alternative, method, draws, seed, missing, device, weights, max_work):
    m.options(device, weights, max_work)
    _choice(design, required)
    c.check_choice(alternative, "alternative", ("greater", "less", "two-sided"))
    c.check_choice(method, "method", ("exact", "monte_carlo"))
    null_effect = c.check_number(null_effect, "null_effect")
    draws = c.check_count(draws, "draws", maximum=1_000_000)
    seed = c.check_count(seed, "seed", minimum=0, maximum=2**63-1)
    outcome, assignment, groups, metadata = _sample(data, y, treatment, strata, missing, max_work)
    adjusted = outcome-null_effect*assignment
    if not bool(torch.isfinite(adjusted).all()):
        raise AnalysisError("numerical_failure", "Sharp-null imputation exceeds finite float64 arithmetic.")
    bits, component_counts, planned_work, plan = _assignments(groups, len(outcome), method, draws, seed, max_work, 16*metadata["n_input"])
    all_bits = torch.cat((assignment[None, :].bool(), bits), 0)
    scaled, reported, scale, tolerance = _statistics(all_bits, adjusted, groups)
    observed, observed_scaled = float(reported[0]), float(scaled[0])
    scaled, reported = scaled[1:], reported[1:]
    if alternative == "greater":
        extreme = scaled >= observed_scaled-tolerance
    elif alternative == "less":
        extreme = scaled <= observed_scaled+tolerance
    else:
        extreme = scaled.abs() >= abs(observed_scaled)-tolerance
    count, total = int(extreme.sum()), len(bits)
    p_value = count/total if method == "exact" else (count+1)/(total+1)
    mc_se = None if method == "exact" else math.sqrt(p_value*(1-p_value)*total)/(total+1)
    encoded = ["".join("1" if bit else "0" for bit in row) for row in bits.tolist()]
    design_rows = [[i, kind, label, len(rows), treated, len(rows)-treated, len(rows)/len(outcome)]
                   for i, (kind, label, rows, treated) in enumerate(groups)]
    design_table = m.frame(design_rows, columns=["stratum", "label_type", "label", "n", "treated_n", "control_n", "statistic_weight"])
    design_table["label"] = pd.Series([row[2] for row in design_rows], dtype=object)
    settings = dict(y=y, treatment=treatment, strata=strata, design=design, null_effect=null_effect,
                    alternative=alternative, method=method, draws=draws, seed=seed,
                    missing=missing, device=device, weights=None, max_work=max_work)
    state = dict(target="Fisher fixed-count constant-additive sharp-null randomization p-value",
                 outcomes=outcome.tolist(), treatment=assignment.tolist(), null_imputed_outcomes=adjusted.tolist(),
                 strata=[dict(type=kind, label=label, positions=rows, n_treated=treated,
                              statistic_weight=len(rows)/len(outcome)) for kind, label, rows, treated in groups],
                 observed_statistic=observed, randomization_statistics=reported.tolist(),
                 normalized_observed_statistic=observed_scaled, normalized_randomization_statistics=scaled.tolist(),
                 statistic_comparison_scale=scale, normalized_statistic_tie_tolerance=tolerance,
                 assignment_bits=encoded, assignment_encoding="original sample row order;1=treatment,0=control",
                 component_assignment_counts=component_counts if method == "exact" else None,
                 extreme=extreme.tolist(), n_extreme=count, n_assignments=total, p_value=p_value,
                 monte_carlo_standard_error=mc_se,
                 monte_carlo_maximum_standard_error=None if method == "exact" else math.sqrt(total/4)/(total+1),
                 planned_work=planned_work, computation_resource_plan=plan,
                 covariance=None, standard_error=None, confidence_interval=None, df=None,
                 inference="complete exact equiprobable assignment support" if method == "exact" else "uniform fixed-count private-RNG Monte Carlo; observed-assignment plus-one correction",
                 assumptions=["declared equiprobable fixed-count random assignment, independently across supplied strata",
                              "constant additive null for every unit", "no interference across units", "complete outcomes for the fixed assignment universe"],
                 statistic="sample-size weighted sum of within-stratum treated-minus-control null-imputed means",
                 rng=None if method == "exact" else dict(engine="torch.Generator(cpu)/randperm", seed=seed, torch_version=str(torch.__version__)))
    return m.result(name, {
        "test": m.frame([[observed, p_value, count, total, mc_se]], columns=["statistic", "p_value", "n_extreme", "n_assignments", "monte_carlo_standard_error"]),
        "design": design_table,
        "assignments": m.frame([[i, code, float(value), bool(flag)] for i, (code, value, flag) in enumerate(zip(encoded, reported.tolist(), extreme.tolist()))],
                               columns=["draw", "assignment", "statistic", "extreme"]),
    }, metadata, settings, state, notes=[
        "Sharp-null assignment p-values are not average-effect confidence intervals or standard errors.",
        "Complete outcomes and the stated fixed-count assignment design are required; no assignment design is inferred from labels.",
        "Monte Carlo standard errors describe simulation uncertainty only." if method == "monte_carlo" else "Every admissible assignment is enumerated exactly.",
    ])


@m.procedure
def randomization_test(data, y, treatment, *, design=None, null_effect=0.0,
                       alternative="two-sided", method="exact", draws=9999, seed=1729,
                       missing="raise", device="cpu", weights=None, max_work=100_000_000):
    """Fisher test for a declared complete-randomized fixed-treated-count design.

All original subjects remain in the assignment universe. Exact enumeration or
uniform private-RNG Monte Carlo tests a constant additive sharp null; Monte Carlo
uses an observed-assignment plus-one p-value and saves every assignment/statistic.
"""
    return _randomization("randomization_test", data, y, treatment, None, design=design,
        required="complete_randomized", null_effect=null_effect, alternative=alternative,
        method=method, draws=draws, seed=seed, missing=missing, device=device, weights=weights, max_work=max_work)


@m.procedure
def stratified_randomization(data, y, treatment, strata, *, design=None, null_effect=0.0,
                             alternative="two-sided", method="exact", draws=9999, seed=1729,
                             missing="raise", device="cpu", weights=None, max_work=100_000_000):
    """Fisher test fixing each typed stratum's observed treatment count.

Every stratum has both arms. The statistic weights each within-stratum contrast
by stratum sample size/total sample size. Exact support is the complete Cartesian
product; Monte Carlo samples independent uniform fixed-count assignments by stratum.
"""
    strata = c.check_name(strata, "strata")
    return _randomization("stratified_randomization", data, y, treatment, strata, design=design,
        required="stratified_randomized", null_effect=null_effect, alternative=alternative,
        method=method, draws=draws, seed=seed, missing=missing, device=device, weights=weights, max_work=max_work)


def _rank_scores(differences):
    used = torch.nonzero(differences != 0).flatten()
    values = differences[used].abs()
    order = torch.argsort(values, stable=True)
    _, inverse, counts = torch.unique_consecutive(values[order], return_inverse=True, return_counts=True)
    starts = torch.cumsum(counts, 0)-counts
    doubled = 2*starts+counts+1
    ranks = torch.empty(len(used), dtype=torch.int64, device="cpu")
    ranks[order] = doubled[inverse]
    return used, ranks


def _rank_distribution(ranks, success, failure, total):
    mass = torch.zeros(total+1, dtype=torch.float64, device="cpu")
    mass[0] = 1
    used = 0
    for rank in ranks.tolist():
        previous = mass[:used+1]
        low, high = previous*failure, previous*success
        if bool(((previous > 0) & ((low == 0) | (high == 0))).any()):
            raise AnalysisError("numerical_failure", "A positive signed-rank assignment probability underflowed; no truncated exact distribution is returned.")
        updated = torch.zeros_like(mass)
        updated[:used+1] = low
        updated[rank:rank+used+1] += high
        mass, used = updated, used+rank
    tolerance = 64*torch.finfo(torch.float64).eps*(len(ranks)+1)
    normalizer = float(mass.sum())
    if not bool(torch.isfinite(mass).all()) or abs(normalizer-1) > tolerance:
        raise AnalysisError("numerical_failure", "The complete signed-rank null distribution failed its normalization gate.")
    tail = torch.flip(torch.cumsum(torch.flip(mass, (0,)), 0), (0,))
    if bool(((tail < -tolerance) | (tail > 1+tolerance)).any()):
        raise AnalysisError("numerical_failure", "Signed-rank tail probabilities exceed their roundoff domain.")
    # This clips verified roundoff at probability endpoints, never assignment
    # probabilities, data, ranks or substantive sensitivity effects.
    return mass, tail.clamp(0, 1), normalizer


@m.procedure
def rosenbaum_rank_bounds(data, y, treatment, pair, *, gammas=(1.0, 1.5, 2.0),
                          alternative="greater", null_effect=0.0, missing="raise",
                          device="cpu", weights=None, max_work=100_000_000):
    """Exact one-sided paired Wilcoxon signed-rank tails under Gamma hidden bias.

Ranks use average tied absolute null-adjusted differences, doubled to integer
scores. Zero differences are conditioned out. Native DP saves all probability
masses and tails; bounds concern the stated additive sharp null, not ATE intervals.
"""
    m.options(device, weights, max_work)
    c.check_choice(alternative, "alternative", ("greater", "less"))
    null_effect = c.check_number(null_effect, "null_effect")
    if not isinstance(gammas, (list, tuple)) or not 1 <= len(gammas) <= 256:
        raise AnalysisError("invalid_option", "gammas must be1..256 strictly increasing finite numbers>=1.")
    gamma_values = [c.check_number(gamma, "gamma", minimum=1) for gamma in gammas]
    if any(a >= b for a, b in zip(gamma_values, gamma_values[1:])):
        raise AnalysisError("invalid_option", "gammas must be strictly increasing without duplicates.")
    differences, pair_rows, metadata = paired_sample(data, y, treatment, pair, missing, max_work)
    adjusted = differences-null_effect
    if not bool(torch.isfinite(adjusted).all()):
        raise AnalysisError("numerical_failure", "Null-adjusted rank differences exceed finite float64 arithmetic.")
    informative = int((adjusted != 0).sum())
    total = informative*(informative+1)
    work_units = 5*metadata["n_input"] + 8*informative*max(1, math.ceil(math.log2(max(informative, 2)))) + 16*len(gamma_values)*informative*(total+1) + 24*len(gamma_values)*(total+1)
    m.work(work_units, max_work, "complete tied-rank DP, all Gamma bounds, masses and tails")
    plan = plan_workspace("paired signed-rank complete distributions and saved state", {
        "original_pair_sample_and_ranks": 2048*metadata["n_input"],
        "live_dynamic_program_vectors": 64*(total+1),
        "all_gamma_mass_tail_tables_and_python_state": 1024*len(gamma_values)*(total+1),
    })
    used, ranks = _rank_scores(adjusted)
    favourable = adjusted[used] > 0 if alternative == "greater" else adjusted[used] < 0
    observed = int(ranks[favourable].sum())
    if int(ranks.sum()) != total:
        raise AnalysisError("numerical_failure", "Doubled average ranks do not sum to the complete rank support.")
    bounds, distributions, rows = [], [], []
    for gamma in gamma_values:
        lower_probability = 1/(1+gamma)
        upper_probability = gamma/(1+gamma)
        lower_mass, lower_tail, lower_norm = _rank_distribution(ranks, lower_probability, upper_probability, total)
        upper_mass, upper_tail, upper_norm = _rank_distribution(ranks, upper_probability, lower_probability, total)
        p_low, p_high = float(lower_tail[observed]), float(upper_tail[observed])
        bounds.append([gamma, observed/2, p_low, p_high, lower_probability, upper_probability])
        distributions.append(dict(gamma=gamma, mass_lower=lower_mass.tolist(), mass_upper=upper_mass.tolist(),
                                  tail_lower=lower_tail.tolist(), tail_upper=upper_tail.tolist(),
                                  normalizer_lower=lower_norm, normalizer_upper=upper_norm))
        rows.extend([[gamma, j, j/2, float(lower_mass[j]), float(upper_mass[j]), float(lower_tail[j]), float(upper_tail[j])]
                     for j in range(total+1)])
    rank_table = _pair_table(pair_rows)
    all_ranks = torch.zeros(len(adjusted), dtype=torch.int64, device="cpu")
    all_ranks[used] = ranks
    rank_table["null_adjusted_difference"] = adjusted.tolist()
    rank_table["doubled_rank"] = all_ranks.tolist()
    rank_table["informative"] = (adjusted != 0).tolist()
    settings = dict(y=y, treatment=treatment, pair=pair, gammas=gamma_values,
                    alternative=alternative, null_effect=null_effect, missing=missing,
                    device=device, weights=None, max_work=max_work)
    state = dict(target="one-sided paired Wilcoxon signed-rank additive-sharp-null sensitivity tails",
                 pair_rows=pair_rows, differences=differences.tolist(), null_adjusted_differences=adjusted.tolist(),
                 informative_pair_indices=used.tolist(), doubled_ranks=ranks.tolist(), informative_pairs=informative,
                 zero_difference_pairs=len(adjusted)-informative, doubled_statistic=observed,
                 bounds=bounds, distributions=distributions, doubled_support=list(range(total+1)),
                 tie_rule="average ranks of tied absolute null-adjusted differences; doubled integer scores; exact zeros conditioned out",
                 probability_roundoff_policy="verified tail roundoff clipped to0..1; no distribution renormalization or branch truncation",
                 planned_work=work_units, computation_resource_plan=plan, covariance=None,
                 standard_error=None, df=None, confidence_interval=None,
                 inference="complete exact native rank DP with independent assignment odds bounded by Gamma",
                 assumptions=["independent assignment across pairs", "within-pair treatment odds ratios bounded by Gamma",
                              "constant additive treatment-effect null for each unit", "pair availability fixed independently of assignment"],
                 vendor_exact_ties_parity_validated=False)
    return m.result("rosenbaum_rank_bounds", {
        "bounds": m.frame(bounds, columns=["gamma", "signed_rank_statistic", "p_lower", "p_upper", "probability_lower", "probability_upper"]),
        "pairs": rank_table,
        "null_distributions": m.frame(rows, columns=["gamma", "doubled_statistic", "signed_rank_statistic", "mass_lower", "mass_upper", "tail_lower", "tail_upper"]),
    }, metadata, settings, state, notes=[
        "Bounds test a paired signed-rank sharp null; they neither estimate a causal effect nor provide its confidence interval.",
        "Average tied ranks are processed exactly by doubled-score DP; zero differences are explicitly conditioned out.",
        "Numerical underflow of any positive branch fails closed; no incomplete null law is substituted.",
    ])
