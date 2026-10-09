"""Explicit cluster and unequal-Bernoulli Fisher assignment laws.

Primary sources: Branson and Bind, https://arxiv.org/abs/1707.04136;
Su and Ding, https://arxiv.org/abs/2104.04647. The latter distinguishes
unit-average effects based on scaled cluster totals from cluster-average effects.
Only the declared constant additive sharp null is tested; no weak-null ATE
confidence interval follows from these randomization p-values.
"""

from __future__ import annotations

from itertools import combinations
import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate import common as c
from openecon.resources import plan_workspace

from . import common as m
from .observational_survival import _SUM_CAPACITY, _sum_rows
from .paired import _pair_key


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise AnalysisError("numerical_failure", f"{name} exceeds finite float64 representation.")
    return value


def _options(design, required, null_effect, alternative, method, draws, seed,
             missing, device, weights, max_work):
    m.options(device, weights, max_work)
    if not isinstance(design, str) or design != required:
        raise AnalysisError("unsupported_design", f"Explicitly declare design='{required}'.")
    if missing != "raise":
        raise AnalysisError("unsupported_missing", "The full original assignment universe requires missing='raise'; no rows may be deleted.")
    c.check_choice(alternative, "alternative", ("greater", "less", "two-sided"))
    c.check_choice(method, "method", ("exact", "monte_carlo"))
    return (c.check_number(null_effect, "null_effect"),
            c.check_count(draws, "draws", maximum=1_000_000),
            c.check_count(seed, "seed", minimum=0, maximum=2**63-1))


def _names(y, treatment, extra, role):
    names = [c.check_name(value, name) for value, name in
             ((y, "y"), (treatment, "treatment"), (extra, role))]
    if len(set(names)) != len(names):
        raise AnalysisError("invalid_spec", "Outcome, treatment and design role must name distinct columns.")
    return names


def _bounded_label(value, role):
    # Check text before JSON escaping can expand a caller-owned large label.
    if isinstance(value, str) and len(value) > 256:
        raise AnalysisError("resource_limit", f"{role} exceeds the 256-byte escaped-label domain.")
    if isinstance(value, int) and value.bit_length() > 850:
        raise AnalysisError("resource_limit", f"{role} exceeds the 256-byte escaped-label domain.")
    encoded = m.canonical(m._label(value))
    if len(encoded.encode("ascii")) > 256:
        raise AnalysisError("resource_limit", f"{role} exceeds the 256-byte escaped-label domain.")


def _label_preflight(data, names, max_work, *, cluster=None):
    if isinstance(data, m.Dataset):
        raise AnalysisError("unsupported_dataset", "Assignment laws require a resident table; no automatic collection is performed.")
    source = c.source(data)
    if len(source) > 100_000:
        raise AnalysisError("resource_limit", "Assignment-law input is limited to 100000 rows.")
    m.work(96*len(source), max_work, "complete original assignment sample")
    for name in names:
        _bounded_label(name, "Column name")
    for value in source.index:
        _bounded_label(value, "Original index label")
    if cluster is not None and cluster in source:
        for value in source[cluster]:
            _bounded_label(value, "Cluster label")
    # This preflight precedes common.sample's copies and escaped state digest.
    # The later complete assignment plan also retains this bound.
    plan_workspace("bounded assignment input labels and sample state", {
        "original_sample_labels_and_copies": 4096*len(source),
    })


def _binary(frame, treatment, *, both):
    if pd.api.types.is_bool_dtype(frame[treatment].dtype):
        raise AnalysisError("invalid_treatment", "Treatment must be numeric 0/1, not Boolean labels.")
    d = m.tensor(frame, treatment)
    if not bool(((d == 0) | (d == 1)).all()) or (both and d.min() == d.max()):
        raise AnalysisError("invalid_treatment", "Treatment must contain numeric 0/1; cluster randomization requires both arms.")
    return d


def _impute(y, d, null_effect):
    shift = d*null_effect
    value = _finite(y-shift, "Sharp-null imputed outcomes")
    if null_effect != 0 and bool(((d == 1) & (value == y)).any()):
        raise AnalysisError("numerical_failure", "A nonzero sharp-null shift is absorbed at the outcome scale; rescale the outcome and null together.")
    return value


def _count_clusters(g, treated, *, exact, limit, max_work):
    count = 1
    for j in range(1, min(treated, g-treated)+1):
        count = count*(g-j+1)//j
        if exact and count > limit:
            m.work(max_work+1, max_work, "complete cluster assignment support")
        if not exact and count.bit_length() > 1075:
            raise AnalysisError("numerical_failure", "A positive uniform cluster assignment mass is below float64 representation.")
    mass = 1/count
    if mass == 0:
        raise AnalysisError("numerical_failure", "A positive uniform cluster assignment mass underflows.")
    return count, mass


def _plan(n_input, n, units, count, max_work, *, cluster):
    # Every coordinate of the native summation expansion may retain 64 parts.
    # Include all parts and all assignments, even when ordinary data use fewer.
    per_cell = 16*_SUM_CAPACITY+32
    work = 96*n_input + per_cell*n*(count+1) + (16*_SUM_CAPACITY*n if cluster else 0)
    m.work(work, max_work, "complete assignment support, probabilities and expansion statistics")
    resource = plan_workspace("cluster/Bernoulli assignments and complete saved state", {
        "sample_topology_and_null_inputs": 4096*n_input,
        "assignments_live_coefficients_and_contributions": 64*n*(count+1),
        "parallel_expansion_parts": 16*_SUM_CAPACITY*(count+1),
        "complete_assignment_tables_and_state": 192*n*count + 1024*count,
        "design_and_aggregate_state": 4096*units,
    })
    return work, resource.record()


def _probability(bits, probabilities):
    mass = torch.ones(len(bits), dtype=torch.float64, device="cpu")
    for i, probability in enumerate(probabilities):
        mass = mass*torch.where(bits[:, i], probability, 1-probability)
        if bool((mass == 0).any()):
            raise AnalysisError("numerical_failure", "A positive retained assignment probability underflows; the complete law is not truncated.")
    return _finite(mass, "Assignment probabilities")


def _statistics(bits, imputed, *, probabilities=None, cluster_codes=None, n_clusters=None,
                n_treated_clusters=None):
    n = len(imputed)
    scale = float(imputed.abs().max()) or 1.0
    values = imputed/scale
    if bool(((imputed != 0) & (values == 0)).any()):
        raise AnalysisError("numerical_failure", "Null-imputed outcomes underflow after statistic normalization.")
    if probabilities is None:
        assigned = bits[:, cluster_codes]
        positive = n_clusters/(n*n_treated_clusters)
        negative = -n_clusters/(n*(n_clusters-n_treated_clusters))
        coefficient = torch.where(assigned,
            torch.tensor(positive, dtype=torch.float64, device="cpu"),
            torch.tensor(negative, dtype=torch.float64, device="cpu"))
    else:
        coefficient = torch.where(bits, 1/probabilities, -1/(1-probabilities))/n
    terms = _finite(coefficient*values[None, :], "HT statistic contributions")
    nonzero = values[None, :] != 0
    if bool((nonzero & (terms == 0)).any()) or bool((nonzero & (terms*scale == 0)).any()):
        raise AnalysisError("numerical_failure", "A nonzero assignment-statistic contribution underflows; rescale the outcome and null together.")
    statistics = _sum_rows(terms.T)
    reported = _finite(statistics*scale, "Assignment statistics")
    if bool(((statistics != 0) & (reported == 0)).any()):
        raise AnalysisError("numerical_failure", "A nonzero assignment statistic underflows on its reported scale.")
    absolute_sum = terms.abs().sum(dim=1)
    tolerance = 64*torch.finfo(torch.float64).eps*float(absolute_sum.max())
    return statistics, reported, scale, tolerance


def _finish(name, tables, metadata, settings, state, bits, masses, statistics,
            reported, scale, tolerance, *, method, alternative, work, plan,
            unit_bits=None):
    observed = float(statistics[0])
    values = statistics[1:]
    comparison_gap = values.abs()-abs(observed) if alternative == "two-sided" else values-observed
    if bool(((comparison_gap != 0) & (comparison_gap.abs() <= tolerance)).any()):
        raise AnalysisError("numerical_failure", "Distinct assignment comparisons are within the recorded roundoff bound; restrict the outcome dynamic range.")
    if alternative == "greater":
        flags = values >= observed-tolerance
    elif alternative == "less":
        flags = values <= observed+tolerance
    else:
        flags = values.abs() >= abs(observed)-tolerance
    count, total = int(flags.sum()), len(bits)
    normalizer = None
    if method == "exact":
        normalizer = float(masses.sum())
        probability_tolerance = 128*torch.finfo(torch.float64).eps*max(1, bits.shape[1])
        if abs(normalizer-1) > probability_tolerance:
            raise AnalysisError("numerical_failure", "The full exact assignment law failed its normalization certificate.")
        p = 1.0 if count == total else float(masses[flags].sum()) if count else 0.0
        if not -probability_tolerance <= p <= 1+probability_tolerance:
            raise AnalysisError("numerical_failure", "The exact probability tail exceeds its roundoff domain.")
        p = min(1.0, max(0.0, p))
        mc_se = None
    else:
        p = (count+1)/(total+1)
        mc_se = math.sqrt(p*(1-p)*total)/(total+1)
    codes = ["".join("1" if value else "0" for value in row) for row in bits.tolist()]
    unit_codes = codes if unit_bits is None else [
        "".join("1" if value else "0" for value in row) for row in unit_bits.tolist()]
    state.update(observed_statistic=float(reported[0]),
                 normalized_observed_statistic=observed,
                 randomization_statistics=reported[1:].tolist(),
                 normalized_randomization_statistics=values.tolist(),
                 statistic_comparison_scale=scale,
                 normalized_statistic_tie_tolerance=tolerance,
                 assignment_bits=codes, unit_assignment_bits=unit_codes,
                 assignment_probabilities=masses.tolist(),
                 probability_mass_sum=normalizer, extreme=flags.tolist(), n_extreme=count,
                 n_assignments=total, p_value=p, monte_carlo_standard_error=mc_se,
                 monte_carlo_maximum_standard_error=None if method == "exact" else math.sqrt(total/4)/(total+1),
                 covariance=None, standard_error=None, confidence_interval=None, df=None,
                 planned_work=work, computation_resource_plan=plan,
                 sum_expansion_capacity=_SUM_CAPACITY,
                 inference="full exact declared assignment law" if method == "exact" else "independent private-RNG assignment draws; observed-assignment plus-one correction",
                 rng=None if method == "exact" else dict(engine="torch.Generator(cpu)", seed=settings["seed"], torch_version=str(torch.__version__)))
    return m.result(name, {
        "test": m.frame([[float(reported[0]), p, count, total, mc_se]], columns=[
            "statistic", "p_value", "n_extreme", "n_assignments", "monte_carlo_standard_error"]),
        **tables,
        "assignments": m.frame([[i, code, unit_code, float(value), float(mass), bool(flag)]
            for i, (code, unit_code, value, mass, flag) in enumerate(zip(
                codes, unit_codes, reported[1:].tolist(), masses.tolist(), flags.tolist(), strict=True))],
            columns=["draw", "assignment", "unit_assignment", "statistic", "probability", "extreme"]),
    }, metadata, settings, state, notes=[
        "Fisher sharp-null p-values do not give weak-null ATE inference or average-effect confidence intervals.",
        "Monte Carlo SE concerns simulation error only; sampled assignment masses are not renormalized." if method == "monte_carlo" else "All assignments and their original design masses are retained; only verified endpoint probability roundoff is clipped.",
    ])


@m.procedure
def cluster_randomization(data, y, treatment, cluster, *, design, null_effect=0.0,
                          alternative="two-sided", method="exact", draws=9999, seed=1729,
                          missing="raise", device="cpu", weights=None, max_work=100_000_000):
    """Fisher test assigning a fixed observed number of complete clusters uniformly.

    Declare design='cluster_randomized'. Every original cluster receives one
    treatment, with both cluster arms represented. The statistic uses fixed-N
    unit-weighted HT scaled cluster totals, including unequal cluster sizes.
    Unitwise Y-null_effect*D imputation precedes every candidate assignment.
    Exact inference enumerates the full cluster support; MC uses private draws.
    """
    null_effect, draws, seed = _options(design, "cluster_randomized", null_effect,
        alternative, method, draws, seed, missing, device, weights, max_work)
    names = _names(y, treatment, cluster, "cluster")
    _label_preflight(data, names, max_work, cluster=cluster)
    topology, design_meta = m.sample(data, names[1:], numeric=[treatment],
                                   missing="raise", max_work=max_work, cost=96)
    observed_units = _binary(topology, treatment, both=True)
    groups, lookup, codes = [], {}, []
    for i, value in enumerate(topology[cluster]):
        try:
            key, label = _pair_key(value)
        except AnalysisError as exc:
            raise AnalysisError("invalid_cluster", "Cluster identities must be finite typed numeric, Boolean or text scalars.") from exc
        if key not in lookup:
            lookup[key] = len(groups)
            groups.append(dict(label_type=key[0], label=label, positions=[], treatment=int(observed_units[i])))
        group = groups[lookup[key]]
        if group["treatment"] != int(observed_units[i]):
            raise AnalysisError("invalid_cluster", "Each original cluster must have exactly one treatment assignment.")
        group["positions"].append(i)
        codes.append(lookup[key])
    n, g = len(topology), len(groups)
    treated = sum(group["treatment"] for group in groups)
    # Reject the complete support before allocating outcome or assignment blocks.
    cost = (16*_SUM_CAPACITY+32)*n
    limit = max(0, (max_work-96*design_meta["n_input"]-16*_SUM_CAPACITY*n)//cost-1)
    universe, mass = _count_clusters(g, treated, exact=method == "exact", limit=limit, max_work=max_work)
    count = universe if method == "exact" else draws
    work, plan = _plan(design_meta["n_input"], n, g, count, max_work, cluster=True)
    rows, metadata = m.sample(data, names, numeric=[y, treatment],
                              missing="raise", max_work=max_work, cost=96)
    outcome = m.tensor(rows, y)
    imputed = _impute(outcome, observed_units, null_effect)
    cluster_codes = torch.tensor(codes, dtype=torch.int64, device="cpu")
    observed = torch.tensor([group["treatment"] for group in groups], dtype=torch.bool, device="cpu")
    bits = torch.zeros((count, g), dtype=torch.bool, device="cpu")
    if method == "exact":
        for i, chosen in enumerate(combinations(range(g), treated)):
            bits[i, list(chosen)] = True
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for row in bits:
            row[torch.randperm(g, generator=generator, device="cpu")[:treated]] = True
    masses = torch.full((count,), mass, dtype=torch.float64, device="cpu")
    all_bits = torch.cat((observed[None, :], bits), dim=0)
    statistics, reported, scale, tolerance = _statistics(all_bits, imputed,
        cluster_codes=cluster_codes, n_clusters=g, n_treated_clusters=treated)
    totals = [float(_sum_rows(imputed[group["positions"], None])[0]) for group in groups]
    design_rows = [[i, group["label_type"], group["label"], len(group["positions"]),
                    group["treatment"], total] for i, (group, total) in enumerate(zip(groups, totals, strict=True))]
    design_table = m.frame(design_rows, columns=["cluster", "label_type", "label", "n", "treatment", "null_imputed_total"])
    design_table["label"] = pd.Series([group["label"] for group in groups], dtype=object)
    metadata.update(n_clusters=g, n_treated_clusters=treated, design_sample_sha256=design_meta["sample_sha256"],
                    assignment_universe="all original complete clusters; no outcome deletion")
    settings = dict(y=y, treatment=treatment, cluster=cluster, design=design, null_effect=null_effect,
        alternative=alternative, method=method, draws=draws, seed=seed, missing=missing,
        device=device, weights=None, max_work=max_work)
    state = dict(outcomes=outcome.tolist(), treatment=observed_units.tolist(),
        null_imputed_outcomes=imputed.tolist(), clusters=groups, cluster_codes=codes,
        cluster_null_imputed_totals=totals, n_clusters=g, n_treated_clusters=treated,
        assignment_universe_size=universe, observed_assignment_probability=mass,
        target="cluster fixed-count Fisher constant-additive sharp-null p-value",
        statistic="fixed-N unit-weighted HT: G/N times treated-minus-control means of cluster totals",
        assignment_encoding="cluster first-appearance order; unit strings in original row order",
        assumptions=["uniform fixed-count random assignment of entire original clusters",
                     "constant additive null at every unit", "no interference across clusters and no within-cluster treatment versions",
                     "complete original outcome/topology sample"])
    return _finish("cluster_randomization", {"design": design_table}, metadata, settings,
        state, bits, masses, statistics, reported, scale, tolerance, method=method,
        alternative=alternative, work=work, plan=plan, unit_bits=bits[:, cluster_codes])


@m.procedure
def bernoulli_randomization(data, y, treatment, probability, *, design, null_effect=0.0,
                            alternative="two-sided", method="exact", draws=9999, seed=1729,
                            missing="raise", device="cpu", weights=None, max_work=100_000_000):
    """Unconditional Fisher test for independent known unequal Bernoulli assignment.

    Declare design='bernoulli_randomized'; probability names the true, externally
    fixed unit-assignment probabilities, not fitted observational propensities.
    The full-N HT statistic includes all-zero/all-one candidate and observed
    assignments. Exact masses are products of the supplied assignment law;
    private MC draws follow that same law, without fixed-count conditioning.
    """
    null_effect, draws, seed = _options(design, "bernoulli_randomized", null_effect,
        alternative, method, draws, seed, missing, device, weights, max_work)
    names = _names(y, treatment, probability, "probability")
    _label_preflight(data, names, max_work)
    topology, design_meta = m.sample(data, names[1:], numeric=names[1:],
                                   missing="raise", max_work=max_work, cost=96)
    d = _binary(topology, treatment, both=False)
    p = m.tensor(topology, probability)
    if pd.api.types.is_bool_dtype(topology[probability].dtype) or not bool(((p > 0) & (p < 1)).all()):
        raise AnalysisError("invalid_probability", "Known assignment probabilities must be numeric and strictly between zero and one.")
    if bool(((1-p) == 1).any()):
        raise AnalysisError("numerical_failure", "An assignment probability is absorbed when forming its positive complement.")
    n = len(topology)
    if method == "exact":
        cost = (16*_SUM_CAPACITY+32)*n
        limit = max(0, (max_work-96*design_meta["n_input"])//cost-1)
        if n >= limit.bit_length():
            m.work(max_work+1, max_work, "full unconditional Bernoulli assignment support")
        count = 1 << n
    else:
        count = draws
    work, plan = _plan(design_meta["n_input"], n, n, count, max_work, cluster=False)
    rows, metadata = m.sample(data, names, numeric=names, missing="raise", max_work=max_work, cost=96)
    outcome = m.tensor(rows, y)
    imputed = _impute(outcome, d, null_effect)
    observed = d.bool()[None, :]
    observed_mass = float(_probability(observed, p)[0])
    if method == "exact":
        bits = ((torch.arange(count, dtype=torch.int64, device="cpu")[:, None]
                 >> torch.arange(n, dtype=torch.int64, device="cpu")[None, :]) & 1).bool()
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        bits = torch.rand((count, n), dtype=torch.float64, generator=generator, device="cpu") < p
    masses = _probability(bits, p)
    all_bits = torch.cat((observed, bits), dim=0)
    statistics, reported, scale, tolerance = _statistics(all_bits, imputed, probabilities=p)
    metadata.update(design_sample_sha256=design_meta["sample_sha256"],
                    assignment_universe="every original binary vector, including all-zero and all-one; no count conditioning")
    settings = dict(y=y, treatment=treatment, probability=probability, design=design,
        null_effect=null_effect, alternative=alternative, method=method, draws=draws,
        seed=seed, missing=missing, device=device, weights=None, max_work=max_work)
    state = dict(outcomes=outcome.tolist(), treatment=d.tolist(), null_imputed_outcomes=imputed.tolist(),
        unit_treatment_probabilities=p.tolist(), unit_control_probabilities=(1-p).tolist(),
        assignment_support=dict(kind="all_binary_vectors", dimension=n, cardinality_power_of_two=n),
        assignment_universe_size=(1 << n) if n <= 1023 else None,
        observed_assignment_probability=observed_mass,
        target="unconditional independent unequal-Bernoulli Fisher constant-additive sharp-null p-value",
        statistic="full-N Horvitz-Thompson contrast of unitwise null-imputed outcomes",
        assignment_encoding="original sample row order; no conditioning on realized treated count",
        assumptions=["independent Bernoulli assignment with supplied true externally fixed probabilities",
                     "constant additive null at every unit", "consistency and no interference",
                     "complete original outcome/design sample; fitted propensity is not an assignment declaration"])
    design_table = m.frame([[i, int(d[i]), float(p[i]), float(1-p[i]), float(outcome[i]), float(imputed[i])]
        for i in range(n)], columns=["position", "treatment", "probability", "control_probability", "outcome", "null_imputed_outcome"])
    return _finish("bernoulli_randomization", {"design": design_table}, metadata, settings,
        state, bits, masses, statistics, reported, scale, tolerance, method=method,
        alternative=alternative, work=work, plan=plan)
