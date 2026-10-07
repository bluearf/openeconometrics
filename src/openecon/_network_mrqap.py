"""Sparse Freedman--Lane and DSP multiple-regression QAP, with joint tests.

The permutation statistic is partial correlation, not an independent-dyad
standard-error test. The reduced-model residual is relabeled at both endpoints
and projected onto the fixed nuisance regressors again. Adding the reduced fit
back before this projection gives the same statistic (Frisch--Waugh--Lovell).
DSP instead relabels the nuisance-residualized focal predictor and projects
it onto the fixed nuisance regressors again (section 3.2.2). Neither method
is an exact general test under arbitrary network dependence.

Dekker, Krackhardt and Snijders (2007), section 3.2.1:
https://www.stats.ox.ac.uk/~snijders/DekkerKrackhardtSnijders.pdf
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import sys

import pandas as pd
import torch

from openecon._network_qap import _edge_map
from openecon.frame import as_frame
from openecon.networks import Network, _error, _integer, _key


class _Vector:
    """A dyad vector with a constant background and sparse actual overrides."""
    def __init__(self, background, entries):
        self.background, self.entries = background, entries


def _dot(a, b, dyads):
    overlap = 0

    def products():
        nonlocal overlap
        for pair, value in a.entries.items():
            other = b.entries.get(pair, b.background)
            overlap += pair in b.entries
            yield value * other
        for pair, value in b.entries.items():
            if pair not in a.entries:
                yield a.background * value
        missing = dyads - len(a.entries) - len(b.entries) + overlap
        if missing:
            yield missing * a.background * b.background

    return math.fsum(products())


def _norm(vector, dyads):
    squared = _dot(vector, vector, dyads)
    if not math.isfinite(squared) or squared < 0:
        _error("precision", "MRQAP residual norm is not representable reliably in float64.")
    return math.sqrt(squared)


def _linear(vectors, coefficients):
    """Combine actual values directly, avoiding cancellation of cross-products."""
    background = math.fsum(c * v.background for c, v in zip(coefficients, vectors))
    support = set().union(*(v.entries for v in vectors))
    entries = {}
    for pair in sorted(support):
        value = math.fsum(c * v.entries.get(pair, v.background)
                          for c, v in zip(coefficients, vectors))
        if value != background:
            entries[pair] = value
    return _Vector(background, entries)


def _normalize(edges, dyads, intercept, role):
    # Complete almost-constant networks need anchor+offset centering: a direct
    # mean rounds away the displacement of [1, 1+ulp, 1]. Absent dyads are zero.
    anchor = edges[min(edges)] if intercept and len(edges) == dyads else 0.
    offset = math.fsum(value - anchor for value in edges.values()) / dyads if intercept else 0.
    background = -anchor - offset if intercept else 0.
    vector = _Vector(background, {pair: (value - anchor) - offset
                                 for pair, value in sorted(edges.items())})
    norm = _norm(vector, dyads)
    if not norm:
        _error("rank_deficient" if role == "predictor" else "undefined_statistic",
               f"MRQAP {role} has no {'centered ' if intercept else ''}dyad variation.")
    return _Vector(background / norm, {pair: value / norm for pair, value in vector.entries.items()}), norm, (anchor, offset)


def _tensor(values):
    return torch.tensor(values, dtype=torch.float64, device="cpu")


def _solve(factor, rhs):
    return torch.cholesky_solve(_tensor(rhs).reshape(-1, 1), factor).flatten().tolist()


def _residual(vector, nuisance, factor, dyads):
    if not nuisance:
        return vector
    coefficients = _solve(factor, [_dot(x, vector, dyads) for x in nuisance])
    return _linear([vector, *nuisance], [1., *(-value for value in coefficients)])


def _statistic(x_residual, y_residual, x_norm, dyads, minimum_norm=0.):
    y_norm = _norm(y_residual, dyads)
    # A reduced response with no residual variation has no partial correlation.
    # Do not quietly discard undefined permutation draws or change their count.
    if not y_norm:
        _error("undefined_statistic", "A reduced or permuted MRQAP response has no residual "
               "variation; no incomplete permutation test is returned.")
    if y_norm <= minimum_norm:
        _error("precision", "A reduced or permuted MRQAP response is numerically annihilated "
               "by the nuisance projection; its partial correlation cannot be reported reliably.")
    value = (_dot(x_residual, y_residual, dyads) / x_norm) / y_norm
    if not math.isfinite(value) or abs(value) > 1 + 512 * sys.float_info.epsilon:
        _error("precision", "MRQAP partial correlation is not representable reliably in float64.")
    return max(-1., min(1., value))


def _restore(value, exponent):
    try:
        restored = math.ldexp(value, exponent)
    except OverflowError:
        _error("precision", "An MRQAP coefficient exceeds finite float64 output range.")
    if not math.isfinite(restored) or (value and restored == 0.):
        _error("precision", "An MRQAP coefficient is not representable as a finite float64 value.")
    return restored


def _permute(vector, permutation, directed):
    entries = {}
    for (u, v), value in vector.entries.items():
        u, v = permutation[u], permutation[v]
        if not directed and u > v:
            u, v = v, u
        entries[u, v] = value
    return _Vector(vector.background, entries)


def _joint_statistic(features, response, dyads, minimum_norm):
    norm = _norm(response, dyads)
    if norm <= minimum_norm:
        _error("undefined_statistic", "Joint MRQAP response has no reliably representable residual variation.")
    gram = _tensor([[_dot(a, b, dyads) for b in features] for a in features])
    eigenvalues = torch.linalg.eigvalsh(gram)
    if float(eigenvalues[0]) <= 512 * sys.float_info.epsilon * len(features) * float(eigenvalues[-1]):
        _error("rank_deficient", "A joint MRQAP reduced/permuted design is rank deficient; no draws are discarded.")
    factor = torch.linalg.cholesky(gram)
    coefficients = _solve(factor, [_dot(x, response, dyads) for x in features])
    fitted = _linear(features, coefficients)
    value = _dot(fitted, fitted, dyads) / (norm * norm)
    if not math.isfinite(value) or not -1e-10 <= value <= 1 + 1e-10:
        _error("precision", "Joint MRQAP partial R-squared is outside its numerical domain.")
    return max(0., min(1., value))


def _adjust(pvalues, method):
    count = len(pvalues)
    if method == "none":
        return list(pvalues)
    if method == "bonferroni":
        return [min(1., p * count) for p in pvalues]
    order = sorted(range(count), key=lambda i: pvalues[i])
    result = [0.] * count
    if method == "holm":
        previous = 0.
        for at, i in enumerate(order):
            previous = min(1., max(previous, pvalues[i] * (count - at)))
            result[i] = previous
    else:
        previous = 1.
        for at in range(count - 1, -1, -1):
            i = order[at]
            previous = min(previous, pvalues[i] * count / (at + 1))
            result[i] = previous
    return result


def qap_regression(graph, predictors, *, values="weight", include_loops=False,
                   permutations=999, seed=0, alternative="two-sided", intercept=True,
                   method="freedman_lane", joint=None, adjustment="none", max_work=50_000_000):
    """OLS coefficients with coefficient-specific Freedman--Lane MRQAP tests.

    ``predictors`` maps 1--64 distinct names to aligned immutable networks.
    Graphs must share exact typed node IDs (including isolates) and directedness.
    Eligible absent dyads are zeros, undirected dyads count once, and loops are
    opt-in. Positive aggregate weights or binary presence can be used.

    Each coefficient is tested conditional on the other fixed predictors using
    node permutations of its reduced-model response residual. Partial
    correlation is the statistic. P-values use conservative plus-one Monte
    Carlo counting. The intercept is estimated but is not permutation-tested.
    Validity requires a suitable linear model and residual node-exchangeability;
    this is an approximate FL test, not DSP, causal inference, or a universal
    correction for arbitrary network dependence. ``method='dsp'`` permutes the
    focal X residual and projects it against Z again. ``joint`` maps named
    coefficient subsets to shared-permutation upper-tail partial-R-squared
    tests. ``adjustment`` selects none/Bonferroni/Holm/BH over all reported
    coefficient and joint tests, excluding the intercept. BH requires
    independence/PRDS; its validity is not asserted for arbitrary dyad dependence.
    """
    if not isinstance(predictors, Mapping) or not 1 <= len(predictors) <= 64:
        _error("invalid_option", "predictors must map 1--64 distinct names to Network snapshots.")
    for name, other in predictors.items():
        if (not isinstance(name, str) or not name.strip() or len(name) > 128
                or name == "Constant" or any(ord(c) < 32 or 127 <= ord(c) < 160 for c in name)):
            _error("invalid_option", "Predictor names must be bounded nonempty strings; 'Constant' is reserved.")
        try:
            name.encode("utf-8")
        except UnicodeError:
            _error("invalid_option", "Predictor names must be valid UTF-8 strings.")
        if not isinstance(other, Network):
            _error("invalid_option", "Every predictor must be a Network snapshot.")
    if not isinstance(values, str) or values not in ("weight", "binary"):
        _error("invalid_option", "values must be 'weight' or 'binary'.")
    if not isinstance(include_loops, bool) or not isinstance(intercept, bool):
        _error("invalid_option", "include_loops and intercept must be boolean.")
    if not isinstance(alternative, str) or alternative not in ("two-sided", "greater", "less"):
        _error("invalid_option", "alternative must be 'two-sided', 'greater', or 'less'.")
    if method not in ("freedman_lane", "dsp") or adjustment not in ("none", "bonferroni", "holm", "bh"):
        _error("invalid_option", "method must be 'freedman_lane' or 'dsp'; adjustment must be none/bonferroni/holm/bh.")
    permutations = _integer(permutations, "permutations", 1_000_000)
    seed = _integer(seed, "seed", 2**63 - 1, zero=True)
    max_work = _integer(max_work, "max_work", 2**63 - 1)
    names = sorted(predictors)
    groups = {} if joint is None else joint
    if not isinstance(groups, Mapping) or len(groups) > 64:
        _error("invalid_option", "joint must map at most 64 distinct hypothesis names to predictor-name lists.")
    groups = dict(groups)
    for name, terms in groups.items():
        if (not isinstance(name, str) or not name.strip() or len(name) > 128
                or any(ord(c) < 32 or 127 <= ord(c) < 160 for c in name)):
            _error("invalid_option", "Joint hypothesis names must be bounded nonempty strings.")
        try:
            name.encode("utf-8")
        except UnicodeError:
            _error("invalid_option", "Joint hypothesis names must be valid UTF-8 strings.")
        if (not isinstance(terms, (list, tuple)) or not terms or any(not isinstance(t, str) or t not in names for t in terms)
                or len(set(terms)) != len(terms)):
            _error("invalid_option", "Every joint hypothesis must name distinct fitted predictors; the intercept is not tested.")
    groups = {name: sorted(terms) for name, terms in sorted(groups.items())}
    graphs = [graph, *(predictors[name] for name in names)]
    n, p = graph.node_count, len(names)
    for other in graphs[1:]:
        if other.directed != graph.directed:
            _error("invalid_option", "MRQAP networks must have the same directedness.")
        if other.node_count != n or any(label not in other._index for label in graph._labels):
            _error("invalid_label", "MRQAP requires identical exact typed node-label sets, including isolates.")
    dyads = n * (n - 1) if graph.directed else n * (n - 1) // 2
    if include_loops:
        dyads += n
    if dyads <= p + int(intercept):
        _error("undefined_statistic", "MRQAP needs more included dyads than fitted model parameters.")
    if dyads > 2**53:
        _error("precision", "The included dyad count exceeds exact float64 integer range.")
    edges = sum(item.edge_count for item in graphs)
    # Upper-bound all residual supports by the union of supplied edge supports.
    # The full requested randomization workload is admitted before maps/factors
    # or RNG allocation. No dyad subsampling, partial p-value or implicit stop.
    setup_work = 8 * n * (p + 1) + 32 * edges * (p + 1)**2 + 64 * (p + 1)**4 + 256
    per_permutation = 4 * n + 32 * edges * p * (p + 1)**2 + 64 * p * (p + 1)**3 + 128
    work_plan = setup_work + permutations * per_permutation
    group_size = sum(len(terms) for terms in groups.values())
    work_plan += (permutations + 1) * (64 * edges * group_size * (p + 1)**2 + 128 * group_size * (p + 1)**3)
    if work_plan > max_work:
        _error("work_budget", "The complete MRQAP permutation work plan exceeds max_work; "
               "raise the explicit budget or request fewer permutations. No partial test is returned.")
    # Simultaneous live-slot envelope: normalized maps E, retained x/y reduced
    # residual maps 2*p*E, scaled input map E, old/new permuted maps 2*E,
    # old/new projected residual maps 4*E, union/sorted-key buffers 2*E and
    # final residual/result slack E. Each of (2*p+12)*E slots receives 512
    # bytes for pair/boxed endpoints, float, hash growth and allocation slack.
    workspace = 512 * n * (p + 1) + 512 * edges * (2 * p + 12) + 1024 * (p + 1)**3 + 16384
    workspace += 512 * edges * (3 * group_size + 2 * len(groups)) + 1024 * len(groups) * (p + 1)**3
    unique = {id(item): item for item in graphs}
    resident = sum(item._base_bytes for item in unique.values())
    for item in unique.values():
        item._guard(workspace + resident - item._base_bytes)
    canonical = {label: i for i, label in enumerate(sorted(graph._labels, key=_key))}
    with torch.device("cpu"), torch.no_grad():
        vectors, exponents, norms, means = [], [], [], []
        for at, item in enumerate(graphs):
            entries, exponent = _edge_map(item, canonical, values, include_loops)
            vector, norm, mean = _normalize(entries, dyads, intercept, "response" if at == 0 else "predictor")
            vectors.append(vector)
            exponents.append(exponent)
            norms.append(norm)
            means.append(mean)
        response, features = vectors[0], vectors[1:]
        gram = _tensor([[_dot(a, b, dyads) for b in features] for a in features])
        eigenvalues = torch.linalg.eigvalsh(gram)
        rank_tolerance = 512 * sys.float_info.epsilon * p * float(eigenvalues[-1])
        if float(eigenvalues[0]) <= rank_tolerance:
            _error("rank_deficient", "MRQAP predictors are rank deficient or too ill-conditioned "
                   "for reliable float64 sufficient-product factorization; no silent column dropping.")
        condition = float(eigenvalues[-1] / eigenvalues[0])
        residual_tolerance = 512 * sys.float_info.epsilon * p * math.sqrt(condition)
        factor, info = torch.linalg.cholesky_ex(gram)
        if int(info):
            _error("precision", "MRQAP sufficient-product factorization failed.")
        beta = _solve(factor, [_dot(x, response, dyads) for x in features])
        coefficients = [_restore(value * norms[0] / norms[i + 1], exponents[0] - exponents[i + 1])
                        for i, value in enumerate(beta)]
        constant = None
        if intercept:
            # Compute in response-scaled units so large raw predictor means do
            # not overflow before cancellation in the final intercept.
            constant_scaled = math.fsum([*means[0], *(-value * norms[0] / norms[i + 1]
                                        * part for i, value in enumerate(beta) for part in means[i + 1])])
            constant = _restore(constant_scaled, exponents[0])
        tests = []
        for j, x in enumerate(features):
            indices = [i for i in range(p) if i != j]
            nuisance = [features[i] for i in indices]
            reduced_factor = torch.linalg.cholesky(gram[indices][:, indices]) if indices else None
            x_residual = _residual(x, nuisance, reduced_factor, dyads)
            y_residual = _residual(response, nuisance, reduced_factor, dyads)
            x_norm = _norm(x_residual, dyads)
            observed = _statistic(x_residual, y_residual, x_norm, dyads, residual_tolerance)
            permutation_floor = _norm(y_residual, dyads) * residual_tolerance
            tests.append((nuisance, reduced_factor, x_residual, y_residual, x_norm, permutation_floor, observed))
        extreme = [0] * p
        joint_tests = []
        for name, terms in groups.items():
            tested = [names.index(term) for term in terms]
            indices = [i for i in range(p) if i not in tested]
            nuisance = [features[i] for i in indices]
            reduced_factor = torch.linalg.cholesky(gram[indices][:, indices]) if indices else None
            xs = [_residual(features[i], nuisance, reduced_factor, dyads) for i in tested]
            ys = _residual(response, nuisance, reduced_factor, dyads)
            floor = _norm(ys, dyads) * residual_tolerance
            observed = _joint_statistic(xs, ys, dyads, residual_tolerance)
            joint_tests.append((name, nuisance, reduced_factor, xs, ys, floor, observed))
        joint_extreme = [0] * len(joint_tests)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        tolerance = 64 * sys.float_info.epsilon
        for _ in range(permutations):
            permutation_tensor = torch.randperm(n, dtype=torch.int64, device="cpu", generator=generator)
            permutation = memoryview(permutation_tensor.numpy())
            for j, (nuisance, reduced_factor, x_residual, y_residual, x_norm, permutation_floor, observed) in enumerate(tests):
                focal = y_residual if method == "freedman_lane" else x_residual
                permuted = _permute(focal, permutation, graph.directed)
                residual = _residual(permuted, nuisance, reduced_factor, dyads)
                if method == "freedman_lane":
                    statistic = _statistic(x_residual, residual, x_norm, dyads, permutation_floor)
                else:
                    permuted_norm = _norm(residual, dyads)
                    if permuted_norm <= x_norm * residual_tolerance:
                        _error("rank_deficient", "A DSP focal residual is annihilated by the nuisance projection; no draws discarded.")
                    statistic = _statistic(residual, y_residual, permuted_norm, dyads, permutation_floor)
                if alternative == "two-sided":
                    extreme[j] += abs(statistic) >= abs(observed) - tolerance
                elif alternative == "greater":
                    extreme[j] += statistic >= observed - tolerance
                else:
                    extreme[j] += statistic <= observed + tolerance
            for j, (_, nuisance, reduced_factor, xs, ys, floor, observed) in enumerate(joint_tests):
                if method == "freedman_lane":
                    py = _residual(_permute(ys, permutation, graph.directed), nuisance, reduced_factor, dyads)
                    statistic = _joint_statistic(xs, py, dyads, floor)
                else:
                    px = [_residual(_permute(x, permutation, graph.directed), nuisance, reduced_factor, dyads) for x in xs]
                    statistic = _joint_statistic(px, ys, dyads, floor)
                joint_extreme[j] += statistic >= observed - tolerance
        full_residual = _linear([response, *features], [1., *(-value for value in beta)])
        residual_fraction = _dot(full_residual, full_residual, dyads)
    rows = [{"term": name, "coefficient": coefficients[j], "statistic": tests[j][-1],
             "pvalue": (extreme[j] + 1) / (permutations + 1), "permutations": permutations,
             "extreme_permutations": extreme[j], "dyads": dyads, "nodes": n}
            for j, name in enumerate(names)]
    if intercept:
        rows.append({"term": "Constant", "coefficient": constant, "statistic": None,
                     "pvalue": None, "permutations": None, "extreme_permutations": None,
                     "dyads": dyads, "nodes": n})
    result = as_frame(pd.DataFrame(rows))
    joint_rows = [dict(hypothesis=name, terms=groups[name], statistic=observed,
                      statistic_name="partial R-squared", alternative="greater",
                      pvalue=(joint_extreme[j] + 1) / (permutations + 1), extreme_permutations=joint_extreme[j])
                  for j, (name, _, _, _, _, _, observed) in enumerate(joint_tests)]
    adjusted = _adjust([row["pvalue"] for row in rows[:p]] + [row["pvalue"] for row in joint_rows], adjustment)
    if adjustment != "none":
        result["adjusted_pvalue"] = [*adjusted[:p], *([None] if intercept else [])]
    for j, row in enumerate(joint_rows):
        row["adjusted_pvalue"] = adjusted[p + j]
    for column in ("permutations", "extreme_permutations"):
        result[column] = pd.array(result[column], dtype="Int64")
    result.attrs.update(kind="network_qap_regression", algorithm="sparse Freedman-Lane semi-partialing MRQAP",
        values=values, weight_semantics="aggregate positive strength" if values == "weight" else "positive-edge presence",
        directed=graph.directed, include_loops=include_loops, intercept=intercept,
        absent_dyads="zero", undirected_dyads="unordered; each pair counted once",
        terms=names, rank=p + int(intercept), residual_degrees_of_freedom=dyads - p - int(intercept),
        model_condition_number=condition, rank_tolerance=rank_tolerance, residual_tolerance=residual_tolerance,
        fit_r_squared=1 - residual_fraction, r_squared_definition="centered" if intercept else "uncentered",
        statistic="partial correlation", statistic_scope="conditional on the other fixed predictors",
        statistic_centering="centered residual correlation" if intercept else "uncentered residual cosine; through-origin model",
        intercept_inference="estimated only; no permutation test for the intercept",
        method="Freedman-Lane coefficient-specific reduced-response residual permutation",
        permutation="uniform with replacement; one shared node permutation per draw at both residual endpoints",
        null="Zero tested coefficient conditional on fixed nuisance predictors and residual node-exchangeability",
        inference_scope="approximate Freedman-Lane MRQAP; no exact generic dependence, DSP or causal claim",
        seed=seed, alternative=alternative, permutations=permutations,
        rng="private CPU Torch generator; canonical typed-label and sorted predictor order",
        pvalue_method="(1 + extreme_permutations) / (1 + permutations)", pvalue_resolution=1 / (permutations + 1),
        comparison_tolerance=tolerance, multiplicity_adjustment="none; per-coefficient p-values",
        scaled_weight_exponents=exponents, device="cpu", dtype="float64", torch_version=torch.__version__,
        memory_scope="O(V + p*E + p^3) workspace plus all resident snapshots under every owned budget; no dense dyad vectors",
        work_used=work_plan, work_estimate=work_plan, work_counting="conservative admitted operation plan", max_work=max_work)
    result.attrs.update(test_method=method, joint_tests=joint_rows, adjustment=adjustment,
        adjustment_family="all reported coefficient and joint hypotheses; intercept excluded",
        multiplicity_adjustment=adjustment if adjustment != "none" else result.attrs["multiplicity_adjustment"],
        multiplicity_assumptions="BH requires independence/PRDS of valid marginal tests; no arbitrary-dependence FDR guarantee" if adjustment == "bh"
            else ("No multiplicity adjustment applied" if adjustment == "none" else "Holm/Bonferroni control FWER when the marginal permutation tests are valid"),
        joint_null="all named coefficients zero conditional on remaining nuisance predictors",
        joint_statistic="partial R-squared; monotone in the joint F statistic at fixed group size/residual df")
    if method == "dsp":
        result.attrs.update(algorithm="sparse double semi-partialing MRQAP",
            method="DSP: permute nuisance-residualized focal X, then regress Y on permuted residual X and fixed Z",
            null="Zero tested coefficient conditional on nuisance predictors and focal-residual node-exchangeability",
            inference_scope="approximate DSP MRQAP; no exact arbitrary-dependence or causal claim",
            permutation="one uniform shared node permutation at both focal residual endpoints; nuisance projected again")
    return result
