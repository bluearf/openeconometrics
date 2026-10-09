"""Descriptive profiles, exact singleton merge-loss quality and partition ARI."""

from __future__ import annotations

import math
from collections import Counter

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.resources import plan_workspace, workspace_budget_bytes

from . import kernel

DEFAULT_WORK = 300_000_000
DEFAULT_BYTES = 128 * 1024**2


def _admit(operation, work, buffers, max_work, max_bytes, *, validation=(0, 0)):
    validation_work, validation_bytes = validation
    work += validation_work
    # Validation temporaries are released before the helper's allocations.
    # Account for the larger of the two peaks, not their simultaneous sum.
    buffers = dict(buffers)
    buffers["prior_validation_peak_margin"] = max(0, validation_bytes - sum(buffers.values()))
    if work > max_work:
        raise AnalysisError("resource_limit", f"{operation} needs {work} declared work units, exceeding max_work.")
    plan = plan_workspace(operation, buffers, budget_bytes=min(max_bytes, workspace_budget_bytes()))
    return {"work_bound": work, "validation_work": validation_work, "workspace": plan.record()}


def _finite(value):
    if not math.isfinite(value):
        raise AnalysisError("numerical_failure", "TwoStep descriptive arithmetic exceeds finite float64.")
    return value


def _dimensions(result):
    """Read only shape metadata admitted by public._validation_cost."""
    state = result.attrs["twostep_state"]
    k = state.get("selected_k")
    noise = state.get("noise_positions")
    if type(k) is not int or not 1 <= k <= len(state["cuts"]) or not isinstance(noise, list) or len(noise) >= len(state["positions"]):
        raise AnalysisError("invalid_state", "TwoStep selected partition dimensions are invalid.")
    return k, len(state["positions"]) - len(noise), len(state["continuous"]) + sum(len(d["levels"]) for d in state["categorical"]) + 1


def profile_tables(state):
    """Construct selected-cut descriptive tables from already validated CFs.

    Population standard deviations use M2/N and training rescaling. They are
    distributions of the clustered observations, without inferential errors.
    """
    cut = state["cuts"][str(state["selected_k"])]
    total = sum(cf["count"] for cf in cut)
    sizes, continuous, categorical = [], [], []
    for label, cf in enumerate(cut, 1):
        sizes.append([label, cf["count"], cf["count"] / total])
        for j, scaling in enumerate(state["scaling"]):
            mean = _finite(scaling["origin"] + (scaling["center"] + scaling["sd"] * cf["mean"][j]))
            sd = _finite(scaling["sd"] * math.sqrt(cf["m2"][j] / cf["count"]))
            continuous.append([label, scaling["name"], cf["count"], mean, sd])
        for j, descriptor in enumerate(state["categorical"]):
            for level, count in zip(descriptor["levels"], cf["categorical_counts"][j]):
                categorical.append([label, descriptor["name"], level[0], level[1], count, count / cf["count"]])
    return {
        "sizes": table(sizes, columns=["cluster", "count", "proportion"]),
        "continuous_profiles": table(continuous, columns=["cluster", "variable", "count", "mean", "population_sd"]),
        "categorical_profiles": table(categorical, columns=["cluster", "variable", "level_type", "level_value", "count", "proportion"]),
    }


@resident_cpu
def twostep_profiles(result, *, device="cpu", weights=None, max_work=DEFAULT_WORK, max_bytes=DEFAULT_BYTES) -> TableSet:
    """Describe the selected partition in original continuous units.

    Counts exclude missing records and recorded small-leaf noise. Population
    standard deviations are descriptive; no SE, covariance, p-value or CI is
    defined for these profiles.
    """
    from .public import _domain, _saved, _state, _validation_cost

    max_work, max_bytes = _domain(device, weights, max_work, max_bytes)
    validation = _validation_cost(result)
    k, _, dimensions = _dimensions(result)
    resources = _admit("TwoStep descriptive profiles", 16 * k * dimensions,
                       {"profile_scalar_buffers": 128 * k * dimensions}, max_work, max_bytes,
                       validation=validation)
    state = _state(result, max_work=max_work, max_bytes=max_bytes)
    return _saved(TableSet(profile_tables(state), title="TwoStep descriptive profiles",
                           method="twostep_profiles", training_state_sha256=result.attrs["state_sha256"],
                           n_input=state["input_n"], n_missing=len(state["missing_positions"]),
                           n_noise=len(state["noise_positions"]), n_clustered=sum(cf["count"] for cf in state["cuts"][str(k)]),
                           n_clusters=k, inference="descriptive; covariance/SE/df/p/CI are not defined",
                           population_sd="sqrt(M2/N), rescaled to original continuous units",
                           device="cpu", dtype="float64", resources=resources))


def _partition(state):
    return {position: label for label, cf in enumerate(state["cuts"][str(state["selected_k"])], 1)
            for position in cf["rows"]}


@resident_cpu
def twostep_quality(result, *, device="cpu", weights=None, max_work=DEFAULT_WORK, max_bytes=DEFAULT_BYTES) -> TableSet:
    """Exact silhouettes using singleton CF pair merge loss, for at most 500 rows.

    Quality uses the saved training observations and selected training cut.
    Missing and noise rows are excluded. A singleton has silhouette zero;
    zero within/between dissimilarities also give zero. One cluster has no
    defined silhouette and is rejected. This is a descriptive diagnostic.
    """
    from .public import _domain, _saved, _state, _validation_cost

    max_work, max_bytes = _domain(device, weights, max_work, max_bytes)
    validation = _validation_cost(result)
    k, n, dimensions = _dimensions(result)
    if n > 500:
        raise AnalysisError("resource_limit", "Exact TwoStep quality accepts at most 500 nonnoise training observations.")
    if k < 2:
        raise AnalysisError("undefined_quality", "Silhouette is undefined for a one-cluster partition.")
    resources = _admit("TwoStep exact singleton-pair quality", 32 * (n * (n - 1) // 2) * dimensions + 8 * n * n,
                       {"pair_dissimilarity": 8 * n * n,
                        "saved_singleton_cf_copies": 128 * n * dimensions,
                        "labels_and_quality_buffers": 128 * n + 8 * n * k}, max_work, max_bytes,
                       validation=validation)
    state = _state(result, max_work=max_work, max_bytes=max_bytes)
    labels = _partition(state)
    positions = sorted(labels)
    complete_index = {position: i for i, position in enumerate(state["positions"])}
    levels = tuple(len(d["levels"]) for d in state["categorical"])
    observations = []
    for position in positions:
        i = complete_index[position]
        observations.append(kernel.singleton(torch.tensor(state["training_numeric"][i], dtype=torch.float64),
                                             torch.tensor(state["training_codes"][i], dtype=torch.int64), levels, position))
    globalvar = torch.tensor(state["global_variance"], dtype=torch.float64)
    distances = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        for j in range(i):
            distances[i, j] = distances[j, i] = kernel.merge_loss(observations[i], observations[j], globalvar)
    members = {label: [i for i, position in enumerate(positions) if labels[position] == label] for label in range(1, k + 1)}
    rows, grouped = [], {label: [] for label in members}
    for i, position in enumerate(positions):
        label = labels[position]
        own = members[label]
        a = float(distances[i, own].sum()) / (len(own) - 1) if len(own) > 1 else 0.0
        b = min(float(distances[i, group].mean()) for other, group in members.items() if other != label)
        silhouette = (b - a) / max(a, b) if len(own) > 1 and max(a, b) > 0 else 0.0
        rows.append([position, label, _finite(a), _finite(b), _finite(silhouette)])
        grouped[label].append(silhouette)
    mean = _finite(math.fsum(row[4] for row in rows) / n)
    return _saved(TableSet({
        "silhouettes": table(rows, columns=["position", "cluster", "within", "nearest_other", "silhouette"]),
        "quality": table([[n, k, mean]], columns=["n_clustered", "clusters", "mean_silhouette"]),
        "cluster_quality": table([[label, len(values), math.fsum(values) / len(values)] for label, values in grouped.items()],
                                 columns=["cluster", "count", "mean_silhouette"]),
    }, title="Exact TwoStep silhouette quality", method="twostep_quality",
        training_state_sha256=result.attrs["state_sha256"], dissimilarity="singleton-pair CF merge loss",
        n_input=state["input_n"], n_missing=len(state["missing_positions"]), n_noise=len(state["noise_positions"]),
        inference="descriptive; covariance/SE/df/p/CI are not defined", singleton_silhouette=0,
        device="cpu", dtype="float64", resources=resources))


def adjusted_rand(left, right):
    """Exact integer contingency arithmetic, invariant to cluster labels."""
    if len(left) != len(right) or len(left) < 2:
        raise AnalysisError("insufficient_sample", "Adjusted Rand requires at least two aligned observations.")
    cells = Counter(zip(left, right))
    rows, columns = Counter(left), Counter(right)
    def choose(n):
        return n * (n - 1) // 2

    total = choose(len(left))
    observed = sum(choose(n) for n in cells.values())
    row_pairs = sum(choose(n) for n in rows.values())
    column_pairs = sum(choose(n) for n in columns.values())
    denominator = total * (row_pairs + column_pairs) - 2 * row_pairs * column_pairs
    score = 2 * (total * observed - row_pairs * column_pairs) / denominator if denominator else 1.0
    return _finite(score), cells


@resident_cpu
def twostep_stability(results, *, device="cpu", weights=None, max_work=DEFAULT_WORK, max_bytes=DEFAULT_BYTES) -> TableSet:
    """Compare 2–10 fitted partitions of the same physical input corpus by ARI.

    Every result pair is aligned on the intersection of nonnoise rows. Saved
    typed raw-input fingerprints, role lists and complete physical positions
    must match. Exclusions are explicit; this comparison does not resample or
    refit and makes no claim of order invariance or statistical uncertainty.
    """
    from .public import _domain, _saved, _state, _validation_cost

    max_work, max_bytes = _domain(device, weights, max_work, max_bytes)
    if not isinstance(results, list) or not 2 <= len(results) <= 10:
        raise AnalysisError("invalid_option", "Supply a list of 2–10 fitted TwoStep results.")
    validation = [_validation_cost(result) for result in results]
    m, n = len(results), len(results[0].attrs["twostep_state"]["positions"])
    resources = _admit("TwoStep adjusted Rand partition stability", 32 * n * (m * (m - 1) // 2),
                       {"aligned_partition_and_contingency_buffers": 128 * n * m * m}, max_work, max_bytes,
                       validation=(sum(cost[0] for cost in validation), max(cost[1] for cost in validation)))
    states = [_state(result, max_work=max_work, max_bytes=max_bytes) for result in results]
    baseline = states[0]
    for state in states[1:]:
        if (state["sample_sha256"] != baseline["sample_sha256"] or state["input_n"] != baseline["input_n"]
                or state["positions"] != baseline["positions"] or state["continuous"] != baseline["continuous"]
                or [d["name"] for d in state["categorical"]] != [d["name"] for d in baseline["categorical"]]):
            raise AnalysisError("sample_mismatch", "TwoStep stability needs the same typed raw input corpus, feature roles and physical positions.")
    partitions = [_partition(state) for state in states]
    pairs, contingency = [], []
    for left in range(m):
        for right in range(left + 1, m):
            common = sorted(set(partitions[left]) & set(partitions[right]))
            a, b = [partitions[left][p] for p in common], [partitions[right][p] for p in common]
            ari, cells = adjusted_rand(a, b)
            pairs.append([left + 1, right + 1, len(common), states[left]["selected_k"], states[right]["selected_k"],
                          n - len(partitions[left]), n - len(partitions[right]), n - len(common), ari])
            contingency.extend([left + 1, right + 1, label_a, label_b, count] for (label_a, label_b), count in sorted(cells.items()))
    return _saved(TableSet({
        "pairs": table(pairs, columns=["left", "right", "n_common", "left_clusters", "right_clusters", "left_noise", "right_noise", "excluded_union", "adjusted_rand"]),
        "contingency": table(contingency, columns=["left", "right", "left_cluster", "right_cluster", "count"]),
    }, title="TwoStep partition stability", method="twostep_stability", comparisons=len(pairs),
        sample_sha256=baseline["sample_sha256"], n_input=baseline["input_n"], n_complete=n,
        n_missing=len(baseline["missing_positions"]), alignment="common nonnoise physical positions per pair",
        degenerate_ari="1 when both partitions have the same trivial equivalence relation",
        inference="descriptive; covariance/SE/df/p/CI are not defined", device="cpu", dtype="float64", resources=resources))
