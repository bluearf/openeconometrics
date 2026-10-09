"""Independent raw-row references for bounded mixed-data TwoStep clustering.

Equations are transcribed from IBM's primary TwoStep algorithms guide pp3–5:
https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/twostep_cluster.pdf
The native contract uses population second moments and a global-minimum IC
selection policy.  It does not claim IBM's automatic distance-jump heuristic.
References recompute summaries and candidate merges from raw NumPy rows,
without using the production score, merge, heap or CF-tree algorithms.
"""

from itertools import combinations
from copy import deepcopy
import math

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.twostep import kernel


def reference_xi(numeric, codes, levels, global_variance):
    n = len(numeric)
    continuous = math.fsum(
        0.5 * math.log(float(global_variance[column]) + float(np.var(numeric[:, column], ddof=0)))
        for column in range(numeric.shape[1])
    )
    entropy = 0.0
    for column, cardinality in enumerate(levels):
        counts = np.bincount(codes[:, column], minlength=cardinality)
        entropy += math.fsum(-(count / n) * math.log(count / n) for count in counts if count)
    return -n * (continuous + entropy)


def reference_loss(numeric, codes, levels, global_variance, left, right):
    together = sorted([*left, *right])
    return math.fsum((
        reference_xi(numeric[left], codes[left], levels, global_variance),
        reference_xi(numeric[right], codes[right], levels, global_variance),
        -reference_xi(numeric[together], codes[together], levels, global_variance),
    ))


def raw_cf(numeric, codes, levels, rows, positions=None):
    values = numeric[rows]
    mean = values.mean(0)
    m2 = ((values - mean) ** 2).sum(0)
    counts = tuple(torch.tensor(np.bincount(codes[rows, column], minlength=cardinality), dtype=torch.int64)
                   for column, cardinality in enumerate(levels))
    physical = rows if positions is None else [positions[row] for row in rows]
    return kernel.CF(len(rows), torch.tensor(mean, dtype=torch.float64),
                     torch.tensor(m2, dtype=torch.float64), counts, tuple(sorted(physical)))


def reference_hierarchy(numeric, codes, levels, global_variance, precluster_rows):
    """Brute-force all active raw-row pairs at every merge; no distance cache."""
    ordered = sorted((sorted(rows) for rows in precluster_rows), key=min)
    active = {i: rows for i, rows in enumerate(ordered)}
    cuts = {len(active): tuple(tuple(rows) for rows in ordered)}
    result = []
    for new_id in range(len(active), 2 * len(active) - 1):
        candidates = []
        for first, second in combinations(active, 2):
            left, right = sorted((first, second), key=lambda key: min(active[key]))
            distance = reference_loss(numeric, codes, levels, global_variance, active[left], active[right])
            candidates.append((distance, min(active[left]), min(active[right]), left, right))
        distance, _, _, left, right = min(candidates)
        rows = sorted([*active.pop(left), *active.pop(right)])
        active[new_id] = rows
        result.append({"left": left, "right": right, "merged": new_id, "distance": distance,
                       "count": len(rows), "clusters_after": len(active)})
        cuts[len(active)] = tuple(tuple(rows) for rows in sorted(active.values(), key=min))
    return cuts, result


def reference_ic(numeric, codes, levels, global_variance, cut):
    score = math.fsum(reference_xi(numeric[list(rows)], codes[list(rows)], levels, global_variance) for rows in cut)
    parameters = len(cut) * (2 * numeric.shape[1] + sum(cardinality - 1 for cardinality in levels))
    n = sum(len(rows) for rows in cut)
    return score, parameters, -2 * score + 2 * parameters, -2 * score + parameters * math.log(n)


@pytest.fixture
def mixed_rows():
    rng = np.random.default_rng(9371)
    x = rng.normal(size=(18, 2)) * [2.0, 0.7] + [3.0, -1.0]
    codes = np.column_stack((np.arange(18) % 3, (np.arange(18) // 3) % 2)).astype(np.int64)
    return x, codes, (3, 2), np.var(x, axis=0, ddof=0)


@pytest.mark.parametrize("family", ["continuous", "categorical", "mixed"])
def test_cf_score_equals_raw_population_variance_and_multinomial_entropy(mixed_rows, family):
    x, codes, levels, global_variance = mixed_rows
    if family == "continuous":
        codes, levels = np.empty((len(x), 0), dtype=np.int64), ()
    elif family == "categorical":
        x, global_variance = np.empty((len(x), 0)), np.empty(0)
    for rows in ([0], [0, 1, 2], [3, 6, 9, 12], list(range(len(x)))):
        summary = raw_cf(x, codes, levels, rows)
        expected = reference_xi(x[rows], codes[rows], levels, global_variance)
        assert kernel.xi(summary, torch.tensor(global_variance, dtype=torch.float64)) == pytest.approx(expected, rel=1e-13, abs=1e-13)


def test_cf_merge_equals_direct_raw_summary_for_unequal_sizes_and_physical_row_ids(mixed_rows):
    x, codes, levels, _ = mixed_rows
    left, right = [0, 3], [1, 7, 8, 13, 17]
    positions = [200 - 3 * i for i in range(len(x))]
    first, second = raw_cf(x, codes, levels, left, positions), raw_cf(x, codes, levels, right, positions)
    actual = kernel.merge(first, second)
    expected = raw_cf(x, codes, levels, [*left, *right], positions)
    assert actual.count == len(left) + len(right)
    assert actual.rows == expected.rows
    np.testing.assert_allclose(actual.mean.numpy(), expected.mean.numpy(), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual.m2.numpy(), expected.m2.numpy(), rtol=1e-13, atol=1e-13)
    for observed, counts in zip(actual.categorical_counts, expected.categorical_counts, strict=True):
        np.testing.assert_array_equal(observed.numpy(), counts.numpy())


@pytest.mark.parametrize("left,right", [([0], [1]), ([0, 3], [1, 7, 8, 13, 17]), ([0, 3, 6], [1, 4, 7, 10])])
def test_cf_merge_distance_is_independent_raw_score_loss(mixed_rows, left, right):
    x, codes, levels, global_variance = mixed_rows
    expected = reference_loss(x, codes, levels, global_variance, left, right)
    actual = kernel.merge_loss(raw_cf(x, codes, levels, left), raw_cf(x, codes, levels, right),
                               torch.tensor(global_variance, dtype=torch.float64))
    assert actual == pytest.approx(expected, rel=1e-13, abs=1e-13)
    assert actual >= 0


def test_cf_heap_hierarchy_matches_complete_pair_raw_reference_at_every_cut(mixed_rows):
    x, codes, levels, global_variance = mixed_rows
    groups = [[0, 3], [1, 2, 8], [4, 5], [6, 7], [9, 10, 11, 12], [13, 14, 15, 16, 17]]
    expected_cuts, expected_merges = reference_hierarchy(x, codes, levels, global_variance, groups)
    summaries = [raw_cf(x, codes, levels, rows) for rows in reversed(groups)]
    actual = kernel.agglomerate(summaries, torch.tensor(global_variance, dtype=torch.float64))
    assert set(actual.cuts) == set(expected_cuts)
    for k, cut in actual.cuts.items():
        assert tuple(cf.rows for cf in cut) == expected_cuts[k]
        for summary in cut:
            expected = raw_cf(x, codes, levels, list(summary.rows))
            np.testing.assert_allclose(summary.mean.numpy(), expected.mean.numpy(), rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(summary.m2.numpy(), expected.m2.numpy(), rtol=1e-13, atol=1e-13)
    assert len(actual.merges) == len(expected_merges)
    for observed, expected in zip(actual.merges, expected_merges, strict=True):
        assert {k: v for k, v in observed.items() if k != "distance"} == {k: v for k, v in expected.items() if k != "distance"}
        assert observed["distance"] == pytest.approx(expected["distance"], rel=1e-13, abs=1e-13)


def test_cf_hierarchy_deterministic_exact_distance_ties_use_first_physical_rows():
    x = np.array([[-1.], [-1.], [1.], [1.]])
    codes, levels = np.empty((4, 0), dtype=np.int64), ()
    groups = [[0], [1], [2], [3]]
    expected_cuts, expected_merges = reference_hierarchy(x, codes, levels, np.array([1.]), groups)
    actual = kernel.agglomerate([raw_cf(x, codes, levels, rows) for rows in groups], torch.tensor([1.], dtype=torch.float64))
    assert [(m["left"], m["right"]) for m in actual.merges] == [(m["left"], m["right"]) for m in expected_merges]
    assert {k: tuple(cf.rows for cf in value) for k, value in actual.cuts.items()} == expected_cuts


def fit_public(data, **options):
    defaults = {"max_preclusters": 32, "max_nodes": 128}
    defaults.update(options)
    return oe.twostep(data=data, **defaults)


def raw_public_encoding(data, state):
    retained = data.iloc[state["positions"]]
    raw = retained[state["continuous"]].to_numpy(dtype=float)
    means, scales = raw.mean(0), np.sqrt(np.var(raw, axis=0, ddof=0))
    numeric = (raw - means) / scales
    category_names = [descriptor["name"] for descriptor in state["categorical"]]
    code_columns, cardinalities = [], []
    kind = {bool: "bool", int: "int", float: "float", str: "str"}
    for name, descriptor in zip(category_names, state["categorical"], strict=True):
        values = retained[name].to_numpy(dtype=object)
        keys = [(kind[type(value)], value) for value in values]
        ordered = list(dict.fromkeys(keys))
        assert descriptor["levels"] == [[label_type, value] for label_type, value in ordered]
        encoded = [ordered.index(key) for key in keys]
        assert descriptor["counts"] == np.bincount(encoded, minlength=len(ordered)).tolist()
        code_columns.append(encoded)
        cardinalities.append(len(ordered))
    codes = np.column_stack(code_columns).astype(np.int64) if code_columns else np.empty((len(retained), 0), dtype=np.int64)
    for i, entry in enumerate(state["scaling"]):
        assert entry["mean"] == pytest.approx(means[i], rel=1e-13, abs=1e-13)
        assert entry["sd"] == pytest.approx(scales[i], rel=1e-13, abs=1e-13)
    np.testing.assert_allclose(state["training_numeric"], numeric, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(state["training_codes"], codes)
    global_variance = np.var(numeric, axis=0, ddof=0)
    np.testing.assert_allclose(state["global_variance"], global_variance, rtol=1e-13, atol=1e-13)
    return numeric, codes, tuple(cardinalities), global_variance


@pytest.mark.parametrize("criterion", ["bic", "aic"])
def test_public_every_cut_information_criterion_uses_cf_score_and_declared_parameter_count(mixed_rows, criterion):
    x, codes, _, _ = mixed_rows
    data = pd.DataFrame({"x": x[:, 0], "u": x[:, 1], "g": codes[:, 0], "h": codes[:, 1]})
    result = fit_public(data, continuous=["x", "u"], categorical=["g", "h"], criterion=criterion, max_clusters=18)
    state = result.attrs["twostep_state"]
    numeric, encoded, levels, global_variance = raw_public_encoding(data, state)
    physical_to_local = {position: i for i, position in enumerate(state["positions"])}
    expected_criteria = []
    for row in result["criteria"].itertuples(index=False):
        k = int(row.clusters)
        cut = [[physical_to_local[position] for position in cf["rows"]] for cf in state["cuts"][str(k)]]
        score, parameters, aic, bic = reference_ic(numeric, encoded, levels, global_variance, cut)
        assert row.score == pytest.approx(score, rel=1e-12, abs=1e-12)
        # The IBM count is J*(2p+sum(L-1)); no additional mixture proportions.
        assert row.parameters == parameters
        assert row.aic == pytest.approx(aic, rel=1e-12, abs=1e-12)
        assert row.bic == pytest.approx(bic, rel=1e-12, abs=1e-12)
        expected_criteria.append((bic if criterion == "bic" else aic, k))
    assert state["selected_k"] == min(expected_criteria)[1]
    assert result.attrs["selection"] == "global-min information criterion"
    assert any("differs" in note and "IBM" in note for note in result.attrs["notes"])


@pytest.mark.parametrize("criterion", ["bic", "aic"])
def test_information_criterion_exact_tie_selects_smaller_cluster_count(criterion):
    if criterion == "bic":
        values = [-1., 1.]
    else:
        # Two 3-row groups have within/global variance r.  Their score gain
        # is 3*log(2/(1+r))=2, exactly offsetting the extra 2-parameter AIC
        # penalty.  This produces tied 1- and 2-cluster criterion values.
        r = 2 * math.exp(-2 / 3) - 1
        a, c = math.sqrt(1 - r), math.sqrt(1.5 * r)
        values = [-a-c, -a, -a+c, a-c, a, a+c]
    result = fit_public(pd.DataFrame({"x": values}), continuous=["x"], criterion=criterion, max_clusters=len(values))
    candidates = result["criteria"].set_index("clusters")[criterion]
    assert candidates.loc[1] == candidates.loc[2]
    assert result.attrs["twostep_state"]["selected_k"] == 1


def test_public_hierarchy_preserves_original_physical_rows_after_missing_drop_and_random_order(mixed_rows):
    x, codes, _, _ = mixed_rows
    data = pd.DataFrame({"x": x[:, 0], "g": codes[:, 0]}, index=np.repeat([9, 4, 1], 6))
    data.iloc[3, data.columns.get_loc("x")] = np.nan
    data.iloc[8, data.columns.get_loc("g")] = np.nan
    original = data.copy(deep=True)
    result = fit_public(data, continuous=["x"], categorical=["g"], n_clusters=3,
                        missing="drop", order="random", seed=413)
    state = result.attrs["twostep_state"]
    keep = [i for i in range(len(data)) if i not in [3, 8]]
    assert state["positions"] == keep and state["missing_positions"] == [3, 8]
    assert sorted(state["order"]) == keep and state["order"] != keep
    numeric, encoded, levels, global_variance = raw_public_encoding(data, state)
    physical_to_local = {position: i for i, position in enumerate(keep)}
    preclusters = [[physical_to_local[position] for position in cf["rows"]] for cf in state["preclusters"]]
    expected_cuts, expected_merges = reference_hierarchy(numeric, encoded, levels, global_variance, preclusters)
    for k, cut in state["cuts"].items():
        actual = tuple(tuple(physical_to_local[position] for position in cf["rows"]) for cf in cut)
        assert actual == expected_cuts[int(k)]
    for actual, expected in zip(state["merges"], expected_merges, strict=True):
        assert (actual["left"], actual["right"], actual["merged"]) == (expected["left"], expected["right"], expected["merged"])
        assert actual["distance"] == pytest.approx(expected["distance"], rel=1e-12, abs=1e-12)
    rows = result["assignments"]
    assert rows.position.tolist() == list(range(len(data)))
    assert rows.status.tolist() == ["missing" if i in [3, 8] else "cluster" for i in range(len(data))]
    assert result.attrs["n_complete"] == len(keep) and result.attrs["n_missing"] == 2
    pd.testing.assert_frame_equal(data, original)


def test_public_typed_categories_distinguish_bool_integer_float_and_string_and_query_maps_are_fixed():
    values = [True, 1, "1", 1., False, 0, "0", 0.]
    data = pd.DataFrame({"g": pd.Series(values * 2, dtype=object), "x": np.arange(16, dtype=float)})
    result = fit_public(data, continuous=["x"], categorical=["g"], n_clusters=3)
    state = result.attrs["twostep_state"]
    raw_public_encoding(data, state)
    assert len(state["categorical"][0]["levels"]) == 8
    assert state["training_codes"][:8] == [[i] for i in range(8)]
    before = deepcopy(state)
    query = pd.DataFrame({"x": [2., 3., 4.], "g": pd.Series([1, True, "1"], dtype=object)})
    assigned = oe.twostep_assign(result, data=query)
    assert assigned["assignments"].status.tolist() == ["cluster"] * 3
    assert state == before
    with pytest.raises(AnalysisError) as caught:
        oe.twostep_assign(result, data=query.assign(g="unseen"))
    assert caught.value.code == "unknown_category"


def test_saved_assignment_uses_nearest_singleton_merge_loss_and_physical_query_positions():
    data = pd.DataFrame({"x": [-5., -4., -3., -2., 2., 3., 4., 5.], "g": ["a", "a", "a", "b", "b", "b", "b", "a"]})
    result = fit_public(data, continuous=["x"], categorical=["g"], n_clusters=2)
    state = result.attrs["twostep_state"]
    numeric, codes, levels, global_variance = raw_public_encoding(data, state)
    query = pd.DataFrame({"x": [-6., np.nan, 0., 6., 6.], "g": ["a", "b", "a", "b", "b"]}, index=[8, 8, 4, 8, 8])
    original = query.copy(deep=True)
    before = deepcopy(state)
    output = oe.twostep_assign(result, data=query, missing="drop")
    assignments = output["assignments"]
    assert assignments.position.tolist() == [0, 1, 2, 3, 4]
    assert assignments.status.tolist() == ["cluster", "missing", "cluster", "cluster", "cluster"]
    expected = []
    for position in [0, 2, 3, 4]:
        query_x = np.array([[(query.x.iloc[position] - data.x.mean()) / np.std(data.x, ddof=0)]])
        query_code = np.array([[0 if query.g.iloc[position] == "a" else 1]], dtype=np.int64)
        losses = []
        for cf in state["cuts"]["2"]:
            rows = cf["rows"]
            loss = math.fsum((
                reference_xi(numeric[rows], codes[rows], levels, global_variance),
                reference_xi(query_x, query_code, levels, global_variance),
                -reference_xi(np.concatenate((numeric[rows], query_x)), np.concatenate((codes[rows], query_code)), levels, global_variance),
            ))
            losses.append(loss)
        label = min(range(len(losses)), key=lambda i: (losses[i], i))
        expected.append((position, label+1, losses[label]))
    for position, label, distance in expected:
        actual = assignments.loc[assignments.position == position].iloc[0]
        assert actual.cluster == label
        assert actual.merge_loss == pytest.approx(distance, rel=1e-12, abs=1e-12)
    assert state == before
    pd.testing.assert_frame_equal(query, original)


def test_saved_cut_reselection_never_refits_or_changes_other_hierarchy_state(monkeypatch):
    data = pd.DataFrame({"x": [-3., -2., -1., 1., 2., 3.]})
    result = fit_public(data, continuous=["x"], n_clusters=3)
    state = deepcopy(result.attrs["twostep_state"])

    def forbidden(*args, **kwargs):
        raise AssertionError("A saved cut must never build a new tree or hierarchy")

    monkeypatch.setattr(kernel, "build_tree", forbidden)
    monkeypatch.setattr(kernel, "agglomerate", forbidden)
    cut = oe.twostep_cut(result, 2)
    assert cut.attrs["twostep_state"]["selected_k"] == 2
    assert cut.attrs["twostep_state"]["cuts"] == state["cuts"]
    assert cut.attrs["twostep_state"]["merges"] == state["merges"]
    assert result.attrs["twostep_state"] == state
    assert set(cut["assignments"].cluster) == {1, 2}


def test_noise_exclusion_changes_ic_sample_size_and_retains_its_physical_training_status():
    data = pd.DataFrame({"x": [-2., -2., -2., 2., 2., 10.]})
    result = fit_public(data, continuous=["x"], n_clusters=2, min_precluster_size=2)
    state = result.attrs["twostep_state"]
    assert state["positions"] == list(range(6)) and state["noise_positions"] == [5]
    assert result["assignments"].status.tolist() == ["cluster"] * 5 + ["noise"]
    assert result.attrs["n_complete"] == 6 and result.attrs["n_clustered"] == 5 and result.attrs["n_noise"] == 1
    for row in result["criteria"].itertuples(index=False):
        assert row.bic == pytest.approx(-2 * row.score + row.parameters * math.log(5), rel=1e-13, abs=1e-13)
    # The documented query policy has no SPSS uniform-noise/reinsertion step.
    query = oe.twostep_assign(result, data=pd.DataFrame({"x": [10.]}))
    assert query["assignments"].status.tolist() == ["cluster"]
