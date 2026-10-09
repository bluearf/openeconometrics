"""CF purity, physical-row/tree invariants, and explicit resource refusals."""

import math

import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.twostep.kernel import (
    CF,
    ROUNDING_RULE,
    agglomerate,
    build_tree,
    merge,
    merge_loss,
    singleton,
    xi,
)


def design(n=35):
    x = torch.arange(n, dtype=torch.float64)[:, None]
    x = torch.cat((x, (x / 3).sin()), dim=1)
    codes = (torch.arange(n) % 3)[:, None]
    return x, codes, (3,), x.var(dim=0, correction=0)


def cf_from_rows(x, codes, levels, rows):
    result = singleton(x[rows[0]], codes[rows[0]], levels, rows[0])
    for row in rows[1:]:
        result = merge(result, singleton(x[row], codes[row], levels, row))
    return result


def tree(x, codes, levels, globalvar, **kwargs):
    options = dict(threshold=0.0, branch_factor=3, max_preclusters=128, max_nodes=512)
    options.update(kwargs)
    return build_tree(x, codes, levels, globalvar, torch.arange(len(x)), **options)


def test_parallel_cf_merge_is_pure_and_preserves_population_moments_counts_rows():
    x, codes, levels, globalvar = design(35)
    a, b = (
        cf_from_rows(x, codes, levels, list(range(0, 17))),
        cf_from_rows(x, codes, levels, list(range(17, 35))),
    )
    saved = [
        (cf.mean.clone(), cf.m2.clone(), tuple(c.clone() for c in cf.categorical_counts), cf.rows)
        for cf in (a, b)
    ]
    combined = merge(a, b)
    assert combined.count == 35 and combined.rows == tuple(range(35))
    torch.testing.assert_close(combined.mean, x.mean(0), rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(combined.m2, ((x - x.mean(0)) ** 2).sum(0), rtol=1e-14, atol=1e-14)
    assert combined.categorical_counts[0].tolist() == [12, 12, 11]
    for cf, (mean, m2, categorical, rows) in zip((a, b), saved):
        assert torch.equal(cf.mean, mean) and torch.equal(cf.m2, m2)
        assert all(torch.equal(a, b) for a, b in zip(cf.categorical_counts, categorical))
        assert cf.rows == rows
    assert merge_loss(a, b, globalvar) == pytest.approx(
        xi(a, globalvar) + xi(b, globalvar) - xi(combined, globalvar), abs=1e-12
    )
    with pytest.raises(AnalysisError, match="overlap"):
        merge(a, a)


@pytest.mark.parametrize("branch_factor", [2, 3, 5])
def test_real_cf_tree_is_balanced_bounded_and_every_internal_summary_matches_its_rows(
    branch_factor,
):
    x, codes, levels, globalvar = design(35)
    result = tree(x, codes, levels, globalvar, branch_factor=branch_factor)
    assert result.depth > 1 and result.split_count > 0 and result.node_count > 1
    depths, nodes, ids, leaves = set(), [], [], []

    def visit(node, depth):
        nodes.append(node["id"])
        assert 1 <= len(node["entries"]) <= branch_factor
        collected = []
        for entry in node["entries"]:
            ids.append(entry["id"])
            state = entry["cf"]
            rows = state["rows"]
            assert state["count"] == len(rows)
            torch.testing.assert_close(
                torch.tensor(state["mean"]), x[rows].mean(0).float(), atol=1e-6, rtol=1e-6
            )
            torch.testing.assert_close(
                torch.tensor(state["m2"]),
                ((x[rows] - x[rows].mean(0)) ** 2).sum(0).float(),
                atol=1e-6,
                rtol=1e-6,
            )
            assert (
                state["categorical_counts"][0]
                == torch.bincount(codes[rows, 0], minlength=3).tolist()
            )
            if node["leaf"]:
                assert entry["child"] is None
                depths.add(depth)
                leaves.extend(rows)
            else:
                assert entry["child"] is not None
                assert sorted(visit(entry["child"], depth + 1)) == rows
            collected.extend(rows)
        return collected

    assert sorted(visit(result.tree, 1)) == list(range(len(x)))
    assert sorted(leaves) == list(range(len(x)))
    assert depths == {result.depth}
    assert len(nodes) == len(set(nodes)) == result.node_count
    assert len(ids) == len(set(ids))
    assert result.tree["automatic_rebuild"] is False
    assert result.tree["distance_roundoff_rule"] == ROUNDING_RULE
    assert [cf.rows for cf in result.preclusters] == [(row,) for row in range(len(x))]


def test_zero_threshold_duplicate_records_absorb_exactly_without_score_cancellation():
    x = torch.cat(
        (torch.zeros(119, dtype=torch.float64), torch.tensor([10.0], dtype=torch.float64))
    )[:, None]
    codes = torch.zeros((120, 1), dtype=torch.int64)
    result = tree(x, codes, (1,), x.var(dim=0, correction=0))
    assert [cf.count for cf in result.preclusters] == [119, 1]
    first = singleton(x[0], codes[0], (1,), 0)
    same = singleton(x[1], codes[1], (1,), 1)
    assert merge_loss(first, same, x.var(dim=0, correction=0)) == 0.0
    assert result.split_count == 0


def test_absolute_absorption_boundary_and_threshold_monotonic_for_two_points():
    x = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    codes = torch.empty((2, 0), dtype=torch.int64)
    variance = torch.tensor([1.0], dtype=torch.float64)
    a, b = singleton(x[0], codes[0], (), 0), singleton(x[1], codes[1], (), 1)
    distance = merge_loss(a, b, variance)
    assert distance == pytest.approx(math.log(2.0), abs=1e-15)
    assert len(tree(x, codes, (), variance, threshold=distance).preclusters) == 1
    assert (
        len(tree(x, codes, (), variance, threshold=math.nextafter(distance, -math.inf)).preclusters)
        == 2
    )


def test_tree_limits_refuse_instead_of_rebuilding_dropping_or_partial_results():
    x, codes, levels, globalvar = design(5)
    with pytest.raises(AnalysisError, match="max_preclusters=2"):
        tree(x, codes, levels, globalvar, max_preclusters=2)
    with pytest.raises(AnalysisError, match="max_nodes=1"):
        tree(x, codes, levels, globalvar, branch_factor=2, max_nodes=1)
    full = tree(x, codes, levels, globalvar, threshold=1000.0, max_preclusters=1, max_nodes=1)
    assert full.preclusters[0].rows == tuple(range(5))
    assert full.node_count == full.depth == 1


def test_heap_hierarchy_stores_every_nested_cut_and_no_unchanged_distance_recomputation():
    x, codes, levels, globalvar = design(20)
    result = tree(x, codes, levels, globalvar)
    hierarchy = agglomerate(result.preclusters, globalvar)
    m = len(result.preclusters)
    assert sorted(hierarchy.cuts) == list(range(1, m + 1))
    assert len(hierarchy.merges) == m - 1
    assert hierarchy.distance_evaluations == (m - 1) ** 2
    previous = {cf.rows for cf in hierarchy.cuts[m]}
    for k in range(m - 1, 0, -1):
        current = {cf.rows for cf in hierarchy.cuts[k]}
        assert sorted(row for group in current for row in group) == list(range(len(x)))
        removed, added = previous - current, current - previous
        assert len(removed) == 2 and len(added) == 1
        assert sorted(row for group in removed for row in group) == list(next(iter(added)))
        previous = current
    assert hierarchy.cuts[1][0].rows == tuple(range(len(x)))


def test_hierarchy_single_leaf_is_complete_and_overlapping_leaves_fail():
    x, codes, levels, globalvar = design(5)
    one = cf_from_rows(x, codes, levels, list(range(5)))
    hierarchy = agglomerate([one], globalvar)
    assert hierarchy.cuts[1][0] is one
    assert hierarchy.merges == () and hierarchy.distance_evaluations == 0
    with pytest.raises(AnalysisError, match="distinct physical rows"):
        agglomerate([one, one], globalvar)


@pytest.mark.parametrize(
    "change",
    [
        "float32",
        "nan",
        "zero_variance",
        "negative_variance",
        "bad_code",
        "nonpermutation",
        "bad_order_type",
        "threshold",
        "branch",
        "leaves",
        "nodes",
    ],
)
def test_strict_native_kernel_admission(change):
    x, codes, levels, globalvar = design(6)
    kwargs = dict(
        X=x,
        codes=codes,
        levels=levels,
        globalvar=globalvar,
        row_order=torch.arange(6),
        threshold=0.0,
        branch_factor=3,
        max_preclusters=128,
        max_nodes=512,
    )
    if change == "float32":
        kwargs["X"] = x.float()
    elif change == "nan":
        x[0, 0] = math.nan
    elif change == "zero_variance":
        globalvar[0] = 0.0
    elif change == "negative_variance":
        globalvar[0] = -1.0
    elif change == "bad_code":
        codes[0, 0] = 4
    elif change == "nonpermutation":
        kwargs["row_order"] = torch.zeros(6, dtype=torch.int64)
    elif change == "bad_order_type":
        kwargs["row_order"] = torch.arange(6, dtype=torch.float64)
    elif change == "threshold":
        kwargs["threshold"] = math.inf
    elif change == "branch":
        kwargs["branch_factor"] = 1
    elif change == "leaves":
        kwargs["max_preclusters"] = 129
    elif change == "nodes":
        kwargs["max_nodes"] = False
    with pytest.raises(AnalysisError):
        build_tree(**kwargs)


def test_degenerate_cf_rejected_before_undefined_logs():
    with pytest.raises(AnalysisError, match="M2"):
        CF(
            1,
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([-1.0], dtype=torch.float64),
            (),
            (0,),
        )
    with pytest.raises(AnalysisError, match="summing to cluster size"):
        CF(
            1,
            torch.empty(0, dtype=torch.float64),
            torch.empty(0, dtype=torch.float64),
            (torch.tensor([2]),),
            (0,),
        )
