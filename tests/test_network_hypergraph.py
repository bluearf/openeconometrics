import itertools

import pytest
import torch

import openecon as oe


def test_incidence_degree_projection_and_typed_roundtrip(tmp_path):
    records = [{"id": 1, "members": [1, "1", "a"], "weight": 2},
               {"id": "1", "members": [1, "a"], "weight": 0},
               {"id": "solo", "members": ["a"], "weight": 3}]
    h = oe.hypergraph(records, nodes=[1, "1", "a", "isolate"])
    assert h.node_count == 4 and h.edge_count == 3
    expected = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.float64)
    assert torch.equal(h.incidence().to_dense(), expected)
    assert torch.equal(h.incidence_matvec(torch.ones(3)), expected.sum(1))
    assert torch.equal(h.incidence_matvec(torch.ones(4), transpose=True), expected.sum(0))
    degree = h.degree()
    assert degree.hyperdegree.tolist() == [2, 1, 3, 0]
    assert degree.strength.tolist() == [2, 2, 5, 0]
    assert type(degree.node[0]) is int and type(degree.node[1]) is str
    assert h.clique_projection().edge_count == 3
    assert h.clique_projection(reducer="count").edges().weight.tolist() == [1, 2, 1]
    normalized = h.clique_projection(reducer="normalized")
    assert sum(normalized.edges().weight) == pytest.approx(2)
    star = h.star_projection()
    assert star.node_count == 7 and star.edge_count == 4
    assert star.metadata["direct_hypergraph_analysis"] is False
    path = h.write(tmp_path / "h.json")
    assert oe.read_hypergraph(path).to_dict() == h.to_dict()
    with pytest.raises(FileExistsError):
        h.write(path)
    copy = h.incidence()
    copy.values().zero_()
    assert torch.equal(h.incidence().to_dense(), expected)
    assert not h.degree().attrs["projected"]


def test_directed_membership_and_projection_oracles():
    h = oe.hypergraph([{"id": "r", "tail": [1, "1"], "head": ["1", "a"], "weight": 6}], directed=True)
    assert h.degree().out_hyperdegree.tolist() == [1, 1, 0]
    assert h.degree().in_hyperdegree.tolist() == [0, 1, 1]
    assert h.incidence(role="tail", weighted=True).values().tolist() == [6, 6]
    with pytest.raises(oe.AnalysisError):
        h.incidence()
    edges = h.clique_projection(reducer="normalized").edges()
    assert {(r.source, r.target): r.weight for r in edges.itertuples()} == {(1, "1"): 2, (1, "a"): 2, ("1", "a"): 2}
    star = h.star_projection()
    assert {(r.source, r.target) for r in star.edges().itertuples()} == {(0, 3), (1, 3), (3, 1), (3, 2)}


@pytest.mark.parametrize("mask", range(1, 32))
def test_small_incidence_independent_oracle(mask):
    members = [i for i in range(5) if mask & (1 << i)]
    records = [{"id": "x", "members": members, "weight": 2}, {"id": "y", "members": [0, 2], "weight": 3}]
    h = oe.hypergraph(records, nodes=range(5))
    table = h.degree()
    assert table.hyperdegree.tolist() == [sum(i in r["members"] for r in records) for i in range(5)]
    assert table.strength.tolist() == [sum(r["weight"] for r in records if i in r["members"]) for i in range(5)]
    expected = {}
    for r in records:
        for a, b in itertools.combinations(sorted(r["members"]), 2):
            expected[a, b] = expected.get((a, b), 0) + r["weight"]
    assert {(r.source, r.target): r.weight for r in h.clique_projection().edges().itertuples()} == expected


@pytest.mark.parametrize("records", [
    [{"id": "x", "members": [1, 1]}], [{"id": "x", "members": []}],
    [{"id": "x", "members": [True]}], [{"id": "x", "members": [None]}],
    [{"id": "x", "members": [1], "weight": -1}], [{"id": "x", "members": [1], "weight": float("nan")}],
    [{"id": "x", "members": [1]}, {"id": "x", "members": [2]}],
    [{"id": "x", "members": [1], "extra": 1}], [{"members": [1]}],
])
def test_invalid_hyperedges(records):
    with pytest.raises(oe.AnalysisError):
        oe.hypergraph(records)


def test_pre_expansion_budgets_and_memory(monkeypatch, tmp_path):
    h = oe.hypergraph([{"id": "large", "members": range(500)}], max_memory_mb=1)
    def blocked(*args, **kwargs):
        pytest.fail("Projection allocation must not run before admission")
    monkeypatch.setattr("openecon._network_hypergraph.network", blocked)
    with pytest.raises(oe.AnalysisError, match="max_edges"):
        h.clique_projection(max_edges=100)
    with pytest.raises(oe.AnalysisError, match="max_work"):
        h.clique_projection(max_edges=200_000, max_work=1)
    with pytest.raises(oe.AnalysisError, match="max_edges"):
        h.star_projection(max_edges=1)
    with pytest.raises(oe.AnalysisError, match="max_memory_mb"):
        h.clique_projection(max_edges=200_000)
    with pytest.raises(oe.AnalysisError, match="max_memberships"):
        oe.hypergraph([{"id": "x", "members": range(50)}], max_memberships=10)
    huge = tmp_path / "too-large.json"
    huge.write_text(" " * 10_000)
    with pytest.raises(oe.AnalysisError, match="pre-parse"):
        oe.read_hypergraph(huge, max_memory_mb=.05)


def test_empty_huge_shape_no_dense_and_default_device():
    torch.set_default_device("meta")
    try:
        h = oe.hypergraph([], nodes=range(2000))
        assert h.incidence().shape == (2000, 0)
        assert h.incidence().device.type == "cpu"
        assert h.degree().hyperdegree.sum() == 0
    finally:
        torch.set_default_device("cpu")
