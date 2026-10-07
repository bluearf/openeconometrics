"""Independent scalar/dense supra equations and lossless identity/resource checks."""
from fractions import Fraction
import itertools
import random

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def edge(identifier, u, v, weight=1., attributes=None):
    return dict(edge_id=identifier, source=u[0], source_layer=u[1], target=v[0], target_layer=v[1],
                weight=weight, attributes={} if attributes is None else attributes)


PAIRS = [(1, "x"), ("1", "x"), (1, "y"), ("isolate", "y")]
EDGES = [edge(1, PAIRS[0], PAIRS[1], 2., {"kind": "loan"}),
         edge("1", PAIRS[0], PAIRS[1], 3., {"kind": "trade"}),
         edge("coupling", PAIRS[0], PAIRS[2], .5),
         edge("return", PAIRS[2], PAIRS[0], 4.),
         edge("loop", PAIRS[2], PAIRS[2], 7.),
         edge("zero", PAIRS[3], PAIRS[1], 0.)]


def graph(directed=True, **kwargs):
    return oe.multilayer_network(EDGES, nodes=PAIRS, layers=["x", "y", "empty"],
        weight="weight", attributes="attributes", directed=directed, **kwargs)


def scalar_adjacency(records, pairs, directed):
    index = {p: i for i, p in enumerate(pairs)}
    matrix = [[Fraction(0) for _ in pairs] for _ in pairs]
    for r in records:
        u, v = index[(r["source"], r["source_layer"])], index[(r["target"], r["target_layer"])]
        w = Fraction(r["weight"])
        matrix[u][v] += w
        if not directed and u != v:
            matrix[v][u] += w
    return matrix


def dense_solve(a, b):
    # Fraction Gaussian elimination, independent of Torch and graph kernels.
    rows = [list(r) + [v] for r, v in zip(a, b)]
    for i in range(len(rows)):
        pivot = next(j for j in range(i, len(rows)) if rows[j][i])
        rows[i], rows[pivot] = rows[pivot], rows[i]
        div = rows[i][i]
        rows[i] = [v / div for v in rows[i]]
        for j in range(len(rows)):
            if j != i:
                factor = rows[j][i]
                rows[j] = [v - factor * w for v, w in zip(rows[j], rows[i])]
    return [r[-1] for r in rows]


def oracle_rank(matrix, probabilities, damping=Fraction(17, 20)):
    n = len(matrix)
    probabilities = [Fraction(p) for p in probabilities]
    total = sum(probabilities)
    p = [v / total for v in probabilities]
    outgoing = [sum(r) for r in matrix]
    transition = [[matrix[i][j] / outgoing[i] if outgoing[i] else p[j] for j in range(n)] for i in range(n)]
    system = [[Fraction(i == j) - damping * transition[j][i] for j in range(n)] for i in range(n)]
    return dense_solve(system, [(1 - damping) * v for v in p])


@pytest.mark.parametrize("directed,transpose", itertools.product([True, False], repeat=2))
def test_rational_matvec_loops_parallel_zero_and_coupling(directed, transpose):
    g = graph(directed)
    a = scalar_adjacency(EDGES, PAIRS, directed)
    x = [Fraction(-1), Fraction(2), Fraction(3), Fraction(10)]
    if transpose:
        a = list(zip(*a))
    expected = [sum(v * w for v, w in zip(r, x)) for r in a]
    result = g.matvec(dict(zip(PAIRS, map(float, x))), transpose=transpose)
    assert result["value"].tolist() == list(map(float, expected))
    assert result[["node", "layer"]].to_records(index=False).tolist() == PAIRS
    assert result.attrs["network"]["dense_adjacency"] is False


@pytest.mark.parametrize("directed,personal", itertools.product([True, False], [False, True]))
def test_rational_stationary_equation(directed, personal):
    g = graph(directed)
    p = [1, 2, 3, 4] if personal else [1] * 4
    expected = oracle_rank(scalar_adjacency(EDGES, PAIRS, directed), p)
    result = g.pagerank(tol=1e-12, max_iter=500, personalization=dict(zip(PAIRS, p)) if personal else None)
    assert result.pagerank.tolist() == pytest.approx(list(map(float, expected)), abs=1e-12)
    assert sum(result.pagerank) == pytest.approx(1.)
    assert result.attrs["converged"]
    assert result.attrs["state_space"] == "node-layer pairs"


@pytest.mark.parametrize("seed", range(24))
@pytest.mark.parametrize("directed", [False, True])
def test_random_independent_equations(seed, directed):
    rng = random.Random(seed)
    pairs = [(n, layer) for layer in [1, "1"] for n in [1, "1", 2]] + [("i", "1")]
    records = [edge(i, rng.choice(pairs[:-1]), rng.choice(pairs[:-1]), rng.randrange(5) / 2) for i in range(22)]
    g = oe.multilayer_network(records, nodes=pairs, layers=[1, "1"], directed=directed, weight="weight")
    a = scalar_adjacency(records, pairs, directed)
    x = [rng.randrange(-5, 5) for _ in pairs]
    expected = [sum(v * w for v, w in zip(r, x)) for r in a]
    assert g.matvec(dict(zip(pairs, x))).value.tolist() == list(map(float, expected))
    p = [rng.randrange(1, 5) for _ in pairs]
    rank = g.pagerank(personalization=dict(zip(pairs, p)), tol=1e-11, max_iter=500)
    assert rank.pagerank.tolist() == pytest.approx(list(map(float, oracle_rank(a, p))), abs=1e-11)


def test_identity_layer_order_attributes_and_lossless_immutable_edit(tmp_path):
    g = graph(node_attributes={PAIRS[0]: {"size": 12}, PAIRS[2]: {"size": "12"}},
              graph_attributes={"name": "test"})
    original = g.edges().to_dict("records")
    assert g.node_count == 4 and g.edge_count == 6 and g.layer_count == 3
    assert g.layers().to_dict("records") == [dict(layer="x", node_count=2), dict(layer="y", node_count=2), dict(layer="empty", node_count=0)]
    h = g.edit_edges(weights={1: 9.}, remove=["return"], attributes={"1": {"kind": "replaced"}},
                     add=[edge("new", ("new", "y"), PAIRS[2], .25)])
    assert g.edges().to_dict("records") == original
    assert h.edge_count == 6 and h.node_count == 5
    assert h.edges().iloc[0].weight == 9.
    assert h.edges().iloc[1]["attr.kind"] == "replaced"
    path = tmp_path / "complete.ndjson"
    h.write(path)
    reopened = oe.read_multilayer_network(path)
    pd.testing.assert_frame_equal(reopened.nodes(), h.nodes())
    pd.testing.assert_frame_equal(reopened.edges(), h.edges())
    pd.testing.assert_frame_equal(reopened.layers(), h.layers())
    assert reopened._graph.graph_attributes == {"name": "test"}
    assert reopened._graph._edge_ids[0:2] == (1, "1")
    with pytest.raises(AnalysisError, match="exists"):
        h.write(path)
    assert not list(tmp_path.glob(".openecon-*"))


def test_layer_extraction_keeps_parallel_zero_isolate_and_orientation():
    g = graph(False)
    x, y, empty = g.layer("x"), g.layer("y"), g.layer("empty")
    assert x._edge_ids == (1, "1") and x._labels == (1, "1")
    assert x.edge_attributes[1] == {"kind": "loan"}
    assert y._labels == (1, "isolate") and y._edge_ids == ("loop",)
    assert empty.node_count == empty.edge_count == 0
    assert g.edge_count == 6


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("policy", ["include", "drop"])
@pytest.mark.parametrize("reducer", ["sum", "count", "binary", "mean", "max", "min"])
def test_explicit_projection_scalar_oracle(directed, policy, reducer):
    g = graph(directed)
    output = g.project(reducer=reducer, inter_layer=policy, attributes="drop", zero="drop")
    grouped = {}
    for r in EDGES:
        if policy == "drop" and r["source_layer"] != r["target_layer"]:
            continue
        u, v = r["source"], r["target"]
        if not directed and (type(u).__name__, str(u)) > (type(v).__name__, str(v)):
            u, v = v, u
        grouped.setdefault((u, v), []).append(Fraction(r["weight"]))
    expected = {}
    for pair, weights in grouped.items():
        value = {"sum": sum(weights), "count": len(weights), "binary": 1,
                 "mean": sum(weights) / len(weights), "max": max(weights), "min": min(weights)}[reducer]
        if value:
            expected[pair] = float(value)
    actual = {(r.source, r.target): r.weight for r in output.edges().itertuples()}
    assert actual.keys() == expected.keys()
    assert list(actual.values()) == pytest.approx([expected[k] for k in actual], abs=1e-14)
    assert output.node_count == 3
    assert output.metadata["multilayer_projection"]["inter_layer"] == policy


def test_projection_conflicts_require_explicit_loss_policy():
    with pytest.raises(AnalysisError, match="attributes"):
        graph().project(reducer="count", inter_layer="drop")
    g = graph(node_attributes={PAIRS[0]: {"value": 1}, PAIRS[2]: {"value": True}})
    with pytest.raises(AnalysisError, match="replicas"):
        g.project(reducer="sum", inter_layer="include")
    with pytest.raises(AnalysisError, match="zero"):
        graph().project(reducer="sum", inter_layer="include", attributes="drop")


@pytest.mark.parametrize("value", [True, 1.5, "", (), {"nested": 1}, None])
def test_invalid_pair_identity(value):
    with pytest.raises(AnalysisError):
        oe.multilayer_network([], nodes=[(value, "x")])


@pytest.mark.parametrize("method,args,options", [
    ("matvec", ({PAIRS[0]: 1},), {}), ("matvec", ({("unknown", "x"): 1},), {}),
    ("matvec", (dict(zip(PAIRS, [1, 2, 3, True])),), {}),
    ("matvec", (dict.fromkeys(PAIRS, 1),), {"transpose": 1}),
    ("matvec", (dict.fromkeys(PAIRS, 1),), {"max_work": 1}),
    ("pagerank", (), {"personalization": {("unknown", "x"): 1}}),
    ("pagerank", (), {"personalization": dict.fromkeys(PAIRS, 0)}),
    ("pagerank", (), {"max_iter": 1}), ("pagerank", (), {"max_work": 1}),
    ("pagerank", (), {"damping": 1}), ("layer", ("unknown",), {}),
    ("edit_edges", (), {"weights": {"unknown": 2}}),
    ("edit_edges", (), {"remove": [1], "weights": {1: 2}}),
    ("edit_edges", (), {"add": [edge("new", (1, "unknown"), PAIRS[0])]}),
    ("project", (), {"reducer": "sum", "inter_layer": "silent"}),
])
def test_explicit_invalid_or_budget_refusal(method, args, options):
    with pytest.raises(AnalysisError):
        getattr(graph(), method)(*args, **options)


def test_empty_unweighted_dangling_and_missing_policy(tmp_path):
    g = oe.multilayer_network([], layers=[1, "1"], nodes=[(1, 1), (1, "1")])
    assert g.pagerank().pagerank.tolist() == [.5, .5]
    assert g.matvec({(1, 1): 3, (1, "1"): 4}).value.tolist() == [0., 0.]
    empty = oe.multilayer_network([], layers=["empty"])
    assert empty.pagerank().empty and empty.matvec({}).empty
    path = tmp_path / "unweighted.ndjson"
    g.write(path)
    assert oe.read_multilayer_network(path).weighted is False
    records = [edge("missing", (None, "x"), (1, "x")), edge("valid", (1, "x"), (2, "x"))]
    with pytest.raises(AnalysisError):
        oe.multilayer_network(records)
    dropped = oe.multilayer_network(records, missing="drop")
    assert dropped.metadata["missing_rows_dropped"] == 1 and dropped.edge_count == 1


@pytest.mark.parametrize("options", [{"layers": ["x", "x"]}, {"layers": [1, True]},
                                     {"nodes": "x"}, {"layers": {}}, {"directed": 1},
                                     {"source": "target"}, {"missing": []}, {"max_memory_mb": .1}])
def test_constructor_refusals(options):
    with pytest.raises(AnalysisError):
        oe.multilayer_network([], **options)


def test_duplicate_typed_edges_and_ingestion_cleanup():
    closed = []
    def records():
        try:
            yield EDGES[0]
            yield EDGES[0]
        finally:
            closed.append(True)
    with pytest.raises(AnalysisError, match="unique"):
        oe.multilayer_network(records(), weight="weight")
    assert closed == [True]
    with pytest.raises(AnalysisError, match="max_memory"):
        oe.multilayer_network((edge(i, (i, "x"), (i + 1, "y")) for i in range(100)), max_memory_mb=1)


@pytest.mark.parametrize("mutate", [
    lambda lines: lines.__setitem__(0, lines[0].replace("/1", "/2")),
    lambda lines: lines.__setitem__(1, lines[1][:-1] + ',"extra":1}'),
    lambda lines: lines.insert(2, lines[1]),
    lambda lines: lines.insert(5, lines[4]),
    lambda lines: lines.__setitem__(-1, lines[-1].replace('"source_layer":"y"', '"source_layer":"unknown"')),
    lambda lines: lines.__setitem__(0, lines[0][:-1] + ',"directed":false}'),
    lambda lines: lines.__setitem__(-1, '{"kind":"edge","weight":NaN}'),
    lambda lines: lines.__setitem__(0, '{"kind":[]}'),
])
def test_strict_interchange_rejects_ambiguity(tmp_path, mutate):
    path = tmp_path / "input.ndjson"
    graph().write(path)
    lines = path.read_text().splitlines()
    mutate(lines)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(AnalysisError):
        oe.read_multilayer_network(path)


def test_sparse_allocation_guard(monkeypatch):
    # A real 5k-state sparse ring with couplings, not an empty shape-only test.
    count = 2500
    records = itertools.chain((edge(i, (i, "a"), ((i + 1) % count, "a")) for i in range(count)),
        (edge("b" + str(i), (i, "b"), ((i + 1) % count, "b")) for i in range(count)),
        (edge("c" + str(i), (i, "a"), (i, "b"), .5) for i in range(count)))
    g = oe.multilayer_network(records, weight="weight", directed=False, max_memory_mb=256)
    for name in ["zeros", "ones", "empty", "full"]:
        original = getattr(torch, name)
        def guarded(*args, _original=original, **kwargs):
            shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args[:2]
            assert tuple(shape) != (g.node_count, g.node_count)
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, guarded)
    assert g.pagerank(max_work=500_000_000).pagerank.tolist() == pytest.approx([1 / 5000] * 5000, abs=1e-12)
    assert g.matvec(dict.fromkeys(g._pairs, 1)).value.tolist() == [2.5] * 5000


def test_large_legal_attribute_record_roundtrip_and_file_budget(tmp_path):
    # Larger than a one-MiB line: every scalar still fits the shared contract.
    attrs = {f"field{i}": "x" * 16000 for i in range(80)}
    g = oe.multilayer_network([edge("a", (1, "x"), (2, "x"), attributes=attrs)], attributes="attributes")
    path = tmp_path / "large.ndjson"
    g.write(path)
    assert path.stat().st_size > 1024**2
    h = oe.read_multilayer_network(path)
    assert h._graph.edge_attributes == {"a": attrs}
    with pytest.raises(AnalysisError) as failure:
        oe.read_multilayer_network(path, max_file_mb=1)
    assert failure.value.code == "network_file_budget"


def test_unweighted_edits_preserve_declared_weight_semantics(tmp_path):
    g = oe.multilayer_network([dict(edge_id=1, source=1, source_layer="x", target=1, target_layer="y")])
    for changed in [g.edit_edges(), g.edit_edges(attributes={1: {"kind": "coupling"}}),
                    g.edit_edges(add=[dict(edge_id=2, source=1, source_layer="y", target=1, target_layer="x")])]:
        assert changed.weighted is False and changed.metadata["weighted"] is False
        path = tmp_path / "unweighted.ndjson"
        changed.write(path, overwrite=True)
        assert oe.read_multilayer_network(path).weighted is False
    assert g.edit_edges(weights={1: 1.}).weighted is True
    assert g.edit_edges(add=[edge(2, (1, "y"), (1, "x"))]).weighted is True


def test_exhausted_import_budget_has_explicit_memory_error():
    with pytest.raises(AnalysisError) as failure:
        oe.multilayer_network([], max_memory_mb=.5)
    assert failure.value.code == "network_memory_budget"
