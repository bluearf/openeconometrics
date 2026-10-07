"""Independent selected-pair oracles across native block-model families."""
from copy import deepcopy
from itertools import repeat
import math

import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError


@pytest.mark.parametrize("directed", [False, True])
@pytest.mark.parametrize("method", ["block_model", "poisson_block_model", "degree_corrected_block_model"])
def test_one_group_expected_pairs_against_count_degree_oracle(directed, method):
    graph = oe.network({"source": [0, 0, 1, 2], "target": [0, 1, 2, 3], "w": [2, 3, 4, 1]},
                       nodes=range(5), weight="w", directed=directed)
    fit = getattr(graph, method)(1, starts=1)
    pairs = [(0, 1), (1, 0), (2, 3), (4, 1), (0, 1)]
    if method == "degree_corrected_block_model":
        pairs += [(0, 0), (4, 4)]
        # Expected degree ratios independently from observed edge counts.
        out = [5, 4, 1, 0, 0] if directed else [7, 7, 5, 1, 0]
        incoming = [2, 3, 4, 1, 0] if directed else out
        total = 10 if directed else 20
        means = [out[u] * incoming[v] / total / (2 if not directed and u == v else 1)
                 for u, v in pairs]
    else:
        dyads = 20 if directed else 10
        means = [3 / dyads if method == "block_model" else 8 / dyads] * len(pairs)
    result = fit.expected_edges(pairs)
    assert result.source.tolist() == [u for u, _ in pairs]
    assert result.target.tolist() == [v for _, v in pairs]
    assert result.mean_count.tolist() == pytest.approx(means)
    probabilities = means if method == "block_model" else [-math.expm1(-v) for v in means]
    assert result.presence_probability.tolist() == pytest.approx(probabilities)
    assert result.identified.all()
    assert result.attrs["sampled"] is False
    assert "\\toprule" in result.to_latex(index=False)


@pytest.mark.parametrize("directed", [False, True])
def test_singleton_zero_stub_groups_and_typed_labels(directed):
    huge = 2**180
    nodes = [1, "1", huge, "zero"]
    graph = oe.network({"source": [1], "target": ["1"], "w": [3]},
                       nodes=nodes, weight="w", directed=directed)
    fit = graph.degree_corrected_block_model(4, initial={node: i for i, node in enumerate(nodes)}, starts=1)
    result = fit.expected_edges(pd.DataFrame({"target": ["1", "1", 1, huge],
        "source": pd.Series([1, huge, "1", huge], dtype=object)}))
    assert result.mean_count.tolist() == [3, 0, 0 if directed else 3, 0]
    assert result.identified.tolist() == [True, False, not directed, False]
    assert result.source.iloc[0] == 1 and type(result.source.iloc[0]) is int
    assert result.source.iloc[1] == huge and type(result.source.iloc[1]) is int
    assert result.source.iloc[2] == "1"


@pytest.mark.parametrize("method", ["block_model", "poisson_block_model", "degree_corrected_block_model"])
def test_singleton_partition_and_empty_output(method):
    graph = oe.network({"source": ["a"], "target": ["b"], "w": [2]}, weight="w")
    fit = getattr(graph, method)(2, initial={"a": 0, "b": 1}, starts=1)
    assert fit.expected_edges([("a", "b")]).mean_count.iloc[0] == (1 if method == "block_model" else 2)
    result = fit.expected_edges([])
    assert list(result.columns) == ["source", "target", "mean_count", "presence_probability", "identified"]
    assert len(result) == 0 and str(result.identified.dtype) == "bool"
    if method != "degree_corrected_block_model":
        with pytest.raises(AnalysisError, match="self-loop"):
            fit.expected_edges([("a", "a")])


def fitted():
    return oe.network({"source": [0], "target": [1], "w": [2]}, weight="w").poisson_block_model(1, starts=1)


@pytest.mark.parametrize("pairs", [None, "01", {0: 1}, [(0,)], [(0, 1, 0)], [[True, 1]], [[0., 1]], [[0, "1"]]])
def test_invalid_pair_requests(pairs):
    with pytest.raises(AnalysisError):
        fitted().expected_edges(pairs)


def test_output_and_memory_preflight_bounded_generator():
    consumed = []
    def pairs():
        for i in range(100):
            consumed.append(i)
            yield 0, 1
    fit = fitted()
    with pytest.raises(AnalysisError, match="max_pairs"):
        fit.expected_edges(pairs(), max_pairs=3)
    assert consumed == [0, 1, 2, 3]
    with pytest.raises(AnalysisError, match="max_pairs"):
        fit.expected_edges([(0, 1)] * 2, max_pairs=1)
    with pytest.raises(AnalysisError, match="max_memory_mb"):
        fit.expected_edges(repeat((0, 1)), max_memory_mb=.001)
    with pytest.raises(AnalysisError, match="max_pairs"):
        fit.expected_edges([], max_pairs=False)


@pytest.mark.parametrize("mutation", ["duplicate", "missing_block", "duplicate_cell", "negative", "bad_flag", "bad_model"])
def test_mutated_saved_result_is_rejected(mutation):
    fit = deepcopy(fitted())
    if mutation == "duplicate":
        fit["membership"].loc[1, "node"] = 0
    elif mutation == "missing_block":
        fit["blocks"] = fit["blocks"].iloc[:0]
    elif mutation == "duplicate_cell":
        fit["blocks"] = pd.concat([fit["blocks"], fit["blocks"]])
    elif mutation == "negative":
        fit["blocks"].loc[0, "mean_count"] = -1
    elif mutation == "bad_flag":
        fit["blocks"]["identified"] = [1]
    else:
        fit["metadata"]["model"] = "anything"
    with pytest.raises(AnalysisError):
        fit.expected_edges([(0, 1)])


@pytest.mark.parametrize("method,title", [("poisson_block_model", "Poisson SBM"),
    ("degree_corrected_block_model", "Degree-corrected Poisson SBM")])
def test_bounded_summary_and_chart_membership(method, title):
    graph = oe.network({"source": [0, 1], "target": [1, 2], "w": [2, 3]}, weight="w")
    fit = getattr(graph, method)(1, starts=1)
    summary = fit.summary()
    assert len(summary) == 11 and summary.Value.iloc[0] == title
    assert summary.attrs["model"] == fit["metadata"]["model"]
    plot = oe.plot.network(graph, groups=fit["membership"])
    assert plot.config["network"]["grouping"] == title + " blocks"


@pytest.mark.parametrize("mutation", ["unnormalized", "negative", "tiny", "flags", "loop_scope", "omega_overflow"])
def test_invalid_corrected_parameters(mutation):
    graph = oe.network({"source": [0], "target": [1], "w": [2]}, weight="w")
    fit = graph.degree_corrected_block_model(1, starts=1)
    if mutation == "unnormalized":
        fit["membership"].loc[0, "theta"] = .7
    elif mutation == "negative":
        fit["membership"].loc[0, "theta"] = -1
    elif mutation == "tiny":
        fit["membership"].loc[0, "theta"] = 1e-300
    elif mutation == "flags":
        fit["membership"].loc[0, "theta_identified"] = False
    elif mutation == "loop_scope":
        fit["metadata"]["self_loops"] = False
    else:
        fit["blocks"]["omega"] = [1e300]
    with pytest.raises(AnalysisError):
        fit.expected_edges([(0, 1)])
