"""Independent graph/singleton metadata oracle; no numerical FE fit shortcut."""
from contextlib import ExitStack

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.dataset import Dataset
from openecon.econometrics.replay_sample import ReplaySample
from openecon.econometrics.streaming_hdfe import _Selection
from openecon.engines.absorb import absorbed_degrees_of_freedom, singleton_mask
from openecon.models import ModelSpec


def structure_data(seed=781123, dimensions=3):
    rng = np.random.default_rng(seed)
    n = 379
    data = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n),
                         "weight": rng.integers(1, 4, size=n),
                         "cluster": np.arange(n)%3, "cluster2": np.arange(n)%7})
    for d in range(dimensions):
        data[f"f{d}"] = rng.integers(0, 41+d*7, size=n)
        data.loc[:3, f"f{d}"] = np.arange(1000+d*10, 1004+d*10)
    return data


def python_singletons(data, dimensions, frequency):
    keep = np.ones(len(data), dtype=bool)
    while True:
        before = keep.copy()
        for name in dimensions:
            values = data[name].to_numpy()
            for group in np.unique(values[keep]):
                selected = keep & (values == group)
                mass = selected.sum()+sum(data.loc[selected, "weight"] >= 2) if frequency else selected.sum()
                if mass == 1:
                    keep[selected] = False
        if np.array_equal(before, keep):
            return keep


@pytest.mark.parametrize("dimensions", [1, 2, 3])
@pytest.mark.parametrize("frequency", [False, True])
@pytest.mark.parametrize("drop", [False, True])
def test_iterative_singletons_and_exact_retained_membership(dimensions, frequency, drop):
    data = structure_data(dimensions=dimensions)
    absorb = [f"f{d}" for d in range(dimensions)]
    spec = ModelSpec(estimator="reghdfe", outcome="y", predictors=["x"], intercept=False,
                     columns={"absorb": absorb}, weights="weight" if frequency else None,
                     weight_type="fweight" if frequency else None)
    sample = ReplaySample(spec, Dataset.from_frame(data), batch_rows=17)
    sample.add_design("mean", intercept=True)
    sample.prepare()
    with ExitStack() as stack:
        selection = _Selection(dimensions, 0)
        stack.callback(selection.close)
        selection.seed(sample, absorb, [], drop)
        expected = python_singletons(data, absorb, frequency) if drop else np.ones(len(data), dtype=bool)
        np.testing.assert_array_equal(selection.filter(data, absorb).numpy(), expected)
        dims = []
        for name in absorb:
            codes, levels = pd.factorize(data.loc[expected, name], sort=True)
            dims.append((torch.from_numpy(codes), len(levels)))
        actual = selection.degrees_of_freedom()
        oracle = absorbed_degrees_of_freedom(dims)
        assert actual == (oracle.levels, oracle.redundant, oracle.nested, oracle.total)
        if not frequency and drop:
            full = []
            for name in absorb:
                codes, levels = pd.factorize(data[name], sort=True)
                full.append((torch.from_numpy(codes), len(levels)))
            np.testing.assert_array_equal(expected, singleton_mask(full).numpy())


@pytest.mark.parametrize("clusters", [["cluster"], ["cluster", "cluster2"], ["cluster2", "cluster"]])
def test_nested_dimensions_skip_graph_rank_in_any_cluster_dimension(clusters):
    data = structure_data(dimensions=3)
    data["f0"] = data.cluster*10+(np.arange(len(data))%2)
    data["f1"] = data.cluster2*10+(np.arange(len(data))//7%2)
    absorb = ["f0", "f1", "f2"]
    spec = ModelSpec(estimator="reghdfe", outcome="y", predictors=["x"], intercept=False,
                     columns={"absorb": absorb}, covariance="cluster", cluster=clusters)
    sample = ReplaySample(spec, Dataset.from_frame(data), batch_rows=17)
    sample.add_design("mean", intercept=True)
    sample.prepare()
    selection = _Selection(3, len(clusters))
    try:
        selection.seed(sample, absorb, clusters, False)
        levels, redundant, nested, total = selection.degrees_of_freedom()
        assert nested == ([True, False, False] if len(clusters) == 1 else [True, True, False])
        assert redundant == ([levels[0], 0, 1] if len(clusters) == 1 else [levels[0], levels[1], 0])
        assert total == sum(levels)-sum(redundant)
    finally:
        selection.close()


def test_disconnected_bipartite_components_independent_graph_oracle():
    rows = [(a, b) for offset in [0, 100, 1000] for a, b in [(offset, offset+1), (offset+1, offset+1), (offset+1, offset+2)]]
    data = pd.DataFrame({"f0": [a for a, _ in rows]*3, "f1": [b for _, b in rows]*3,
                         "x": np.arange(27), "y": np.sin(np.arange(27))})
    spec = ModelSpec(estimator="reghdfe", outcome="y", predictors=["x"], intercept=False, columns={"absorb": ["f0", "f1"]})
    sample = ReplaySample(spec, Dataset.from_frame(data), batch_rows=17)
    sample.add_design("mean", intercept=True)
    sample.prepare()
    selection = _Selection(2, 0)
    try:
        selection.seed(sample, ["f0", "f1"], [], False)
        assert selection.degrees_of_freedom() == ([6, 6], [0, 3], [False, False], 9)
    finally:
        selection.close()
