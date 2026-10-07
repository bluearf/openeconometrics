"""Count-fit stopping contracts and independent restart reproducibility."""
from unittest.mock import patch
import math

import pytest

from openecon import _network_dc_sbm as dc
from openecon import _network_poisson as poisson
from openecon.networks import network


@pytest.fixture(params=[poisson, dc])
def solver(request):
    module = request.param
    fit = module.poisson_block_model if module is poisson else module.degree_corrected_block_model
    return module, fit


def count_graph(directed=False):
    rows = [{"source": a, "target": b, "count": 11 if a // 3 == b // 3 else 1}
            for a in range(6) for b in range(6) if (a != b if directed else a < b)]
    return network(rows, nodes=range(6), directed=directed, weight="count")


@pytest.mark.parametrize("directed", [False, True])
def test_each_restart_initialization_is_independent_of_earlier_iteration_limits(solver, directed):
    module, fit = solver
    graph = count_graph(directed)
    original = module._State

    def initializations(limit):
        captured = []

        def record(graph, membership, *args, **kwargs):
            captured.append(list(membership))
            return original(graph, membership, *args, **kwargs)

        with patch.object(module, "_State", record):
            fit(graph, 2, initial=[0, 1, 0, 1, 0, 1], starts=4,
                seed=37, max_iter=limit)
        return captured

    assert initializations(1) == initializations(25)


@pytest.mark.parametrize("directed", [False, True])
def test_every_restart_reports_objective_membership_changes_and_its_seed(solver, directed):
    _, fit = solver
    result = fit(count_graph(directed), 2, initial=[0, 1, 0, 1, 0, 1],
                 starts=3, seed=37, max_iter=25)
    meta = result["metadata"]
    assert [run["seed"] for run in meta["start_fits"]] == [37, 38, 39]
    for run in meta["start_fits"]:
        assert len(run["membership_changes"]) == len(run["likelihood_gains"]) == run["iterations"]
        assert sum(run["membership_changes"]) == run["moves"]
        assert all(0 <= changed <= 6 for changed in run["membership_changes"])
        assert math.fsum(run["likelihood_gains"]) == pytest.approx(
            run["log_likelihood"] - run["likelihood_history"][0], abs=1e-10)
        assert run["stopping_reason"] == ("no_admissible_move" if run["converged"] else "max_iter")
        if run["converged"]:
            assert run["membership_changes"][-1] == 0
    selected = meta["start_fits"][meta["selected_start"]]
    assert meta["stopping_reason"] == selected["stopping_reason"]
    assert meta["selected_seed"] == selected["seed"]


def test_fixed_profile_and_unique_partition_have_distinct_stopping_reasons(solver):
    _, fit = solver
    graph = count_graph()
    fixed = fit(graph, 2, initial=[0, 1, 0, 1, 0, 1], starts=1, max_iter=0)
    assert not fixed["metadata"]["converged"]
    assert fixed["metadata"]["stopping_reason"] == "fixed_partition"
    unique = fit(graph, 1, starts=1)
    assert unique["metadata"]["converged"]
    assert unique["metadata"]["stopping_reason"] == "unique_partition"
    for result in [fixed, unique]:
        run = result["metadata"]["start_fits"][0]
        assert run["membership_changes"] == run["likelihood_gains"] == []


def test_seed_wraps_within_the_public_seed_range(solver):
    _, fit = solver
    result = fit(count_graph(), 2, starts=3, seed=2**63 - 1)
    assert [run["seed"] for run in result["metadata"]["start_fits"]] == [2**63 - 1, 0, 1]
