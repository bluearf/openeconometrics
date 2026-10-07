"""Independent data-separation and panel regressions for the synthetic lab."""
from collections import Counter
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.console import ConsoleSession
from openecon.output_events import validate_output_events
from openecon.plot_artifacts import checked_plot_path
from openecon.workspace import Workspace


SOURCE = Path(__file__).resolve().parents[1] / "docs/examples/network_advanced_lab.py"
REDUCED = dict(firms=144, mrqap_nodes=36, permutations=19, betweenness_samples=8,
               link_candidates=200, block_starts=1, block_iterations=4, skills=24)


@pytest.fixture(scope="module")
def lab():
    spec = importlib.util.spec_from_file_location("owned_advanced_network_example", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _edges(graph):
    return {(a, b): float(w) for a, b, w in
            graph.edges()[["source", "target", "weight"]].itertuples(index=False, name=None)}


def _payload_edges(payload):
    labels = {row["id"]: row["identity"]["value"] for row in payload["nodes"]}
    return {(labels[row["source"]], labels[row["target"]]): row["weight"]
            for row in payload["edges"]}


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


@pytest.fixture(scope="module")
def computation(lab, tmp_path_factory):
    captures = dict(fits={}, predictions=[], links={}, qap={})
    outputs = []
    folder = tmp_path_factory.mktemp("lab-no-default-exports")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as monkeypatch, ExitStack() as stack:
        monkeypatch.chdir(folder)
        for name in ["poisson_block_model", "degree_corrected_block_model"]:
            original = getattr(oe.Network, name)

            def record_fit(self, *args, _name=name, _original=original, **kwargs):
                captures["fits"][_name] = dict(edges=_edges(self), options=kwargs)
                return _original(self, *args, **kwargs)

            stack.enter_context(patch.object(oe.Network, name, record_fit))
        original_prediction = oe.NetworkBlockResult.expected_edges

        def record_prediction(self, pairs, **kwargs):
            pairs = list(pairs)
            frame = original_prediction(self, pairs, **kwargs)
            captures["predictions"].append((pairs, frame.copy()))
            return frame

        stack.enter_context(patch.object(oe.NetworkBlockResult, "expected_edges", record_prediction))
        original_links = oe.Network.link_prediction

        def record_links(self, pairs, **kwargs):
            pairs = list(pairs)
            frame = original_links(self, pairs, **kwargs)
            captures["links"][kwargs["method"]] = dict(pairs=pairs, frame=frame.copy())
            return frame

        stack.enter_context(patch.object(oe.Network, "link_prediction", record_links))
        original_qap = oe.Network.qap_regression

        def record_qap(self, predictors, **kwargs):
            captures["qap"] = dict(response=_edges(self), predictors={name: _edges(graph)
                for name, graph in predictors.items()})
            return original_qap(self, predictors, **kwargs)

        stack.enter_context(patch.object(oe.Network, "qap_regression", record_qap))
        try:
            result = lab.run_lab(REDUCED, display_callback=outputs.append)
            assert list(folder.iterdir()) == [], "run_lab wrote unsolicited export files"
        finally:
            torch.set_num_threads(threads)
    return result, captures, outputs


def _frames(result):
    return {row["label"]: row["network"]
            for row in result["charts"]["timeline"].config["network"]["frames"]}


def test_lab_reports_scope_finite_results_and_preserves_method_metadata(computation):
    result, _, outputs = computation
    meta, proof = result["metadata"], result["proof"]
    assert meta["nodes"] == REDUCED["firms"] and len(meta["months"]) == 8
    assert meta["config"] == {**REDUCED, "months": 8, "groups": 6, "seed": 731}
    assert meta["output_count"] == len(outputs) == len(result["tables"]) + len(result["charts"]) <= 20
    assert meta["default_exports"] is False and meta["out_of_core"] is False
    assert meta["resident_sparse"] is True and meta["analytical_device"] == "cpu"
    assert meta["dtype"] == "float64" and meta["synthetic"] is True
    assert "not whole-process RSS" in meta["memory_budget_scope"]
    json.dumps(meta, allow_nan=False)
    json.dumps(proof, allow_nan=False)
    assert proof["pagerank_sum"] == pytest.approx(1, abs=1e-8)
    assert proof["brokerage"]["samples"] == REDUCED["betweenness_samples"]
    assert proof["brokerage"]["exact"] is False
    for name, frame in result["tables"].items():
        assert frame.attrs["synthetic"] is True and frame.attrs["caption"]
        for row in frame.to_dict(orient="records"):
            for field, value in row.items():
                if value is None or pd.isna(value):
                    if name == "block_calibration":
                        assert field in {"mean_poisson_nll", "positive_mae", "zero_mae", "mae", "count_bias", "presence_brier"}
                        if field == "mean_poisson_nll":
                            assert row["impossible_positive_pairs"] > 0 or row["evaluated_pairs"] == 0
                        elif field == "positive_mae":
                            assert row["positive_pairs"] == 0
                        elif field == "zero_mae":
                            assert row["zero_pairs"] == 0
                        else:
                            assert row["evaluated_pairs"] == 0
                    else:
                        assert name == "mrqap" and row["term"] == "Constant"
                        assert field in {"statistic", "pvalue", "permutations", "extreme_permutations"}
                elif isinstance(value, (int, float)):
                    assert math.isfinite(value), (name, field, value)
    attrs = result["tables"]["mrqap"].attrs
    assert attrs["kind"] == "network_qap_regression"
    assert attrs["device"] == "cpu" and attrs["dtype"] == "float64"
    assert attrs["pvalue_resolution"] == pytest.approx(1 / (REDUCED["permutations"] + 1))
    assert "approximate" in attrs["inference_scope"] and "node-exchangeability" in attrs["null"]
    assert result["tables"]["temporal_route"].attrs["traversal"].startswith("strictly increasing")
    assert result["tables"]["turnover"].attrs["weight_changes"] == "ignored; presence only"


def test_full_chart_payloads_reconcile_counts_and_temporal_contact_witnesses(computation):
    result, _, _ = computation
    charts, meta = result["charts"], result["metadata"]
    assert {name: spec.kind for name, spec in charts.items()} == {
        "shock_counts": "d3", "system": "network", "resilience": "d3",
        "timeline": "network", "block_intensity": "d3", "skill_network": "network"}
    frames = _frames(result)
    assert list(frames) == meta["months"]
    # Start on the default first frame. The installed renderer's explicit
    # initial-frame path can overlap worker jobs and trigger its watchdog.
    assert "frame_index" not in charts["timeline"].config["options"]
    assert charts["timeline"].config["options"]["layout"] == "fixed"
    total = Counter()
    positions = {}
    for index, frame in enumerate(frames.values()):
        assert not frame["sampled"] and frame["directed"]
        assert frame["shown_node_count"] == frame["node_count"] == REDUCED["firms"]
        assert frame["shown_edge_count"] == frame["edge_count"] == meta["monthly_edges"][index]
        weights = _payload_edges(frame)
        assert all(value > 0 and value.is_integer() for value in weights.values())
        for node in frame["nodes"]:
            identity = (node["identity"]["type"], node["identity"]["value"])
            assert node["fixed"] is True
            assert node["x"] == pytest.approx(3 * node["longitude"])
            assert node["y"] == pytest.approx(-3 * node["latitude"])
            point = (node["x"], node["y"])
            assert positions.setdefault(identity, point) == point
        total.update(weights)
        assert sum(weights.values()) == result["tables"]["scenarios"].actual_count.iloc[index]
    system = charts["system"].config["network"]
    assert not system["sampled"] and system["node_count"] == system["shown_node_count"] == meta["nodes"]
    assert system["edge_count"] == system["shown_edge_count"] == meta["aggregate_edges"] == len(total)
    assert _payload_edges(system) == dict(total)
    assert sum(total.values()) == meta["aggregate_count"]
    skills = charts["skill_network"].config["network"]
    assert not skills["sampled"] and skills["shown_node_count"] == REDUCED["skills"]
    assert skills["shown_edge_count"] == result["proof"]["bipartite"]["projection_edges"]
    route = result["tables"]["temporal_route"]
    assert route.attrs["reachable"] and len(route) == route.attrs["hops"]
    assert route.position.tolist() == sorted(set(route.position))
    assert route.source.iloc[0] == "F0000" and route.target.iloc[-1] == "F0008"
    assert route.target.iloc[:-1].tolist() == route.source.iloc[1:].tolist()
    for row in route.itertuples():
        assert row.layer == meta["months"][row.position]
        assert (row.source, row.target) in _payload_edges(frames[row.layer])
    assert route.attrs["arrival_layer"] == route.layer.iloc[-1]
    for row in result["tables"]["turnover"].itertuples():
        before, after = set(_payload_edges(frames[row.from_layer])), set(_payload_edges(frames[row.to_layer]))
        assert row.edges_added == len(after - before)
        assert row.edges_removed == len(before - after)
        assert row.edges_persisted == len(before & after)
        assert row.edge_jaccard == pytest.approx(len(before & after) / len(before | after))


def test_common_holdout_dyads_exposure_and_mae_are_recomputed_from_actual_forecasts(computation):
    result, capture, _ = computation
    proof = result["proof"]["block_holdout"]
    frames = _frames(result)
    assert proof["train_month"] < proof["test_month"]
    assert proof["exposure_months"] == 1 and proof["evaluation_mask"] == "common identified intersection"
    assert not proof["likelihoods_compared"] and not proof["global_optimum_certified"]
    assert not proof["known_planted_initialization"]
    first, second = capture["predictions"]
    pairs = first[0]
    assert pairs == second[0] and len(pairs) == len(set(pairs)) == proof["pairs"] == REDUCED["link_candidates"]
    assert all(a != b for a, b in pairs) and _digest(pairs) == proof["pairs_sha256"]
    common = [a and b for a, b in zip(first[1].identified, second[1].identified)]
    observed_map = _payload_edges(frames[proof["test_month"]])
    observed = [observed_map.get(pair, 0) for pair in pairs]
    assert proof["future_nonzero"] == sum(value > 0 for value in observed)
    for row, prediction in zip(result["tables"]["block_validation"].itertuples(), [first[1], second[1]]):
        assert capture["fits"][row.model]["edges"] == _payload_edges(frames[proof["train_month"]])
        assert "initial" not in capture["fits"][row.model]["options"]
        assert list(zip(prediction.source, prediction.target)) == pairs
        assert row.identified_pairs == sum(common) and row.model_identified_pairs == sum(prediction.identified)
        errors = [abs(mean - actual) for mean, actual, admitted in zip(prediction.mean_count, observed, common) if admitted]
        assert row.mae_identified == pytest.approx(sum(errors) / len(errors))
        assert row.zero_baseline_mae == pytest.approx(sum(value for value, admitted in zip(observed, common) if admitted) / sum(common))
        assert row.iterations <= REDUCED["block_iterations"] and row.starts == REDUCED["block_starts"]
        trace = proof["optimization_traces"][row.model]
        assert row.stopping_reason == trace[0]["stopping_reason"]
        assert row.selected_seed == trace[0]["seed"]


def test_count_calibration_uses_common_dyads_and_training_only_baseline(computation):
    result, capture, _ = computation
    proof = result["proof"]["block_holdout"]
    frames = _frames(result)
    first, second = capture["predictions"]
    common = [a and b for a, b in zip(first[1].identified, second[1].identified)]
    observed_map = _payload_edges(frames[proof["test_month"]])
    actual = [observed_map.get(pair, 0.) for pair, use in zip(first[0], common) if use]
    training = _payload_edges(frames[proof["train_month"]])
    rate = sum(value for (a, b), value in training.items() if a != b) / (REDUCED["firms"] * (REDUCED["firms"] - 1))
    assert proof["training_rate_baseline"]["mean_count"] == pytest.approx(rate)
    assert proof["training_rate_baseline"]["train_month"] < proof["test_month"]
    predictions = [[mean for mean, use in zip(frame.mean_count, common) if use]
                   for _, frame in [first, second]] + [[0.] * len(actual), [rate] * len(actual)]
    for row, means in zip(result["tables"]["block_calibration"].itertuples(), predictions):
        assert row.evaluated_pairs == len(actual) and row.exposure_months == 1
        assert row.observed_count == sum(actual) and row.predicted_count == pytest.approx(sum(means))
        assert row.positive_pairs == sum(value > 0 for value in actual)
        assert row.zero_pairs == sum(value == 0 for value in actual)
        for field, condition in [("mae", lambda y: True), ("positive_mae", lambda y: y > 0),
                                 ("zero_mae", lambda y: y == 0)]:
            errors = [abs(mean - value) for value, mean in zip(actual, means) if condition(value)]
            assert getattr(row, field) == pytest.approx(sum(errors) / len(errors))
        assert row.count_bias == pytest.approx(sum(mean - value for value, mean in zip(actual, means)) / len(actual))
        assert row.presence_brier == pytest.approx(sum(
            (1 - math.exp(-mean) - (value > 0))**2 for value, mean in zip(actual, means)) / len(actual))
        impossible = sum(value > 0 and mean == 0 for value, mean in zip(actual, means))
        assert row.impossible_positive_pairs == impossible
        if impossible:
            assert pd.isna(row.mean_poisson_nll)
        else:
            nll = sum(mean + math.log(math.factorial(int(value))) -
                      (value * math.log(mean) if value else 0.) for value, mean in zip(actual, means)) / len(actual)
            assert row.mean_poisson_nll == pytest.approx(nll, abs=1e-12)


@pytest.mark.parametrize("means", [[.5, 1., 0., 0., 10.], [.5, 1., 0., 2., 10.]])
def test_count_scores_do_not_clip_impossible_events_or_include_unidentified_pairs(lab, means):
    actual = torch.tensor([0., 2., 0., 3., 99.], dtype=torch.float64)
    predicted = torch.tensor(means, dtype=torch.float64)
    mask = torch.tensor([True, True, True, True, False])
    scores = lab._count_scores(actual, predicted, mask)
    assert scores["evaluated_pairs"] == 4 and scores["observed_count"] == 5
    assert scores["positive_pairs"] == scores["zero_pairs"] == 2
    assert scores["zero_mae"] == .25
    assert scores["predicted_count"] == sum(means[:4])
    if means[3] == 0:
        assert scores["impossible_positive_pairs"] == 1 and scores["mean_poisson_nll"] is None
    else:
        expected = (.5 + 1 + math.log(2) + 2 + math.log(6) - 3 * math.log(2)) / 4
        assert scores["impossible_positive_pairs"] == 0
        assert scores["mean_poisson_nll"] == pytest.approx(expected)
    json.dumps(scores, allow_nan=False)


@pytest.mark.parametrize("observed,mask", [([0., 0.], [True, True]), ([2., 3.], [True, True]),
                                        ([2., 3.], [False, False])])
def test_count_scores_keep_empty_positive_zero_and_evaluation_subsets_explicit(lab, observed, mask):
    scores = lab._count_scores(torch.tensor(observed, dtype=torch.float64),
                              torch.ones(2, dtype=torch.float64), torch.tensor(mask))
    if not scores["positive_pairs"]:
        assert scores["positive_mae"] is None
    if not scores["zero_pairs"]:
        assert scores["zero_mae"] is None
    if not scores["evaluated_pairs"]:
        assert scores["mae"] is scores["presence_brier"] is scores["mean_poisson_nll"] is None
    json.dumps(scores, allow_nan=False)


def test_mrqap_lag_response_and_sector_sample_have_no_future_information(computation):
    result, capture, _ = computation
    proof, frames = result["proof"]["mrqap"], _frames(result)
    sample = proof["sample_nodes"]
    assert len(sample) == len(set(sample)) == REDUCED["mrqap_nodes"]
    assert Counter(int(node[1:]) % 12 for node in sample) == dict.fromkeys(range(12), len(sample) // 12)
    assert _digest(sample) == proof["sample_sha256"]
    assert proof["lag_month"] < proof["response_month"] < result["proof"]["links"]["outcome_months"][0]
    assert proof["sample_selected_without_outcomes"] and not proof["causal"]
    assert proof["pvalues_exploratory"] and not proof["exchangeability_verified"]
    selected = set(sample)
    for name, month in [("response", proof["response_month"]), ("lagged_count", proof["lag_month"])]:
        expected = {pair: math.log1p(value) for pair, value in _payload_edges(frames[month]).items()
                    if set(pair) <= selected}
        actual = capture["qap"]["response"] if name == "response" else capture["qap"]["predictors"][name]
        assert actual == pytest.approx(expected)
    # Spherical law of cosines is independent of the example's haversine
    # calculation; planar degree-distance regressions would fail this check.
    locations = {node["identity"]["value"]: (math.radians(node["latitude"]),
                 math.radians(node["longitude"])) for node in frames[proof["lag_month"]]["nodes"]}
    affinity = capture["qap"]["predictors"]["geographic_affinity"]
    for (a, b), actual in list(affinity.items())[:20]:
        lat_a, lon_a = locations[a]
        lat_b, lon_b = locations[b]
        cosine = math.sin(lat_a) * math.sin(lat_b) + math.cos(lat_a) * math.cos(lat_b) * math.cos(lon_a - lon_b)
        distance = 6371.0088 * math.acos(min(1., max(-1., cosine)))
        assert actual == pytest.approx(1 / (1 + distance / 1000), abs=1e-9)
    frame = result["tables"]["mrqap"]
    assert set(frame.dyads) == {len(sample) * (len(sample) - 1)}
    for row in frame[frame.term != "Constant"].itertuples():
        assert row.pvalue == pytest.approx((1 + row.extreme_permutations) / (1 + row.permutations))
        assert row.permutations == REDUCED["permutations"]
    constant = frame[frame.term == "Constant"].iloc[0]
    assert pd.isna(constant.pvalue) and pd.isna(constant.statistic)


def test_frozen_training_nonedge_candidates_and_matched_flow_certificates(computation):
    result, capture, _ = computation
    proof, frames = result["proof"], _frames(result)
    links = proof["links"]
    first, second = capture["links"].values()
    pairs = first["pairs"]
    assert pairs == second["pairs"] and len(pairs) == len(set(pairs)) == REDUCED["link_candidates"]
    assert _digest(pairs) == links["candidates_sha256"]
    assert max(links["training_months"]) < min(links["outcome_months"])
    training = {tuple(sorted(pair)) for month in links["training_months"] for pair in _payload_edges(frames[month])}
    future = {tuple(sorted(pair)) for month in links["outcome_months"] for pair in _payload_edges(frames[month])}
    assert not set(pairs) & training and all(a < b for a, b in pairs)
    assert sum(pair in future for pair in pairs) == links["future_positives"]
    assert not links["probabilities_calibrated"]
    for row in result["tables"]["ranked_links"].itertuples():
        assert row.appeared_later == int((row.source, row.target) in future)
    scenarios = result["tables"]["scenarios"]
    assert (scenarios.actual_count <= scenarios.control_count).all()
    assert scenarios.actual_count.iloc[:4].tolist() == scenarios.control_count.iloc[:4].tolist()
    flow = proof["flow"]
    assert flow["shock"] <= flow["control"] and "proxy" in flow["capacity_semantics"]
    assert len(flow["certificates"]) == 2
    for row in flow["certificates"]:
        assert row["certified"] and row["month"] == flow["matched_month"]
        assert row["max_flow"] == pytest.approx(row["cut_capacity"], abs=1e-8)
        assert row["conservation_error"] < 1e-8 and row["cut_edges"] > 0
    for name in ["targeted", "random"]:
        curve = result["tables"]["resilience"][name].tolist()
        assert all(0 <= value <= 1 for value in curve)
        assert all(after <= before for before, after in zip(curve, curve[1:]))


def test_changing_block_holdout_cannot_change_fits_predictions_or_training_baseline(lab, computation, tmp_path, monkeypatch):
    baseline, _, _ = computation
    simulate = lab._simulate
    month = baseline["proof"]["block_holdout"]["test_month"]

    def doubled_holdout(c):
        state = simulate(c)
        graph = state["actual"][month]
        state["actual"][month] = graph.edit_edges(weights={pair: 2 * count
                                                         for pair, count in _edges(graph).items()})
        mask = state["monthly"].month == month
        state["monthly"].loc[mask, "actual_count"] *= 2
        return state

    monkeypatch.setattr(lab, "_simulate", doubled_holdout)
    monkeypatch.chdir(tmp_path)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        changed = lab.run_lab(REDUCED)
    finally:
        torch.set_num_threads(threads)
    original, revised = [result["proof"]["block_holdout"] for result in [baseline, changed]]
    assert original["future_nonzero"] == revised["future_nonzero"] > 0
    for field in ["pairs_sha256", "training_rate_baseline", "optimization_traces"]:
        assert revised[field] == original[field]
    for before, after in zip(original["calibration"], revised["calibration"]):
        assert after["predicted_count"] == before["predicted_count"]
        assert after["evaluated_pairs"] == before["evaluated_pairs"]
        assert after["observed_count"] == 2 * before["observed_count"]
        assert after["positive_pairs"] == before["positive_pairs"]
    assert list(tmp_path.iterdir()) == []


def test_erasing_only_future_outcomes_cannot_change_candidates_scores_or_fits(lab, computation, tmp_path, monkeypatch):
    baseline, _, _ = computation
    simulate = lab._simulate

    def no_future(c):
        state = simulate(c)
        for month in list(state["actual"])[4:]:
            graph = state["actual"][month]
            state["actual"][month] = graph.edit_edges(remove=list(_edges(graph)))
            mask = state["monthly"].month == month
            state["monthly"].loc[mask, ["actual_count", "actual_edges", "retention"]] = 0
        return state

    monkeypatch.setattr(lab, "_simulate", no_future)
    monkeypatch.chdir(tmp_path)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        changed = lab.run_lab(REDUCED)
    finally:
        torch.set_num_threads(threads)
    assert list(tmp_path.iterdir()) == []
    assert baseline["proof"]["links"]["future_positives"] > 0
    assert changed["proof"]["links"]["future_positives"] == 0
    assert changed["proof"]["links"]["candidates_sha256"] == baseline["proof"]["links"]["candidates_sha256"]
    assert changed["proof"]["mrqap"]["sample_sha256"] == baseline["proof"]["mrqap"]["sample_sha256"]
    pd.testing.assert_frame_equal(changed["tables"]["block_validation"], baseline["tables"]["block_validation"])
    pd.testing.assert_frame_equal(changed["tables"]["block_calibration"], baseline["tables"]["block_calibration"])
    pd.testing.assert_frame_equal(changed["tables"]["mrqap"], baseline["tables"]["mrqap"])
    columns = ["source", "target", "score"]
    pd.testing.assert_frame_equal(changed["tables"]["ranked_links"][columns], baseline["tables"]["ranked_links"][columns])


@pytest.mark.parametrize("scores,outcomes,ap,auc", [
    ([3, 3, 2, 1], [1, 0, 1, 0], 7 / 12, 5 / 8),
    ([1, 1, 1, 1], [1, 0, 1, 0], .5, .5),
    ([2, 1], [1, 0], 1., 1.),
    ([2, 1], [0, 0], None, None),
    ([2, 1], [1, 1], 1., None),
])
def test_ranking_utilities_match_hand_computed_ties_and_undefined_auc(lab, scores, outcomes, ap, auc):
    result = lab._ranking_metrics(scores, outcomes, k=2)
    assert result["average_precision"] == (pytest.approx(ap) if ap is not None else None)
    assert result["roc_auc"] == (pytest.approx(auc) if auc is not None else None)
    assert result["prevalence"] == pytest.approx(sum(outcomes) / len(outcomes))


def test_exact_example_runs_as_whole_panel_script_with_inherited_runtime_argv(computation, tmp_path):
    _, _, expected_outputs = computation
    workspace = Workspace(tmp_path / "owned-panel")
    session = ConsoleSession(workspace)
    try:
        setup = session.execute("import sys\nsys.argv=['openecon-runtime','--port','0','--data-root','owned']\n"
                                f"NETWORK_LAB_CONFIG = {REDUCED!r}")
        assert setup["status"] == "ok"
        source = SOURCE.read_text()
        run = session.execute(source, timeout_seconds=90)
        assert run["status"] == "ok", run.get("error")
        assert "Display limit reached" not in run["stdout"]
        assert len(run["outputs"]) == len(expected_outputs) == 18
        validate_output_events(run["events"], run["stdout"], run["outputs"])
        assert [event["index"] for event in run["events"] if event["type"] == "output"] == list(range(len(expected_outputs)))
        for output, expected in zip(run["outputs"], expected_outputs):
            if isinstance(expected, oe.DataFrame):
                assert output["type"] == "table"
                assert output["data"]["columns"] == list(expected.columns)
                assert output["data"]["total_rows"] == len(expected)
            else:
                assert output["type"] == "plot"
                actual = output["data"]
                if "artifact" in actual:
                    actual = json.loads(checked_plot_path(workspace.path, actual["artifact"]).read_text())
                from openecon_charts.timeline import unpack
                actual = unpack(actual)
                assert actual == expected.model_dump()
        assert Workspace(workspace.path).console_history()[-1]["code"] == source
        assert Workspace(workspace.path).console_history()[-1]["outputs"] == run["outputs"]
    finally:
        session.close()
