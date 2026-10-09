"""Independent complete-fit plus empirical-target PSU replication checks."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError


_spec = importlib.util.spec_from_file_location(
    "complete_empirical_target_oracle",
    Path(__file__).resolve().parents[1]/"scripts/validate_survey_replicate_margins.py",
)
reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference)
FAMILIES = ("logit", "probit", "poisson")
METHODS = reference.METHODS
TARGETS = ("mean", "ame")


def fixture(*, fpc=False):
    frame = reference.primitive_frame()
    if fpc:
        frame["N"] = np.repeat([4, 6, 8], 16)
    design = oe.survey_design(frame, weights="w", psu="p", strata="h", fpc="N" if fpc else None)
    return frame, design, reference.bootstrap_weights(frame)


def specification(target):
    return {"profiles": reference.PROFILES if target == "mean" else None,
            "variables": ["x", "z"] if target == "ame" else None}


def fit(frame, design, family, method, target, *, supplied=None, **overrides):
    options = {**reference.default_options(method), **specification(target), "tolerance": 1e-11}
    options.update(overrides)
    if method == "bootstrap" and supplied is None:
        supplied = reference.bootstrap_weights(frame)
    if supplied is not None:
        options.pop("replicates", None)
        options["replicate_weights"] = supplied
    return oe.survey_margins_replicate(
        frame, design, "c" if family == "poisson" else "b", ["x", "z"],
        family=family, method=method, target=target, **options,
    )


def assert_reference(actual, expected, *, alpha=.05, null=0):
    assert actual.labels == tuple(expected["labels"])
    assert actual.df == expected["df"]
    np.testing.assert_allclose(actual.estimates, expected["estimates"], atol=4e-9, rtol=5e-8)
    np.testing.assert_allclose(actual.covariance, expected["covariance"], atol=4e-9, rtol=5e-8)
    np.testing.assert_allclose(actual.replicate_estimates, expected["replicas"], atol=4e-9, rtol=5e-8)
    source = actual.source_result
    np.testing.assert_allclose(source.coefficients, expected["coefficients"], atol=4e-9, rtol=5e-8)
    np.testing.assert_allclose(source.covariance, expected["coefficient_covariance"], atol=4e-9, rtol=5e-8)
    record = source.metadata["replication"]
    assert record["replicate_ids"] == expected["ids"]
    assert record["failed_replicates"] == []
    np.testing.assert_allclose(record["replicate_estimates"], expected["coefficient_replicas"], atol=4e-9, rtol=5e-8)
    np.testing.assert_allclose(record["variance_multipliers"], expected["multipliers"], atol=1e-13, rtol=1e-13)
    assert source.metadata["sample_positions"] == np.flatnonzero(expected["selected"]).tolist()
    table = actual.to_frame()
    assert table.index.tolist() == expected["labels"]
    assert table.columns.tolist() == ["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"]
    np.testing.assert_allclose(table.to_numpy(), reference.inference(
        expected["estimates"], expected["covariance"], expected["df"], alpha=alpha, null=null), atol=4e-8, rtol=5e-8)
    np.testing.assert_allclose(table.attrs["covariance_matrix"], expected["covariance"], atol=4e-9, rtol=5e-8)
    assert table.attrs["survey_replicate_margins_state"] == actual.to_state()
    assert table.attrs["empirical_covariate_uncertainty"] is True
    assert table.attrs["coefficient_uncertainty"] is True


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("target", TARGETS)
def test_all_24_complete_refits_and_empirical_targets_with_full_covariance_t_inference_and_restore(family, method, target):
    frame, design, _ = fixture()
    original = frame.copy(deep=True)
    expected = reference.oracle(frame, family, method, target=target, **reference.default_options(method))
    actual = fit(frame, design, family, method, target, alpha=.1, null=.15)
    assert_reference(actual, expected, alpha=.1, null=.15)
    assert np.count_nonzero(abs(expected["covariance"]-np.diag(expected["covariance"].diagonal())) > 1e-10) == (6 if target == "mean" else 2)
    assert actual.family == family and actual.method == method and actual.target == target
    assert actual.source_result.metadata["n_design"] == 48
    state = actual.to_state()
    restored = oe.SurveyReplicateMarginsResult.from_state(json.dumps(state, allow_nan=False))
    assert restored.to_state() == state
    pd.testing.assert_frame_equal(restored.to_frame(), actual.to_frame())
    assert oe.SurveyReplicateMarginsResult.from_state(state).to_state() == state
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("target", TARGETS)
def test_population_covariate_reweighting_is_distinct_from_fixed_distribution_replay_and_coefficient_delta(family, method, target):
    frame, design, _ = fixture()
    expected = reference.oracle(frame, family, method, target=target, **reference.default_options(method))
    actual = fit(frame, design, family, method, target)
    assert_reference(actual, expected)
    # Fixed original averaging weights omit sampled covariate-distribution
    # uncertainty even if all coefficient refits themselves are retained.
    gap = np.max(abs(expected["covariance"]-expected["fixed_covariance"]))
    assert gap > 1e-10
    assert np.max(abs(np.asarray(actual.covariance)-expected["delta_covariance"])) > 1e-10


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("target", TARGETS)
def test_alternate_centering_recomputes_the_complete_target_vector(family, method, target):
    frame, design, _ = fixture()
    center = "original" if method in {"bootstrap", "jackknife"} else "replicate_mean"
    options = {**reference.default_options(method), "centering": center}
    expected = reference.oracle(frame, family, method, target=target, **options)
    actual = fit(frame, design, family, method, target, centering=center)
    assert_reference(actual, expected)
    assert actual.source_result.metadata["replication"]["centering"] == center


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("target", TARGETS)
def test_domain_missing_keeps_original_psu_geometry_and_uses_replica_specific_denominators(family, method, target):
    frame, design, weights = fixture()
    frame.index = np.tile([37, -2, 37, 9], 12)
    frame.iloc[40:, frame.columns.get_loc("domain")] = 0
    outcome = "c" if family == "poisson" else "b"
    frame.iloc[40:, frame.columns.get_loc(outcome)] = np.nan
    frame.iloc[2, frame.columns.get_loc("x")] = np.nan
    before = frame.copy(deep=True)
    with pytest.raises(AnalysisError):
        fit(frame, design, family, method, target, domain="domain")
    actual = fit(frame, design, family, method, target, domain="domain", missing="drop",
                 supplied=weights if method == "bootstrap" else None)
    expected = reference.oracle(frame, family, method, target=target, domain="domain", **reference.default_options(method))
    assert_reference(actual, expected)
    source = actual.source_result
    assert source.metadata["outcome_exclusions"] == [2]
    assert source.metadata["out_of_domain_positions"] == list(range(40, 48))
    assert source.metadata["n_used"] == 39 and source.metadata["n_design"] == 48
    assert actual.df == 3
    for diagnostic, replica_weights in zip(source.metadata["replication"]["replicate_convergence"], expected["weights"]):
        members = expected["selected"] & (replica_weights > 0)
        assert diagnostic["n_used"] == int(members.sum())
        assert diagnostic["sample_positions"] == np.flatnonzero(members).tolist()
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("target", TARGETS)
def test_common_scale_of_base_and_supplied_psu_weights_preserves_complete_targets(family, method, target):
    frame, design, weights = fixture()
    baseline = fit(frame, design, family, method, target)
    scaled = frame.assign(w=frame.w*37)
    declaration = oe.survey_design(scaled, weights="w", psu="p", strata="h")
    actual = fit(scaled, declaration, family, method, target, supplied=weights*37 if method == "bootstrap" else None)
    np.testing.assert_allclose(actual.estimates, baseline.estimates, atol=4e-9, rtol=5e-8)
    np.testing.assert_allclose(actual.replicate_estimates, baseline.replicate_estimates, atol=4e-9, rtol=5e-8)
    np.testing.assert_allclose(actual.covariance, baseline.covariance, atol=4e-9, rtol=5e-8)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("center", ["original", "stratum_mean"])
def test_jackknife_census_fpc_keeps_certainty_psu_weights_and_complete_target_variance(family, target, center):
    frame, _, _ = fixture(fpc=True)
    frame.loc[frame.h == 0, "N"] = 2
    design = oe.survey_design(frame, weights="w", psu="p", strata="h", fpc="N")
    expected = reference.oracle(frame, family, "jackknife", target=target, centering=center)
    actual = fit(frame, design, family, "jackknife", target, centering=center)
    assert_reference(actual, expected)
    assert len(expected["ids"]) == 4
    assert all(not identity.startswith("stratum-0") for identity in expected["ids"])
    np.testing.assert_allclose(expected["weights"][:, :16], np.broadcast_to(frame.w[:16], (4, 16)))


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("method", METHODS)
def test_saturated_two_group_empirical_ame_has_closed_form_without_likelihood_optimizer(family, method):
    frame, _, _ = fixture()
    frame["x"] = np.tile([0., 0., 0., 0., 1., 1., 1., 1.], 6)
    design = oe.survey_design(frame, weights="w", psu="p", strata="h")
    options = reference.default_options(method)
    supplied = reference.bootstrap_weights(frame)
    actual = oe.survey_margins_replicate(
        frame, design, "c" if family == "poisson" else "b", ["x"], family=family, method=method,
        target="ame", variables=["x"], tolerance=1e-11, **options,
        **({"replicate_weights": supplied} if method == "bootstrap" else {}),
    )
    weights, _, multipliers, strata, df = reference.replica_plan(frame, method, **options)
    y = frame.c.to_numpy() if family == "poisson" else frame.b.to_numpy()
    def target(weight):
        mean = [weight[frame.x == group]@y[frame.x == group]/weight[frame.x == group].sum() for group in (0, 1)]
        if family == "logit":
            eta = special.logit(mean)
        elif family == "probit":
            eta = stats.norm.ppf(mean)
        else:
            eta = np.log(mean)
        slope = eta[1]-eta[0]
        indexes = eta[0]+slope*frame.x.to_numpy()
        return np.array([slope*(weight@reference.response_derivative(family, indexes))/weight.sum()])
    estimate = target(frame.w.to_numpy())
    replicas = np.array([target(weight) for weight in weights])
    covariance = reference.target_covariance(replicas, estimate, multipliers, options.get("centering", "original"), strata)
    assert actual.labels == ("AME:x",) and actual.df == df
    np.testing.assert_allclose(actual.estimates, estimate, atol=4e-10, rtol=5e-9)
    np.testing.assert_allclose(actual.replicate_estimates, replicas, atol=4e-10, rtol=5e-9)
    np.testing.assert_allclose(actual.covariance, covariance, atol=4e-10, rtol=5e-9)
    np.testing.assert_allclose(actual.to_frame().to_numpy(), reference.inference(estimate, covariance, df), atol=4e-8, rtol=5e-8)


def test_example_all_eight_states_and_displayed_payloads_match_independent_full_target_oracle(tmp_path):
    path = Path(__file__).resolve().parents[1]/"docs/examples/survey_replicate_margins_eight.py"
    displayed = []
    scope = {"display": displayed.append, "__name__": "__source_example__"}
    exec(compile(path.read_text(), str(path), "exec"), scope)
    states, inputs, tables = (scope[name] for name in ("states", "oracle_inputs", "poststates"))
    assert len(displayed) == 8 and tuple(states) == reference.CASES
    receipt = reference.verify(states, inputs, tables)
    assert receipt["gates"] == 8 and receipt["full_covariance_verified"] is True
    assert receipt["population_distribution_uncertainty"] is True
    assert receipt["native_display_verified"] is False
    for case, table in zip(reference.CASES, displayed):
        pd.testing.assert_frame_equal(table, scope["models"][case].to_frame())
    payload = {"states": states, "poststates": tables, "oracle_inputs": inputs}
    saved = tmp_path/"margins.json"
    saved.write_text(json.dumps(payload, allow_nan=False))
    assert json.loads(saved.read_text()) == payload
    # Object keys carry no sequence semantics. Target ordering is explicitly
    # declared by labels, including after canonical/sorted JSON serialization.
    sorted_payload = json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))
    assert reference.verify(sorted_payload["states"], sorted_payload["oracle_inputs"],
                            sorted_payload["poststates"])["gates"] == 8
    for case in reference.CASES:
        state = oe.SurveyReplicateMarginsResult.from_state(json.dumps(sorted_payload["states"][case]))
        pd.testing.assert_frame_equal(state.to_frame(), scope["models"][case].to_frame())
    primitive_mutated = deepcopy(payload)
    primitive_mutated["states"]["brr_mean"]["metadata"]["primitive"]["groups"][0] = 1
    with pytest.raises(AssertionError):
        reference.verify(primitive_mutated["states"], primitive_mutated["oracle_inputs"],
                         primitive_mutated["poststates"])
    mutated = deepcopy(tables)
    mutated["brr_mean"]["data"][0][0] += .1
    with pytest.raises(AssertionError):
        reference.verify(states, inputs, mutated)
    execution = {"status": "ok", "error": None, "outputs": [
        {"type": "table", "latex": "source transport test", "data": {
            "columns": tables[case]["columns"], "rows": tables[case]["data"],
            "index": [[label] for label in tables[case]["index"]], "index_names": [None],
            "total_rows": len(tables[case]["index"]), "total_columns": 6,
        }} for case in reference.CASES
    ]}
    # Tests transport validation, not evidence of actual installed execution.
    assert reference.verify(states, inputs, tables, execution)["native_display_verified"] is True
    execution["outputs"][3]["data"]["index"][0] = ["wrong label"]
    with pytest.raises(AssertionError):
        reference.verify(states, inputs, tables, execution)
