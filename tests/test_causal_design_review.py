"""Cross-family regressions for numerical support, exact labels and CPU routing."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.resources import use_workspace_budget


def _case(name, data=None):
    if data is None:
        data = pd.DataFrame(
            {
                "t": [0, 0, 0, 1, 1, 1],
                "x": [0.0, 1.0, 2.0, 0.25, 1.0, 1.75],
                "y": [0.0, 1.0, 2.0, 1.0, 2.0, 3.0],
                "pair": ["a", "b", "c", "a", "b", "c"],
                "time": [1.0, 2.0, 4.0, 1.0, 3.0, 4.0],
                "event": [1, 0, 1, 0, 1, 1],
            }
        )
    function = getattr(oe, name)
    if name == "ebalance":
        return function(data, "t", ["x"])
    if name == "cem":
        return function(data, "t", ["x"], cutpoints={"x": [1.0]})
    if name == "balance":
        return function(data, "t", ["x"])
    if name == "rosenbaum_bounds":
        return function(data, "y", "t", "pair")
    if name == "paired_randomization":
        return function(data, "y", "t", "pair", design="paired_randomized")
    if name == "treatment_cdf":
        return function(data, "y", "t", design="randomized", thresholds=[1.0])
    if name == "treatment_quantile":
        return function(data, "y", "t", design="randomized", quantiles=[0.5], reps=19, seed=701)
    return function(data, "time", "event", "t", design="randomized", tau=3.0)


NAMES = (
    "ebalance",
    "cem",
    "balance",
    "rosenbaum_bounds",
    "paired_randomization",
    "treatment_cdf",
    "treatment_quantile",
    "treatment_rmst",
)


@pytest.mark.parametrize("name", NAMES)
def test_explicit_cpu_route_survives_non_cpu_global_factory_context(name):
    expected = oe.causal_design_save(_case(name))
    with torch.device("meta"):
        assert torch.empty(0).device.type == "meta"
        observed = oe.causal_design_save(_case(name))
        assert torch.empty(0).device.type == "meta"
    assert observed == expected


@pytest.mark.parametrize("name", NAMES)
def test_all_methods_reject_selected_sample_workspace_before_computation(name):
    n = 10_000
    data = pd.DataFrame(
        {
            "t": [0] * (n // 2) + [1] * (n // 2),
            "x": np.linspace(-1.0, 1.0, n),
            "y": np.linspace(-1.0, 1.0, n),
            "pair": list(range(n // 2)) * 2,
            "time": [4.0] * n,
            "event": [1] * n,
        }
    )
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        _case(name, data)
    assert error.value.code == "workspace_limit"


@pytest.mark.parametrize("offset", [0.0, 1e-11])
def test_entropy_joint_hull_boundary_cannot_pass_strict_numerical_support(offset):
    # All donors satisfy x+z <= 1. Positive donor weights put their means
    # strictly below the supporting face, even though both coordinate ranges
    # individually contain the target. Coordinate tests alone are insufficient.
    data = pd.DataFrame(
        {
            "t": [0, 0, 0, 0, 1, 1],
            "x": [0.0, 1.0, 0.0, 0.2, 0.5, 0.5],
            "z": [0.0, 0.0, 1.0, 0.2, 0.5 + offset, 0.5 + offset],
        }
    )
    with pytest.raises(AnalysisError) as error:
        oe.ebalance(data, "t", ["x", "z"])
    assert error.value.code in (
        "infeasible_balance",
        "insufficient_support",
        "balance_nonconvergence",
    )


def test_entropy_interior_near_supporting_face_remains_admitted():
    data = pd.DataFrame(
        {
            "t": [0, 0, 0, 0, 1, 1],
            "x": [0.0, 1.0, 0.0, 0.2, 0.5, 0.5],
            "z": [0.0, 0.0, 1.0, 0.2, 0.49, 0.49],
        }
    )
    result = oe.ebalance(data, "t", ["x", "z"])
    probabilities = np.asarray(result.attrs["state"]["control_probability_weights"])
    assert (probabilities > 0).all()
    np.testing.assert_allclose(
        probabilities @ data.loc[data.t == 0, ["x", "z"]], [0.5, 0.49], atol=1e-9
    )


@pytest.mark.parametrize("name", ["ebalance", "balance"])
def test_positive_weights_cannot_silently_normalize_to_zero(name):
    if name == "ebalance":
        data = pd.DataFrame(
            {
                "t": [0, 0, 0, 1, 1],
                "x": [0.0, 1.0, 2.0, 0.5, 1.5],
                "w": [1.0, 1.0, 1.0, 1e-250, 1e150],
            }
        )
        kwargs = {"base_weights": "w"}
    else:
        data = pd.DataFrame(
            {"t": [0, 0, 1, 1], "x": [0.0, 1.0, 2.0, 3.0], "w": [1e-250, 1e150, 1e-250, 1e150]}
        )
        kwargs = {"balance_weights": "w"}
    with pytest.raises(AnalysisError) as error:
        getattr(oe, name)(data, "t", ["x"], **kwargs)
    assert error.value.code in ("numerical_failure", "invalid_weights", "insufficient_support")


def test_cem_boolean_and_numeric_categories_do_not_create_false_common_support():
    data = pd.DataFrame({"t": [0, 1, 0, 1], "cat": pd.Series([True, 1, False, 0], dtype=object)})
    with pytest.raises(AnalysisError) as error:
        oe.cem(data, "t", [], cutpoints={}, categorical=["cat"])
    assert error.value.code == "no_common_support"


def test_cem_exact_type_preservation_keeps_distinct_supported_strata():
    data = pd.DataFrame({"t": [0, 1, 0, 1], "cat": pd.Series([True, True, 1, 1], dtype=object)})
    result = oe.cem(data, "t", [], cutpoints={}, categorical=["cat"])
    assert len(result["strata"]) == 2
    np.testing.assert_array_equal(result["weights"].weight, 1.0)


@pytest.mark.parametrize("name", NAMES)
def test_artifact_preserves_table_order_dtypes_and_every_scientific_value(name):
    original = _case(name)
    artifact = json.loads(json.dumps(oe.causal_design_save(original)))
    restored = oe.causal_design_load(artifact)
    assert list(restored) == list(original)
    for table in original:
        pd.testing.assert_frame_equal(restored[table], original[table])
    assert oe.causal_design_save(restored) == artifact


def test_large_typed_numeric_pair_identity_survives_table_and_complete_artifact():
    identifier = 2**60 + 1
    data = pd.DataFrame(
        {
            "pair": pd.Series([identifier, identifier, 1.5, 1.5], dtype=object),
            "t": [0, 1, 0, 1],
            "y": [0.0, 1.0, 0.0, 2.0],
        }
    )
    result = oe.rosenbaum_bounds(data, "y", "t", "pair")
    assert int(result["pairs"].pair.iloc[0]) == identifier
    assert result.attrs["state"]["pair_rows"][0][0] == identifier
    artifact = json.loads(json.dumps(oe.causal_design_save(result)))
    restored = oe.causal_design_load(artifact)
    assert int(restored["pairs"].pair.iloc[0]) == identifier
    pd.testing.assert_frame_equal(restored["pairs"], result["pairs"])


def test_balance_overflow_cannot_turn_a_scientific_infinite_cell_into_missing_on_load():
    data = pd.DataFrame({"t": [0, 0, 1, 1], "x": [-1e-150, 1e-150, -1e150, 1e150]})
    with pytest.raises(AnalysisError) as error:
        oe.balance(data, "t", ["x"])
    assert error.value.code == "numerical_failure"


def test_fixed_weight_hc1_scores_are_formed_before_squaring_tiny_probabilities():
    data = pd.DataFrame(
        {"t": [0, 0, 1, 1], "x": [0.0, 1e150, -1e-100, 1e-100], "w": [1.0, 1e-250, 1.0, 1.0]}
    )
    result = oe.balance(data, "t", ["x"], balance_weights="w")
    # The small donor's probability squared is outside float64, but its
    # weighted outcome score and score square are representable. The control
    # HC1 variance is4e-200 and treated HC1 variance is1e-200.
    np.testing.assert_allclose(result["covariance"].iloc[0, 0], 5e-200, rtol=1e-13, atol=0)
    np.testing.assert_allclose(
        result["balance"].std_error.iloc[0], np.sqrt(5e-200), rtol=1e-13, atol=0
    )


def test_varying_bootstrap_quantile_contrasts_do_not_claim_zero_empirical_variance():
    data = pd.DataFrame(
        {"t": [0, 0, 0, 1, 1, 1], "y": np.array([0.0, 1.0, 2.0, 1.0, 2.0, 3.0]) * 1e-180}
    )
    with pytest.raises(AnalysisError) as error:
        oe.treatment_quantile(data, "y", "t", design="randomized", quantiles=[0.5], reps=29)
    assert error.value.code == "numerical_failure"


def test_nonconstant_balance_outcomes_do_not_claim_zero_empirical_variance():
    data = pd.DataFrame({"t": [0, 0, 1, 1], "x": np.array([0.0, 1.0, 2.0, 3.0]) * 1e-180})
    with pytest.raises(AnalysisError) as error:
        oe.balance(data, "t", ["x"])
    assert error.value.code == "numerical_failure"


@pytest.mark.parametrize("key,value", [("procedure", "cem"), ("n", 999), ("positions", [99])])
def test_save_refuses_top_level_scientific_metadata_inconsistent_with_hashed_state(key, value):
    result = _case("balance")
    result.attrs[key] = value
    with pytest.raises(AnalysisError) as error:
        oe.causal_design_save(result)
    assert error.value.code == "invalid_result"
