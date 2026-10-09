"""Evidence gates must reject wrong inference and broken physical identity."""
import copy
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).parents[1] / "benchmarks/native_model_matrix_replay.py"
    spec = importlib.util.spec_from_file_location("native_matrix_replay", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def bundle():
    import openecon as oe
    return oe.ols(data=oe.example(), y="wage", x=["education", "experience"]).model_dump()


@pytest.mark.parametrize("mutation", ["covariance", "sample", "p_value", "diagnostic", "prediction"])
def test_matrix_rejects_corrupted_full_inference_sample_diagnostic_and_prediction(bundle, mutation):
    changed = copy.deepcopy(bundle)
    if mutation == "covariance":
        changed["covariance_matrix"][0][1] += 1
    elif mutation == "sample":
        changed["dropped_rows"] += 1
    elif mutation == "p_value":
        changed["coefficients"][0]["p_value"] = .5
    elif mutation == "diagnostic":
        changed["tests"]["model"]["statistic"] += 10
    else:
        changed["predictions"][0]["fitted"] += 5
    result = module().compare(bundle, changed, relative=1e-9, absolute=1e-10)
    assert result["status"] == "failed" and result["errors"]


def test_matrix_requires_each_full_covariance_row_and_each_reference_metric(bundle):
    changed = copy.deepcopy(bundle)
    changed["covariance_matrix"].pop()
    changed["metrics"].pop("r_squared")
    result = module().compare(bundle, changed, relative=1e-9, absolute=1e-10)
    assert any("covariance_matrix" in error for error in result["errors"])
    assert any("metrics/r_squared" in error for error in result["errors"])


def test_matrix_compares_preview_by_physical_position_instead_of_list_order(bundle):
    changed = copy.deepcopy(bundle)
    changed["predictions"].reverse()
    result = module().compare(bundle, changed, relative=1e-9, absolute=1e-10)
    assert result["status"] == "passed"
    assert result["prediction_rows_compared"] == len(bundle["predictions"])
