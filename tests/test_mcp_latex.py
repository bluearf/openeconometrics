"""Saved MCP publication exports preserve inference without exposing observation rows."""

from copy import deepcopy
import importlib
import json

import numpy as np
import pandas as pd
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from openecon.data import DataError
from openecon.latex import Latex
from openecon.mcp_latex import MAX_LATEX_COEFFICIENTS, result_latex_fields
from openecon.mcp_server import compact_result, create_mcp_server
from openecon.workspace import Workspace


@pytest.fixture
def saved(tmp_path):
    rng = np.random.default_rng(2047)
    label = r"income_%\input{untrusted-label}"
    outcome = "Ücret & wages"
    x = rng.normal(size=36)
    frame = pd.DataFrame(
        {
            label: x,
            outcome: 2 + 0.7 * x + rng.normal(size=36),
            "frequency_%": np.tile([1, 2, 3], 12),
            "private": [f"private-person-{i}" for i in range(36)],
        }
    )
    frame.loc[3, label] = np.nan
    source = tmp_path / "data.csv"
    frame.to_csv(source, index=False)
    store = Workspace(tmp_path / "workspace")
    dataset = store.import_file(source)
    result = store.run_analysis(
        dataset["id"],
        {
            "outcome": outcome,
            "predictors": [label],
            "covariance": "HC3",
            "weights": "frequency_%",
            "weight_type": "fweight",
            "missing": "drop",
            "options": {"hansen": True},
        },
    )
    return store, result


def test_get_result_opt_in_preserves_default_schema_and_saved_values_without_refit(
    saved, monkeypatch
):
    store, original = saved
    server = create_mcp_server(store.path)
    tool = server._tool_manager.get_tool("get_result")
    assert tool.parameters["properties"]["include_latex"] == {
        "default": False,
        "title": "Include Latex",
        "type": "boolean",
    }
    assert tool.parameters["required"] == ["result_id"]
    record = store.result_path / (original["id"] + ".json")
    before = record.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("Reading saved LaTeX must not fit, import data, or publish a record")

    for name in (
        "compute_analysis",
        "run_analysis",
        "load_frame",
        "import_file",
        "save_agent_result",
    ):
        monkeypatch.setattr(Workspace, name, forbidden)
    default = tool.fn(original["id"])
    explicit_false = tool.fn(original["id"], include_latex=False)
    exported = tool.fn(original["id"], include_latex=True)
    assert default == explicit_false == compact_result(original)
    assert {key: value for key, value in exported.items() if not key.startswith("latex")} == default
    assert exported["observation_data_included"] is False
    assert exported["latex_math"].startswith(r"\begin{array}")
    assert record.read_bytes() == before
    assert len(store.list_results()) == 1
    assert store.display_history() == []


def test_saved_weighted_hansen_sample_notes_and_escaped_labels_are_preserved(saved):
    _, original = saved
    result = deepcopy(original)
    result["warnings"].append(r"Caution & \input{warning-code}")
    result["spec"]["outcome"] = Latex(r"literal\input{outcome-code}")
    before = deepcopy(result)
    exported = result_latex_fields(result)
    source = exported["latex"]
    notes = " ".join(exported["latex_notes"])
    for text in (
        "HC3 heteroskedasticity-robust standard errors",
        "frequency weights",
        "physical estimation sample 35 of 36 observations",
        "1 observations excluded",
        "coefficient-specific Bell-McCaffrey/Satterthwaite",
        "Hansen (2025)",
        "model F(",
    ):
        assert text in source or text in notes
    assert f"effective observations = {original['nobs']}" in notes
    assert r"frequency\_\%" in source
    assert r"income\_\%\textbackslash{}input\{untrusted-label\}" in source
    assert r"literal\textbackslash{}input\{outcome-code\}" in source
    assert r"\input{" not in source and r"\input{" not in exported["latex_math"]
    assert exported["latex_packages"] == ["amsmath", "booktabs", "adjustbox"]
    assert result == before


def test_fresh_projection_omits_stored_tex_and_all_prediction_state(saved, monkeypatch):
    _, original = saved
    result = deepcopy(original)
    secret = "hidden-observation-sentinel"
    result.update(latex=r"\input{stored-code}", latex_math=secret, latex_notes=[secret])
    result["extra"].update(
        fitted_rows=[secret],
        query=[[secret]],
        query_estimates=[42.0],
        smoother_state={"observations": [secret]},
        penalized_state={"rows": [secret]},
    )
    result["predictions"] = [{"row": 999, "private": secret}]
    result["sample_positions"] = [secret]
    result["covariance_matrix"] = [[secret]]
    result["inference"]["observation_influence"] = [secret]
    result["provenance"]["training_rows"] = [secret]
    renderer = importlib.import_module("openecon.latex")
    actual = renderer.display_latex

    def check_projection(model, **kwargs):
        assert model.predictions == model.sample_positions == model.covariance_matrix == []
        assert secret not in model.model_dump_json()
        assert kwargs == {"max_rows": None, "max_columns": None}
        return actual(model, **kwargs)

    monkeypatch.setattr(renderer, "display_latex", check_projection)
    exported = result_latex_fields(result)
    assert secret not in json.dumps(exported)
    assert "stored-code" not in json.dumps(exported)


@pytest.mark.parametrize("target_location", ["extra", "inference"])
def test_prediction_only_results_fail_before_renderer_can_export_hidden_queries(
    saved, monkeypatch, target_location
):
    _, original = saved
    result = deepcopy(original)
    result[target_location]["target"] = "prediction"
    result["inference"]["available"] = False
    result["extra"].update(
        smoother_state={}, fitted_rows=["private"], query=[[123.0]], query_estimates=[456.0]
    )
    renderer = importlib.import_module("openecon.latex")
    monkeypatch.setattr(
        renderer,
        "display_latex",
        lambda *args, **kwargs: pytest.fail("Unsafe prediction rendering"),
    )
    with pytest.raises(DataError, match="Prediction-only") as error:
        result_latex_fields(result)
    assert error.value.code == "LATEX_UNAVAILABLE"


def test_unavailable_and_unknown_ols_inference_are_not_invented(saved):
    _, original = saved
    unavailable = deepcopy(original)
    unavailable["inference"]["available"] = False
    with pytest.raises(DataError, match="unavailable coefficient inference"):
        result_latex_fields(unavailable)
    unknown = deepcopy(original)
    del unknown["inference"]["use_t"]
    with pytest.raises(DataError, match="OLS inference distribution is unavailable"):
        result_latex_fields(unknown)


def test_actual_synthetic_provenance_adds_note_without_changing_saved_warnings(saved):
    _, original = saved
    result = deepcopy(original)
    result["provenance"]["synthetic_data"] = True
    before = deepcopy(result)
    exported = result_latex_fields(result)
    assert "Synthetic example data" in exported["latex"]
    assert exported["latex_notes"][-1].startswith("Synthetic example data")
    assert result == before


def test_complete_math_longtable_and_package_hints_have_no_preview_truncation(saved):
    _, original = saved
    result = deepcopy(original)
    prototype = result["coefficients"][1]
    result["coefficients"] = [
        {**prototype, "term": f"long coefficient {i} " + "x" * 45} for i in range(24)
    ]
    for key in ("coefficient_df", "coefficient_scale", "hansen"):
        result["inference"].pop(key, None)
    exported = result_latex_fields(result)
    assert r"\begin{longtable}" in exported["latex"]
    assert "long coefficient 23" in exported["latex_math"]
    assert "Preview:" not in exported["latex_math"]
    assert exported["latex_packages"] == ["amsmath", "booktabs", "longtable", "array"]


def test_limits_and_incomplete_saved_model_tests_fail_explicitly_without_truncation(
    saved, monkeypatch
):
    _, original = saved
    oversized = deepcopy(original)
    oversized["coefficients"] = [original["coefficients"][0]] * (MAX_LATEX_COEFFICIENTS + 1)
    with pytest.raises(DataError, match="No table was truncated") as error:
        result_latex_fields(oversized)
    assert error.value.code == "LATEX_LIMIT"
    invalid = deepcopy(original)
    invalid["tests"]["model"].pop("df2")
    with pytest.raises(DataError, match="cannot safely produce"):
        result_latex_fields(invalid)
    monkeypatch.setattr("openecon.mcp_latex.MAX_MATH_BYTES", 8)
    with pytest.raises(DataError, match="No table was truncated"):
        result_latex_fields(original)


@pytest.mark.parametrize("field,value", [("df2", "unknown"), ("df", 0), ("p_value", 1.1)])
def test_invalid_saved_model_test_is_rejected_instead_of_published(saved, field, value):
    _, original = saved
    result = deepcopy(original)
    result["tests"]["model"][field] = value
    with pytest.raises(DataError, match="cannot safely produce") as error:
        result_latex_fields(result)
    assert error.value.code == "LATEX_UNAVAILABLE"


def test_complete_saved_chi_square_test_and_all_source_bounds(saved):
    _, original = saved
    result = deepcopy(original)
    result["tests"]["model"] = {
        "distribution": "chi2",
        "statistic": 4.1,
        "df": 2,
        "p_value": 0.12,
    }
    exported = result_latex_fields(result)
    assert "model Wald chi-square(2) = 4.1, p = 0.12" in " ".join(exported["latex_notes"])
    for warning in ("large input " * 30000, "\\" * 48000):
        oversized = deepcopy(original)
        oversized["warnings"].append(warning)
        with pytest.raises(DataError, match="No table was truncated") as error:
            result_latex_fields(oversized)
        assert error.value.code == "LATEX_LIMIT"


def test_get_result_reports_actionable_mcp_error_while_default_remains_retrievable(
    saved, monkeypatch
):
    store, original = saved
    server = create_mcp_server(store.path)
    tool = server._tool_manager.get_tool("get_result")
    monkeypatch.setattr("openecon.mcp_latex.MAX_MATH_BYTES", 8)
    with pytest.raises(ToolError, match=r"\[LATEX_LIMIT\].*without include_latex"):
        tool.fn(original["id"], include_latex=True)
    assert tool.fn(original["id"]) == compact_result(original)
