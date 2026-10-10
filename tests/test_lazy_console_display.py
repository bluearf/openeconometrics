"""Dataset display and variables never enumerate a full replayable source."""
import pandas as pd

from openecon.console_worker import _execute, _variables, validate_result
from openecon.dataset import Dataset


def test_unknown_rows_display_is_a_bounded_latex_preview():
    seen = []
    def factory():
        for start in range(0, 5000, 50):
            seen.append(start)
            yield pd.DataFrame({"x": range(start, start + 50)})
    source = Dataset.from_batches(factory, ["x"])
    assert _variables({"df": source})[0]["preview"] == "unknown rows × 1 columns · batch source"
    assert not seen
    result = _execute("display(df)", {"df": source}, "lazy-display")
    assert result["status"] == "ok", result["error"]
    validate_result({"kind": "result", "id": "lazy-display", "result": result}, "lazy-display")
    table = result["outputs"][0]
    assert table["type"] == "table" and len(table["data"]["rows"]) == 50
    assert table["data"]["total_rows_known"] is False
    assert "total row count was not computed" in table["latex"]
    assert seen == [0]


def test_known_rows_display_preserves_global_shape_and_reports_first_rows():
    source = Dataset.from_frame(pd.DataFrame({"x": range(5000)}))
    result = _execute("display(df)", {"df": source}, "known-display")
    assert result["status"] == "ok", result["error"]
    data = result["outputs"][0]["data"]
    assert data["total_rows"] == 5000 and len(data["rows"]) == 50
    assert "total_rows_known" not in data
    assert _variables({"df": source})[0]["preview"].startswith("5,000 rows")


def test_posterior_types_render_real_tables_with_posterior_uncertainty_labels():
    from openecon.econometrics.bayesian.commands import (
        bayes_contrast, bayes_draws, bayes_linear,
    )
    posterior = bayes_linear(
        data=pd.DataFrame({"y": [1.0, 2.0, 4.0], "x": [0.0, 1.0, 2.0]}),
        y="y", x=["x"],
        prior={"mean": [0.0, 0.0], "scale_matrix": [[1.0, 0.0], [0.0, 1.0]],
               "shape": 2.0, "scale": 1.0},
    )
    namespace = {
        "posterior": posterior,
        "contrast": bayes_contrast(result=posterior, weights={"x": 1.0}),
        "draws": bayes_draws(result=posterior, draws=8, seed=735),
    }
    result = _execute("display(posterior)\ndisplay(contrast)\ndisplay(draws)",
                      namespace, "posterior-display")
    assert result["status"] == "ok", result["error"]
    validate_result({"kind": "result", "id": "posterior-display", "result": result},
                    "posterior-display")
    assert len(result["outputs"]) == 3
    for table in result["outputs"]:
        assert table["type"] == "table"
        assert table["data"]["rows"]
        assert "tabular" in table["latex"]
    columns = result["outputs"][0]["data"]["columns"]
    assert "Posterior std. dev." in columns and "Student-t scale" in columns
    assert "Credible lower" in columns and "Credible upper" in columns
    assert not {"p", "p-value", "CI lower", "CI upper", "t", "z"}.intersection(columns)
    assert "Probability above threshold" in result["outputs"][1]["data"]["columns"]
