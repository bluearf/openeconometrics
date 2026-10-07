"""Real Python display/export and safe presentation of saved cloud results."""
from copy import deepcopy
import json

import pytest

from openecon.console_worker import _execute
from openecon.output_latex import (
    MAX_LATEX_BYTES, MAX_MATH_BYTES, MAX_OUTPUT_BYTES, PUBLICATION_STYLE,
    add_output_latex, enrich_record, validate_latex_fields,
)
from openecon.team_server import validate_worker_result
from openecon.team_store import TeamError


def test_pooled_timeline_console_display_latex_and_shared_validation_preserve_graph():
    import openecon as oe
    from openecon.team_output import validate_plot
    record = _execute("""graph=oe.network([{'source':1,'target':'1','weight':2}],
nodes=[1,'1'], directed=True,weight='weight')
snapshots=oe.network_snapshots({str(i):graph for i in range(8)},ordered=True)
plot=oe.plot.network(snapshots,title='Pooled timeline',frame_index=3)
display(plot)
""", {"oe": oe}, "pooled-latex")
    assert record["status"] == "ok" and not record["stdout"]
    output, = record["outputs"]
    assert output["type"] == "plot" and "tabular" in output["latex"]
    assert output["data"]["config"]["network"]["encoding"] == "timeline-pool-v1"
    shared = validate_plot(output["data"])
    assert len(shared["config"]["network"]["frames"]) == 8
    assert shared["config"]["options"]["frame_index"] == 3
    assert {x["identity"]["type"] for x in shared["config"]["network"]["nodes"]} == {"integer", "string"}
    original = deepcopy(output["data"])
    assert enrich_record(record)["outputs"][0]["latex"] == output["latex"]
    assert output["data"] == original
    malformed = deepcopy(original)
    malformed["config"]["network"]["base"]["nodes"][0] = 99999
    with pytest.raises(ValueError):
        validate_plot(malformed)


def test_console_dataframe_model_plot_and_explicit_latex_share_exports():
    import openecon as oe
    record = _execute("""df = oe.example()
display(df.head(3))
model = oe.ols(data=df, y='wage', x=['education', 'experience'])
display(model)
display(oe.plot.coefficients(model))
df.head(2).to_latex()
""", {"oe": oe}, "latex-test")
    assert record["status"] == "ok", record["error"]
    assert [item["type"] for item in record["outputs"]] == ["table", "model", "plot", "latex"]
    for item in record["outputs"]:
        assert isinstance(item["latex"], str) and item["latex"]
    table, model, plot, explicit = record["outputs"]
    assert table["data"]["total_rows"] == 3
    assert "education" in table["latex"]
    assert model["data"]["nobs"] == 480
    assert "experience" in model["latex"]
    assert "tikzpicture" in plot["latex"]
    assert "tabular" in explicit["data"]
    assert table["latex_math"] and model["latex_math"]
    assert table["latex_style"] == model["latex_style"] == PUBLICATION_STYLE


def test_auto_table_export_matches_bounded_visible_data_without_hidden_rows():
    import openecon as oe
    record = _execute("df = oe.DataFrame({'x': range(75)})\ndf", {"oe": oe}, "bounded-test")
    assert record["status"] == "ok", record["error"]
    item = record["outputs"][0]
    assert len(item["data"]["rows"]) == 50
    assert item["data"]["total_rows"] == 75
    assert "% Displayed 50 of 75 rows" in item["latex"]
    assert "74" not in item["latex"]
    assert len(json.dumps(record["outputs"]).encode()) < MAX_OUTPUT_BYTES


def test_forecast_notes_survive_console_json_and_saved_preview_enrichment():
    import openecon as oe

    frame = oe.DataFrame({"forecast": range(75)})
    notes = ["Beyond one-step: plug-in approximation.", r"Parameter uncertainty excluded & \input{unsafe}"]
    frame.attrs["publication_notes"] = notes
    direct = frame.to_latex(notes=["User note."])
    assert direct.notes == [*notes, "User note."]
    record = _execute("display(frame)", {"frame": frame}, "forecast-notes")
    assert record["status"] == "ok", record["error"]
    item, = record["outputs"]
    assert item["data"]["publication_notes"] == item["latex_notes"] == notes
    assert len(item["data"]["rows"]) == 50
    assert "plug-in approximation" in item["latex"]
    assert r"\input{unsafe}" not in item["latex"]
    saved = json.loads(json.dumps(record))
    original = deepcopy(saved)
    del saved["outputs"][0]["latex"]
    enriched = enrich_record(saved)
    assert enriched["outputs"][0]["latex_notes"] == notes
    assert enriched["outputs"][0]["latex"] == original["outputs"][0]["latex"]
    run = {"id": "forecast-notes", "code": "display(frame)", "created_at": "2026-10-07",
           "generation": 1, "email": "actor@example.com"}
    shared, _ = validate_worker_result(
        {"execution_id": "forecast-notes", "record": record, "generated_files": []}, run
    )
    assert shared["outputs"] == record["outputs"]


@pytest.mark.parametrize("notes", [None, "text", [1], ["x" * (MAX_MATH_BYTES + 1)]])
def test_serialized_forecast_notes_reject_invalid_or_oversized_data(notes):
    item = {"type": "table", "data": {"columns": ["forecast"], "rows": [[1.0]],
            "publication_notes": notes, "total_rows": 1, "total_columns": 1}}
    with pytest.raises(ValueError):
        add_output_latex(item)


def test_index_only_dataframe_displays_every_row_and_exports_the_same_index():
    import openecon as oe
    record = _execute("oe.DataFrame(index=['row_a', 'row_b'])", {"oe": oe}, "index-only")
    assert record["status"] == "ok" and not record["stdout"]
    item, = record["outputs"]
    assert item["type"] == "table"
    assert item["data"]["rows"] == [[], []]
    assert item["data"]["index"] == [["row_a"], ["row_b"]]
    assert item["data"]["total_rows"] == 2
    assert r"row\_a" in item["latex"] and r"row\_b" in item["latex_math"]


def test_large_math_preview_keeps_complete_valid_latex_source():
    import openecon as oe
    frame = oe.DataFrame({str(i): ["x" * 500] * 10 for i in range(20)})
    source = frame.to_latex()
    assert len(source.math.encode()) > MAX_MATH_BYTES
    assert len(source.encode()) < MAX_LATEX_BYTES
    record = _execute("source", {"source": source}, "large-preview")
    assert record["status"] == "ok" and not record["stdout"]
    item, = record["outputs"]
    assert item["data"] == item["latex"] == source
    assert item["latex_math"] is None


def test_saved_history_exports_escape_cells_and_do_not_rewrite_stored_record():
    original = {"outputs": [{"type": "table", "data": {
        "columns": ["x_y"], "rows": [[r"A&B\input{secret}"]],
        "index": [["a_b"]], "index_names": ["row%"],
        "total_rows": 1, "total_columns": 1,
    }}]}
    baseline = deepcopy(original)
    enriched = enrich_record(original)
    assert original == baseline
    source = enriched["outputs"][0]["latex"]
    assert r"x\_y" in source and r"A\&B" in source
    assert r"\input{secret}" not in source
    assert enriched["outputs"][0]["latex_math"]


def test_saved_automatic_exports_upgrade_from_original_numbers_without_rewriting():
    import openecon as oe
    model = oe.ols(data=oe.example(), y="wage", x=["education"])
    original = {"outputs": [
        {"type": "table", "data": {"columns": ["x"], "rows": [[1.25]],
         "index": [[0]], "total_rows": 1, "total_columns": 1},
         "latex": "old table", "latex_math": "old preview"},
        {"type": "model", "data": model.model_dump(mode="json"),
         "latex": "old model", "latex_math": "old preview"},
    ]}
    baseline = deepcopy(original)
    enriched = enrich_record(original)
    assert original == baseline
    for before, after in zip(original["outputs"], enriched["outputs"], strict=True):
        assert after["data"] == before["data"]
        assert after["latex"] != before["latex"]
        assert after["latex_math"] != before["latex_math"]
        assert after["latex_style"] == PUBLICATION_STYLE
    assert "1.2500" in enriched["outputs"][0]["latex"]
    assert enrich_record(enriched) == enriched


def test_user_written_tex_remains_exactly_as_written():
    source = r"\begin{tabular}{|r|}\hline 42 \\ \hline\end{tabular}"
    item = {"type": "latex", "data": source, "latex": source,
            "latex_math": None}
    baseline = deepcopy(item)
    assert add_output_latex(item) == baseline
    assert "latex_style" not in item


def test_styled_charts_keep_options_latex_and_interleaved_execution_order_when_shared():
    import openecon as oe
    run = {"id": "chart-options", "code": "", "created_at": "2026-10-04", "generation": 1,
           "email": "actor@example.com"}
    record = _execute("""print('before')
display(oe.plot.scatter(data={'x': [1, 10], 'y': [2, 20]}, x='x', y='y',
    x_scale='log', color='#123456', x_label='Income', grid=False))
print('between')
display(oe.plot.barh(data={'region': ['A', 'B'], 'value': [4, 8]},
    x='region', y='value', xlim=(0, 10), x_label='Amount', legend=False))
print('after')
""", {"oe": oe}, "chart-options")
    assert record["status"] == "ok", record["error"]
    assert record["events"] == [
        {"type": "stdout", "text": "before\n"}, {"type": "output", "index": 0},
        {"type": "stdout", "text": "between\n"}, {"type": "output", "index": 1},
        {"type": "stdout", "text": "after\n"},
    ]
    payload = {"execution_id": "chart-options", "record": record, "generated_files": []}
    shared, generated = validate_worker_result(payload, run)
    assert generated == []
    assert shared["events"] == record["events"]
    for original, output in zip(record["outputs"], shared["outputs"], strict=True):
        assert output["data"] == original["data"]
        assert output["latex"] == original["latex"]
    assert shared["outputs"][0]["data"]["config"]["options"]["x_scale"] == "log"
    assert "xmode=log" in shared["outputs"][0]["latex"]
    assert "Income" in shared["outputs"][0]["latex"]
    assert shared["outputs"][1]["data"]["config"]["options"]["xlim"] == [0, 10]
    assert "Amount" in shared["outputs"][1]["latex"]


@pytest.mark.parametrize("field,value", [
    ("latex", {}), ("latex", None), ("latex_math", []),
    ("latex", "a" * (MAX_LATEX_BYTES + 1)),
    ("latex_math", "a" * (MAX_MATH_BYTES + 1)),
    ("latex", "ğ" * MAX_LATEX_BYTES),
    ("latex_style", {}), ("latex_style", None),
    ("latex_style", "unknown-style"),
    ("latex_notes", {}), ("latex_notes", [42]), ("latex_notes", None),
    ("latex_notes", ["ğ" * MAX_MATH_BYTES]),
])
def test_bad_presentation_is_rejected_before_sharing_with_another_member(field, value):
    run = {"id": "run1", "code": "1", "created_at": "2026-10-02", "generation": 1,
           "email": "actor@example.com"}
    item = {"type": "latex", "data": "x", field: value}
    payload = {"execution_id": "run1", "record": {"status": "ok", "outputs": [item]},
               "generated_files": []}
    with pytest.raises(TeamError):
        validate_worker_result(payload, run)


def test_explicit_latex_and_older_json_outputs_remain_valid_cloud_results():
    run = {"id": "run1", "code": "1", "created_at": "2026-10-02", "generation": 1,
           "email": "actor@example.com"}
    outputs = [{"type": "text", "data": "old result"},
               {"type": "latex", "data": r"\frac{1}{2}", "latex": r"\frac{1}{2}",
                "latex_math": r"\frac{1}{2}"}]
    payload = {"execution_id": "run1", "record": {"status": "ok", "outputs": outputs},
               "generated_files": []}
    record, generated = validate_worker_result(payload, run)
    assert record["outputs"] == outputs
    assert generated == []
    validate_latex_fields({"latex": "x", "latex_math": None})
