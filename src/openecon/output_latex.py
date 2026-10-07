"""LaTeX presentation for bounded, already serialized console outputs.

The public control process can enrich saved results without importing pandas,
Torch, executing source, or modifying the stored execution record.
"""
from __future__ import annotations

from copy import deepcopy
import json

MAX_LATEX_BYTES = 512 * 1024
MAX_MATH_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
PUBLICATION_STYLE = "publication-v2"


def validate_latex_fields(item: dict) -> None:
    data = item.get("data")
    if item.get("type") == "table" and isinstance(data, dict) and "publication_notes" in data:
        validate_latex_fields({"latex_notes": data["publication_notes"]})
    if "latex_style" in item and item["latex_style"] not in ("publication-v1", PUBLICATION_STYLE):
        raise ValueError("Invalid LaTeX presentation style.")
    if "latex_notes" in item:
        notes = item["latex_notes"]
        if (not isinstance(notes, list) or not all(isinstance(note, str) for note in notes)
                or len(json.dumps(notes, ensure_ascii=False).encode("utf-8")) > MAX_MATH_BYTES):
            raise ValueError("Invalid or oversized publication notes.")
    for name, maximum in (("latex", MAX_LATEX_BYTES), ("latex_math", MAX_MATH_BYTES)):
        if name not in item:
            continue
        value = item[name]
        if name == "latex_math" and value is None:
            continue
        if not isinstance(value, str) or len(value.encode("utf-8")) > maximum:
            raise ValueError("Invalid or oversized LaTeX presentation.")


def add_output_latex(item: dict) -> dict:
    """Add a representation of the displayed data, never hidden dataset rows."""
    from openecon.latex import display_latex, table_latex

    validate_latex_fields(item)
    kind, data = item.get("type"), item.get("data")
    # An old automatic table may already have a diagnostic LaTeX export. Rebuild
    # that presentation from the saved numbers while preserving explicit user
    # TeX and the execution record. Current exports do not need regeneration.
    publication_table = kind in {"table", "model"}
    if "latex" in item and (
        not publication_table or item.get("latex_style") == PUBLICATION_STYLE
    ):
        return item
    if kind == "table":
        notes = data.get("publication_notes", [])
        validate_latex_fields({"latex_notes": notes})
        source = table_latex(data["columns"], data["rows"], index=data.get("index"),
                             index_names=data.get("index_names"), notes=notes)
        if notes:
            item["latex_notes"] = notes
        rows, columns = len(data["rows"]), len(data["columns"])
        if data.get("total_rows_known") is False:
            source = type(source)(f"% Displayed the first {rows} rows; total row count was not computed.\n" + str(source),
                                  math=getattr(source, "math", None))
        elif data.get("total_rows", rows) > rows or data.get("total_columns", columns) > columns:
            note = (f"% Displayed {rows} of {data.get('total_rows', rows)} rows and "
                    f"{columns} of {data.get('total_columns', columns)} columns.\n")
            source = type(source)(note + str(source), math=getattr(source, "math", None))
        math = getattr(source, "math", None)
    elif kind == "model":
        from openecon.models import ResultBundle
        fields = {key: value for key, value in data.items() if key != "display_omitted"}
        fields.setdefault("covariance_matrix", [])
        fields.setdefault("sample_positions", [])
        model = ResultBundle.model_validate(fields)
        source, math = display_latex(model, max_rows=50, max_columns=30)
        item["latex_notes"] = source.notes
    elif kind == "plot":
        from openecon.plotting import PlotSpec
        from openecon_charts.timeline import unpack
        source, math = PlotSpec(**unpack(data)).to_latex(), None
    elif kind == "latex":
        source, math = str(data), item.get("latex_math")
    elif kind == "text":
        source, math = display_latex(data)
    else:
        return item
    # Never cut TeX inside an environment or load an enormous math array into
    # the browser. A full dataframe can still be written explicitly to a file.
    if len(str(source).encode("utf-8")) > MAX_LATEX_BYTES:
        return item
    if math is not None and len(math.encode("utf-8")) > MAX_MATH_BYTES:
        math = None
    item.update(latex=str(source), latex_math=math)
    if publication_table:
        item["latex_style"] = PUBLICATION_STYLE
    validate_latex_fields(item)
    return item


def enrich_record(record: dict) -> dict:
    """Give older saved results the same export support without rewriting them."""
    enriched = deepcopy(record)
    outputs = enriched.get("outputs", [])
    for position, item in enumerate(outputs):
        candidate = deepcopy(item)
        try:
            add_output_latex(candidate)
            proposed = [*outputs[:position], candidate, *outputs[position + 1:]]
            if len(json.dumps(proposed, ensure_ascii=False).encode("utf-8")) <= MAX_OUTPUT_BYTES:
                outputs[position] = candidate
        except (KeyError, TypeError, ValueError):
            # Old records may predate the current table/model schema. Their
            # existing readable representation remains available.
            continue
    return enriched
