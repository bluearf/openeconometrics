"""Marginal Product and Labor Demand: run the whole file after importing the supplied workbook."""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import pandas as pd
import torch
import openecon as oe
from openecon.models import ResultBundle

DATA_FILE = "production_input.xlsx"
LAB = "09-production-input"
TITLE = "Marginal Product and Labor Demand"


def load_data(data_path=None):
    """Read the prepared file or its most recent imported workspace snapshot."""
    if data_path is not None:
        return oe.DataFrame(pd.read_excel(Path(data_path).expanduser(), sheet_name="Data"))
    candidates = [Path(DATA_FILE)]
    if globals().get("__file__"):
        candidates.insert(0, Path(__file__).resolve().with_name(DATA_FILE))
    for candidate in candidates:
        if candidate.is_file():
            return oe.DataFrame(pd.read_excel(candidate, sheet_name="Data"))
    workspace_path = os.environ.get("OPENECON_WORKSPACE")
    if workspace_path:
        from openecon.workspace import Workspace

        workspace = Workspace(workspace_path)
        matches = [
            r for r in workspace.list_datasets() if r["name"] in (DATA_FILE, Path(DATA_FILE).stem)
        ]
        if matches:
            return oe.DataFrame(workspace.load_frame(matches[0]["id"]))
    raise FileNotFoundError(f"Import {DATA_FILE} without renaming it, or supply data_path.")


def mean(values):
    return math.fsum(values) / len(values)


def variance(values):
    center = mean(values)
    return math.fsum((x - center) ** 2 for x in values) / (len(values) - 1)


def histogram(values, bins=18):
    counts, edges = torch.histogram(torch.tensor(values, dtype=torch.float64), bins=bins)
    return {"x": ((edges[1:] + edges[:-1]) / 2).tolist(), "y": counts.tolist()}


def panel(x, y, *, kind="line", title=TITLE, xlabel="Observation", ylabel="Value"):
    return {
        "kind": kind,
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "series": [{"label": "Computed values", "x": list(x), "y": list(y)}],
    }


def chart(panel):
    """Draw a chart_data panel with OpenEconometrics Charts, as the handout figure does."""
    series = panel["series"]
    first = series[0]
    options = {"title": panel["title"]}
    if panel.get("value_limits"):
        options["xlim" if "low" in first else "ylim"] = list(panel["value_limits"])
    if panel.get("segments"):
        options["annotations"] = [
            {
                "type": "segment",
                "text": g["label"],
                **{k: g[k] for k in ("x0", "y0", "x1", "y1", "color", "align", "dx", "dy") if k in g},
            }
            for g in panel["segments"]
        ]
    if "low" in first:
        records = [
            {"term": str(term), "estimate": estimate, "ci_low": low, "ci_high": high}
            for term, estimate, low, high in zip(first["x"], first["y"], first["low"], first["high"])
        ]
        plot = oe.plot.coefficients(
            {"coefficients": records}, x_label=panel["ylabel"], y_label=panel["xlabel"], **options
        )
    elif panel["kind"] == "bar":
        frame = {panel["xlabel"]: [str(v) for v in first["x"]], **{s["label"]: s["y"] for s in series}}
        plot = oe.plot.bar(
            data=frame,
            x=panel["xlabel"],
            y=[s["label"] for s in series],
            legend=len(series) > 1,
            y_label=panel["ylabel"],
            **options,
        )
    else:
        plotter = oe.plot.scatter if panel["kind"] == "scatter" else oe.plot.line
        plot = plotter(
            data={"x": first["x"], "y": first["y"]},
            x="x",
            y="y",
            x_label=panel["xlabel"],
            y_label=panel["ylabel"],
            **options,
        )
    for line in panel.get("reference_lines", []):
        plot = (plot.hline if line["axis"] == "y" else plot.vline)(line["value"], text=line["label"])
    return plot


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def pack(value):
    if isinstance(value, ResultBundle):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {
            "tables": {k: pack(v) for k, v in value.items()},
            "attrs": clean(getattr(value, "attrs", {})),
        }
    return {
        "index": clean(value.index.tolist()),
        "columns": clean(value.columns.tolist()),
        "values": clean(value.values.tolist()),
        "attrs": clean(value.attrs),
    }


def analyze(raw):
    price = 2.0
    wage = 5.0
    fixed = 10.0
    labor = (price * 5 / wage) ** 2
    q = 10 * math.sqrt(labor)
    profits = price * raw.output - wage * raw.labor - fixed
    summary = {
        "output_price": price,
        "wage": wage,
        "optimal_labor": labor,
        "optimal_output": q,
        "marginal_product_at_optimum": 5 / math.sqrt(labor),
        "profit_after_fixed_cost": price * q - wage * labor - fixed,
        "grid_profit_maximum": float(profits.max()),
    }
    native = oe.DataFrame({"value": list(summary.values())}, index=list(summary))
    shown = raw.labor <= 16
    demand = {
        **panel(
            raw.labor[shown].tolist(),
            (price * 5 / raw.labor[shown] ** 0.5).tolist(),
            xlabel="Labor (worker-days)",
            ylabel="Value of marginal product (currency/worker-day)",
        ),
        "reference_lines": [
            {"axis": "y", "value": wage, "label": f"Wage = {wage:g}"},
            {"axis": "x", "value": labor, "label": f"Profit-maximizing labor = {labor:g}"},
        ],
    }
    checks = {
        "workbook_output_is_ten_root_labor": max(abs(raw.output - 10 * raw.labor**0.5)) < 1e-9,
        "value_marginal_product_equals_wage": abs(price * 5 / math.sqrt(labor) - wage) < 1e-12,
        "analytic_matches_grid": abs(
            summary["grid_profit_maximum"] - summary["profit_after_fixed_cost"]
        )
        < 1e-12,
    }
    return (
        summary,
        native,
        [demand],
        checks,
        {},
    )


def run_lab(output_dir=None, display_callback=None, data_path=None):
    raw = load_data(data_path)
    summary, native, panels, checks, models = analyze(raw)
    if not checks or not all(checks.values()):
        raise AssertionError(checks)
    numbers = {
        k: float(v)
        for k, v in summary.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    table = oe.DataFrame({"value": list(numbers.values())}, index=list(numbers))
    result = {
        "metadata": {
            "lab": LAB,
            "title": TITLE,
            "synthetic": True,
            "data_file": DATA_FILE,
            "raw_rows": len(raw),
        },
        "summary": clean(summary),
        "models": {k: pack(v) for k, v in models.items()},
        "tables": {} if isinstance(native, ResultBundle) else {"native": pack(native)},
        "chart_data": {"panels": panels},
        "checks": clean(checks),
    }
    latex = str(table.to_latex(precision=6, caption=TITLE + " (original synthetic data)"))
    if display_callback is not None:
        display_callback(
            oe.DataFrame(pd.concat(native, names=["procedure"]))
            if isinstance(native, dict)
            else native
        )
        display_callback(chart(panels[0]))
        display_callback(table)
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "reference.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        (target / "table.tex").write_text(latex)
        if json.loads((target / "reference.json").read_text()) != result:
            raise AssertionError("Saved complete result differs from returned state")
    print(TITLE + ": " + ", ".join(f"{k}={v:.6g}" for k, v in numbers.items()))
    return result


if __name__ == "__main__":
    lab_result = run_lab(display_callback=globals().get("display"))
