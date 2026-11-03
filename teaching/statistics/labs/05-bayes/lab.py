"""What Does a Positive Test Tell Us?: run the whole file after importing the supplied workbook."""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import pandas as pd
import torch
import openecon as oe
from openecon.models import ResultBundle

DATA_FILE = "antigen_tests.xlsx"
LAB = "05-bayes"
TITLE = "What Does a Positive Test Tell Us?"
DATA_SOURCE = "Pray et al. 2021, MMWR Table 2"
LICENCE = "public domain"
# Column roles; identical to "roles" in the lab's dataset.json.
ROLES = {
    "unit": "specimen_pair_id",
    "condition": "pcr_positive",
    "test": "antigen_positive",
    "group": "symptomatic",
}


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


def outcomes(frame):
    """Confusion-matrix counts and the rates that divide them by a row or a column."""
    d = frame[ROLES["condition"]] == 1
    a = frame[ROLES["test"]] == 1
    tp = int((d & a).sum())
    fp = int((~d & a).sum())
    fn = int((d & ~a).sum())
    tn = int((~d & ~a).sum())
    return {
        "pairs": len(frame),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "prevalence": (tp + fn) / len(frame),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "positive_predictive_value": tp / (tp + fp),
        "negative_predictive_value": tn / (tn + fn),
    }


def predictive_value(base_rate, sensitivity, specificity):
    """Bayes' theorem: the true-positive route over all routes into a positive result."""
    true_route = sensitivity * base_rate
    false_route = (1 - specificity) * (1 - base_rate)
    return true_route / (true_route + false_route)


def analyze(raw):
    native = oe.crosstab(raw, ROLES["condition"], ROLES["test"], expected=True, exact=False)
    overall = outcomes(raw)
    groups = {
        name: outcomes(raw[raw[ROLES["group"]] == code])
        for name, code in (("asymptomatic", 0), ("symptomatic", 1))
    }
    asym, sym = groups["asymptomatic"], groups["symptomatic"]
    summary = {
        **overall,
        "bayes_predictive_value": predictive_value(
            overall["prevalence"], overall["sensitivity"], overall["specificity"]
        ),
    }
    for name, rates in groups.items():
        for key in ("pairs", "prevalence", "sensitivity", "specificity", "positive_predictive_value"):
            summary[f"{name}_{key}"] = rates[key]
    summary["asymptomatic_rates_at_symptomatic_prevalence"] = predictive_value(
        sym["prevalence"], asym["sensitivity"], asym["specificity"]
    )
    summary["asymptomatic_ppv_at_one_percent_prevalence"] = predictive_value(
        0.01, asym["sensitivity"], asym["specificity"]
    )
    cells = ("true_positive", "false_positive", "false_negative", "true_negative")
    checks = {
        "confusion_matrix_partition": sum(overall[c] for c in cells) == len(raw),
        "groups_partition_every_cell": all(asym[c] + sym[c] == overall[c] for c in cells),
        "bayes_matches_positive_denominator": abs(
            summary["bayes_predictive_value"] - overall["positive_predictive_value"]
        )
        < 1e-14,
        "bayes_matches_each_group": all(
            abs(
                predictive_value(r["prevalence"], r["sensitivity"], r["specificity"])
                - r["positive_predictive_value"]
            )
            < 1e-14
            for r in groups.values()
        ),
        "lower_base_rate_lowers_ppv": summary["asymptomatic_ppv_at_one_percent_prevalence"]
        < asym["positive_predictive_value"],
    }
    labels = ["True positive", "False positive", "False negative", "True negative"]
    return (
        summary,
        native,
        [
            {
                **panel(
                    labels,
                    [asym[c] for c in cells],
                    kind="bar",
                    title="Test outcomes by symptom status",
                    xlabel="Antigen result against RT-PCR",
                    ylabel="Paired swabs",
                ),
                "series": [
                    {"label": name.capitalize(), "x": labels, "y": [rates[c] for c in cells]}
                    for name, rates in groups.items()
                ],
            },
            panel(
                ["Asymptomatic", "Symptomatic", f"Asym. at {sym['prevalence']:.1%}"],
                [
                    asym["positive_predictive_value"],
                    sym["positive_predictive_value"],
                    summary["asymptomatic_rates_at_symptomatic_prevalence"],
                ],
                kind="bar",
                title="Positive predictive value",
                xlabel="Group (last bar: asymptomatic rates at symptomatic prevalence)",
                ylabel="Share of antigen positives that are PCR positive",
            ),
        ],
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
            "synthetic": False,
            "data_file": DATA_FILE,
            "data_source": DATA_SOURCE,
            "licence": LICENCE,
            "raw_rows": len(raw),
        },
        "summary": clean(summary),
        "models": {k: pack(v) for k, v in models.items()},
        "tables": {"native": pack(native)},
        "chart_data": {"panels": panels},
        "checks": clean(checks),
    }
    latex = str(table.to_latex(precision=6, caption=f"{TITLE} ({DATA_SOURCE}; {LICENCE})"))
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
