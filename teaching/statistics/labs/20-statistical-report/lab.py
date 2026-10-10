"""From a Question to a Statistical Report: run the whole file after importing the supplied workbook."""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import pandas as pd
import torch
import openecon as oe
from openecon.models import ResultBundle

DATA_FILE = "campus_cafe_offer.xlsx"
LAB = "20-statistical-report"
TITLE = "From a Question to a Statistical Report"


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
    data = raw.dropna(subset=["spending"])
    native = oe.ttest(data, "spending", by="offer", missing="raise")
    row = native["test"].loc["unequal_variances"]
    secondary = oe.prtest(raw, "returned", by="offer", positive=1, missing="raise")
    a = data.loc[data.offer == "A", "spending"].tolist()
    b = data.loc[data.offer == "B", "spending"].tolist()
    se = math.sqrt(variance(a) / len(a) + variance(b) / len(b))
    difference = mean(a) - mean(b)
    pvalues = [float(row["p_value"]), float(secondary.attrs["p_value"])]
    order = sorted(range(2), key=pvalues.__getitem__)
    adjusted = [0.0, 0.0]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (2 - rank) * pvalues[i]))
        adjusted[i] = running
    summary = {
        "randomized_customers": len(raw),
        "spending_observed": len(data),
        "spending_missing": len(raw) - len(data),
        "n_A_spending": len(a),
        "n_B_spending": len(b),
        "mean_A_spending": mean(a),
        "mean_B_spending": mean(b),
        "spending_difference_A_minus_B": difference,
        "welch_se": float(row["std_error"]),
        "welch_df": float(row["df"]),
        "welch_t": float(row["statistic"]),
        "primary_p": pvalues[0],
        "primary_ci_low": float(row["ci_low"]),
        "primary_ci_high": float(row["ci_high"]),
        "return_rate_difference_A_minus_B": float(secondary.loc["diff", "proportion"]),
        "secondary_p": pvalues[1],
        "holm_primary_p": adjusted[0],
        "holm_secondary_p": adjusted[1],
    }
    checks = {
        "primary_welch_se": abs(row["std_error"] - se) < 1e-12,
        "primary_t_uses_same_rows": abs(row["statistic"] - difference / se) < 1e-10,
        "sample_accounting": len(a) + len(b) + int(raw.spending.isna().sum()) == len(raw),
        "holm_does_not_reduce_p": all(x >= p for x, p in zip(adjusted, pvalues)),
        "distinct_customer_ids": raw.customer_id.is_unique,
    }
    return (
        summary,
        native,
        [
            panel(
                ["A", "B"],
                [mean(a), mean(b)],
                kind="bar",
                xlabel="Randomized offer",
                ylabel="Observed mean spending (currency)",
            )
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
            "synthetic": True,
            "data_file": DATA_FILE,
            "raw_rows": len(raw),
        },
        "summary": clean(summary),
        "models": {k: pack(v) for k, v in models.items()},
        "tables": {"native": pack(native)},
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
        first = panels[0]
        series = first["series"][0]
        chart_frame = oe.DataFrame({"x": series["x"], "y": series["y"]})
        plotter = getattr(oe.plot, {"hist": "bar"}.get(first["kind"], first["kind"]))
        display_callback(plotter(data=chart_frame, x="x", y="y", title=first["title"]))
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
