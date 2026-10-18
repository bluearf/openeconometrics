"""Comparing Ordinal Customer Ratings: run the whole file after importing the supplied workbook."""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import pandas as pd
import torch
import openecon as oe
from openecon.models import ResultBundle

DATA_FILE = "customer_ratings.xlsx"
LAB = "18-rank-comparisons"
TITLE = "Comparing Ordinal Customer Ratings"


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
    native = oe.ranksum(raw, "rating", "service", exact=False, missing="raise")
    a = raw.loc[raw.service == "A", "rating"].tolist()
    b = raw.loc[raw.service == "B", "rating"].tolist()
    u = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
    prob = u / (len(a) * len(b))
    ranks = raw.rating.rank(method="average")
    rank_sum = float(ranks[raw.service == "A"].sum())
    summary = {
        "n_A": len(a),
        "n_B": len(b),
        "U_A": float(native.attrs["u1"]),
        "rank_sum_A": rank_sum,
        "probability_A_exceeds_B_with_half_ties": float(native.attrs["porder"]),
        "z_without_continuity_correction": float(native.attrs["z"]),
        "asymptotic_two_sided_p": float(native.attrs["p_value"]),
        "tied_pair_fraction": sum(x == y for x in a for y in b) / (len(a) * len(b)),
    }
    checks = {
        "native_U_matches_pair_enumeration": abs(float(native.attrs["u1"]) - u) < 1e-12,
        "rank_sum_identity": abs(rank_sum - len(a) * (len(a) + 1) / 2 - u) < 1e-12,
        "native_probability_matches_pair_comparison": abs(float(native.attrs["porder"]) - prob)
        < 1e-12,
        "all_ranks_sum_correctly": abs(float(ranks.sum()) - len(raw) * (len(raw) + 1) / 2) < 1e-12,
    }
    ratings = list(range(1, 8))
    by_service = {
        name: raw.loc[raw.service == name, "rating"].value_counts().reindex(ratings, fill_value=0)
        for name in ("A", "B")
    }
    checks["counts_cover_every_customer"] = sum(int(c.sum()) for c in by_service.values()) == len(raw)
    return (
        summary,
        native,
        [
            {
                **panel(
                    ratings,
                    by_service["A"].tolist(),
                    kind="bar",
                    title="Ratings by Service",
                    xlabel="Rating (ordered 1–7)",
                    ylabel="Customers",
                ),
                "series": [
                    {"label": f"Service {name}", "x": ratings, "y": [int(v) for v in counts]}
                    for name, counts in by_service.items()
                ],
            }
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
