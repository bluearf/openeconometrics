"""Original synthetic wage-distribution laboratory; run the complete document."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import openecon as oe


# Original dataset-generation seed (provenance only).
SEED = 2022026
DATA_FILE = 'wage_distributions.xlsx'


def load_data(data_path=None):
    """Read the supplied workbook or its imported OpenEconometrics snapshot."""
    import os
    import pandas as pd

    if data_path is not None:
        return oe.DataFrame(pd.read_excel(Path(data_path).expanduser(), sheet_name="Data", engine="openpyxl"))
    candidates = [Path(DATA_FILE)]
    if globals().get("__file__"):
        candidates.insert(0, Path(__file__).resolve().with_name(DATA_FILE))
    for candidate in candidates:
        if candidate.is_file():
            return oe.DataFrame(pd.read_excel(candidate, sheet_name="Data", engine="openpyxl"))
    workspace_path = os.environ.get("OPENECON_WORKSPACE")
    if workspace_path:
        from openecon.workspace import Workspace
        workspace = Workspace(workspace_path)
        matches = [record for record in workspace.list_datasets()
                   if record["name"] in (DATA_FILE, Path(DATA_FILE).stem)]
        if matches:
            # The workspace lists the most recent import first.
            return oe.DataFrame(workspace.load_frame(matches[0]["id"]))
    raise FileNotFoundError(
        f"Import {DATA_FILE} into OpenEconometrics without renaming it, "
        f"or call run_lab(data_path='/path/to/{DATA_FILE}')."
    )


def _percentile(values, probability):
    position = len(values) * probability
    if position.is_integer():
        return (values[int(position) - 1] + values[int(position)]) / 2
    return values[math.ceil(position) - 1]


def _histogram(values, bins=24):
    tensor = torch.tensor(values, dtype=torch.float64)
    counts, edges = torch.histogram(tensor, bins=bins)
    return {"x": ((edges[:-1] + edges[1:]) / 2).tolist(), "y": counts.tolist()}


def run_lab(output_dir=None, display_callback=None, data_path=None):
    raw = load_data(data_path)
    data = raw.dropna(subset=["hourly_wage"]).copy()
    values = sorted(data.hourly_wage.tolist())
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    data["log_wage"] = [math.log(value) for value in data.hourly_wage]
    native = oe.describe(
        data,
        ["hourly_wage", "log_wage"],
        stats=["n", "mean", "std_dev", "min", "p25", "p50", "p75", "max"],
    )
    median = _percentile(values, 0.5)
    mean_without_max = sum(values[:-1]) / (n - 1)
    top_count = math.ceil(n * 0.01)
    summary = {
        "generated_workers": len(raw),
        "observed_wages": n,
        "missing_wages": len(raw) - n,
        "mean_wage": mean,
        "median_wage": median,
        "sample_sd": math.sqrt(variance),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "iqr": _percentile(values, 0.75) - _percentile(values, 0.25),
        "minimum_wage": values[0],
        "maximum_wage": values[-1],
        "mean_without_largest": mean_without_max,
        "mean_change_remove_largest": mean - mean_without_max,
        "median_without_largest": _percentile(values[:-1], 0.5),
        "mean_log_wage": float(data.log_wage.mean()),
        "geometric_mean_wage": math.exp(float(data.log_wage.mean())),
        "top_one_percent_count": top_count,
        "top_one_percent_wage_share": sum(values[-top_count:]) / sum(values),
    }
    checks = {
        "missing_count_explicit": n == 795 and len(raw) - n == 5,
        "native_mean_matches_direct_sum": abs(float(native.loc["hourly_wage", "mean"]) - mean)
        < 1e-12,
        "native_sd_matches_centered_formula": abs(
            float(native.loc["hourly_wage", "std_dev"]) - math.sqrt(variance)
        )
        < 1e-12,
        "native_percentiles_match_stated_rule": all(
            abs(float(native.loc["hourly_wage", key]) - summary[key]) < 1e-12
            for key in ("p25", "p75")
        )
        and abs(float(native.loc["hourly_wage", "p50"]) - median) < 1e-12,
        "jensen_inequality": summary["geometric_mean_wage"] <= mean,
        "all_observed_wages_positive": values[0] > 0,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    panels = []
    for column, title, label in [
        ("hourly_wage", "Wages on the original scale", "Hypothetical currency/hour"),
        ("log_wage", "The same workers on a log scale", "Natural log of wage"),
    ]:
        histogram = _histogram(data[column].tolist())
        panels.append(
            {
                "kind": "bar",
                "title": title,
                "xlabel": label,
                "ylabel": "Workers per equal-width bin",
                "series": [{"label": "Workers", **histogram}],
            }
        )
        checks[f"{column}_histogram_retains_every_worker"] = sum(histogram["y"]) == n
    result = {
        "metadata": {
            "lab": "02-wage-distributions",
            "title": "A First Look at Wages",
            "seed": SEED,
            "synthetic": True,
            "percentile_rule": "Stata empirical percentile: ceiling np; average adjacent order statistics when np integer",
            "wage_unit": "hypothetical currency/hour",
            "missing_rule": "exclude five missing wages explicitly",
        },
        "summary": summary,
        "models": {},
        "chart_data": {"panels": panels},
        "checks": checks,
    }
    latex = native.to_latex(
        precision=3, caption="Original synthetic wage and log-wage distributions"
    )
    if display_callback is not None:
        display_callback(native)
        display_callback(
            oe.plot.hist(
                data=data, x="hourly_wage", bins=24, title="Original synthetic hourly wages"
            )
        )
        display_callback(
            oe.plot.hist(
                data=data, x="log_wage", bins=24, title="The same observed wages after taking logs"
            )
        )
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "reference.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (target / "table.tex").write_text(str(latex), encoding="utf-8")
        if json.loads((target / "reference.json").read_text()) != result:
            raise AssertionError("Explicit export readback differs.")
    print(
        f"795 observed synthetic wages, 5 missing. Mean {mean:.3f}; median {median:.3f}; geometric mean {summary['geometric_mean_wage']:.3f}."
    )
    return result


if __name__ == "__main__":
    lab_result = run_lab(display_callback=globals().get("display"))
