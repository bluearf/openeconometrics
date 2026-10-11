"""Why Do Averages Vary Less Than Observations?: run the whole file after importing the supplied workbook."""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import pandas as pd
import torch
import openecon as oe
from openecon.models import ResultBundle

DATA_FILE = "spending_population.xlsx"
LAB = "08-sampling-means"
TITLE = "Why Do Averages Vary Less Than Observations?"


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
    population = torch.tensor(raw.spending.tolist(), dtype=torch.float64)
    g = torch.Generator().manual_seed(991008)
    means = {
        n: population[torch.randint(len(population), (1600, n), generator=g)].mean(1)
        for n in (5, 40)
    }
    mu = float(population.mean())
    sigma = float(population.std(correction=0))
    native = oe.DataFrame(
        {
            "sample_size": [5, 40],
            "replications": [1600, 1600],
            "mean_of_means": [float(means[n].mean()) for n in (5, 40)],
            "empirical_sd": [float(means[n].std()) for n in (5, 40)],
            "theoretical_se": [sigma / math.sqrt(n) for n in (5, 40)],
        }
    )
    summary = {
        "population_size": len(raw),
        "population_mean": mu,
        "population_sd_denominator_N": sigma,
        "replications": 1600,
        "mean_at_n5": float(means[5].mean()),
        "mean_at_n40": float(means[40].mean()),
        "empirical_se_n5": float(means[5].std()),
        "empirical_se_n40": float(means[40].std()),
        "theoretical_se_n5": sigma / math.sqrt(5),
        "theoretical_se_n40": sigma / math.sqrt(40),
    }
    checks = {
        "means_within_population_support": all(
            float(v.min()) >= float(population.min()) and float(v.max()) <= float(population.max())
            for v in means.values()
        ),
        "larger_samples_have_less_dispersion": float(means[40].std()) < float(means[5].std()),
        "mean_close_relative_to_monte_carlo_error": all(
            abs(float(v.mean()) - mu) < 5 * sigma / math.sqrt(n * len(v)) for n, v in means.items()
        ),
    }
    charts = [
        panel(
            **histogram(means[n].tolist()),
            kind="hist",
            title=f"Sampling means: n={n}",
            xlabel="Sample mean spending",
            ylabel="Replications",
        )
        for n in (5, 40)
    ]
    return summary, native, charts, checks, {}


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
