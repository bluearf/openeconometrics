"""Weak Relevance and IV Sensitivity: run the whole file after importing the supplied workbook."""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import pandas as pd
import torch
import openecon as oe
from openecon.models import ResultBundle

DATA_FILE = "weak_instruments.xlsx"
LAB = "16-weak-instruments"
TITLE = "Weak Relevance and IV Sensitivity"


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


def coef(model, name):
    return next(r for r in model.coefficients if r.term == name)


def fit(data, predictors, y="y", covariance="HC3", **kwargs):
    return oe.ols(
        data=oe.DataFrame(data),
        y=y,
        x=predictors,
        covariance=covariance,
        missing="raise",
        device="cpu",
        **kwargs,
    )


def valid(models):
    return {
        "complete_model_json_roundtrip": all(
            ResultBundle.model_validate_json(m.model_dump_json()).model_dump(mode="json")
            == m.model_dump(mode="json")
            for m in models.values()
        ),
        "unique_retained_positions": all(
            len(m.sample_positions) == m.nobs == len(set(m.sample_positions))
            for m in models.values()
        ),
        "covariance_positive_semidefinite": all(
            float(
                torch.linalg.eigvalsh(torch.tensor(m.covariance_matrix, dtype=torch.float64)).min()
            )
            >= -1e-8
            for m in models.values()
        ),
    }


def analyze(raw):
    strong = oe.ivregress(
        data=raw,
        y="y",
        x=["control"],
        endog=["x"],
        instruments=["z"],
        covariance="unadjusted",
        small=True,
        missing="raise",
    )
    weakdata = raw.assign(x=raw.weak_x, y=raw.y - 1.2 * raw.x + 1.2 * raw.weak_x)
    weak = oe.ivregress(
        data=weakdata,
        y="y",
        x=["control"],
        endog=["x"],
        instruments=["z"],
        covariance="unadjusted",
        small=True,
        missing="raise",
    )
    fs = fit(raw, ["control", "z"], y="x", covariance="nonrobust")
    fw = fit(weakdata, ["control", "z"], y="x", covariance="nonrobust")
    models = {
        "strong_first_stage": fs,
        "weak_first_stage": fw,
        "strong_iv": strong,
        "weak_iv": weak,
    }
    summary = {
        "strong_first_stage_F": coef(fs, "z").statistic ** 2,
        "weak_first_stage_F": coef(fw, "z").statistic ** 2,
        "strong_iv_x": coef(strong, "x").estimate,
        "weak_iv_x": coef(weak, "x").estimate,
        "strong_iv_se": coef(strong, "x").std_error,
        "weak_iv_se": coef(weak, "x").std_error,
    }
    checks = {
        "weak_instrument_relevance_lower": summary["weak_first_stage_F"]
        < summary["strong_first_stage_F"],
        "same_structural_effect": max(abs((weakdata.y - 1.2 * weakdata.x) - (raw.y - 1.2 * raw.x)))
        < 1e-12,
    }
    checks.update(valid(models))
    return (
        summary,
        models["weak_iv"],
        [
            panel(
                raw.x.tolist(),
                raw.y.tolist(),
                kind="scatter",
                xlabel="X (predictor units)",
                ylabel="Y (outcome units)",
            )
        ],
        checks,
        models,
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
