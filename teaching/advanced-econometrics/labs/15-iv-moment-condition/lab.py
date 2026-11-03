"""Instrumental Variables and the Projection Formula: run the whole file after importing the supplied workbook."""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import pandas as pd
import torch
import openecon as oe
from openecon.models import ResultBundle

DATA_FILE = "maimonides_classes.xlsx"
LAB = "15-iv-moment-condition"
TITLE = "Instrumental Variables and the Projection Formula"
DATA_SOURCE = "Angrist and Lavy 1999 replication data, Harvard Dataverse"
LICENCE = "CC0 1.0"
# Column roles; identical to "roles" in the lab's dataset.json.
ROLES = {
    "y": "avgverb",
    "x": "classize",
    "z": "func1",
    "controls": ["tipuach", "c_size"],
    "enrolment": "c_size",
    "cluster": "schlcode",
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
    y, x, z, controls, cluster = (ROLES[k] for k in ("y", "x", "z", "controls", "cluster"))
    without_enrolment = [c for c in controls if c != ROLES["enrolment"]]
    data = raw[[cluster, *controls, x, z, y]]
    ols = fit(data, [*controls, x], y=y, covariance="nonrobust")
    first = fit(data, [*controls, z], y=x, covariance="nonrobust")
    first_clustered = fit(data, [*controls, z], y=x, covariance="cluster", cluster=cluster)

    def iv(exogenous, **options):
        return oe.ivregress(
            data=data, y=y, x=exogenous, endog=[x], instruments=[z], small=True, missing="raise", **options
        )

    two_stage = iv(controls, covariance="unadjusted")
    two_stage_clustered = iv(controls, cluster=cluster)
    pd_only = iv(without_enrolment, cluster=cluster)
    X = torch.tensor(data[controls + [x]].values, dtype=torch.float64)
    Z = torch.tensor(data[controls + [z]].values, dtype=torch.float64)
    ones = torch.ones(len(data), 1, dtype=torch.float64)
    X, Z = torch.cat([ones, X], dim=1), torch.cat([ones, Z], dim=1)
    Y = torch.tensor(data[y].tolist(), dtype=torch.float64)
    X_hat = Z @ torch.linalg.lstsq(Z, X).solution
    b = torch.linalg.solve(X.T @ X_hat, X_hat.T @ Y)
    enrolment = data[ROLES["enrolment"]]
    rule = enrolment / ((enrolment - 1) // 40 + 1)
    models = {
        "ordinary": ols,
        "first_stage": first,
        "first_stage_clustered": first_clustered,
        "two_stage_least_squares": two_stage,
        "two_stage_least_squares_clustered": two_stage_clustered,
        "percent_disadvantaged_only": pd_only,
    }
    summary = {
        "classes": len(data),
        "schools": int(data[cluster].nunique()),
        "ols_class_size": coef(ols, x).estimate,
        "first_stage_instrument_slope": coef(first, z).estimate,
        "first_stage_cluster_F": coef(first_clustered, z).statistic ** 2,
        "iv_class_size": coef(two_stage, x).estimate,
        "iv_class_size_classical_se": coef(two_stage, x).std_error,
        "iv_class_size_cluster_se": coef(two_stage_clustered, x).std_error,
        "iv_class_size_cluster_ci_low": coef(two_stage_clustered, x).ci_low,
        "iv_class_size_cluster_ci_high": coef(two_stage_clustered, x).ci_high,
        "iv_class_size_percent_disadvantaged_only": coef(pd_only, x).estimate,
        "iv_class_size_percent_disadvantaged_only_cluster_se": coef(pd_only, x).std_error,
        "matrix_iv_class_size": float(b[-1]),
    }
    checks = {
        "projection_formula_matches_iv": abs(float(b[-1]) - coef(two_stage, x).estimate) < 1e-9,
        "clustering_changes_only_the_covariance": abs(
            coef(two_stage_clustered, x).estimate - coef(two_stage, x).estimate
        )
        < 1e-10,
        "instrument_follows_maimonides_rule": float((data[z] - rule).abs().max()) < 1e-12,
        "same_complete_sample": all(m.nobs == len(data) for m in models.values()),
    }
    checks.update(valid(models))
    return (
        summary,
        models["two_stage_least_squares"],
        [
            panel(
                data[z].tolist(),
                data[x].tolist(),
                kind="scatter",
                title="First stage: class size against Maimonides' rule",
                xlabel="Predicted class size, func1 (pupils)",
                ylabel="Actual class size (pupils)",
            ),
            panel(
                ["OLS + c_size", "2SLS", "2SLS + c_size"],
                [coef(ols, x).estimate, coef(pd_only, x).estimate, coef(two_stage, x).estimate],
                kind="bar",
                title="Class-size slope by estimator",
                xlabel="Estimator (each controls for tipuach)",
                ylabel="Reading points per pupil",
            ),
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
            "synthetic": False,
            "data_file": DATA_FILE,
            "data_source": DATA_SOURCE,
            "licence": LICENCE,
            "raw_rows": len(raw),
        },
        "summary": clean(summary),
        "models": {k: pack(v) for k, v in models.items()},
        "tables": {} if isinstance(native, ResultBundle) else {"native": pack(native)},
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
