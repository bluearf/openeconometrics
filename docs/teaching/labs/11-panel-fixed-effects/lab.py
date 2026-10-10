"""Original synthetic teaching lab; run the complete file in a new Python document.

Import the supplied named workbook or keep it beside this source file.
run_lab(output_dir=None, display_callback=None, data_path=None) writes only on explicit request.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import openecon as oe
from openecon.models import ResultBundle


def tensor(values):
    return torch.tensor(list(values), dtype=torch.float64)


def coef(model, name):
    return next(row for row in model.coefficients if row.term == name)


def matrix_ols(x, y, kind="HC1", groups=None):
    beta = torch.linalg.lstsq(x, y).solution
    residual = y - x @ beta
    bread = torch.linalg.inv(x.T @ x)
    n, k = x.shape
    if kind == "nonrobust":
        covariance = float(residual.square().sum()) / (n - k) * bread
    else:
        leverage = ((x @ bread) * x).sum(1)
        correction = 1.0
        if kind == "HC1":
            correction = n / (n - k)
        elif kind == "HC2":
            residual = residual / torch.sqrt(1 - leverage)
        elif kind == "HC3":
            residual = residual / (1 - leverage)
        scores = x * residual[:, None]
        if groups is not None:
            labels = sorted(set(groups))
            scores = torch.stack([scores[tensor(groups) == group].sum(0) for group in labels])
            correction = len(labels) / (len(labels) - 1) * (n - 1) / (n - k)
        covariance = correction * bread @ (scores.T @ scores) @ bread
    return beta, covariance


def compare_linear(model, names, beta, covariance):
    order = [names.index(row.term) for row in model.coefficients]
    error = max(
        abs(row.estimate - float(beta[names.index(row.term)])) for row in model.coefficients
    )
    expected = covariance[order][:, order]
    cov_error = float((tensor_matrix(model.covariance_matrix) - expected).abs().max())
    return error, cov_error


def tensor_matrix(values):
    return torch.tensor(values, dtype=torch.float64)


def complete(
    slug, title, data, summary, models, chart_data, checks, output_dir, display_callback, displays
):
    checks = dict(checks)
    checks["full_result_json_roundtrip"] = all(
        ResultBundle.model_validate_json(model.model_dump_json()).model_dump(mode="json")
        == model.model_dump(mode="json")
        for model in models.values()
    )
    checks["all_models_keep_sample"] = all(
        model.nobs == len(data) and model.dropped_rows == 0 for model in models.values()
    )
    if not all(checks.values()):
        raise AssertionError(f"{slug} verification failed: {checks}")
    records = data.to_dict(orient="records")
    result = {
        "metadata": {
            "lab": slug,
            "title": title,
            "synthetic": True,
            "source": "Original synthetic OpenEconometrics teaching data; not real observations.",
            "data_file": DATA_FILE,
            "openecon_version": oe.__version__,
            "torch_version": str(torch.__version__),
            "dtype": "float64",
            "missing": "raise",
            "weights": None,
            "analysis_rows": len(data),
            "data_sha256": hashlib.sha256(
                json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
        },
        "summary": summary,
        "models": {name: model.model_dump(mode="json") for name, model in models.items()},
        "chart_data": chart_data,
        "checks": checks,
    }
    table = oe.regression_table(
        models,
        caption=title + ": original synthetic data",
        label="tab:" + slug,
        stars=False,
        precision=4,
        notes="Original synthetic observations. Inference conventions and exact estimation samples are retained in the model notes; no empirical policy claim.",
    )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "reference.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (destination / "table.tex").write_text(str(table) + "\n", encoding="utf-8")
        if json.loads((destination / "reference.json").read_text(encoding="utf-8")) != result:
            raise AssertionError("Saved complete-result readback differs.")
    if display_callback is not None:
        for item in displays:
            display_callback(item)
        display_callback(table)
    else:
        for model in models.values():
            print(model.summary())
    print(f"{title}: {len(data)} original synthetic observations.")
    return result


DATA_FILE = "firm_panel.xlsx"
FIRMS = 60
PERIODS = 6


def load_data(data_path=None):
    """Read the supplied workbook or its imported OpenEconometrics snapshot."""
    import os
    import pandas as pd

    if data_path is not None:
        return oe.DataFrame(
            pd.read_excel(Path(data_path).expanduser(), sheet_name="Data", engine="openpyxl")
        )
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
        matches = [
            record
            for record in workspace.list_datasets()
            if record["name"] in (DATA_FILE, Path(DATA_FILE).stem)
        ]
        if matches:
            # The workspace lists the most recent import first.
            return oe.DataFrame(workspace.load_frame(matches[0]["id"]))
    raise FileNotFoundError(
        f"Import {DATA_FILE} into OpenEconometrics without renaming it, "
        f"or call run_lab(data_path='/path/to/{DATA_FILE}')."
    )


def run_lab(output_dir=None, display_callback=None, data_path=None):
    data = load_data(data_path)
    pooled = oe.ols(
        data=data,
        y="productivity",
        x=["investment", "trend"],
        covariance="cluster",
        cluster="firm",
        missing="raise",
        device="cpu",
    )
    fe = oe.xtreg(
        data=data,
        y="productivity",
        x=["investment", "trend"],
        panel="firm",
        time="year",
        model="fe",
        covariance="cluster",
        cluster="firm",
        missing="raise",
    )
    x = torch.column_stack((tensor(data.investment), tensor(data.trend)))
    y = tensor(data.productivity)
    xw = x - x.reshape(FIRMS, PERIODS, 2).mean(1).repeat_interleave(PERIODS, 0)
    yw = y - y.reshape(FIRMS, PERIODS).mean(1).repeat_interleave(PERIODS)
    slopes = torch.linalg.lstsq(xw, yw).solution
    residual = yw - xw @ slopes
    bread = torch.linalg.inv(xw.T @ xw)
    scores = (xw * residual[:, None]).reshape(FIRMS, PERIODS, 2).sum(1)
    correction = FIRMS / (FIRMS - 1) * (len(data) - 1) / (len(data) - 3)
    slope_covariance = correction * bread @ (scores.T @ scores) @ bread
    intercept = y.mean() - x.mean(0) @ slopes
    beta = torch.cat((intercept[None], slopes))
    mapping = torch.cat((-x.mean(0)[None, :], torch.eye(2, dtype=torch.float64)))
    covariance = mapping @ slope_covariance @ mapping.T
    b_error, v_error = compare_linear(fe, ["Intercept", "investment", "trend"], beta, covariance)
    px = torch.column_stack((torch.ones(len(data), dtype=torch.float64), x))
    pb, pv = matrix_ols(px, y, groups=data.firm.tolist())
    pb_error, pv_error = compare_linear(pooled, ["Intercept", "investment", "trend"], pb, pv)
    p, f = coef(pooled, "investment"), coef(fe, "investment")
    summary = {
        "nobs": len(data),
        "firms": FIRMS,
        "periods": PERIODS,
        "true_investment_slope": 2.0,
        "pooled_investment": p.estimate,
        "pooled_se": p.std_error,
        "pooled_ci_low": p.ci_low,
        "pooled_ci_high": p.ci_high,
        "fe_investment": f.estimate,
        "fe_se": f.std_error,
        "fe_ci_low": f.ci_low,
        "fe_ci_high": f.ci_high,
        "fe_trend": coef(fe, "trend").estimate,
        "inference_df": fe.inference["df_inference"],
        "within_r_squared": fe.metrics["r_squared_within"],
        "max_coefficient_error": max(b_error, pb_error),
        "max_covariance_error": max(v_error, pv_error),
    }
    checks = {
        "independent_pooled_and_within_coefficients": max(b_error, pb_error) < 1e-9,
        "independent_cluster_covariances": max(v_error, pv_error) < 1e-8,
        "within_means_zero": float(xw.reshape(FIRMS, PERIODS, 2).mean(1).abs().max()) < 1e-12,
        "enough_independent_clusters": fe.inference["df_inference"] == FIRMS - 1,
        "same_sample_positions": fe.sample_positions
        == pooled.sample_positions
        == list(range(len(data))),
    }
    chart_data = {
        "figure": {
            "kind": "interval",
            "title": "Between-firm differences versus within-firm changes",
            "xlabel": "Productivity points per investment unit",
            "ylabel": "Model",
            "series": [
                {
                    "label": "95% firm-cluster intervals",
                    "x": [p.estimate, f.estimate],
                    "y": ["Pooled + trend", "Firm FE + trend"],
                    "low": [p.ci_low, f.ci_low],
                    "high": [p.ci_high, f.ci_high],
                }
            ],
        },
        "within_scatter": [
            {"investment_within": float(a), "productivity_within": float(b)}
            for a, b in zip(xw[:, 0], yw)
        ],
    }
    displays = [
        pooled,
        fe,
        oe.plot.scatter(
            data=oe.DataFrame(chart_data["within_scatter"]),
            x="investment_within",
            y="productivity_within",
            title="Original within-firm deviations (trend still present)",
        ),
    ]
    return complete(
        "11-panel-fixed-effects",
        "Comparing Firms to Themselves",
        data,
        summary,
        {"Pooled, cluster firm": pooled, "Firm FE, cluster firm": fe},
        chart_data,
        checks,
        output_dir,
        display_callback,
        displays,
    )


if __name__ == "__main__":
    panel_fixed_effects_lab_result = run_lab(display_callback=globals().get("display"))
