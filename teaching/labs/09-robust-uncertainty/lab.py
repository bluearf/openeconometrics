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


DATA_FILE = "household_spending.xlsx"


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
    models = {
        kind: oe.ols(
            data=data,
            y="spending",
            x=["income_thousands"],
            covariance=kind,
            missing="raise",
            device="cpu",
        )
        for kind in ["nonrobust", "HC1", "HC3"]
    }
    x = torch.column_stack(
        (torch.ones(len(data), dtype=torch.float64), tensor(data.income_thousands))
    )
    y = tensor(data.spending)
    summary = {
        "nobs": len(data),
        "income_min": float(data.income_thousands.min()),
        "income_max": float(data.income_thousands.max()),
    }
    errors = []
    for kind, model in models.items():
        beta, covariance = matrix_ols(x, y, kind)
        errors.append(compare_linear(model, ["Intercept", "income_thousands"], beta, covariance))
        row = coef(model, "income_thousands")
        for key, value in {
            "estimate": row.estimate,
            "se": row.std_error,
            "ci_low": row.ci_low,
            "ci_high": row.ci_high,
        }.items():
            summary[kind + "_" + key] = value
    fitted = models["HC3"]
    residual = y - x @ tensor(
        [coef(fitted, "Intercept").estimate, coef(fitted, "income_thousands").estimate]
    )
    cutoff = float(data.income_thousands.median())
    low = tensor(data.income_thousands) <= cutoff
    summary.update(
        {
            "median_income_thousands": cutoff,
            "low_income_residual_rms": float(residual[low].square().mean().sqrt()),
            "high_income_residual_rms": float(residual[~low].square().mean().sqrt()),
            "max_coefficient_error": max(e[0] for e in errors),
            "max_covariance_error": max(e[1] for e in errors),
        }
    )
    checks = {
        "independent_ols_coefficients": summary["max_coefficient_error"] < 1e-9,
        "independent_three_covariances": summary["max_covariance_error"] < 1e-8,
        "identical_coefficients": max(coef(m, "income_thousands").estimate for m in models.values())
        - min(coef(m, "income_thousands").estimate for m in models.values())
        < 1e-12,
        "identical_sample_positions": all(
            m.sample_positions == list(range(len(data))) for m in models.values()
        ),
        "declared_student_t_df": all(
            m.inference["df_inference"] == len(data) - 2 and m.inference["use_t"]
            for m in models.values()
        ),
    }
    chart_data = {
        "panels": [
            {
                "kind": "scatter",
                "title": "One line, increasing outcome spread",
                "xlabel": "Monthly income (thousands of currency units)",
                "ylabel": "Monthly spending (currency units)",
                "series": [
                    {
                        "label": "Synthetic households",
                        "x": data.income_thousands.tolist(),
                        "y": data.spending.tolist(),
                    }
                ],
            },
            {
                "kind": "interval",
                "title": "Same slope; different covariance conventions",
                "xlabel": "Income slope (currency units per thousand)",
                "ylabel": "Covariance",
                "series": [
                    {
                        "label": "95% intervals",
                        "x": [summary[k + "_estimate"] for k in models],
                        "y": ["Classical", "HC1", "HC3"],
                        "low": [summary[k + "_ci_low"] for k in models],
                        "high": [summary[k + "_ci_high"] for k in models],
                    }
                ],
            },
        ],
        "residuals": [
            {"income": float(a), "residual": float(b)}
            for a, b in zip(data.income_thousands, residual)
        ],
    }
    displays = [
        fitted,
        oe.plot.scatter(
            data=data, x="income_thousands", y="spending", title="Synthetic income and spending"
        ),
    ]
    return complete(
        "09-robust-uncertainty",
        "Same Coefficient, Different Standard Error?",
        data,
        summary,
        models,
        chart_data,
        checks,
        output_dir,
        display_callback,
        displays,
    )


if __name__ == "__main__":
    robust_uncertainty_lab_result = run_lab(display_callback=globals().get("display"))
