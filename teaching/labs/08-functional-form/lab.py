"""Original log-wage curvature and retransformation teaching example."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import openecon as oe
from openecon.models import ResultBundle


# Original dataset-generation seed (provenance only).
SEED = 8082026
DATA_FILE = 'wage_profiles.xlsx'


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


def run_lab(output_dir=None, display_callback=None, data_path=None):
    data = load_data(data_path)
    linear = oe.ols(
        data=data,
        y="log_wage",
        x=["education", "experience"],
        covariance="HC3",
        missing="raise",
        device="cpu",
    )
    quadratic = oe.ols(
        data=data,
        y="log_wage",
        x=["education", "experience", "experience_squared"],
        covariance="HC3",
        missing="raise",
        device="cpu",
    )
    coefficients = {row.term: row for row in quadratic.coefficients}
    linear_coefficients = {row.term: row for row in linear.coefficients}
    x = torch.tensor(
        [
            [1, *row]
            for row in data[["education", "experience", "experience_squared"]].itertuples(
                index=False, name=None
            )
        ],
        dtype=torch.float64,
    )
    y = torch.tensor(data.log_wage.tolist(), dtype=torch.float64)
    beta = torch.linalg.lstsq(x, y).solution
    bread = torch.linalg.inv(x.T @ x)
    residual = y - x @ beta
    leverage = ((x @ bread) * x).sum(1)
    scores = x * (residual / (1 - leverage))[:, None]
    covariance = bread @ scores.T @ scores @ bread
    b1, b2 = coefficients["experience"].estimate, coefficients["experience_squared"].estimate
    effects = []
    for years in (5, 20, 35):
        derivative = quadratic.lincom({"experience": 1, "experience_squared": 2 * years})
        exact_change = b1 + b2 * (2 * years + 1)
        effects.append(
            {
                "experience": years,
                "log_derivative": derivative["estimate"],
                "derivative_se": derivative["std_error"],
                "derivative_ci_low": derivative["ci_low"],
                "derivative_ci_high": derivative["ci_high"],
                "one_year_log_change": exact_change,
                "one_year_exact_percent": 100 * math.expm1(exact_change),
            }
        )
    grid = list(range(41))
    qprofile = [
        coefficients["Intercept"].estimate
        + 14 * coefficients["education"].estimate
        + b1 * years
        + b2 * years**2
        for years in grid
    ]
    lprofile = [
        linear_coefficients["Intercept"].estimate
        + 14 * linear_coefficients["education"].estimate
        + linear_coefficients["experience"].estimate * years
        for years in grid
    ]
    true_geometric_20 = math.exp(1.6 + 0.07 * 14 + 0.05 * 20 - 0.0008 * 20**2)
    true_mean_20 = true_geometric_20 * math.exp((0.16 + 0.004 * 20) ** 2 / 2)
    derivative_error = 0.0
    for row in effects:
        years = row["experience"]
        vector = torch.tensor([0, 0, 1, 2 * years], dtype=torch.float64)
        derivative_error = max(
            derivative_error, abs(row["derivative_se"] ** 2 - float(vector @ covariance @ vector))
        )
    checks = {
        "same500_workers": quadratic.nobs == linear.nobs == 500
        and quadratic.sample_positions == linear.sample_positions,
        "independent_ols": float(
            (
                beta
                - torch.tensor(
                    [row.estimate for row in quadratic.coefficients], dtype=torch.float64
                )
            )
            .abs()
            .max()
        )
        < 1e-10,
        "independent_hc3": float(
            (covariance - torch.tensor(quadratic.covariance_matrix, dtype=torch.float64))
            .abs()
            .max()
        )
        < 1e-10,
        "derivative_contrast_covariance": derivative_error < 1e-10,
        "exact_one_year_changes": all(
            abs(
                row["one_year_log_change"]
                - (
                    b1 * (row["experience"] + 1)
                    + b2 * (row["experience"] + 1) ** 2
                    - b1 * row["experience"]
                    - b2 * row["experience"] ** 2
                )
            )
            < 1e-12
            for row in effects
        ),
        "retransformation_mean_exceeds_geometric": true_mean_20 > true_geometric_20,
        "full_result_roundtrips": all(
            ResultBundle.model_validate_json(m.model_dump_json()).model_dump(mode="json")
            == m.model_dump(mode="json")
            for m in (linear, quadratic)
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    education = coefficients["education"]
    summary = {
        "nobs": 500,
        "education_log_coefficient": education.estimate,
        "education_approx_percent": 100 * education.estimate,
        "education_exact_percent": 100 * math.expm1(education.estimate),
        "education_exact_percent_ci_low": 100 * math.expm1(education.ci_low),
        "education_exact_percent_ci_high": 100 * math.expm1(education.ci_high),
        "experience_coefficient": b1,
        "experience_squared_coefficient": b2,
        "turning_point": -b1 / (2 * b2),
        "linear_experience_coefficient": linear_coefficients["experience"].estimate,
        "linear_r_squared": linear.metrics["r_squared"],
        "quadratic_r_squared": quadratic.metrics["r_squared"],
        "known_dgp_geometric_wage_at_education14_experience20": true_geometric_20,
        "known_dgp_mean_wage_at_education14_experience20": true_mean_20,
        "known_dgp_retransformation_factor_at20": true_mean_20 / true_geometric_20,
    }
    for row in effects:
        for key, value in row.items():
            if key != "experience":
                summary[f"experience{row['experience']}_{key}"] = value
    panels = [
        {
            "kind": "line",
            "title": "Curvature in predicted log wages, education fixed at 14",
            "xlabel": "Experience (years)",
            "ylabel": "Fitted log hourly wage",
            "series": [
                {"label": "Linear experience", "x": grid, "y": lprofile},
                {"label": "Quadratic experience", "x": grid, "y": qprofile},
            ],
        },
        {
            "kind": "line",
            "title": "The marginal association changes along the curve",
            "xlabel": "Experience (years)",
            "ylabel": "100 × log-wage derivative",
            "series": [
                {
                    "label": "Approximate percentage change per extra year",
                    "x": grid,
                    "y": [100 * (b1 + 2 * b2 * years) for years in grid],
                }
            ],
        },
    ]
    result = {
        "metadata": {
            "lab": "08-functional-form",
            "title": "Modeling Percentage Changes and Curvature",
            "seed": SEED,
            "synthetic": True,
            "covariance": "HC3",
            "outcome": "natural log hourly wage",
            "inference_df": 496,
        },
        "summary": summary,
        "models": {
            "linear_log_wage": linear.model_dump(mode="json"),
            "quadratic_log_wage": quadratic.model_dump(mode="json"),
        },
        "chart_data": {"panels": panels, "local_effects": effects},
        "checks": checks,
    }
    table = oe.regression_table(
        {"Linear experience": linear, "Quadratic experience": quadratic},
        precision=5,
        stars=False,
        caption="Original synthetic log-wage models: the same 500 workers",
    )
    if display_callback is not None:
        display_callback(quadratic)
        display_callback(oe.DataFrame(effects))
        display_callback(
            oe.plot.line(
                data={"experience": grid, "log_wage": qprofile},
                x="experience",
                y="log_wage",
                title="Predicted synthetic log wage at 14 education years",
            )
        )
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "reference.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (target / "table.tex").write_text(str(table), encoding="utf-8")
        if json.loads((target / "reference.json").read_text()) != result:
            raise AssertionError("Explicit export readback differs.")
    print(
        f"Synthetic education association: {summary['education_exact_percent']:.3f}% per year; experience curve turning point {summary['turning_point']:.3f} years."
    )
    print(
        f"Known DGP at education 14 / experience 20: geometric wage {true_geometric_20:.3f}, arithmetic mean {true_mean_20:.3f}."
    )
    return result


if __name__ == "__main__":
    lab_result = run_lab(display_callback=globals().get("display"))
