"""Original synthetic group-specific wage profiles and interaction contrasts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import openecon as oe
from openecon.models import ResultBundle


# Original dataset-generation seed (provenance only).
SEED = 7072026
DATA_FILE = 'wage_interactions.xlsx'


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
    predictors = ["experience_centered", "sector", "sector_experience"]
    full = oe.ols(
        data=data, y="hourly_wage", x=predictors, covariance="HC3", missing="raise", device="cpu"
    )
    additive = oe.ols(
        data=data,
        y="hourly_wage",
        x=["experience_centered", "sector"],
        covariance="HC3",
        missing="raise",
        device="cpu",
    )
    raw = oe.ols(
        data=data,
        y="hourly_wage",
        x=["experience", "sector", "sector_experience_raw"],
        covariance="HC3",
        missing="raise",
        device="cpu",
    )
    coefficients = {row.term: row for row in full.coefficients}
    raw_coefficients = {row.term: row for row in raw.coefficients}
    x = torch.tensor(
        [[1, *row] for row in data[predictors].itertuples(index=False, name=None)],
        dtype=torch.float64,
    )
    y = torch.tensor(data.hourly_wage.tolist(), dtype=torch.float64)
    beta = torch.linalg.lstsq(x, y).solution
    bread = torch.linalg.inv(x.T @ x)
    residual = y - x @ beta
    leverage = ((x @ bread) * x).sum(1)
    scores = x * (residual / (1 - leverage))[:, None]
    covariance = bread @ scores.T @ scores @ bread
    gaps = []
    for years in (10, 15, 25):
        contrast = full.lincom({"sector": 1, "sector_experience": years - 15})
        gaps.append({"experience": years, **contrast})
    slope_a = coefficients["experience_centered"].estimate
    slope_b = full.lincom({"experience_centered": 1, "sector_experience": 1})
    grid = list(range(5, 31))
    profile_a = [coefficients["Intercept"].estimate + slope_a * (value - 15) for value in grid]
    profile_b = [
        value
        + coefficients["sector"].estimate
        + coefficients["sector_experience"].estimate * (years - 15)
        for years, value in zip(grid, profile_a)
    ]
    manual_variances = []
    for row in gaps:
        vector = torch.tensor([0, 0, 1, row["experience"] - 15], dtype=torch.float64)
        manual_variances.append(float(vector @ covariance @ vector))
    checks = {
        "same480_workers_all_models": all(
            m.nobs == 480 and m.dropped_rows == 0 and m.sample_positions == full.sample_positions
            for m in (full, additive, raw)
        ),
        "independent_ols": float(
            (beta - torch.tensor([row.estimate for row in full.coefficients], dtype=torch.float64))
            .abs()
            .max()
        )
        < 1e-10,
        "independent_hc3": float(
            (covariance - torch.tensor(full.covariance_matrix, dtype=torch.float64)).abs().max()
        )
        < 1e-10,
        "contrast_covariance_includes_cross_terms": all(
            abs(row["std_error"] ** 2 - variance) < 1e-10
            for row, variance in zip(gaps, manual_variances)
        ),
        "centering_changes_reference_not_profiles": abs(
            coefficients["sector"].estimate
            - (
                raw_coefficients["sector"].estimate
                + 15 * raw_coefficients["sector_experience_raw"].estimate
            )
        )
        < 1e-10,
        "group_specific_slopes": abs(
            slope_b["estimate"] - slope_a - coefficients["sector_experience"].estimate
        )
        < 1e-12,
        "full_result_roundtrips": all(
            ResultBundle.model_validate_json(m.model_dump_json()).model_dump(mode="json")
            == m.model_dump(mode="json")
            for m in (full, additive, raw)
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    summary = {
        "nobs": 480,
        "group_a_workers": 240,
        "group_b_workers": 240,
        "slope_a": slope_a,
        "slope_b": slope_b["estimate"],
        "slope_difference": coefficients["sector_experience"].estimate,
        "slope_difference_se": coefficients["sector_experience"].std_error,
        "slope_difference_ci_low": coefficients["sector_experience"].ci_low,
        "slope_difference_ci_high": coefficients["sector_experience"].ci_high,
        "gap_at_15": coefficients["sector"].estimate,
        "raw_group_coefficient_at_zero": raw_coefficients["sector"].estimate,
        "additive_constant_gap": next(
            row.estimate for row in additive.coefficients if row.term == "sector"
        ),
    }
    for row in gaps:
        for key in ("estimate", "std_error", "ci_low", "ci_high"):
            summary[f"gap_at_{row['experience']}_{key}"] = row[key]
    panels = [
        {
            "kind": "line",
            "title": "Fitted group-specific wage profiles",
            "xlabel": "Experience (years)",
            "ylabel": "Hourly wage (currency/hour)",
            "series": [
                {"label": "Group A", "x": grid, "y": profile_a},
                {"label": "Group B", "x": grid, "y": profile_b},
            ],
        },
        {
            "kind": "interval",
            "title": "The conditional group gap changes with experience",
            "xlabel": "Experience (years)",
            "ylabel": "B minus A wage gap (currency/hour)",
            "series": [
                {
                    "label": "HC3 gap and 95% interval",
                    "x": [row["experience"] for row in gaps],
                    "y": [row["estimate"] for row in gaps],
                    "low": [row["ci_low"] for row in gaps],
                    "high": [row["ci_high"] for row in gaps],
                }
            ],
        },
    ]
    result = {
        "metadata": {
            "lab": "07-interactions",
            "title": "Do Groups Have Different Wage Profiles?",
            "seed": SEED,
            "synthetic": True,
            "covariance": "HC3",
            "reference_experience": 15,
            "inference_df": 476,
        },
        "summary": summary,
        "models": {
            "interaction": full.model_dump(mode="json"),
            "additive": additive.model_dump(mode="json"),
            "uncentered": raw.model_dump(mode="json"),
        },
        "chart_data": {"panels": panels, "gaps": gaps},
        "checks": checks,
    }
    table = oe.regression_table(
        {"Additive": additive, "Group-specific slopes": full},
        precision=4,
        stars=False,
        caption="Original synthetic group wage profiles; experience centered at 15 years",
    )
    if display_callback is not None:
        display_callback(full)
        display_callback(oe.DataFrame(gaps))
        display_callback(
            oe.plot.line(
                data={"experience": grid, "gap": [b - a for a, b in zip(profile_a, profile_b)]},
                x="experience",
                y="gap",
                title="Fitted group B minus A hourly wage gap",
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
        f"Synthetic wage slopes: A {slope_a:.4f}, B {slope_b['estimate']:.4f}; gap at 15 years {summary['gap_at_15']:.4f}."
    )
    return result


if __name__ == "__main__":
    lab_result = run_lab(display_callback=globals().get("display"))
