"""Lab 04: analyze the prepared original synthetic class-size workbook.

Import class_size.xlsx into OpenEconometrics without renaming it, then run
this complete Python document. Ordinary Python reads the sibling workbook;
run_lab(data_path="/path/to/class_size.xlsx") selects an explicit file.
The default run writes no files. Schools are simulated, not actual pupils.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

import openecon as oe
from openecon.models import ResultBundle


# Original dataset-generation seed (provenance only).
SEED = 4204
SCHOOLS = 240
TRUE_SLOPE = -1.8
DATA_FILE = 'class_size.xlsx'


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


def manual_ols_hc3(data):
    """Independently calculate centered scalar OLS and its HC3 sandwich.

    This reference uses only the displayed algebra and Torch arithmetic. It
    never calls a fitted model's coefficient, residual or covariance methods.
    """
    x = torch.tensor(data["class_size"].tolist(), dtype=torch.float64)
    y = torch.tensor(data["test_score"].tolist(), dtype=torch.float64)
    centered = x - x.mean()
    slope = (centered * (y - y.mean())).sum() / centered.square().sum()
    intercept = y.mean() - slope * x.mean()
    design = torch.stack((torch.ones_like(x), x), dim=1)
    bread = torch.linalg.inv(design.T @ design)
    residual = y - intercept - slope * x
    leverage = ((design @ bread) * design).sum(dim=1)
    adjusted_squared = (residual / (1.0 - leverage)).square()
    covariance = bread @ (design.T @ (design * adjusted_squared[:, None])) @ bread
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "covariance": covariance.tolist(),
        "residuals": residual.tolist(),
        "max_leverage": float(leverage.max()),
    }


def run_lab(output_dir=None, display_callback=None, data_path=None):
    """Fit, explain and verify the lab; optionally save complete results."""
    data = load_data(data_path)
    robust = oe.ols(
        data=data,
        y="test_score",
        x=["class_size"],
        covariance="HC3",
        missing="raise",
        device="cpu",
    )
    classical = oe.ols(
        data=data,
        y="test_score",
        x=["class_size"],
        covariance="nonrobust",
        missing="raise",
        device="cpu",
    )
    coefficients = {term.term: term for term in robust.coefficients}
    slope = coefficients["class_size"]
    intercept = coefficients["Intercept"]
    classical_slope = next(c for c in classical.coefficients if c.term == "class_size")
    manual = manual_ols_hc3(data)
    covariance_error = float(
        (
            torch.tensor(robust.covariance_matrix, dtype=torch.float64)
            - torch.tensor(manual["covariance"], dtype=torch.float64)
        ).abs().max()
    )
    coefficient_error = max(
        abs(slope.estimate - manual["slope"]),
        abs(intercept.estimate - manual["intercept"]),
    )
    restored = ResultBundle.model_validate_json(robust.model_dump_json())
    residuals = manual["residuals"]
    residual_mean = sum(residuals) / SCHOOLS
    covariance_matrix = torch.tensor(robust.covariance_matrix, dtype=torch.float64)
    se_error = max(
        abs(term.std_error - math.sqrt(float(covariance_matrix[i, i])))
        for i, term in enumerate(robust.coefficients)
    )
    checks = {
        "all_rows_retained": robust.nobs == SCHOOLS and robust.dropped_rows == 0,
        "same_sample_for_covariance_comparison": robust.sample_positions
        == classical.sample_positions,
        "manual_coefficient_max_abs_error": coefficient_error,
        "manual_hc3_covariance_max_abs_error": covariance_error,
        "standard_error_max_abs_error": se_error,
        "residual_mean_abs": abs(residual_mean),
        "coefficients_match_manual": coefficient_error < 1e-9,
        "hc3_covariance_matches_manual": covariance_error < 1e-8,
        "standard_errors_match_covariance": se_error < 1e-10,
        "covariance_does_not_change_coefficient": abs(
            slope.estimate - classical_slope.estimate
        ) < 1e-12,
        "explicit_t_inference": robust.inference["distribution"] == "t"
        and robust.inference["df_inference"] == SCHOOLS - 2,
        "json_roundtrip_preserves_full_result": restored.model_dump(mode="json")
        == robust.model_dump(mode="json"),
    }
    assert all(value for value in checks.values() if isinstance(value, bool)), checks
    summary = {
        "nobs": robust.nobs,
        "class_size_mean": float(data["class_size"].mean()),
        "class_size_min": float(data["class_size"].min()),
        "class_size_max": float(data["class_size"].max()),
        "test_score_mean": float(data["test_score"].mean()),
        "intercept": intercept.estimate,
        "slope": slope.estimate,
        "slope_hc3_se": slope.std_error,
        "slope_classical_se": classical_slope.std_error,
        "slope_ci_low": slope.ci_low,
        "slope_ci_high": slope.ci_high,
        "slope_p_value": slope.p_value,
        "r_squared": robust.metrics["r_squared"],
        "prediction_at_20": intercept.estimate + 20.0 * slope.estimate,
        "prediction_at_25": intercept.estimate + 25.0 * slope.estimate,
        "five_student_reduction": -5.0 * slope.estimate,
        "five_student_reduction_ci_low": -5.0 * slope.ci_high,
        "five_student_reduction_ci_high": -5.0 * slope.ci_low,
        "true_slope": TRUE_SLOPE,
        "manual_max_leverage": manual["max_leverage"],
    }
    rows = data.to_dict(orient="records")
    data_digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    line = [
        {"class_size": value, "test_score": intercept.estimate + slope.estimate * value}
        for value in [18.0 + 0.25 * i for i in range(49)]
    ]
    chart_data = {
        "scatter": rows,
        "fitted_line": line,
        "residuals": [
            {"class_size": row["class_size"], "residual": error}
            for row, error in zip(rows, residuals)
        ],
    }
    result = {
        "metadata": {
            "lab": "04",
            "title": "Does Class Size Predict Test Scores?",
            "synthetic": True,
            "seed": SEED,
            "schools": SCHOOLS,
            "openecon_version": oe.__version__,
            "dtype": "float64",
            "device": "cpu",
            "covariance": "HC3",
            "confidence_level": 0.95,
            "inference_distribution": "Student t",
            "inference_df": SCHOOLS - 2,
            "missing": "raise",
            "weights": None,
            "data_sha256": data_digest,
            "source": "Prepared original simulated school-level teaching workbook.",
            "interpretation": "A teaching simulation; no empirical education-policy claim.",
        },
        "summary": summary,
        "models": {
            "HC3": robust.model_dump(mode="json"),
            "Classical": classical.model_dump(mode="json"),
        },
        "chart_data": chart_data,
        "checks": checks,
    }
    publication_table = oe.regression_table(
        {"HC3 robust": robust, "Classical": classical},
        caption="Class size and test scores: original synthetic data",
        label="tab:lab04-class-size",
        term_labels={"class_size": "Students per class"},
        precision=3,
        stars=False,
        notes="Identical 240-school sample in both columns; only covariance differs. "
        "School outcomes are simulated, not empirical policy evidence.",
    )
    print(robust.summary())
    print(
        f"Five fewer students: {summary['five_student_reduction']:.3f} predicted score "
        f"points (95% CI {summary['five_student_reduction_ci_low']:.3f}, "
        f"{summary['five_student_reduction_ci_high']:.3f}). Synthetic data only."
    )
    if display_callback is not None:
        display_callback(publication_table)
        display_callback(
            oe.plot.scatter(
                data=data,
                x="class_size",
                y="test_score",
                title="Simulated schools: class size and test score",
            )
        )
        display_callback(
            oe.plot.scatter(
                data=oe.DataFrame(chart_data["residuals"]),
                x="class_size",
                y="residual",
                title="Residual spread grows with simulated class size",
            )
        )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "reference.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (destination / "table.tex").write_text(str(publication_table), encoding="utf-8")
        saved = json.loads((destination / "reference.json").read_text(encoding="utf-8"))
        assert saved == result, "Saved complete-result readback differs."
    return result


if __name__ == "__main__":
    class_size_lab_result = run_lab(display_callback=globals().get("display"))
