"""Original observational-school capstone with explicit common-sample reporting."""

import json
from pathlib import Path

import pandas as pd
import torch

import openecon as oe
from openecon.models import ResultBundle

DATA_FILE = 'school_tutoring.xlsx'
TRUE_EFFECT = 1.2


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


def manual_hc3(data):
    columns = [torch.ones(len(data), dtype=torch.float64)]
    columns.extend(
        torch.tensor(data[name].tolist(), dtype=torch.float64)
        for name in ("tutoring_hours", "prior_score", "resource_index")
    )
    x = torch.stack(columns, dim=1)
    y = torch.tensor(data.final_score.tolist(), dtype=torch.float64)
    beta = torch.linalg.lstsq(x, y).solution
    inverse = torch.linalg.inv(x.T @ x)
    residual = y - x @ beta
    leverage = ((x @ inverse) * x).sum(1)
    scaled = residual / (1 - leverage)
    covariance = inverse @ (x.T @ (x * scaled.square()[:, None])) @ inverse
    return {
        "beta": beta.tolist(),
        "covariance": covariance.tolist(),
        "fitted": (x @ beta).tolist(),
        "residuals": residual.tolist(),
    }


def run_lab(output_dir=None, display_callback=None, data_path=None):
    data = load_data(data_path)
    common = data.dropna(
        subset=["final_score", "tutoring_hours", "prior_score", "resource_index"]
    ).copy()
    common = common.reset_index(drop=True)
    models = {
        "unadjusted_common": oe.ols(
            data=common,
            y="final_score",
            x=["tutoring_hours"],
            covariance="HC3",
            missing="raise",
            device="cpu",
        ),
        "adjusted": oe.ols(
            data=common,
            y="final_score",
            x=["tutoring_hours", "prior_score", "resource_index"],
            covariance="HC3",
            missing="raise",
            device="cpu",
        ),
        "without_resources": oe.ols(
            data=common,
            y="final_score",
            x=["tutoring_hours", "prior_score"],
            covariance="HC3",
            missing="raise",
            device="cpu",
        ),
        "unadjusted_all": oe.ols(
            data=data,
            y="final_score",
            x=["tutoring_hours"],
            covariance="HC3",
            missing="raise",
            device="cpu",
        ),
    }
    adjusted = models["adjusted"]
    manual = manual_hc3(common)
    coefficients = {c.term: c for c in adjusted.coefficients}
    main = coefficients["tutoring_hours"]
    scenarios = pd.DataFrame(
        {"tutoring_hours": [2.0, 6.0], "prior_score": [60.0, 60.0], "resource_index": [0.0, 0.0]}
    )
    prediction = adjusted.predict(data=scenarios, interval="mean")
    serialized = adjusted.model_dump(mode="json")
    restored = ResultBundle.model_validate(serialized)
    comparison = []
    for name, fit in models.items():
        row = next(c for c in fit.coefficients if c.term == "tutoring_hours")
        comparison.append(
            {
                "model": name,
                "nobs": fit.nobs,
                "estimate": row.estimate,
                "std_error": row.std_error,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
            }
        )
    checks = {
        "manual_coefficient_max_error": max(
            abs(a.estimate - b) for a, b in zip(adjusted.coefficients, manual["beta"], strict=True)
        ),
        "manual_covariance_max_error": float(
            (
                torch.tensor(adjusted.covariance_matrix, dtype=torch.float64)
                - torch.tensor(manual["covariance"], dtype=torch.float64)
            )
            .abs()
            .max()
        ),
        "same_primary_sample": all(
            models[name].sample_positions == list(range(len(common)))
            and models[name].nobs == len(common)
            for name in ("unadjusted_common", "adjusted", "without_resources")
        ),
        "explicit_complete_case_count": len(common) == 277 and len(data) == 300,
        "prediction_contrast_error": abs(
            float(prediction.xb.iloc[1] - prediction.xb.iloc[0]) - 4 * main.estimate
        ),
        "full_state_roundtrip": restored.model_dump(mode="json") == serialized,
        "df_matches_parameter_count": adjusted.inference["df_inference"] == len(common) - 4,
    }
    for key, value in checks.items():
        assert value if isinstance(value, bool) else value < 1e-7, (key, value)
    figure = {
        "kind": "scatter",
        "title": "Residuals challenge the adequacy of a fitted report",
        "xlabel": "Adjusted fitted score",
        "ylabel": "Residual score",
        "series": [
            {"label": "277 complete-case schools", "x": manual["fitted"], "y": manual["residuals"]}
        ],
    }
    result = {
        "metadata": {
            "lab_id": "20",
            "title": "From Research Question to Finished Report",
            "data_source": DATA_FILE,
            "data_kind": "original synthetic observational schools",
            "covariance": "HC3",
            "missing": "explicit complete-case sample, then raise",
            "unobserved_confounder": "motivation affects tutoring and final score; intentionally unavailable to analyst",
            "retained_school_ids": common.school.tolist(),
            "excluded_school_ids": data.loc[data.resource_index.isna(), "school"].tolist(),
            "openecon_version": oe.__version__,
        },
        "summary": {
            "nobs": len(common),
            "original_nobs": len(data),
            "excluded_nobs": len(data) - len(common),
            "true_structural_effect": TRUE_EFFECT,
            "adjusted_tutoring": main.estimate,
            "adjusted_se": main.std_error,
            "adjusted_ci_low": main.ci_low,
            "adjusted_ci_high": main.ci_high,
            "adjusted_p_value": main.p_value,
            "unadjusted_tutoring": comparison[0]["estimate"],
            "without_resources_tutoring": comparison[2]["estimate"],
            "all_sample_unadjusted_tutoring": comparison[3]["estimate"],
            "r_squared": adjusted.metrics["r_squared"],
            "scenario_contrast": 4 * main.estimate,
            "scenario_low_prediction": float(prediction.xb.iloc[0]),
            "scenario_high_prediction": float(prediction.xb.iloc[1]),
            "df_inference": adjusted.inference["df_inference"],
        },
        "models": {name: fit.model_dump(mode="json") for name, fit in models.items()},
        "predictions": {"records": prediction.to_dict(orient="records"), "attrs": prediction.attrs},
        "chart_data": {"figure": figure, "specification_comparison": comparison},
        "checks": checks,
    }
    table = oe.regression_table(
        {
            "Unadjusted, common": models["unadjusted_common"],
            "Adjusted": adjusted,
            "Without resources": models["without_resources"],
        },
        precision=3,
        stars=False,
        caption="Synthetic observational schools: a reproducible association report",
        notes=[
            "Same 277 complete-case schools; HC3 standard errors and Student t inference.",
            "Unobserved motivation remains a confounder; adjusted associations are not policy effects.",
        ],
    )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "reference.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (destination / "table.tex").write_text(str(table) + "\n", encoding="utf-8")
    if display_callback is not None:
        display_callback(adjusted)
        display_callback(oe.DataFrame(comparison))
        display_callback(prediction)
        display_callback(
            oe.plot.scatter(
                data={"fitted": manual["fitted"], "residual": manual["residuals"]},
                x="fitted",
                y="residual",
                title=figure["title"],
            )
        )
    else:
        print(adjusted.summary())
        print(pd.DataFrame(comparison).to_string(index=False))
        print(prediction.to_string(index=False))
    return result


if __name__ == "__main__":
    report_lab_result = run_lab(globals().get("LAB_OUTPUT_DIR"), globals().get("display"))
