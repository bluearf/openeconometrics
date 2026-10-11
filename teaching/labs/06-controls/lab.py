"""Lab 06: education, experience and log earnings on one shared sample.

Import wages_and_controls.xlsx into OpenEconometrics without renaming it,
then run this entire document. Ordinary Python reads the sibling workbook;
run_lab(data_path="/path/to/wages_and_controls.xlsx") selects an explicit file.
The prepared workers are synthetic. Default execution writes no files.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import openecon as oe
import torch
from openecon.models import ResultBundle


# Original dataset-generation seed (provenance only).
SEED = 6112026
GENERATED_ROWS = 600
TRUE_EDUCATION_SLOPE = 0.08
TRUE_EXPERIENCE_SLOPE = 0.03
DATA_FILE = 'wages_and_controls.xlsx'


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


def _coefficients(model):
    return {coefficient.term: coefficient for coefficient in model.coefficients}


def _matrix_ols_hc1(x, y):
    """Independent matrix calculation, including finite-sample HC1 scaling."""
    beta = torch.linalg.lstsq(x, y).solution
    residual = y - x @ beta
    bread = torch.linalg.inv(x.T @ x)
    scores = x * residual[:, None]
    n, k = x.shape
    covariance = (n / (n - k)) * bread @ (scores.T @ scores) @ bread
    return beta, covariance


def _t_975(degrees_of_freedom):
    """Independent Student-t quantile by Simpson integration and bisection.

    This small numerical oracle is for the lab's CI check; estimation and
    inference shown to students use the native OpenEconometrics result.
    """
    df = degrees_of_freedom
    normalizer = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)

    def area(upper):
        intervals = 512
        step = upper / intervals

        def density(value):
            return normalizer * (1 + value * value / df) ** (-(df + 1) / 2)

        total = density(0) + density(upper)
        total += sum((4 if i % 2 else 2) * density(i * step) for i in range(1, intervals))
        return step * total / 3

    low, high = 0.0, 4.0
    for _ in range(42):
        middle = (low + high) / 2
        if area(middle) < 0.475:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def run_lab(output_dir=None, display_callback=None, data_path=None):
    """Fit two specifications, validate their uncertainty, and return results.

    With no output_dir this function does not write files. The callback is the
    workbench's display function; ordinary Python can omit it. All checks raise
    on failure so a broken run cannot silently produce reference artifacts.
    """
    raw = load_data(data_path)
    data = raw.dropna(subset=["education", "experience", "log_earnings"]).reset_index(drop=True)
    short = oe.ols(data=data, y="log_earnings", x=["education"], covariance="HC1", missing="raise", device="cpu")
    adjusted = oe.ols(data=data, y="log_earnings", x=["education", "experience"], covariance="HC1", missing="raise", device="cpu")
    short_coefficients = _coefficients(short)
    adjusted_coefficients = _coefficients(adjusted)
    n = len(data)
    education = torch.tensor(data["education"].tolist(), dtype=torch.float64)
    experience = torch.tensor(data["experience"].tolist(), dtype=torch.float64)
    outcome = torch.tensor(data["log_earnings"].tolist(), dtype=torch.float64)
    ones = torch.ones(n, dtype=torch.float64)
    short_x = torch.column_stack((ones, education))
    adjusted_x = torch.column_stack((ones, education, experience))
    oracle_short, covariance_short = _matrix_ols_hc1(short_x, outcome)
    oracle_adjusted, covariance_adjusted = _matrix_ols_hc1(adjusted_x, outcome)

    # Residualize both variables on an intercept and experience (FWL theorem).
    controls = torch.column_stack((ones, experience))
    residual_education = education - controls @ torch.linalg.lstsq(controls, education).solution
    residual_outcome = outcome - controls @ torch.linalg.lstsq(controls, outcome).solution
    fwl_slope = float((residual_education @ residual_outcome) / (residual_education @ residual_education))

    # Exact *sample* omitted-variable identity for these nested linear models.
    delta = float(((education - education.mean()) @ (experience - experience.mean())) / ((education - education.mean()).square().sum()))
    short_slope = short_coefficients["education"].estimate
    adjusted_slope = adjusted_coefficients["education"].estimate
    experience_slope = adjusted_coefficients["experience"].estimate
    difference = short_slope - adjusted_slope
    decomposition = experience_slope * delta

    beta_error, covariance_error, ci_error = 0.0, 0.0, 0.0
    for model, names, beta, covariance in (
        (short, ["Intercept", "education"], oracle_short, covariance_short),
        (adjusted, ["Intercept", "education", "experience"], oracle_adjusted, covariance_adjusted),
    ):
        fitted = _coefficients(model)
        order = [names.index(item.term) for item in model.coefficients]
        native_covariance = torch.tensor(model.covariance_matrix, dtype=torch.float64)
        expected_covariance = covariance[order][:, order]
        covariance_error = max(covariance_error, float((native_covariance - expected_covariance).abs().max()))
        critical = _t_975(n - len(names))
        for index, name in enumerate(names):
            item = fitted[name]
            estimate, standard_error = float(beta[index]), math.sqrt(float(covariance[index, index]))
            beta_error = max(beta_error, abs(item.estimate - estimate))
            ci_error = max(ci_error, abs(item.ci_low - (estimate - critical * standard_error)), abs(item.ci_high - (estimate + critical * standard_error)))

    models = {"Education only": short.model_dump(mode="json"), "Education and experience": adjusted.model_dump(mode="json")}
    roundtrip = all(ResultBundle.model_validate_json(json.dumps(payload)).model_dump(mode="json") == payload for payload in models.values())
    checks = {
        "same_sample": short.sample_positions == adjusted.sample_positions and short.nobs == adjusted.nobs == 588,
        "no_implicit_exclusions": short.dropped_rows == adjusted.dropped_rows == 0,
        "independent_coefficients": beta_error < 1e-10,
        "independent_hc1_covariance": covariance_error < 1e-10,
        "independent_student_t_intervals": ci_error < 1e-8,
        "fwl_identity": abs(fwl_slope - adjusted_slope) < 1e-10,
        "sample_ovb_identity": abs(difference - decomposition) < 1e-10,
        "full_result_json_roundtrip": roundtrip,
    }
    if not all(checks.values()):
        raise AssertionError(f"Lab 06 verification failed: {checks}")

    adjusted_education = adjusted_coefficients["education"]
    summary = {
        "generated_rows": GENERATED_ROWS,
        "analysis_rows": n,
        "excluded_missing_experience": len(raw) - n,
        "education_mean": float(education.mean()),
        "experience_mean": float(experience.mean()),
        "education_experience_correlation": float(torch.corrcoef(torch.stack((education, experience)))[0, 1]),
        "short_education_coefficient": short_slope,
        "short_education_se": short_coefficients["education"].std_error,
        "short_education_ci_low": short_coefficients["education"].ci_low,
        "short_education_ci_high": short_coefficients["education"].ci_high,
        "adjusted_education_coefficient": adjusted_slope,
        "adjusted_education_se": adjusted_education.std_error,
        "adjusted_education_ci_low": adjusted_education.ci_low,
        "adjusted_education_ci_high": adjusted_education.ci_high,
        "adjusted_experience_coefficient": experience_slope,
        "adjusted_experience_se": adjusted_coefficients["experience"].std_error,
        "adjusted_experience_ci_low": adjusted_coefficients["experience"].ci_low,
        "adjusted_experience_ci_high": adjusted_coefficients["experience"].ci_high,
        "adjusted_education_approx_percent": 100 * adjusted_slope,
        "adjusted_education_exact_percent": 100 * math.expm1(adjusted_slope),
        "adjusted_education_exact_percent_ci_low": 100 * math.expm1(adjusted_education.ci_low),
        "adjusted_education_exact_percent_ci_high": 100 * math.expm1(adjusted_education.ci_high),
        "experience_on_education_slope": delta,
        "short_minus_adjusted": difference,
        "sample_ovb_component": decomposition,
        "fwl_education_slope": fwl_slope,
        "max_coefficient_error": beta_error,
        "max_covariance_error": covariance_error,
        "max_interval_error": ci_error,
    }
    analysis_records = data[["worker_id", "education", "experience", "log_earnings", "hourly_earnings"]].to_dict(orient="records")
    result = {
        "metadata": {
            "lab": "06-controls",
            "title": "What Changes When We Add Controls?",
            "data_kind": "original synthetic teaching data; no real workers",
            "seed": SEED,
            "data_source": DATA_FILE,
            "torch_version": str(torch.__version__),
            "true_education_slope": TRUE_EDUCATION_SLOPE,
            "true_experience_slope": TRUE_EXPERIENCE_SLOPE,
            "covariance": "HC1",
            "inference": "Student t; n-k residual degrees of freedom; 95% intervals",
            "sample_rule": "drop rows missing any variable in either specification before both fits; then missing='raise'",
            "analysis_worker_ids": data["worker_id"].tolist(),
            "data_sha256": hashlib.sha256(json.dumps(analysis_records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        },
        "summary": summary,
        "models": models,
        "chart_data": {
            "workers": analysis_records,
            "partial_regression": [{"residual_education": float(x), "residual_log_earnings": float(y)} for x, y in zip(residual_education, residual_outcome)],
            "education_coefficients": [
                {"model": "Education only", "estimate": short_slope, "ci_low": short_coefficients["education"].ci_low, "ci_high": short_coefficients["education"].ci_high},
                {"model": "Education and experience", "estimate": adjusted_slope, "ci_low": adjusted_education.ci_low, "ci_high": adjusted_education.ci_high},
            ],
        },
        "checks": checks,
    }
    table = oe.regression_table(
        {"Education only": short, "Education and experience": adjusted},
        caption="Education and synthetic log hourly earnings: a common sample",
        label="tab:lab06", precision=4, stars=False,
        term_labels={"education": "Education (years)", "experience": "Experience (years)"},
        notes="Original synthetic workers. Both models use the same 588 complete cases. HC1 standard errors; 95 percent Student-t inference. Estimates are not population evidence.",
    )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "reference.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (destination / "table.tex").write_text(str(table), encoding="utf-8")
        saved = json.loads((destination / "reference.json").read_text(encoding="utf-8"))
        if saved != result:
            raise AssertionError("Saved reference readback differs from the validated run.")
    if display_callback is not None:
        display_callback(short)
        display_callback(adjusted)
        display_callback(table)
        partial = oe.DataFrame(result["chart_data"]["partial_regression"])
        display_callback(oe.plot.scatter(data=partial, x="residual_education", y="residual_log_earnings", title="Education and log earnings after removing experience"))
    return result


if __name__ == "__main__":
    lab_results = run_lab(display_callback=globals().get("display"))
    s = lab_results["summary"]
    print(f"Synthetic workers: {s['analysis_rows']} of {s['generated_rows']}; 12 missing experience reports excluded before both fits.")
    print(f"Education coefficient: {s['short_education_coefficient']:.4f} without experience; {s['adjusted_education_coefficient']:.4f} with experience (HC1 SE {s['adjusted_education_se']:.4f}).")
    print(f"Exact fitted earnings ratio: {s['adjusted_education_exact_percent']:.2f}% per education year, holding experience fixed; this is not a population causal claim.")
