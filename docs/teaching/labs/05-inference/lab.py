"""Original confidence-interval and hypothesis-test teaching example."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import openecon as oe
from openecon.models import ResultBundle


# Original dataset-generation seed (provenance only).
SEED = 5052026
DATA_FILE = 'wage_inference.xlsx'


def _t_quantile(probability, df):
    normalizer = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)

    def area(upper):
        intervals = 512
        step = upper / intervals
        total = 0.0
        for index in range(intervals + 1):
            value = index * step
            weight = 1 if index in (0, intervals) else 4 if index % 2 else 2
            total += weight * normalizer * (1 + value * value / df) ** (-(df + 1) / 2)
        return total * step / 3

    low, high = 0.0, 5.0
    for _ in range(44):
        middle = (low + high) / 2
        if area(middle) < probability - 0.5:
            low = middle
        else:
            high = middle
    return (low + high) / 2


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
    fitted = {
        str(level): oe.ols(
            data=data,
            y="hourly_wage",
            x=["education"],
            covariance="HC3",
            alpha=1 - level,
            missing="raise",
            device="cpu",
        )
        for level in (0.90, 0.95, 0.99)
    }
    model = fitted["0.95"]
    slope = next(row for row in model.coefficients if row.term == "education")
    x = torch.column_stack(
        (
            torch.ones(140, dtype=torch.float64),
            torch.tensor(data.education.tolist(), dtype=torch.float64),
        )
    )
    y = torch.tensor(data.hourly_wage.tolist(), dtype=torch.float64)
    beta = torch.linalg.lstsq(x, y).solution
    bread = torch.linalg.inv(x.T @ x)
    residual = y - x @ beta
    leverage = ((x @ bread) * x).sum(dim=1)
    scores = x * (residual / (1 - leverage))[:, None]
    covariance = bread @ scores.T @ scores @ bread
    intervals = []
    ci_error = 0.0
    for level, fit in fitted.items():
        row = next(item for item in fit.coefficients if item.term == "education")
        critical = _t_quantile((1 + float(level)) / 2, 138)
        ci_error = max(
            ci_error,
            abs(row.ci_low - (row.estimate - critical * row.std_error)),
            abs(row.ci_high - (row.estimate + critical * row.std_error)),
        )
        intervals.append(
            {
                "confidence": float(level),
                "estimate": row.estimate,
                "std_error": row.std_error,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "width": row.ci_high - row.ci_low,
            }
        )
    benchmark = model.lincom({"education": 1}, constant=-1.0)
    covariance_error = float(
        (torch.tensor(model.covariance_matrix, dtype=torch.float64) - covariance).abs().max()
    )
    checks = {
        "all140_workers_retained": all(
            fit.nobs == 140 and fit.dropped_rows == 0 for fit in fitted.values()
        ),
        "same_fit_different_confidence": max(
            abs(row["estimate"] - slope.estimate) for row in intervals
        )
        < 1e-12,
        "independent_ols": float(
            (torch.tensor([c.estimate for c in model.coefficients], dtype=torch.float64) - beta)
            .abs()
            .max()
        )
        < 1e-10,
        "independent_hc3_covariance": covariance_error < 1e-9,
        "independent_t_interval_quantiles": ci_error < 1e-8,
        "nested_confidence_intervals": intervals[0]["width"]
        < intervals[1]["width"]
        < intervals[2]["width"],
        "nonzero_null_contrast": abs(benchmark["estimate"] - (slope.estimate - 1)) < 1e-12,
        "full_model_roundtrip": all(
            ResultBundle.model_validate_json(fit.model_dump_json()).model_dump(mode="json")
            == fit.model_dump(mode="json")
            for fit in fitted.values()
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    summary = {
        "nobs": 140,
        "estimate": slope.estimate,
        "hc3_se": slope.std_error,
        "zero_null_t": slope.statistic,
        "zero_null_p": slope.p_value,
        "ci95_low": slope.ci_low,
        "ci95_high": slope.ci_high,
        "benchmark_one_t": benchmark["statistic"],
        "benchmark_one_p": benchmark["p_value"],
        "benchmark_one_difference": benchmark["estimate"],
        "two_year_contrast": 2 * slope.estimate,
        "two_year_contrast_se": 2 * slope.std_error,
        "two_year_ci_low": 2 * slope.ci_low,
        "two_year_ci_high": 2 * slope.ci_high,
        "ci90_low": intervals[0]["ci_low"],
        "ci90_high": intervals[0]["ci_high"],
        "ci99_low": intervals[2]["ci_low"],
        "ci99_high": intervals[2]["ci_high"],
    }
    result = {
        "metadata": {
            "lab": "05-inference",
            "title": "How Precise Is Our Estimate?",
            "seed": SEED,
            "synthetic": True,
            "covariance": "HC3",
            "inference_df": 138,
            "units": "currency/hour per education year",
        },
        "summary": summary,
        "models": {key: fit.model_dump(mode="json") for key, fit in fitted.items()},
        "chart_data": {
            "intervals": intervals,
            "figure": {
                "kind": "interval",
                "title": "One estimate, three confidence levels",
                "xlabel": "Confidence level",
                "ylabel": "Education coefficient (currency/hour/year)",
                "series": [
                    {
                        "label": "HC3 estimate and interval",
                        "x": ["90%", "95%", "99%"],
                        "y": [row["estimate"] for row in intervals],
                        "low": [row["ci_low"] for row in intervals],
                        "high": [row["ci_high"] for row in intervals],
                    }
                ],
            },
        },
        "checks": checks,
    }
    table = oe.regression_table(
        {"90 percent": fitted["0.9"], "95 percent": model, "99 percent": fitted["0.99"]},
        precision=4,
        stars=False,
        caption="Synthetic education-wage association: confidence levels differ; fit does not",
    )
    if display_callback is not None:
        display_callback(model)
        display_callback(oe.DataFrame(intervals))
        display_callback(
            oe.plot.coefficients(
                {
                    "coefficients": [
                        {"term": f"{int(row['confidence'] * 100)} percent", **row}
                        for row in intervals
                    ]
                },
                title="HC3 slope confidence intervals",
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
        f"Synthetic education coefficient {slope.estimate:.4f}, HC3 SE {slope.std_error:.4f}; 95% CI [{slope.ci_low:.4f}, {slope.ci_high:.4f}]."
    )
    print(f"Two-sided zero-null p={slope.p_value:.6g}; benchmark-one p={benchmark['p_value']:.6g}.")
    return result


if __name__ == "__main__":
    lab_result = run_lab(display_callback=globals().get("display"))
