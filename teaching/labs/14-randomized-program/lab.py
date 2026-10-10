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


DATA_FILE = "training_program.xlsx"


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
    unadjusted = oe.ols(
        data=data, y="earnings", x=["offer"], covariance="HC2", missing="raise", device="cpu"
    )
    adjusted = oe.ols(
        data=data,
        y="earnings",
        x=["offer", "baseline_score"],
        covariance="HC2",
        missing="raise",
        device="cpu",
    )
    selected = oe.ols(
        data=data, y="earnings", x=["participated"], covariance="HC2", missing="raise", device="cpu"
    )
    offered = data.offer == 1
    y1 = tensor(data.loc[offered, "earnings"])
    y0 = tensor(data.loc[~offered, "earnings"])
    diff = float(y1.mean() - y0.mean())
    neyman_se = float((y1.var(unbiased=True) / len(y1) + y0.var(unbiased=True) / len(y0)).sqrt())
    errors = []
    for model, names in [
        (unadjusted, ["offer"]),
        (adjusted, ["offer", "baseline_score"]),
        (selected, ["participated"]),
    ]:
        x = torch.column_stack(
            (torch.ones(len(data), dtype=torch.float64), *[tensor(data[name]) for name in names])
        )
        beta, covariance = matrix_ols(x, tensor(data.earnings), "HC2")
        errors.append(compare_linear(model, ["Intercept", *names], beta, covariance))
    u, a, s = coef(unadjusted, "offer"), coef(adjusted, "offer"), coef(selected, "participated")
    take1 = float(data.loc[offered, "participated"].mean())
    take0 = float(data.loc[~offered, "participated"].mean())
    true_compliers = float((data.take_if_offer - data.take_if_no_offer).mean())
    summary = {
        "nobs": len(data),
        "offered": int(offered.sum()),
        "not_offered": int((~offered).sum()),
        "mean_offered": float(y1.mean()),
        "mean_not_offered": float(y0.mean()),
        "takeup_offered": take1,
        "takeup_not_offered": take0,
        "itt": u.estimate,
        "itt_se": u.std_error,
        "itt_ci_low": u.ci_low,
        "itt_ci_high": u.ci_high,
        "adjusted_itt": a.estimate,
        "adjusted_se": a.std_error,
        "adjusted_ci_low": a.ci_low,
        "adjusted_ci_high": a.ci_high,
        "participation_association": s.estimate,
        "participation_se": s.std_error,
        "true_participation_effect": 30.0,
        "true_finite_sample_offer_effect": 30 * true_compliers,
        "true_complier_fraction": true_compliers,
        "baseline_mean_offered": float(data.loc[offered, "baseline_score"].mean()),
        "baseline_mean_not_offered": float(data.loc[~offered, "baseline_score"].mean()),
        "max_coefficient_error": max(e[0] for e in errors),
        "max_covariance_error": max(e[1] for e in errors),
    }
    checks = {
        "randomized_equal_allocation": len(y1) == len(y0) == 300,
        "independent_mean_difference": abs(diff - u.estimate) < 1e-9,
        "independent_neyman_hc2_se": abs(neyman_se - u.std_error) < 1e-9,
        "independent_three_ols_coefficients": summary["max_coefficient_error"] < 1e-8,
        "independent_three_hc2_covariances": summary["max_covariance_error"] < 1e-7,
        "monotone_potential_takeup": bool((data.take_if_offer >= data.take_if_no_offer).all()),
        "all_randomized_applicants_followed": not data.isna().any().any(),
        "same_sample_positions": all(
            m.sample_positions == list(range(len(data))) for m in [unadjusted, adjusted, selected]
        ),
    }
    chart_data = {
        "panels": [
            {
                "kind": "interval",
                "title": "Offer effects with and without a baseline control",
                "xlabel": "Earnings difference (currency units)",
                "ylabel": "Intent-to-treat estimate",
                "series": [
                    {
                        "label": "95% HC2 intervals",
                        "x": [u.estimate, a.estimate],
                        "y": ["Unadjusted offer", "Baseline-adjusted offer"],
                        "low": [u.ci_low, a.ci_low],
                        "high": [u.ci_high, a.ci_high],
                    }
                ],
            },
            {
                "kind": "bar",
                "title": "Noncompliance in both randomized groups",
                "xlabel": "Assignment",
                "ylabel": "Participation rate",
                "series": [
                    {"label": "Observed uptake", "x": ["No offer", "Offer"], "y": [take0, take1]}
                ],
            },
        ]
    }
    displays = [
        unadjusted,
        adjusted,
        oe.DataFrame(
            [
                {
                    "assignment": "No offer",
                    "n": len(y0),
                    "mean_earnings": float(y0.mean()),
                    "participation_rate": take0,
                },
                {
                    "assignment": "Offer",
                    "n": len(y1),
                    "mean_earnings": float(y1.mean()),
                    "participation_rate": take1,
                },
            ]
        ),
    ]
    return complete(
        "14-randomized-program",
        "Evaluating a Randomized Training Program",
        data,
        summary,
        {
            "Unadjusted offer HC2": unadjusted,
            "Baseline-adjusted offer HC2": adjusted,
            "Participation association HC2": selected,
        },
        chart_data,
        checks,
        output_dir,
        display_callback,
        displays,
    )


if __name__ == "__main__":
    randomized_program_lab_result = run_lab(display_callback=globals().get("display"))
