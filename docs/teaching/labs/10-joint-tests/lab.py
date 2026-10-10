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


DATA_FILE = "productivity_joint_tests.xlsx"


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
    model = oe.ols(
        data=data,
        y="productivity",
        x=["coaching", "software", "tenure"],
        covariance="HC1",
        missing="raise",
        device="cpu",
    )
    x = torch.column_stack(
        (
            torch.ones(len(data), dtype=torch.float64),
            tensor(data.coaching),
            tensor(data.software),
            tensor(data.tenure),
        )
    )
    beta, covariance = matrix_ols(x, tensor(data.productivity))
    b_error, v_error = compare_linear(
        model, ["Intercept", "coaching", "software", "tenure"], beta, covariance
    )
    joint = model.test(["coaching", "software"])
    equal = model.test({"coaching": 1, "software": -1})
    combined = model.lincom({"coaching": 1, "software": 1})
    r = torch.tensor([[0.0, 1, 0, 0], [0.0, 0, 1, 0]], dtype=torch.float64)
    rb = r @ beta
    manual_f = float(rb @ torch.linalg.solve(r @ covariance @ r.T, rb) / 2)
    contrast = torch.tensor([0.0, 1, 1, 0], dtype=torch.float64)
    manual_sum = float(contrast @ beta)
    manual_sum_se = float((contrast @ covariance @ contrast).sqrt())
    c1, c2 = coef(model, "coaching"), coef(model, "software")
    summary = {
        "nobs": len(data),
        "coaching_estimate": c1.estimate,
        "coaching_se": c1.std_error,
        "coaching_p": c1.p_value,
        "software_estimate": c2.estimate,
        "software_se": c2.std_error,
        "software_p": c2.p_value,
        "joint_f": joint["statistic"],
        "joint_p": joint["p_value"],
        "equal_f": equal["statistic"],
        "equal_p": equal["p_value"],
        "combined_estimate": combined["estimate"],
        "combined_se": combined["std_error"],
        "combined_ci_low": combined["ci_low"],
        "combined_ci_high": combined["ci_high"],
        "coaching_software_correlation": float(data.coaching.corr(data.software)),
        "slope_covariance": float(covariance[1, 2]),
        "max_coefficient_error": b_error,
        "max_covariance_error": v_error,
    }
    checks = {
        "independent_coefficients": b_error < 1e-9,
        "independent_hc1_covariance": v_error < 1e-8,
        "independent_joint_wald_f": abs(manual_f - joint["statistic"]) < 1e-8,
        "independent_sum_contrast": abs(manual_sum - combined["estimate"]) < 1e-9
        and abs(manual_sum_se - combined["std_error"]) < 1e-9,
        "joint_dimensions": joint["df_num"] == 2 and joint["df_denom"] == len(data) - 4,
        "complete_sample_positions": model.sample_positions == list(range(len(data))),
    }
    chart_data = {
        "figure": {
            "kind": "interval",
            "title": "Individual channels and their combined change",
            "xlabel": "Productivity points per specified change",
            "ylabel": "Contrast",
            "series": [
                {
                    "label": "95% HC1 intervals",
                    "x": [c1.estimate, c2.estimate, combined["estimate"]],
                    "y": ["Coaching +1 hour", "Software +1 hour", "Both +1 hour"],
                    "low": [c1.ci_low, c2.ci_low, combined["ci_low"]],
                    "high": [c1.ci_high, c2.ci_high, combined["ci_high"]],
                }
            ],
        },
        "joint_test": joint,
        "equal_test": equal,
        "sum_contrast": combined,
    }
    displays = [
        model,
        oe.DataFrame(
            [
                {"claim": "Both slopes zero", "F": joint["statistic"], "p_value": joint["p_value"]},
                {"claim": "Equal slopes", "F": equal["statistic"], "p_value": equal["p_value"]},
            ]
        ),
    ]
    return complete(
        "10-joint-tests",
        "Testing Several Claims Together",
        data,
        summary,
        {"HC1 productivity": model},
        chart_data,
        checks,
        output_dir,
        display_callback,
        displays,
    )


if __name__ == "__main__":
    joint_tests_lab_result = run_lab(display_callback=globals().get("display"))
