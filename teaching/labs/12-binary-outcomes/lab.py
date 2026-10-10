"""Original synthetic teaching lab; run the complete file in a new Python document.

Import the supplied named workbook or keep it beside this source file.
run_lab(output_dir=None, display_callback=None, data_path=None) writes only on explicit request.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
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


DATA_FILE = "labor_force_participation.xlsx"


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
    predictors = ["education", "children", "age"]
    lpm = oe.ols(
        data=data, y="participating", x=predictors, covariance="HC1", missing="raise", device="cpu"
    )
    logit = oe.logit(
        data=data, y="participating", x=predictors, covariance="nonrobust", missing="raise"
    )
    probit = oe.probit(
        data=data, y="participating", x=predictors, covariance="nonrobust", missing="raise"
    )
    names = ["Intercept", *predictors]
    x = torch.column_stack(
        (torch.ones(len(data), dtype=torch.float64), *[tensor(data[name]) for name in predictors])
    )
    y = tensor(data.participating)
    beta = torch.zeros(4, dtype=torch.float64)
    for _ in range(30):
        p = torch.sigmoid(x @ beta)
        step = torch.linalg.solve(x.T @ (x * (p * (1 - p))[:, None]), x.T @ (y - p))
        beta += step
        if float(step.abs().max()) < 1e-12:
            break
    p = torch.sigmoid(x @ beta)
    covariance = torch.linalg.inv(x.T @ (x * (p * (1 - p))[:, None]))
    b_error, v_error = compare_linear(logit, names, beta, covariance)
    lb, lv = matrix_ols(x, y)
    lbe, lve = compare_linear(lpm, names, lb, lv)
    pb = tensor([coef(probit, name).estimate for name in names]).requires_grad_()

    def probit_loss(parameters):
        eta = x @ parameters
        return -(y * torch.special.log_ndtr(eta) + (1 - y) * torch.special.log_ndtr(-eta)).sum()

    gradient = torch.autograd.functional.jacobian(probit_loss, pb)
    hessian = torch.autograd.functional.hessian(probit_loss, pb)
    pve = float((tensor_matrix(probit.covariance_matrix) - torch.linalg.inv(hessian)).abs().max())
    ame = oe.margins(logit, variables=["education"], data=data, method="ame")
    ame_record = ame.to_dict(orient="records")[0]
    # The parameter derivative is aggregated first, then delta-method variance.
    b_native = tensor([coef(logit, name).estimate for name in names]).requires_grad_()

    def mean_effect(parameters):
        probability = torch.sigmoid(x @ parameters)
        return (probability * (1 - probability)).mean() * parameters[1]

    manual_ame = mean_effect(b_native)
    ame_gradient = torch.autograd.functional.jacobian(mean_effect, b_native)
    manual_ame_se = float(
        (ame_gradient @ tensor_matrix(logit.covariance_matrix) @ ame_gradient).sqrt()
    )
    grid = pd.DataFrame(
        {"education": [10 + 0.25 * i for i in range(33)], "children": 1.0, "age": 35.0}
    )
    curves = []
    for label, model in [("LPM", lpm), ("Logit", logit), ("Probit", probit)]:
        prediction = oe.predict(model, data=grid)
        values = prediction["xb" if label == "LPM" else "response"].tolist()
        curves.append({"label": label, "x": grid.education.tolist(), "y": values})
    education = coef(logit, "education")
    summary = {
        "nobs": len(data),
        "participation_rate": float(data.participating.mean()),
        "lpm_education": coef(lpm, "education").estimate,
        "lpm_education_se": coef(lpm, "education").std_error,
        "logit_education": education.estimate,
        "logit_education_se": education.std_error,
        "logit_education_odds_ratio": math.exp(education.estimate),
        "probit_education": coef(probit, "education").estimate,
        "logit_education_ame": float(ame_record["estimate"]),
        "logit_education_ame_se": float(ame_record["std_error"]),
        "logit_education_ame_ci_low": float(ame_record["ci_low"]),
        "logit_education_ame_ci_high": float(ame_record["ci_high"]),
        "prediction_education_14_child_1_age_35": curves[1]["y"][16],
        "max_coefficient_error": max(b_error, lbe),
        "max_covariance_error": max(v_error, lve, pve),
        "probit_score_max_abs": float(gradient.abs().max()),
    }
    checks = {
        "binary_coding": set(data.participating) == {0.0, 1.0},
        "independent_logit_newton_coefficients": b_error < 1e-7,
        "independent_logit_information_covariance": v_error < 1e-7,
        "independent_lpm_hc1": lbe < 1e-9 and lve < 1e-9,
        "probit_score_and_information": float(gradient.abs().max()) < 1e-5 and pve < 1e-7,
        "independent_average_effect": abs(
            float(manual_ame.detach()) - summary["logit_education_ame"]
        )
        < 1e-8
        and abs(manual_ame_se - summary["logit_education_ame_se"]) < 1e-8,
        "probabilities_in_unit_interval": all(
            0 <= value <= 1 for series in curves[1:] for value in series["y"]
        ),
    }
    chart_data = {
        "figure": {
            "kind": "line",
            "title": "Fitted participation probability at age 35, one child",
            "xlabel": "Education (years)",
            "ylabel": "Probability of participation",
            "series": curves,
        },
        "education_margins": ame.to_dict(orient="records"),
    }
    displays = [
        logit,
        probit,
        ame,
        oe.plot.line(
            data=oe.DataFrame(
                {"education": grid.education.tolist(), "probability": curves[1]["y"]}
            ),
            x="education",
            y="probability",
            title="Logit probability at fixed age and children",
        ),
    ]
    return complete(
        "12-binary-outcomes",
        "Explaining Labor-Force Participation",
        data,
        summary,
        {"LPM HC1": lpm, "Logit information": logit, "Probit information": probit},
        chart_data,
        checks,
        output_dir,
        display_callback,
        displays,
    )


if __name__ == "__main__":
    binary_outcomes_lab_result = run_lab(display_callback=globals().get("display"))
