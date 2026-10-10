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


DATA_FILE = "schooling_instruments.xlsx"


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


def manual_iv(data):
    ones = torch.ones(len(data), dtype=torch.float64)
    x = torch.column_stack((ones, tensor(data.background), tensor(data.schooling)))
    z = torch.column_stack((ones, tensor(data.background), tensor(data.offer)))
    y = tensor(data.log_earnings)
    xhat = z @ torch.linalg.lstsq(z, x).solution
    bread = torch.linalg.inv(xhat.T @ x)
    beta = bread @ (xhat.T @ y)
    residual = y - x @ beta
    scores = xhat * residual[:, None]
    covariance = len(data) / (len(data) - 3) * bread @ (scores.T @ scores) @ bread.T
    return beta, covariance


def run_lab(output_dir=None, display_callback=None, data_path=None):
    supplied = load_data(data_path)
    columns = ["person", "offer", "background", "schooling", "log_earnings"]
    data = supplied[columns].copy()

    def fit_iv(frame):
        return oe.ivregress(
            data=frame,
            y="log_earnings",
            x=["background"],
            endog=["schooling"],
            instruments=["offer"],
            covariance="robust",
            small=True,
            missing="raise",
        )

    iv = fit_iv(data)
    weak_data = data.copy()
    weak_data["schooling"] = supplied["schooling_weak"]
    weak_data["log_earnings"] = supplied["log_earnings_weak"]
    weak = fit_iv(weak_data)
    invalid_data = data.copy()
    invalid_data["schooling"] = supplied["schooling_invalid"]
    invalid_data["log_earnings"] = supplied["log_earnings_invalid"]
    invalid = fit_iv(invalid_data)
    ols = oe.ols(
        data=data,
        y="log_earnings",
        x=["background", "schooling"],
        covariance="HC1",
        missing="raise",
        device="cpu",
    )
    first = oe.ols(
        data=data,
        y="schooling",
        x=["background", "offer"],
        covariance="HC1",
        missing="raise",
        device="cpu",
    )
    errors = []
    for model, frame in [(iv, data), (weak, weak_data), (invalid, invalid_data)]:
        beta, covariance = manual_iv(frame)
        errors.append(
            compare_linear(model, ["Intercept", "background", "schooling"], beta, covariance)
        )
    first_f = first.test(["offer"])
    native_first = iv.extra["first_stage"][0]
    strong_c, weak_c, bad_c, ols_c = [coef(m, "schooling") for m in [iv, weak, invalid, ols]]
    summary = {
        "nobs": len(data),
        "true_schooling_slope": 0.08,
        "ols_schooling": ols_c.estimate,
        "ols_se": ols_c.std_error,
        "iv_schooling": strong_c.estimate,
        "iv_se": strong_c.std_error,
        "iv_ci_low": strong_c.ci_low,
        "iv_ci_high": strong_c.ci_high,
        "first_stage_offer": coef(first, "offer").estimate,
        "first_stage_f": native_first["f_statistic"],
        "first_stage_partial_r_squared": native_first["partial_r_squared"],
        "weak_first_stage_f": weak.extra["first_stage"][0]["f_statistic"],
        "weak_iv_schooling": weak_c.estimate,
        "weak_iv_se": weak_c.std_error,
        "weak_iv_ci_low": weak_c.ci_low,
        "weak_iv_ci_high": weak_c.ci_high,
        "invalid_iv_schooling": bad_c.estimate,
        "invalid_iv_se": bad_c.std_error,
        "invalid_iv_ci_low": bad_c.ci_low,
        "invalid_iv_ci_high": bad_c.ci_high,
        "inference_df": iv.inference["df_inference"],
        "max_coefficient_error": max(e[0] for e in errors),
        "max_covariance_error": max(e[1] for e in errors),
    }
    checks = {
        "independent_three_2sls_coefficients": summary["max_coefficient_error"] < 1e-8,
        "independent_three_2sls_sandwiches": summary["max_covariance_error"] < 1e-7,
        "native_first_stage_matches_ols_wald": abs(
            native_first["f_statistic"] - first_f["statistic"]
        )
        < 1e-7,
        "same_sample_ols_iv_firststage": iv.sample_positions
        == ols.sample_positions
        == first.sample_positions
        == list(range(len(data))),
        "explicit_small_sample_t": iv.inference["df_inference"] == len(data) - 3
        and iv.inference["use_t"],
    }
    chart_data = {
        "figure": {
            "kind": "interval",
            "title": "Strong, weak and invalid instruments answer different concerns",
            "xlabel": "Log earnings per schooling year",
            "ylabel": "Specification",
            "series": [
                {
                    "label": "95% conventional robust intervals",
                    "x": [strong_c.estimate, weak_c.estimate, bad_c.estimate],
                    "y": [
                        "Strong valid offer",
                        "Weak valid offer",
                        "Strong offer with direct effect",
                    ],
                    "low": [strong_c.ci_low, weak_c.ci_low, bad_c.ci_low],
                    "high": [strong_c.ci_high, weak_c.ci_high, bad_c.ci_high],
                }
            ],
        },
        "first_stage": native_first,
    }
    displays = [iv, first, weak, invalid]
    return complete(
        "13-instrumental-variables",
        "Estimating Effects with an Instrument",
        data,
        summary,
        {
            "OLS descriptive": ols,
            "Strong valid IV": iv,
            "Weak valid IV": weak,
            "Invalid exclusion IV": invalid,
            "Strong first stage": first,
        },
        chart_data,
        checks,
        output_dir,
        display_callback,
        displays,
    )


if __name__ == "__main__":
    instrumental_variables_lab_result = run_lab(display_callback=globals().get("display"))
