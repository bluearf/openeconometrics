"""Original synthetic persistent price and demand series; gap-aware HAC inference."""

import json
from pathlib import Path

import pandas as pd
import torch

import openecon as oe

DATA_FILE = 'advertising_and_sales.xlsx'
LAGS = 4


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


def manual_hac(data, lags=LAGS):
    """Independent OLS sandwich using pairs whose CALENDAR distance equals lag."""
    x = torch.stack(
        [
            torch.ones(len(data), dtype=torch.float64),
            torch.tensor(data.advertising.tolist(), dtype=torch.float64),
        ],
        dim=1,
    )
    y = torch.tensor(data.sales.tolist(), dtype=torch.float64)
    inverse = torch.linalg.inv(x.T @ x)
    beta = torch.linalg.lstsq(x, y).solution
    residual = y - x @ beta
    scores = x * residual[:, None]
    meat = scores.T @ scores
    lookup = {int(t): i for i, t in enumerate(data.quarter)}
    for lag in range(1, lags + 1):
        cross = torch.zeros((2, 2), dtype=torch.float64)
        for t, i in lookup.items():
            if t + lag in lookup:
                cross += torch.outer(scores[i], scores[lookup[t + lag]])
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    covariance = len(data) / (len(data) - 2) * inverse @ meat @ inverse
    return {
        "beta": beta.tolist(),
        "covariance": covariance.tolist(),
        "residuals": residual.tolist(),
    }


def run_lab(output_dir=None, display_callback=None, data_path=None):
    data = load_data(data_path)
    models = {
        "HC1": oe.ols(
            data=data, y="sales", x=["advertising"], covariance="HC1", missing="raise", device="cpu"
        )
    }
    for lag in (4, 8):
        models[f"HAC{lag}"] = oe.ols(
            data=data,
            y="sales",
            x=["advertising"],
            covariance="hac",
            time="quarter",
            lags=lag,
            kernel="bartlett",
            missing="raise",
            device="cpu",
        )
    manual = manual_hac(data)
    main = models["HAC4"]
    covariance = torch.tensor(main.covariance_matrix, dtype=torch.float64)
    expected = torch.tensor(manual["covariance"], dtype=torch.float64)
    rows = {
        name: next(c for c in fit.coefficients if c.term == "advertising")
        for name, fit in models.items()
    }
    checks = {
        "manual_coefficient_error": abs(rows["HAC4"].estimate - manual["beta"][1]),
        "manual_covariance_max_error": float((covariance - expected).abs().max()),
        "covariance_choice_coefficient_error": max(
            abs(c.estimate - rows["HAC4"].estimate) for c in rows.values()
        ),
        "same_sample": all(
            fit.sample_positions == list(range(197)) and fit.nobs == 197 for fit in models.values()
        ),
        "actual_calendar_preserved": data.quarter.iloc[0] == 1
        and data.quarter.iloc[-1] == 200
        and not data.quarter.isin([40, 41, 95]).any(),
        "t_df": main.inference["df_inference"] == 195,
    }
    for key, value in checks.items():
        assert bool(value) if not isinstance(value, float) else value < 1e-8, (key, value)
    comparison = [
        {
            "covariance": name,
            "estimate": row.estimate,
            "std_error": row.std_error,
            "ci_low": row.ci_low,
            "ci_high": row.ci_high,
        }
        for name, row in rows.items()
    ]
    intervals = {
        "kind": "interval",
        "title": "One OLS coefficient, three uncertainty calculations",
        "xlabel": "Covariance specification",
        "ylabel": "Sales units per advertising unit",
        "series": [
            {
                "label": "Estimate and 95% interval",
                "x": ["HC1", "HAC(4)", "HAC(8)"],
                "y": [row["estimate"] for row in comparison],
                "low": [row["ci_low"] for row in comparison],
                "high": [row["ci_high"] for row in comparison],
            }
        ],
    }
    residuals = {
        "kind": "line",
        "title": "Persistent residual shocks in the observed calendar",
        "xlabel": "Quarter (40, 41 and 95 unavailable)",
        "ylabel": "OLS residual",
        "series": [{"label": "Residual", "x": data.quarter.tolist(), "y": manual["residuals"]}],
    }
    result = {
        "metadata": {
            "lab_id": "18",
            "title": "Accounting for Persistent Shocks",
            "data_source": DATA_FILE,
            "data_kind": "original synthetic quarterly advertising and sales",
            "calendar_exclusions": [40, 41, 95],
            "kernel": "bartlett",
            "lags": 4,
            "small_sample_correction": "N/(N-K)",
            "missing": "raise",
            "openecon_version": oe.__version__,
        },
        "summary": {
            "nobs": 197,
            "calendar_span": 200,
            "true_slope": 1.4,
            "slope": rows["HAC4"].estimate,
            "hc1_se": rows["HC1"].std_error,
            "hac4_se": rows["HAC4"].std_error,
            "hac8_se": rows["HAC8"].std_error,
            "hac4_ci_low": rows["HAC4"].ci_low,
            "hac4_ci_high": rows["HAC4"].ci_high,
            "df_inference": 195,
        },
        "models": {name: fit.model_dump(mode="json") for name, fit in models.items()},
        "chart_data": {"panels": [residuals, intervals], "comparison": comparison},
        "checks": checks,
    }
    table = oe.regression_table(
        models,
        precision=3,
        stars=False,
        caption="Synthetic sales: HC1 and calendar-aware HAC",
        notes=[
            "Identical 197 observations; original quarter labels retained after three explicit exclusions.",
            "Bartlett HAC uses the declared calendar distance; Student t(195) intervals.",
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
        display_callback(main)
        display_callback(oe.DataFrame(comparison))
        display_callback(
            oe.plot.line(
                data={"quarter": data.quarter.tolist(), "residual": manual["residuals"]},
                x="quarter",
                y="residual",
                title=residuals["title"],
            )
        )
    else:
        print(main.summary())
        print(pd.DataFrame(comparison).to_string(index=False))
    return result


if __name__ == "__main__":
    hac_lab_result = run_lab(globals().get("LAB_OUTPUT_DIR"), globals().get("display"))
