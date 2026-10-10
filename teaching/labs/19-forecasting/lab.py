"""Train-only native AR(1) and a fixed-origin held-out inflation forecast."""

import json
from pathlib import Path


import openecon as oe

DATA_FILE = 'forecast_series.xlsx'
TRAIN_N = 160
HORIZON = 20


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
    train = data.iloc[:TRAIN_N].copy()
    test = data.iloc[TRAIN_N:].copy()
    model = oe.arima(
        data=train,
        y="inflation",
        order=(1, 0, 0),
        time="quarter",
        method="ml",
        covariance="nonrobust",
        missing="raise",
        alpha=0.05,
        ljung_lags=8,
    )
    forecasts = oe.forecast(model, steps=HORIZON, alpha=0.05)
    terms = {c.term: c for c in model.coefficients}
    mean, phi, sigma = (
        terms["Intercept"].estimate,
        terms["ARMA:L1.ar"].estimate,
        terms["/sigma"].estimate,
    )
    last = float(train.inflation.iloc[-1])
    manual_points = [mean + phi**h * (last - mean) for h in range(1, HORIZON + 1)]
    manual_se = [
        sigma * sum(phi ** (2 * j) for j in range(h)) ** 0.5 for h in range(1, HORIZON + 1)
    ]
    evaluation = forecasts.copy()
    evaluation["actual"] = test.inflation.to_list()
    evaluation["train_mean"] = float(train.inflation.mean())
    evaluation["last_value"] = last
    mse = {
        name: float(((evaluation[name] - evaluation.actual) ** 2).mean())
        for name in ("forecast", "train_mean", "last_value")
    }
    checks = {
        "point_recursion_max_error": max(
            abs(a - b) for a, b in zip(forecasts.forecast, manual_points, strict=True)
        ),
        "forecast_se_max_error": max(
            abs(a - b) for a, b in zip(forecasts.std_error, manual_se, strict=True)
        ),
        "training_sample_only": model.nobs == TRAIN_N
        and model.sample_positions == list(range(TRAIN_N)),
        "origin_before_test": int(train.quarter.max()) < int(test.quarter.min()),
        "forecast_calendar": forecasts.period.to_list()
        == test.quarter.to_list()
        == list(range(161, 181)),
        "manual_mse_error": abs(
            mse["forecast"]
            - sum((p - a) ** 2 for p, a in zip(manual_points, test.inflation, strict=True))
            / HORIZON
        ),
        "interval_contains_point": bool(
            (
                (forecasts.ci_low < forecasts.forecast) & (forecasts.forecast < forecasts.ci_high)
            ).all()
        ),
        "stationary_fitted_ar": abs(phi) < 1,
    }
    for key, value in checks.items():
        assert value if isinstance(value, bool) else value < 1e-8, (key, value)
    panels = [
        {
            "kind": "line",
            "title": "A fixed-origin forecast uses no held-out outcomes",
            "xlabel": "Quarter",
            "ylabel": "Synthetic inflation, percent",
            "legend": "above",
            "series": [
                {
                    "label": "Observed training series",
                    "x": train.quarter.iloc[-40:].tolist(),
                    "y": train.inflation.iloc[-40:].tolist(),
                },
                {
                    "label": "Held-out actual",
                    "x": test.quarter.tolist(),
                    "y": test.inflation.tolist(),
                },
                {
                    "label": "AR(1) forecast and 95% prediction interval",
                    "x": forecasts.period.tolist(),
                    "y": forecasts.forecast.tolist(),
                    "low": forecasts.ci_low.tolist(),
                    "high": forecasts.ci_high.tolist(),
                },
            ],
        },
        {
            "kind": "bar",
            "title": "Forecast errors on the same 20 held-out quarters",
            "xlabel": "1: AR(1); 2: training mean; 3: last value",
            "ylabel": "Mean squared error (percentage points squared)",
            "series": [{"label": "Held-out MSE", "x": [1, 2, 3], "y": list(mse.values())}],
        },
    ]
    result = {
        "metadata": {
            "lab_id": "19",
            "title": "Forecasting Inflation Without Looking Ahead",
            "data_source": DATA_FILE,
            "data_kind": "original synthetic stationary inflation series",
            "train_end": 160,
            "test_start": 161,
            "forecast_origin": 160,
            "horizon": HORIZON,
            "evaluation": "one fixed origin, horizons 1 through 20; not 20 rolling one-step forecasts",
            "interval_limit": "normal innovations-only prediction interval; parameter uncertainty excluded",
            "openecon_version": oe.__version__,
        },
        "summary": {
            "nobs": TRAIN_N,
            "test_nobs": HORIZON,
            "mean": mean,
            "ar1": phi,
            "sigma": sigma,
            "first_forecast": float(forecasts.forecast.iloc[0]),
            "first_se": float(forecasts.std_error.iloc[0]),
            "last_forecast": float(forecasts.forecast.iloc[-1]),
            "last_se": float(forecasts.std_error.iloc[-1]),
            "ar1_mse": mse["forecast"],
            "mean_mse": mse["train_mean"],
            "naive_mse": mse["last_value"],
            "training_mean": float(train.inflation.mean()),
            "last_training_value": last,
        },
        "models": {"ar1": model.model_dump(mode="json")},
        "forecast": {"records": forecasts.to_dict(orient="records"), "attrs": forecasts.attrs},
        "chart_data": {"panels": panels, "evaluation": evaluation.to_dict(orient="records")},
        "checks": checks,
    }
    table = oe.regression_table(
        {"Training sample AR(1)": model},
        precision=3,
        stars=False,
        caption="Synthetic inflation: train-only AR(1)",
        notes=[
            "Quarters 1–160 only; exact Gaussian maximum likelihood.",
            "Intercept is the long-run mean, not the recursion constant.",
            "Forecast evaluation uses previously untouched quarters 161–180.",
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
        display_callback(model)
        display_callback(oe.DataFrame(evaluation))
        display_callback(
            oe.plot.line(
                data=forecasts,
                x="period",
                y="forecast",
                title="Synthetic AR(1) dynamic forecast from quarter 160",
            )
        )
        display_callback(
            oe.DataFrame([{"method": name, "mse": value} for name, value in mse.items()])
        )
    else:
        print(model.summary())
        print(evaluation.to_string(index=False))
        print("Held-out MSE:", mse)
    return result


if __name__ == "__main__":
    forecast_lab_result = run_lab(globals().get("LAB_OUTPUT_DIR"), globals().get("display"))
