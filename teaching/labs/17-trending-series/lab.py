"""Original independent random walks: levels, differences and native unit-root tests."""

import json
from pathlib import Path

import torch

import openecon as oe

DATA_FILE = 'trending_series.xlsx'


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


def manual_adf(series):
    """ADF(1), constant+trend: independent OLS t ratio, no DF p-value substitution."""
    y = torch.tensor(series.tolist(), dtype=torch.float64)
    changes = y[1:] - y[:-1]
    response = changes[1:]
    x = torch.stack(
        [y[1:-1], changes[:-1], torch.arange(2, len(y), dtype=y.dtype), torch.ones_like(response)],
        dim=1,
    )
    inv = torch.linalg.inv(x.T @ x)
    beta = torch.linalg.lstsq(x, response).solution
    resid = response - x @ beta
    variance = (resid.square().sum() / (len(response) - x.shape[1])) * inv
    return float(beta[0] / variance[0, 0].sqrt())


def table_state(table):
    return {
        "columns": table.columns.tolist(),
        "records": table.to_dict(orient="records"),
        "attrs": table.attrs,
    }


def run_lab(output_dir=None, display_callback=None, data_path=None):
    data = load_data(data_path)
    differences = (
        data.assign(change_x=data.index_x.diff(), change_y=data.index_y.diff()).iloc[1:].copy()
    )
    differences = differences.reset_index(drop=True)
    levels = oe.ols(
        data=data, y="index_y", x=["index_x"], covariance="nonrobust", missing="raise", device="cpu"
    )
    changes = oe.ols(
        data=differences,
        y="change_y",
        x=["change_x"],
        covariance="HC3",
        missing="raise",
        device="cpu",
    )
    adf_level = oe.dfuller(data, "index_y", time="quarter", lags=1, trend="trend")
    adf_change = oe.dfuller(differences, "change_y", time="quarter", lags=1, trend="constant")
    kpss_level = oe.kpss(data, "index_y", time="quarter", lags=4, trend=True)
    kpss_change = oe.kpss(differences, "change_y", time="quarter", lags=4, trend=False)
    level = next(c for c in levels.coefficients if c.term == "index_x")
    change = next(c for c in changes.coefficients if c.term == "change_x")
    xx = torch.tensor(data.index_x.tolist(), dtype=torch.float64)
    yy = torch.tensor(data.index_y.tolist(), dtype=torch.float64)
    manual_slope = float(
        ((xx - xx.mean()) * (yy - yy.mean())).sum() / (xx - xx.mean()).square().sum()
    )
    checks = {
        "level_slope_error": abs(level.estimate - manual_slope),
        "adf_statistic_error": abs(adf_level.attrs["statistic"] - manual_adf(data.index_y)),
        "levels_sample": levels.sample_positions == list(range(240)) and levels.nobs == 240,
        "differences_sample": changes.sample_positions == list(range(239)) and changes.nobs == 239,
        "difference_calendar": differences.quarter.tolist() == list(range(2, 241)),
        "adf_observation_count": adf_level.attrs["nobs"] == 238 and adf_change.attrs["nobs"] == 237,
        "different_null_distributions": adf_level.attrs["distribution"] == "Dickey-Fuller",
    }
    for key, value in checks.items():
        assert value if isinstance(value, bool) else value < 1e-7, (key, value)
    diagnostics = {
        "adf_level": table_state(adf_level),
        "adf_difference": table_state(adf_change),
        "kpss_level": table_state(kpss_level),
        "kpss_difference": table_state(kpss_change),
    }
    panels = [
        {
            "kind": "line",
            "title": "Independent synthetic random walks with drift",
            "xlabel": "Quarter",
            "ylabel": "Index level",
            "series": [
                {"label": name, "x": data.quarter.tolist(), "y": data[name].tolist()}
                for name in ("index_x", "index_y")
            ],
        },
        {
            "kind": "scatter",
            "title": "Changes remove the accumulated history",
            "xlabel": "Change in X",
            "ylabel": "Change in Y",
            "series": [
                {
                    "label": "Quarterly changes",
                    "x": differences.change_x.tolist(),
                    "y": differences.change_y.tolist(),
                }
            ],
        },
    ]
    result = {
        "metadata": {
            "lab_id": "17",
            "title": "The Trap of Trending Series",
            "data_source": DATA_FILE,
            "data_kind": "original synthetic independent random walks with drift",
            "level_inference_warning": "usual stationary-regression reference distribution is invalid here",
            "difference_first_original_quarter": 2,
            "openecon_version": oe.__version__,
        },
        "summary": {
            "nobs": 240,
            "difference_nobs": 239,
            "level_slope": level.estimate,
            "level_se": level.std_error,
            "level_p_value": level.p_value,
            "level_r_squared": levels.metrics["r_squared"],
            "difference_slope": change.estimate,
            "difference_se": change.std_error,
            "difference_p_value": change.p_value,
            "difference_ci_low": change.ci_low,
            "difference_ci_high": change.ci_high,
            "adf_level_statistic": adf_level.attrs["statistic"],
            "adf_level_p_value": adf_level.attrs["p_value"],
            "adf_difference_statistic": adf_change.attrs["statistic"],
            "adf_difference_p_value": adf_change.attrs["p_value"],
            "kpss_level_statistic": kpss_level.attrs["statistic"],
            "kpss_difference_statistic": kpss_change.attrs["statistic"],
        },
        "models": {
            "levels": levels.model_dump(mode="json"),
            "differences": changes.model_dump(mode="json"),
        },
        "diagnostics": diagnostics,
        "chart_data": {"panels": panels},
        "checks": checks,
    }
    table = oe.regression_table(
        {"Levels (spurious)": levels, "First differences": changes},
        precision=3,
        stars=False,
        caption="Independent synthetic series: levels versus changes",
        notes=[
            "Different outcomes and samples; coefficients are not the same estimand.",
            "The usual level-regression significance is misleading for independent nonstationary series.",
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
        display_callback(levels)
        display_callback(changes)
        display_callback(adf_level)
        display_callback(adf_change)
        display_callback(kpss_level)
        display_callback(kpss_change)
        display_callback(
            oe.plot.scatter(
                data=differences,
                x="change_x",
                y="change_y",
                title="Independent synthetic quarterly changes",
            )
        )
    else:
        print(levels.summary())
        print(changes.summary())
        print(adf_level.to_string(index=False))
        print(adf_change.to_string(index=False))
    return result


if __name__ == "__main__":
    trends_lab_result = run_lab(globals().get("LAB_OUTPUT_DIR"), globals().get("display"))
