"""Original synthetic sharp RD, fixed bandwidths and bias-corrected inference."""

import json
from pathlib import Path

import pandas as pd
import torch

import openecon as oe

DATA_FILE = 'scholarship_cutoff.xlsx'
CUTOFF = 50.0
TRUE_EFFECT = 3.0


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


def manual_rd(data, h=12.0, b=18.0):
    """Independent local-polynomial normal equations and HC0 influence variance."""
    x = torch.tensor(data.score.tolist(), dtype=torch.float64) - CUTOFF
    y = torch.tensor(data.credits.tolist(), dtype=torch.float64)
    sides = []
    for side in (x < 0, x >= 0):
        keep = side & (x.abs() < max(h, b))
        z, outcome = x[keep], y[keep]
        r = torch.stack([torch.ones_like(z), z], dim=1)
        rq = torch.stack([torch.ones_like(z), z, z.square()], dim=1)
        wh, wb = (1 - z.abs() / h).clamp_min(0) / h, (1 - z.abs() / b).clamp_min(0) / b
        inv_p = torch.linalg.inv(r.T @ (wh[:, None] * r))
        inv_q = torch.linalg.inv(rq.T @ (wb[:, None] * rq))
        beta_p = inv_p @ (r.T @ (wh * outcome))
        beta_q = inv_q @ (rq.T @ (wb * outcome))
        moment = (r * wh[:, None]).T @ (z / h).square()
        projection = wb * (rq @ inv_q[:, 2])
        qrows = r * wh[:, None] - h**2 * projection[:, None] * moment[None, :]
        beta_bc = inv_p @ (qrows.T @ outcome)
        influence_p = (r * wh[:, None]) @ inv_p[:, 0]
        influence_bc = qrows @ inv_p[:, 0]
        sides.append(
            {
                "conventional": float(beta_p[0]),
                "robust": float(beta_bc[0]),
                "variance_conventional": float(
                    ((outcome - r @ beta_p) * influence_p).square().sum()
                ),
                "variance_robust": float(((outcome - rq @ beta_q) * influence_bc).square().sum()),
            }
        )
    return {
        "conventional": sides[1]["conventional"] - sides[0]["conventional"],
        "robust": sides[1]["robust"] - sides[0]["robust"],
        "se_conventional": (sides[0]["variance_conventional"] + sides[1]["variance_conventional"])
        ** 0.5,
        "se_robust": (sides[0]["variance_robust"] + sides[1]["variance_robust"]) ** 0.5,
    }


def run_lab(output_dir=None, display_callback=None, data_path=None):
    data = load_data(data_path)
    models = {}
    for label, h, b in (("main", 12.0, 18.0), ("narrow", 8.0, 12.0), ("wide", 16.0, 24.0)):
        models[label] = oe.rdrobust(
            data=data,
            y="credits",
            running="score",
            cutoff=CUTOFF,
            h=h,
            b=b,
            p=1,
            q=2,
            kernel="triangular",
            vce="hc0",
            missing="raise",
            alpha=0.05,
        )
    main = models["main"]
    rows = {row.term: row for row in main.coefficients}
    robust, conventional = rows["Robust"], rows["Conventional"]
    manual = manual_rd(data)
    checks = {
        "conventional_estimate_error": abs(conventional.estimate - manual["conventional"]),
        "conventional_se_error": abs(conventional.std_error - manual["se_conventional"]),
        "robust_estimate_error": abs(robust.estimate - manual["robust"]),
        "robust_se_error": abs(robust.std_error - manual["se_robust"]),
        "sharp_assignment": bool((data.awarded == (data.score >= CUTOFF).astype(int)).all()),
        "full_sample_retained": main.nobs == 600 and main.sample_positions == list(range(600)),
        "local_counts": (
            main.metrics["n_h_left"] == int(((data.score < 50) & (data.score > 38)).sum())
            and main.metrics["n_h_right"] == int(((data.score >= 50) & (data.score < 62)).sum())
        ),
        "rbc_interval_width": robust.ci_low < robust.estimate < robust.ci_high,
    }
    for key, value in checks.items():
        assert value if isinstance(value, bool) else value < 1e-8, (key, value)
    sensitivity = []
    for label, model in models.items():
        row = next(c for c in model.coefficients if c.term == "Robust")
        sensitivity.append(
            {
                "specification": label,
                "h": model.metrics["h_left"],
                "b": model.metrics["b_left"],
                "estimate": row.estimate,
                "std_error": row.std_error,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "local_n": int(model.metrics["n_h_left"] + model.metrics["n_h_right"]),
            }
        )
    figure = {
        "kind": "scatter",
        "title": "A synthetic grant threshold and first-year credits",
        "xlabel": "Eligibility score",
        "ylabel": "Credits completed",
        "series": [
            {
                "label": label,
                "x": data.loc[mask, "score"].tolist(),
                "y": data.loc[mask, "credits"].tolist(),
            }
            for label, mask in (("Not awarded", data.awarded == 0), ("Awarded", data.awarded == 1))
        ],
    }
    result = {
        "metadata": {
            "lab_id": "16",
            "title": "What Happens Around an Eligibility Cutoff?",
            "data_source": DATA_FILE,
            "data_kind": "original synthetic independent students",
            "cutoff": CUTOFF,
            "h": 12.0,
            "b": 18.0,
            "p": 1,
            "q": 2,
            "kernel": "triangular",
            "vce": "hc0",
            "inference": "normal RBC",
            "missing": "raise",
            "openecon_version": oe.__version__,
        },
        "summary": {
            "nobs": 600,
            "true_effect": TRUE_EFFECT,
            "conventional_estimate": conventional.estimate,
            "conventional_se": conventional.std_error,
            "robust_estimate": robust.estimate,
            "robust_se": robust.std_error,
            "robust_ci_low": robust.ci_low,
            "robust_ci_high": robust.ci_high,
            "robust_p_value": robust.p_value,
            "local_left": int(main.metrics["n_h_left"]),
            "local_right": int(main.metrics["n_h_right"]),
            "h": 12.0,
            "b": 18.0,
        },
        "models": {key: model.model_dump(mode="json") for key, model in models.items()},
        "chart_data": {"figure": figure, "bandwidth_sensitivity": sensitivity},
        "checks": checks,
    }
    table = oe.regression_table(
        {"h=12, b=18": main},
        precision=3,
        stars=False,
        caption="Synthetic sharp RD: conventional and robust bias-corrected inference",
        notes=[
            "Fixed bandwidths; independent students; triangular kernel; HC0 variance.",
            "The three rows are alternative estimates/inference for one discontinuity.",
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
        display_callback(oe.plot.scatter(data=data, x="score", y="credits", title=figure["title"]))
        display_callback(main)
        display_callback(oe.DataFrame(sensitivity))
    else:
        print(main.summary())
        print(pd.DataFrame(sensitivity).to_string(index=False))
    return result


if __name__ == "__main__":
    rd_lab_result = run_lab(globals().get("LAB_OUTPUT_DIR"), globals().get("display"))
