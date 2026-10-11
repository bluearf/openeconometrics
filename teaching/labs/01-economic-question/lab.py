"""Original synthetic study-design laboratory; run the entire Python document."""

from __future__ import annotations

import json
from pathlib import Path

import openecon as oe


# Original dataset-generation seed (provenance only).
SEED = 1012026
DATA_FILE = 'training_and_wages.xlsx'


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
    cells = []
    for skill in (0, 1):
        for trained in (0, 1):
            selected = data[(data.prior_skill == skill) & (data.trained == trained)]
            cells.append(
                {
                    "prior_skill": skill,
                    "trained": trained,
                    "workers": len(selected),
                    "mean_wage": float(selected.hourly_wage.mean()),
                }
            )
    means = {(row["prior_skill"], row["trained"]): row["mean_wage"] for row in cells}
    group_means = data.groupby("trained").hourly_wage.mean()
    raw = float(group_means[1] - group_means[0])
    low = means[0, 1] - means[0, 0]
    high = means[1, 1] - means[1, 0]
    standardized = 0.5 * low + 0.5 * high
    skill_shares = data.groupby("trained").prior_skill.mean()
    reconstructed = {}
    for trained in (0, 1):
        share = float(skill_shares[trained])
        reconstructed[trained] = (1 - share) * means[0, trained] + share * means[1, trained]
    checks = {
        "all_workers_accounted_for": sum(row["workers"] for row in cells) == 600,
        "four_nonempty_comparison_cells": all(row["workers"] > 0 for row in cells),
        "composition_identity": max(abs(reconstructed[g] - float(group_means[g])) for g in (0, 1))
        < 1e-12,
        "standardization_identity": abs(
            standardized - (0.5 * (means[0, 1] + means[1, 1]) - 0.5 * (means[0, 0] + means[1, 0]))
        )
        < 1e-12,
        "simpson_reversal_in_this_example": raw < 0 < low and high > 0,
        "no_missing_required_values": not data.isna().any().any(),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    summary = {
        "workers": len(data),
        "trained_workers": int(data.trained.sum()),
        "untrained_workers": int((data.trained == 0).sum()),
        "trained_mean_wage": float(group_means[1]),
        "untrained_mean_wage": float(group_means[0]),
        "raw_wage_gap": raw,
        "low_skill_gap": low,
        "high_skill_gap": high,
        "balanced_skill_gap": standardized,
        "trained_high_skill_share": float(skill_shares[1]),
        "untrained_high_skill_share": float(skill_shares[0]),
        "structural_training_effect": 4.0,
    }
    comparisons = [
        {"comparison": "Raw groups", "wage_gap": raw},
        {"comparison": "Equal skill composition", "wage_gap": standardized},
    ]
    result = {
        "metadata": {
            "lab": "01-economic-question",
            "title": "The Economic Question Comes First",
            "seed": SEED,
            "synthetic": True,
            "unit": "worker",
            "wage_unit": "hypothetical currency/hour",
        },
        "summary": summary,
        "models": {},
        "checks": checks,
        "chart_data": {
            "cells": cells,
            "figure": {
                "kind": "bar",
                "title": "Changing the comparison changes the question",
                "xlabel": "Comparison",
                "ylabel": "Trained minus untrained wage (currency/hour)",
                "series": [
                    {
                        "label": "Wage gap",
                        "x": [r["comparison"] for r in comparisons],
                        "y": [r["wage_gap"] for r in comparisons],
                    }
                ],
            },
        },
    }
    table = oe.DataFrame(cells)
    latex = table.to_latex(
        index=False, precision=3, caption="Original synthetic workers: comparison cells"
    )
    if display_callback is not None:
        display_callback(table)
        display_callback(oe.DataFrame(comparisons))
        display_callback(
            oe.plot.bar(
                data=comparisons,
                x="comparison",
                y="wage_gap",
                title="Raw and standardized descriptive contrasts",
            )
        )
    if output_dir is not None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "reference.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (target / "table.tex").write_text(str(latex), encoding="utf-8")
        if json.loads((target / "reference.json").read_text()) != result:
            raise AssertionError("Explicit export changed the complete result.")
    print(
        f"Original synthetic workers: 600. Raw wage gap {raw:.3f}; equal-skill-composition gap {standardized:.3f} currency/hour."
    )
    return result


if __name__ == "__main__":
    lab_result = run_lab(display_callback=globals().get("display"))
