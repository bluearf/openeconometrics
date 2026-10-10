"""Lab 15: an original, fully synthetic common-adoption DiD example.

Run this complete file after importing policy_panel.xlsx into OpenEconometrics,
or keep the workbook beside this file in ordinary Python. No files are written
unless an explicit output directory is given.
"""

import json
from pathlib import Path

import torch

import openecon as oe


DATA_FILE = 'policy_panel.xlsx'
POLICY_YEAR = 2018
TRUE_EFFECT = 3.0
CONCURRENT_SHOCK = 2.0
N_REGIONS = 60
YEARS = list(range(2015, 2021))


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


def coefficient(model, term):
    return next(row for row in model.coefficients if row.term == term)


def manual_cluster_check(data):
    """Independent balanced-panel demeaning and scalar sandwich calculation.

    Group effects are nested in the 60 clusters. With one treatment slope and
    six time effects, the didregress correction counts K=7, rather than all
    region dummies. No OpenEconometrics fitting/covariance helper is called.
    """
    dtype = torch.float64
    shape = (N_REGIONS, len(YEARS))
    outcome = torch.tensor(data.employment_rate.to_list(), dtype=dtype).reshape(shape)
    treatment = torch.tensor(data.treated_post.to_list(), dtype=dtype).reshape(shape)

    def within(values):
        return values - values.mean(1, keepdim=True) - values.mean(0, keepdim=True) + values.mean()

    y_within, d_within = within(outcome), within(treatment)
    bread_inverse = (d_within * d_within).sum()
    estimate = (d_within * y_within).sum() / bread_inverse
    residuals = y_within - estimate * d_within
    cluster_scores = (d_within * residuals).sum(1)
    n = len(data)
    k = 1 + len(YEARS)
    adjustment = N_REGIONS / (N_REGIONS - 1) * (n - 1) / (n - k)
    variance = adjustment * (cluster_scores**2).sum() / bread_inverse**2
    return {"estimate": float(estimate), "std_error": float(variance.sqrt()), "k": k}


def group_means(data):
    means = data.groupby(["year", "treated"]).employment_rate.mean().unstack("treated")
    return [{
        "year": int(year),
        "control": float(row[0]),
        "treated": float(row[1]),
        "gap": float(row[1] - row[0]),
    } for year, row in means.iterrows()]


def run_lab(output_dir=None, display_callback=None, data_path=None):
    """Fit the policy and identification-failure examples; return complete state."""
    data = load_data(data_path)
    did = oe.didregress(
        data=data, y="employment_rate", treatment="treated_post",
        group="region", time="year", covariance="cluster", cluster="region",
        missing="raise", alpha=0.05,
    )
    interaction = oe.ols(
        data=data, y="employment_rate", x=["treated", "post", "treated_post"],
        covariance="cluster", cluster="region", missing="raise", alpha=0.05,
    )
    unclustered = oe.didregress(
        data=data, y="employment_rate", treatment="treated_post",
        group="region", time="year", covariance="HC1", missing="raise", alpha=0.05,
    )

    cells = data.groupby(["treated", "post"]).employment_rate.mean()
    control_pre, control_post = float(cells[0, 0]), float(cells[0, 1])
    treated_pre, treated_post = float(cells[1, 0]), float(cells[1, 1])
    control_change = control_post - control_pre
    treated_change = treated_post - treated_pre
    four_cell_did = treated_change - control_change
    post_gap = treated_post - control_post
    effect = coefficient(did, "ATET:r1vs0.treated_post")
    interaction_effect = coefficient(interaction, "treated_post")
    unclustered_effect = coefficient(unclustered, "ATET:r1vs0.treated_post")
    manual = manual_cluster_check(data)

    # An unrelated, treated-only 2018 shock is observationally indistinguishable
    # from policy treatment in this design. Its timing leaves all pre-data intact.
    contaminated = data.copy()
    contaminated["employment_rate"] += CONCURRENT_SHOCK * contaminated.treated_post
    confounded = oe.didregress(
        data=contaminated, y="employment_rate", treatment="treated_post",
        group="region", time="year", covariance="cluster", cluster="region",
        missing="raise", alpha=0.05,
    )
    confounded_effect = coefficient(confounded, "ATET:r1vs0.treated_post")

    checks = {
        "balanced_region_year_panel": bool(
            len(data) == N_REGIONS * len(YEARS)
            and not data.duplicated(["region", "year"]).any()
            and data.groupby("region").size().eq(len(YEARS)).all()
        ),
        "no_missing_observations": bool(not data.isna().any().any()),
        "employment_rates_in_range": bool(data.employment_rate.between(0, 100).all()),
        "same_estimation_sample": bool(
            did.sample_positions == interaction.sample_positions == list(range(len(data)))
            and did.dropped_rows == interaction.dropped_rows == 0
        ),
        "four_cell_abs_error": abs(effect.estimate - four_cell_did),
        "interaction_abs_error": abs(effect.estimate - interaction_effect.estimate),
        "covariance_choice_coefficient_abs_error": abs(effect.estimate - unclustered_effect.estimate),
        "manual_within_abs_error": abs(effect.estimate - manual["estimate"]),
        "manual_cluster_se_abs_error": abs(effect.std_error - manual["std_error"]),
        "cluster_df_matches_assignment_units": did.inference["df_inference"] == N_REGIONS - 1,
        "concurrent_shock_abs_error": abs(confounded_effect.estimate - effect.estimate - CONCURRENT_SHOCK),
        "parallel_trends_p_unchanged_abs_error": abs(
            did.tests["parallel_trends"]["p_value"] - confounded.tests["parallel_trends"]["p_value"]
        ),
        "granger_p_unchanged_abs_error": abs(
            did.tests["granger"]["p_value"] - confounded.tests["granger"]["p_value"]
        ),
    }
    for name, value in checks.items():
        if isinstance(value, bool):
            assert value, name
        else:
            assert value < 1e-8, (name, value)

    summary = {
        "nobs": did.nobs,
        "regions": N_REGIONS,
        "treated_regions": N_REGIONS // 2,
        "control_regions": N_REGIONS // 2,
        "periods": len(YEARS),
        "pre_periods": 3,
        "post_periods": 3,
        "policy_year": POLICY_YEAR,
        "true_effect": TRUE_EFFECT,
        "control_pre": control_pre,
        "control_post": control_post,
        "treated_pre": treated_pre,
        "treated_post": treated_post,
        "control_change": control_change,
        "treated_change": treated_change,
        "post_only_gap": post_gap,
        "did_estimate": effect.estimate,
        "did_std_error": effect.std_error,
        "did_ci_low": effect.ci_low,
        "did_ci_high": effect.ci_high,
        "did_p_value": effect.p_value,
        "hc1_std_error": unclustered_effect.std_error,
        "hc1_ci_low": unclustered_effect.ci_low,
        "hc1_ci_high": unclustered_effect.ci_high,
        "df_inference": did.inference["df_inference"],
        "parallel_trends_p_value": did.tests["parallel_trends"]["p_value"],
        "granger_p_value": did.tests["granger"]["p_value"],
        "confounded_estimate": confounded_effect.estimate,
        "confounded_std_error": confounded_effect.std_error,
        "confounded_ci_low": confounded_effect.ci_low,
        "confounded_ci_high": confounded_effect.ci_high,
        "concurrent_shock": CONCURRENT_SHOCK,
        "manual_covariance_k": manual["k"],
    }
    result = {
        "metadata": {
            "lab_id": "15", "title": "Before and After a Policy Reform",
            "data_kind": "original synthetic balanced region-year panel",
            "data_source": DATA_FILE,
            "outcome_unit": "employment rate, percent of working-age population",
            "effect_unit": "percentage points", "covariance": "cluster",
            "cluster": "region", "inference_distribution": "Student t(59)",
            "missing": "raise", "alpha": 0.05, "openecon_version": oe.__version__,
            "policy_year": POLICY_YEAR, "runtime_dependencies": ["openecon", "pandas", "torch"],
        },
        "summary": summary,
        "models": {
            "did": did.model_dump(mode="json"),
            "interaction": interaction.model_dump(mode="json"),
            "unclustered_did": unclustered.model_dump(mode="json"),
            "confounded_did": confounded.model_dump(mode="json"),
        },
        "chart_data": {"group_means": group_means(data), "confounded_group_means": group_means(contaminated)},
        "checks": checks,
    }

    table = oe.regression_table(
        {"Policy design": did, "Concurrent shock": confounded}, precision=3, stars=False,
        caption="Synthetic policy illustration: clustered difference-in-differences",
        label="tab:lab15-did", term_labels={"ATET:r1vs0.treated_post": "Treated x post (pp)"},
        notes=["Original synthetic data; not evidence about an actual policy.",
               "Region and year effects absorbed; 60 region clusters; t(59) inference.",
               "The second outcome adds an unrelated two-point treated-only post-policy shock."],
    )
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "reference.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (destination / "table.tex").write_text(str(table) + "\n", encoding="utf-8")

    if display_callback is not None:
        display_callback(oe.DataFrame([
            {"group": "Never treated", "pre": control_pre, "post": control_post, "change": control_change},
            {"group": "Treated", "pre": treated_pre, "post": treated_post, "change": treated_change},
        ]))
        display_callback(did)
        display_callback(oe.plot.line(
            data=result["chart_data"]["group_means"], x="year", y="gap",
            title="Synthetic treated-minus-control employment gap (policy begins in 2018)",
        ))
        display_callback(confounded)
    else:
        print(did.summary())
        print(f"Four-cell DiD: {four_cell_did:.6f} pp; true simulated effect: {TRUE_EFFECT:.1f} pp")
        print(f"Concurrent-shock estimate: {confounded_effect.estimate:.6f} pp")
    return result


if __name__ == "__main__":
    lab_result = run_lab(
        output_dir=globals().get("LAB_OUTPUT_DIR"),
        display_callback=globals().get("display"),
    )
