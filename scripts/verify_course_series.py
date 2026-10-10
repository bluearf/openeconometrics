"""Focused execution, workbook and independent-reference checks for a course.

Uses disposable workspaces and the existing complete-result/native-worker
verifier. This is a teaching-content check, not the full SDK gate.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tempfile

import pandas as pd

import verify_teaching_labs as shared

ROOT = Path(__file__).resolve().parents[1]
COURSES = ("statistics", "microeconomics", "advanced-econometrics")


def configure(course):
    base = ROOT / "docs/teaching" / course
    catalog = json.loads((base / "catalog.json").read_text())
    shared.LABS = {lab["slug"]: lab["display_types"] for lab in catalog["labs"]}
    return base, catalog


def workbook_check(base, folder):
    metadata = json.loads((folder / "dataset.json").read_text())
    source = folder / metadata["filename"]
    namespace = runpy.run_path(str(base / "instructors/generators" / f"{folder.name}.py"))
    expected = pd.DataFrame(namespace["make_data"]())
    with pd.ExcelFile(source) as workbook:
        shared.require(
            workbook.sheet_names == ["Data", "Dictionary"], "Workbook sheet contract differs"
        )
        actual = pd.read_excel(workbook, sheet_name="Data")
        dictionary = pd.read_excel(workbook, sheet_name="Dictionary")
    pd.testing.assert_frame_equal(
        actual, expected, check_dtype=False, check_exact=False, rtol=1e-13, atol=1e-13
    )
    shared.require(actual.isna().equals(expected.isna()), "Missing mask differs")
    shared.require(
        list(dictionary.iloc[: len(expected.columns), 0]) == list(expected.columns),
        "Dictionary column order differs",
    )
    return {
        "rows": len(actual),
        "columns": list(actual),
        "missing_cells": int(actual.isna().sum().sum()),
        "sha256": digest(source),
        "raw_rows_and_order": "passed",
        "tolerance": 1e-13,
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_statistics(folder, summary):
    """Development-only SciPy references, separate from shipped lab arithmetic."""
    import numpy as np
    from scipy import stats

    metadata = json.loads((folder / "dataset.json").read_text())
    data = pd.read_excel(folder / metadata["filename"], sheet_name="Data")
    number = int(folder.name[:2])
    checks = {}

    def equal(label, value, key):
        shared.compare(float(value), summary[key], label)
        checks[label] = True

    if number in (1, 2, 3, 6, 7):
        column = {1: "monthly_spending", 2: "spending", 3: "minutes", 6: "orders", 7: "grams"}[
            number
        ]
        values = data[column].dropna().to_numpy()
        mean_key = {
            1: "mean_spending",
            2: "mean",
            3: "mean_minutes",
            6: "empirical_mean",
            7: "sample_mean_grams",
        }[number]
        equal("numpy_mean", np.mean(values), mean_key)
        if number == 2:
            equal("numpy_median", np.median(values), "median")
        if number == 3:
            equal(
                "numpy_sample_variance", np.var(values, ddof=1), "sample_variance_minutes_squared"
            )
        if number == 6:
            equal("scipy_binomial_tail", stats.binom.sf(4, 10, 0.3), "p_at_least_five")
        if number == 7:
            equal("scipy_normal_tail", stats.norm.sf(1), "model_fraction_above_1012")
    if number in (4, 5):
        first, second = ("member", "purchase") if number == 4 else ("defect", "alert")
        subset = data.loc[data[second] == 1, first]
        equal(
            "conditional_frequency_from_selected_rows",
            subset.mean(),
            "p_member_given_purchase" if number == 4 else "positive_predictive_value",
        )
    if number in (9, 10):
        column, mu = ("minutes", 10) if number == 9 else ("milliliters", 500)
        values = data[column].dropna().to_numpy()
        fit = stats.ttest_1samp(values, mu)
        se = stats.sem(values)
        low, high = stats.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=se)
        if number == 9:
            equal("scipy_mean_interval_low", low, "ci95_low")
            equal("scipy_mean_interval_high", high, "ci95_high")
            low99, high99 = stats.t.interval(0.99, len(values) - 1, loc=values.mean(), scale=se)
            equal("scipy_99_interval_low", low99, "ci99_low")
            equal("scipy_99_interval_high", high99, "ci99_high")
        else:
            equal("scipy_t_statistic", fit.statistic, "t")
            equal("scipy_t_two_sided_p", fit.pvalue, "two_sided_p")
            equal("scipy_mean_interval_low", low, "ci_low_ml")
            equal("scipy_mean_interval_high", high, "ci_high_ml")
    if number in (12, 20):
        by, column = ("method", "minutes") if number == 12 else ("offer", "spending")
        a = data.loc[data[by] == "A", column].dropna().to_numpy()
        b = data.loc[data[by] == "B", column].dropna().to_numpy()
        fit = stats.ttest_ind(a, b, equal_var=False)
        interval = fit.confidence_interval(0.95)
        equal("scipy_welch_t", fit.statistic, "welch_t")
        equal("scipy_welch_df", fit.df, "welch_df")
        equal("scipy_welch_p", fit.pvalue, "welch_p" if number == 12 else "primary_p")
        equal(
            "scipy_welch_interval_low",
            interval.low,
            "welch_ci_low" if number == 12 else "primary_ci_low",
        )
        equal(
            "scipy_welch_interval_high",
            interval.high,
            "welch_ci_high" if number == 12 else "primary_ci_high",
        )
    if number == 13:
        complete = data.dropna(subset=["after", "before"])
        fit = stats.ttest_rel(complete.after, complete.before)
        interval = fit.confidence_interval(0.95)
        equal("scipy_paired_t", fit.statistic, "t")
        equal("scipy_paired_p", fit.pvalue, "two_sided_p")
        equal("scipy_paired_interval_low", interval.low, "ci_low")
        equal("scipy_paired_interval_high", interval.high, "ci_high")
    if number in (11, 14):
        equal("scipy_normal_two_sided_p", 2 * stats.norm.sf(abs(summary["z"])), "two_sided_p")
        if number == 11:
            ci = stats.binomtest(int(data.opt_in.sum()), len(data)).proportion_ci(method="wilson")
            equal("scipy_wilson_low", ci.low, "wilson_low")
            equal("scipy_wilson_high", ci.high, "wilson_high")
    if number == 15:
        complete = data.dropna(subset=["hours", "score"])
        fit = stats.pearsonr(complete.hours, complete.score)
        equal("scipy_pearson_r", fit.statistic, "pearson_r")
        equal("scipy_pearson_p", fit.pvalue, "two_sided_p")
    if number == 16:
        fit = stats.chi2_contingency(pd.crosstab(data.channel, data.member), correction=False)
        equal("scipy_pearson_chi_square", fit.statistic, "pearson_chi_squared")
        equal("scipy_chi_square_p", fit.pvalue, "p_value")
    if number == 17:
        groups = [v.score.to_numpy() for _, v in data.groupby("format")]
        fit = stats.f_oneway(*groups)
        equal("scipy_anova_F", fit.statistic, "F")
        equal("scipy_anova_p", fit.pvalue, "p_value")
    if number == 18:
        a, b = [v.rating.to_numpy() for _, v in data.groupby("service")]
        fit = stats.mannwhitneyu(a, b, method="asymptotic", use_continuity=False)
        equal("scipy_midrank_U", fit.statistic, "U_A")
        equal("scipy_tie_corrected_p", fit.pvalue, "asymptotic_two_sided_p")
    if number in (8, 19):
        values = data.spending.to_numpy()
        key = "theoretical_se_n40" if number == 8 else "empirical_distribution_exact_se"
        equal(
            "numpy_empirical_distribution_standard_error",
            np.std(values, ddof=0) / np.sqrt(40 if number == 8 else len(values)),
            key,
        )
    shared.require(bool(checks), "No independent scientific reference was exercised")
    return checks


def snippet_check(folder):
    namespace = runpy.run_path(str(folder / "lab.py"))
    namespace["display"] = lambda value: None
    snippets = re.findall(r"```python\n(.*?)```", (folder / "README.md").read_text(), re.S)
    shared.require(bool(snippets), "Student application snippet is missing")
    with contextlib.redirect_stdout(io.StringIO()):
        for snippet in snippets:
            exec(compile(snippet, str(folder / "README.md"), "exec"), namespace)
    return {"executed_python_snippets": len(snippets)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", choices=COURSES, required=True)
    parser.add_argument("--labs", nargs="+")
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-receipt", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    base, catalog = configure(args.course)
    if args.worker:
        shared.standalone_worker(args.worker.resolve(), args.worker_receipt.resolve())
        return
    names = args.labs or [lab["slug"] for lab in catalog["labs"]]
    shared.require(
        args.labs is not None or len(names) == catalog["expected_labs"], "Incomplete course"
    )
    records = []
    for name in names:
        folder = base / "labs" / name
        text = (folder / "README.md").read_text()
        from teaching_content import assert_student_content

        assert_student_content(text, name)
        shared.require(
            (base / "instructors" / f"{name}.md").is_file(), "Instructor companion missing"
        )
        with tempfile.TemporaryDirectory(prefix=f"course-{args.course}-") as temporary:
            empty = Path(temporary) / "empty"
            empty.mkdir()
            receipt = Path(temporary) / "receipt.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--course",
                    args.course,
                    "--worker",
                    str(folder),
                    "--worker-receipt",
                    str(receipt),
                ],
                cwd=empty,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=90,
            )
            shared.require(completed.returncode == 0, completed.stdout + completed.stderr)
            standalone = json.loads(receipt.read_text())
        reference = json.loads((folder / "reference.json").read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            console = shared.verify_console(folder, reference)
        if args.course == "statistics":
            independent = independent_statistics(folder, reference["summary"])
        else:
            from teaching_references import microeconomics, advanced_econometrics

            independent = (
                microeconomics if args.course == "microeconomics" else advanced_econometrics
            )(folder, reference)
        records.append(
            {
                "lab": name,
                "status": "passed",
                "source_sha256": digest(folder / "lab.py"),
                "reference_sha256": digest(folder / "reference.json"),
                "workbook": workbook_check(base, folder),
                "standalone": standalone,
                "console": console,
                "independent_reference": independent,
                "student_snippets": snippet_check(folder),
            }
        )
        print(name, "passed", flush=True)
    destination = args.output_report or ROOT / f"artifacts/teaching/{args.course}-verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "status": "passed",
                "course": args.course,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "labs_passed": len(records),
                "numerical_tolerance": shared.TOLERANCE,
                "labs": records,
            },
            indent=2,
        )
        + "\n"
    )
    print("Verification receipt:", destination)


if __name__ == "__main__":
    main()
