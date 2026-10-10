# Instructor companion · Lab 20

## Teaching sequence

Ask groups to write a short result paragraph including raw and retained counts, contrast direction, interval, multiplicity convention and missingness limitation.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Spending: 240−6=234, comprising 120 A and 114 B. Return: all 240.

**B.** Observed A spending is -2.15781 currency relative to B, with interval [-3.64342, -0.672206]. The negative sign means observed B spending is higher.

**C.** The smaller raw p 0.00460324 is doubled to 0.00920647; the larger 0.673808 remains 0.673808 after monotonicity.

**D.** Differential missing spending can break comparability after assignment. An assumption or missing-outcome sensitivity analysis is needed.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/20-statistical-report/campus_cafe_offer.xlsx) for the published analysis. The [original generator](generators/20-statistical-report.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/20-statistical-report/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/20-statistical-report/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: primary welch se, primary t uses same rows, sample accounting, holm does not reduce p, distinct customer ids. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Randomization as a cure for differential missingness; secondary outcomes sharing the primary denominator; switching the primary outcome after inspecting p values; adjusted and raw p values left unlabeled.
