# Instructor companion · Lab 13

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** The intercept plus every unit dummy is exactly collinear.

**B.** Within-unit deviations, giving slope 1.28195.

**C.** 30 units, giving 29 t degrees of freedom.

**D.** No. Time-varying confounders and failures of strict exogeneity remain possible.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/13-fixed-effects-algebra/fixed_effects_algebra.xlsx) for the published analysis. The [original generator](generators/13-fixed-effects-algebra.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/13-fixed-effects-algebra/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/13-fixed-effects-algebra/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: within dummy slope equivalence, balanced unique panel time, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Fixed effects remove time-invariant unit components, not time-varying confounding. Time-invariant regressors cannot be separately identified from unit intercepts.
