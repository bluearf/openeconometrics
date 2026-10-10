# Instructor companion · Lab 14

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** One per unit: 180−150=30.

**B.** Consecutive temporal changes must be formed within the correct unit.

**C.** Yes. Adjacent differences share the same level error with opposite signs.

**D.** No. Equality is special for two-period panels under matching conventions; longer panels use different transformations.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/14-first-differences/first_differences.xlsx) for the published analysis. The [original generator](generators/14-first-differences.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/14-first-differences/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/14-first-differences/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: one first period lost per unit, positive remaining time, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Differencing can amplify measurement error and does not remove time-varying confounding. With more than two periods its efficiency and estimand weighting can differ from within estimation.
