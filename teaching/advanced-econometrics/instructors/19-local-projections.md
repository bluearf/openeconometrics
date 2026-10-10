# Instructor companion · Lab 19

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** 0.8×0.5²=0.2.

**B.** Each extra lead loses one final outcome; the first lag-initialization row is also dropped.

**C.** No. Native coefficient intervals are pointwise for separate regressions.

**D.** Not without an additional identification strategy for the shock.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/19-local-projections/local_projections.xlsx) for the published analysis. The [original generator](generators/19-local-projections.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/19-local-projections/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/19-local-projections/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: one tail row lost each horizon, same hac convention, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** An observed exposure is not automatically an exogenous shock. Pointwise significance across horizons does not provide simultaneous coverage, and horizon/sample conventions must be reported.
