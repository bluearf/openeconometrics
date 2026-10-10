# Instructor companion · Lab 04

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** No. Use the full covariance expression; the sum SE is 0.0645731.

**B.** No. Correlation 0.999558 is below one in magnitude.

**C.** Observed X and Z move together, so their combined contribution can be identified more precisely than either separate slope.

**D.** No. It concerns identification strength/precision within the design; exogeneity is a different issue.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/04-collinearity/collinearity.xlsx) for the published analysis. The [original generator](generators/04-collinearity.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/04-collinearity/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/04-collinearity/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: full rank not exact collinearity, sum variance uses covariance, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** High collinearity does not itself cause endogeneity, and deleting a substantively required control can introduce bias. Precision depends on the contrast of interest and where predictions are made.
