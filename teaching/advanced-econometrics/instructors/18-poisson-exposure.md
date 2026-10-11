# Instructor companion · Lab 18

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Expected count doubles at fixed covariates; expected rate does not.

**B.** No. Log exposure enters with coefficient fixed at one.

**C.** The expected rate multiplies by 1.4995 per unit X.

**D.** Not generally. The conditional variance assumption would need revision or a robust procedure.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/18-poisson-exposure/poisson_exposure.xlsx) for the published analysis. The [original generator](generators/18-poisson-exposure.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/18-poisson-exposure/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/18-poisson-exposure/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: positive exposure, count support, intercept score zero, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The crude aggregate rate is not a covariate-adjusted effect. Exposure errors, overdispersion, excess zeros and dependence would require additional modeling decisions.
