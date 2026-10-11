# Instructor companion · Lab 20

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** The fitted effect at baseline X=0, 1.1486.

**B.** τ+δ×0.0476494 = 1.17948.

**C.** HC3 linear-contrast uncertainty conditional on the observed baseline design, SE=0.135705.

**D.** The estimand, complete and retained sample, treatment/covariate definitions, all declared specifications, full covariance, interval convention, sensitivity discussion and saved outputs.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/20-specification-report/specification_report.xlsx) for the published analysis. The [original generator](generators/20-specification-report.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/20-specification-report/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/20-specification-report/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: interaction preserved, average effect identity, same sample all specifications, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Specification choices should be stated before searching for significance. This original randomized simulation does not establish an empirical policy effect; population generalization requires a target population and sampling argument.
