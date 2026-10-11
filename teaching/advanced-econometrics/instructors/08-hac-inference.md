# Instructor companion · Lab 08

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** No. The X slope remains 1.26994.

**B.** With L=4 they are .8 and .2.

**C.** n/(n−k)=240/237.

**D.** No. Nonstationarity and identification require a separate model and analysis.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/08-hac-inference/hac_inference.xlsx) for the published analysis. The [original generator](generators/08-hac-inference.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/08-hac-inference/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/08-hac-inference/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: covariance only preserves slope, regular unique time, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** HAC is not a cure for omitted dynamics, endogenous regressors or nonstationarity. Different lag choices change the estimated covariance and should be sensitivity analyses rather than significance searches.
