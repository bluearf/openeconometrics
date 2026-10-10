# Instructor companion · Lab 07

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** 40 groups, rather than 240 independent rows.

**B.** No. It remains 1.20802.

**C.** G−1=39.

**D.** No. The realized within-group score covariance can change uncertainty differently across coefficients.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/07-clustered-inference/clustered_inference.xlsx) for the published analysis. The [original generator](generators/07-clustered-inference.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/07-clustered-inference/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/07-clustered-inference/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: covariance only preserves coefficients, df is clusters minus one, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** With very few clusters, ordinary cluster asymptotics may be unreliable. Choosing groups after searching for favorable significance does not supply a valid sampling design.
