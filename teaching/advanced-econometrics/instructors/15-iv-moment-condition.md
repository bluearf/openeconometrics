# Instructor companion · Lab 15

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** The control itself; Z is the excluded instrument.

**B.** Native 1.07921 and matrix 1.07921 agree.

**C.** No. Relevance and orthogonality to the structural error are separate conditions.

**D.** It uses fitted-exposure regression residuals instead of the correct structural IV covariance.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/15-iv-moment-condition/iv_moment_condition.xlsx) for the published analysis. The [original generator](generators/15-iv-moment-condition.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/15-iv-moment-condition/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/15-iv-moment-condition/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: projection formula matches iv, same complete sample, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Exclusion is an assumption about the mechanism, not proved by first-stage strength. Exactly identified IV supplies no overidentification test, and weak instruments can make usual inference unreliable.
