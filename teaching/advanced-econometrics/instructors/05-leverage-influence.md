# Instructor companion · Lab 05

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** k/n=3/240=0.0125.

**B.** No. It depends on predictor locations and the design.

**C.** The sample and point estimates; the X slope changes by -0.328882.

**D.** No. It retains the row while adjusting its covariance contribution.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/05-leverage-influence/leverage_influence.xlsx) for the published analysis. The [original generator](generators/05-leverage-influence.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/05-leverage-influence/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/05-leverage-influence/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: leverage trace equals rank, flagged leverage above average, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Sensitivity does not justify removing a valid row merely to change significance. Data-quality evidence, sampling scope and robustness should be discussed separately.
