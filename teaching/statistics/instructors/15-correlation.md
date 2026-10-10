# Instructor companion · Lab 15

## Teaching sequence

Contrast correlation’s unit-free symmetry with a regression slope. Discuss two different mechanisms that could produce the same scatter.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** 160−2 = 158.

**B.** r stays unchanged under positive scaling; a points-per-minute slope is one sixtieth of a points-per-hour slope.

**C.** 0.425076 is the fraction fitted by a simple linear model with an intercept, not a causal share.

**D.** No. For example a symmetric quadratic association can have zero linear correlation.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/15-correlation/study_hours_scores.xlsx) for the published analysis. The [original generator](generators/15-correlation.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/15-correlation/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/15-correlation/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native correlation matches centered crossproduct, pair count matches common sample, correlation bounded. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Correlation as causation; r as a slope; pairwise denominators mixed across variables; zero correlation as independence without distributional assumptions.
