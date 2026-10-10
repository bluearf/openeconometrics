# Instructor companion · Lab 12

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** The quadratic coefficient, -0.0256374.

**B.** The marginal slope at X=8.05467, equal to -0.131454.

**C.** Both complete bases span the same column space.

**D.** Generally no. The transformed constant component would be missing.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/12-polynomial-centering/polynomial_centering.xlsx) for the published analysis. The [original generator](generators/12-polynomial-centering.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/12-polynomial-centering/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/12-polynomial-centering/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: identical fitted values, centered slope interpretation, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Dropping the quadratic term after centering changes the model. Centering alone does not justify extrapolation outside the observed X support.
