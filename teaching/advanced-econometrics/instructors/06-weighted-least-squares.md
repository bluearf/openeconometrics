# Instructor companion · Lab 06

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Those with larger supplied error variance, because weight=1/variance.

**B.** Yes. Its transformed column is sqrt(weight), not a column of ones.

**C.** No. A common positive factor cancels from the weighted normal equations.

**D.** The mechanism is heteroskedastic, violating its constant-variance assumption.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/06-weighted-least-squares/weighted_least_squares.xlsx) for the published analysis. The [original generator](generators/06-weighted-least-squares.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/06-weighted-least-squares/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/06-weighted-least-squares/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: transformed design identity, strictly positive weights, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Misspecified weights can reduce efficiency or alter the appropriate covariance. Survey and frequency weights require their own contracts and must not be relabeled as precision weights.
