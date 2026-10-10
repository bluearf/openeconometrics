# Instructor companion · Lab 03

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** 1.5×1/(1+1)=0.75.

**B.** Finite sample covariances between independent draws are not exactly zero.

**C.** No. It changes estimated uncertainty, not the measurement-error estimand.

**D.** Independent additive outcome noise primarily increases variance without the same attenuation denominator; correlated errors require separate analysis.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/03-measurement-error/measurement_error.xlsx) for the published analysis. The [original generator](generators/03-measurement-error.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/03-measurement-error/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/03-measurement-error/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: same units and sample, positive measurement variance, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Nonclassical error can bias estimates in other directions. HC3 does not correct measurement bias, and an estimated reliability from an unvalidated proxy is not automatically known.
