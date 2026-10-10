# Instructor companion · Lab 01

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** They agree at 1.18823 up to numerical precision on the same rows.

**B.** The intercept and all controls being partialled out, here the observed z.

**C.** The full model; its HC3 SE is 0.111878.

**D.** No. It explains the controlled linear coefficient; exogeneity requires an independent argument.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/01-partialling-out/partialling_out.xlsx) for the published analysis. The [original generator](generators/01-partialling-out.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/01-partialling-out/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/01-partialling-out/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: partial slope identity, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Partialling out is an algebraic identity. It does not remove unobserved confounding or justify using residual-regression degrees of freedom for the original model.
