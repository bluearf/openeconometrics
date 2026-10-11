# Instructor companion · Lab 09

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Divide W=510.212 by q=2 to obtain F=255.106.

**B.** No. R selects only x and z.

**C.** They omit the full restriction covariance and answer different hypotheses.

**D.** No. It rejects their simultaneous equality to zero.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/09-joint-restrictions/joint_restrictions.xlsx) for the published analysis. The [original generator](generators/09-joint-restrictions.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/09-joint-restrictions/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/09-joint-restrictions/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: F is wald over q, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A joint rejection says at least one restricted direction differs; it does not prove each slope differs from zero or establish causal effects.
