# Instructor companion · Lab 02

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Both the Z outcome effect and Z-on-X relationship are positive here, so the gap is positive.

**B.** 0.735798 × 0.741588 = 0.545659.

**C.** The auxiliary slope is zero and the two exposure slopes agree.

**D.** No. That would change the estimand and may bias the intended causal comparison.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/02-omitted-variable-bias/omitted_variable_bias.xlsx) for the published analysis. The [original generator](generators/02-omitted-variable-bias.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/02-omitted-variable-bias/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/02-omitted-variable-bias/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: sample decomposition exact, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** An exact sample decomposition is not a generic justification for adding every available variable. A control’s causal role and measurement quality matter.
