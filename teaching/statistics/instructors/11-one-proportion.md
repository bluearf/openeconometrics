# Instructor companion · Lab 11

## Teaching sequence

Write two SE formulas side by side before showing either interval. Keep fractions distinct from percentage points.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** 137/400 = 0.3425.

**B.** The observed rate is below .4; the two-sided p counts deviations in either direction.

**C.** The native interval is Wald; the additional interval is Wilson. They use different constructions.

**D.** No. Within-user dependence changes the effective information and requires a dependence-aware design.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/11-one-proportion/app_opt_in.xlsx) for the published analysis. The [original generator](generators/11-one-proportion.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/11-one-proportion/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/11-one-proportion/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native z uses null variance, wilson interval inside probability support, binary coding preserved. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Using successes as the denominator; interpreting p as P(null true); confusing a Wald interval with Wilson; ignoring repeated users.
