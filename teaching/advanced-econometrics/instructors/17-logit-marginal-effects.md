# Instructor companion · Lab 17

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Log odds; X coefficient=0.951863.

**B.** exp(βX)=2.59053.

**C.** Multiply 0.190116 by 100 per unit X; this is a point average derivative.

**D.** No. It verifies the point AME and full coefficient covariance, while an AME interval would require additional propagation.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/17-logit-marginal-effects/logit_marginal_effects.xlsx) for the published analysis. The [original generator](generators/17-logit-marginal-effects.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/17-logit-marginal-effects/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/17-logit-marginal-effects/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: probabilities in support, intercept score zero, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Odds ratios are not risk ratios or percentage-point effects. A descriptive AME is not causal without an identification argument; uncertainty for the AME requires its own delta or resampling calculation.
