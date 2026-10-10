# Instructor companion · Lab 17

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Private marginal cost plus the external damage of 10 per unit.

**B.** 60−.5Q=20+.5Q gives Q=40.

**C.** It is a per-unit marginal incentive, 10; total original damage is 500.

**D.** No. It is a transfer in this benchmark; actual external damage is already subtracted.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/17-external-cost/external_cost.xlsx) for the published analysis. The [original generator](generators/17-external-cost.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/17-external-cost/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/17-external-cost/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: social foc, corrective tax implements optimum. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A real corrective tax requires evidence about marginal damage and responses. Total damage at the original output is not the optimal unit tax.
