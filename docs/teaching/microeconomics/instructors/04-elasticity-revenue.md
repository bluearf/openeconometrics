# Instructor companion · Lab 04

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** No. −2 is constant, but P/Q varies; at P=20 elasticity is -0.5.

**B.** 120−4P=0 gives P=30 and R=1800.

**C.** Its magnitude diverges near P=60; the expression is undefined at Q=0.

**D.** Generally no. Profit also subtracts the cost of producing the demand quantity.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/04-elasticity-revenue/elasticity_revenue.xlsx) for the published analysis. The [original generator](generators/04-elasticity-revenue.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/04-elasticity-revenue/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/04-elasticity-revenue/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: interior revenue foc, elasticity uses local quantity. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A local elasticity need not predict a large price change. Revenue excludes production costs, so its maximum is not a general recommendation for a firm’s price.
