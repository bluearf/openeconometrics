# Instructor companion · Lab 09

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** 2×5/sqrt(L)=5 gives L=4.

**B.** 10 sqrt(4)=20.

**C.** No. It changes profit and possible exit considerations, not value marginal product.

**D.** Marginal revenue rather than market price would value an additional unit of output.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/09-production-input/production_input.xlsx) for the published analysis. The [original generator](generators/09-production-input.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/09-production-input/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/09-production-input/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: value marginal product equals wage, analytic matches grid. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The first-order condition assumes an interior choice and price taking. It does not account for indivisible workers, adjustment costs or an uncertain output price.
