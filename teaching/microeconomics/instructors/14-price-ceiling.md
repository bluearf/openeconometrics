# Instructor companion · Lab 14

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** The short side supplies 30 units.

**B.** 70−30=40.

**C.** 1250−1050=200.

**D.** Generally not. Some high-value buyers could be excluded; the stated allocation is a benchmark.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/14-price-ceiling/price_ceiling.xlsx) for the published analysis. The [original generator](generators/14-price-ceiling.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/14-price-ceiling/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/14-price-ceiling/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: binding ceiling, quantity short side, surplus partition. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The consumer surplus estimate is conditional on rationing. A lower posted price does not prove every buyer benefits or that everyone can obtain the good.
