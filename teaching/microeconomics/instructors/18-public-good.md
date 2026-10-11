# Instructor companion · Lab 18

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Both people consume the same nonrival provision; an extra unit benefits both.

**B.** 70−1.5Q=25 gives Q=30.

**C.** A: 10 and B: 15 currency/unit.

**D.** No. It presumes known benefits and does not create incentives to reveal or fund them.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/18-public-good/public_good.xlsx) for the published analysis. The [original generator](generators/18-public-good.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/18-public-good/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/18-public-good/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: samuelson condition, positive individual marginal benefits. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The optimality condition is not a mechanism guaranteeing voluntary financing. Information, distribution and free-riding require separate institutional analysis.
