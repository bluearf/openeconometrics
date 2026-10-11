# Instructor companion · Lab 06

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Both rise 25%; X becomes 20 and Y 45.

**B.** Its uncompensated demand depends only on its own price and the fixed Y expenditure share.

**C.** No. Both income derivatives are positive.

**D.** No. It follows from this specific uncompensated demand system.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/06-demand-comparative-statics/demand_comparative_statics.xlsx) for the published analysis. The [original generator](generators/06-demand-comparative-statics.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/06-demand-comparative-statics/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/06-demand-comparative-statics/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: all budgets exhausted, homogeneous demand. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** These elasticities are imposed by Cobb–Douglas preferences. They do not establish actual consumer responses or substitution patterns across arbitrary goods.
