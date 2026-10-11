# Instructor companion · Lab 13

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** 100−2Q=20 gives Q=40.

**B.** 1600−200=1400.

**C.** 0.5×(80−40)×(60−20)=800.

**D.** No. Transfers change distribution; deadweight loss is the value of forgone beneficial trades.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/13-monopoly-welfare/monopoly_welfare.xlsx) for the published analysis. The [original generator](generators/13-monopoly-welfare.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/13-monopoly-welfare/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/13-monopoly-welfare/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: marginal revenue equals mc, lost trade triangle. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Price discrimination, dynamic innovation and entry can change these comparisons. The exercise is a static partial-equilibrium model, and producer surplus excludes the separately stated fixed cost.
