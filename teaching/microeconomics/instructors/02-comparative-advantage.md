# Instructor companion · Lab 02

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** A, with cost 0.5 below B’s 2.

**B.** 0.5 < 1 < 2.

**C.** A retains 40 X and receives 20 Y; B retains 20 Y and receives 20 X. Both bundles exceed their own linear frontiers at those X levels.

**D.** No. Demand and feasible trade volumes must also determine terms of trade.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/02-comparative-advantage/comparative_advantage.xlsx) for the published analysis. The [original generator](generators/02-comparative-advantage.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/02-comparative-advantage/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/02-comparative-advantage/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: price between opportunity costs, labor constraints bind. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Opportunity costs alone do not determine the equilibrium world price or distribution of gains. Transport costs, adjustment, multiple factors and preferences can change the conclusion.
