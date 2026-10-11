# Instructor companion · Lab 01

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** No. It rises with X and is 1.5 at X=24.

**B.** 2 sqrt(1600−24²)=64.

**C.** It is feasible but technically inefficient under this model.

**D.** It expands feasible choices; realized welfare also depends on preferences and the chosen allocation.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/01-production-frontier/production_frontier.xlsx) for the published analysis. The [original generator](generators/01-production-frontier.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/01-production-frontier/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/01-production-frontier/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: frontier identity, negative frontier slope. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A frontier cannot choose an allocation without preferences or values. A movement along it differs from an outward shift caused by new resources or technology.
