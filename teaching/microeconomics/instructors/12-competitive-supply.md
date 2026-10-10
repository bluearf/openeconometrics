# Instructor companion · Lab 12

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** (12−2)/.2=50.

**B.** The fixed cost remains unavoidable, giving -100.

**C.** Yes, whenever revenue covers variable cost but not the fixed cost.

**D.** No. Convex costs and comparison with feasible boundaries are also needed.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/12-competitive-supply/competitive_supply.xlsx) for the published analysis. The [original generator](generators/12-competitive-supply.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/12-competitive-supply/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/12-competitive-supply/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: no negative output, shutdown keeps fixed cost, interior foc. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A short-run supply rule does not establish long-run industry equilibrium or entry. Positive profit can attract entry if technology and market conditions permit it.
