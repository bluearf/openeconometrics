# Instructor companion · Lab 03

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** 120−2P=−20+2P gives P=35 and Q=50.

**B.** 20 units/day.

**C.** The equilibrium price rises by 5; moving along the shifted demand curve offsets 10 units.

**D.** An outward supply shift raises quantity but lowers price in this model.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/03-market-equilibrium/market_equilibrium.xlsx) for the published analysis. The [original generator](generators/03-market-equilibrium.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/03-market-equilibrium/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/03-market-equilibrium/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: market clears, shift increases price and quantity. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A static crossing does not describe adjustment speed, rationing institutions or uncertainty. Observed price/quantity pairs alone would not identify separate demand and supply slopes.
