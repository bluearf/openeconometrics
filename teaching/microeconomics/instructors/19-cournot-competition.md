# Instructor companion · Lab 19

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** (100−20)/(2+1)=26.6667.

**B.** Monopoly output 40 from the same demand and marginal cost.

**C.** No. Individual output falls, while aggregate output rises.

**D.** No. Under standard unconstrained homogeneous-good assumptions, price competition can produce marginal-cost pricing with two firms.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/19-cournot-competition/cournot_competition.xlsx) for the published analysis. The [original generator](generators/19-cournot-competition.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/19-cournot-competition/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/19-cournot-competition/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: symmetric best response, more firms lower price. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The number of firms is an external scenario input. Entry costs, asymmetric costs, capacity and repeated-game collusion can produce different outcomes.
