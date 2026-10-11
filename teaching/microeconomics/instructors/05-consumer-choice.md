# Instructor companion · Lab 05

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** αm=.4×120=48 currency.

**B.** 3×16+2×36=120.

**C.** Not without additional cardinal and interpersonal assumptions; here they only rank this consumer’s bundles.

**D.** The budget set and optimal quantities remain unchanged; real purchasing opportunities are the same.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/05-consumer-choice/consumer_choice.xlsx) for the published analysis. The [original generator](generators/05-consumer-choice.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/05-consumer-choice/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/05-consumer-choice/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: budget exhausted, tangency. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Tangency is not universal: perfect substitutes or quantity constraints can produce corner solutions. This utility specification imposes constant shares rather than estimating preferences.
