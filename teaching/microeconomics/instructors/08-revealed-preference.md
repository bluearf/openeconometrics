# Instructor companion · Lab 08

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Use observation i’s price vector for every alternative bundle.

**B.** 0 under tolerance 1e−10.

**C.** No. Finite choices can be rationalized by many preference structures.

**D.** No. The script tests direct two-choice WARP conditions; it does not calculate the transitive closure.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/08-revealed-preference/revealed_preference.xlsx) for the published analysis. The [original generator](generators/08-revealed-preference.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/08-revealed-preference/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/08-revealed-preference/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: chosen bundles exhaust budget, no strict two choice cycle. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** No observed violation does not identify unique preferences or prove consistency in unobserved budgets. A full GARP analysis would additionally examine transitive revealed-preference chains.
