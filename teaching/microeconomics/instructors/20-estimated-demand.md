# Instructor companion · Lab 20

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** A 1% price rise corresponds locally to about -1.22405% lower quantity, holding income fixed.

**B.** 100(1.1^βP−1)=-11.0116%.

**C.** No. It adjusts covariance; exogeneity is a separate condition.

**D.** All supplied price, income and quantity values are strictly positive; log zero would be undefined.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/20-estimated-demand/estimated_demand.xlsx) for the published analysis. The [original generator](generators/20-estimated-demand.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/20-estimated-demand/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/20-estimated-demand/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: matrix coefficients match, full result roundtrip, all observations used. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The exogeneity argument applies to this mechanism. An actual demand study would need an identification design, instruments or other evidence; a high fit statistic would not establish it.
