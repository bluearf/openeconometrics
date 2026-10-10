# Instructor companion · Lab 16

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** 40−30=10.

**B.** 10×60=600.

**C.** That omits the 600 paid by government and counts a transfer as a net gain.

**D.** Yes, for a sufficiently large positive external benefit or another specified distortion; neither exists in this benchmark.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/16-subsidy-welfare/subsidy_welfare.xlsx) for the published analysis. The [original generator](generators/16-subsidy-welfare.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/16-subsidy-welfare/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/16-subsidy-welfare/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: subsidy wedge, overproduction, fiscal cost subtracted. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The conclusion depends on the no-externality benchmark. Positive spillovers could justify subsidizing some production, while costly public finance could make the fiscal burden larger.
