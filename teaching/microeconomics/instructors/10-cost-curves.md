# Instructor companion · Lab 10

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** 100/50+2+.1×50=9.

**B.** 2+.2×50=12.

**C.** The workbook uses integer q while the derivative permits fractional output.

**D.** No. Dividing total cost by q requires q>0; fixed total cost remains defined.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/10-cost-curves/cost_curves.xlsx) for the published analysis. The [original generator](generators/10-cost-curves.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/10-cost-curves/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/10-cost-curves/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: mc crosses atc at minimum, fixed cost per unit falls. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Cost curves describe a specified technology and input-price environment. They do not determine market demand or prove that a firm can sell output at a profitable price.
