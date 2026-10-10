# Instructor companion · Lab 07

## Teaching sequence

Ask students to solve the defining condition algebraically before running the complete file. Compare that solution with the plotted scenarios and the numerical table. Discuss why the model assumptions matter for the final exercise.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** -5.44394 + -2.55606 = -8.

**B.** Utility, verified as 26.0273 before and 26.0273 after compensation.

**C.** That would be Slutsky compensation. Hicks compensation minimizes spending needed for original utility.

**D.** X is normal and becomes relatively dearer; lower real income further reduces its demand.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/07-substitution-income/substitution_income.xlsx) for the published analysis. The [original generator](generators/07-substitution-income.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/07-substitution-income/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/07-substitution-income/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: effects add to total, utility held fixed. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** For inferior goods the income effect can oppose substitution. This example’s signs follow from normal Cobb–Douglas demand. Compensated income is a hypothetical comparison, not an actual transfer received.
