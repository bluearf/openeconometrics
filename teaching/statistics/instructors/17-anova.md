# Instructor companion · Lab 17

## Teaching sequence

Use the variance decomposition first. Only then open the post-hoc table and compare its question with the omnibus test.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** (292.857/2)/(4051.27/132) = 4.77098.

**B.** 292.857 + 4051.27 = 4344.12 up to rounding.

**C.** Three: A−B, A−C and B−C. Bonferroni adjusts for all 3.

**D.** No. It rejects their joint equality. Read adjusted pairwise results separately.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/17-anova/training_formats.xlsx) for the published analysis. The [original generator](generators/17-anova.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/17-anova/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/17-anova/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: sum of squares decomposition, native F matches mean square ratio, native eta squared matches ss ratio, between df plus within df. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Omnibus rejection as every pair differing; ignoring multiple comparisons; F as a directional effect; η² as causal attribution.
