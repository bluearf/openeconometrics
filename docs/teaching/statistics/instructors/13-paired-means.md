# Instructor companion · Lab 13

## Teaching sequence

Have students join observations by worker ID before subtracting. Contrast the paired variance with an incorrect independent calculation.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** 60 raw − 1 incomplete = 59 complete pairs.

**B.** 2.70573/sqrt(59) = 0.352257.

**C.** After minus before is positive, so retained workers improved on average.

**D.** No. A before/after comparison lacks an untreated counterfactual; concurrent changes can explain it.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/13-paired-means/worker_before_after.xlsx) for the published analysis. The [original generator](generators/13-paired-means.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/13-paired-means/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/13-paired-means/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: paired se uses differences, t orientation after before, variance of difference identity, unique pairs. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Counting measurements as independent workers; reversing the subtraction; retaining an incomplete pair; treating time change as treatment causality.
