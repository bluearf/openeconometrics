# Instructor companion · Lab 10

## Teaching sequence

Put statistical and operational questions beside each other. Require units for the effect and a separate interpretation of its dimensionless test statistic.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Compute (500.755−500)/0.204864 = 3.68705 up to displayed rounding. The ratio is dimensionless.

**B.** Under the specified null sampling model, outcomes with |T| at least 3.68705 have probability 0.000414531. This conditions on H0; it is not a probability assigned to H0 after observing the data.

**C.** Divide the ml discrepancy and SE by 1000. Their ratio and p value do not change. The null becomes 0.5 liters.

**D.** No. Equality testing and equivalence within a margin answer different questions. A tolerance must be specified and an appropriate uncertainty-based equivalence assessment conducted.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/10-mean-test/filling_volumes.xlsx) for the published analysis. The [original generator](generators/10-mean-test.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/10-mean-test/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/10-mean-test/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native t matches direct formula, df matches retained n, p in unit interval, two sided tail sum. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** p as P(H0 true); a significant result as a large operational error; non-rejection as equivalence; deciding a one-sided direction after seeing the observed sign.
