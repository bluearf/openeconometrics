# Instructor companion · Lab 12

## Teaching sequence

Fix A−B on the board and trace the same sign through the table, interval and words.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** A−B = -2.92812 minutes. Negative means A is faster on average in this sample.

**B.** -2.92812/1.22106 = -2.39802.

**C.** Satterthwaite matches an estimated variance distribution; df need not equal nA+nB−2.

**D.** No. Those measurements are dependent pairs; analyze within-pair differences.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/12-independent-means/delivery_methods.xlsx) for the published analysis. The [original generator](generators/12-independent-means.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/12-independent-means/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/12-independent-means/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native se matches unpooled formula, welch df matches satterthwaite, difference orientation, samples partition raw rows. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Pooling despite unequal variances; treating independent groups as pairs; interpreting an observed method difference as causality.
