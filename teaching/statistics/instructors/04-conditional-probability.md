# Instructor companion · Lab 04

## Teaching sequence

Draw the two-by-two table on the board and have students circle a different denominator for each conditional question.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Use the number of member visits. The resulting fraction is 0.625532. Dividing the same cell by all 600 visits instead gives the joint fraction 0.245.

**B.** Membership among purchasers is 0.617647. Its denominator is purchaser visits, so it need not equal purchase among members.

**C.** The joint probability would equal $P(M)P(P)$. In this sample, 0.245 differs from the marginal product 0.155361. Sampling uncertainty is a separate question from computing those frequencies.

**D.** No. Selection into membership can reflect prior engagement or spending. A randomized membership offer with a defined follow-up could support a different causal comparison.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/04-conditional-probability/customer_events.xlsx) for the published analysis. The [original generator](generators/04-conditional-probability.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/04-conditional-probability/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/04-conditional-probability/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: four cells partition visits, conditional times marginal equals joint, bayes reverses denominator. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Reversing a conditional without changing the denominator; assuming independence; confusing visit and customer weights; reading association as a membership treatment effect.
