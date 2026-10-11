# Instructor companion · Lab 03

## Teaching sequence

Write units beside every equation. Use a minutes-to-hours conversion to make the squared units of variance concrete.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Variance is in minutes squared; standard deviation and IQR are in minutes. The observed values are 62.508, 7.9062 and 11.2892 respectively.

**B.** Divide variance by 3600 and standard deviation by 60. The latter becomes 0.13177 hours. Dividing variance by 60 would be dimensionally incorrect.

**C.** The mean increases by five minutes. All centered deviations and pairwise differences remain the same, so variance, standard deviation and IQR are unchanged. CV decreases because its denominator increases.

**D.** It describes individual delivery variation. Under independent sampling the mean's standard error is $s/\sqrt{160}$, and a confidence interval additionally requires a critical value and inferential assumptions.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/03-spread/delivery_variation.xlsx) for the published analysis. The [original generator](generators/03-spread.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/03-spread/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/03-spread/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native sd matches centered sum, rescaling sd, rescaling variance, ordered quartiles. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Variance in minutes; standard deviation as a confidence interval; confusing n and n−1; interpreting CV on a scale without a meaningful zero.
