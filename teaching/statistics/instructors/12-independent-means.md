# Instructor companion · Lab 12

## Teaching sequence

Write A − B on the board before anything runs and trace that sign through the estimate, the t statistic, the interval and the final sentence. Have students read the group standard deviations before the test, so the size of method B's variance term in the standard error is expected rather than surprising.

Use the slight overlap of the two single-group intervals as a deliberate trap, then show why the standard error of a difference is smaller than the sum of the two standard errors. Present the pooled row as a design choice made before looking at p values, and close by separating the mean comparison from claims about individual deliveries and about causation.

## Worked exercise answers

**1.** For B − A the estimate is +2.93 minutes, t = +2.398 and the 95% interval is 0.51 to 5.35 minutes (the A − B endpoints negated and swapped). The standard error 1.221, the Welch degrees of freedom 108.4 and the two-sided p value 0.0182 do not change; the two one-sided p values exchange their labels. The substantive sentence (A's mean is lower) is unchanged.

**2.** `oe.ttest(raw, "minutes", by="method", alpha=0.01, missing="raise")` gives a 99% Welch interval of −6.13 to 0.27 minutes, using the critical value 2.622 with 108.4 degrees of freedom. It contains zero while the 95% interval does not, because the two-sided p value 0.0182 is below 0.05 but above 0.01. The point estimate is the same; only the critical value changed.

**3.** With 170 B deliveries, $v_B = 109.88/170 = 0.646$, so SE = √(0.198 + 0.646) = 0.919 and t = −2.928/0.919 = −3.19 (Welch df 231.2). Doubling A to 130 instead gives $v_A = 12.89/130 = 0.099$, SE = √(0.099 + 1.293) = 1.180 and t = −2.48. Adding B deliveries helps far more because B's term is about 87% of the variance of the difference; shrinking A's already small term barely moves the standard error.

**4.** If $v_A = 0$, Satterthwaite's formula reduces to $v_B^2/(v_B^2/84) = 84 = n_B - 1$: the uncertainty of the standard error would come only from B's 85 deliveries. A's term is small but not zero, so it adds some information and ν rises above 84; it stays well below 148 because ν reaches the pooled count only when each term is proportional to its group's degrees of freedom ($v_A/64 = v_B/84$), and here B's term is far larger. ν is noninteger because it is an approximation matched to the variance of an estimated quantity, not a count of observations.

**5.** The pooled variance is $s_p^2 = (64 \times 12.89 + 84 \times 109.88)/148 = 67.94$, so the pooled SE² is $67.94/65 + 67.94/85 = 1.045 + 0.799 = 1.844$ and SE = 1.358. Welch's terms are 0.198 and 1.293, summing to 1.491. Pooling assigns a variance dominated by B to group A, inflating A's term more than it deflates B's, because the inflated term is divided by the smaller n. When the larger group has the larger variance, the pooled test is conservative. If the 65-delivery group had been the noisy one (same two standard deviations swapped), the pooled SE would be 1.220 against Welch's 1.357, so pooling would understate the uncertainty.

**6.** `(raw.loc[raw.method == "B", "minutes"] < raw.loc[raw.method == "A", "minutes"].mean()).sum()` gives 34: 34 of 85 method B deliveries (40%) took less time than method A's mean of 33.45 minutes. A lower mean is a statement about averages; many individual B deliveries still beat a typical A delivery because B is so variable. A supported rewrite: "In these 150 synthetic deliveries, method A's mean time was 2.93 minutes shorter than method B's (Welch 95% interval 0.51 to 5.35 minutes shorter, p = 0.018). The file does not record how deliveries were allocated to methods, so this is a descriptive difference in averages, not evidence that switching to A would shorten deliveries."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/12-independent-means/delivery_methods.xlsx) for the published analysis. The [original generator](generators/12-independent-means.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/12-independent-means/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/12-independent-means/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native se matches unpooled formula, welch df matches satterthwaite, difference orientation, samples partition raw rows. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Pooling despite unequal variances; treating independent groups as pairs; interpreting an observed method difference as causality.
