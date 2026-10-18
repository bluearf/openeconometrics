# Instructor companion · Lab 09

## Teaching sequence

Fix the sample first: 72 encounters, blanks at encounters 11 and 41, 70 observed waits used for both confidence levels. Then separate the standard deviation of waits (2.30 minutes) from the standard error of the mean (0.275 minutes) before writing the interval formula.

Before running, collect predictions for the pause question and ask which piece of the formula changes between 95% and 99%. When showing the output, point to the two native tables: `statistics` holds the interval for the mean, while `test` holds the interval for $\mu-10$. Spend time on the repeated-sampling meaning of confidence and on the difference between an interval for the mean and a range for individual waits.

## Worked exercise answers

**1.** The 99% half-width is $2.649 \times 0.2751 = 0.729$ minutes, matching $(11.330-9.872)/2$. The widths are 1.098 minutes (95%) and 1.458 minutes (99%), a ratio of $1.458/1.098 = 1.328$, which equals the ratio of critical values $2.649/1.995$. Halving the 95% half-width to 0.274 minutes needs about four times as many observations; with $s = 2.30$ the exact t-based requirement is $n = 273$ (the simple $4 \times 70 = 280$ is slightly conservative because the critical value also falls as the degrees of freedom grow).

**2.** Multiply every quantity measured in minutes by 60: mean 636.07 seconds, standard deviation 138.12 seconds, standard error 16.51 seconds, 95% interval 603.14 to 669.00 seconds (99%: 592.34 to 679.80 seconds). The sample size, degrees of freedom, critical values, confidence levels and the t statistic for a correspondingly converted null (600 seconds) do not change.

**3.** The 90% interval is 10.14 to 11.06 minutes (critical value 1.667, half-width 0.459, width 0.917). The three intervals are nested and share the center 10.60 minutes; lower confidence gives a narrower interval: 90% width 0.917, 95% width 1.098, 99% width 1.458 minutes.

**4.** Only 13 of the 70 observed waits (18.6%) lie inside the 95% confidence interval for the mean. The prediction range is $10.601 \pm 1.995 \times 2.302 \times \sqrt{1 + 1/70} = 10.601 \pm 4.625$, or 5.98 to 15.23 minutes, and it contains 68 of the 70 waits. The confidence interval describes uncertainty about the mean, which shrinks with $\sqrt n$; the prediction range must include the full variation of a single wait, which does not shrink with $n$.

**5.** The 95% interval (10.05 to 11.15) excludes 10, so the two-sided p value is below 0.05; the 99% interval (9.87 to 11.33) includes 10, so the p value is above 0.01. The `test` table reports $t = 2.185$ with 69 degrees of freedom and two-sided $p = 0.0323$, consistent with that bracket. The test's interval for $\mu - 10$ is 0.05 to 1.15 minutes.

**6.** For example: "The 95% confidence interval for the population mean wait is 10.05 to 11.15 minutes. The procedure that produced it covers the true mean in 95% of repeated samples under its assumptions; this particular interval either contains the mean or does not. It says nothing about the share of customers who wait that long; individual waits in the sample range from about 5.5 to 16 minutes."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/09-confidence-intervals/service_waits.xlsx) for the published analysis. The [original generator](generators/09-confidence-intervals.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/09-confidence-intervals/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/09-confidence-intervals/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native se matches sample formula, df is n minus one, ci symmetric about mean, higher confidence wider. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Confidence as a posterior probability; individual coverage as mean coverage; a narrower 99% interval; reading difference-from-null endpoints as mean endpoints.
