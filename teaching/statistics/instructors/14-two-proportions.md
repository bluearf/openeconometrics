# Instructor companion · Lab 14

## Teaching sequence

Draw one number line in probabilities and label it in percentage points before the test runs, then have students state the difference as A − B in both units. Ask which denominator belongs to the pooled rate and which to each group's rate, and pose the pause question before the two standard errors are computed.

After the test and interval, spend time on the interval's width: the non-rejection is the least informative part of the result. Close with the design questions the workbook raises, the alternating assignment and the unstated follow-up window.

## Worked exercise answers

**1.** Relative to B, A's rate is (0.260 − 0.2967)/0.2967 = −12.4%, about 12% lower. Relative to A, B's rate is (0.2967 − 0.260)/0.260 = +14.1%. The two percents differ because each divides the same 3.67-point gap by a different baseline. "3.7 percent lower" mislabels a percentage-point gap as a relative change; correct versions are "3.7 percentage points lower" or "about 12% lower than B's rate."

**2.** Using the unpooled standard error, −0.03667/0.03656 = −1.0028, against −1.0020 with the pooled one. They are close because the groups are the same size and the rates are close: with equal $n$, $SE_0^2 - SE_1^2 = (\hat p_A-\hat p_B)^2/(2n) = 0.0367^2/600 = 2.2\times10^{-6}$, so the pooled version is always slightly larger and the gap grows with the squared rate difference. Unequal sizes and rates make them diverge, in either direction. For example, 100 recipients at 0.10 against 900 at 0.40 gives $SE_0 = 0.051$ and $SE_1 = 0.034$; swapping the rates (100 at 0.40, 900 at 0.10) gives $SE_0 = 0.035$ and $SE_1 = 0.050$.

**3.** With 1,200 per message, $SE_0 = \sqrt{0.2783 \times 0.7217 \times 2/1200} = 0.0183$, z = −2.004 and two-sided p = 0.045. The unpooled standard error is 0.0183, and the 95% interval is −0.0725 to −0.0008 (−7.25 to −0.08 percentage points). The test now rejects equal rates at 5%, narrowly, and the interval just excludes zero. The rates, the −3.67-point estimate and the pooled rate are unchanged; both standard errors halve because the sample is four times larger.

**4.** With `positive=0` the group proportions become 0.740 and 0.703, the difference A − B becomes +0.0367, z becomes +1.002 and the interval becomes −0.035 to 0.108, the old endpoints negated and swapped. The standard errors (0.0253, 0.0264, 0.0366 and the null 0.0366) and the two-sided p value 0.316 are unchanged because $p(1-p)$ is the same for $p$ and $1-p$. Recoding the event changes the sign of the comparison, not its evidence.

**5.** `oe.prtest(raw, "response", by="message", positive=1, alpha=0.10, missing="raise")` gives a 90% interval for A − B of −0.0968 to 0.0235, or −9.7 to +2.3 percentage points. It extends well past −2 points and slightly past +2, so the data do not show that the messages are interchangeable. Equivalence at the 2-point margin would require the whole 90% interval to lie between −2 and +2 points (the two one-sided tests procedure at the 5% level), which would need a much larger sample.

**6.** For example: "In 600 synthetic recipients, message A's response rate was 3.67 percentage points below message B's (26.0% against 29.7%; 95% interval −10.8 to +3.5 points). The difference is not statistically significant at the 5% level (z = −1.00, p = 0.32), but the interval is too wide to show that the messages perform the same."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/14-two-proportions/message_responses.xlsx) for the published analysis. The [original generator](generators/14-two-proportions.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/14-two-proportions/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/14-two-proportions/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native z uses pooled null, native interval se is unpooled, group counts partition, percentage point scale. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Relative percentages as percentage points; treating non-rejection as equality; mixing pooled and unpooled variance.
