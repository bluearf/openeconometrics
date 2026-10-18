# Instructor companion · Lab 10

## Teaching sequence

Put the statistical and operational questions side by side on the board before any calculation: "Is the process mean different from 500 ml?" and "Does the deviation matter?" Fix the two-sided alternative from the purpose of the check, then build the t statistic as a deviation in milliliters divided by a standard error in milliliters.

Before running, collect answers to the pause question about the two-sided test and the positive deviation. After running, insist that students report the deviation (0.755 ml) and its interval (0.35 to 1.16 ml) before the p value, and that they name a tolerance before calling the result important. Use the histogram to separate the process mean from individual containers.

## Worked exercise answers

**1.** $t = (500.755 - 501)/0.2049 = -1.19$ with 79 degrees of freedom, two-sided $p = 0.236$; `oe.ttest(raw, "milliliters", mu=501, missing="raise")` gives the same values. The data do not reject a process mean of 501 ml, consistent with 501 lying inside the 95% interval for the mean (500.35 to 501.16 ml). Failing to reject 501 does not show that the mean equals 501.

**2.** For $H_1: \mu > 500$, the one-sided p value is `p_greater` = 0.000207, half the two-sided value because the deviation lies in the hypothesized direction. For $H_1: \mu < 500$, `p_less` = 0.99979: a sample mean above 500 provides no evidence at all of underfilling, and the two one-sided p values sum to one. The one-sided version is legitimate only if the direction was fixed before seeing the data.

**3.** The deviation is 0.000755 liters and the standard error 0.000205 liters. Both change by the same factor of 1/1000, so $t = 3.69$, the 79 degrees of freedom and $p = 0.0004$ are unchanged; the null becomes 0.5 liters.

**4.** 28 of the 80 containers (0.35) hold less than 500 ml. Under a normal model with mean 500.76 ml and SD 1.83 ml, $z = (500 - 500.755)/1.832 = -0.41$ and $\Phi(-0.41) = 0.340$, or about 27 of 80 containers. The observed and model fractions agree closely. Neither contradicts the test, which concerns the process mean; a mean above target with a standard deviation of 1.83 ml still leaves about a third of containers below 500 ml.

**5.** The 95% interval for the difference is 0.35 to 1.16 ml. With a 1 ml tolerance, the interval straddles the tolerance, so the data cannot tell whether the deviation is operationally important or not. With a 0.25 ml tolerance, the whole interval lies above the tolerance, so the data indicate a deviation that matters in practice. Only the second case supports that conclusion.

**6.** For example: "The mean fill is 0.755 ml above the 500 ml target (95% interval 0.35 to 1.16 ml). If the process were centered on 500 ml, a sample of 80 containers would deviate this much in either direction with probability 0.0004, so the data are strong evidence that the process mean is not exactly 500 ml. The p value is not the probability that the machine is correctly calibrated, and whether a deviation of this size is large depends on a tolerance specified in advance."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/10-mean-test/filling_volumes.xlsx) for the published analysis. The [original generator](generators/10-mean-test.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/10-mean-test/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/10-mean-test/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native t matches direct formula, df matches retained n, p in unit interval, two sided tail sum. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** p as P(H0 true); a significant result as a large operational error; non-rejection as equivalence; deciding a one-sided direction after seeing the observed sign.
