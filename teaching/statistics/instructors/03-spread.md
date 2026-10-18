# Instructor companion · Lab 03

## Teaching sequence

Write units beside every equation. Work one deviation by hand (delivery 1) before the full sum, and use the minutes-to-hours conversion to make the squared units of variance concrete. Point out that the file holds a single delivery process: the lab builds the measures needed to compare reliability, rather than comparing two services.

Before running, use the pause question in Section 5 (divide the variance by 60 or by 3,600?). Connect each code line in the second snippet to the variance formula before interpreting the output.

## Worked exercise answers

**1.** The sum of squared deviations is 9,938.76 minutes squared. Dividing by $n=160$ gives 62.117 minutes squared and a standard deviation of 7.8815 minutes, compared with 62.508 and 7.9062 using $n-1=159$. The variance differs by the factor $159/160$, so it is 0.39 minutes squared (0.6%) smaller, and the standard deviation about 0.3% smaller. With 160 deliveries, losing one degree of freedom matters little; with 5 deliveries it would matter much more.

**2.** The variance is $62.508/3{,}600=0.017363$ hours squared. The IQR is $11.2892/60=0.18815$ hours. The standard deviation, 0.13177 hours, is given in the lab table.

**3.** The mean becomes 40.688 minutes. The variance (62.508 minutes squared), standard deviation (7.9062 minutes) and IQR (11.2892 minutes) do not change: adding 5 minutes moves every delivery and the mean by the same amount, so every deviation is unchanged, and both quartiles move by 5. The CV falls to $7.9062/40.688=0.1943$ because its denominator grows while its numerator does not. This is one reason the CV needs a scale with a meaningful zero.

**4.** One standard deviation either side of the mean is 27.78 to 43.59 minutes; 109 of 160 deliveries (68.1%) fall within it. Two standard deviations either side is 19.88 to 51.50 minutes; 155 of 160 deliveries (96.9%) fall within it, leaving 5 outside.

**5.** $s/\sqrt{160}=7.9062/12.649=0.6250$ minutes. The standard deviation describes how much individual deliveries vary around the mean. The standard error describes how much the mean of 160 independent deliveries would vary from sample to sample, which is why it shrinks with $\sqrt n$. A confidence interval for the mean would additionally need a critical value and the inferential assumptions behind it.

**6.** For example: "Across 160 synthetic deliveries, the mean delivery time is 35.69 minutes with a sample standard deviation of 7.91 minutes (variance 62.51 minutes squared, dividing by n − 1); the middle half of deliveries span 11.29 minutes." "Very consistent" should be dropped unless it is tied to a benchmark, such as a promised delivery window or a comparison process, which these data do not supply. A variance on its own is in squared units and cannot be compared with the mean.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/03-spread/delivery_variation.xlsx) for the published analysis. The [original generator](generators/03-spread.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/03-spread/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/03-spread/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native sd matches centered sum, rescaling sd, rescaling variance, ordered quartiles. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Variance in minutes; standard deviation as a confidence interval; confusing n and n−1; interpreting CV on a scale without a meaningful zero.
