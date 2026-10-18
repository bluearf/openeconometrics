# Instructor companion · Lab 18

## Teaching sequence

Start with two units that have the same X and Z but different exposures, and ask what a sensible model should predict for their counts and their rates. Write $\mu_i=e_i\exp(w_i'\beta)$, take logs to show the offset, and derive the score equations so students predict that fitted events will sum to 871 before running.

After running, read the coefficient table on the log-rate scale, convert to rate ratios, and compare the intercept rate with the crude rate. Use the Pearson dispersion and the figure's uneven scatter to separate the mean model (which drives consistency) from the variance assumption (which drives the model-based standard errors).

## Worked exercise answers

**1.** Exponentiating the coefficient interval gives 1.403 to 1.603 for the X rate ratio, a 40% to 60% higher rate per unit of X (point estimate 50%). For two units of X the ratio is $e^{2\times0.4051}=2.249$ with interval $e^{2\times0.3386}=1.968$ to $e^{2\times0.4717}=2.569$, a 125% higher rate (97% to 157%).

**2.** Doubling every exposure changes only the intercept: it falls from 0.3137 to −0.3794, a drop of exactly $\ln2=0.6931$, with the same standard error (0.0393). The X and Z coefficients and their standard errors are unchanged. Because $\ln(2e_i)=\ln2+\ln e_i$, the extra $\ln2$ in the offset is absorbed by the intercept, so fitted counts are identical and fitted rates per new exposure unit are halved. Changing the exposure unit rescales the baseline rate, never the rate ratios.

**3.** With `log_exposure` estimated freely, its coefficient is 0.842 (standard error 0.087, 95% interval 0.671 to 1.013), and the X coefficient is 0.403. The test of a coefficient of one gives $z=(0.842-1)/0.0873=-1.81$, two-sided p about 0.07, so proportionality is not rejected at 5%. A value below one would mean expected events grow less than proportionally with exposure, for example because risk declines with time at risk; the offset model would then misstate rates for short and long exposures.

**4.** Without exposure, the X coefficient is 0.397 (standard error 0.033) and the intercept 0.727, close to the offset model's X coefficient; fitted events still sum to 871 because the intercept score still holds. The X coefficient barely moves because exposure is almost uncorrelated with X in this sample (correlation 0.014). Omitting exposure biases the X coefficient when exposure is related to X, because the X coefficient then absorbs differences in opportunity rather than differences in rate.

**5.** Robust standard errors: intercept 0.0395 (model-based 0.0393), X 0.0291 (0.0340), Z 0.0293 (0.0321). A Pearson dispersion of 1.083 would suggest model-based standard errors about $\sqrt{1.083}=1.04$ times too small if the variance were proportional to the mean, but the sandwich is smaller for X and Z. The dispersion is an average; the sandwich depends on how squared residuals co-vary with $x_i^2$ and $z_i^2$, and with dispersion this close to one the difference mostly reflects sampling noise in the sandwich. Neither statistic indicates a problem here.

**6.** For example: "Holding Z fixed, one more unit of X multiplies the expected number of events per exposure unit by 1.50 (95% interval 1.40 to 1.60). This compares rates, not raw counts; actual event counts also depend on each unit's exposure."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/18-poisson-exposure/poisson_exposure.xlsx) for the published analysis. The [original generator](generators/18-poisson-exposure.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/18-poisson-exposure/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/18-poisson-exposure/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: positive exposure, count support, intercept score zero, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The crude aggregate rate is not a covariate-adjusted effect. Exposure errors, overdispersion, excess zeros and dependence would require additional modeling decisions.
