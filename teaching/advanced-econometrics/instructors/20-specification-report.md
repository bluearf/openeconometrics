# Instructor companion · Lab 20

## Teaching sequence

Before running, state the estimand (the sample average effect of randomized assignment) and write the three specifications. Ask students what each treatment coefficient estimates, and point out that `interaction` uses the raw, uncentered X (mean 0.048) even though the workbook dictionary calls it centered.

After running, read the specification table for sample, estimate, standard error and target together. Then derive $\hat\tau+\hat\delta\bar x$ and its variance $a'\hat Va$ by hand from the covariance matrix before comparing with `oe.lincom`. Close with what a report must retain and with the difference between conditioning on the observed covariates and targeting a population.

## Worked exercise answers

**1.** Arm means of the outcome are 3.1903 (156 treated) and 2.0511 (144 controls); their difference, 1.1393, is the unadjusted coefficient. With outcome variances $s_1^2$ and $s_0^2$ by arm, $\sqrt{s_1^2/155+s_0^2/143}=0.16783$, identical to the reported HC3 standard error: with a single binary regressor every leverage is $1/n_g$, so HC3 divides each arm's sum of squared residuals by $(n_g-1)^2$. The unequal-variance (Neyman) form $\sqrt{s_1^2/156+s_0^2/144}=0.16727$ is slightly smaller; HC2 would reproduce it exactly.

**2.** Arm means: X 0.0197 (treated) and 0.0779 (control), a gap of −0.0582; Z 0.1644 and 0.1529, a gap of 0.0115. With adjusted coefficients $\hat\beta_x=0.7536$ and $\hat\beta_z=0.2983$, $1.1796+0.7536(-0.0582)+0.2983(0.0115)=1.1796-0.0438+0.0034=1.1393$, the unadjusted estimate exactly. The adjusted residuals are orthogonal to the intercept and to `treat`, so they average zero in each arm; taking arm means of the fitted adjusted equation gives the identity. Here the treated arm happens to have lower baseline X, which depresses the raw difference by about 0.044.

**3.** With `treat * (x - x.mean())`, the treatment coefficient is 1.179483 with HC3 standard error 0.135705 and interval 0.9124 to 1.4466, exactly the `lincom` result; the other coefficients, including the interaction slope 0.648191, are unchanged. Since $D(x-\bar x)=Dx-\bar xD$, the new treatment coefficient is $\tau+\delta\bar x$. The regressors span the same column space, so fitted values, residuals and leverages are identical, and the HC3 covariance transforms linearly with the reparametrization.

**4.** At X = −1: contrast 0.500, standard error 0.213, 95% interval 0.081 to 0.920. At X = 1: 1.797, standard error 0.199, interval 1.406 to 2.188. The variance $\hat V_{\tau\tau}+x^2\hat V_{\delta\delta}+2x\hat V_{\tau\delta}$ is a quadratic in $x$ with its minimum at $x=-\hat V_{\tau\delta}/\hat V_{\delta\delta}=0.062$, close to the mean of X where the data are densest; moving away in either direction adds $x^2\hat V_{\delta\delta}$.

**5.** $s_x=1.0300$, so $\hat\delta^2s_x^2/n=0.6482^2\times1.0609/300=0.001486$. The total variance is $0.018416+0.001486=0.019902$, a standard error of 0.1411 (95% interval about 0.902 to 1.457) against 0.1357. The cross term is asymptotically zero because the coefficient estimates' errors are uncorrelated with the sampling error of $\bar x$ when the conditional mean is correctly specified. The 0.1357 standard error refers to the average effect over these 300 units' observed baseline values; the 0.1411 standard error refers to a population average effect whose X distribution is estimated by this sample.

**6.** For example: "The treatment coefficient 1.149 is the fitted contrast at baseline X = 0; the contrast rises by 0.648 per unit of the uncentered baseline X. Averaged over the sample, where X has mean 0.048, the fitted contrast is 1.179 (HC3 standard error 0.136)."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/20-specification-report/specification_report.xlsx) for the published analysis. The [original generator](generators/20-specification-report.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/20-specification-report/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/20-specification-report/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: interaction preserved, average effect identity, same sample all specifications, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Specification choices should be stated before searching for significance. This original randomized simulation does not establish an empirical policy effect; population generalization requires a target population and sampling argument.
