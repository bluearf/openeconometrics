# Instructor companion · Lab 20

## Teaching sequence

Before fitting, have students derive why the log-log coefficient is an elasticity by exponentiating the model, and answer the "Pause before running" question in words, including what is held fixed. Point out the two loose dictionary labels: `income` is an index (the dictionary lists "currency"), and `quantity` is the quantity purchased at the market's price, not a demand curve.

After fitting, read the output in three separate passes: the coefficients as elasticities, the HC3 intervals as precision, and R² as fit. Then spend time on Section 6: the estimate is a demand elasticity only because prices were generated independently of demand disturbances, and no regression statistic can show that.

## Worked exercise answers

**1.** For a 20% price cut the exact change is $100(0.8^{-1.22405}-1)=+31.41\%$; the approximation $-20\times(-1.22405)=+24.48\%$ understates it. On a constant-elasticity curve quantity is convex in the price change: price cuts raise quantity by more than the linear approximation, and price rises lower it by less (Section 4's −11.01% versus −12.24%). The linear approximation errs most for large changes.

**2.** Revenue is multiplied by $1.1\times1.1^{\beta_P}=1.1^{1+\beta_P}=1.1^{-0.22405}$, a change of $-2.11\%$. Because demand is elastic ($|\beta_P|>1$), the proportional fall in quantity outweighs the proportional rise in price, so revenue falls.

**3.** `lincom` gives $\hat\beta_P+1=-0.2240$ with HC3 standard error 0.0411, $t=-5.45$ on 177 degrees of freedom ($p\approx1.7\times10^{-7}$) and a 95% interval from −0.305 to −0.143. The interval excludes zero, so unit elasticity is rejected at the 5% level: in this model demand is more elastic than unit elastic. The estimate is simply the price coefficient plus one, so its standard error is unchanged.

**4.** A 10% income increase predicts $100(1.1^{0.52141}-1)=+5.10\%$ quantity at a given price. The positive income elasticity makes the good normal; because it is below one, quantity rises less than proportionally with income (an income-inelastic good, often called a necessity).

**5.** With log price alone the coefficient is −1.2291 (HC3 SE 0.0526), almost identical to −1.2240, while R² falls from 0.832 to 0.750. The coefficient barely moves because log price and log income are nearly uncorrelated in these records (correlation −0.012), so omitting income leaves it in the error without biasing the price coefficient; it only adds residual variance, which lowers R² and widens the standard error. If income were correlated with price, for example if richer markets had higher prices, omitting it would bias the price coefficient toward zero or even make it positive, since the price term would pick up part of the income effect.

**6.** For example: "In 180 synthetic markets with exogenously assigned prices, the fitted constant-elasticity model predicts that a 10% price increase lowers quantity purchased by about 11.0% (exact calculation from an elasticity of −1.224), holding income fixed. The model assumes one elasticity for all markets; the R² of 0.83 describes fit and does not establish that the coefficient is causal, which rests on how prices were assigned."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/20-estimated-demand/estimated_demand.xlsx) for the published analysis. The [original generator](generators/20-estimated-demand.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/20-estimated-demand/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/20-estimated-demand/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: matrix coefficients match, full result roundtrip, all observations used. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The exogeneity argument applies to this mechanism. An actual demand study would need an identification design, instruments or other evidence; a high fit statistic would not establish it.
