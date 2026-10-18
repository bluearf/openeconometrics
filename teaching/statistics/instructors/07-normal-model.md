# Instructor companion · Lab 07

## Teaching sequence

Start by writing the model (mean 1,000 g, SD 12 g) and the batch summary (999.67 g, 11.51 g) side by side, and insist on separate symbols for each. Convert a few weights to standard scores and back before showing any probability, so students treat z as a position on a rescaled axis rather than a probability.

Before running, collect predictions for the pause question: should the mean of the population-standardized scores be exactly zero? Use the output (−0.028 and 0.959) to show the two linear relations from Section 3, then introduce $\Phi$ and the tail area. Finish by contrasting the model probability 0.1587 with the observed fraction 0.138 and asking what kind of variation separates them.

## Worked exercise answers

**1.** $(1006-1000)/12 = 0.5$: half a model standard deviation above the mean. $(979-1000)/12 = -1.75$: one and three-quarter standard deviations below. A score of −1.5 corresponds to $1000 + 12(-1.5) = 982$ grams. None of these is a probability.

**2.** 478 of the 500 parcels weigh between 976 and 1,024 grams, an observed fraction of 0.956, against the model probability 0.9545. Eleven parcels weigh 976 grams or less and eleven weigh 1,024 grams or more. The agreement is close, but with only 22 parcels outside the range the tails are thinly observed.

**3.** By symmetry, $P(W<988) = P(Z<-1) = P(Z>1) = 0.1587$, and $P(988<W<1012) = 1 - 2(0.1587) = 0.6827$. In the batch, 76 parcels (0.152) weigh less than 988 grams and 355 (0.710) weigh between 988 and 1,012 grams; together with the 69 above 1,012 grams they account for all 500.

**4.** With the sample mean 999.67 g and sample SD 11.51 g, a sample-based score of 1 corresponds to $999.67 + 11.51 = 1011.18$ grams. These scores have mean 0 and SD 1 by construction (up to floating-point rounding). 77 parcels (0.154) exceed 1,011.18 grams, eight more than the 69 above 1,012 grams, because the threshold moved down by about 0.82 grams: the batch mean is 0.33 g below 1,000 and its SD is 0.49 g smaller than 12. The sample-based score answers "how unusual within this batch," not "how far from the stated process."

**5.** The expected count is $500 \times 0.1587 = 79.33$ and its standard deviation is $\sqrt{500 \times 0.1587 \times 0.8413} = 8.17$ parcels. The observed 69 lies $(69-79.33)/8.17 = -1.26$ standard deviations from the expectation. Equivalently, the fraction 0.138 is 0.0207 below 0.1587 with a model standard error of 0.0163. A deviation of about 1.3 standard deviations is common under the model, so the shortfall is not surprising and gives no strong evidence against it.

**6.** For example: "A weight of 1,012 grams is a standard score of 1 relative to the stated model (mean 1,000 g, SD 12 g). If that model holds, the probability that a parcel exceeds 1,012 grams is 0.159; in this batch, 69 of 500 parcels (0.138) did. A bell-shaped histogram is consistent with normality but does not establish it." Relevant evidence for a real line includes the distribution's shape in the tails (for example a large sample or a normal quantile plot), stability of the process over time (control charts, drift and shifts), scale calibration and resolution, and dependence between consecutive parcels.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/07-normal-model/parcel_weights.xlsx) for the published analysis. The [original generator](generators/07-normal-model.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/07-normal-model/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/07-normal-model/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: linear standardization mean, linear standardization sd, tail complement. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Treating a z score as a probability; forcing sample moments to equal population moments; normality from standardization alone; extrapolating a synthetic quality model to production.
