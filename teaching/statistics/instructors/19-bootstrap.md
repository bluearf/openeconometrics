# Instructor companion · Lab 19

## Teaching sequence

Keep the prepared sample fixed. Start from the skewed baskets (median 17.74, mean 24.93, one basket at 229.88) and the symmetric t interval, then show one resampling index vector with its repeated and missing basket numbers before presenting the distribution of 1,999 means.

Pose the pause question about sampling without replacement before running. After running, compare the bootstrap standard error with its exact empirical target, then the percentile and t intervals, and finish by separating sampling variation from Monte Carlo variation.

## Worked exercise answers

**1.** The first resample (`indices[0]`) contains 57 distinct baskets, 63.3% of 90, close to the expected share 0.634, or 57.08 baskets. Its most repeated basket appears 4 times. Averaged over all 1,999 resamples, the number of distinct baskets is 57.14.

**2.** $2.894/\sqrt{1999} = 0.0647$ currency units. The bootstrap mean differs from the sample mean by $24.892 - 24.928 = -0.036$, about 0.55 Monte Carlo standard errors, which is ordinary simulation noise and no cause for concern.

**3.** With seed 1 the bootstrap standard error is 2.925 and the percentile interval is 20.15 to 31.43, against 2.865 and 20.25 to 31.17 with seed 991019. The baskets and the exact target, 2.894, are unchanged; only the random resamples differ. This is Monte Carlo variation, which more replications would shrink, not sampling variation, which only new data could change.

**4.** Percentile interval: $24.93 - 20.25 = 4.68$ below and $31.17 - 24.93 = 6.24$ above, a ratio of 1.33. The t interval reaches 5.78 on each side, a ratio of exactly 1 by construction. The percentile interval reaches further upward because the baskets are right-skewed: resamples that include the large baskets, especially basket 7, produce high means, and there is no comparably extreme small basket.

**5.** Without basket 7 the 89 remaining baskets have mean 22.62, bootstrap standard error 1.762 and percentile interval 19.41 to 26.16 (t interval 19.05 to 26.20). One basket raises the mean by 2.30 and the bootstrap standard error by about 63%, and with it removed the percentile and t intervals are much closer to each other. The bootstrap can only recombine observed baskets, so its picture of uncertainty depends heavily on whether rare large baskets happen to be in the sample.

**6.** For example: "Resampling the 90 synthetic baskets 1,999 times gives a 95% percentile interval of 20.25 to 31.17 currency units for mean spending. If the baskets are independent and representative, intervals built this way should cover the population mean in roughly 95% of samples." The original wrongly says the bootstrap "proves" something, treats the interval as a probability statement about a fixed mean, and ignores that the interval is approximate and inherits the sample's limitations.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/19-bootstrap/bootstrap_baskets.xlsx) for the published analysis. The [original generator](generators/19-bootstrap.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/19-bootstrap/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/19-bootstrap/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: bootstrap replicates within observed support, bootstrap center close with monte carlo tolerance, bootstrap se close to empirical target, retains all baskets per resample. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Resampling as new data collection; without-replacement bootstrap; percentile intervals as universally reliable; confusing baskets with replication means.
