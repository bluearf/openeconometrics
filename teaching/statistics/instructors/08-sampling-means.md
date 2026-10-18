# Instructor companion · Lab 08

## Teaching sequence

Use separate labels for population size $N$, sample size $n$ and replications $B$ from the start; students often confuse all three. Show the skewed population first (SD 34.90 above a mean of 31.25; only 322 of 1,000 people spend more than the mean) so that the later symmetry of the $n=40$ means is visibly a property of averages, not of people.

Derive $\operatorname{Var}(\bar X)=\sigma^2/n$ from the sum of independent variances before running, then collect predictions for the pause question. After running, read the table in pairs (centers, then spreads) and separate sampling error, which depends on $n$, from Monte Carlo error, which depends on $B$. Point out that the two histogram panels have different horizontal scales.

## Worked exercise answers

**1.** With divisor $N=1000$, $\sigma = 34.9047$; with divisor $N-1=999$, the value is 34.9222. The ratio is $\sqrt{1000/999}=1.0005$, a difference of 0.017 currency units. Divisor $N$ is correct because the file is the whole population. The choice matters more when the number of values is small, for example a sample of 5, where $\sqrt{5/4}=1.118$.

**2.** $34.9047/\sqrt{10} = 11.04$ and $34.9047/\sqrt{160} = 2.76$ currency units per day. A standard error of at most 2 requires $n \ge (34.9047/2)^2 = 304.6$, so $n = 305$.

**3.** Empirical over theoretical: $15.0153/15.6099 = 0.962$ at $n=5$ and $5.5149/5.5189 = 0.999$ at $n=40$. The empirical ratio $15.0153/5.5149 = 2.723$ compares with $\sqrt8 = 2.828$. The discrepancies are Monte Carlo error from using 1,600 replications; the estimated standard deviation of a skewed sampling distribution (as at $n=5$) is itself relatively noisy, so the larger gap appears there.

**4.** Monte Carlo standard errors are $34.9047/\sqrt{5\times1600}=0.390$ for $n=5$ and $34.9047/\sqrt{40\times1600}=0.138$ for $n=40$. The gaps are $31.1490-31.2451=-0.096$, or $-0.25$ Monte Carlo standard errors, and $31.4171-31.2451=+0.172$, or $+1.25$. Neither is larger than Monte Carlo error would readily produce. With $B=6400$ the Monte Carlo standard errors halve to 0.195 and 0.069; the standard error of an individual sample mean, $\sigma/\sqrt n$, does not change.

**5.** The correction is $\sqrt{995/999}=0.998$ for $n=5$, giving 15.58, and $\sqrt{960/999}=0.980$ for $n=40$, giving 5.41 currency units. A 10% reduction requires $\sqrt{(1000-n)/999}=0.9$, so $1000-n = 0.81 \times 999 = 809.2$ and $n \approx 191$, about 19% of the population. The correction is small here because both sample sizes are tiny fractions of $N$.

**6.** The population did not change; only the distribution of sample means narrowed. For example: "The spending of individual people is unchanged (population SD 34.90). Means of samples of 40 vary less across samples (SD 5.51) than means of samples of 5 (SD 15.02), because averaging more independent draws lets high and low values offset each other."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/08-sampling-means/spending_population.xlsx) for the published analysis. The [original generator](generators/08-sampling-means.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/08-sampling-means/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/08-sampling-means/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: means within population support, larger samples have less dispersion, mean close relative to monte carlo error. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** CLT as normality of raw data; population SD as the SE; increasing replications as increasing sample size; applying a finite-population correction to with-replacement draws.
