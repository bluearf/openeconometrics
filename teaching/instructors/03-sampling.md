# Instructor companion · Lab 03

Keep three counts separate on the board: observations per sample, replications of the sample, and the number of independent populations (one fixed mechanism here). The small-sample estimated-scale coverage is an effective discussion trigger: known-scale normal coverage can look approximately nominal while the studentized statistic behaves differently.

## Worked exercise answers

1. With population SD 3.544371466, theoretical SEs at n=20, 80 and 640 are approximately 0.792545554, 0.396272777 and 0.140103584. Increasing n fourfold halves SE; the formula does not depend on the number of Monte Carlo replications.
2. With 4,800 replications, Monte Carlo SEs are approximately half those with 1,200. The estimator's theoretical SE at a fixed n is unchanged. Coverage estimates may change because simulation noise changes.
3. Require separate counts of standardized means below −1.959964 and above +1.959964. Near-95% combined coverage does not imply equal tail errors or a normal sampling distribution. Do not require tails to sum to exactly 5% in finite simulation.
4. Recompute $\mu=\exp(1+1.2^2/2)$ and the corresponding lognormal variance before standardizing. More severe skew and larger tail influence can make normal approximation slower at a fixed n. Grade the actually executed simulation rather than an assumed monotonic pattern in every finite realization.
5. Covariance terms in the variance of the mean cease to vanish. Positive within-sample covariance generally increases the sampling variance relative to the independent formula. Require a specified dependence mechanism and independently simulated samples.
6. Individual SD is 3.544371; one mean's SE is this quantity divided by sqrt(n); the Monte Carlo average of 1,200 independent means has that SE divided again by sqrt(1200). The three spreads refer to different random quantities.

## Scientific and reproduction notes

The experiment uses independent lognormal draws, not a bootstrap from a finite file. Analytic moments provide the oracle. Each n has 1,200 replications and the script checks simulated mean proximity using Monte Carlo SE, empirical SD against the analytic SE, native descriptive results against direct Torch summaries, histogram counts, and the exact square-root ratio. `models` is intentionally empty.

The [full reference](../labs/03-sampling/reference.json) records numerical summaries and figure bins. Native outputs: table followed by three histograms. All normal-reference intervals deliberately use 1.95996398454; no claim of exact Student-t coverage is made for lognormal individual outcomes. Explicit regeneration uses `run_lab(output_dir="docs/teaching/labs/03-sampling")`. Default runs do not write files.
