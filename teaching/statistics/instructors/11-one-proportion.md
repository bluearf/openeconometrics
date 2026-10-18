# Instructor companion · Lab 11

## Teaching sequence

Start with the denominator: 137 opt-ins out of 400 invited users, with refusals counted. Write the Bernoulli variance $p(1-p)$ and then the two standard-error formulas side by side, one evaluated at $p_0 = 0.4$ for the test and one at $\hat p = 0.3425$ for the Wald interval, before showing either result.

Before running, collect answers to the pause question. After running, point to the two standard errors in the output (0.0245 in the attributes, 0.0237 in the displayed row) and then introduce the Wilson interval as the set of benchmarks a score test would not reject. Keep fractions, percentages and percentage points distinct throughout.

## Worked exercise answers

**1.** Rejection at 5% requires $|z| \ge 1.96$, that is $|\hat p - 0.4| \ge 1.96 \times 0.02449 = 0.0480$. Below 160, $\hat p \le 0.3520$ means at most $400 \times 0.3520 = 140.8$, so 140 opt-ins is the largest count that rejects ($z = -2.04$, $p = 0.041$); 141 gives $z = -1.94$, $p = 0.052$. Above 160, at least 180 opt-ins are needed ($z = 2.04$, $p = 0.041$); 179 gives $p = 0.052$.

**2.** The gap is $0.3425 - 0.40 = -0.0575$, or 5.75 percentage points below the benchmark. Relative to the benchmark it is $-0.0575/0.40 = -14.4\%$. A report should normally give the percentage-point gap with the two rates, and label any relative change explicitly, because "6% lower" could mean either 6 points or 6% of 40%.

**3.** The 99% critical value is 2.576, so the Wald interval is $0.3425 \pm 2.576 \times 0.02373 = 0.3425 \pm 0.0611$, or 0.281 to 0.404. It contains 0.40, consistent with a two-sided p value (0.019) above 0.01 but below 0.05, since the 95% intervals exclude 0.40. The Wald interval uses the standard error at $\hat p$ while the test uses the standard error at $p_0$, so the two are not exact inverses and can disagree near the boundary. (The 99% Wilson interval, 0.284 to 0.406, also contains 0.40.)

**4.** With $k = 1$ and $n = 10$, $\hat p = 0.1$ and the Wald interval is $0.1 \pm 1.96 \times 0.0949$, or −0.086 to 0.286: its lower limit is an impossible negative probability. The Wilson interval is 0.018 to 0.404 (center 0.211, half-width 0.193) and stays within 0 to 1. Under a benchmark of 0.40 the expected counts are 4 opt-ins and 6 refusals, too few for the normal approximation; an exact binomial test would be appropriate.

**5.** If each of 100 users gave the same answer four times, the data contain only 100 independent responses. The null standard error becomes $\sqrt{0.24/100} = 0.0490$, twice 0.0245, and with the same observed rate $z = -0.0575/0.0490 = -1.17$, two-sided $p = 0.24$. Treating the 400 rows as independent would understate the standard error and turn an unremarkable deviation into a "significant" one.

**6.** For example: "Of 400 invited users, 137 opted in (34.25%), 5.75 percentage points below the 40% benchmark. A two-sided z test gives $p = 0.019$, so the data are inconsistent with a 40% rate at the 5% level; the 95% Wilson interval is 29.8% to 39.0%." The p value is computed assuming the rate is 40%; it is not the probability that the rate is 40%, and $1 - p$ is not the probability that the rate is below 40%.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/11-one-proportion/app_opt_in.xlsx) for the published analysis. The [original generator](generators/11-one-proportion.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/11-one-proportion/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/11-one-proportion/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native z uses null variance, wilson interval inside probability support, binary coding preserved. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Using successes as the denominator; interpreting p as P(null true); confusing a Wald interval with Wilson; ignoring repeated users.
