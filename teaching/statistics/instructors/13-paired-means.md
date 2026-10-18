# Instructor companion · Lab 13

## Teaching sequence

Have students find worker 9's blank cell and state the complete-pairs rule before any code runs, then write the direction after − before on the board. Ask them to compare the three standard deviations in the statistics table before the test, so the small spread of changes is the puzzle that the variance identity resolves.

Work the identity with the lab's covariance, then contrast the paired standard error with the incorrect two-group calculation. Use the scatterplot to locate the diagonal of no change, and close by separating a before-and-after difference from an intervention effect.

## Worked exercise answers

**1.** The mean of all 60 `before` values is 29.24 and the mean of the 59 `after` values is 31.45, a difference of 2.20 tasks per day, about 0.044 below the paired mean change of 2.25. The two means now describe different sets of workers. Worker 9's baseline value, 31.83, is above the 59-worker baseline mean of 29.20, so including it raises the before mean and shrinks the apparent change. The gap is small here only because one worker is missing.

**2.** With the column standard deviations, SE = √(6.025²/59 + 5.371²/59) = √(0.615 + 0.489) = 1.051 tasks per day and t = 2.246/1.051 = 2.14 (Welch df 114.5, p = 0.035, 95% interval 0.16 to 4.33). The estimate is the same, but the standard error is about three times the paired 0.352 because the two-group formula ignores the covariance of 28.92 between a worker's two measurements. It treats differences in workers' levels as noise in the change.

**3.** With r = 0.5, $s_d^2 = 36.30 + 28.85 - 2(0.5)(6.025)(5.371) = 65.15 - 32.36 = 32.79$, so $s_d = 5.73$, SE = 5.73/√59 = 0.745 and t = 2.246/0.745 = 3.01. Weaker pairing leaves more of the worker-level variation in the changes. At r = 0 the paired standard error would equal the two-group value from Exercise 2, 1.051.

**4.** 44 of the 59 workers have a positive change and 15 have a negative change; none is exactly zero. A positive mean change says the average worker in this group produced more at follow-up. It does not say every worker improved: about a quarter of them declined, and the mean says nothing about which ones.

**5.** `oe.ttest(pairs, "after", paired_with="before", mu=2, missing="raise")` reports a difference from the benchmark of 0.246 tasks per day, t = 0.697 on 58 degrees of freedom and a two-sided p value of 0.489; its interval for $\mu_d - 2$ is −0.46 to 0.95. The data do not reject a mean gain of exactly 2. The 95% interval for the change, 1.54 to 2.95, contains values on both sides of 2, so the data neither establish that the gain is at least 2 nor rule it out. Showing "at least 2" would need a one-sided test chosen in advance; its p value here is 0.244.

**6.** For example: "Among 59 synthetic workers with both measurements, completed tasks were 2.25 per day higher at follow-up than at baseline on average (95% interval 1.54 to 2.95). The data contain no comparison group, so the change cannot be attributed to any program." Defending the original sentence would need a comparison group of similar workers who did not take part, measured at the same two times, preferably with participation assigned at random, and an analysis of the difference in their changes.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/13-paired-means/worker_before_after.xlsx) for the published analysis. The [original generator](generators/13-paired-means.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/13-paired-means/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/13-paired-means/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: paired se uses differences, t orientation after before, variance of difference identity, unique pairs. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Counting measurements as independent workers; reversing the subtraction; retaining an incomplete pair; treating time change as treatment causality.
