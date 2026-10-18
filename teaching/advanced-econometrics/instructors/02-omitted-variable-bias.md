# Instructor companion · Lab 02

## Teaching sequence

Show the short and full slopes first and ask where the missing 0.546 went. Derive the identity by substituting the fitted full regression into the short slope's numerator, and stress which normal equations remove the intercept and residual terms. Then state the population formula and contrast the two: one is arithmetic on a sample, the other a probability limit under a model.

Before running, ask students to predict the sign of the gap. After running, keep the decomposition (exact, assumption-free) separate from the decision whether `z` belongs in the model (a causal judgment) and from the uncertainty of the gap (which needs the joint distribution of the fits).

## Worked exercise answers

**1.** The short regression of `y` on `z` has slope 1.4069 and the full `z` coefficient is 0.7358, a gap of 0.6711. The regression of `x` on `z` has slope 0.5482, and $1.2243\times0.5482=0.6711$. The identity is symmetric in which regressor is omitted.

**2.** The `x` slopes (1.7699 short, 1.2243 full) and both gaps (0.5457) are unchanged. The `z` coefficient is divided by 10 (0.07358) and the auxiliary slope is multiplied by 10 (7.416). Rescaling the omitted variable moves units between the two factors; the product, which is in outcome units per unit of `x`, cannot change.

**3.** On the first 120 rows the short slope is 1.8845. Subtracting the 240-row full slope gives 0.6603, which no longer equals the 240-row product 0.5457: the substitution in Section 2 uses the full regression's normal equations, which hold only on the rows it was fitted to. Refitting all three on the first 120 rows restores the identity: controlled slope 1.2927, observed gap 0.5918 and product 0.5918.

**4.** The sample covariance of `x` and `z` is 0.8148 and the sample variance of `x` is 1.0988; their ratio is 0.7416, the auxiliary slope. With $r_{xz}=0.6376$, $s_z=1.2192$ and $s_x=1.0482$, $\hat\pi_x=r_{xz}s_z/s_x=0.7416$ and the gap is $\hat\beta_z r_{xz}s_z/s_x=0.7358\times0.6376\times1.1631=0.5457$.

**5.** With 500 resamples drawn by `numpy.random.default_rng(2026)` and least-squares refits, the bootstrap standard deviation of the gap is about 0.081 and the 2.5–97.5 percentile interval is about 0.397 to 0.718 (bootstrap mean 0.549). Students' values will differ slightly with the seed and the number of draws. Every resample must refit the short, full and auxiliary regressions on the same resampled rows: the identity, and the dependence between $\hat\beta_z$ and $\hat\pi_x$, hold only within one common sample. Within every resample the observed and decomposed gaps agree to about $10^{-14}$.

**6.** For example: "In 240 synthetic units, adding `z` lowers the least-squares slope on `x` from 1.770 to 1.224; the 0.546 difference equals the `z` coefficient (0.736) times the slope of `z` on `x` (0.742). The controlled slope is the relevant comparison only if `z` is a confounder rather than a mediator or collider, and even then it is causal only if no other omitted variable moves with `x`."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/02-omitted-variable-bias/omitted_variable_bias.xlsx) for the published analysis. The [original generator](generators/02-omitted-variable-bias.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/02-omitted-variable-bias/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/02-omitted-variable-bias/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: sample decomposition exact, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** An exact sample decomposition is not a generic justification for adding every available variable. A control’s causal role and measurement quality matter.
