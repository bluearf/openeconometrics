# Instructor companion · Lab 13

## Teaching sequence

Before running, write the unit-intercept model, explain why one dummy must go, and derive the within equation from $M_D$ and the Frisch–Waugh–Lovell theorem. Ask students which records would add nothing to the slope (units whose X never changes).

After the run, confirm the two routes agree, then spend the time on inference: the CR1 factor with $K=32$, the 29 cluster degrees of freedom, why demeaned-data software understates the standard error, and why the dummy standard errors are not usable (the cluster covariance has rank 2).

## Worked exercise answers

**1.** From unit means and the within slopes, $\hat\alpha_0=1.6895$ and $\hat\alpha_1=1.1917$. The dummy regression gives an intercept of 1.6895 and $1.6895+(-0.4978)=1.1917$ for unit 1. The two routes give the same unit levels, not only the same slopes.

**2.** The pooled slope is 1.790 (cluster SE 0.073), well above 1.282. Units with higher mean X have higher recovered intercepts: the correlation between $\bar x_i$ and $\hat\alpha_i$ across the 30 units is 0.85. The pooled regression credits those level differences to X, which pushes its slope upward.

**3.** `fit` keeps an intercept, so the demeaned regression has $K=3$. Its X slope is 1.28195 (the intercept is zero to rounding), but its cluster SE is 0.0805 against 0.0881. It uses $c=\tfrac{30}{29}\cdot\tfrac{179}{177}=1.0462$ instead of 1.2512; because the residuals and the slope block of the sandwich are identical, the SE ratio is $\sqrt{1.0462/1.2512}=\sqrt{148/177}=0.914$, an understatement of about 8.6%. Without an intercept ($K=2$) the SE is 0.0803.

**4.** The unit mean of `x` is constant within units, so it is a linear combination of the intercept and the dummies. The software reports "Collinear terms omitted: unit_29" and keeps the new column. The X slope is unchanged at 1.28195. The new coefficient (0.160) has no meaning as an effect: it only reparameterizes the unit intercepts after one dummy was dropped.

**5.** With period dummies for periods 1–5 added, the X slope is 1.2712 with cluster SE 0.0922 (29 degrees of freedom). The period dummies remove shocks common to all units in a period, which unit effects alone leave in the within deviations.

**6.** For example: "In a balanced synthetic panel of 30 units over 6 periods, within-unit variation gives a fixed-effects slope of 1.282 outcome units per X unit, holding Z fixed (standard error clustered by unit 0.088, 29 degrees of freedom; 95% interval 1.102 to 1.462). The estimate removes time-invariant unit differences and is causal only if within-unit errors satisfy strict exogeneity and no time-varying confounder moves with X."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/13-fixed-effects-algebra/fixed_effects_algebra.xlsx) for the published analysis. The [original generator](generators/13-fixed-effects-algebra.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/13-fixed-effects-algebra/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/13-fixed-effects-algebra/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: within dummy slope equivalence, balanced unique panel time, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Fixed effects remove time-invariant unit components, not time-varying confounding. Time-invariant regressors cannot be separately identified from unit intercepts.
