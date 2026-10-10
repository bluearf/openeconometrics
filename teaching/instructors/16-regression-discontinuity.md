# Instructor notes · Lab 16: What Happens Around an Eligibility Cutoff?

The [student handout](../labs/16-regression-discontinuity/README.md) and
[executable lab](../labs/16-regression-discontinuity/lab.py) use an original
sharp grant-eligibility design. All observations are synthetic. The central
distinctions are local versus whole-group comparisons, point-fit versus bias
bandwidths, and bias-corrected point estimation versus robust bias-corrected
inference.

## Teaching approach

Ask what a comparison of score 49.9 with score 50.1 requires before showing
the regression output. Then ask why the two nearest individual outcomes are
not the conditional-mean jump. The curvature term makes separate local fits
and a bias stage substantive rather than decorative options.

Do not describe bias correction as a guarantee of moving closer to the true
effect in every sample. Here the conventional estimate is 4.198 and the
corrected estimate is 4.579, while the known effect is 3.000. The robust
interval covers three; the middle row's interval and the narrow-bandwidth
robust interval do not. This makes repeated-sample coverage and finite-sample
realizations worth discussing explicitly. Retain the declared seed rather
than selecting a visually convenient result.

## Worked exercise answers

1. The full recipient comparison mixes the jump with the smooth preparation
   terms $0.2(X-50)+0.006(X-50)^2$. RD evaluates the two fitted limits at the
   same score. The target is local to 50, with a causal interpretation requiring
   continuity of the relevant potential outcomes and a valid assignment story.

2. For each side, form $R=(1,X-50)$ and diagonal triangular weights $W_h$.
   The local coefficients are $(R'W_hR)^{-1}R'W_hY$. The difference of the
   right and left intercepts is **4.1984836595**. Separate slopes permit
   different local gradients; forcing one slope is an additional restriction.

3. Point weights are positive for scores strictly between 38 and 50 on the
   left and from 50 to strictly below 62 on the right. No observation lands
   exactly on a boundary in this realization. Counts are **122 and 127**,
   total **249**, from full-side counts **280 and 320**. The bias fit uses
   **175 left and 198 right**, total **373**, with bandwidth 18.

4. Bias correction produces **4.5789526415**. The middle row retains the
   conventional SE **0.7339060690**. The robust row uses SE **0.8941394900**,
   accounting for variability from estimating the bias adjustment. Report the
   robust interval **[2.8264714439, 6.3314338391]**, with normal inference.
   The rows are alternatives for one parameter, not three independent effects.

5. With $(h,b)=(8,12)$, the estimate is **5.394343** and interval
   **[3.224379, 7.564307]** from 159 point-fit observations. With $(16,24)$,
   it is **4.233580**, interval **[2.710455, 5.756706]**, from 334.
   The wider neighborhoods trade locality against precision; none should be
   selected because its p-value or relation to the known truth is attractive.

6. If priority registration also changes at 50, its effect is perfectly
   coincident with grant eligibility. The discontinuity cannot separate the
   two without additional variation or a design argument about the bundled
   policy. Administrative timing, take-up, and outcomes related to each
   mechanism could inform a different analysis.

7. An acceptable paragraph reports a **4.58-credit local discontinuity at
   score 50**, robust interval **[2.83, 6.33]**, triangular local-linear
   bandwidth 12 and quadratic bias bandwidth 18. It conditions attribution on
   continuity, nonmanipulation, and no simultaneous cutoff policy, and states
   that all observations are synthetic. It does not generalize the effect to
   every score or an actual grant program.

**Prepared input:** [scholarship_cutoff.xlsx](../labs/16-regression-discontinuity/scholarship_cutoff.xlsx). Use the stored observations for the published analysis. The [original generator](generators/16-regression-discontinuity.py) supports new dataset editions; update results and answers when changing observations.

## Reproduction and verification evidence

Run the complete source in a fresh session. Explicit exports use
`run_lab(output_dir="rd-output")`; the checked-in
[full reference state](../labs/16-regression-discontinuity/reference.json)
contains three native models and the complete plot and sensitivity data.

`manual_rd` independently builds both local polynomial normal equations,
the bias-corrected influence rows, and HC0 variances. Conventional and robust
point estimates and standard errors match native `oe.rdrobust`; maximum
observed discrepancy is below $6\times10^{-14}$. This checks uncertainty as
well as the coefficient. Assignment, all 600 retained sample positions, and
positive point-fit counts are also checked.

Fresh source execution and Ruff passed. Execution through the actual
`ConsoleSession` worker passed with output order **plot, model, table** and
no worker error. The whole-file result is `rd_lab_result`. Default execution
writes no exports; `reference.json` and `table.tex` require an explicit
destination. These results verify this source runtime, not a separate
installer or hosted deployment.
