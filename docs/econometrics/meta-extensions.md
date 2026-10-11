# Meta-analysis extensions: three-level models, CR2 inference and bias sensitivity

This page covers the MARKET-174 extensions that sit beside
[independent-study meta-analysis](../meta-analysis.md) and
[dependent effects with a declared sampling covariance](dependent-meta.md):

| Function | Kind | Purpose |
| --- | --- | --- |
| `oe.meta_multilevel` | procedure (`EXPORTS`) | Three-level random-effects meta-analysis and meta-regression, ML/REML |
| `oe.meta_multilevel_predict` | procedure | Saved-state conditional means and new-study latent-effect intervals |
| `oe.meta_cr2` | procedure | CR2 study-robust covariance, Satterthwaite t and HTZ joint tests |
| `oe.meta_trimfill` | procedure | Duval–Tweedie trim-and-fill (L0/R0) |
| `oe.meta_pet_peese` | procedure | PET, PEESE and conditional PET-PEESE |
| `oe.meta_selection` | procedure | Vevea–Hedges step-function selection model, estimated or fixed weights |

Every route is resident CPU float64 Torch with complete inputs; there is no
missing-row deletion, Dataset/accelerator route, observation weighting or
categorical expansion. Funnel asymmetry, trim-and-fill, PET-PEESE and
selection models are model-dependent sensitivity analyses: none of them
establishes publication bias or recovers the true effect.

## Three-level random effects (`meta_multilevel`)

Effects `i` are nested in independent studies `j`:
`y_ij = x_ij' beta + u_j + w_ij + e_ij`, `u_j ~ N(0, sigma2_study)`,
`w_ij ~ N(0, sigma2_effect)`, `e_j ~ N(0, S_j)` (Konstantopoulos 2011,
*Research Synthesis Methods* 2:61–76). The marginal block is
`V_j = S_j + sigma2_study 11' + sigma2_effect I`, which is the model fitted by
the metafor author's `rma.mv(yi, V, random = ~ 1 | study/effect)`
([reference](https://wviechtb.github.io/metafor/reference/rma.mv.html)).
`S` comes from a `vi` column (independent sampling errors) or a labelled
covariance DataFrame whose axes are the effect labels; cross-study entries
must be exactly zero and each within-study block exactly symmetric SPD with
condition ratio at most 1e10. Exactly one of `vi`/`covariance` is required.

**Likelihood.** ML is `-1/2 [N log 2pi + log|V| + r'V^-1 r]` at the GLS
coefficients; REML is `-1/2 [(N-p) log 2pi + log|V| + log|X'V^-1X| - log|X'X| + r'V^-1 r]`
(the full-REML constant used by metafor). Both components are optimized
jointly over the nonnegative quadrant. The solver evaluates the origin, both
one-component faces (each by a bounded 61-point log grid with every
positive-to-nonpositive score bracket refined by safeguarded Newton/bisection),
and interior log-scale Newton starts from a 15×15 batched grid, followed by a
natural-scale Newton polish. The largest likelihood is retained and must pass a
KKT check: interior components have scaled score `|s_a|(1+sigma2_a)/N <= 1e-7`,
zero components a nonpositive scaled score. All scores and observed Hessians
are analytic (`tr(P V_a)`, `y'P V_a P y`, `tr(P V_a P V_b)`,
`y'P V_a P V_b P y`), computed on padded study blocks without an N×N matrix.

**Inference.** Coefficients use the plug-in GLS covariance `(X'V^-1X)^-1` with
`z` or `t(N-p)` (metafor `test="t"`, residual df) and `QM` as chi-square or
`F(m, N-p)`. `QE` uses the sampling covariance only (df `N-p`). Each variance
component reports: the observed-information Wald standard error (inverse
negative Hessian of the profiled likelihood in the variance scale, as metafor's
`cvvc`; omitted when a component is on the zero boundary), a likelihood-ratio
statistic against the model with that component fixed at zero with both the
naive chi-square(1) p-value and the 50:50 boundary-mixture p-value, and a
profile-likelihood interval (`confint.rma.mv` convention: drop of
`chi2_{1,level}/2`; lower limit 0 when the zero profile is inside; an upper
limit beyond the admitted variance range is reported as null).
Multilevel I² follows the author note
([I² for multilevel models](https://www.metafor-project.org/doku.php/tips:i2_multilevel_multivariate)):
typical sampling variance `(N-p)/tr(P_0)` with `P_0` from `W = S^-1`.
AIC/BIC/AICc count `p+2` parameters and use `N-p` for REML.

**Prediction and state.** `meta_multilevel_predict` reads the checksum-protected
state (design, outcome, every within-study sampling block, labels, components,
coefficients, likelihood receipts and component inference), replays GLS at the
saved components without the optimizer and rejects any incoherent or tampered
state. Intervals add `sigma2_study + sigma2_effect` (a new study) to the mean
variance and use the fitted z/t rule; future sampling error and component
uncertainty are excluded.

Limits: 3..2000 effects, 2..1000 studies, at most 100 effects per study and
200,000 padded block entries, 1..16 coefficients, 50,000 likelihood
evaluations; at least one study must contain two effects (otherwise the two
components are not separately identified). Failure codes include
`invalid_spec`, `missing_values`, `invalid_covariance`, `covariance_alignment`,
`unsupported_dependence`, `unidentified_components`, `singular_design`,
`variance_not_converged`, `resource_limit`, `unsupported_device`,
`unsupported_weights`, `unsupported_dataset` and `invalid_result_state`.

## CR2 small-sample inference (`meta_cr2`)

`meta_cr2(result, contrasts=None, null=None)` accepts a saved
`meta_multilevel` or `meta_dependent` fit (including one already passed through
`meta_dependent_robust`). The working model is the fitted inverse-variance
model: `Theta_j = V_j`, `W_j = V_j^-1`, `M = (X'WX)^-1`. With `Theta_j = L_j L_j'`,
the bias-reduced-linearization adjustment is
`A_j = L_j [L_j'(Theta_j - X_j M X_j')L_j]^-1/2 L_j'` and
`V^CR2 = M (sum_j X_j'W_j A_j e_j e_j' A_j' W_j X_j) M` (Tipton 2015;
Pustejovsky and Tipton 2018, *JBES* 36:672–683). For a contrast `c`, with
`p_j = M X_j'W_j A_j` and `P_jk = c'p_j(delta_jk Theta_j - X_j M X_k')p_k'c`,
the Satterthwaite df is `(sum_j P_jj)^2 / sum_jk P_jk^2`. Joint hypotheses
use the HTZ approximation: `Omega = sum_j P_jj`, standardized blocks
`B_jk = Omega^-1/2 P_jk Omega^-1/2`, `nu = q(q+1)/sum_st sum_jk (B_jk[s,t]B_jk[t,s] + B_jk[s,s]B_jk[t,t])`,
`F = (nu-q+1)/(nu q) Q ~ F(q, nu-q+1)`. These are the conventions of
[clubSandwich](https://github.com/cran/clubSandwich) `vcovCR(type="CR2")`,
`coef_test(test="Satterthwaite")` and `Wald_test(test="HTZ")` for rma.mv fits
without user weights; the block decomposition avoids clubSandwich's dense
cluster arrays. The default joint test covers all moderators; labelled
contrasts (a DataFrame naming every coefficient) with scalar, list or labelled
Series nulls add per-row Satterthwaite tests and their joint HTZ test.

Differences: a study whose adjustment block is singular (full leverage, for
example a moderator that is an indicator of one study) raises
`degenerate_inference`; clubSandwich would substitute a generalized inverse.
HTZ p-values are null when `nu-q+1 <= 0`. Coefficients and variance
components are never refitted; Satterthwaite/HTZ df are working-model
approximations, not exact small-sample guarantees.

## Trim-and-fill (`meta_trimfill`)

Duval and Tweedie (2000) with the metafor implementation
([reference](https://wviechtb.github.io/metafor/reference/trimfill.html)):
the side is declared or set by the sign of the `sqrt(vi)` meta-regression slope
(negative slope: missing on the right). Effects are sorted (stable), the
trimmed intercept-only fit (common, DL, ML or REML) is recomputed until the
missing-study count repeats, ranks of `|y-beta|` break ties by position
(`ties.method="first"`), L0 is `(4 S_r - k(k+1))/(2k-1)` with its standard
error and R0 is the rightmost run length minus one with `p = 0.5^(k0+1)`.
Filled studies mirror the `k0` most extreme effects around the final trimmed
estimate and keep their variances; the adjusted model refits observed plus
filled studies with the same estimator and inference. Up to 1000 observed
studies; an undefined R0 run, a trimmed sample below two studies or
non-convergence within `max_iterations` raise explicit errors.

## PET, PEESE and conditional PET-PEESE (`meta_pet_peese`)

PET regresses `yi` on `sqrt(vi)` and PEESE on `vi` by weighted least squares
with weights `1/vi` and an unrestricted multiplicative dispersion
(`RSS_w/(k-2)`), reporting full 2×2 covariances and `t(k-2)` inference
(Stanley and Doucouliagos 2014, *Research Synthesis Methods* 5:60–78). The
conditional estimate reports the PEESE intercept when the PET intercept test
rejects at the declared `alpha` and `alternative`, otherwise PET; the rule is
recorded, and the reported interval does not account for this selection step.
At least ten studies with varying standard errors are required.

## Step-function selection models (`meta_selection`)

One-sided p-values `p_i = 1 - Phi(y_i/sqrt(v_i))` (or `Phi(...)` for
`alternative="less"`) fall into intervals `(a_{j-1}, a_j]` defined by the
declared cutpoints (1 closes the last interval). The log likelihood is
`sum_i [log omega_{g(i)} + log phi(y_i; x_i'beta, v_i+tau2) - log sum_j omega_j B_ij]`
with `B_ij` the normal probability of interval `j` (Hedges 1992; Vevea and
Hedges 1995, *Psychometrika* 60:419–435), `omega_1 = 1`. Weights are
estimated or, for Vevea and Woods (2005) sensitivity analyses, fixed.
Estimation maximizes an analytic gradient over `(beta, log tau2, log omega)` by
BFGS with a Newton polish; the zero-heterogeneity boundary is fitted
separately, compared and KKT-checked. Standard errors come from the
observed information in natural parameters (`beta`, `tau2`, `omega`; a zero
`tau2` is fixed), as in metafor's `selmodel`
([reference](https://wviechtb.github.io/metafor/reference/selmodel.html));
weights are tested against 1 with Wald z statistics and lower CI limits are
truncated at 0. The likelihood-ratio test compares with the unadjusted ML
random-effects (or common-effect) model on `J-1` df; a within-model LRT tests
`tau2 = 0`. Empty intervals with estimated weights (`empty_interval`), weights
running outside `[1e-6, 1e6]` (`selection_not_identified`) and failed
optimization (`selection_not_converged`) are refused; two-sided p-value models
are not offered.

## Validation evidence

`tests/test_econ_meta_extensions.py` (52 tests) compares against the committed
fixture produced by `scripts/generate_meta_extension_oracles.{py,R}` (R 4.6.1,
metafor 5.2-1, metadat 1.6-0, clubSandwich 0.7.0, weightr 2.0.2; see
`tests/fixtures/meta_extensions/README.md`), including the published
Konstantopoulos (2011) and Hackshaw (1998) records, and against dense
NumPy/SciPy likelihoods and statsmodels WLS computed in the test. Measured
maximum deviations on 2026-10-10:

| Comparison | max abs | max rel |
| --- | --- | --- |
| three-level sigma2 vs rma.mv (BFGS reltol 1e-15) | 9.5e-9 | 4.1e-7 |
| three-level coefficients / covariance | 3.1e-9 / 2.6e-9 | 3.2e-8 / 1.4e-7 |
| three-level log likelihood, LRT, profile limits | 1.8e-14, 2.1e-13, 5.1e-14 | 8.3e-13 |
| component Wald covariance vs `cvvc` | 3.5e-10 | 2.1e-7 |
| CR2 covariance (three-level, dependent) | 3.7e-10 | 3.7e-7 |
| Satterthwaite df / HTZ df2 | 3.5e-8 | 7.3e-9 |
| trim-and-fill k0, se, adjusted fit, filled effects | 5.4e-13 | 2.3e-11 |
| PET/PEESE coefficients and covariance vs lm | 2.7e-14 | 6.7e-14 |
| selection log likelihood / LRT vs selmodel | 1.2e-10 / 2.4e-10 | 3.7e-11 |
| selection parameters vs selmodel | 9.2e-6 | 1.4e-5 (beta near zero: 1.3e-6 abs) |
| selection parameters vs weightr (L-BFGS-B defaults) | 2.0e-3 | 9.2e-3 |

For the zero-boundary three-level case and every selection model the
likelihood reached here is never lower than the reference optimizer's
(asserted in the tests), so the remaining parameter differences reflect the
reference stopping rules; three-level estimates are also checked against an
independent dense NumPy/SciPy maximization and selection stationarity
against an independent NumPy likelihood. This is agreement with
development references, not licensed vendor parity. Simulation-based coverage
of the new procedures, installed-application acceptance and public release
evidence are not established here.
