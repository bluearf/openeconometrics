# Wild bootstrap for IV, two-way clustering and binary ML

Three restricted (null-imposed) wild bootstrap routes extend the one-way OLS
`wild_cluster_test` / `wild_cluster_confidence_set`:

| Function | Method | Model and clustering |
| --- | --- | --- |
| `oe.iv_wild_test`, `oe.iv_wild_confidence_set` | Wild restricted efficient (WRE) bootstrap, Davidson and MacKinnon (2010); 2SLS t and Anderson–Rubin tests; test inversion | linear IV, one endogenous regressor; heteroskedastic or one-way clusters |
| `oe.wild_multiway_test` | Two-way wild cluster restricted (WCR) bootstrap, MacKinnon, Nielsen and Webb (2021) | unweighted OLS result, two cluster dimensions |
| `oe.score_bootstrap_test` | Score bootstrap, Kline and Santos (2012) | binary logit/probit ML; heteroskedastic or one-way clusters |

Common contract. CPU float64 Torch; private seeded generator
(`torch.Generator(cpu).manual_seed(seed)`), the B × G multiplier matrix drawn
in replication-row blocks (CPU sampling is serial, so the stream does not
depend on the block size — tested); Rademacher (±1), Webb six-point
(±√1.5, ±1, ±√0.5) or Mammen two-point multipliers, all float64.
P-values are plus-one Monte Carlo `(1 + #exceedances)/(B + 1)`; draws within a
relative 1e-10 of the observed statistic count as exceedances (wild draws can
reproduce the observed statistic exactly, for example when every cluster
receives the same multiplier, and rounding must not break that tie in the
anti-conservative direction). Any replication with a zero or singular
studentizing variance raises `bootstrap_failure`; nothing is dropped.
`max_work` bounds the estimated floating-point work before allocation.
Weights are not supported by these routes. Reported statistics carry the
HC1/CR1-type factors; those factors cancel between the observed and bootstrap
statistics, so p-values do not depend on them.

## IV: wild restricted efficient bootstrap

```python
res = oe.iv_wild_test(data=df, y="wage", endog="educ", instruments=["qob1", "qob2"],
                      x=["exper"], null=0.08, cluster="state", wild="webb", reps=9999)
ar = oe.iv_wild_test(data=df, y="wage", endog="educ", instruments=["qob1", "qob2"],
                     x=["exper"], null=0.08, method="ar", cluster="state")
cs = oe.iv_wild_confidence_set(data=df, y="wage", endog="educ",
                               instruments=["qob1", "qob2"], x=["exper"],
                               method="ar", cluster="state", reps=999)
```

Model `y1 = β y2 + X1 γ + u1`, `y2 = X1 π1 + X2 π2 + u2`. Under `H0: β = β0`:

1. restricted structural OLS of `y1 − β0 y2` on `X1` gives `γ̃` and `ũ1`;
2. the efficient reduced form is OLS of `y2` on `[X1, X2, ũ1]`; `π̃` are its
   `X = [X1, X2]` coefficients and `Ũ2 = y2 − X π̃` (Davidson and MacKinnon's
   "efficient" step; for this null it coincides with the restricted LIML
   system estimates used by boottest);
3. each draw multiplies **both** residual vectors by the same multiplier of
   the observation (heteroskedastic) or cluster: `y2* = X π̃ + Ũ2 v*`,
   `y1* = β0 y2* + X1 γ̃ + ũ1 v*`;
4. the 2SLS t statistic is recomputed with the same studentization.

The 2SLS estimate and robust variance are evaluated by Frisch–Waugh with
`z = M_X1 P_X y2`: `β̂ = z'y1 / z'y2`, `û = M_X1(y1 − β̂ y2)` and
`Var(β̂) = c Σ_g (Σ_{i∈g} z_i û_i)² / (z'z)²`, exactly the β element of the full
2SLS sandwich, with `c = n/(n−k)` (HC1) or `G/(G−1)·(n−1)/(n−k)` (CR1),
`k = k1 + 1`. All draws are vectorised: per draw only n-vectors are
projected on the fixed instrument bases. `tail` gives the symmetric |t|
(default), equal-tail, upper or lower p-value.

Anderson–Rubin (`method="ar"`): the robust Wald statistic of the excluded
instruments in the OLS regression of `y1 − β0 y2` on `[X1, X2]` (asymptotic
χ²(l2) shown as a diagnostic). Its WRE bootstrap perturbs only that
regression's restricted residuals, `(y1 − β0 y2)* = X1 γ̃ + ũ1 v*` — a
restricted wild (cluster) bootstrap of the AR regression, as in Roodman et al.
(2019, section 6.1). AR jointly tests `β = β0` and the exclusion restrictions
and does not require identification.

Confidence sets. `iv_wild_confidence_set` inverts either test on a grid
(caller grid, or 2SLS estimate ± `width` robust SE over `points` values),
with the **same** multipliers for every null so that `p(β0)` is a fixed
function. Between adjacent grid points with different decisions the boundary
is bisected `refine` times; each reported boundary is a certified bracket
`[rejected side, accepted side]` of width `gap / 2^refine` (tested by
re-evaluating both ends). Crossings between two grid points with the same
decision cannot be excluded; the maximum grid gap is reported. Components may
be empty, one interval or disjoint. For AR the tails are decided by the limit
`β0 → ±∞`: AR is scale invariant in `y1 − β0 y2`, so the limit is the wild
bootstrap Wald test of the instruments in the first-stage regression of
`y2`; if it accepts, an edge-touching component is reported as unbounded,
otherwise the side beyond the grid stays "not determined". For the 2SLS t
inversion, tails beyond the grid are always "not determined".

Domain: one endogenous regressor (several are refused), at least one excluded
instrument, exogenous regressors and categorical exogenous terms through the
usual IV design and collinearity screen, complete rows or the declared
missing policy, heteroskedastic or one-way clusters, no weights or absorbed
fixed effects, 2SLS only (no LIML/GMM/k-class bootstrap estimators).

## Two-way clustering: WCR bootstrap

```python
fit = oe.ols(data=df, y="y", x=["treat", "x"])
res = oe.wild_multiway_test(fit, data=df, null={"treat": 0.0},
                            clusters=["state", "year"], bootstrap_cluster="fewest",
                            reps=9999, wild="webb")
```

Following the "Multiway Wild (Cluster) Bootstrap Algorithms" of MacKinnon,
Nielsen and Webb (CREATES Research Paper 2020-06, published in JBES 2021):
the statistic uses the three-term CRVE
`V3 = (X'X)⁻¹[c_G M_G + c_H M_H − c_I M_I](X'X)⁻¹` with
`c_D = |D|(N−1)/((|D|−1)(N−k))` and `I` the nonempty intersections (their
equation (8) factors); when V3 is not positive semidefinite it is replaced by
its eigenvalue truncation (Cameron, Gelbach and Miller), in the sample and in
every draw (counts reported). The WCR bootstrap uses restricted OLS,
multiplies the restricted residuals by multipliers clustered by **one**
dimension, the intersection, or observations (`bootstrap_cluster`:
`"fewest"` default — the dimension with fewer clusters, for which the paper's
simulations show modest evidence —, a cluster column name,
`"intersection"`, `"observation"`), regenerates `y* = X b̃ + u*`, re-estimates
OLS and V3*. One restriction: signed t with symmetric/equal-tail/upper/lower
p-values; several: Wald, upper tail. The asymptotic reference reported next to
the bootstrap is `t(min(G, H) − 1)` (one restriction) or `χ²(q)`.
Two dimensions only; unweighted OLS results only. The paper gives conditions
under which each bootstrap clustering choice is valid; none of them matches
two-dimensional dependence exactly.

## Binary ML: Kline–Santos score bootstrap

```python
res = oe.score_bootstrap_test(data=df, y="employed", x=["training", "age"],
                              null={"training": 0.0}, model="probit",
                              cluster="site", reps=9999, method="lm")
```

Kline and Santos (2012, equations (12)–(14), Examples 4.1–4.2): with unit
score contributions `s_c` (observations or cluster sums) at an estimate θ,
`A = R J(θ)⁻¹` (J minus the Hessian) and Σ the recentred sample covariance of
the `s_c` with factor G/(G−1),

`T = (A Σ A')^{-1/2} A Σ_c s_c`, `T* = (A Σ* A')^{-1/2} A Σ_c s_c W_c`,

with Σ* the recentred covariance of the perturbed `s_c W_c`. No model is
re-estimated in the draws. `method="lm"` (default) evaluates scores and
Hessian at the null-restricted MLE (robust score statistic); `method="wald"`
uses the unrestricted MLE and the robust Wald statistic of `R θ̂ − r`. Logit
and probit are fitted by Newton with analytic gradients and Hessians, stable
`log Φ` tails for probit, step halving and explicit non-convergence /
separation failures (`convergence_failure`). Recentring follows Kline and
Santos (the mean over units); Roodman et al. (2019, appendix A.3) recentre
cluster sums by cluster observation shares instead, which differs only for
unequal cluster sizes.

## Evidence

`tests/test_econ_iv_wild_bootstrap.py` compares against independent
references with **identical multiplier matrices**:

- WRE: full 2SLS sandwich on `[ŷ2, X1]` in NumPy (not Frisch–Waugh), the
  explicit restricted and efficient reduced-form fits, and a NumPy loop over
  draws — t statistics agree to ≤ 1e-9 relative for HC1 and CR1 with equal and
  unequal clusters and all three multiplier laws; public p-values (all four
  tails) and AR statistics/p-values reproduce exactly from the regenerated
  stream; confidence-set grid p-values, certified brackets and the AR limit
  test are re-evaluated by the oracle; an irrelevant-instrument design yields
  the unbounded AR status.
- Multiway: the three-term CRVE equals statsmodels' two-way cluster
  covariance (`cov_type="cluster"` with two group columns) to 1e-9; WCR draws
  for each bootstrap clustering equal a NumPy loop with explicit V3 and
  eigenvalue truncation to 1e-9; a k > min(G, H) design exercises truncation.
- Score bootstrap: unrestricted MLEs equal statsmodels `Logit`/`Probit` and
  restricted MLEs equal statsmodels binomial GLM fits with an offset (≤ 1e-7);
  LM and Wald statistics and every bootstrap draw are recomputed in NumPy.

R `fwildclusterboot` was not available in the provisioned oracle library, so
no boottest/fwildclusterboot numbers are compared; this is not a vendor parity
claim. `scripts/generate_iv_wild_bootstrap_oracles.py` records seeded bounded
null-size and power cells in `tests/fixtures/iv_wild_bootstrap/size_power.json`.
Run of 2026-10-10 (torch 2.14.0, one thread, B = 199, Webb multipliers,
α = 0.05, 400 datasets per cell, every dataset completed, zero failed calls):

| Cell | Bootstrap rate (Wilson 95%) | Conventional asymptotic rate (Wilson 95%) |
| --- | --- | --- |
| WRE 2SLS t, strong instruments, 12 unequal clusters (4–60 obs), null | 0.050 (0.033–0.076) | 0.178 (0.143–0.218), CR1 normal |
| WRE 2SLS t, weak instruments, same clusters, null | 0.043 (0.027–0.067) | 0.433 (0.385–0.481) |
| WRE AR, weak instruments, null | 0.055 (0.037–0.082) | 0.233 (0.194–0.276), χ² |
| WRE 2SLS t, strong, β = 1.6 vs null 1.0 (power) | 0.295 (0.252–0.341) | 0.595 (oversized reference) |
| Two-way WCR, G = 12, H = 8 (bootstrap by H), null | 0.053 (0.035–0.079) | 0.090 (0.066–0.122), t(7) |
| Two-way WCR, slope 0.25 (power) | 0.333 (0.288–0.380) | 0.378 |
| Score LM logit, 15 clusters, null | 0.060 (0.041–0.088) | 0.085 (0.061–0.116), χ² |
| Score LM logit, β = 0.4 (power) | 0.810 (0.769–0.845) | 0.838 |

These are descriptive results for the declared designs (unequal few clusters
and weak instruments included), not a proof of size control for every
data-generating process; asymptotic power figures are not size-adjusted.

References: [Davidson and MacKinnon (2010)](https://doi.org/10.1198/jbes.2009.07221);
[Roodman, MacKinnon, Nielsen and Webb (2019), Stata Journal 19(1)](https://repec.econ.au.dk/repec/creates/rp/18/rp18_34.pdf);
[MacKinnon, Nielsen and Webb (2021)](https://doi.org/10.1080/07350015.2019.1677473)
([working paper](https://repec.econ.au.dk/repec/creates/rp/20/rp20_06.pdf));
[Kline and Santos (2012)](https://eml.berkeley.edu/~pkline/papers/ScoreFinal_web.pdf)
([NBER w16127](https://doi.org/10.3386/w16127)).
