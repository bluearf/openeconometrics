# S/MM regression and panel quantiles via moments

`oe.sreg`, `oe.mmreg`, and `oe.panel_mmqr` are separate registered estimators.
The existing `rreg` (Cook screening, Huber then biweight) and `qreg` families
retain their original semantics. S/MM regression is resistant to contamination;
the “MM” in panel MM-QR means method of moments and does **not** confer a
high-breakdown contamination guarantee.

All estimator kernels use CPU float64 PyTorch. NumPy/SciPy are used only by
development oracles. These methods require an in-memory table. `Dataset`
input raises `streaming_unsupported` without collecting it. Model-owned design,
joint scores, panel means, and covariance geometry pass the shared workspace
budget before allocation; that budget excludes caller input, Python result
objects, private BLAS buffers, and allocator overhead. S initialization has
work proportional to `starts * N * K`; this is not a promise of unlimited scale.

```python
import openecon as oe

s = oe.sreg(data=df, y="wage", x=["education", "experience"],
            starts=500, seed=24)
mm = oe.mmreg(data=df, y="wage", x=["education", "experience"],
              efficiency_tune=4.685061, starts=500, seed=24)
q = oe.panel_mmqr(data=panel_df, y="wage", x=["experience", "union"],
                  panel="person", time="year", quantiles=[.1, .5, .9])
print(mm.summary())
print(q.to_latex())
```

## S and MM scientific contract

Write (r_i=y_i-x_i'\beta). The normalized Tukey bisquare is

\[
\rho_c(u)=\begin{cases}
1-[1-(u/c)^2]^3,& |u|\le c,\\
1,& |u|>c.
\end{cases}
\]

The S scale solves (N^{-1}\sum_i\rho_{c_0}(r_i/s)=b), and the S
coefficient estimate approximately minimizes this positive scale over
coefficients. `breakdown=b` defaults to .5 and is restricted to [.05,.5].
When `scale_tune` is omitted, deterministic Gaussian integration calibrates

\[
E[\rho_{c_0}(Z)]=b,\quad Z\sim N(0,1).
\]

For (b=.5), (c_0=1.54764498). This normalization is Gaussian-consistent
asymptotically, without a finite-sample scale correction. Supplying
`scale_tune` overrides consistency calibration, and the result carries a
warning. High-breakdown claims require general-position regressors and an
actual S minimum; finite-sample breakdown also depends on design dimension.
The recorded (b) is the **nominal asymptotic** breakdown target.

Initialization uses a local seeded Torch generator and nonsingular elemental
subsets of (K) observations; the global RNG is unchanged. OLS is one
additional candidate. The ten candidates with the smallest initial scale
receive full S refinement; candidates that fail to converge are counted and
excluded, and an explicit failure is raised if none converges. The result
records the seed, valid starts, converged retained starts, scale-equation
error, and score errors. Fitting requires (N>2K), after collinearity
omission. A zero scale or an exact fit is an error. **Finite random starts do
not certify the nonconvex global minimum or theoretical breakdown.**

MM starts at the estimated S coefficients and keeps the S scale fixed.
It minimizes (\sum_i\rho_{c_1}(r_i/s_S)), with (c_1\ge c_0) so that

\[
\rho_{c_1}(u)\le\rho_{c_0}(u),\qquad
\sup\rho_{c_1}=\sup\rho_{c_0}=1.
\]

IRLS with guarded step halving decreases this objective. Convergence requires
both a small coefficient change and a small score. The initial/final MM
objectives and fixed-scale status are persisted. These conditions are the
ones that preserve the initial estimator's breakdown in the MM construction;
they do not turn the finite S search into a proof of breakdown.

For (\psi_c(u)=u[1-(u/c)^2]^2 I(|u|<c)), Gaussian asymptotic efficiency is

\[
\operatorname{ARE}(c)=\frac{(E\psi'_c(Z))^2}{E[\psi_c(Z)^2]}.
\]

Independent SciPy integration gives S efficiency approximately .286826 and
MM efficiency .95 at default `efficiency_tune=4.685061`. This is an
asymptotic central-Gaussian property; finite samples and contamination change
efficiency. A separate seeded Monte Carlo report records that uncertainty.

No observation weights are supported. `nonrobust` covariance assumes
homoskedastic symmetric errors and uses

\[
\widehat V=\frac{N}{N-K}s_S^2
\frac{\overline{\psi_c(u)^2}}{\overline{\psi'_c(u)}^2}(X'X)^{-1}.
\]

`robust` and one-dimensional `cluster` use the full estimating-equation
sandwich, including uncertainty of the S scale. For S the equations are

\[
g_i=(x_i\psi_{c_0}(u_{S,i}),\ \rho_{c_0}(u_{S,i})-b).
\]

MM stacks (x_i\psi_{c_1}(u_{MM,i})) before those S equations.
The derivative includes the coefficient/log-scale cross blocks. `robust`
uses HC1 (N/(N-K)); `cluster` sums all joint scores inside the declared
clusters and applies (G/(G-1)(N-1)/(N-K)). The reported coefficient block
uses Student t with (N-K), or (G-1), degrees of freedom. This differs from
the pseudovalue covariance used by `rreg`.

## Panel MM-QR scientific contract

The supported model is the linear same-design location-scale panel model
of Machado and Santos Silva:

\[
y_{it}=\alpha_i+x_{it}'\beta+
(\delta_i+x_{it}'\gamma)U_{it},\qquad
\delta_i+x_{it}'\gamma>0.
\]

The standardized error has a common continuous distribution, is independent
of the strictly exogenous regressors, and satisfies (E[U]=0) and

\[
E|U|=1.
\]

The paper's principal theorems assume independent errors/covariates across
individuals and time, sufficient finite moments, a uniformly positive scale,
and well-conditioned within-design limits. The cluster sandwich permits
within-cluster dependence under the corresponding cluster moment/CLT
conditions; it does not establish consistency after arbitrary departures
from the common strictly exogenous location-scale restrictions.

The estimator proceeds as follows:

1. Within-panel OLS estimates \(\beta\), and panel means recover \(\alpha_i\).
2. With location residual \(r\), compute
   \(a=2r[I(r\ge0)-\hat\lambda]\),
   \(\hat\lambda=N^{-1}\sum I(r\ge0)\).
3. Within-panel OLS of \(a\) on the same encoded \(X\) estimates \(\gamma\),
   and panel means recover \(\delta_i\).
4. The standardized residual is \(u=r/(\delta_i+x'\gamma)\). Its empirical
   quantile \(q_\tau\) gives the slope \(\beta+q_\tau\gamma\) and individual
   effect \(\alpha_i+q_\tau\delta_i\).

The centered-sign scale transformation is the authors' simplification in
section 3.1. It handles asymmetry and differs from fitting plain absolute
residuals. Quantiles use the empirical inverse CDF with the midpoint at an
exact integer \(N\tau\), Hyndman-Fan type 2.

The supported quantile domain is [.05,.95]; probabilities must be distinct.
Every retained panel needs at least three observations, and each fitted scale
must be positive above a relative numerical floor. Nonpositive scales cause
`nonpositive_scale`; there is no clipping, absolute-value repair, or silent
observation deletion. The guarantee of non-crossing applies only at design
points where the fitted scale is positive. Location/scale regressors use the
same encoded design; nonlinear scale functions, endogenous regressors,
different scale covariates, weights, additional absorbed effects and
split-panel bias correction are outside this first contract.

Time-invariant and within-collinear terms are omitted with recorded names.
There is no identified global intercept (`ModelSpec.intercept=False`);
individual location/scale/quantile effects appear in `extra.individual_effects`.
Rows are stably sorted by panel/time with original sample positions retained.
Duplicate panel/time values and insufficient panels/periods fail explicitly.

### Joint process covariance

Let \(L=[X,D]\) contain the full individual dummy design, and
\(h_i=2[I(r_i\ge0)-\hat\lambda]\). The estimating equations are

\[
L'r=0,\qquad L'(a-s)=0,\qquad
\sum_i[\tau-I(r_i/s_i\le q_\tau)]=0.
\]

The negative Jacobian has location/scale diagonal blocks \(L'L\),
scale/location block \(L'\operatorname{diag}(h)L\), and quantile row blocks

\[
f(q_\tau)\sum_i L_i/s_i,\qquad
q_\tau f(q_\tau)\sum_i L_i/s_i,\qquad Nf(q_\tau).
\]

The derivative with respect to the centered-sign probability vanishes because
\(L'r=0\) when both equations use the same design. The density is a Gaussian
KDE of standardized residuals, using an explicit bandwidth or Silverman's
robust bandwidth. The individual-effect blocks are eliminated analytically;
the runtime does not allocate an \(N\times I\) dummy matrix or \(I^2\)
nuisance covariance.

This joint influence matrix propagates location, scale, individual-effect
and quantile estimation into the full covariance across the requested
quantile slopes. Its primitive covariance and parameter order are saved in
`extra.joint_moment_covariance` and `extra.joint_moment_order`; the reported
`covariance_matrix` contains every cross-quantile block. `HC0` uses independent
observation scores. `robust` clusters on the panel. `cluster` uses one column
that must nest panels. Cluster CR1 is
\(G/(G-1)(N-1)/(N-I-K)\), and coefficient inference uses Student t with
\(G-1\) degrees of freedom. `HC0` uses \(N-I-K\) degrees of freedom.

These are a joint empirical profile-moment sandwich and density estimator,
**not a claim of covariance parity with `xtqreg` 1.5**. The full dense dummy
Jacobian is independently checked on synthetic panels, and a full sparse
dummy Jacobian checks the published-author data fixture.

The paper's quantile slopes/scale inference requires growing panel lengths
and \(n=o(T)\) for a centered asymptotic distribution. Short panels suffer
incidental-parameter bias, which a cluster covariance does not correct.
Every result warns about fixed-T bias; a further warning is emitted when
the number of individuals is not small relative to the shortest panel.
The three-observation guard is an identification floor, **not** a statement
that confidence intervals from a short panel have valid coverage.

## Verification and reference boundaries

`tests/test_robust_smm.py` checks bisquare tuning, Gaussian efficiency,
scale roots, S/MM coefficients against independent nonlinear optimization,
covariance against numerical score derivatives, bad-leverage/vertical
contamination at 20% and 40% with amplitude increasing from 1,000 to
1,000,000, fixed-scale objective descent, seed isolation, sample alignment,
rank omission, strict JSON reload, LaTeX, resource rejection and lazy imports.

`tests/test_panel_mmqr.py` checks all covariance/process blocks against an
independent full dense dummy GMM system, distributional effects,
non-crossing on the fitted positive-scale domain, sample/missing/time/rank
alignment, strict JSON/LaTeX and the failure domains.

The frozen public fixture reproduces the **authors' published**
[nlswork example](https://jmcss.som.surrey.ac.uk/MM-QR-JK.do), from
[Stata Press release-17 demonstration data](https://www.stata-press.com/data/r17/u.html).
It keeps individuals with at least 10 source observations, before listwise
missing exclusion, and estimates `ln_wage` on `age ttl_exp tenure not_smsa south`
at .1/.5/.9. The selected data have 11,541 complete observations and 982
individuals. Their final panel lengths range from 5 to 15; this is a published
algorithm replication with explicit finite-T warnings, not coverage evidence.
The CSV preserves the original Stata floating point values and source row IDs.
Raw source/selected CSV/author-command checksums and independent frozen
coefficients/full covariance are recorded alongside it.

`scripts/validate_robust_mmqr.py` regenerates the fixture using NumPy/SciPy
and a full sparse dummy moment system, separate from the runtime's profiled
PyTorch implementation. **Stata itself has not been executed**, and
`stata_parity_validated` remains false. Published-author coefficient-algorithm
replication, independent scientific covariance checks, local source tests,
frozen runtime/installed application and public release are separate proof
layers. This change supplies the first three; it supplies no application
build, platform installation or release evidence.

`scripts/validate_robust_efficiency.py` supplies a seeded clean-Gaussian Monte
Carlo, paired Monte Carlo uncertainty, independent efficiency integrals, and
a separate severe-contamination experiment. The resulting evidence is in
`docs/evidence/market-132-robust/scientific-validation.json`.

## Primary sources

- Yohai (1987), [High Breakdown-Point and High Efficiency Robust Estimates
  for Regression](https://doi.org/10.1214/aos/1176350366), *Annals of Statistics*
  15(2), 642–656: three-stage MM construction, common bound/ordered losses,
  fixed robust scale, asymptotic efficiency and initial-estimator breakdown.
- Machado and Santos Silva (2019), [Quantiles via Moments](https://doi.org/10.1016/j.jeconom.2019.04.009),
  *Journal of Econometrics* 213(1), 145–173;
  [author manuscript](https://openresearch.surrey.ac.uk/esploro/outputs/99516973202346),
  section 3.1 and appendix: location/scale panel moments, individual
  distributional effects, positivity, no crossing and incidental-parameter bias.
- Machado and Santos Silva,
  [published `xtqreg` implementation](https://ideas.repec.org/c/boc/bocode/s458523.html),
  version 1.5 (30 September 2021), used as the published coefficient-algorithm
  reference; [author research page](https://jmcss.som.surrey.ac.uk/research.html)
  links the nlswork example and discusses its jackknife/cluster route.

The appendix of the accessible manuscript defines the centered scale
disturbance with a `-1` term. The author-command covariance code uses a
different pooled-moment construction. We therefore identify the exact
joint profile-moment covariance supplied here rather than labeling it
as reproduction of the published command's covariance.
