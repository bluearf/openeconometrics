# Binary mediator natural contrasts

`mediation_binary` fits one binary mediator with a logit or probit link and
one Gaussian, logit, probit or Poisson outcome. These eight combinations use
native Torch CPU float64. Treatment and mediator are numeric `0/1`, with both
levels present. Binary outcomes use the same exact coding; Poisson outcomes
are nonnegative integer counts. Gaussian outcomes are continuous. Numeric
controls enter both equations. An optional treatment–mediator interaction
enters the outcome equation.

```python
result = oe.mediation_binary(
    data=data, y="outcome", treatment="treatment", mediator="mediator",
    controls=["baseline"],
    mediator_link="logit", outcome_model="poisson",
    interaction=True, covariance="HC0",
)
restored = oe.mediation_binary_restore(result)
```

## Target and response scale

Let `p(a,c)` be the fitted probability that the mediator equals one under
treatment `a` and controls `c`. Let `g(a,m,c)` be the fitted outcome mean.
For the retained control rows `c_i`, the reported four means are

\[
\mu(a,a')=\frac1n\sum_{i=1}^{n}
 \{[1-p(a',c_i)]g(a,0,c_i)+p(a',c_i)g(a,1,c_i)\}.
\]

The sum over the two mediator states is exact. The empirical control rows are
held fixed. This target differs from evaluating the nonlinear models at mean
controls, and its covariance does not include uncertainty from estimating a
superpopulation distribution of controls. Gaussian means use the identity
link, binary outcome means are probabilities, and Poisson means are expected
counts. Effects are response differences, rather than link-scale coefficients,
odds ratios, rate ratios or coefficient products.

The saved mean order is `mu00,mu10,mu01,mu11`. The five contrasts are:

| Effect | Response difference |
| --- | --- |
| PNDE | `mu10 - mu00` |
| TNDE | `mu11 - mu01` |
| PNIE | `mu01 - mu00` |
| TNIE | `mu11 - mu10` |
| TE | `mu11 - mu00` |

Consequently `TE=PNDE+TNIE=TNDE+PNIE`. The names distinguish direct and
indirect contrasts when the outcome includes interaction. These algebraic
identities also make the five-effect covariance singular; a global five-effect
Wald test would require independent contrast coordinates.

## Likelihood and complete uncertainty

The likelihood factors into a Bernoulli mediator model and the conditional
outcome model. Every observation contributes to both fitted stages. Gaussian
likelihood includes its normalization constant and retains `log_sigma` as a
joint nuisance parameter, with the ML residual variance denominator `n`.
Poisson likelihood includes `log(y!)`. The full observed information is
calculated for all fitted parameters; its cross-equation block is zero by
likelihood factorization.

`covariance="OIM"` uses inverse observed information under the selected
conditional probability models. `covariance="HC0"` uses

\[
 V=H^{-1}\left(\sum_i s_i s_i^{\mathsf T}\right)H^{-1},
\]

where each `s_i` stacks both equations' scores, including Gaussian scale.
The empirical cross-equation score blocks are retained. HC0 has no degrees of
freedom correction. It addresses misspecification of the score covariance
under independent observations; it does not repair incorrect mean models,
confounding or lack of overlap.

Automatic differentiation of the exact standardization formula gives a joint
Jacobian `G`. Complete mean covariance is `G V G'`. Linear response contrasts
produce the complete five-effect covariance, including all off-diagonal
entries. Scale has a zero derivative for Gaussian response means, while its
score and covariance remain saved. Coefficient and effect inference uses
asymptotic normal references; pointwise intervals are not simultaneous bands.
There is no Gaussian scale hypothesis test with an invented null value.

## Interpretation and bounds

The default interpretation is an adjusted model contrast. A causal label
requires explicit declarations of consistency, positivity, sequential
ignorability and absence of treatment-induced mediator–outcome confounding.
These declarations are user-supplied assumptions, not findings from the fit.
Controls must precede treatment for that interpretation. Treatment
randomization alone does not remove mediator–outcome confounding. The
potential-outcome definitions and identification conditions follow
[Imai, Keele and Tingley (2010)](https://imai.fas.harvard.edu/research/files/BaronKenny.pdf).
The authors' [R mediation implementation](https://github.com/kosukeimai/mediation/blob/master/R/mediate.R)
provides a separate primary reference for treatment-specific direct and
indirect estimands. This implementation derives exact binary-state sums and
joint delta covariance directly; it does not copy R estimation code.

This bounded contract supports iid observations, at most 4,096 retained rows
and ten numeric controls. GPU execution, weights, clustered or panel
covariance, multiple or serial mediators, categorical/continuous treatments,
continuous mediators, exposure offsets, sensitivity analysis, bootstrap
inference, missing-data imputation and superpopulation target uncertainty
remain outside this stage. Rank deficiency, separation, degenerate Gaussian
residual scale, nonfinite calculations and failed fit convergence are refused;
regularization does not invent identification. Sample handling is explicit,
and both equations use the same retained rows.

## Persistence and independent evidence

The complete sealed `binary_mediation_state` preserves input rows, model
options, fit parameters, scores, observed information, bread, all covariance
matrices, counterfactual rows, delta Jacobian and inference settings.
`mediation_binary_restore` accepts a fitted result or JSON-roundtripped
attributes and rebuilds tables without optimizing either model. The optional
confidence level changes only inference derived from saved estimates and
covariance.

The fourteen complete tables are `inputs`, `parameters`, `information`,
`bread`, `scores`, `model_loglikelihood`, `covariance`, `counterfactual_rows`,
`means`, `means_covariance`, `effects`, `effects_covariance`, `delta_jacobian`
and `fit_summary`. Full JSON and LaTeX exports retain them all. The
[eight-model synthetic example](../examples/binary_mediation.py) displays
parameters and effects, saves every table and verifies exact state restoration.

Development-only SciPy likelihood optimization, analytic scores and observed
Hessians independently check all eight combinations with and without
interaction, OIM and HC0 including cross-stage blocks, full Gaussian/Poisson
likelihood constants, finite-difference delta Jacobians, complete covariance
and inference. Saturated binary-arm cases also have direct cell-mean oracles.
These bounded checks are separate from frozen runtime, native app and public
release evidence; the broader MARKET-167 scope remains open.
