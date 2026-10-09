# MARKET-155: Bayesian and posterior workflow scope decision

Status: planning deliverable; implementation deferred. Preserve the earlier
`Hayır` decisions for general MCMC, Bayesian regression/group tests, variable
selection and averaging; BVAR/TVP-VAR retain their `Sonra` status. Activate a
conjugate iid Gaussian pilot only after a concrete posterior-prediction use
case and a reviewed prior/result contract. Frequentist SEs and `ResultBundle`
inference are not posterior draws or posterior uncertainty.

Proposed `oe.bayes_linear` and other `bayes_*` names are unregistered draft
names. Define a versioned **PosteriorBundle** before exporting them: priors and
their parameterization, likelihood/data provenance, ordered chains/draws,
warmup boundaries, summaries, diagnostics, posterior-prediction state and
external draw-artifact hashes. Do not populate frequentist p/CI fields with
posterior probabilities or credible intervals.

## Deferred phase matrix

| Phase | Domain and dependencies | Acceptance gate |
| --- | --- | --- |
| BAYES-1 | Normal linear regression with proper normal–inverse-gamma prior; analytic iid posterior; one-sample/group contrasts derived from this model | Hand-computed posterior precision/mean/shape/scale and Student-t predictions; unit/scaling and prior-sensitivity fixtures; no chain diagnostics for analytic results |
| BAYES-2 | Posterior state, credible intervals, posterior-predictive draws and explicit point-null comparisons; depends on BAYES-1 | Exact round trip, full posterior covariance, credible vs predictive interval separation; Bayes factors only with specified proper hypotheses/normalizing constants |
| BAYES-3 | Native multiple-chain sampling and diagnostics, then logit/GLM; depends on the result contract and reviewed sampler algorithm | Analytic-target recovery, all-chain/all-parameter diagnostics, independent sampler comparison, simulation-based calibration; failed diagnostics retained and predictions qualified |
| BAYES-4 | Hierarchical likelihoods and variable selection/BMA; depends on BAYES-3 | Prior and parameterization sensitivity, label/design/likelihood agreement, marginal-likelihood reference; model weights cannot be exponentiated frequentist AIC weights |
| BAYES-5 | BVAR with an explicit Minnesota or conjugate matrix prior, then TVP-VAR as a separate state/sampler stage | Joint posterior coefficients/covariance and identification-dependent posterior IRFs; forecast draws include state/parameter uncertainty; observed random-walk/MCMC reference |

## Exact pilot definition

Use `y | β,σ² ~ N(Xβ,σ²I)`, `β | σ² ~ N(m0,σ²V0)`,
`σ² ~ IG(a0,b0)` with density proportional to
`(σ²)^(-a0-1) exp(-b0/σ²)`. Require proper positive shape/scale and positive
definite V0; preserve prior units and intercept treatment. Then
`VN=(V0⁻¹+X'X)⁻¹`, `mN=VN(V0⁻¹m0+X'y)`, `aN=a0+n/2`,
`bN=b0+(y'y+m0'V0⁻¹m0−mN'VN⁻¹mN)/2`.
Evaluate via stable factorizations; independently verify the algebra. Marginal
β follows a multivariate t with `2aN` df and **scale matrix** `(bN/aN)VN`;
its covariance exists only for `aN>1` and is `(bN/(aN−1))VN`. Predictions for
new outcomes add observation variance; conditional-mean uncertainty does not.

Default missing policy is raise; explicit complete-case dropping preserves
original row positions and counts. Reject nonfinite priors/data, unsupported
weights, endogenous sampling, unsupported covariance requests and accidental
global RNG mutation. Rank-deficient X with a proper prior is a distinct valid
posterior case, not a claim of frequentist identification. Report the prior's
regularization and support; improper priors cannot yield arbitrary Bayes factors.

## Sampler and calibration plan

Chain seeds, initialization, warmup/adaptation, iteration limits and retained
draws must be explicit. First sampler choice is a separately reviewed project
decision; importing Stan as a production estimation backend is not this plan.
Log posterior, gradients, transformations/Jacobians and native sampler steps
need independent references. Diagnostics record rank-normalized/folded split
R-hat, bulk/tail ESS and MCSE for every parameter and important prediction.
Sampler-specific divergences and acceptance diagnostics must be preserved.
Thresholds are declared protocol gates, not proof of convergence.
[Stan posterior diagnostics](https://mc-stan.org/docs/reference-manual/analysis.html).

Predeclare parameter/data simulation under the proper prior, posterior rank
calculation, autocorrelation treatment, grid size, seeds, failed-chain
denominators and a simultaneous rank-uniformity acceptance rule. Use exact
conjugate targets first, then an independent development-only sampler.
Simulation-based calibration checks the algorithm under its declared model;
it does not validate real-world prior suitability.
[SBC protocol](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html).

Persist draw dtype/order, warmup exclusions and all failed-chain diagnostics.
Avoid embedding unbounded draws in JSON; use bounded immutable local artifacts
with checksums and verified access. Reopening must reproduce saved summaries
and seeded predictions without retraining or dropping problematic chains.
Budget data, factorizations and `chains × retained draws × parameters × 8`
bytes before sampling; also bound work by model gradient cost, iterations and
posterior-predictive draws. CPU float64 is the first gate; Dataset/GPU/cloud
draw distribution each needs separate evidence.

Open implementation: [GitHub #41](https://github.com/bluearf/openecon/issues/41).
All numerical/sampler/platform gates above are future work, not passed tests.
