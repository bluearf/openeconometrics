# Fixed single-target delta sensitivity

`mi_delta(data, columns, *, target, kind, delta, m=5, seed=0, predictors=None,
prior_scale=2.5, proposal_scale=0.5, mh_burn=100, mh_steps=100,
max_work=100000000)` returns an immutable `MIDeltaResult` (a validated `MIResult` subclass) with method `mi_chained`
and operation `fixed_single_target_pattern_mixture`.

Exactly one selected column, `target`, must be incomplete. Every other selected
column must be complete. `predictors=None` uses all other selected columns; an
explicit unique name sequence selects a subset, and `[]` fits an intercept-only
model. All original rows, index labels (including duplicates) and observed cells
are preserved. Only resident finite numeric panels are admitted. Dataset,
weights, GPU/device options, categorical predictors and multiple incomplete
targets are outside this interface.

The caller declares the finite constant sensitivity parameter `delta`. It is
unidentifiable from the observed outcomes. The observed-data model and its
posterior target are identical for every delta; the missing-data predictive
model changes on the declared scale:

| kind | Observed working model | Missing predictive model |
| --- | --- | --- |
| `normal` | Gaussian linear regression | `Normal(X beta + delta, sigma²)` |
| `logit` | Bernoulli logistic regression | `Bernoulli(sigmoid(X beta + delta))` |
| `poisson` | Poisson log-linear regression | `Poisson(exp(X beta + delta))` |

Normal regression draws the full conjugate posterior under
`p(beta, sigma²) ∝ 1/sigma²`. A full-rank observed design, positive residual
sum of squares and positive residual degrees of freedom are required. Each
imputation draws both coefficients and residual variance, then predictive
noise. Delta shifts missing outcomes in their original units.

Logistic and Poisson coefficients, including the intercept, have independent
proper `Normal(0, prior_scale²)` priors in raw predictor units. A symmetric
random-walk Metropolis kernel targets the observed posterior; each imputation
starts an independent chain at zero and retains its final coefficient vector
after `mh_burn + mh_steps` transitions. Acceptance diagnostics describe the
finite computation. No finite-chain stationarity or convergence is asserted.
The proper prior permits a single observed binary class or all-zero counts and
rank-deficient predictor designs. This is a declared prior model, not a
frequentist separation correction.

For a fixed coefficient draw, a logistic delta multiplies missing conditional
odds by `exp(delta)`, and a Poisson delta multiplies missing conditional means
by `exp(delta)`. Count draws are sampled from the shifted Poisson mean. Integer
draws are never multiplied afterward. Observed Poisson outcomes and sampled
counts are integers in `[0, 1000000]`; predictive means must be finite, positive
and at most `1000000`. Overflow, underflow to zero, and draws outside the bound
fail explicitly. No mean or count clipping is applied.

A seed is local to CPU Torch. Imputation `i` uses `(seed + i - 1) mod 2^63`;
global RNG state is untouched. With the same seed, changing delta leaves each
observed posterior draw and observed likelihood unchanged. Gaussian missing
values shift by delta with the same residual noise.

Admission permits 1–10000 rows, 1–16 selected columns and 1–100 imputations.
MH burn is 0–100000 and steps 1–100000. Estimated work and a named workspace
plan are checked before selected tensor/design/sampler allocations. The plan
includes retained completions, final coefficient and predictive diagnostics,
Python result state, and temporary checksum/JSON copies. It estimates named
buffers rather than process RSS; caller-owned input and allocator overhead are
outside that scope. Work above `max_work` or workspace above the configured
budget is refused before computation.

Saved metadata retains the exact missing and observed positions, predictor
terms, prior, declared delta and scale, per-imputation coefficient draws,
Gaussian variance draws, shifted and unshifted missing linear predictors,
predictive means/probabilities, full observed-data log likelihood, MH acceptance
counts and seeds. `MIDeltaResult.model_validate_json` and validated copying recheck common full-state
integrity, original geometry and observed-cell preservation, plus the source-based
link equations, outcome support, priors, model likelihood and resource estimates.
These scientific checks also reject inconsistent metadata if its digest was
recomputed. A base `MIResult` restore does not perform the extra sensitivity checks. The SHA-256 digest
detects inconsistent saved state; it is not authentication. The result exposes
completed datasets, an observed/missing count table and LaTeX rendering.

The delta conditioning convention is documented by the official
[MICE MNAR reference](https://amices.org/mice/reference/mice.impute.mnar.html).
This interface implements a constant, single-target conditional sensitivity
model, not that package's general NARFCS specification language. The Poisson
likelihood, normal coefficient prior and predictive sampling follow the model
structure described in [Stan's posterior prediction guide](https://mc-stan.org/docs/2_29/stan-users-guide/posterior-prediction-for-regressions.html).
Independent NumPy/SciPy tests check Gaussian model/predictive moments, discrete
log-posterior gradients, link-scale effects, full likelihoods and exact support.
Neither implementation parity with MICE/Stan nor arbitrary finite-chain
posterior accuracy is claimed. Delta zero alone does not prove MAR, and this
helper does not establish congeniality with a downstream analysis model or
identify an MNAR mechanism.
