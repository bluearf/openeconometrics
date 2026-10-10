# Proper-prior point-null model comparison

`bayes_hypothesis` completes exact point-null comparisons for resident iid
Gaussian regressions fitted with `bayes_linear`. It compares the complete proper
normal–inverse-gamma alternative with full-row-rank linear equalities
`C beta = values`. Coefficient order includes the intercept when fitted.
`PosteriorHypothesisComparison` retains both complete models, the original
source/index/sample and all original-unit uncertainty. Bayesian credible
intervals, model probabilities and evidence have their own fields; ordinary
frequentist p-values or standard errors are not substituted.

```python
from openecon.econometrics.bayesian.hypothesis import (
    bayes_hypothesis, bayes_hypothesis_predict, bayes_hypothesis_restore,
)

comparison = bayes_hypothesis(
    result=posterior,
    constraints=[[0.0, 1.0]],  # Intercept, slope
    values=[0.0],
    labels=["zero slope"],
    model_prior_odds=1.0,     # P(null) / P(alternative), explicitly declared
)
tables = comparison.summary()
saved = comparison.model_dump_json()
reopened = bayes_hypothesis_restore(saved=saved)
prediction = bayes_hypothesis_predict(result=reopened, data=query, target="null")
```

## The complete conditional null prior

Write the alternative prior as

\[
\beta\mid\sigma^2\sim N(m_0,\sigma^2V_0),\qquad
\sigma^2\sim IG(a_0,b_0),
\]

where the inverse-gamma density uses `exp(-b0 / sigma²)`. For `r` independent
rows of `C`, put `G = C V0 C'` and `h = values - C m0`. The null uses the
**joint alternative prior conditioned on the equalities**:

\[
a_{0c}=a_0+r/2,\quad b_{0c}=b_0+h'G^{-1}h/2,
\]
\[
m_{0c}=m_0+V_0C'G^{-1}h,\quad
V_{0c}=V_0-V_0C'G^{-1}CV_0.
\]

Conditioning changes the variance prior as well as the coefficient prior. A
null that retains the original marginal inverse-gamma prior would represent a
different model. The conditional nuisance-prior requirement is the reason the
Savage–Dickey density ratio is valid here. See the primary treatments by
[Mulder, Wagenmakers and Marsman (2020)](https://arxiv.org/abs/2004.09899) and
[Heck (2019)](https://bpspsychub.onlinelibrary.wiley.com/doi/10.1111/bmsp.12150).

The algebra can be written with `V0 = L L'`, orthonormal `U` spanning
`null(C L)`, and `K = L U`:

\[
\beta=m_{0c}+K\theta,\qquad \theta\mid\sigma^2\sim N(0,\sigma^2I_{k-r}).
\]

Production constructs an equivalent explicit affine constraint chart in prior
diagonal units and whitens its full proper-prior precision. This avoids
subtracting covariance matrices and avoids cancellation in an orthogonal QR
loading for a directly fixed coefficient. A declared single-coordinate
equality has an exactly zero free loading row and its declared fixed value;
its collapsed credible interval cannot exclude that value due to roundoff.
Complete pivoting chooses the free chart after prior-metric rank admission.

With `W = X K`, `e = y - X m0c`, the free posterior is

\[
V_\theta=(I+W'W)^{-1},\qquad m_\theta=V_\theta W'e,
\]
\[
a_c=a_{0c}+n/2,\quad
b_c=b_{0c}+\{\|e-Wm_\theta\|^2+\|m_\theta\|^2\}/2.
\]

The state stores the full original-unit posterior mean `m0c + K mtheta` and
conditional scale `K Vtheta K'`, including off-diagonal uncertainty. Its
marginal Student-t scale is `(bc/ac) K Vtheta K'`; its covariance is
`bc/(ac-1) K Vtheta K'` when the moment exists. These are distinct quantities.
The moment denominator is retained as `a0 + (n+r-2)/2`; this preserves a tiny
proper shape even when the displayed posterior shape rounds to exactly one.
The null variance mean exists mathematically for every admitted positive
`a0` because `n+r >= 2`. A moment exceeding float64 range is refused explicitly.

When every coefficient is fixed (`r = k`), the free dimension is exactly zero.
The state contains empty free-coordinate vectors/matrices, deterministic
coefficients, zero coefficient scale/covariance and collapsed mean credible
intervals. The inverse-gamma variance posterior remains stochastic; future
observation intervals and draws retain independent Gaussian noise. There is no
dummy free parameter or artificial positive-definite original covariance.

## Model evidence and prior odds

The reported orientation is always

\[
BF_{01}=p(C\beta=d\mid y,H_1)/p(C\beta=d\mid H_1),
\]

so positive `log_bayes_factor_null_alternative` supports the declared null.
Both densities are exact multivariate Student-t densities. Constraint rows are
equilibrated in the proper prior metric; the common row-coordinate Jacobian
cancels from the ratio. Their saved individual log-density fields explicitly
name this coordinate convention. The independent restricted-model marginal
likelihood is also retained rather than defined by adding the Bayes factor to
the alternative likelihood.

Integer/half-integer gamma increments and `log1p` scale increments avoid
subtracting large log-gammas, including declared alternative prior shapes up to
`1e12`. A finite log Bayes factor remains available when its exponential lies
outside representable float64 range; the ordinary Bayes-factor field is then
`None` with explicit `overflow` or `underflow` status.

Posterior model odds equal `BF01 * model_prior_odds`. Omitting prior odds leaves
posterior model probabilities unavailable. A continuous posterior contrast
probability is a different estimand. Point-null posterior probabilities are not
deduced from how many continuous posterior draws happen to equal zero.

## Predictions, exact draws and complete saved state

`bayes_hypothesis_predict` takes `target="null"` or `"alternative"`. It returns
original query row labels, conditional-mean credible intervals and new-outcome
predictive intervals. Its attributes retain full cross-row Student-t scales and
covariances. Independent observation variance is added only for outcomes.

`bayes_hypothesis_draws` returns `PosteriorHypothesisDraws`. It samples the joint
inverse-gamma variance and Gaussian free coefficients with a locally seeded CPU
generator. In the all-fixed null, only variance and future outcomes vary. Draw
state retains the complete original query values/dtypes/index, missing policy
and selected row positions. `bayes_hypothesis_draws_restore` replays the exact
posterior/query/count/seed record. These are exact conjugate draws, with no
warmup, chain R-hat, ESS or MCMC convergence claim.

Every public summary, prediction, draw, restore and payload serialization
rechecks source binding, equality geometry, normalizers, full uncertainty,
resource receipts and the digest. Typed `model_copy` and `model_construct`
objects receive the same semantic checks. Wrong cached shapes and nonfinite,
boolean or overflowing numeric primitives are admitted/refused before tensor
construction. Digest rehashing cannot authorize a numerically inconsistent
posterior. Restoring the comparison performs deterministic analytic replay;
restoring draws performs deterministic local seeded draw replay.
Encoded JSON reserves 64 times its UTF-8 byte count for decoder, dense-object
and typed-validation buffers before either decoder runs. Typed deep copies
reserve the complete resident object/copy geometry and validate semantics
before copying, including objects created through `model_construct`.
JSON serialization reserves complete metadata and depth-dependent indentation
before allocating output and enforces the same 32 MiB encoded envelope.
Excluding the payload from a portable dump still validates the complete model.
`summary()` and `to_tables()` return native `TableSet` model/coefficient tables
and full original-unit prior/posterior conditional scales, Student-t scales
and covariance matrices, each through the same replay boundary.

## Declared resource and numerical boundaries

The alternative contract bounds original rows at 10,000 and original
coefficients at 33. Constraints must be resident lists/tuples with `1 <= r <= k`;
generators and redundant rows are refused. Constraint rows are first normalized
by their own maximum magnitude and then equilibrated in prior coordinates.
The normalized Gram minimum eigenvalue must exceed
`256 * float64_epsilon * r * maximum_eigenvalue`. This is a declared numerical
rank-resolution boundary, not a statistical requirement that changes the
requested hypothesis dimension. Unresolved directions are refused; there is
no pseudoinverse, clipping or covariance repair.

Full joint prediction matrices admit at most 512 original query rows. Exact
draws admit at most 10,000 draws and the existing 2,000,000 joint-value bound.
The inherited `max_work`/`max_bytes` and the current workspace budget apply to
complete source replay, free designs, constraint factorizations, full query
matrices and serialized draw copies before construction. Saved original
admission receipts remain provenance; current live limits are rechecked
independently. Estimates are buffer/work contracts, not a process-RSS limit or
a statistical sample-size condition. All production algebra uses native CPU
float64 Torch; SciPy, NumPy numerical kernels and mpmath are reference-only.
Ambient default dtype/device and RNG state are preserved.

## Independent verification and remaining family scope

The tests integrate the constrained joint prior independently in observation
space, compare complete original-unit uncertainty, and use 90-digit original
prior integration for high-shape odd/even cases. They cover row-unit and
nonorthogonal coefficient-coordinate changes, full affine draw/prediction
covariance, all-fixed support, complete typed source/index binding, semantic
cache forgery and pre-allocation work/shape/primitive guards.

Run the portable saved example and standalone independent audit:

```shell
PYTHONPATH=src python examples/bayesian_hypothesis.py
python scripts/verify_bayesian_hypothesis_oracles.py \
  --input /tmp/bayesian-hypothesis-example/comparison.json \
  --report /tmp/bayesian-hypothesis-example/oracle.json
```

The standalone audit imports neither Torch nor OpenEconometrics. It compares
observation-space integration, SciPy original-constraint Student-t densities
and 90-digit mpmath marginal likelihoods. Its dense reference implementation
admits up to 200 sample rows; that development-only bound does not change the
production 10,000-row contract. Installed application execution, persisted
result reopening and integration acceptance remain separate verification gates.

This exact conditional NIG scope does not complete the entire Bayesian parent.
General posterior/GLM multiple-chain samplers and diagnostics (BAYES-3),
hierarchical priors and selection/BMA (BAYES-4), and proper-prior BVAR/TVP work
(BAYES-5) remain explicit in the [family plan](../research/bayesian.md).
