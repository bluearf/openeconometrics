# Proper conjugate Gaussian posterior

This implements the analytic regression and group-contrast domain in
[MARKET-625](https://linear.app/bluearf/issue/MARKET-625), the first implementation
stage under [MARKET-362](https://linear.app/bluearf/issue/MARKET-362).
It provides proper normal–inverse-gamma inference, exact saved analytic state,
linear contrasts, conditional-mean credible intervals, future-observation
predictive intervals and bounded independent joint posterior/predictive draws.

## Model and prior

The caller declares an iid, homoskedastic Gaussian outcome model:

```
y | beta, sigma² ~ Normal(X beta, sigma² I)
beta | sigma² ~ Normal(m0, sigma² V0)
sigma² ~ InverseGamma(a0, b0)
```

The inverse-gamma density is proportional to
`(sigma²)^(-a0-1) exp(-b0/sigma²)`. Shape and scale are strictly positive;
`V0` must be finite, exactly symmetric and positive definite. An explicit prior
is required. Its coefficient order is `Intercept`, when enabled, followed by
the ordered predictor names. No implicit flat/improper prior is introduced.
`shape` is bounded above by `1e12` for the native special-function domain.
Prior means/scales and the data must fit the supported float64 numerical range.

The posterior is computed using native CPU float64 Torch Cholesky solves:

```
VN = inverse(inverse(V0) + X' X)
mN = VN (inverse(V0) m0 + X' y)
aN = a0 + n/2
bN = b0 + ((y-X mN)'(y-X mN) + (mN-m0)' inverse(V0) (mN-m0))/2
```

The residual/prior-penalty form of `bN` avoids subtracting large sufficient
statistics. Marginal `beta` has a multivariate Student-t distribution with
`2 aN` degrees of freedom and **scale matrix** `(bN/aN) VN`. Its covariance is
`(bN/(aN-1)) VN` only when `aN > 1`; otherwise covariance and posterior standard
deviations are unavailable while credible/predictive intervals remain defined.
The normalized proper-prior log marginal likelihood retains every determinant
and likelihood constant. It does not automatically define competing point-null
hypotheses, model probabilities or Bayes factors.

Log evidence uses integer gamma recurrences and a shifted, stable half-step
ratio from [DLMF 5.11.8](https://dlmf.nist.gov/5.11.E8). It groups variance-scale
terms with `log1p` and retains the residual penalty before adding it to `b0`.
Thus a high prior shape does not erase likelihood constants, and a penalty
smaller than one float64 unit of `b0` still contributes to the evidence.

For a linear contrast `w' beta`, the Student-t scale is
`sqrt(w' ((bN/aN) VN) w)`. Posterior probability above a caller threshold is
labelled as a Bayesian probability. For a new row `q`, the conditional-mean
scale is `sqrt((bN/aN) q' VN q)` and the future-observation scale adds
`bN/aN` before taking the square root. All these distributions use `2 aN` df.
Predictive uncertainty includes a future iid observation; credible uncertainty
for the conditional mean does not.

`bayes_predict` retains full cross-row conditional-mean uncertainty as
`attrs["mean_scale_factor"]`: its product with its transpose is the complete
joint Student-t scale matrix. Add
`attrs["outcome_independent_scale"] * I` for future-observation joint scale.
Convert scale to covariance only when
`attrs["scale_to_covariance_multiplier"]` is available. This factorized
representation avoids allocating an unbounded dense query-by-query matrix.

The prior can regularize a rank-deficient or wider-than-sample design.
`source_rank` records that condition. A valid proper posterior in this domain
does not establish frequentist identification or instrument validity.

## Usage

```python
import pandas as pd
import openecon as oe

data = pd.DataFrame({"y": [1., 2., 3.], "x": [0., 1., 2.]})
prior = oe.NormalInverseGammaPrior(
    mean=(0., 0.), scale_matrix=((1., 0.), (0., 1.)),
    shape=2., scale=1.,
)
posterior = oe.bayes_linear(data=data, y="y", x=["x"], prior=prior)
posterior.summary()
restored = oe.PosteriorBundle.model_validate_json(posterior.model_dump_json())
oe.bayes_contrast(result=restored, weights={"x": 1.}).summary()
query = pd.DataFrame({"x": [.5, 1.5]})
oe.bayes_predict(result=restored, data=query)
draws = oe.bayes_draws(result=restored, draws=100, seed=735, data=query)
oe.PosteriorDraws.model_validate_json(draws.model_dump_json())
posterior.to_latex()
```

Group-mean differences are declared design-column contrasts of this same
regression model. The caller supplies numeric indicators/contrasts and a proper
prior in that basis. There is no hidden categorical coding, automatic reference
level or default prior on groups. Zero contrast vectors are exact degenerate
targets; their interval is the single target value.

## Source, uncertainty and persistence

`PosteriorBundle` is a separately typed immutable, versioned result. It has no
frequentist p-value, t/z statistic, frequentist confidence interval, or MCMC
diagnostic fields. Its tables and LaTeX exports use posterior means, posterior
standard deviations where they exist, Student-t scales and credible limits.

State records the declared prior, original selected source columns/values and
dtypes, typed original index, missing policy, original complete-case positions,
ordered coefficient terms, all posterior matrices, normalized log evidence,
source rank and integrity digests. Supported indices include ordinary duplicate
labels, RangeIndex, categorical, timezone-aware datetime, timedelta and
MultiIndex, with typed scalar/date/tuple labels. Unsupported index kinds fail.

Validation reconstructs the declared typed source and exact sample, checks the
index round trip, recomputes the conjugate algebra and every posterior matrix,
credible limit, rank and digest. Rehashed but numerically forged saved results
are refused, including tiny-unit covariance changes. Saved predictions accept
a `PosteriorBundle`, its mapping or JSON. Validation uses analytic sufficient
statistics and performs no sampling-chain fit or optimizer retraining.
Integrity digests detect corruption; they are not signatures or authentication.

`PosteriorContrast` saves the full required `PosteriorBundle` target, named
coefficient order, weights, threshold and credible probability. Its validator
revalidates that target and recomputes the mean, full Student-t quadratic scale,
posterior variance availability, credible limits and probability above the
threshold. A target digest alone cannot restore a contrast. Constructed/copied
live targets and fully rehashed altered inference are revalidated as well.

`bayes_draws` uses its own CPU generator. It draws inverse-gamma variance and
conditionally normal coefficients jointly, then optionally conditional means
and future iid observations. These are independent analytic posterior draws;
there is no warmup, chain convergence, R-hat or ESS claim. `PosteriorDraws`
embeds its exact posterior target, seed, query design/index and bounded arrays.
Its saved-state validator replays the same target/seed and checks every draw.
Exact seeded draw reproduction is scoped to the recorded native Torch runtime;
the analytic posterior/interval state is portable independently of RNG versions.
Ambient Torch dtype, default device and global RNG state remain unchanged.

## Input and resource domains

Input is a resident pandas/OpenEconometrics DataFrame with finite real numeric
columns, excluding booleans/complex values. Integer source values must be exactly
representable in float64. At most 10,000 **original** rows and 32 predictors
are admitted, plus an optional intercept. The default missing policy raises;
`missing="drop"` retains complete cases with original positions. Prediction
has its own explicit missing policy and preserves query index/positions.
No intercept and an intercept-only model are supported; a zero-width design
and an empty complete-case sample are refused.

Named data/state/factorization/index/query/draw buffers and operation counts are
admitted before numerical/source copies. `max_bytes` defaults to 128 MiB and is
also bounded by the current global workspace budget. `max_work` defaults to
100,000,000 conservative operation units. These are estimated buffer/work
contracts, not a process-RSS guarantee; caller-owned data and private BLAS
workspace are excluded. The maximum row/predictor limits do not guarantee every
combination fits the default budgets. Saved metadata and JSON are bounded
before typed copying; JSON payloads are limited to 32 MiB. Individual index
strings are limited to 10,000 characters and model column names to 1,000.

Draw count is 1–10,000, seed is an integer in `[0, 2**63-1]`, and complete
coefficient/variance/mean/outcome draw arrays are limited jointly to 2,000,000
values. Unsupported weights, cluster/HAC covariance, Dataset/streaming,
GPU, endogenous sampling, heteroskedastic likelihood and automatic category
encoding are outside this method contract and are not silently approximated.

## Verification and remaining family scope

The source suite compares all coefficients, off-diagonal scale/covariance,
inverse-gamma parameters, normalized log evidence, contrast probabilities,
credible/predictive intervals and full cross-row prediction scale with
independent NumPy sufficient-statistic algebra and SciPy distributions.
Hand-computed, prior/unit scaling, rank-deficient, infinite-variance,
missing/index, resource, global-default/RNG and rehashed-state refusal cases
are included. A fixed seed recovery protocol checks variance marginals and
conditionally standardized joint coefficient draws with predeclared tolerances.

The standalone development oracle
`scripts/verify_bayesian_conjugate_oracles.py` imports neither Torch nor
OpenEconometrics and checks a saved result or the receipt emitted by
`examples/bayesian_conjugate.py`. Run it with:

```text
python scripts/verify_bayesian_conjugate_oracles.py --posterior receipt.json --report oracle-report.json
```

Source tests, independent receipt checks, source-built frozen execution,
installed native Run/export/full Quit/reopen and any licensed vendor reference
are separate evidence layers. This method document does not claim those
delivery layers have been completed by its presence in the repository.

Remaining **MARKET-362** work includes proper competing-hypothesis/Bayes-factor
workflows beyond continuous contrasts, native multiple-chain sampling and
diagnostics, logit/GLM posterior families, hierarchical models, variable
selection/BMA, BVAR posterior forecasts/IRFs and separate TVP-VAR state/samplers.
The first analytic stage does not complete the general Bayesian family.

Normal/inverse-gamma conventions and Gaussian conjugacy are described in
[Murphy's original-author derivation](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf).
Future sampling diagnostics and calibration have separate protocols in the
[Bayesian family plan](../research/bayesian.md).
