# Discrete fully conditional multiple imputation

`mi_discrete` imputes incomplete count, ordinal and nominal columns of a resident
numeric panel. It returns `MIDiscreteResult`, a full-state `MIResult` subclass
whose saved JSON revalidates the discrete model and predictive state. The stored
MI family is `mi_chained`; `metadata["api"] == "mi_discrete"` identifies this
operation. This delivery covers MARKET-433 (Poisson), MARKET-434 (ordered logit)
and MARKET-435 (baseline multinomial logit).

```python
import openecon as oe

result = oe.mi_discrete(
    frame, ["visits", "rating", "sector", "income"],
    methods={"visits": "poisson", "rating": "ordinal", "sector": "multinomial"},
    categories={"rating": [1, 2, 3], "sector": [10, 20, 30]},
    m=5, seed=2026, burn=10, iterations=5,
    predictors={"visits": ["income"], "rating": ["income"], "sector": ["income"]},
    prior_scale=2.5, proposal_scale=0.5, mh_burn=100, mh_steps=100,
)
completed = result.dataset(imputation=1)
saved = result.model_dump_json()
restored = oe.MIDiscreteResult.model_validate_json(saved)
```

`methods` must specify exactly every incomplete selected column. Complete
numeric columns can serve as predictors. Categorical targets require explicitly
declared, distinct finite numeric codes, with 3–8 levels and every level observed.
The ordinal list declares the order; the first multinomial code declares the
baseline. Codes are preserved in the completed data. No inferred order, string
encoding, absent-level borrowing, or automatic category reduction is provided.
Poisson observed outcomes must be integer counts between 0 and 1,000,000.

`predictors` optionally maps every incomplete target to a sequence of distinct
other selected column names. An empty sequence permits a count/multinomial
intercept-only model or an ordinal threshold-only model. If omitted, every other
selected column is used. Categorical predictor codes enter as raw numeric values;
the function does not infer dummy variables. Proper priors retain all declared
coefficients, including in a rank-deficient design. Predictor scaling affects the
prior and proposal, so its interpretation belongs to the stated model.

## Conditional models and draws

For Poisson targets, `log(lambda_i) = X_i beta`, including an intercept. The
posterior combines the ordinary Poisson likelihood with independent
`Normal(0, prior_scale**2)` coefficient priors. An accepted parameter state produces
integer Poisson predictive draws. Every rate used for prediction must be finite
and at most 1,000,000; exceeding this boundary raises an error. Likelihood proposal
overflow yields a negative-infinite target and rejection. Neither rates nor
outcomes are clipped, and predictive counts retain unbounded Poisson support.

Ordinal targets use `eta_i = X_i beta` without an intercept. For `K` declared
levels, the sampled coordinates are `beta`, the first cutpoint `a`, and `K-2`
log-gap coordinates `g`. Cutpoints are `a`, `a + exp(g_1)`, and subsequent cumulative
positive gaps. The model uses ordered-logit probabilities: the first and last
levels are logistic tail probabilities; middle levels are differences of adjacent
logistic CDFs. Their log probabilities use positive gaps directly to avoid
cancellation. Independent Gaussian priors are defined **in these sampled
coordinates**, so no change-of-variables Jacobian is added. Numerically unresolved
or overflowing cutpoint proposals are rejected.

Multinomial targets include an intercept and fix the first declared category's
linear predictor to zero. The remaining `K-1` coefficient vectors receive
independent Gaussian priors of the stated scale. A stable log-softmax likelihood
defines the posterior, and category draws use its probabilities in the declared
code order. No empirical baseline selection or probability clipping occurs.

Every conditional update uses symmetric isotropic Gaussian random-walk
Metropolis in the declared coordinates, with covariance
`proposal_scale**2 * I`. It runs `mh_burn + mh_steps` transitions, carries its last
parameter state to the next sweep, then draws missing outcomes. All randomness
uses a private CPU generator. Independent imputation seeds are `seed + chain`
modulo `2**63`, beginning with `chain = 0`. Initial missing values are independently
sampled from each target's observed values; targets are visited in selected-column
order. Each chain returns its completed panel after `burn + iterations` sweeps.

The Metropolis transition targets the stated posterior rather than a Laplace or
asymptotic approximation. Its finite output is not claimed to have reached
stationarity. The conditional models need not define a compatible joint model.
Iteration counts and acceptance rates do not establish convergence; the saved
metadata explicitly records convergence, stationarity and compatibility as
unassessed. This scope provides no vendor-parity claim.

## State, bounds and verification

All observed cells, row positions and original indexes, including duplicate labels,
are preserved. No rows or predictors are silently dropped. Each result stores the
declared methods, category order/baseline, predictor/update order, prior coordinates,
MH proposal/acceptance/rejection counts and finite posterior-density traces. Each
imputation retains the last posterior coordinates, cutpoints or coefficient
matrices, conditional observed/missing designs and missing means/probabilities.
These designs describe the target's update in the final sweep: later target
updates can subsequently change a predictor in the final completed panel.

`MIDiscreteResult` rechecks category/count support, model geometry, proper-prior
declarations, diagnostics and resource estimates. It recomputes the last recorded
posterior density and missing probabilities/means from the saved conditional
design and parameter state. It checks completed summaries and observed predictor
preservation without claiming that a finite random draw proves convergence.
The inherited full-state checksum detects accidental state changes; it is not a
signature authenticating data to a third party.

Admission is limited to 10,000 resident rows, 16 selected numeric columns and 100
imputations. `burn + iterations <= 1,000`, conditional parameter dimension is at
most 128, and trace size is at most 100,000 variable updates. Work estimates must
fit `max_work` (default 100,000,000). Additional workspace plans account for
conditional designs, category probabilities, proposal vectors, traces, final
models, and their serialized/checksum state before these buffers are created.
Large requests can be refused even when individual shape bounds are satisfied.
Kernels use native Torch on CPU in float64; Dataset/streaming inputs, weights,
device fallback and unbounded automatic retries are outside this API.

The scoped tests compare original PMFs and gradients with independent NumPy/SciPy
calculations, scalar/two-dimensional numerical posterior quadrature, predictive
moments, conditional probability readback, exact support, observed/index
preservation, seeded replay and global-RNG isolation. They also exercise scalar,
category/count and resource refusals, saved-state mutation rejection and ambient
dtype/device independence. These are bounded method tests, not a claim about
arbitrary FCS convergence or all missingness mechanisms.

The probability definitions follow the primary [Stan Poisson reference](https://mc-stan.org/docs/functions-reference/unbounded_discrete_distributions.html#poisson-distribution)
and [Stan ordered-logit and categorical-logit reference](https://mc-stan.org/docs/functions-reference/bounded_discrete_distributions.html).
The finite FCS workflow and its conditional-model scope are informed by the
original [mice paper](https://www.jstatsoft.org/article/view/v045i03).
