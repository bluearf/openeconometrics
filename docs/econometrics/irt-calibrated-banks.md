# Known calibrated IRT banks and exact finite posterior

This method slice accepts supplied item parameters and computes person posteriors
under a caller-declared finite prior. It does not estimate item parameters, infer
calibration uncertainty, or accept fitted MML results as known banks. All numerical
kernels use native resident CPU Torch in float64; SciPy is used only by independent
test oracles.

## Binary bank: MARKET-473

```python
bank = oe.irt_bank_binary(
    ["item_a", "item_b"],
    discrimination=[1.2, .8], difficulty=[-.5, .8],
    guessing=[0., .2], upper=[1., .9],
)
```

For category code 1, the probability is

\[
P_j(1\mid\theta)=c_j+(u_j-c_j)\operatorname{logistic}\{a_j(\theta-b_j)\}.
\]

Category 0 has the complementary probability. `guessing=None` means zero and
`upper=None` means one. The implementation evaluates both probabilities directly
in log space, including these endpoints, rather than taking the logarithm of a
rounded sigmoid. The supplied discrimination is positive; a fixed lower and upper
asymptote are part of the known item model.

## Polytomous bank: MARKET-474

```python
bank = oe.irt_bank_polytomous(
    ["graded", "partial", "nominal"],
    family=["grm", "gpcm", "nrm"],
    thresholds=[[-1., .2, 1.], [.8, -.5], None],
    discrimination=[1.3, .7, None],
    slopes=[None, None, [0., -.8, 1.2]],
    intercepts=[None, None, [0., .5, -.2]],
    scores=[None, None, [3, 0, 3]],
)
```

`family` may be one string for all items or a per-item list. Each parameter is a
per-item list; an irrelevant row in a mixed bank must be `None`. GRM and GPCM
discrimination defaults to one if omitted. No parameter aliases or conversions
from another implementation's slope/difficulty convention are performed.

* **GRM:** ordered, strictly increasing thresholds define cumulative probabilities
  \(P(Y\geq k\mid\theta)=\operatorname{logistic}\{a(\theta-b_k)\}\).
  Category probabilities are adjacent cumulative differences. Endpoint and middle
  probabilities are evaluated with stable log expressions. A positive threshold
  gap is accepted even when it is subnormal; tied thresholds are refused.
* **GPCM:** for category \(k\), the logit is
  \(\sum_{h=1}^{k}a(\theta-b_h)\), with category zero's logit equal to zero.
  The `thresholds` argument contains step difficulties; steps need not be ordered.
* **NRM:** category logits are \(a_k\theta+d_k\) and probabilities are their
  softmax. The baseline slope and intercept must both be exactly zero, and slopes
  must not all be equal. Category labels do not imply an ordinal response order.

Response category codes are always integers `0..K-1`. Scoring values are a
separate map used by downstream score-distribution operations. GRM/GPCM default to
`0..K-1`; NRM requires explicit integer values `0..10`. Duplicate and gapped values
are valid. For example, nominal scoring map `[3, 0, 3]` still has three response
categories; its first and third categories contribute to the same total-score
value. Scoring values do not enter the item response likelihood.

## Exact finite posterior: MARKET-477

```python
posterior = oe.irt_posterior(
    bank, data=responses,
    support=[-2., -.3, 1., 2.5], masses=[.1, .2, .5, .2],
)
```

The support and masses specify the actual discrete latent model, not a numerical
quadrature approximation to a continuous normal model. For person \(i\), with
observed item set \(O_i\),

\[
\pi_{iq}=
\frac{m_q\prod_{j\in O_i}P_j(y_{ij}\mid\theta_q)}
{\sum_r m_r\prod_{j\in O_i}P_j(y_{ij}\mid\theta_r)}.
\]

The computation sums log probabilities and normalizes with log-sum-exp. Item
responses are conditionally independent given one latent trait. Item parameters
and prior masses are treated as fixed. Reported uncertainty is conditional on
those choices and does not include estimated-parameter uncertainty.

Every input person remains in its physical source position. `None`, `NaN`,
`pd.NA`, and `pd.NaT` are missing; observed `-1`, booleans, fractional values,
infinities, and numeric strings are refused. Missing items omit only their own
likelihood factors. An all-missing person receives the declared prior, with log
evidence zero up to floating-point rounding. Duplicate index labels are preserved
and distinguished by `source_position`.

The result includes all posterior masses, declared prior, selected responses,
physical sample positions and index labels. The summary reports mean, standard
deviation, entropy, log evidence, and inverse discrete CDF quantiles at `.025`,
`.5`, and `.975`. A quantile is the smallest support point whose cumulative mass
reaches the target; intervals therefore stay on the support. A one-point prior
has zero posterior standard deviation and entropy.

## Admission bounds and persistence

* Banks admit `1..16` uniquely named items, with names of `1..256` characters and
  `2..6` categories per item. Constructors require primitive lists and finite
  primitive numbers; booleans are not numerical parameters.
* Discrimination is `.1..5`; binary difficulties and ordinal step/threshold
  values are `-8..8`. Binary guessing is `0...4`, upper is `.6..1`, with
  guessing strictly below upper. NRM slopes are `-5..5` and intercepts `-12..12`.
* Finite support has `1..101` sorted, unique points in `-8..8`; masses must be
  positive and sum to one within `1e-12`. Masses are never silently normalized.
* Data must be a resident pandas DataFrame, a resident column mapping, or explicit
  records, with all bank items present exactly once in a DataFrame. The limit is
  `1..1000` people. There is no implicit Dataset or tensor-device transfer.
  Index labels must be finite primitive scalars; strings have at most 256
  characters and numeric labels are bounded by `2**53-1`.
  Column mappings are positional, including Series with different indexes;
  use a DataFrame when its row labels should be preserved. Mapping/record inputs
  receive physical row labels `0..n-1`.
* `max_bytes` defaults to 128 MiB and is combined with the current global workspace
  budget. Plans name numerical, full-state/table serialization, item-probability,
  and selected-response buffers before their numerical allocations. The default
  `max_work` is 300 million conservative work units; the finite-posterior charge
  is `128*n*q*j + 128*q*sum(K_j) + 128*n*q`. These are admission estimates, not a
  process RSS or wall-time guarantee. Caller input, private allocator/BLAS
  workspace, and unrelated process memory remain outside that contract.
  An additional conservative complete-summary bound includes Unicode JSON
  escaping and every repeated row label. Outputs exceeding the 8 MiB portable
  domain are refused before finite-posterior tensors; a short character count
  alone does not establish that long Unicode labels are portable.

`oe.summary_state(result)` contains complete portable bank or posterior state,
including parameter maps, input category codes, prior, posterior, evidence,
metadata and all canonical tables. Full-summary JSON input is bounded to 8 MiB.
Both `oe.restore_summary` and the specialized `oe.irt_bank_restore` /
`oe.irt_posterior_restore` preserve full tables and exact LaTeX. Use the specialized
restore functions before further calculations: they validate primitive schemas,
state checksums, canonical table dimensions and cells, and sample alignment.
Posterior restoration additionally recomputes the finite Bayes calculation and
checks its posterior and evidence; rehashing incoherent numerical state does not
make it valid. Fitted models and state-only handcrafted envelopes are refused.

Downstream operations first use `_posterior_input` for primitive shape validation,
then admit their aggregate work and buffers through `_posterior` before numerical
verification. The private response kernel permits scoring evaluations on
`-12..12`; this does not enlarge the finite-prior support domain.

## Independent validation and references

Tests compare known binary, GRM, GPCM and NRM probabilities with independently
implemented NumPy/SciPy formulas, and enumerate finite-prior Bayes calculations
for complete, partial and all-missing response patterns. They check uncertainty,
evidence, discrete quantile ties, score-map/category distinctions, extreme logits,
subnormal GRM gaps, CPU residency under an alternate default device, complete
JSON/LaTeX restoration, rehashed numerical tampering, and refusal before bounded
tensor/table allocations. This validates the declared parameterizations and
finite model; it does not establish general vendor equivalence or estimated IRT
calibration coverage.

Primary method references are the official Stata
[nominal-response model manual](https://www.stata.com/manuals/irtirtnrm.pdf),
[partial-credit model manual](https://www.stata.com/manuals/irtirtpcm.pdf), and
the original-author [mirt item-model documentation](https://philchalmers.github.io/mirt/reference/mirt.html).
The known-parameter and declared finite-prior contract above is narrower than the
estimation workflows described in those references.
