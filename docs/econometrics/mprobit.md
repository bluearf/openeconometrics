# Bounded multinomial probit

`mprobit` fits a Gaussian random-utility choice model to complete resident long
data with two or three known alternatives. Each case supplies alternative-level
numeric attributes, availability and exactly one available chosen alternative.
The fitted catalogue fixes the alternative identities and their order. Cases
may have different available subsets; every available alternative contributes
to the likelihood denominator. Missing values and invalid choice sets fail
explicitly.

The utility for alternative `j` is `x[j]' beta + epsilon[j]`. With the first
catalogue alternative as reference, the model identifies error differences.
The second alternative's difference variance is fixed to one, removing the
utility scale indeterminacy. For three alternatives, the normalized covariance
is

```text
Gamma = [[1, sd3 * rho3],
         [sd3 * rho3, sd3**2]]
```

Free covariance estimation includes `sd3 > 0` and `-1 < rho3 < 1` jointly with
the coefficients. The saved covariance, score and observed-information tables
use these physical parameters. A caller may instead pass an explicit positive
definite normalized `covariance` matrix; fixed covariance coordinates are held
known and do not appear as estimated parameters. Two-alternative fits use the
normalized scalar covariance `[[1]]`. A common case utility shift does not
change probabilities. A common alternative-invariant attribute cannot identify
an additional coefficient.

The supported free numerical interior is narrower: `0.05 < sd3 < 20` and
`abs(rho3) < 0.98`; fixed covariance permits these endpoints. Every alternative
comparison must also retain variance and correlation complement above `1e-8`.
These are declared computational conditioning guards, distinct from the
mathematical positive-definite domain. Raw numeric magnitudes are bounded by
`1e12`. Likelihood and query evaluation additionally require every absolute
standardized Gaussian event threshold, and both conditional-normal thresholds
of every bivariate event, to be at most `16`. This computational
derivative-accuracy guard applies before evaluating likelihoods, predictions or
delta-method derivatives. Requests beyond it fail explicitly; thresholds and
probabilities are never clamped. The bound keeps Gaussian boundary densities
and their conditional factors within the normal float64 range before raw-cell
scaling. It is separate from the raw-input magnitude bound and from the
mathematical Gaussian model. Three deterministic covariance
starts are compared; accepted fits are
stationary interior local likelihood maxima, without a global-maximum guarantee.

Two-choice probabilities reduce to a univariate normal CDF, using the variance
of the appropriate difference even when the available pair omits the reference
alternative. Three-choice probabilities are bivariate normal orthant integrals.
The production engine uses deterministic native float64 Torch quadrature and
differentiates its likelihood, without importing a third-party estimator. The
bounded numerical integral and its convergence checks are part of the saved
contract; they are not simulated-choice likelihood with caller Monte Carlo
draws. [Train, chapter 5](https://eml.berkeley.edu/books/choice2nd/Ch05_p97-133.pdf)
explains the random-utility difference normalization and probit choice integrals.

## Fit and uncertainty

```python
import openecon as oe

fit = oe.mprobit(data, "chosen", ["price", "quality"],
                 case="case", alternative="alternative",
                 alternatives=["base", "second", "third"], available="available",
                 vce="cr0", cluster="respondent")
restored = oe.mprobit_restore(fit.attrs)
```

OIM uncertainty inverts the observed information of the complete physical
coefficient/covariance vector. HC0 uses case likelihood scores. CR0 first sums
case scores within the caller's respondent clusters, then forms the sandwich.
These are asymptotic covariance conventions, without small-sample correction
or Student-t claims. Original-case scores retain the admitted case grouping;
long rows are not independent score units. Unidentified parameters, boundary
covariance, invalid information, failed convergence and exhausted resource
budgets fail explicitly. No penalty or covariance regularization silently
repairs a refused model.

The sealed JSON-compatible fit state retains primitive inputs, typed case and
alternative identities, choice sets, covariance normalization, physical
parameter order, likelihood and inference settings. `mprobit_restore` verifies
the stored numerical contract and rebuilds complete tables without optimizing.
The checksum detects inconsistent changes; it does not authenticate the data
or sampling provenance.

## Saved prediction

```python
predictions = oe.mprobit_predict(fit, query,
    targets=[{"case": "A", "alternative": "second"}], kind="probability")
log_predictions = oe.mprobit_predict(fit, query,
    targets=[{"case": "A", "alternative": "second"}], kind="log_probability")
```

Query data need the fitted raw attributes and case/alternative/availability
columns, with no choice or respondent-cluster column. Unknown alternatives are
refused. New cases and unbalanced subsets of known alternatives are admitted.
Requested unavailable alternatives have structural probability zero; their log
probability is undefined and explicitly refused. A singleton query choice set
has structural probability one and zero parameter derivative.

Default prediction materializes every available alternative only when the
complete requested count is at most 256. A larger request needs an explicit
subset; this never truncates the alternatives used to evaluate probabilities.
Each target may override the global `kind` with its own probability/log kind.
The `predictions` table reports the estimate, asymptotic standard error, interval
and support. `jacobian` and `covariance` preserve every requested target's full
physical derivative and joint covariance. `joint_jacobian` and
`joint_covariance` additionally retain log probabilities and log odds for
nonstructural targets; their order is declared in settings. Probability intervals
transform normal log-odds intervals. If displayed probability endpoints round
together, their log endpoints remain available. Log-probability upper limits
are projected to zero. A zero first-order variance has explicitly unavailable
normal inference; known structural probabilities retain point intervals.

## Raw-cell effects and fixed averages

```python
effects = oe.mprobit_margins(fit, query, targets=[
    {"outcome": "second", "changed": "third", "attribute": "price"},
], average=True, weights=[1., 2., 3.])
elasticities = oe.mprobit_margins(fit, query, attribute="price",
                                  elasticity=True)
```

An effect is `d P(outcome) / d x(changed, attribute)` with every other raw cell
fixed. An elasticity is `x(changed, attribute) * d log P(outcome) / d x`, avoiding
division by an underflowed probability. Exact native mixed derivatives propagate
joint coefficient and free covariance uncertainty, including their cross terms.
There is no formula-expansion or automatic categorical-change interpretation.
Adding up outcome effects is zero, and summing the effects of a common attribute
shift across available alternatives is zero.

Explicit targets name exactly `outcome`, `changed` and `attribute`. The
`attribute` shorthand requests all supported own/cross pairs for one fitted raw
attribute; omitting both requests all fitted attributes and supported pairs.
Each pair requires simultaneous available support. The `support` table retains
every query case's inclusion and reason; missing/unavailable pairs do not enter
that pair's average. Unavailable raw-cell probability effects are structural
zeros, documented in support. Elasticities require a strictly positive changed
attribute in every eligible case and a defined available outcome probability;
no additional domain filtering occurs.

`average=True` adds fixed case-weighted means, with each pair's complete support,
positive weight denominator and normalized weights. Weights are finite and
strictly positive, supplied in query case order, a complete case mapping, or
typed `{"case": label, "weight": value}` records. Integer and string labels
remain distinct. Rescaling all weights does not change the average. The
complete `jacobian` and `covariance` contain every per-case effect followed by
all averages, including cross-case and effect/average covariance. Query choice
sets and weights are fixed; this is parameter uncertainty without empirical
evaluation-population sampling uncertainty or a causal claim.

## Bounds and evidence

Fit admission is bounded to two or three alternatives, at most eight numeric
attributes and 512 cases. Prediction/margins have at most 256 declared targets
and 8,192 complete pair/case support rows. Explicit `max_work` and `max_bytes`
admit complete quadrature, mixed derivatives, full target covariance and table
storage before evaluation. Large joint covariance requests may therefore be
refused below the row bound. These are resident implementation resource guards,
not statistical sample-size limits. No denominator is truncated to fit a budget.

The [runnable example](../examples/mprobit.py) retains OIM, HC0, CR0, exact saved
replay, a fixed binary reduction, probabilities/log probabilities, raw-cell
effects/elasticities and fixed averages. Analytical binary reductions and an
independent adaptive Gaussian-integral oracle test physical derivatives and
complete joint covariance. Source tests, compiled frozen runtime, actual native
Run and full Quit/relaunch persistence are distinct acceptance layers. Licensed
vendor execution and blanket Stata parity remain unclaimed. General larger
choice catalogues, random coefficients, panel latent-error integration,
multinomial simulation estimators, Dataset streaming and GPU execution are
outside this bounded implementation.
