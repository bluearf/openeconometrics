# Strict rank-ordered logit

`rologit` estimates a common numeric attribute coefficient vector from full or
partial rankings. Each row is one alternative in one choice case. Rank **1 is
best**, followed by distinct consecutive ranks; **0 means unranked**. A partial
ranking asserts that every ranked alternative is preferred to the remaining
unranked alternatives. It does not assert an order among those remaining rows.

```python
import openecon as oe

fit = oe.rologit(data, "rank", ["price", "quality"],
                 case="choice", alternative="alternative",
                 available="available", vce="cr0", cluster="respondent")
restored = oe.rologit_restore(fit)
first = oe.rologit_predict(restored, data=new_choices, mode="first")
stages = oe.rologit_predict(restored, data=ranked_choices, mode="stages")
effects = oe.rologit_margins(restored, data=new_choices, mode="first")
```

These are specialized `TableSet` procedures. They do not change the generic
binary matched-set `clogit` route or register a generic saved `oe.predict`
adapter. Read each named table directly and use `result.to_latex()` to export
all tables. The accompanying [synthetic example](../examples/rank_ordered.py)
saves complete tables, attributes and LaTeX and checks JSON restoration.

## Likelihood and identification

For a case's observed prefix, multiply the chosen alternative's softmax
probability at each successive stage, removing only previously chosen rows.
Unranked available alternatives stay in each denominator. The terminal
singleton of a full ranking has probability one and contributes zero score
and information. A first-choice-only ranking reduces to multinomial choice;
later ranks contribute additional data. This is the strict ranking random
utility model under iid extreme-value errors and the associated IIA
assumption, derived in [Train, chapter 7, section 7.3.1](https://eml.berkeley.edu/books/choice2nd/Ch07_p151-182.pdf).

Specify alternative attributes explicitly. A common intercept and columns
constant within every choice case are unidentified and refused. Supply any
alternative indicator columns yourself, leaving one reference level out.
Identification is checked after within-case centering and scaling, and final
unregularized observed information must be positive definite. Finite interior
estimates must satisfy the actual score/Newton-step stationarity checks;
separation and failed convergence are not repaired with an implicit penalty.

Every case needs at least two available alternatives and one observed rank.
Unavailable rows must have rank zero. Typed case/alternative keys are retained
without string coercion; duplicate alternatives in a case are refused.
All selected numeric cells must be finite. No missing rows are silently
removed, and no categories are implicitly encoded.

## Covariance and inference

`vce="oim"` uses the inverse observed information. `vce="hc0"` forms an
uncorrected sandwich from complete original choice-case scores. `vce="cr0"`
first sums those scores over the supplied respondent cluster and uses that
cluster meat. A respondent cluster must be constant within a choice case.
Successive stages of one ranking are never treated as independent subjects.
The coefficient, bread, meat and full covariance tables retain every block.

Reported inference uses asymptotic normal reference distributions without a
finite-sample multiplier or invented residual degrees of freedom. Fixed
availability and covariates define the conditional target. Degenerate
first-order delta uncertainty is explicitly unavailable instead of being
reported as a point confidence interval. Cluster inference requires a
substantive independence justification between respondents; the grouping
column itself does not establish it.

## Saved prediction and substitution

The saved state contains the complete selected inputs and fitted quantities.
Restoration validates the schema and canonical checksum, rebuilds the risk
sets, likelihood, scores, information and covariance, and checks stationarity
without fitting again. The checksum detects corruption; numerical replay
also rejects internally inconsistent resealed state. It is not a signature
that authenticates who produced the model.

First-choice prediction permits new cases and requires no observed rank.
Stage prediction uses a supplied strict ranking prefix. Requested alternatives
retain the complete available denominator, even when only a subset of output
targets is materialized. Removed and unavailable rows have structural zero
probability. Ranking probability and log probability describe the supplied
full or partial prefix. Joint delta covariance retains shared coefficient
uncertainty across cases and stages. Stable log-scale targets distinguish
floating-point tail rounding from structural zero support.

Explicit targets select output components while preserving every denominator:

```python
probabilities = oe.rologit_predict(fit, data=new_choices,
                                 targets=[(1000, "A"), (1000, "B")])
prefix = oe.rologit_predict(fit, data=ranked_choices, mode="stages",
                          targets=[(1000, 2, "B"),
                                   {"case": 1000, "kind": "ranking"}])
cross_effect = oe.rologit_margins(fit, data=new_choices,
                                targets=[(1, "A", "B", "price")])
```

Choice targets are `(case, alternative)` in first mode or
`(case, stage, alternative)` in stages mode. Margin targets specify
`(stage, outcome alternative, changed alternative, attribute)` and average
over their explicit case support. A stage is one-based. Case and alternative
labels keep their original integer or string type.

Own/cross substitution differentiates an outcome alternative's probability
with respect to one named attribute of a changed alternative. Its analytic
parameter Jacobian supplies complete delta uncertainty. Margins average over
choice cases with both alternatives eligible at the requested stage;
unbalanced alternative sets therefore have explicitly recorded pair support.
The `support` table records every target and query case, its inclusion flag
and the reason for exclusion. A known unavailable or previously removed
alternative has a structural attribute effect of zero. Those excluded zero
effects stay outside the eligible-pair average denominator, so this is a
pair-conditional average rather than an all-case average. An alternative
absent from a case, or a stage beyond its observed prefix, has no defined
derivative; no attributes are invented for it.

Optional `case_weights` must map every query case to a fixed positive weight
between `1e-12` and `1e12`, inclusive. Weights are normalized separately over
each target's eligible cases. They change that target average only, are not
estimation weights and do not change the fitted sandwich.

Positive-attribute elasticities use the closed-form expression directly,
avoiding division by a rounded zero probability. The changed alternative's
attribute must be strictly positive in every eligible case. An elasticity
whose outcome probability is structurally zero is undefined, despite its
zero attribute effect. If the outcome remains eligible and the changed
alternative is excluded, its elasticity is structurally zero when that
changed attribute is positive.

Per-case Jacobians together with full parameter covariance preserve every
joint covariance block through the lossless factorization
`per_case_jacobian @ coefficient_covariance @ per_case_jacobian.T`.
Materialized average-target covariance is bounded and named. Pointwise
confidence intervals do not establish simultaneous coverage or causal
attribute effects.

## Limits and evidence

Resident CPU float64 input is bounded to 4,096 rows, 512 choice cases,
20 available alternatives per case and eight explicit attributes. Joint
postestimation requests are bounded to 256 targets. Margins additionally
require at most 8,192 complete support rows (targets times query cases) and
8,192 eligible per-case margin rows. Resource refusals occur before the
named derivative and covariance tensor allocations and do not thin risk
sets. Exact accepted budgets and options are retained in each result.

Positive-rank ties, fitted weights, a common intercept, implicit missing-row
deletion, nested or random-coefficient ranking, multinomial probit, Dataset,
CUDA and MPS are outside this contract. The broader MARKET-164 remains open.
Independent development likelihood/derivative and simulation checks are
separate from frozen-runtime, installed native, licensed vendor and public
release evidence. No global vendor parity claim is made.
