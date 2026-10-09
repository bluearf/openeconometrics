# Two-level nested logit

`nlogit` fits disjoint nests from alternative rows with explicit numeric
attributes, a binary chosen column, typed case/alternative/nest keys and
availability. Shared coefficients and free nest dissimilarities are estimated
jointly. The specialized `TableSet` helpers are `nlogit`, `nlogit_restore`,
`nlogit_predict` and `nlogit_margins`; they have no generic `oe.fit`/`oe.predict`
adapter. Read named tables or export all tables with `result.to_latex()`.

```python
import openecon as oe
fit = oe.nlogit(data, "chosen", ["price", "quality"], case="choice",
                alternative="alternative", nest="nest", available="available",
                vce="cr0", cluster="respondent")
restored = oe.nlogit_restore(fit)
probabilities = oe.nlogit_predict(restored, data=new_choices)
effects = oe.nlogit_margins(restored, data=new_choices,
    targets=[{"outcome": "A", "changed": "B", "attribute": "price"}])
```

The [synthetic example](../examples/nested_logit.py) saves complete tables,
attributes and LaTeX separately from the selected displayed tables.

## Model and identification

For utility \(V_j=x_j'\beta\), dissimilarity \(\lambda_m\), and available
alternatives in nest \(m\), define
\(q_{j|m}=\exp(V_j/\lambda_m)/\sum_{k\in m}\exp(V_k/\lambda_m)\),
\(I_m=\log\sum_{k\in m}\exp(V_k/\lambda_m)\), and
\(Q_m=\exp(\lambda_m I_m)/\sum_l\exp(\lambda_l I_l)\).
Then \(P_j=Q_m q_{j|m}\), with root scale normalized to one. This is the
utility-consistent two-level GEV formulation in
[Train, chapter 4, sections 4.2.2–4.2.4](https://eml.berkeley.edu/books/choice2nd/Ch04_p76-96.pdf).
The numerical implementation is original native float64 Torch.

Each alternative retains one stable typed nest assignment; integer and
string labels remain distinct. Supply alternative/nest indicators explicitly
and omit suitable reference levels. No intercept or category encoding is
added. Columns constant within every case are unidentified. Common case
centering precedes utility multiplication and preserves between-nest levels.

The mathematical RUM domain is \((0,1]\). This bounded implementation
supports fixed nest/lambda pairs in \([0.001,1]\) and free estimates strictly
between 0.001 and 0.999. The computational floor bounds the utility division
and high-order derivative tapes; it is not a statistical lower-bound theorem.
`fixed_dissimilarity` supplies those explicit pairs.
Unspecified nonsingleton nests estimate lambda jointly; globally singleton
nests use fixed one. A free nest needs jointly available alternatives and
positive full joint information. Fixing all lambdas to one reproduces
conditional multinomial logit. Empty nests contribute zero; active singleton
conditional probabilities and one-active-nest probabilities are exactly one.
Disjoint cases with one active nest can identify free dissimilarities when
other informative fixed-dissimilarity cases anchor the shared coefficients.
All-free geometry with no between-nest cases is scale confounded and fails
the joint-information check.

The joint likelihood need not be globally concave. Deterministic multiple
starts are recorded and the best certified finite stationary maximum is
selected, without a global optimum guarantee. Physical beta/lambda
stationarity and positive observed information are required; a vanishing
sigmoid-coordinate gradient alone is insufficient. Free boundary estimates,
separation and unidentified models fail. Fixed lambda one is supported;
estimated-boundary Wald/LR theory is outside this contract.

## Sample, inference and persistence

Each case needs 2..20 available alternatives and exactly one chosen available
row. Unavailable rows must be unchosen. Availability accepts booleans or exact
numeric 0/1. All selected numeric cells, including unavailable rows, must be
finite; missing rows are never silently deleted. Resident DataFrames and
column mappings support at most 4,096 rows, 512 cases, eight attributes and
eight nests. Work/workspace admission precedes numerical allocation. These
are implementation resource limits, not statistical requirements.

`oim` inverts full joint observed information. `hc0` uses whole original
choice-case scores; `cr0` first sums them within respondents, whose key is
constant within each case. Complete beta/lambda cross covariance, bread,
meat and scores are saved. Reference distributions are asymptotic normal,
without a degrees-of-freedom multiplier. Between-respondent independence
needs a study-design justification.

Restoration validates sealed complete inputs, typed nest geometry, fixed/free
declarations, physical parameters, diagnostics and inference. It numerically
replays likelihood, scores, information, covariance and stationarity without
optimization. The checksum detects corruption; it is not authentication.

## Queries and substitution

New cases and alternatives in declared nests are supported. A known
alternative cannot change its nest. Targets select alternative, within-nest,
nest or log probabilities while retaining every available denominator row.
Full joint parameter delta covariance includes shared beta/lambda uncertainty.
Unavailable alternatives and empty nests have structural zero probability;
undefined conditional or log targets fail. Stable log targets retain tail
information after ordinary probabilities round or underflow.

For outcome \(i\) in nest \(a\), changed alternative \(j\) in nest \(b\),
and attribute \(r\), the analytical effect is

\[
\frac{\partial P_i}{\partial x_{jr}}=\beta_r P_i\left[
1(a=b)\left\{\frac{1(i=j)}{\lambda_a}
+\left(1-\frac1{\lambda_a}\right)q_{j|b}\right\}-P_j\right].
\]

Elasticity multiplies the bracket by \(\beta_r x_{jr}\), requires a positive
changed attribute, and avoids dividing by an underflowed outcome probability.
Fixed case weights standardize over simultaneous eligible outcome/changed
support. Per-case support and exclusions, Jacobians, average Jacobians and
full covariance are retained. Query sets and weights are fixed; estimated
weight uncertainty is unsupported. Zero first-order variance does not imply
an invented point confidence interval. Materialization is bounded to 256
joint targets and 8,192 per-case margin rows, without denominator truncation.

This delivery supports resident CPU float64. Fit weights, Dataset, CUDA/MPS,
cross-nested or deeper trees, multinomial probit, automatic nest selection and
proprietary-vendor parity are unsupported. MARKET-164 remains open for the
broader choice family and licensed-reference acceptance.
