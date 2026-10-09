# Exact finite-posterior IRT draws and replicate predictions

These helpers use a caller-declared fixed item bank and a finite latent prior.
They condition on those item parameters and on each person's observed answers.
They do not estimate item parameters or a population distribution, propagate
calibration uncertainty, or claim equivalence to continuous-prior quadrature.
The posterior support and prior masses are part of the scientific input, not
an automatically chosen numerical integration grid.

```python
draws = oe.irt_plausible_values(posterior, draws=19, seed=83)
prediction = oe.irt_predictive(posterior, items=["item1", "item3"])
score_distribution = oe.irt_test_score_distribution(
    posterior, items=["item1", "item3"], level=.95, max_support=256
)
```

The selected prediction items may include answered or unanswered items. The
posterior always conditions on the original observed-item pattern, including
answers to items outside the selected prediction subset. A missing answer
contributes no likelihood factor. A fully missing pattern retains the declared
prior. Original row positions and source indices remain separate, including
duplicate source indices.

## Finite posterior and reproducible draws

For support points `theta[q]` with strictly positive prior masses `pi[q]`
already summing to one, the
posterior mass for a response pattern `y` is

\[
w_q(y)=\frac{\pi_q\prod_{j\in\mathrm{observed}(y)}P_{j,y_j}(\theta_q)}
 {\sum_r\pi_r\prod_{j\in\mathrm{observed}(y)}P_{j,y_j}(\theta_r)}.
\]

The implementation evaluates these quantities in the log domain. No Gaussian
approximation replaces the finite posterior. `irt_plausible_values` samples
support indices by inverse CDF using an explicitly seeded private CPU Torch
generator. Returned values are supported posterior draws; the draw table is
complete, and repeated calls with identical inputs and seed reproduce it.
The caller's global random state is unchanged. Empirical summaries of a small
draw set still have Monte Carlo error; deterministic posterior moments are
available separately.

The support contains 1..101 sorted distinct points in `[-8, 8]`. All supplied
masses are strictly positive and must already sum to one within the stated
input tolerance; the API does not silently normalize an arbitrary weight list.

Posterior draws alone do not implement the complete plausible-value survey
methodology used by NAEP. That methodology also uses an estimated population
structure model with background variables. See the primary
[NCES description of plausible-value generation](https://nces.ed.gov/nationsreportcard/tdw/analysis/est_pv.aspx).

## Future replicate item scores and their joint covariance

The prediction target is a new locally independent response replicate
`Y_rep`, conditional on the person's posterior. It is not uncertainty about
an answer that is already observed. Reusing answered items defines a posterior
replicate check; it does not provide held-out predictive validation.

Each category has a declared integer score `s[j,k]`. Category codes identify
response alternatives; scores specify the reported quantity. In particular,
NRM category labels have no inherent ordinal or numerical-score meaning, and
NRM category slope order need not match score order. The primary
[Stata NRM specification](https://www.stata.com/manuals/irtirtnrm.pdf) uses
category-specific slope/intercept logits with a baseline constraint. Explicit
score maps may contain duplicates and gaps.

At each support point define

\[
m_{jq}=\sum_k s_{jk}P_{jk}(\theta_q),\qquad
v_{jq}=\sum_k s_{jk}^2P_{jk}(\theta_q)-m_{jq}^2.
\]

For the selected item score vector `S_rep`, the exact finite-posterior mean
and covariance are

\[
\mu=\sum_qw_qm_q,\qquad
C=\operatorname{diag}\!\left(\sum_qw_qv_q\right)
 +\sum_qw_q(m_q-\mu)(m_q-\mu)^\mathsf T.
\]

The first term is conditional response variation. The second term accounts
for shared posterior theta uncertainty and usually introduces cross-item
dependence. This is the law of total covariance under local independence.
The returned covariance concerns item **scores**, not a full stacked category
indicator covariance. It may be singular: an item whose categories all share
one score has exactly zero score variance. Total score mean is `sum(mu)` and
total score variance is `sum(C)`, including both off-diagonal contributions.

## Predictive total-score distribution

For every support point, form each item's score PMF by adding probabilities
of categories with the same score. Convolve the selected item PMFs conditional
on that theta, then mix the resulting total-score PMFs with posterior masses:

\[
f(t\mid y)=\sum_q w_q(y)
 [z^t]\prod_j\left\{\sum_kP_{jk}(\theta_q)z^{s_{jk}}\right\}.
\]

Mixing each item's probabilities before convolution loses shared-theta
dependence. The primary
[Lord–Wingersky algorithm extension](https://pmc.ncbi.nlm.nih.gov/articles/PMC4366368/)
describes conditional score recursion and its polytomous extension. Here the
final mixture uses the declared finite posterior for the particular person.

The complete output includes the integer score support, PMF, CDF and
equal-tail quantiles. Impossible scores in gaps retain zero probability.
Quantiles use the discrete inverse CDF, the smallest score whose cumulative
probability reaches the requested probability. These are predictive credible
interval endpoints, not confidence intervals for the observed total score.
The equal-tail probabilities must be representably distinct from zero, one
and each other around `.5`; levels that collapse those probabilities in
float64 are refused before posterior reconstruction.

An independent dependence check uses two binary items and two equally weighted
posterior support points. Both items have conditional success probabilities
`.1` and `.9`. The correct total-score PMF is `[.41, .18, .41]`, cross-item
covariance is `.16`, and total variance is `.82`. Convolving marginal success
probabilities `.5` instead gives the incorrect PMF `[.25, .5, .25]`.

## Bounded execution and portable evidence

The three helpers use CPU float64 calculations, reject other requested
devices, and admit their additional work and named buffers together with
posterior reconstruction before constructing tensors. Prediction is bounded
to 128 people; posterior draws support up to 1000 people and 99 draws per
person. Scores are
caller-declared nonnegative integers `0..10`. Total-score support is bounded
by the caller's `max_support` before convolution; its default is 256.
Default work and named-buffer ceilings are 300 million units and 128 MiB,
also subject to the current workspace budget. These bounds concern admitted
numerical/output buffers, not caller objects or process RSS.

Complete predictive JSON has a separate 32 MiB limit. Before numerical
posterior reconstruction, the helpers bound every output row, numeric cell,
row label and item name using the full writer's Unicode escaping convention.
The bound also includes complete input state and metadata. Twice this
conservative serialized size participates in the combined workspace plan.
Long labels, many draws or a large covariance table can therefore be refused
within the people/item limits. The result is never truncated to satisfy the
portable-state limit. Core posterior reuse has its own conservative eight-MiB
full-summary domain, checked before numerical allocation.

Complete predictive results retain their finite posterior input and settings
for portable `oe.summary_state` / `oe.restore_summary` JSON restoration.
Each consumer semantically validates its input finite posterior and canonical
tables; input checksums detect accidental drift and are not authenticated
signatures. Generic summary restoration preserves a predictive artifact and
does not rerun its calculations. Roundtrip checks compare every table and
persisted attribute, including draw settings, item selection and source-index
alignment.
Numeric source labels also retain an exact JSON/table/LaTeX roundtrip. The
complete output is checked for actual exportability before a successful return.

`tests/test_irt_calibrated_predictive.py` uses independent NumPy item formulas
and exhaustive response-pattern enumeration on small mixed banks. It checks
category marginals, full score covariance, PMF/CDF/quantiles, shared-theta
dependence, one-point priors, score gaps/duplicates, missing patterns, row
identity, subset selection, private seeded draws, portable state and admission
errors, including escaped Unicode output limits before tensor reconstruction.
These are bounded method-level oracles. Licensed software execution,
continuous quadrature equivalence, survey inference and delivery/runtime
evidence are separate acceptance layers.
