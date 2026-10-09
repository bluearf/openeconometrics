# Bounded scored full-profile conjoint

This family implements MARKET-288..295 under MARKET-173. It provides scored
full-profile studies, declared plans, individual utilities and conditional
preference calculations. It does not close the broader ranked/sequence,
choice-based conjoint or general orthogonal-array-search backlog.

```python
import openecon as oe

attributes = {"brand": ["a", "b"], "price": [10, 20, 30, 40],
              "speed": ["low", "high"], "quality": ["basic", "premium"]}
full = oe.conjoint_plan(attributes)
profiles = full["profiles"]
# Choose and record training and holdout profiles BEFORE collecting scores.
training = profiles.iloc[[i for i in range(len(profiles)) if i % 4 != 0]]
held = profiles.iloc[::4]
# responses: one row per subject/profile_id, numeric score where larger is preferred.
fit = oe.conjoint_fit(training, responses, attributes, factors={"price": "ideal"})
predicted = oe.conjoint_predict(fit, held)
validated = oe.conjoint_holdout(fit, held, holdout_responses)
importance = oe.conjoint_importance(fit)
shares = oe.conjoint_simulate(fit, held, method="logit", temperature=1.0)
oe.conjoint_save(fit, "my-study.json")
restored = oe.conjoint_load("my-study.json")
```

The runnable [synthetic example](../examples/conjoint_eight.py) contains all eight
procedures and all data locally. No external customer data is accessed.

## Plans and geometry

`conjoint_plan` emits the complete Cartesian product in declared attribute/level
order, with stable string IDs; the last attribute varies fastest. It never
silently subsamples an oversized design. Each attribute has 2..32 distinct
nonmissing scalar levels. IDs/levels are finite numbers or nonempty strings up
to 128 characters; booleans and composite labels are unsupported. Numeric
integer/float promotion preserves equality; strings remain distinct from numbers.

`conjoint_orthogonal(attributes, base, generators, signs=...)` constructs a
declared regular two-level fraction. All independent base combinations are
enumerated, first base factor fastest. Generator words contain distinct base
names; low/high levels are -1/+1 characters, multiplied by a declared +/-1 sign.
Every non-base factor needs one nonconstant character distinct from all other
main effects. Main effects are balanced and mutually orthogonal; interactions
can be aliased. This is a declared construction, without automatic optimal-plan
or arbitrary mixed-level orthogonal-array search.

`conjoint_diagnostics` reports the actual design rank, condition, full main-effect
Gram, declared category counts, between-factor centred block cross-products and
Cramer's V. Within-factor effects-coded columns need not be mutually orthogonal.
For all-discrete two-level factors it additionally audits every exact signed
alias among intercept, main effects and two-factor interactions. This does not
rule out higher-order aliases. Rank-deficient diagnostics remain inspectable;
the fitting procedure rejects rank deficiency and condition above 1e10.

## Individual utilities and uncertainty

`conjoint_fit` requires explicit profile-ID/subject/score columns in long form.
The plan ID order determines the fitted design; responses join by IDs, never
their row position. Each subject must score every training profile exactly once.
Duplicate pairs, missing IDs, unknown IDs or nonfinite scores fail. With
`missing="drop_subjects"`, whole incomplete respondents are explicitly listed;
no subject is fitted on a quietly shortened profile plan. Original response
positions and scalar index labels are recorded after the keyed join.

Discrete factors use sum-zero effects coding, with the final declared level
equal to the negative sum of the other utilities. Linear and quadratic (`ideal`)
factors fit centred/scaled numeric columns. A full linear transformation exports
raw-scale coefficients and the entire raw covariance, including intercept and
cross-factor covariance. Numeric level magnitudes are bounded by 1e6 and score
magnitudes by 1e12. A quadratic may be ideal, anti-ideal or effectively linear;
its curvature is estimated without forcing a maximum.

The solver is direct float64 QR. Residual variance uses df=training profiles minus
free coefficients. Individual coefficient and level-utility tables carry
estimate, SE, df, two-sided t/p and the requested t interval. This is conventional
conditional iid OLS covariance; Gaussian within-respondent errors justify finite
sample t inference. Zero variance yields SE=0/point intervals with undefined
t/p. Weighted, robust, clustered and group-level sampling covariance are
unsupported. Equal-subject mean utilities/coefficients are explicitly descriptive.

`conjoint_predict` reuses saved training anchors, coding, individual coefficients
and full covariance, with no refit. It returns all subject/profile predictions,
conditional fitted-mean SE and t CI. Unknown levels are rejected even for numeric
factors. These are fitted-mean intervals, without new response noise or
cross-subject uncertainty. Training predictions are permitted; validation uses
the separate leakage guard below.

## Validation, importance and simulation

`conjoint_holdout` refuses both training IDs and renamed training factor
combinations. It requires complete held-out scores for every retained saved
subject and records any explicit whole-subject exclusions, including entirely
absent subjects. It reports descriptive Pearson correlation, tie-adjusted
Kendall tau-b, MAE and RMSE. Constant scores/predictions produce explicit
undefined-correlation flags. Holdout scores never estimate utilities. Pearson
holdout validation is our additional descriptive target; no IBM holdout Pearson
significance or proprietary output equivalence is claimed.

`conjoint_importance` computes each attribute's utility range over the declared
finite level universe and normalizes those ranges within each subject. Group
importance averages the individual normalized percentages equally. It separately
labels the different quantity obtained by normalizing ranges of mean utilities.
A zero total range is explicitly undefined, preventing an incomplete group
average. Importance has no fabricated standard error or sampling interval.

`conjoint_simulate` accepts declared known-level alternatives. First-choice
assigns mass equally to exact maxima, or to maxima within an explicitly supplied
absolute `tie_tolerance`. BTL divides positive predicted scores by their sum;
one nonpositive score aborts the entire calculation instead of shifting scores
or dropping respondents. Logit applies stable softmax at a declared positive
temperature and accepts any finite score. The latter differs from legacy IBM
positive-score respondent filtering. All methods average conditional probabilities
equally across saved respondents. Shares are model-based descriptions, without
population market-share CI, consumer sampling claims or CBC model estimation.

## Budgets and persistence

Domain: 4096 profiles, 16 attributes, 128 coefficients, 128 subjects and 100000
response cells. Every numerical procedure checks named conservative workspace
buffers against the actual task budget and work against `max_work` (default
100000000) before the numerical allocations. Holdout pair comparisons have their
own quadratic work/buffer guards. These are buffer estimates, not a total
process-memory promise; resident tabular inputs and integrity serialization are
separate from the numerical buffer estimate.

Resident CPU float64 only. Dataset, CUDA/MPS, observation/respondent weights,
rank/sequence input and unsupported covariance fail explicitly. CPU placement
does not inherit the caller's default device/dtype. `conjoint_save/load` retains
all tables, full covariance, coding, declared levels, original sample identities,
options, diagnostics and state inside a checksummed artifact up to 32 MiB.
Any modified result/state/table is refused before postestimation or saving.
Console previews are not complete artifacts; save the entire JSON explicitly.

## Primary references and bounded comparison

* [IBM original CONJOINT algorithms](https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/conjoint.pdf): effects coding, numeric centring, OLS, utility covariance, individual-normalized importance and preference models. Our complete raw covariance/prediction oracles are independent NumPy/SciPy development calculations; they do not run in the implementation.
* [IBM CONJOINT command](https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=conjoint-overview-command): full-profile score/rank distinction, utility factor models and holdout roles. Our supported input is scored, explicitly keyed long form.
* [NIST published resolution-IV design](https://www.itl.nist.gov/div898/handbook/pri/section3/eqns/2to4m1.txt): the exact eight-run D=ABC fixture and its pair-interaction aliases.
* [NIST fractional-design tables](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3347.htm): declared generator and resolution conventions.

No licensed IBM/Stata executable comparison, blanket parity, GPU validation or
public release is claimed. Source, package, frozen, installed native and restart
verification are recorded separately in the acceptance evidence.
