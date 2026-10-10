# Bounded scored full-profile conjoint

This family implements MARKET-288..295 and MARKET-785..792 under MARKET-173. It provides scored
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
t/p. The default equal-subject mean utilities/coefficients remain descriptive;
`conjoint_group_mean` explicitly requests the separate sampling target below.
Observation/respondent weights remain unsupported.

`covariance="hc0"`, `"hc1"`, `"hc2"` or `"hc3"` supplies a full within-respondent
heteroskedastic covariance. With scaled design X, residual e and B=(X'X)^-1,
HC0 is B X' diag(e²) X B. HC1 multiplies HC0 by n/(n-p). HC2 and HC3 replace
e² with e²/(1-h) and e²/(1-h)², respectively. Unit leverage (1-h <= 1e-12)
is refused for HC2/HC3; no profile is discarded or leverage clipped.

`covariance="cr0"` or `"cr1"`, with `cluster="session"`, groups profile scores
within **each respondent** using that response column. Cluster labels are joined
by original subject/profile keys, retaining typed scalar IDs and all original
row identities. CR0 sums cluster-score outer products without a correction;
CR1 multiplies it by G/(G-1)*(n-1)/(n-p). At least two clusters must remain for
every complete respondent. Missing/invalid labels fail; an explicitly dropped
incomplete respondent is excluded as a whole. Cross-respondent and multiway
profile clustering are outside this contract.

HC reporting uses t(n-p), and CR uses respondent-specific t(G-1), for marginal
p-values and intervals. These are **approximate reporting conventions**, not
exact finite-sample robust coverage or a remedy for few clusters. Residual
variance diagnostics retain n-p; saved records retain inference df, correction
multiplier and complete ordered cluster IDs. Every covariance is transformed
in full to raw coefficient and level-utility scales.

`conjoint_predict` reuses saved training anchors, coding, individual coefficients
and full covariance, with no refit. It returns all subject/profile predictions,
conditional fitted-mean SE and method-specific t CI. Unknown levels are rejected even for numeric
factors. These are fitted-mean intervals, without new response noise or
cross-subject uncertainty; robust fits retain their approximate HC/CR df convention. Training predictions are permitted; validation uses
the separate leakage guard below.

## Saved contrasts and respondent means

```python
robust = oe.conjoint_fit(training, responses, attributes,
                        factors={"price": "ideal"}, covariance="hc3")
contrasts = oe.conjoint_contrast(
    robust, {"price 10 to 20": {"price:linear": 10, "price:quadratic": 300}},
    null={"price 10 to 20": 0},
)
group = oe.conjoint_group_mean(robust)
oe.conjoint_save(contrasts, "price-contrast.json")
oe.conjoint_save(group, "respondent-mean.json")
```

`conjoint_contrast` accepts named linear maps of **saved raw coefficient terms**,
plus an optional complete map of null values. It returns every respondent's
estimates, full joint target covariance, method-specific pointwise t inference
and a joint test. The nonrobust Gaussian model uses exact conditional F(q,n-p).
HC/CR uses asymptotic Wald chi-square(q), explicitly without a finite-cluster
exactness claim. Empty/zero/redundant maps, unknown/nonfinite weights or nulls,
and singular joint covariance fail without pseudoinverse repair. Targets are
computed in the saved scaled fit basis to avoid unnecessary raw-polynomial
cancellation. Neither fitting nor the original score data are needed again.

`conjoint_group_mean` targets the population mean of fitted respondent
coefficients **assuming independent iid sampled respondents**. It uses the mean
raw coefficient vector and its complete empirical covariance
sum((b_i-bbar)(b_i-bbar)')/[m(m-1)], with m>=2. The same linear map supplies the
complete joint level-utility covariance. Within-person fitting noise is already
part of the between-respondent dispersion; adding individual covariances would
double count that component. Marginal t(m-1) is approximate unless the fitted
respondent vectors are Gaussian. A convenience sample does not acquire a
population sampling justification by calling this procedure. No importance,
preference-share, survey-weighted or cross-respondent dependency CI is supplied.
Zero empirical variance remains explicit, with point intervals/undefined tests.

Both procedures retain complete checked tables and state in the existing JSON
artifact format and restore without optimization. Existing v1 artifacts remain
readable. Work/workspace limits admit full target matrices before allocation;
CPU, resident-table, no-weight and integrity policies continue to apply.

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

## Explicit bootstrap sampling laws

MARKET-805..812 add eight remaining uncertainty domains through three APIs:

```python
for law in ("residual", "rademacher", "mammen", "pairs"):
    uncertainty = oe.conjoint_bootstrap_fit(restored, method=law,
                                            replications=499, seed=1729)
    oe.conjoint_save(uncertainty, f"{law}-uncertainty.json")

importance_uncertainty = oe.conjoint_bootstrap_importance(restored)
for method in ("first_choice", "btl", "logit"):
    shares_uncertainty = oe.conjoint_bootstrap_shares(restored, held, method=method)
```

Choose the law that matches the sampling design; these are different assumptions.
`conjoint_bootstrap_fit` uses saved keyed training scores, original design anchors
and full OLS refits. It supplies every raw coefficient and finite-level utility,
their complete **joint** covariance, bootstrap mean/bias and marginal percentile
intervals. Original subjects, profile IDs, source positions/labels, all random
draws, scaled design, target map and every replica are retained.

* `residual`: fixed profiles with independent homoskedastic errors. Center the
  fitted residuals, multiply by sqrt(n/(n-p)), resample them with replacement,
  add to the original fitted scores and refit against the original X.
* `rademacher`: fixed profiles with independent, potentially heteroskedastic
  errors. Multiply each HC2-adjusted residual e/sqrt(1-h) by independent +/-1
  with equal probabilities, add to fitted scores and refit. Unit leverage fails.
* `mammen`: the same fixed-design HC2 construction, with multipliers
  (1-sqrt(5))/2 and (1+sqrt(5))/2, probabilities (sqrt(5)+1)/(2sqrt(5)) and
  (sqrt(5)-1)/(2sqrt(5)). Their first three moments are 0, 1, 1.
* `pairs`: iid **random** profile-score rows within a respondent. Resample the
  complete (X,y) row with replacement, retaining the original coding. This is
  not conditional fixed-plan sampling. Any rank-deficient/ill-conditioned
  replica aborts the complete operation; there is no redraw, skipped replica,
  regularization or pseudoinverse.

These four laws assume independent profile errors and reject CR0/CR1 fits;
clustered profile resampling needs a separate block law. They estimate sampling
uncertainty of fitted utilities, without new-response prediction intervals.

The two group APIs instead resample **whole independent iid respondents**. They
reuse all saved individual targets and never resample profiles or refit people.
`conjoint_bootstrap_importance` targets the mean of individually normalized
importance percentages over the declared levels. It differs from normalizing
the ranges of mean utilities; any undefined individual's importance fails the
complete operation. `conjoint_bootstrap_shares` targets mean conditional fitted
shares for the supplied alternative set, preserving exact/declared ties,
positive-score BTL and logit temperature rules. It retains every individual
vector, selected respondent positions, full covariance and every replica.

Group inference concerns the population mean of these **fitted individual
targets** under iid respondent sampling. It does not remove fitting noise or
nonlinear estimation bias, turn a convenience sample into a random sample,
fit a CBC model or calibrate true population market shares. Within-person CR
fits are permitted here because entire respondents remain the sampling units;
cross-respondent dependence and survey weights remain unsupported.

All procedures use 20..1999 replications (default 499) and a declared integer
seed, with type-7 linear-interpolated empirical percentiles at (1-level)/2 and
(1+level)/2. Covariance uses the complete replica matrix with divisor B-1.
Covariance can be singular because of utility constraints or shares summing
to one; it is returned intact without an invented inverse. Zero empirical
variance has an explicit flag. These are approximate **marginal** intervals;
no joint confidence band, bootstrap-t/BCa refinement, p-value or exact coverage
is claimed. Small B has coarse Monte Carlo precision. There is no universal
nominal-coverage claim for small, discrete or nonregular cases.

Every operation admits cumulative numerical work and complete draw/replica/
covariance/serialization buffers before numerical refits. The entire result
must fit the existing 32 MiB artifact bound. A saved result restores complete
tables and LaTeX without refitting or drawing new randomness. Independent tests
compare every replica against raw-basis NumPy least squares and all moments/
quantiles against NumPy, including exhaustively enumerated Rademacher and iid
respondent laws. These are algorithm/law checks, not a finite-sample coverage
certificate or a whole-current-main desktop release.

## Budgets and persistence

Domain: 4096 profiles, 16 attributes, 128 coefficients, 128 subjects and 100000
response cells. Every numerical procedure checks named conservative workspace
buffers against the actual task budget and work against `max_work` (default
100000000) before the numerical allocations. Holdout pair comparisons have their
own quadratic work/buffer guards. These are buffer estimates, not a total
process-memory promise; resident tabular inputs and integrity serialization are
separate from the numerical buffer estimate.

Resident CPU float64 only. Dataset, CUDA/MPS, observation/respondent weights,
rank/sequence input and unsupported covariance domains fail explicitly. CPU placement
does not inherit the caller's default device/dtype. `conjoint_save/load` retains
all tables, full covariance, coding, declared levels, original sample identities,
options, diagnostics and state inside a checksummed artifact up to 32 MiB.
Any modified result/state/table is refused before postestimation or saving.
Console previews are not complete artifacts; save the entire JSON explicitly.

## Primary references and bounded comparison

* [IBM original CONJOINT algorithms](https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/conjoint.pdf): effects coding, numeric centring, OLS, utility covariance, individual-normalized importance and preference models. Our complete raw covariance/prediction oracles are independent NumPy/SciPy development calculations; they do not run in the implementation.
* [IBM CONJOINT command](https://www.ibm.com/docs/en/spss-statistics/32.0.0?topic=conjoint-overview-command): full-profile score/rank distinction, utility factor models and holdout roles. Our supported input is scored, explicitly keyed long form.
* [Stata regress methods and formulas](https://www.stata.com/manuals/rregress.pdf): independently implemented HC0/HC1/HC2/HC3 score sandwiches, one-way cluster correction and df conventions. This is a published-method reference, not licensed executable parity.
* [Efron and Tibshirani (1986)](https://doi.org/10.1214/ss/1177013815): bootstrap sampling, standard errors and intervals. Our saved finite draw matrix and marginal quantile convention are explicit.
* [Flachaire, original author manuscript](https://www.math.kth.se/matstat/gru/sf2930/papers/Flachaire_03a.pdf): distinct wild and pairs regression sampling laws and leverage-adjusted residual constructions. Our intervals are unstudentized percentile approximations; the manuscript's refined hypothesis-test performance is not claimed.
* [NIST published resolution-IV design](https://www.itl.nist.gov/div898/handbook/pri/section3/eqns/2to4m1.txt): the exact eight-run D=ABC fixture and its pair-interaction aliases.
* [NIST fractional-design tables](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3347.htm): declared generator and resolution conventions.

No licensed IBM/Stata executable comparison, blanket parity, GPU validation or
public release is claimed. Source, package, frozen, installed native and restart
verification are recorded separately in the acceptance evidence.
