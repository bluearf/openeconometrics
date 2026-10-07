# Long-run, structural and decomposition models

The public functions below use repository-owned float64 Torch kernels. SciPy,
statsmodels and arch are development references only. Each fit returns a
`ResultBundle` with the full reported covariance, uncertainty, original input
row positions, explicit specification, reproducible inference settings and
JSON/LaTeX output. This is bounded method coverage, not global Stata parity.
These 14 routes accept in-memory tabular inputs. Dataset replay/streaming and
common saved predict/margins adapters are not provided for this wave; requests
are rejected instead of silently collecting a Dataset or inventing predictions.

## Panel ARDL

```python
pmg = oe.pmg(data=panel, y="y", x=["x"], panel="unit", time="t", p=2, q=2)
mg = oe.mg(data=panel, y="y", x=["x"], panel="unit", time="t", p=2, q=2)
dfe = oe.dfe(data=panel, y="y", x=["x"], panel="unit", time="t", p=2, q=2)
comparison = oe.panel_ardl_hausman(mg, pmg)
```

All models estimate `Δy = φ*(L(y) - θ*L(x)) + short-run differences +
deterministics`. PMG constrains long-run slopes and permits unit-specific
adjustment, short-run slopes and Gaussian variances. Its profile likelihood
uses native BFGS with a declared iteration/convergence contract; full joint
observed-Hessian covariance includes each log variance as nuisance state.
MG fits unrestricted unit ECMs and reports equal-unit averages and
between-unit covariance with t(G−1) inference. DFE constrains ECM slopes and
variance while retaining unit deterministic terms. Its Gaussian covariance
and long-run ratios use the full linear covariance and delta transform.
`p,q` are common integer orders 1..12; `trend` is n/c/ct, default c.
Coefficients include adjustment and short-run terms; unit designs, effective
rows, full covariances and convergence diagnostics remain in saved state.

Hausman comparisons require identical source data, sample and p/q/trend.
A non-positive-definite covariance difference returns an explicitly undefined
comparison; it is never projected to a positive matrix. Unbalanced panels with
consecutive integer periods are supported. Interior missing rows, gaps,
singular designs and unidentified adjustment ratios fail with structured
errors. No period is compressed or imputed. Cross-unit independence and the
homogeneity restrictions are maintained model assumptions.

The profile likelihood and full observed Hessian are checked against an
independently optimized SciPy/NumPy Gaussian ECM likelihood. MG/DFE and the
Hausman statistic have separate matrix references. The method follows
[Pesaran, Shin and Smith's PMG model](https://www.econ.ed.ac.uk/papers/id16_esedps.pdf).

## Cointegrating regressions

```python
fm = oe.fmols(data=series, y="y", x=["x"], time="t", bandwidth=6)
dyn = oe.dols(data=series, y="y", x=["x"], time="t", leads=2, lags=2,
              covariance="hac", kernel="parzen", bandwidth=6)
cc = oe.ccr(data=series, y="y", x=["x"], time="t")
pooled = oe.panel_fmols(data=panel, y="y", x=["x"], panel="unit", time="t")
mean_group = oe.panel_dols(data=panel, y="y", x=["x"], panel="unit", time="t",
                           pooling="mean_group")
```

FMOLS retains the endogeneity correction and one-sided serial-correlation
bias. CCR retains the short/long-run transformations. Both store full innovation
LRV and one-sided covariance. DOLS includes contemporaneous differences and
explicit leads/lags; fixed/AIC/BIC/HQIC selection is performed on a common
maximum-trim sample, followed by re-estimation on the selected model's sample.
Full DOLS covariance includes nuisance difference coefficients.

The deterministic basis uses original periods 1..T (n/c/ct/ctt).
Bartlett/Parzen/QS kernels use explicit integer fixed bandwidth, default 4;
automatic bandwidth and separate regressor trends are outside this version.
QS retains every available lag; Bartlett/Parzen support is finite.
`df_adjust=True` multiplies covariance by effective T/(T−k), while coefficient
inference stays asymptotic normal. These fits assume I(1) regressors and an I(0)
cointegrating error; fitting them does not test cointegration.

Pooled panel models constrain level slopes but retain unit deterministic terms
and, for DOLS, unit augmentation. Pooled two-step covariance uses unit-specific
unrestricted long-run variances; HAC DOLS replays within-unit scores with no
cross-boundary lag products. Mean group averages slopes equally, retains
between-unit covariance and uses t(G−1). Neither implies cross-unit dependence
correction. Each unit's coefficients, covariance and effective rows are saved.

The single-series parameter/full-covariance references use
[arch's cointegrating models](https://arch.readthedocs.io/en/stable/unitroot/cointegration.html),
including EIA/FRED WTI/Brent data. arch restarts deterministic time after trim;
tests explicitly change its coefficient/covariance basis to original periods.
Panel pooled coefficients and covariance use a separate NumPy block-design
reference. Missing/gaps/rank/LRV domain errors are checked independently.

## Structural VAR, local projections and connectedness

```python
structural = oe.svar(data=series, y=["growth", "inflation"], time="t", lags=1,
                     short_run=[[None, 0], [None, None]])
responses = oe.svar_irf(structural, steps=8, draws=200, seed=12)
projection = oe.lp(data=series, y="y", x=["shock"], time="t", horizons=8)
instrumented = oe.lpiv(data=series, y="y", x=["shock"], instruments=["z"],
                       time="t", horizons=8)
panel_response = oe.panel_lp(data=panel, y="y", x=["shock"], panel="unit",
                            time="t", horizons=8)
shares = oe.connectedness(structural, horizon=10)
rolling = oe.rolling_connectedness(data=series, y=["growth", "inflation"],
                                  time="t", window=80, horizon=10, lags=1)
```

Exact SVAR accepts K(K−1)/2 zero restrictions on impact and/or long-run impact
for K=2..6. It checks local rotation rank, residual accuracy and sign anchors.
Long-run restrictions require a stable VAR. Over/underidentified and nonzero
restrictions are rejected. The coefficient table explicitly represents the
reduced-form VAR; structural impact matrices are separate saved state.

Sign models accept named response/shock/horizon/±1 records, a seed, draw budget
and required accepted count. Haar O(K) rotations correct QR diagonal signs.
Insufficient admissible draws fail explicitly. Sign and exact-zero systems are
separate targets in this version. Sign IRF intervals are conditional identified
set quantiles, **not sampling confidence intervals**. Exact-model intervals
simulate the joint Gaussian reduced-form coefficient covariance and Wishart
innovation covariance and re-identify every draw; percentile settings and seed
are recorded. An invalid draw fails rather than silently changing the protocol.
Proxy SVAR and panel VAR are outside scope.

LP has exactly one original-unit shock, numeric controls and lagged
outcome/shock/control columns. Horizon-specific future boundaries and original
rows are saved. Cumulative responses target `y(t+h)−y(t−1)`. Time-series
inference uses full cross-horizon Bartlett HAC with HC1 and bandwidth at least
the requested maximum horizon. Panel LP adds unit intercepts and full
unit-cluster CR1 covariance with t(G−1). LP-IV uses explicit excluded instruments,
QR projection and projected-design rank checks; rank establishes numerical
relevance, not instrument validity. Exclusion/exogeneity remains an assumption.

DY uses generalized FEVD for horizons 0..H−1 and normalizes each response row to
100%. Own shares are excluded from FROM/TO/net/total connectedness. Net is
TO−FROM and total is the mean FROM share. The result records raw/normalized
shares, variable ordering, horizon and fitted window size. Rolling windows
retain actual consecutive integer start/end periods and never compress gaps.
FRED macro data checks impact/IRF points, full covariance, simulation intervals
and generalized FEVD against independent statsmodels/NumPy references.

## Mediation and Oaxaca–Blinder

```python
paths = oe.mediation(data=frame, y="y", x=["a"], mediators=["m1", "m2"],
                     controls=["c"], serial=True)
conditional = oe.mediation(data=frame, y="y", x=["a"], mediators=["m1"],
                           moderator="w", at=[-1, 0, 1])
natural = oe.mediation(data=frame, y="binary", x=["a"], mediators=["m1"],
                       outcome_model="probit")
gap = oe.oaxaca(data=frame, y="y", x=["x", "category"], categorical=["category"],
                group="group", groups=["A", "B"], fold=2, reference="pooled",
                reps=200, seed=9)
components = oe.oaxaca_details(gap)
```

Mediation defaults to **associational** effects. Causal labeling requires
explicit consistency, positivity, sequential ignorability and no exposure-induced
mediator/outcome-confounding declarations. The result records that declarations
are not verified from observed data. One exposure and 1..8 observed continuous
mediators are supported. Linear parallel/ascending serial paths include direct,
indirect, total and each path, at explicit moderator values. Exposure→mediator,
mediator→outcome and direct moderation are separately selectable.

All cross-equation score blocks are retained, including weighted linear
fits; native autodiff applies the full joint delta covariance (HC1 or CR1).
Numeric controls are supported; categorical controls must be explicitly encoded.
Logit/probit outcomes support unweighted parallel Gaussian continuous mediators.
They target population-average natural effects on probability scale: logit uses
seeded antithetic integration; probit uses an analytic Gaussian mixture.
Mediator covariance is estimated and retained in nuisance covariance and delta
inference. Per-mediator nonlinear components use declared order-dependent
switching; serial/weighted nonlinear targets are rejected. This is an observed
path framework, not general SEM/GSEM. Published R mediation framing data checks
Gaussian/probit natural effects; independent equations check full joint covariance.
See the [mediation authors' paper](https://doi.org/10.18637/jss.v059.i05) for
identification assumptions; this implementation's inference/integration
protocol is explicitly different from the R package defaults.

Oaxaca explicitly orders two groups, preserving A−B. Reference choices are A,
B, pooled with group dummy, Neumark without group dummy, Reimers and Cotton
(count shares). Three-fold endowments/coefficients/interaction and two-fold
explained/unexplained totals and detailed components are retained. Weighted
means/WLS use declared aweights. Full-category sum-to-zero normalization makes
categorical details invariant to the omitted fitting category.
Stratified pairs bootstrap preserves each group's size; seed/reps, covariance
and detailed percentile intervals are saved. Degenerate draws fail explicitly
instead of being discarded. CCARD replication and separate weighted/dummy
algebra validate results. The statsmodels reference spells Neumark as
`nuemark`, and its automatically swapped Cotton sample sizes need explicit
independent count-share algebra; tests document both reference quirks.
