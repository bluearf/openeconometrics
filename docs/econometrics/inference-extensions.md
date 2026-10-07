# Inference extensions: validated domains

These code-first procedures use native float64 Torch calculations. Their independent
development references are recorded in `tests/fixtures/inference-2026-10-07.json`;
`scripts/generate_inference_reference.py` reproduces those fixtures. Reference
libraries are not runtime dependencies. This is method/option validation, not
blanket Stata parity or a signed desktop release claim.

## Phillips–Ouliaris

```python
po = oe.po_zt(df, "output", ["capital", "employment"], time="year",
              trend="trend", kernel="bartlett", bandwidth=5)
za = oe.po_za(df, "output", ["capital", "employment"], bandwidth=5)
```

`phillips_ouliaris(..., test="Za"|"Zt")` also exposes both statistics. The null
is no cointegration; lower statistics reject. Deterministic cases are `none`,
`constant`, `trend`, `quadratic`; kernels are Bartlett, Parzen and quadratic
spectral. Metadata records original T, AR sample T−1, series count, kernel,
bandwidth and variances. The default bandwidth is `floor(4*(T/100)^(2/9))`,
not arch's automatic bandwidth. Supply an explicit bandwidth for replication.

The retained [arch 8.0.0 implementation and calibration](https://github.com/bashtage/arch/blob/v8.0.0/arch/unitroot/_phillips_ouliaris.py)
cover 2–6 total series; the minimum sample varies by case and is checked before
inference. Below it, `inference="none"` returns statistics only. Complete regular
calendars and full rank are required. There is no dimensional extrapolation.
Resident N is capped at one million and covariance work at 100 million
observation-lag products. The response-surface attribution and license are in
`docs/reference-licenses/arch.txt`.

## Structural weak-IV inference

```python
ar = oe.iv_weak_test(data=df, y="outcome", x=["control"], endog=["treatment"],
                    instruments=["z1", "z2"], null=1.0, method="ar")
clr = oe.iv_weak_test(data=df, y="outcome", endog=["treatment"],
                     instruments=["z1", "z2"], null=1.0, method="clr")
region = oe.iv_ar_confidence_set(data=df, y="outcome", endog="treatment",
                               instruments=["z1", "z2"])
```

AR tests a complete endogenous coefficient vector, with included exogenous
nuisance coefficients projected out. Under iid Gaussian homoskedastic errors it
uses exact F(k, N−C−k). HC0, one-way cluster G/(G−1), and HAC use asymptotic
Wald chi-square(k), with the full null reduced-form covariance retained.
HAC requires ordered, regular `time` and explicit `lags`. Weights and untested
endogenous nuisance coefficients are outside this contract.

Moreira CLR retains the full reduced-form covariance and conditional numerical
quadrature. Its validated domain is one endogenous coefficient with iid
homoskedastic covariance; robust/cluster/HAC CLR is rejected. The confidence-set
API analytically inverts scalar iid AR. Empty, disjoint and unbounded sets remain
distinct; `None` denotes an infinite endpoint. CLR confidence inversion and
robust/subvector inversions are not supplied. Test and coefficient fixtures
match [ivmodels 0.10.0](https://ivmodels.readthedocs.io/en/stable/api.html).

`ivregress(method="fuller", fuller_alpha=1)` uses LIML κ−α/(N−L).
`method="kclass", kappa=...` accepts explicit κ in [0,1], including OLS at 0
and 2SLS at 1. These conventional model covariances do not replace weak-IV tests.
`stock_yogo(result)` supplies the single-endogenous 2SLS 10%-maximum-size
reference for a nominal 5% Wald test, with 1–30 excluded instruments. It requires
unweighted iid covariance and never assigns the threshold to robust KP output.
The numeric reference is [WWC 4.0 Table II.7, reproducing Stock–Yogo Table 5.2](https://ies.ed.gov/ncee/wwc/Docs/referenceresources/wwc_standards_handbook_v4.pdf).

## PANIC selection and trend calibration

```python
panic = oe.xtpanic(df, "output", "country", "year", factors="icp2",
                  max_factors=5, lags="bic", maxlag=3, trend="trend",
                  inference="bridge", pooling="independent")
```

Fixed factors/manual lags and `inference="none"` retain their previous behavior.
`icp1`, `icp2`, `icp3` implement [Bai–Ng 2002 equation (9)](https://www.ssc.wisc.edu/~bhansen/718/BaiNg2002.pdf)
on the PANIC common differenced sample, allowing zero factors. Candidate criteria,
chosen count and lack of per-unit standardization are recorded. AIC/BIC/sequential
t lag selection uses a common maximum-lag comparison sample for each component,
then refits that component using its chosen lag. Selection does not estimate
finite-sample uncertainty about the number of pervasive factors.

The explicit `bridge` option calibrates [Bai–Ng 2004 Theorem 3.1(1)](https://users.ssc.wisc.edu/~behansen/718/BaiNg2004.pdf)
for trend idiosyncratic components. It simulates the Karhunen–Loève Brownian-bridge
integral with 65,536 draws, 512 modes and local seed 271828; omitted modes use
their exact tail expectation. Metadata distinguishes Monte Carlo CDF error,
truncation error and the asymptotic limit from finite-sample calibration.
The maximum binomial CDF standard error is approximately 0.00196. Calibration
does not modify global Torch random state and reserves 4 MiB within `memory_mb`.
Trend pooling is available only with bridge calibration and the explicit
independence assumption. The default trend inference still reports idiosyncratic
statistics without p-values. Common MQ tests retain published quantiles only.

## RD and manipulation

```python
rd = oe.rdrobust(data=df, y="outcome", running="score", fuzzy="received",
                 covariates=["baseline"], weights="observation_weight",
                 masspoints="adjust", bwcheck=10)
kink = oe.rdrobust(data=df, y="outcome", running="score", p=2, deriv=1)
density = oe.rddensity(data=df, running="score", h=[0.4, 0.5])
```

RD uses a common weighted polynomial-residualized covariate slope on both sides;
singular adjustment covariates are rejected. Nonnegative observation weights
multiply kernel weights; zero-weight rows follow the shared sample contract.
`deriv<=p` targets a derivative jump; 1 is a kink, and fuzzy kink uses the ratio
of derivative jumps. `masspoints="adjust"` uses unique pilot counts and the
preliminary minimum unique-value bandwidth, `check` records duplicates without
adjustment, and `off` disables adjustment. Equidistant nearest-neighbor matches
use the reference's relative √machine-epsilon tolerance.

Covariates, weights and derivatives require resident DataFrames. Dataset replay
rejects these options explicitly and supports the existing unadjusted RD plus
mass-point selection. `rddensity` is a distinct manipulation diagnostic on the
running variable: explicit bandwidths, unrestricted local polynomials,
jackknife joint left/right covariance, and order q=p+1 inference. Automatic
density bandwidths, restricted fits and plug-in density VCE remain outside the
validated domain. These are compared with the official
[rdrobust](https://github.com/rdpackages/rdrobust) and
[rddensity](https://github.com/rdpackages/rddensity) implementations, including
bandwidths, local N, estimates and standard errors.

## Evidence boundary

The committed checks validate scientific calculations, missing/calendar/rank
guards, topology of AR sets, full covariance, JSON persistence and LaTeX output.
Project-source tests execute actual owned pip/uv subprocesses, a local source
build, cancellation, active-generation readback and inert manifest reopening.
Pinned Git transport and custom-index policy also have deterministic tests.
Live QA built PyPA sampleproject at commit
`621e4974ca25ce531773def586ba3ed8e736b3fc`, resolved its peppercorn dependency,
installed a SHA-256 verified HTTPS wheel, and resolved an explicit HTTPS index
with uv. These checks do not assert access to a private authenticated index. Native desktop,
Windows/CUDA, cloud transfer, signing and notarization are separate acceptance
layers and are not implied by these source tests.
