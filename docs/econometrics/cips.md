# Cross-sectionally augmented panel unit-root tests

`oe.xtcips` implements native float64 Torch CADF and CIPS/CIPS* statistics.
The same procedures are available as `oe.xtunitroot(..., test="cips")` and
`oe.xtunitroot(..., test="cadf")`; the latter returns individual statistics.

```python
import openecon as oe

cips = oe.xtcips(data=df, y="income", panel="country", time="year", lags=2)
cadf = oe.xtcips(df, "income", "country", "year", lags="bic", maxlag=3,
                 individual=True)
display(cips)
display(cadf)
latex = cips.to_latex(notes=cips.attrs["notes"])
```

The augmented regression follows [Pesaran (2007), equation 54](https://doi.org/10.1002/jae.951):
individual lagged level, lagged cross-section mean, current and `p` lagged mean
differences, and `p` lagged individual differences. `trend="none"`, `"constant"`
and `"trend"` control unit deterministic terms. The OLS t ratio on the
individual lagged level is CADF; its unit average is CIPS. CIPS* first clips
the individual ratios at the published deterministic-case truncation bounds.
`truncated=True` selects CIPS* as the primary statistic; both averages appear
in the summary table. The alternative is stationarity for a nonzero fraction
of panels, rather than proof that every individual series is stationary.
The no-intercept finite-T reference assumes zero initial levels. No per-unit
centering is applied without an intercept; a nonzero initial cross-section
mean may affect finite-T similarity. Intercept/trend regressions absorb unit
initial-level shifts without changing their regression space.

Complete, balanced common-date panels are required. Integer dates must be
consecutive, including exact large signed/unsigned integer dates. Datetimes
must share an inferable regular calendar frequency; month starts and business
days are supported. Duplicate keys, missing observations, a zero common mean,
rank-deficient regressions and unresolved exact fits produce explicit errors.
Rows are sorted by exact identifiers/dates; input data is not changed.

Integer `lags` applies to every unit. Automatic AIC/BIC/HQIC compares lag orders
0..`maxlag` on the common holdback sample, then refits each unit at its selected
order. Fixed lags avoid lag-selection uncertainty. No unit or regressor is
silently dropped when a candidate design is not identified.

Tables I(a-c) and II(a-c) provide the individual and averaged left-tail
1%, 5%, 10% quantiles. Separate parenthesized cells are retained for truncated
statistics, including sparse parenthesized rows. The numerical tables were
transcribed from the author's [Cambridge working paper](https://doi.org/10.17863/CAM.5082)
and checked against the corresponding published tables. Rows index time `T`,
columns cross-section `N`; off-grid quantiles use documented bilinear
interpolation. `T` means potential differences, so a dataset with 51 levels
uses the T=50 row. Lag augmentation uses the paper's same asymptotic reference;
finite-sample automatic-lag selection is not separately calibrated.

There is no extrapolation outside 10..200 in either N or T. For longer/larger
panels, `inference="none"` calculates the statistics and leaves quantiles and
decisions absent. Exact p-values are never inferred from three quantiles or
replaced by a standard normal/t distribution. Decisions and interpolated
critical values remain approximate, subject to the paper's factor assumptions.

This is the one-factor cross-section-mean augmentation. It does not implement
PANIC or an arbitrary multifactor test. Batched, scaled Householder QR uses
bounded fitting workspace: `block_size`<=512 and `memory_mb`<=1024. The reported
estimate excludes in-memory input/index/output arrays, the common mean series,
allocator overhead and private BLAS workspaces. CPU float64 is the tested
device/precision; general GPU or out-of-core support is not established.

Tests compare every deterministic/lag design and automatic lag choice with
independent NumPy least-squares calculations, verify published table cells
and sparse parentheses, exact date identities, serialization/LaTeX, workspace
limits and unchanged source data. `stata_parity_validated=False` remains until
direct Stata comparisons are run.
