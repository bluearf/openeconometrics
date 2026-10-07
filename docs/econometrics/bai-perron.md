# Bai–Perron pure multiple-change regression

`oe.bai_perron(data, y, x, ...)` implements the **pure structural-change**
model of Bai and Perron (2003): all supplied regression coefficients may change
at common unknown dates. It supports mean shifts (`x=[]`), changing intercepts
and changing coefficients on fixed exogenous regressors. Numeric results use
native CPU float64 Torch, without an external estimation engine.

```python
result = oe.bai_perron(df, "y", ["x"], time="period",
                      max_breaks=2, trim=.15, replications=499, seed=2026)
print(result["tests"])
print(result["models"])
```

For each break count `k=0,...,max_breaks`, the method minimizes the sum of
segment OLS residual sums of squares over **every** admissible partition.
The segment length is `h=max(ceil(trim*T), q+2, min_segment or 0)`.
The dynamic programming recurrence is

`SSR(s,t) = min_j [SSR(s-1,j) + segment_SSR(j,t)]`.

Dates are zero-based **first observations of the following regime**, with
matching time values in `models.break_periods`. Singular segment designs are
excluded using a fixed-design rank tolerance; the excluded count is recorded.
If any requested break count has no admissible partition, the call refuses
instead of quietly reducing the maximum count. Missing inputs, duplicate or
gapped time keys and collinear full designs also refuse. Datetimes must have a
regular inferred frequency; integer periods must be consecutive. Sorting
retains original row positions and index labels in `sample`. `settings` stores
all scientific metadata as JSON values in ordinary table cells, so console
persistence does not depend on preserving pandas attributes.

Under **independent Gaussian errors with common variance, independent of fixed
exogenous X**, the classical spherical-error statistic is

`supF(k) = [(SSR(0)-SSR(k))/(k*q)] / [SSR(k)/(T-(k+1)*q)]`.

`UDmax` is the maximum of these statistics over the prespecified break counts.
The method simulates standard normal errors conditional on exactly the supplied
design and repeats the entire segment/date search in each draw. Adding `X*beta`
and multiplying the errors by a nonzero scale leave these statistics unchanged,
so null coefficients and variance need not be estimated for calibration.
The method reports `(1 + number of null statistics >= observed)/(B+1)` and
empirical null quantiles, rather than invented or universal critical values.
It uses a local seeded Torch CPU generator, a fixed batch size of 16, and saves
the seed, draw count, successful/failed counts and denominator. A numerical
failure aborts the call; it cannot silently shrink the draw denominator.

`models` contains optimum SSRs and an explicitly defined BIC:
`T*log(SSR/T) + ((k+1)*q+k)*log(T)`. This includes estimated break dates in the
penalty. BIC selects a model; the calibrated tests address **no break**, rather
than the selected number of breaks. `coefficients` and `covariance` describe
the selected partition. Covariance uses a common pooled error variance and is
conditional on the estimated dates. **It is not selective inference**: there
are no coefficient p-values or confidence intervals, no date confidence
intervals, and no sequential `k` versus `k+1` test.

The `errors` argument accepts only `"iid_gaussian"`. HAC, heteroskedastic,
autoregressive-error and partial-change models are unsupported. An iid
assumption cannot be diagnosed automatically from arbitrary observed data;
the caller must justify it. No general Stata, EViews or full Bai–Perron parity
claim follows from this bounded contract.

The quadratic search is explicitly bounded: at most 1000 rows, eight parameters,
five breaks and 9999 null draws, subject to a more restrictive 256 MiB planned
workspace and 250 million planned work units. These are allocation/work limits,
not measured whole-process RSS limits. Refusal occurs before segment arrays;
large datasets must be deliberately restricted and cannot be automatically
collected from `Dataset`.

The independent oracle in `tests/test_econ_bai_perron.py` enumerates all
partitions and fits each segment with NumPy least squares in original units.
It validates dates, SSRs, statistics, BIC, coefficients, covariance and the
entire null-search calibration. `scripts/validate_bai_perron.py` declares two
T/q cells, null and two-shift alternatives, 200 independent outer samples and
199 inner draws per sample, complete failure denominators, and 1/5/10% levels.
Source, frozen execution and native/persistence acceptance are distinct proofs.

Source: [Bai and Perron (2003), sections 3.2–3.3 and 5.1–5.2](https://doi.org/10.1002/jae.659)
and [the authors' paper](https://www.columbia.edu/~jb3064/papers/2003_Computation_and_analysis_of_multiple_structural_changes.pdf).
