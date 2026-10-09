# Saved VAR identification with an external proxy

`oe.proxy_svar` adds one explicitly normalized external-instrument shock to a
saved reduced-form VAR. The original VAR coefficients, sample and covariance
are preserved; no VAR refitting occurs. The complete output is a `TableSet`
with point responses, impacts, descriptive diagnostics and saved state.

For reduced-form innovations `u_t`, external proxy `z_t` and target impact
column `b`, exclusion of every other structural shock implies
`Gamma = Cov(u_t, z_t)` is proportional to `b`. Normalizing response `r` to
impact `c` yields `b = c * Gamma / Gamma_r` and response `Phi_h b`. This is the
single-shock moment/ratio identification in equations (2.7)–(2.9) of
[Montiel Olea, Stock and Watson (2021)](https://www.princeton.edu/~mwatson/papers/JOE_Publication_SVARIV.pdf).
The same paper explains why weak proxies invalidate ordinary ratio-based Wald
inference. This implementation returns point estimates and descriptive moments
without confidence intervals or a strength cutoff.

```python
import openecon as oe

var_result = oe.var(data=series, y=["output", "inflation", "rate"],
                    time="period", lags=2, maxlag=2)
# External observations contain exactly the VAR's retained estimation periods.
external = proxy_series.loc[
    proxy_series.period.isin(series.iloc[var_result.sample_positions].period)
]
identified = oe.proxy_svar(
    var_result, data=series, proxy_data=external,
    proxy="announcement_surprise", key="period", normalize="rate", impact=.25,
    source="Versioned published announcement surprise series with stated units",
    exogeneity_assumed=True, steps=12,
)
state_json = oe.summary_state(identified)
restored = oe.restore_summary(state_json)
replayed = oe.proxy_svar_irf(restored, steps=20)
```

`exogeneity_assumed=True` records the analyst's exclusion assumption. It does
not establish instrument validity or strength. In addition to exclusion and
relevance, interpretation requires the reduced-form VAR to correctly span
recoverable innovations and a constant impact relationship across the sample.
The point ratio remains sensitive to a small normalization covariance.

The named response has exactly the supplied signed impact at horizon zero.
There is no automatic sign flip or standard-deviation normalization. Multiplying
the proxy by a nonzero scalar, adding a constant or permuting its keyed rows
leaves the normalized responses unchanged, subject to float64 arithmetic. The
proxy hash follows the aligned key/value sequence, so source value changes
remain visible.

The numerical denominator guard compares `abs(Gamma_r)` with `tolerance`
times the product of proxy and anchor residual standard deviations. The default
relative tolerance is `1e-12`, configurable in `1e-14..1e-6`. This guard refuses
numerically undefined ratios; it is not a weak-instrument F criterion. Raw
proxy variance, anchor covariance/correlation and their descriptive R-squared
are reported without p-values, critical values or causal-validity verdicts.

## Admitted domain and saved sample

The initial route admits numeric, unweighted resident CPU float64 `oe.var`
results with 2–12 endogenous variables, 1–12 lags, a constant, no trend or
exogenous terms, and 20–8192 retained estimation rows. The saved VAR must have
an integer time column; source verification admits at most 32768 original rows
including excluded endpoints. Input periods remain exact integers; floats, dates and
strings are not silently coerced into time keys. `Dataset` inputs are refused
before iteration, and no GPU or streaming route is advertised.

Supply the exact original VAR model-input rows and physical order. The saved
model-input hash is checked before reconstructing all equation residuals from
the saved coefficients. Sorting of the time column, initial lag boundaries
and any already admitted endpoint missing exclusions follow the fitted VAR's
saved sample positions. The separate proxy table contains exactly those
retained estimation periods, in any row order. Missing, extra or duplicate
periods and incomplete/nonfinite proxy values raise; they never rescreen or
refit the VAR on a convenient subset.

The helper saves the full autoregressive matrices, residual matrix, proxy
values and means, ordered periods, original sample positions, centered joint
proxy/residual moment matrix, normalization and source fingerprints. Response
propagation follows the saved VAR stability state; unstable results are not
silently stabilized. An arithmetic work bound and up-front workspace plan
refuse large requests without truncation. The memory estimate covers declared
numeric/state buffers, not caller-owned objects or private allocator storage.

`proxy_svar_irf` uses only the versioned saved state. Complete JSON export and
restoration retain all rows beyond console preview limits and require no data
access, identification call or model fitting. Saved state is validated for
finite dimensions, variable order and agreement with its named normalization.

The remaining structural shocks are unidentified. This route returns no full
structural shock system, FEVD, historical decomposition, delta-method ratio
interval, bootstrap or weak-proxy robust confidence set. Those outputs require
their own identification and inference validation; existing SVAR zero/sign
restrictions, generalized connectedness and LP/IV-LP keep their own contracts.

## Independent verification

`tests/test_next_eight_proxy_svar.py` independently reconstructs every equation
residual from saved coefficients with NumPy. Complete sample covariances and
explicit anchor ratios match at `2e-13` absolute tolerance. A separate block
companion-matrix power calculation verifies the noncommuting multi-lag point
responses. The tests include shuffled original data, admitted missing endpoints,
permuted external keys, signed proxy transformations, exact file JSON replay
with fitting prohibited, and malformed sample/hash/normalization/state cases.
Work, workspace, integer-key, finite-arithmetic, Dataset and foreign-default
device refusals are covered. These are native source numerical/persistence
checks; they do not establish licensed vendor equivalence, general statistical
coverage or a new installed desktop release.
