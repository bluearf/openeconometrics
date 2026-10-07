# Panel residual diagnostics

`oe.xtserial` tests serial correlation in idiosyncratic panel errors.
`oe.xttest3` tests whether fixed-effects residual variances differ across units.
Both procedures return finite JSON dictionaries with the statistic, reference
distribution, degrees of freedom, p-value, sample counts and provenance.

```python
import openecon as oe

serial = oe.xtserial(data=df, y="wage", x=["education", "experience"],
                     panel="person", time="year")

fe = oe.xtreg(data=df, y="wage", x=["experience"],
             panel="person", time="year", model="fe")
heteroskedasticity = oe.xttest3(fe, data=df)
```

## Wooldridge–Drukker serial correlation test

The procedure follows [Drukker (2003), section 2](https://doi.org/10.1177/1536867X0300300206)
and the publisher's [xtserial reference procedure](https://www.stata-journal.com/software/sj3-2/st0039/xtserial.ado).
It fits `Δy = ΔXβ + e` without a constant, then fits `e_it = ρ e_i,t−1 + u_it`
without a constant. Under the null of no first-order correlation in the level
idiosyncratic error, `ρ = −0.5`. The test is `(ρ̂ + 0.5)² / Var(ρ̂)`, with
panel-cluster CR1 covariance and `F(1, G−1)` inference, where `G` counts panels
contributing to the second regression.

Inputs: `data`, `y`, predictor sequence `x`, `panel`, `time`; optional
`categorical`, `missing="drop"`, and `time_delta`. The differenced regression
omits time-invariant or collinear terms and records them in `omitted_terms`.
At least two panels must provide three consecutive complete observations.
The first regression uses every available consecutive difference, including
panels that cannot contribute a lagged residual to the second regression.

Numeric time uses exact integer periods and defaults to increment 1.
Use `time_delta=2` for periods spaced by two integer units. Datetimes require
an explicit fixed duration such as `time_delta="1D"`; calendar months should
be represented as integer monthly periods. Gaps stay gaps. Missing data are
dropped before constructing differences; they never join distant periods.
Choose `missing="raise"` to reject incomplete model inputs.

The result includes `correlation`, `std_error`, `null_correlation=-0.5`,
`nobs_fd`, second-regression `nobs`, `n_groups`, `n_groups_input`,
`design_rank`, original positional `sample_positions`, `time_delta`, and warnings.
Zero residual variation or a singular cluster variance raises an explicit error.

## Modified Wald groupwise heteroskedasticity test

The method follows [Baum's xttest3 description](https://ideas.repec.org/c/boc/bocode/s414801.html)
and the maintained [method source](http://fmwww.bc.edu/repec/bocode/x/xttest3.ado),
version 1.0.8 dated 4 October 2024. The equations are implemented independently
in Torch; the source procedure is not bundled.

For each unit, `s_i² = mean_t(e_it²)` and
`V_i = sum_t((e_it² − s_i²)²) / (T_i (T_i−1))`.
The common variance is the population variance of all idiosyncratic FE residuals,
and the statistic is `sum_i((s_i² − s²)² / V_i)`, with asymptotic `chi²(G)` inference.
Every denominator uses that unit's actual `T_i`, including unbalanced panels.

`oe.xttest3(result, data=df)` requires an unweighted `xtreg(model="fe")` result
and exactly its original input data. It verifies the data hash, estimation
positions and coefficient estimates, then rebuilds idiosyncratic residuals by
the within transformation. A persisted `ResultBundle` can be used directly.
Robust/cluster covariance on the original FE fit does not change the residual test.

Every panel must contain at least three estimation observations and nondegenerate
squared residuals. Undefined variance terms raise errors instead of being silently
omitted. This differs deliberately from the reference procedure's absolute
small-variance filter and preserves invariance to outcome units. `xtgls` and
weighted FE results are outside this procedure's current contract.

The chi-square calibration is asymptotic. Baum's [help](http://fmwww.bc.edu/repec/bocode/x/xttest3.hlp)
reports low power in panels with many units and few periods; a nonsignificant
result in that setting is weak evidence of homoskedasticity.

## Computation and validation

Numerical work uses float64 Torch, QR least squares and vectorized group sums.
No observation loop or observation-by-observation covariance matrix is formed.
Residuals are normalized internally to avoid overflow/underflow in fourth moments;
the statistic is unchanged by this normalization. pandas handles columns,
labels, missingness and sorting only.
The procedures currently use in-memory model tables; they do not provide an
out-of-core or distributed panel execution path.

The existing panel Pearson-correlation kernel normalizes each input and its
centered values separately, including positive weights, before computing
within/between/overall fit metrics. This avoids tiny-unit underflow, large-unit
overflow and rejecting a correlation solely because the two variables have
different units. Constant effective samples still return no correlation.

Independent tests compare `xtserial` with two statsmodels regressions and CR1
cluster covariance, and `xttest3` with NumPy dummy-variable regression and direct
group formulas. Coverage includes balanced/unbalanced panels, gaps, missingness,
categorical/collinear terms, fixed-duration datetime grids, large integer time
indices, JSON serialization, saved-result validation and outcome-unit changes.
Numerical regression tests independently verify the correlation kernel under
separate input scales from `1e-300` through `1e300` and weighted sample changes.
`provenance["stata_parity_validated"]` remains `False`: these are independent
algorithm checks, not proof from an actual Stata execution.
