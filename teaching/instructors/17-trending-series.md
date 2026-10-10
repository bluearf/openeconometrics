# Instructor notes · Lab 17: The Trap of Trending Series

The [student handout](../labs/17-trending-series/index.md) uses two independent
random walks with drift. The [source](../labs/17-trending-series/lab.py) reads
the prepared workbook, fits levels and differences, and runs native ADF and KPSS tests.
The lesson is about a valid numerical calculation with an invalid usual
inferential interpretation, not a software malfunction.

## Teaching approach

Before disclosing independence, show the level regression's $R^2$ and t
statistic and ask what evidence would be needed to interpret the relationship.
Then disclose the generator and have students identify why accumulation
creates persistent level patterns. Avoid turning the exercise into a rule
that every trending variable should always be differenced.

Require each test interpretation to begin by naming its null. ADF and KPSS
reject in different directions and use different reference distributions.
Emphasize that stationarity around a level, stationarity around a deterministic
trend, and a random walk with drift are distinct specifications.

## Worked exercise answers

1. A random walk expands to $X_t=X_0+dt+\sum_{j=1}^t\varepsilon_j$.
   Independent shocks give accumulated variance $t\sigma^2$, and a one-time
   shock remains permanently in later levels. In a stationary AR(1), its
   contribution decays as $\rho^h$ and unconditional variance is
   $\sigma^2/(1-\rho^2)$ under stationary initialization.

2. The level regression estimates slope **0.9544288568**, conventional
   SE **0.0170441143**, t **55.9976**, and $R^2$ **0.9294548731**.
   These describe the sample line and requested formula correctly. The usual
   stationary-regression significance argument does not apply to these
   independent integrated levels. A p-value near $4.97\times10^{-139}$ does
   not supply an economic mechanism or identify causality.

3. Differencing gives $\Delta X_t=0.35+\varepsilon_{xt}$ and
   $\Delta Y_t=0.25+\varepsilon_{yt}$. Independent innovations imply a
   population change slope of zero. The actual estimate is
   **−0.0154836828**, with an intercept **0.315952** rather than exactly
   0.25 because the realized sample mean differs from its expectation.

4. One additional X index-point increment is associated with approximately
   **−0.0155 Y index-point increment**. HC3 SE is **0.0674128301**;
   interval **[−0.1482885767, 0.1173212111]**; p **0.818535**.
   The model uses **239 changes**, originally quarters 2–240. Its slope
   answers a different question from the level slope.

5. ADF has a unit-root null; sufficiently negative values reject. For levels
   with a constant, trend, and one lagged difference, statistic
   **−2.7499390993** is above the 5% cutoff **−3.4290999472**, so do not
   reject at 5%; approximate p **0.215938**. For changes with a constant and
   one lag, statistic **−10.7925916539** is below cutoff **−2.8738137461**;
   approximate p **$2.11\times10^{-19}$**. The DF reference is nonstandard;
   an ordinary Student t critical value would give the wrong test.

6. KPSS reverses the null. Level statistic **0.2434047637** exceeds the
   trend-stationarity 1% value **0.216**, so the table notes p below 0.01.
   Change statistic **0.1214658835** is below the level-stationarity 10%
   value **0.347**, so the notes state p above 0.10. Stored values 0.01 and
   0.10 are interpolation bounds, not exact tail probabilities.

7. Two integrated series linked by an economic equilibrium may have a
   stationary linear combination. Differencing both and ignoring that
   relationship can lose information about long-run adjustment. A suitable
   extension asks whether a theoretically justified cointegrating relation
   exists and how deviations from it affect later changes. Independent
   random walks here do not have a constructed shared equilibrium.

**Prepared input:** [trending_series.xlsx](../labs/17-trending-series/trending_series.xlsx). Use the stored observations for the published analysis. The [original generator](generators/17-trending-series.py) supports new dataset editions; update results and answers when changing observations.

## Reproduction and verification evidence

The [full reference state](../labs/17-trending-series/reference.json) retains
the native level and difference models plus complete ADF/KPSS tables and
their notes. Explicit exports use `run_lab(output_dir="trends-output")`.
The stored calendar starts at 1 for levels and 2 for differences.

An independent centered-slope calculation matches the level estimate within
$4\times10^{-16}$. `manual_adf` forms the ADF(1) auxiliary design with lagged
level, lagged change, trend, and constant, computes its classical t ratio,
and matches the native statistic within $4\times10^{-13}$. It does not
substitute a Student t p-value for the DF probability. Checks also verify
240 level rows, 239 differences, 238 level ADF rows, and 237 difference ADF
rows. Full fitted models retain covariance and exact sample positions.

Fresh source execution and Ruff passed. Native `ConsoleSession` execution
passed without error, with output order **model, model, table, table, table,
table, plot**. The result global is `trends_lab_result`. Default execution
writes no files. These checks establish calculation and source-runtime
behavior; they do not make the deliberately spurious level interpretation
valid or assert acceptance in a separate installed release.
