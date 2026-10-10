# Instructor notes · Lab 18: Accounting for Persistent Shocks

The [student handout](../labs/18-serial-correlation/README.md) compares HC1,
Bartlett HAC(4), and Bartlett HAC(8) for one original synthetic advertising
regression. The [source](../labs/18-serial-correlation/lab.py) uses a persistent
regressor and persistent independent demand shocks, with three explicitly
unavailable calendar quarters.

## Teaching approach

Have students distinguish the coefficient-estimation problem from the
covariance-estimation problem before revealing the comparison table. Ask
whether 197 rows necessarily mean 197 independent pieces of information.
Then use the original quarter labels to show why missing observations change
which score pairs exist at each lag.

Do not describe HAC as a generic switch that repairs every time-series model.
Its scope is uncertainty for a specified estimator under appropriate
regularity conditions. Exogeneity, structural change, nonstationarity, and
sample selection remain substantive concerns. The source's Student t(195)
reference is a reporting approximation, not an exact finite-sample theorem
for arbitrary serial dependence.

## Worked exercise answers

1. Under suitable exogeneity, serially correlated errors can leave the OLS
   population moment identifying the coefficient intact while adding
   cross-period terms to its sampling variance. If advertising responds to
   unobserved demand shocks, that moment itself can fail. HAC changes the
   covariance estimate, not the coefficient or assignment mechanism.

2. HC1 permits observation-specific variance but omits serial score
   covariance. HAC includes weighted products of $x_t\hat u_t$ and
   $x_{t+\ell}\hat u_{t+\ell}$. Regressor persistence matters because slope
   variation depends on these score products, not residual variance alone.

3. At four lags, Bartlett weights are **0.8, 0.6, 0.4, 0.2**. At eight,
   they are $8/9,7/9,\ldots,1/9$. The native finite-sample multiplier is
   $197/195$, with two coefficients including the constant. The bread is
   $(X'X)^{-1}$ on both sides of the weighted score meat.

4. Quarters **39 and 42** would incorrectly become lag-one neighbors if the
   missing 40 and 41 were compressed away. Their actual distance is three.
   Similarly, **94 and 96** have distance two rather than one. Errors also
   propagate to other lag definitions after exclusions. Keeping original
   calendar labels ensures pairs are included by actual distance.

5. The main slope is **1.4201321133 thousand sales units per ten thousand
   currency units of advertising**. HAC(4) SE is **0.1975588889**, interval
   **[1.0305056723, 1.8097585543]**, with 197 observations and t(195)
   reference. It describes a quantity association; profit requires prices,
   costs, and an appropriate causal interpretation.

6. HC1 SE is **0.1146086951**, interval **[1.194100, 1.646164]**;
   HAC(8) SE is **0.2246019473**, interval **[0.977171, 1.863093]**.
   All slopes are identical. Positive persistence increases uncertainty in
   this realization, but covariance products can have either sign and weights
   change with bandwidth. There is no universal monotonic ordering.

7. Examples include reverse causality from anticipated demand, a structural
   break in the conditional mean, a spurious regression of integrated series,
   and outcome-dependent missingness. Appropriate responses might require
   assignment-based identification, a break model or subsample argument,
   stationarity/cointegration analysis, or evidence about selection. Requesting
   a different covariance estimator alone does not address those questions.

**Prepared input:** [advertising_and_sales.xlsx](../labs/18-serial-correlation/advertising_and_sales.xlsx). Use the stored observations for the published analysis. The [original generator](generators/18-serial-correlation.py) supports new dataset editions; update results and answers when changing observations. Keep the calendar gaps at quarters 40, 41 and 95. Renumbering the remaining rows would change the time-distance convention.

## Reproduction and verification evidence

The [full reference state](../labs/18-serial-correlation/reference.json) saves
all three models, the residual path, and the comparison data. It records the
excluded quarters **40, 41, 95** and the 200-quarter span. Explicit exports
use `run_lab(output_dir="hac-output")`.

`manual_hac` independently solves OLS, forms the full score sandwich, and
uses a calendar lookup for lag pairs. The complete covariance matrix matches
the native HAC(4) result within $5\times10^{-14}$; the slope discrepancy is
below $5\times10^{-16}$. Other checks confirm identical 197-observation
samples, no hidden row drop, retained calendar gaps, and 195 inference
degrees of freedom. The true generated slope is 1.4, which is not supplied
to any native fit.

Fresh source execution and Ruff passed. The actual `ConsoleSession` worker
completed without error, returning **model, table, plot**. The whole-file
result global is `hac_lab_result`. Default execution writes no exports;
complete JSON and the LaTeX table require an explicit output directory.
This evidence covers the current source runtime and does not establish a
different installer or cloud deployment's behavior.
