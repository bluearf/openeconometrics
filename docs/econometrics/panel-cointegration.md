# Heterogeneous panel cointegration

`oe.xtcointtest(..., test="pedroni")` and `test="westerlund"` fit independent
unit-specific cointegrating vectors. Both require complete resident DataFrames
with a balanced consecutive clock, at least two units, at least 20 periods,
one to seven regressors, and sufficient rows for the selected augmentation.
Weights, automatic missing-row removal and Dataset collection are unsupported.

`deterministic="none"`, `"constant"` (default), or `"trend"` declares the unit
deterministic terms. `demean=True` first removes period cross-sectional means;
this alone does not establish cross-sectional independence.

Pedroni `ar="panel"` reports group-rho (modified PP), group-t (PP) and group
ADF. `ar="same"` reports the four within-dimension counterparts including
panel-v. ADF partials both level and difference residuals on lagged differences
before constructing the statistic. Pooling uses the conditional long-run
variance of the difference-regression innovations, following the original
Pedroni algorithm. Individual slope rank failures are errors.

Westerlund is the **2005 variance-ratio test**, with group-mean (`ar="panel"`,
some panels cointegrated) and pooled (`ar="same"` or `all_panels=True`, all
panels cointegrated) alternatives. It is not the 2007 ECM test. It uses full
level residual cumulative sums and does not use ADF lags or a HAC kernel;
those common API settings only apply to Pedroni/Kao.

Pedroni normalizes using published Pedroni (1999) Table 2 and (2004) Corollary 1
numerical Brownian moments (100,000 draws, T=1000). Westerlund uses the separate
Hlouskova–Wagner (2009) Tables 20–25 T=500 approximation of large-T Brownian
moments (100,000 draws). These are numerical large-sample approximations;
small-T nuisance dependence is not eliminated. Panel-v rejects in the upper
tail; other tests reject in the lower tail. Normal p-values and 5% standardized
critical values are returned alongside raw statistics and moment provenance.
The method and finite-sample conventions are not identical to current Stata;
no full-precision Stata parity is claimed.

Pedroni's `kernel_lags=<int>` declares a fixed HAC truncation. With no truncation,
the default selects unit Bartlett bandwidths by the Newey–West (1994) plug-in
algorithm. Automatic bandwidths for Parzen/QS are rejected; fixed bandwidths
support all existing kernels. Selected bandwidths are recorded for every unit.
Kao keeps its historical fixed rule default; `bandwidth="auto"` explicitly
selects plug-in bandwidths per unit and averages the matrix HAC contributions.
Kao's original fixed-bandwidth statistics are preserved.

For dependent units, `bootstrap=B, seed=..., block_length=...` additionally
returns common circular-block p-values and raw-statistic critical values.
It integrates centered joint differences under no cointegration, using shared
time block indices for all units and variables, preserving cross-sectional
dependence. Stationarity and adequate block approximation of joint differences
are assumptions. The finite bootstrap is conditional on that declared null,
not a replacement proof of unrestricted finite-sample size. Asymptotic and
bootstrap columns remain separate. A local CPU generator preserves global RNG;
failed or singular replicas fail the requested bootstrap instead of being omitted.

Workspace admission precedes sorting and tensor allocation. `max_work` bounds
declared regression work across all replicas, with at most 5000 replicas stored.
Complete unit labels, sample dimensions, settings and full bootstrap raw
statistics are retained as JSON-safe table attributes.

Validation uses independent NumPy SVD/residual-maker/HAC formulas for both AR
domains and all deterministic cases, independent common-block bootstrap nulls,
an independent Brownian simulation checking the Westerlund moment approximation,
and Stata's public fictitious 100-unit replication data with frozen independent
raw-statistic references. The source data are public numerical examples;
no licensed Stata executable was used.

Sources: [Pedroni 1999](https://web.williams.edu/Economics/wp/pedronicriticalvalues.pdf),
[Pedroni 2004](https://web.williams.edu/Economics/wp/pedronipc-rev-f.pdf),
[Westerlund 2005](https://doi.org/10.1080/07474930500243019),
[Hlouskova and Wagner 2009](https://irihs.ihs.ac.at/id/eprint/1945/1/es-244.pdf),
[Stata methods and public examples](https://www.stata.com/manuals/xtxtcointtest.pdf).
