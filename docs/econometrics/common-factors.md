# Common-factor panel models

`oe.xtcce(data=..., y=..., x=[...], panel=..., time=..., model="ccemg")`
provides six distinct native CPU float64 QR estimators. No external estimator
package is called. ModelSpec, sample positions, full slope covariance and
unit coefficients/covariance survive ResultBundle serialization.

| Model | Unit design and common-factor controls | Inference target |
| --- | --- | --- |
| ccemg | Current x, current cross-sectional y/x means | Mean of heterogeneous unit slopes |
| ccep | Current x, unit-specific loadings on current y/x means | Shared slopes, unit-cluster sandwich |
| amg | Current x and estimated common process from pooled first differences with differenced time dummies | Mean of heterogeneous unit slopes |
| dcce | Lagged y, current/lagged x; current and lagged y/x means | Mean short-run slopes |
| csardl | Dynamic CCE plus unit-specific long-run ratios | Mean of unit ratios; full within-unit delta covariance and between-unit long-run covariance |
| csdl | Current x and current/lagged first differences of x; current y mean and current/lagged x means | Direct mean long-run level slopes |

Unit intercepts are always included; `trend=True` adds unit trends. CSA uses
every complete input row in each period, before trimming any initial lags.
Static models admit observed-row unbalanced panels, including internal gaps
when the global calendar is consecutive. AMG differences only consecutive
within-unit pairs. Dynamic and lag-augmented models require a balanced,
consecutive shared clock. `missing="raise"` is default; `"drop"` records the
complete model-input sample and may render a dynamic panel inadmissible.

Default outcome lags are one for dcce/csardl and zero otherwise; x lags are one
for csardl/csdl and zero otherwise. Dynamic CSA lags default to floor(T^(1/3));
static CSA lags default to zero. `cs_lags` can override this. AMG rejects CSA
lags. Unit slopes must have full rank. Dependent nuisance columns may be
omitted, with their names recorded; no deficient unit is silently omitted.
CS-ARDL rejects unstable unit AR roots and undefined long-run denominators.

Mean-group V is the covariance of unit estimates divided by G; it is normal
asymptotic inference across units, requiring at least three units. CCEP uses a
unit-cluster sandwich with the declared G/(G-1) correction. Unit residual V
accounts for all retained nuisance columns. Common-process/CSA estimation
uncertainty relies on the large-N common-factor asymptotics; it is not a
finite-N joint likelihood covariance. Factor rank/span, sufficiently weak
idiosyncratic dependence and static strict/dynamic weak exogeneity are
assumptions, not consequences of xtcd/xthst diagnostic results.

Resident frames only; weights and categorical terms are unsupported. Workspace
admission occurs before dense designs, and `max_work` limits declared QR work.
Resource rejection does not silently collect a Dataset or change an estimator.

Verification: independent joint-design NumPy SVD checks all coefficients,
off-diagonal V, unit V and nonlinear long-run delta V in factor-driven panels;
the published 48-country Eberhardt macro panel checks CCEMG point/SE/CI and AMG
point/z at printed precision. These checks establish the stated method domains.

References: [Pesaran (2006)](https://doi.org/10.1111/j.1468-0262.2006.00692.x),
[Eberhardt (2012)](https://lezme.github.io/markuseberhardt/xtmgSJ.pdf),
[Chudik and Pesaran (2015)](https://doi.org/10.1016/j.jeconom.2015.03.007),
[Chudik et al. (2016)](https://doi.org/10.1108/S0731-905320160000036013),
and the authors' [xtdcce2 documentation](https://github.com/JanDitzen/xtdcce2).
