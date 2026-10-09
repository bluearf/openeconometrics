# Eight method-level closure contracts

Predeclared integration scope, 7 October 2026, source baseline `374d2b7`.
This delivery targets #35, #39, #40, #46, #52, #61, #62 and #68. Existing
validated methods are reused; public routes below fill the remaining core gaps.
Independent numerical checks, licensed vendor execution and installed desktop
evidence remain distinct. No global parity flag changes.

| Issue | New bounded contract | Existing methods/evidence to recheck |
| --- | --- | --- |
| #35 | Fisher–Johansen aggregation from supplied, explicitly sourced rank-test p-values, independent units, no invented p-value interpolation | PO calibration, Pedroni/Westerlund/common-block bootstrap, FMOLS/DOLS/CCR and panel estimators |
| #39 | Restored IV specification plus exactly matched original resident data; structural AR/CLR tests and scalar iid AR inversion | Weak/strong IV, full reduced-form covariance, confidence-set topology and independent fixtures |
| #40 | Restricted wild-cluster linear tests, OLS/explicit absorbed/panel FE, CR1 studentization, Rademacher/Mammen/Webb, bounded complete Rademacher enumeration and scalar finite-grid inversion | Independent restricted-refit draws, unequal clusters, controlled null-size diagnostic |
| #46 | No new estimator: recheck finite-filter conditional CSS/GPH/state/forecast domains and installed receipts | Fractional weights, d=0 limits, likelihood gradients, full covariance and finite-model forecasting |
| #52 | Key-aligned cross-sectional SAR 2SLS using supplied excluded instruments, iid/HC0 covariance and full delta multiplier impacts | ML SAR/SEM/SDM/SAC, sparse weights, Moran and restored prediction |
| #61 | No new estimator: recheck independent-study effect preparation, pooling/regression/diagnostics/plots | Published BCG fixture, heterogeneity/inference/CI/PI and source/frozen/installed receipts |
| #62 | Normal paired/cluster mean design, fixed-effect one-way ANOVA/regression F power, proportional-hazards logrank event planning; inversion/rounding and scenario output | Mean/proportion/correlation power and CI-width planning |
| #68 | Whole-panel pairs bootstrap for MM-QR; explicit balanced-even-calendar split-panel bias correction, with bootstrap recomputing all three fits | Published nlswork fixture, full location/scale/quantile covariance, equivariance and saved quantile means |

New numerical APIs use native float64 CPU Torch, unweighted complete numeric
resident domains with declared resource limits. Dataset/GPU collection, automatic
critical-value extrapolation, generic weak-ID cutoff claims and silently omitted
failed draws are outside these contracts. Staged spatial-panel, IV/multiway wild,
multilevel meta, sequential design and specialized break/Fourier cointegration
work remains visible separately. The balanced split-panel correction assumes
time-homogeneous parameters and a leading 1/T bias expansion; it is not a generic
short-panel correction or a new FE quantile objective.

## Structural IV state and tests

`iv_saved_weak_test(result, data=..., null=..., method="ar")` verifies the
original model columns, values and row order, then reconstructs the retained
sample and matrices without refitting the IV coefficients. AR supports iid,
HC0 (`robust`), one-way cluster and HAC structural nulls. Moreira CLR remains
scalar iid homoskedastic. A complete endogenous null is required; untested
endogenous nuisance coefficients are not profiled by an invented subvector
procedure. The saved fit covariance is the default; changing it is explicit.

`iv_saved_ar_confidence_set` uses the existing analytic iid scalar inversion.
It retains empty, disjoint and unbounded sets and the boundary polynomial.
There is no robust/cluster/HAC/CLR analytic interval substitute.

## Restricted wild-cluster inference

`wild_cluster_test` fits the constrained linear null and multiplies its
residuals by independent cluster weights. Each draw refits the unrestricted
linear design and recomputes CR1 studentization. The full dummy rank is used
in `G/(G-1)*(N-1)/(N-K_full)`, including for absorbed/panel FE. FE groups must
nest within the one independent bootstrap cluster; weighted, categorical,
disconnected/collinear, nonlinear, IV and multiway designs fail explicitly.
Only reported slopes can be restricted in an FE model.

Rademacher, Mammen and Webb weights use local seeded float64 generators.
Monte Carlo p-values use `(exceedances+1)/(B+1)`. Complete Rademacher
enumeration with G<=12 uses `exceedances/2^G`; this is complete *conditional
bootstrap* enumeration, not an exact population/assignment test. A relative
1e-12 equality guard is reported. Singular studentization aborts; no failed
draws are silently discarded. N<=8192, K_full<=256, 49..100000 sampled draws,
2 million multiplier cells and explicit work/workspace admission apply.

`wild_cluster_confidence_set` inverts the same scalar restricted test over
2..201 prespecified increasing values with shared multipliers. It returns
accepted grid components and neighboring rejected boundary brackets. The
maximum gap is saved. No interpolation, convexification or unbounded-tail
claim is made outside the tested grid. See [#117](https://github.com/bluearf/openecon/issues/117)
for continuous inversion and IV/multiway/nonlinear stages.

The controlled 200-trial G=8 unequal-cluster Gaussian-intercept experiment
enumerates 256 draws per trial and records 5 rejections at alpha=.05:
rate .025, Wilson 95% [.0107,.0572]. This is one fixed-design diagnostic,
consistent with conservatism here, not a universal finite-G calibration.
[MacKinnon, Nielsen and Webb (2022)](https://doi.org/10.1016/j.jeconom.2022.04.001)
is the primary restricted-wild reference.

## Cointegration calibration boundary

`fisher_johansen` combines supplied unit rank-test p-values as `-2 sum(log p)`
with chi-square(2N). Every unit must test the same rank/deterministic null;
the caller declares the original calibration and independence explicitly.
Zero/invalid p-values and duplicate units fail; no tails are clipped. It does
not manufacture MacKinnon–Haug–Michelis response-surface probabilities from
sparse critical tables. [EViews' panel cointegration documentation](https://help.eviews.com/content/coint-Panel_Cointegration_Testing.html)
describes the independence requirement and the automatic MHM variant.

Existing `xtcointtest` supports Pedroni and Westerlund **2005 residual
variance-ratio** statistics, with common circular-block null bootstrap for
their declared dependent panel domain. It is not Westerlund 2007 ECM.
Automatic Fisher calibration and the ECM stage remain [#116](https://github.com/bluearf/openecon/issues/116).
Original-paper break/Fourier/threshold tests remain [#115](https://github.com/bluearf/openecon/issues/115).
The validated time-series/panel long-run estimators and PO calibrations are
documented in [panel-cointegration.md](panel-cointegration.md) and
[inference-extensions.md](inference-extensions.md).

## Spatial IV and restored network means

`sar_iv` estimates `y=rho Wy+X beta+epsilon` using explicit excluded instruments
and native full-rank QR 2SLS. It never allocates an N by N instrument projector.
The iid covariance is `SSE/(N-K)*(X'PzX)^-1`; HC0 uses the projected-design
sandwich. Excluded instruments, exogenous controls, outcome and key roles
must be distinct. W and the instruments are assumed exogenous. This is not
weak-identification-robust SAR inference or spatial-error GMM.

N<=max_n (default512, hard2048), K<=32, L<=64 and N>L>=K apply. Dense network
multiplier work is admitted before allocation. Missing-data subgraphs and
re-normalization are explicit. Unconstrained rho outside the nonsingular
stable domain fails; it is never clipped. Effective keyed W/hash, structural
residuals and full covariance survive ResultBundle JSON.

`sar_iv_predict` reads restored parameters plus complete effective network
keys and numeric X, in any order; no outcomes/instruments/refit are needed.
The mean is `(I-rho W)^-1 X beta`. Normal delta mean CIs and multiplier-impact
CIs use full beta/rho covariance, including off-diagonal entries. They exclude
future innovations and W/IV uncertainty. Spatial-panel and further IV/GMM
variants remain [#118](https://github.com/bluearf/openecon/issues/118).

## Panel quantile resampling and split restrictions

`panel_mmqr_bootstrap` samples whole independent individuals with replacement
and relabels repeated copies. Each draw recomputes location, centered-sign
scale, error quantiles and all quantile slopes. Coarser-cluster fits, weights,
categorical predictors, changed term support and failed draws fail explicitly.
49..10000 draws and a declared structural work/workspace budget apply.

With `split_panel=True`, balanced common equally spaced even T>=12 is required.
The coefficient-level combination is `2*full-(half1+half2)/2`; its bootstrap
recomputes all three fits jointly. The user must justify homogeneous time
parameters and a leading 1/T coefficient-bias expansion. There is no generic
fixed-T coverage guarantee. This linear split-panel extrapolation is distinct
from the author's [component-wise gamma/quantile jackknife example](https://jmcss.som.surrey.ac.uk/MM-QR-JK.do);
the author example is not claimed as an identical correction oracle.
[Dhaene and Jochmans' split-panel principle](https://doi.org/10.1093/restud/rdv007)
does not automatically establish its assumptions for every MM-QR dataset.

The 100-trial-per-calendar Gaussian location-scale bias diagnostic covers
T=12/24/48. Two T=12 trials fail the positive-scale domain in a half panel;
all attempted/valid/failed counts are retained, and reported bias means are
conditional on admitted fits. The correction does not improve every quantile
in every cell. This evidence supports the explicit short-T restriction and
conditional nature of the extrapolation, not an automatic improvement claim.

The saved result carries joint cross-quantile bootstrap covariance, every
draw, original keys and all component states. `panel_mmqr_resampled_predict`
combines saved known-individual conditional means with the same weights.
Corrected curves can cross; no monotonicity is manufactured. No individual
prediction SE is derived from slope-only covariance. Existing published-author
nlswork, joint dense-dummy moment, equivariance and quantile-order fixtures
remain the base MM-QR evidence; source tests are not licensed xtqreg execution.

## Prospective planning and persistence

Paired normal means use known difference SD derived from the prespecified
two SDs and correlation. Fixed-size independent cluster means use
`DE=1+(m-1)ICC`; the normal known-covariance approximation is explicit.
One-way balanced ANOVA and fixed-design regression use noncentral F laws,
with native Poisson mixtures and verified integer/effect inversion. The
published Stata regression fixture `(R2_full-R2_reduced)/(1-R2_full)=.125`,
five slopes/two tested restrictions, alpha=.05/power=.8 yields N=81.
The calculation uses prespecified fixed-design signal, not observed power.

`power_logrank` adds Schoenfeld proportional-hazards normal event planning.
Its shift is `log(HR)*sqrt(D*a*(1-a))`; events are integer and `a` is the
prespecified constant group-2 risk-set information fraction, not enrollment
allocation. An optional event fraction converts to expected enrollment
`N=ceil(D/fraction)`. It does not infer censoring/accrual distributions.
`planning_scenarios` and `planning_plot` preserve all chosen assumptions and
return editable curves. Known-SD CI-width planning remains `precision_mean`.
Sequential looks/stopping boundaries remain [#120](https://github.com/bluearf/openecon/issues/120).

All new summaries support `oe.summary_state(output)` and
`oe.restore_summary(json_string)` with complete tables, covariance, draw
metadata and component states. Save the JSON to an owned file and read it
back. Console tables are previews (50 rows, 500 characters per cell); large
settings identify their full-state export, rather than claiming preview
persistence. The runnable [integration example](../examples/eight_capability_closures.py)
exercises file readback and restored predictions.

## Reused completed cores and evidence layers

#46 uses the existing [fractional-memory contract](fractional-memory.md):
finite-filter zero-prehistory conditional CSS, not exact stationary ML;
GPH, d=0 ARMA checks, full-information covariance and model-consistent saved
forecast innovation covariance. Source/frozen/installed receipts remain pinned
under `docs/evidence/fractional-memory-2026-10-07` to their recorded revision.

#61 uses the independent-study [meta-analysis contract](../meta-analysis.md):
MD/SMD/OR/RR/Fisher-z preparation, common/DL/ML/REML, HK variants, regression,
CI/PI and diagnostics/plots. The published BCG fixture and 3200-run coverage
receipt are bounded evidence. Dependent-effect multilevel/multivariate work
remains [#119](https://github.com/bluearf/openecon/issues/119).

The new integration receipt is under
`docs/evidence/eight-capability-closures-2026-10-07`. It distinguishes source
oracles, imports-blocked representative execution and installed-wheel execution
from installed desktop/UI evidence. This delivery does not certify a new
desktop release, Windows/CUDA behavior or global vendor parity.

Integration note: ANOVA/regression/ICC procedures reuse the independently merged
PR #114 implementation (`power_designs.py`); there is one authoritative public
route per method. Paired-known-SD and logrank reuse the independently merged
PR #121 implementation (`planning_extended.py`), alongside its two-proportion,
two-correlation, slope, McNemar and additional precision designs. This delivery
adds isolated scenario/plot helpers in `planning_scenarios.py`. The F-law
N<=20000, df<=40000 and noncentrality<=4096 budgets remain in force.
