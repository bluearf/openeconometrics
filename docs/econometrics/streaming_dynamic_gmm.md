# Full-source dynamic-panel GMM replay

The `xtdpd` replay adapter estimates difference and system GMM from every
eligible observation in a replayable `Dataset`. The `xtabond` and
`xtdpdsys` convenience functions construct the corresponding `xtdpd`
specifications, so they share this numerical path. The model, instrument
semantics and inferential assumptions remain those described in
[dynamic panel-data GMM](dpanel.md).

```python
import openecon as oe

fit = oe.xtdpd(
    data=oe.scan("panel.parquet"), y="output", x=["capital"],
    panel="firm", time="year", system=True, twostep=True,
    covariance="robust", collapse=True,
    gmm=[{"columns": ["output"], "lags": [2, 4], "collapse": True}],
)
print(fit.tests["ar2"])
print(fit.to_latex())
```

## Complete equations and global moments

Owned SQLite tables retain and sort the complete projected input by panel
identity and time. Instruments may use columns absent from `x`; these
columns participate in missing-value selection and source verification.
Under `missing="drop"`, missing inputs are removed before forming lags,
transformations and instruments. Integer periods retain exact signed
int64 identity, including periods above `2**53`; gaps remain gaps and
duplicate panel periods are refused. Fractional and boolean periods are
refused. As in the existing native model, datetime values become
consecutive ranks of the globally observed dates. The result warns that
calendar gaps are not detected under that datetime convention; explicit
regular integer periods preserve calendar gaps when required.

One complete panel at a time supplies the existing native first-difference
or forward-orthogonal-deviation transformation. Global instrument
coordinates are discovered before fitting. Every panel uses those same
coordinates, including zero cells for unavailable lag instruments.
Global TSQR screens transformed regressor and instrument rank and
accumulates the positive factors of `sum(Z_i' H_i Z_i)`. Compensated
complete-panel sums accumulate `A = sum(Z_i' X_i)` and
`b = sum(Z_i' y_i)`; the native float64 Torch GMM kernels solve their
small global problem. There is no estimator delegation and no collected
full-sample `X`, `Z` or panel-by-instrument moment matrix.

One-step residual replays supply every panel moment
`g_i = Z_i' (y_i - X_i beta_1)`. Their TSQR factor supplies the complete
two-step weight matrix. Two-step robust uncertainty includes the
[Windmeijer finite-sample correction](https://www.sciencedirect.com/science/article/pii/S0304407604000387),
computed by another complete-panel pass. System-GMM level means and the
reported intercept/covariance transformation are global, including time
dummies. They are not estimated separately within each numerical block.

## Supported options and diagnostics

Replay covers the existing native specification: one/two-step;
difference/system equations; first differences/orthogonal deviations;
`h=1,2,3`; dynamic order; GMM/IV instrument groups with equation roles,
finite/all lag ranges, collapse and permitted passthrough; time dummies;
constant options; robust/nonrobust covariance and `small` inference.
User weights, categorical predictors, bootstrap and additional cluster
columns remain outside the native `xtdpd` contract.

Arellano–Bond serial-correlation tests, one-step Sargan, two-step Hansen,
instrument-subset difference tests and the joint coefficient Wald test
use complete-source global quantities. Serial-correlation variance
includes the global cross terms rather than treating panels as separate
tests. Native singular-weight warnings, generalized inverses,
underidentification guards and unavailable-test notes are retained.

Difference GMM requires the declared lagged-level moment conditions and
serial-correlation assumptions. System GMM additionally requires the
restrictions that justify lagged differences as instruments for the level
equation. Robust covariance and a Hansen test do not establish those
restrictions. Many instruments can weaken the Hansen and subset tests;
`collapse` and a justified lag cap control width. These limitations and
instrument semantics follow [Roodman's methodological account](https://www.cgdev.org/publication/how-do-xtabond2-introduction-difference-and-system-gmm-stata-working-paper-103)
and the [author's `xtabond2` help](https://github.com/droodman/xtabond2/blob/master/xtabond2.hlp).

Reported observation counts follow the native convention: transformed
equations for difference GMM and level equations for system GMM. The
400 fitted values retained for display are the first sorted reported
equations; coefficients and diagnostics use all equations. Saved
`ResultBundle` JSON/LaTeX include the original source fingerprint,
physical-position digest, lag-only exclusions, instrument/rank counts
and both complete-input and effective-equation row counts. The raw
source is replayed and verified after numerical work; a changed source
is refused.

## Resource boundary

This adapter is bounded in the number of panels, not in each panel's
length. One complete panel's local transformation, instruments and
time-grid workspace must fit the snapshotted numerical budget before
its SQL rows are fetched. The lookup grid spans the global integer-period
range, including gaps. Thus a very long individual panel, widely spaced
period codes or a wide instrument set can be refused even when the total
number of panels is small. An `ahreg` replay has a different row-wise
lag-window implementation and does not require a whole panel in memory.

Numerical storage consists of a source-reader block, one guarded panel,
small global factors/diagnostic accumulators and a fixed SQLite cache.
Global regressor terms are capped at384 and planned instrument columns
at2000 before constructing their labels or tensor matrices. Uncollapsed
instruments and time dummies also require bounded global period metadata.
The configured budget may impose a tighter limit, especially because
GMM factor workspace grows quadratically with instrument width. The
replay numerical ceiling is128MiB even when the wider application budget
is larger. Reported resource plans include the largest checked complete
panel and its simultaneous global factors, source buffers and cache.
This estimates named live buffers rather than total process RSS.

Temporary disk grows with the complete projected input, ordering indexes,
panel metadata and selected-position table. A conservative free-space
guard runs before creating the owned temporary store, and the store is
removed on success or failure. All numerical work in this adapter is CPU
float64. Source-reader batch size does not guarantee that a complete
panel fits, and no universal GPU, timing or arbitrary-row-count promise
follows from replay support.

## Verification

Independent NumPy tests construct balanced difference-GMM equations and
check one/two-step estimates, robust covariance, the Windmeijer correction,
Sargan/Hansen statistics and the global AR-test variance. Native comparisons
cover both equation systems, both transformations, all three `h` choices,
robust/nonrobust covariance, small inference, time dummies, custom
instrument roles, unbalanced/missing/gapped data, global collinearity,
datetime ranks and exact large integer periods. Other checks exercise
JSON/LaTeX, full results beyond the display limit, external-estimator
import refusal, caller-device restoration, source mutation, resource
refusal and temporary-file cleanup.
