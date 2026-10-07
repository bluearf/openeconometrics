# Full-source dynamic-panel replay

The native `ahreg` replay path implements the Anderson–Hsiao AR(1)
first-difference IV model on all eligible `Dataset` rows. It preserves the
[native model and assumptions](ahreg.md), rather than estimating a sampled
subset or treating rows from different panels as independent.

```python
import openecon as oe

result = oe.ahreg(
    data=oe.scan("panel.parquet"), y="output", x=["capital"],
    panel="firm", time="year", instrument="levels", covariance="cluster",
)
print(result.to_latex())
```

Owned SQLite tables sort projected complete observations by panel identity
and exact signed int64 period. SQL lag windows generate only consecutive
three-period windows (`levels`) or four-period windows (`differences`).
Time gaps stay gaps; dates and fractional/bool periods are refused. Period
identity beyond `2**53` is preserved without a float64 round trip. Missing
rows are removed before the lag windows under `missing="drop"`; a missing
interior row therefore creates a genuine gap. Duplicate panel periods are
an error. This ordering is read in bounded numerical blocks, so even one
long panel does not need to fit in RAM.

The exogenous regressors are `D.x`, the endogenous regressor is `D.L.y`,
and the excluded instrument is `L2.y` or `D.L2.y`. A global TSQR of
`[D.x, instrument, D.L.y, D.y]` preserves the complete crossproducts;
the existing native IV QR kernels operate on that small factor. Full
ordered replays compute structural residuals, panel score sums and the
panel-clustered first-stage excluded-instrument F test. `robust` retains
the native convention of panel-clustered CR1 uncertainty. `nonrobust`
retains the induced MA(1) error covariance: diagonal2 and −1 between
consecutive equations, with `RSS/[2(N-K)]` scaling. Cross-block adjacent
equations contribute to that covariance, while gaps/panel boundaries do
not.

Every coefficient, covariance, uncentered differenced-equation R²,
RMSE, group size and first-stage statistic uses all eligible equations.
The model is exactly identified; its empty overidentification test is
reported honestly. First-stage relevance is not weak-instrument-robust
AR/CLR inference. User weights and categorical regressors remain outside
the existing native `ahreg` contract.

The saved result contains at most 400 fitted values for display, exact
effective equation counts/physical-position digest, original source
fingerprint and lag-window exclusions. The source is verified again
after estimation. `ResultBundle` JSON/LaTeX follow the native contract.
Selection digests are in physical input order; displayed fitted values
are the first sorted complete equations.

Numerical storage scales with the configured row block and small global
factors; panel covariance sums spill to owned SQLite. Disk grows with the
complete projected observations and differenced equations. A snapshotted
buffer budget reserves the source reader, report sample, SQL caches,
factors and live row arrays before estimation. Insufficient scratch
space, RAM budget, consecutive windows, panels, identification or
float64 precision produces an explicit error, and owned temporary
files are cleaned on success and failure. CPU float64 work is not a
universal GPU or timing claim. This entry does not claim that the other
dynamic-panel GMM procedures are automatically covered by AH replay.

Independent NumPy tests construct the consecutive windows and 2SLS
equations, panel sandwich and MA(1) covariance directly. Native parity
covers both instruments, all existing covariance modes, no/exogenous
predictors, unbalanced/gapped data, exact large integer periods,
split numerical blocks and all equations beyond the display limit.
