# Eight saved target extensions

These adapters evaluate existing fits from a JSON-restored `ResultBundle` without
refitting. Fit coverage and prediction coverage are separate: this delivery adds
the saved targets below, rather than introducing eight new estimators.

| Saved fit | Interface | Accepted target |
| --- | --- | --- |
| `sreg`, `mmreg` | `oe.predict`, `oe.margins` | Robust fitted linear location, using the recorded S/MM estimator and covariance. This is not a general conditional-mean claim. |
| `ivcue` | `oe.predict`, `oe.margins` | Structural `X beta` at supplied exogenous and endogenous regressor values. Evaluation does not require instruments. Inference assumes strong identification. |
| `mixedflex` | `oe.predict`, `oe.margins` | Unconditional population fixed portion `X beta`, integrating mean-zero Gaussian random effects. Evaluation does not require group identifiers. |
| `sar`, `sac`, `sdm` | `oe.spatial_predict` | Reduced-form mean on the complete effective saved graph. |
| `sem` | `oe.spatial_predict` | `X beta` on that complete saved graph, retaining its jointly estimated coefficient covariance. |

## Common linear interfaces

```python
import openecon as oe
from openecon.models import ResultBundle

saved = ResultBundle.model_validate_json(fitted.model_dump_json())
oe.predict(saved, new_rows, interval="mean", alpha=.05)
oe.predict(saved, new_rows, kind="xb")
oe.predict(saved, new_rows, kind="stdp")
oe.predict(saved, new_rows, kind="derivative", term="income", interval="mean")
oe.margins(saved, "income", data=new_rows, method="ame")
```

Original fitted category coding is reused; new categories are refused. Complete
parameter names and covariance are matched even after a joint permutation.
Missing=`drop` retains the original query positions and duplicate index labels,
with missing predictions at excluded rows. Replayable Dataset inputs produce
complete disk-backed predictions and global AME/MEM using the existing bounded
batch routes. Evaluation weights do not invent a weighted fitted model.

S/MM mean intervals use the saved Student reference law; CUE and mixedflex use
their saved asymptotic normal law. Mixedflex variance-component Jacobian columns
are exactly zero for this population mean, while its full covariance is retained.
Conditional group predictions, BLUP uncertainty and future-observation intervals
are outside these four extensions. Existing finite-start S estimation remains an
approximate nonconvex search; these adapters do not improve its optimum guarantee.

Metadata, coefficient roles, coding and covariance inconsistencies are rejected.
The legacy fitting payloads do not carry signed originals: a coherent arbitrary
replacement of every numerical value and associated metadata cannot be detected.

## Complete graph means

```python
saved = ResultBundle.model_validate_json(fitted_spatial.model_dump_json())
target = oe.spatial_predict(saved, data=all_effective_units,
                            alpha=.05, max_n=512, max_work=250_000_000)
target["means"]
target["jacobian"]
target["mean_covariance"]
state = oe.summary_state(target)      # every row, full matrices and metadata
restored = oe.restore_summary(state)  # no fit or numerical recomputation
```

Every effective saved key must appear exactly once. Arbitrary query ordering and
changed numeric X are allowed, including constant or rank-deficient query designs.
If fitting dropped rows, use the effective induced and renormalized saved graph,
not the original larger graph in the specification. Saved weight hashes and, for
SAC, the separate error graph are validated. Outcomes are unnecessary.

With `A = I - rho W`, SAR/SAC use `solve(A, X beta)` and SDM uses
`solve(A, X beta + W X_slopes theta)`. SEM uses `X beta`. The mean gradient includes
`solve(A, W mean)` for rho and the complete beta/theta derivatives. Lambda and
log innovation variance have zero direct mean gradients. Covariance is the full
`J V J'`, including beta/rho/theta cross blocks and off-diagonal covariance between
query units. Normal confidence intervals describe parameter uncertainty conditional
on this fixed graph and X; they exclude innovation and graph-sampling uncertainty.

The TableSet preserves means, keyed query rows and encoded index labels, the full
parameter Jacobian and covariance, the full mean covariance, and settings. Use
`summary_state` for complete persistence; ordinary console table previews truncate.
The current shared summary schema preserves index labels and omits index names;
store names in explicit metadata when a later reconstruction needs them. The native
examples retain those names and reattach them after restoration.
The bounded resident route permits N<=512 and at most 128 parameters, admits its
workspace before allocation, and checks solve backward residuals. Caller ceilings
may be lower. New nodes, replacement networks, partial graphs, missing predictors,
Dataset streaming, spatial IV fits, conditional realized errors and observation
intervals require other adapters and are refused.

The population mixed target agrees with the definition of `xb` in the
[Stata mixed postestimation manual](https://www.stata.com/manuals/memixedpostestimation.pdf).
The reduced-form spatial target is documented in the
[Stata spatial postestimation manual](https://www.stata.com/manuals15/spspregresspostestimation.pdf).
These are target-definition references; no licensed vendor execution is claimed.
See the frozen scientific protocol (internal evidence excluded from this public snapshot)
and its independent algebra/delta checks for the accepted scientific scope.
