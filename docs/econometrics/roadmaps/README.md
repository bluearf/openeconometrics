# Research method stages

These are implementation proposals and acceptance gates, audited against
`40add89` on 7 October 2026. They preserve earlier deferred/out-of-scope decisions.
Closing the named **planning** records does not implement their future estimators.
The original GitHub implementation issues remain open.

| Planning record | Deliverable | Implemented here | Open numerical stages |
| --- | --- | --- | --- |
| [MARKET-144](https://linear.app/bluearf/issue/MARKET-144) | [Survey](survey.md) | [MARKET-202](https://linear.app/bluearf/issue/MARKET-202): single-stage declaration only | MARKET-210, MARKET-211, MARKET-212 |
| [MARKET-186](https://linear.app/bluearf/issue/MARKET-186) | [Specialized quantiles](specialized-quantiles.md) | None | MARKET-213, MARKET-214, MARKET-215 |
| [MARKET-188](https://linear.app/bluearf/issue/MARKET-188) | [Advanced state space](state-space.md) | None | MARKET-216, MARKET-217, MARKET-218 |

The machine-readable [stage manifest](stages.json) binds every stage to its
record, source paths, evidence requirements and current support decision.
Existing qreg/ARDL/VAR/UCM/general proper-prior sspace and completed method work
are foundations to reuse, not new support claims. Open PR91's advertised partial-observation
claim is unverified and its eight-line difference is documentation only. No GPU, licensed-vendor, public
release or installed-native claim is introduced by this plan.

## Shared promotion gates

Before an implementation stage can become a supported capability:

1. Define the estimand, identifying assumptions, equation/parameter ordering,
   supported options, sample/missing/category and weight semantics. Keep original
   physical rows/dates and identify each intentional exclusion.
2. Publish CPU float64 Torch kernels and complete covariance/SE, distribution,
   df, p/CI, objective, convergence and diagnostics. An unidentified target has
   no fabricated inference. Missing a requested output fails explicitly.
3. Independently reproduce an analytical or original-author numerical fixture
   with sample/options and provenance/license/hash recorded. Small positive
   examples, invariances and unsupported/adversarial geometries all matter.
4. Integrate the appropriate registry, ModelSpec/ResultBundle or typed procedure
   result, prediction/inference and publication LaTeX. Persist full specifications,
   sample/data hashes, uncertainty and support exclusions; read back without a
   refit. Update generated capability/editor help and a runnable example.
5. Declare resident/Dataset, device, workspace, operation and iteration domains.
   First CPU/resident support does not authorize materializing arbitrary Datasets
   or claiming CUDA. Reject work-budget exhaustion and incompatible options;
   no silent fallback, PSD projection, data truncation or failed-replicate removal.
6. Record source/oracle validation, licensed vendor comparisons, frozen runtime,
   installed desktop/display and restart persistence separately. Vendor binaries,
   author data and restricted fixtures require access/license review before use.
   `stata_parity_validated` remains false.

Priority is an implementation recommendation: survey's descriptive Taylor kernel
first; IVQR and QARDL after independent reference conventions; proper-prior
missing-state reconciliation before diffuse/factor work. DSGE and research
quantile extensions retain their earlier scope decisions. Dates are not delivery
promises. Child stages remain Backlog until their own evidence gates pass.

## Baseline audit

The existing registry, `analysis_contracts.py`, quantile kernels, ARDL/UCM and
`tsworkflows/sspace.py` were read at the pinned source. The historical
`reports/OpenEcon ekonometri tam ekleme planı.md` is traceability, not current
implementation evidence. PR91 head `830b9c3` was fetched and its sspace diff
inspected: eight added lines are documentation only. Its open
status and absence of a merge commit were read back. Extend the shared filter
in place after defining sample preparation, all-missing periods, smoothing and
inference; neither compared source implements the advertised missing-data claim.

Primary references linked in each plan were opened on 7 October 2026. Reading a
paper/manual is not numerical reproduction. No new author/vendor numerical
fixture has been executed for the planned estimators in this change.
The PR91 description advertises partial-observation support, but the actual
`sspace.py` difference is **eight docstring lines only**. Both compared Kalman
implementations require complete finite measurement errors. The advertised
candidate is therefore not an implemented missing-data foundation.
