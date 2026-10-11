# Meta-analysis packaged backend acceptance: eight remaining scopes

Starting source: `7a195f6231023c025104115b2e3c8ebea411ff51` (fetched main).
The estimator implementation was already merged in PR249. This work adds
reproducible acceptance evidence, not eight estimators. MARKET-629–636 and
their earlier dependent-meta acceptance are preserved and not counted again.

| Child | Remaining acceptance |
| --- | --- |
| MARKET-816 | Published three-level ML: full inference, components and portable result |
| MARKET-817 | Three-level REML with labeled correlated sampling blocks and moderators |
| MARKET-818 | Saved three-level new-study prediction after backend/worker restart |
| MARKET-819 | Three-level CR2, Satterthwaite and prespecified joint HTZ tests |
| MARKET-820 | CR2 for study/effect/common dependent models, including CR1 input |
| MARKET-821 | Trim-and-fill L0/R0: complete observed/filled samples and inference |
| MARKET-822 | PET, PEESE, full WLS inference and conditional selection rule |
| MARKET-823 | Estimated/fixed step-function selection and zero-heterogeneity cases |

Before acceptance, run the existing independent R/dense-reference and negative
fixtures without changing their limits. Build a fresh CPU float64 frozen
backend from this source. Verify every relevant compiled module against the
committed source and require SciPy/statsmodels to be absent from execution.
Run the same complete cases in source and frozen execution and compare every
table cell, axis, dtype, table order, attribute and LaTeX output. Save the full
results outside console previews. Stop the owned backend/workers, reopen its
isolated project and compare saved script, history, events and outputs. Restore
all full results and regenerate supported prediction/CR2 outputs without the
variance optimizer; retain negative-case receipts.

Require the genuine quick gate on the current PR head/base, integration/tree
readback and tracker readback. The daily SDK gate remains a separate scope;
this acceptance does not establish simulation coverage, installed native UI,
Windows acceptance, licensed vendor parity or a public release. Parent
MARKET-174 and GitHub119 remain open for their broader scientific scope.

After local tests, remove only this worktree's new runtime/build outputs and
temporary web dependencies. Preserve the user's installed application, existing
environments, synthetic saved projects/full results and evidence. Use the
ownership-aware runtime shutdown helper in `finally` on every outcome.
