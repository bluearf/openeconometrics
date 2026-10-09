# Deferred analytical families: scope and acceptance plans

These deliver the **planning tasks** MARKET-146, 155, 156 and 169. Their
implementation work remains open in GitHub #32, #41, #42 and #56 and under
[MARKET-73](https://linear.app/bluearf/issue/MARKET-73).
Completing a planning issue does not add an estimator or change the earlier
`Hayır` / `Sonra` product decisions. Each phase below is **deferred** until
the product explicitly activates it; no implementation date is promised.

Source baseline: `e5db5d0806428c8776f9a77e4416d720ed8e0220` (7 October 2026).
The old [model plan](<../../reports/OpenEcon ekonometri tam ekleme planı.md>)
is historical intent; current registry/export source determines availability.
Draft PR #91 was inspected and does not supply these four general families.

| Planning issue | Decision and phases | Open implementation tracker |
| --- | --- | --- |
| [MARKET-146](https://linear.app/bluearf/issue/MARKET-146) | [Continuous CFA, latent SEM and subsequent extensions](sem.md) | [#32](https://github.com/bluearf/openecon/issues/32) |
| [MARKET-155](https://linear.app/bluearf/issue/MARKET-155) | [Conjugate posterior contract before samplers and BVAR](bayesian.md) | [#41](https://github.com/bluearf/openecon/issues/41) |
| [MARKET-156](https://linear.app/bluearf/issue/MARKET-156) | [Saved prediction and split isolation before model families](supervised.md) | [#42](https://github.com/bluearf/openecon/issues/42) |
| [MARKET-169](https://linear.app/bluearf/issue/MARKET-169) | [Mixture likelihood, regularity and latent classes](mixtures.md) | [#56](https://github.com/bluearf/openecon/issues/56) |

## Shared implementation gates

Every future phase must record the exact outcome/target, rows retained and
their original positions, missing policy, category maps, supported weights,
parameter order and every covariance entry. Unsupported options fail explicitly.
Predictive intervals, posterior credible intervals and frequentist confidence
intervals retain different labels and schema semantics.

Numerical runtime remains native CPU float64 Torch with local random generators.
Independent NumPy/R/vendor software is a development oracle only. Current
registered estimators, Dataset routes, user projects, identifiers and auth are
preserved. New families start resident-only: no inherited Dataset replay,
streaming or GPU claim. Admit the complete data/model/state footprint before
allocation and before starting iterative work; validate positive memory/work
limits against the global workspace budget. Cancellation must retain earlier
saved results and report an incomplete fit, never truncate the sample or starts.

A versioned saved-state adapter precedes a public registry entry. It must round
trip fitted state, category/transform metadata, full uncertainty and predictions
without refitting. Numerical references compare full precision under the same
likelihood, sample and options, with tolerances declared before inspection.
Simulation protocols predeclare seeds, grids, failure denominators and decision
rules; a failed calibration stays in the evidence and blocks that inference.

Source tests, independent references, source-built frozen execution, installed
native Run/export/restart and any cloud/public distribution are separate gates.
Licensed Stata/SPSS/EViews execution is an additional named reference gate,
never inferred from documentation or another package's agreement. None of these
future gates has been run for the planned methods in this delivery.

The machine-readable [phase register](phases.json) keeps every deferred phase
and its implementation tracker. The planning audit (internal evidence excluded from this public snapshot)
checks present-source names, historical decisions, local references and phase
coverage. It is a document/source audit, not scientific validation of future code.
