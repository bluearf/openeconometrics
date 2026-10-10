# Eight-group completion program

The user explicitly authorized implementation of **all remaining scope in all
eight comparison groups** on 2026-10-08. This supersedes the historical research
deferral, but does not change the scientific acceptance criteria. A completed
child stage does not close its parent or its comparison group.

Baseline: `d92e12b4289e59439449e7653e24df11fd06343b`.
Owner: chat `01a11245-1367-79a0-98ed-f048cc3dfbf4`, branch
`codex/eight-gap-completion`.

| Group | Remaining parent scopes | First active stage |
| --- | --- | --- |
| Bayesian | MARKET-362: posterior workflows, native MCMC/GLM, hierarchical/BMA, BVAR/TVP-VAR | MARKET-625: proper conjugate Gaussian; MARKET-680: proper exact linear-null evidence/Bayes factors |
| Latent models | MARKET-360: CFA, latent paths/means, scores/fit, multigroup/MAR/robust, generalized/multilevel outcomes | MARKET-626: continuous CFA; MARKET-646: continuous joint Gaussian latent paths/means and complete inference |
| Endogenous and choice | MARKET-168: generalized/multiple first stages and treatment/ERM; MARKET-164: remaining multinomial probit and broader choice extensions | Preserve streamed-control ownership; nested-choice children MARKET-601–608 are already Done via PR #198 |
| Inference and quantiles | MARKET-153: robust/subvector CLR; MARKET-213: IVQR; MARKET-214: QARDL; MARKET-368: wider wild-bootstrap inversion | MARKET-627: scalar iid IVQR; MARKET-637: QARDL; MARKET-681: full-real-line scalar iid CLR inversion |
| Survival | MARKET-163: Fine-Gray/Gray, covariate interval Cox, frailty/recurrent extensions | MARKET-684: Fine-Gray/Gray with estimated censoring nuisance and joint CIF uncertainty |
| Time series | MARKET-162: seasonal engines/workflows; MARKET-216: exact diffuse; MARKET-217: dynamic factors; MARKET-218: DSGE; MARKET-161: remaining volatility families | MARKET-628: fixed-system exact diffuse; MARKET-647: known-finite-prior conditional AR(1) dynamic factors |
| Survey and missing data | MARKET-211: complex survey; MARKET-145: remaining MI inference, compatibility and sensitivity | Fully stratified third-stage survey is merged via PR #203; preserve its delivery and prior MI cores/extensions |
| Supervised and mixture models | MARKET-363: supervised algorithms and validation; MARKET-364: finite mixtures and latent classes | MARKET-682: fixed-K Gaussian regression mixtures; MARKET-683: train-only partitions and weighted CART with complete pruning |

## Acceptance for every stage

1. Implement the declared algorithm, sample rules, uncertainty and failure domains.
2. Verify against independent numerical or primary-author references, including
   full covariance/likelihood and adversarial cases rather than coefficients alone.
3. Validate versioned saved state semantically without refitting. Preserve source
   identity, typed indexes, parameter order, assumptions and budget records.
4. Verify relevant frozen-package behavior and a dedicated installed macOS
   application's full Run, Quit, process exit and reopen lifecycle.
5. Merge the exact tested head after a fresh current-base gate, then read back the
   child issue's actual Linear status. Keep parent and remaining stages open.

Some existing parents require licensed Stata/EViews evidence. Vendor installation
or reference-output availability was requested; independent source verification
can proceed while that answer is pending. Papers and SciPy comparisons are not
licensed vendor execution. Public release, signing and Windows release evidence
remain separate from source and dedicated macOS QA application evidence.

## Ownership and status

As of 2026-10-09, MARKET-625–628 have renewed source, frozen-package and actual
native acceptance after the Bayesian moment, subnormal replay, admission and
macOS layout repairs. The repaired evidence (internal evidence excluded from this public snapshot)
preserves both historical failures and the later actual Run, full Quit/process
exit and visible cold reopen. The current-main bridge is scoped explicitly;
exact final current-head hosted/local gates, merge and Linear completion remain
separate.

MARKET-646, MARKET-647 and MARKET-680–683 have complete public/editor integration
and actual frozen native acceptance (internal evidence excluded from this public snapshot)
at `e13836da`: 674 checks per Python, six independent actual-state oracles,
eight UI runs, all 151 fitting/restoration tables, and unchanged full state,
sources and history after complete process exit and visible cold reopen.
Final merge gates, merge and Linear completion remain pending. MARKET-637 still
fails three of eleven strict original-author targets; those failures and the
unchanged tolerance remain visible. MARKET-684's multi-group censoring covariance
has a documented different author nuisance target; its author comparison is
still failed, while the independent common-score derivative check passes.

The next separately owned scopes are MARKET-685 (native Gaussian/logit/Poisson
HMC), MARKET-686 (finite bootstrap random forests), MARKET-687 (Gaussian
NARFCS), MARKET-688 (ETS MNN), MARKET-689 (covariate interval Weibull) and
MARKET-690 (proper finite Gaussian BMA). Their individual source/calibration,
integration, native and merge boundaries must be verified separately. None of
these stages closes its parent or comparison group.

Existing streamed-control, fully-stratified-survey and Windows-release worktrees
are preserved. Completed nested-choice, missing-data and survey stages are not
rebuilt or counted again. The research phase manifest retains the historical
deferral together with the newer authorization for all remaining scope.
