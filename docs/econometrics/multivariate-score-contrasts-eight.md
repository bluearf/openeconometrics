# Simultaneous fixed-query score contrasts: eight acceptance domains

Preregistered 10 October 2026 before implementation. Starting main
`c41b4f7397a70adad5a28b59db6099f4217e4597`; owner chat
`01a11344-6de7-7b92-b2f6-56e95ef62894`, branch
`codex/multivariate-score-contrasts-eight`. Parent MARKET-185 stays open.
Children MARKET-793–800 cover frequency covariance/correlation PCA (2),
IID/frequency paired X/Y CCA (2), and IID/frequency unrotated/full-target
multifactor principal-factor regression scores (4).

Existing training fits and marginal query-score APIs are unchanged. New APIs
`pca_fweight_bootstrap_score_contrasts`, `canon_bootstrap_score_contrasts` and
`factor_bootstrap_score_contrasts` take a complete original training bootstrap,
independently fixed query measurements, and an independently fixed contrast
matrix. The entire saved point and every seeded training refit must replay.
Each draw retains its own centering, scales and accepted scoring coefficients.

## Hypothesis coordinates and inference

Columns of C are **all physical query rows**, then score axes in their saved
order, including incomplete rows. For labelled DataFrames require exactly
`query[0]:Comp1`, `query[0]:Comp2`, … (CCA axes X:Can1,…,Y:Can1,…).
Unlabelled real numeric matrices use precisely that same order. Contrast row
labels are unique nonempty strings. A nonzero coefficient on any missing query
coordinate refuses the hypothesis; zero coefficients permit that missing row.
No deletion of contrast terms or renormalization changes the family.

For complete joint score vector s, point theta=C s and draw theta_b=C s_b.
Save every transformed draw, B-1 joint covariance, SE, bootstrap bias, marginal
linear percentile intervals, and the centered deviations
`z_bj=(theta_bj-theta_j)/SE_j`. The SE is the **one fixed bootstrap SE**, shared
across draws. Let `M_b=max_j(abs(z_bj))` and `c=linear_quantile(M, confidence)`.
Simultaneous basic bands are `theta_j ± c SE_j`. Their family is exactly the
declared rows of C, including signs, cross-query differences and CCA X/Y
combinations. No covariance inverse or rank repair is used: duplicated and
dependent contrasts preserve singular joint covariance if every variance is
strictly positive to working precision. Zero/numerically cancelled variance
refuses the entire family, without a positive variance floor.

These are first-order centered fixed-scale maximum-deviation bootstrap bands.
They are not per-draw studentized bootstrap-t or stepdown adjusted p-values;
p-values and inferential degrees of freedom remain unavailable. Their coverage
requires IID independent training units, finite fourth moments, fixed finite
query/contrast family, regular identified population spectral/loading geometry,
and positive first-order variances. Sign charts, targets, components, queries and
contrast families must not be selected from the training sample. Frequencies
denote literal independent units. The empirical band construction alone does
not prove finite-sample coverage; Monte Carlo quantile error remains. At least
one expected upper-tail rank is required: `(B+1)*(1-confidence)>=1`.

The dependence-preserving maximum bootstrap principle is described by
[Romano and Wolf (2005), sections 3–5](https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffa286d4d1/etca.pdf).
The fixed-scale single-step construction above is the explicit implementation
contract; no claim to implement that paper's complete studentized stepdown
procedure is made. Intervals concern training-estimator score functionals,
not observation noise or individual latent posterior uncertainty.

## Resources, persistence and acceptance

CPU float64, original source limits, at most 128 physical query-score
coordinates and 64 contrast rows. Admit bounded metadata/shape, complete source
replay plus projection work (250 million units), aggregate workspace and the
complete 32 MiB export **before source replay or contrast tensor conversion**.
These are implementation/resource bounds. Reject nonfinite, boolean/complex,
ragged, mislabelled or invalid contrast inputs without executing a refit.

Preserve every source score table (including every original fit table), exact
settings, query identities/missing positions, full physical C and retained C,
all deviations/maxima/critical values and the complete result. JSON roundtrip
must retain table names/order/dtypes/LaTeX. Re-running from the restored original
fit, query and C must reproduce every numerical result. Checksums alone do not
validate numerical training state; forged and resealed fits must fail replay.

Independent NumPy/SciPy complete training refits and literal-frequency expansion
validate all eight Gaussian/skewed domains, every projected draw, joint covariance,
bias, percentiles, centered maximum and bands. Include shifted/scaled data,
signed/scaled/permuted/redundant contrasts, missing/zero-count/typed identities,
nonzero weights on missing rows, zero-variance contrasts, semantic forgeries,
preallocation budgets and exact legacy/catalogue preservation.

Source/frozen/native persistence and current-head/base quick CI are distinct.
Follow current AGENTS.md: focused scientific tests and the fresh required quick
gate; the complete registered SDK gate runs daily, not after every commit.
Verify exact real merge parents/tree and all eight fresh Done readbacks before
claiming closure. After local testing, remove only this batch's temporary app,
installer and generated binary/build files; keep saved projects/results and
committed evidence. Broader parent scope remains open.

## Acceptance receipt

The complete dated evidence pack (internal evidence excluded from this public snapshot)
records302 scientific/preservation cases,1,050 catalogue/integration cases, nine
browser catalogue cases, all eight frozen199-draw domains, independent1,592 full
refits, complete saved-state restart/reconstruction and owned temporary-binary
cleanup. Native installed UI and public release are separate from this change.
