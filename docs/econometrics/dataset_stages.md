# Dataset fit stages after the original 80 routes

At the start of MARKET-519, registry and dispatch were refreshed from main
`9f74901e17f63706a9cd674b753a4aef8c63568c`: 148 registered estimators,
80 Dataset fit routes and 68 gaps. The method-level snapshot in
`../evidence/market-519/route-matrix.json` (internal evidence excluded from this public snapshot)
records each gap's exact source contract for weights, categories, covariance,
roles and options. A family helper or a resident fit does not establish a
Dataset fit route. The original 80-method fixture roster and its historical
receipts keep their original denominator.

The first delivery implements **four** additional Dataset routes:
`bspline_regress`, `rcs_regress`, `fp_regress` and `mfp_regress`. Coverage was
**84 of 148**, with **64 methods still unsupported on Dataset**, at the
first delivery snapshot. After integration of main `337c0465478128848179146df14f0e9598efd6f6`,
the newly registered resident `umidas` route makes current source support
**84 of 149**, with **65 methods still unsupported on Dataset**. The
refreshed method/family/algorithm matrix (internal evidence excluded from this public snapshot)
records the current gap contracts. This is a staged delivery, not an
all-estimator or all-option streaming claim.

## Delivered numerical and sample contract

All four methods use numeric unweighted predictors, an intercept, iid Gaussian
OLS covariance and conditional Student-t inference. Global complete cases are
selected under `missing='raise'/'drop'` across the outcome and every declared
predictor. Estimates, full covariance, RSS, physical sample count, residual df,
p values and intervals refer to the complete retained source, never the bounded
reporting sample. Categories, weights and alternative VCEs remain unsupported
by these estimators, as declared in their resident manifests.

The Dataset route has no resident smoothing 100,000-row ceiling; an explicit
`max_work` guard and preplanned workspace limit its computation. Only projected
source blocks, the bounded TSQR factor tree, small SVD/covariance state and the
first 400 reporting rows are resident. The workspace contract estimates named
live buffers; it is not an RSS limit. Every full pass validates projected values
and retained original row positions. Saved results carry full-source sample and
position hashes; observation-length position/fitted/residual arrays are omitted.

For spline defaults, SQLite indexes order the **entire retained sample** on
disk. Adjacent ranks are linearly interpolated as type-7 quantiles; there is no
sampled quantile, sketch or batch-specific knot. Main/temporary SQLite caches
are included in the memory plan, scratch free space is checked, the temporary
owned database is removed on completion/failure, and its measured disk size is
recorded. Explicit complete knot/boundary declarations avoid this state.

Augmented global TSQR compresses `[basis, outcome]` without using normal
equations for coefficient solving. A small float64 SVD with the full-sample
rank threshold produces coefficients and inverse geometry; a complete-source
residual pass supplies RSS and `RSS/(N-K) * (X'X)^-1`. MFP uses all eight simple
and eight repeated-power terms in one master factor, then solves every one of
the 45 null/FP1/FP2 candidates from that factor. Each candidate retains the
complete-sample objective. Its existing 4/3/2 df closed F tests are approximate
adaptive comparisons. MFP searches one nonlinear variable with fixed linear
adjustments; it is not general multivariable MFP selection. Final intervals and
tests condition on the fixed/selected transformation and model, excluding
selection uncertainty.

## Saved Dataset evaluation

`smoothing_predict(result, data=dataset, interval=True, missing='drop')` replays
the saved basis into an owned, indexed Parquet Dataset. It never refits powers,
knots or boundaries. The `row` column/index identifies original query positions;
missing query rows are dropped and counted. Pointwise mean intervals use the
saved complete covariance and residual df. B-splines still reject extrapolation;
RCS retains its saved linear tails; FP retains the strictly positive scaled
domain.

`smoothing_margins(result, data=dataset, variable=None|'predictor', interval=True)`
reduces the complete fixed query population's basis or analytic first-partial
basis **before** applying covariance (`L V L'`). Optional Dataset averaging
weights are an explicit finite nonnegative **column name**, normalized globally;
whole-row Python weight lists remain a resident-table API. Evaluation weights
do not add weighted fitting support. Zero-weight query rows contribute no mass;
invalid predictor domains are still validated on complete query rows.

These are family helpers. The common `oe.predict`/`oe.margins` adapter catalogue
is unchanged; no implicit common adapter is claimed. Dataset paired contrasts,
standalone row derivative helpers, GAM/local/MARS replay and fit remain explicit
unsupported domains in this stage. Resident APIs keep their existing behavior.

Saved ResultBundle JSON restores the digest-bound basis and full covariance;
restored Dataset prediction/margins and both model/margins LaTeX are verified.
Publication preserves conditional inference and selection-uncertainty notes.

## Remaining family algorithm matrix

These are engineering requirements for future stages, not implemented Dataset
support or performance promises. Method-level sample/weight/category/VCE
contracts are retained separately in the JSON matrix.

| Family (initial gaps) | Exact state or replay required | Main acceptance difficulty / suggested order |
|---|---|---|
| smoothing (8) | Delivered four global transformed TSQR routes. GAM needs global centering/penalty and trace df; local kernels need disk exact neighborhoods/pairs; MARS needs candidate replay. | First delivery prioritizes full conditional inference. Four local/adaptive/penalized routes remain. |
| panel (4) | Disk panel moments/means, global between/within or CCE factors, instrument geometry and actual-row scores. | Reuse exact panel machinery; CRE precedes heterogeneous CCE/HT moment corrections. |
| iv (1) | CUE repeatedly rebuilds exact moment-weight covariance and joint derivatives from original rows. | Global optimization and weak-identification/score inference; cannot substitute 2SLS. |
| teffects (2) | Heterogeneous DID needs complete cohort nuisance/influence state; synthetic control needs complete donor/time geometry and placebos. | Define target/sample and nuisance inference before replay. |
| mixed (5) | Global covariance-component likelihood/profile state and integrated random-effect group scores; disk whole-group state. | Correlated/random-slope covariance, nuisance uncertainty, count/quadrature convergence. |
| spatial (5) | Complete keyed graph, global spatial operators/log determinant or a proved exact sparse method, row alignment and multiplier uncertainty. | Existing guarded dense graph domain does not become out-of-core by replaying rows. |
| regularized (11) | Exact fold-specific global moments/factors for squared loss/PLS; IRLS replay for GLM; disk neighborhoods for kernels; cross-fit nuisance state for orthogonal targets. | High usage: fixed ridge/PLS/CV first, preserving predictive-only inference; kernel/global search and honest orthogonal inference are harder. |
| tsworkflows (3) | Ordered filter/optimizer states, global presample initialization, derivative/information state; disk lagged MIDAS design. | MIDAS factor replay first, then exact filter likelihood and forecast boundaries. |
| mgarch (3) | Ordered multivariate variance/correlation/tangent recursions, joint constrained ML Hessian and nuisance covariance. | Parameter constraints, complete information and approximate versus exact forecast distinctions. |
| robust (3) | Exact disk medians/scales, global multistart/subsample selection, score/Jacobian replay; panel MMQR joint quantile covariance. | An ordinary OLS compression cannot recover residual-dependent robust estimators. |
| longrun (5) | Disk time/panel ordering, exact lag/lead design, long-run kernel corrections and global transformed moments. | DOLS factor route precedes FMOLS/CCR long-run and panel corrections. |
| panel_ardl (3) | Disk complete panel lag/error-correction designs, joint/heterogeneous optimizer and covariance scores. | Distinguish MG, DFE and PMG restrictions, group/sample df and optimizer convergence. |
| structural (4) | Global VAR identification or exact horizon IV factors and cross-horizon joint influence state. | LP factors first; joint covariance, lag/sample boundaries and identified SVAR uncertainty required. |
| decomposition (4) | Complete group/counterfactual moments and joint equations; binary Fairlie needs exact ranks and permutation replay. | Fixed linear Oaxaca geometry before nonlinear mediation/Fairlie uncertainty. |
| fractional (1) | Ordered fractional filters and exact initialization, global likelihood and all parameter derivatives. | Long memory cannot silently become truncated or block-reset filtering. |
| causal (6) | Complete cross-fit nuisance/fold moments, disk tree membership/search and orthogonal influence state; joint transport/policy targets. | Preserve held-out alignment, overlap, nuisance uncertainty and exact search; no resident forest disguised as Dataset support. |

## Verification and sources

`../evidence/market-519/README.md` (internal evidence excluded from this public snapshot) separates
source, frozen and installed-runtime proof. The portable physical verifier
`scripts/validate_dataset_smoothing.py` generates independent development
references, then its `verify` function needs no SciPy/statsmodels and runs inside
the native runtime. Eight cases cover 250,000 physical rows in both CSV and
Parquet, missing rows, full inference, saved Dataset prediction/weighted margins,
JSON envelopes and LaTeX. Frozen/installed acceptance is joined by MARKET-515;
the source test or caller's stage label alone is not installed proof.

Basis/reference definitions: [R Core `bs` fixed knots/boundaries and saved replay](https://stat.ethz.ch/R-manual/R-devel/library/splines/html/bs.html),
[R Core saved spline prediction](https://stat.ethz.ch/R-manual/R-devel/library/splines/html/predict.bs.html),
[Royston and Altman (1994), fractional polynomial regression](https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2986270),
and [Demmel et al., communication-avoiding TSQR](https://doi.org/10.1137/080731992).
Independent SciPy/NumPy references are development tools, never runtime
estimation dependencies. The spline parameterization, bounded scientific domain
and MFP approximate closed-testing semantics are those of the existing resident
implementation; no complete R/Stata parity is inferred from these checks.
