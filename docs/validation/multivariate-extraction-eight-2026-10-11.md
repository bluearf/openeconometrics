# Eight remaining factor-analysis scopes

Acceptance contract recorded before public integration and validation, current main `ba011676` (11 October 2026).
Linear MARKET-824–831 under MARKET-185; older completed methods are not counted.

1. GLS extraction minimizes the explicitly declared concentrated uniqueness-scaled criterion `0.5 sum_tail (gamma-1)^2`; fixed factor count, interior/Heywood diagnostics and local minimum certification. This is not fixed-R-inverse weighted residual GLS.
2. Generalized image extraction uses conditional variances `D=diag(1/diag(R^-1))`, eigenpairs of `D^-1/2 R D^-1/2`, and loadings `D^1/2 V (gamma-1)/sqrt(gamma)`. It is distinct from existing image-covariance PCA.
3. Full oblique target rotation minimizes all-cell half residual SS on unit-factor-variance oblique geometry. Finite full-rank target, fixed caller axes/signs and locally full-rank tangent are required. No global optimum/uniqueness claim.
4–8. Fixed one-factor unrotated ML, minres, alpha, image-covariance and generalized-image estimator-functional IID bootstrap. Refit complete raw rows, recalculate sample moments and retain a fixed sign anchor; all requested draws must pass. These are separate newly supported method domains, not five copies of PF acceptance.

The five bootstrap targets are statistical estimator functionals under IID complete-case sampling, fixed dimension/method/count, finite fourth moments, interior nonsingular population correlation and locally regular extraction/sign geometry. Population regularity is assumed, not proved by finite sample checks. ML/minres require interior strict local minima; alpha requires a regular fixed-point map; image methods require separated retained eigenvalues. No model-selection, rotation, cluster/survey/analytic weights, summary-matrix bootstrap, latent posterior, p-value, inference-df, simultaneous or exact finite-sample coverage claim.

Every planned parameter vector and index draw, full joint B-1 covariance, marginal linear percentile intervals, source alignment, method diagnostics and optimizer-free portable restoration persist. Any failed draw refuses the complete result with full attempted/success/failure denominator. CPU float64 and explicit work/storage bounds; no SciPy/statsmodels fitting in shipped runtime.

Validation: independent primary formulas, NumPy/SciPy development oracles with convergence code0, complete covariance/CI checks, negative geometry/sample/resource cases, source/full typed saved result/replay, fresh frozen backend and real worker/server restart. Native installed UI, Windows and public release are separate and not claimed by this backend change. Focused tests sequential with numerical threads1; fresh required quick gate on current candidate/base before normal merge and tracker readback. Daily full gate preserved; dispatch only if material regression scope requires it under AGENTS.md.

Primary algorithm source: [IBM FACTOR Algorithms, pp.3–6 and8–9](https://public.dhe.ibm.com/software/analytics/spss/support/Stats/Docs/Statistics/Algorithms/14.0/factor.pdf) (generalized image and uniqueness-scaled GLS formulas). The implemented convention is stated explicitly rather than inferred from a method name. Existing ML/PF/minres/alpha/image-covariance and orthogonal-target conventions remain unchanged.

After tests, delete only this batch's generated runtime/build/web-dependency/app/installer files. Preserve this batch's synthetic saved projects/results/evidence, previous artifacts/meta files, primary environment and other worktrees.
