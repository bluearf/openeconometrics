# Exhaustive proper Gaussian Bayesian model averaging

MARKET-690 is a bounded part of MARKET-362. This family averages a **declared finite set of all optional subsets**. Public lazy exports, capability metadata and offline editor help expose the fit, prediction, independent draws and typed replay APIs. Installed application execution and merge/tracker acceptance remain separate gates. It does not complete the Bayesian parent.

## Model and sample contract

`bayes_bma(data=..., y=..., optional=..., forced=..., priors=..., model_prior_odds=...)` always includes an intercept and all forced predictors. At most eight optional and eight forced predictors are supported. Integer mask `m` includes optional coordinate `j` exactly when `m & (1 << j)`. Supply exactly `2**len(optional)` proper `NormalInverseGammaPrior` objects and positive finite model odds in mask order. Within each mask, prior coordinates are `Intercept`, forced names, then present optional names in their declared order. Every prior covariance scale must be positive definite; shape and inverse-gamma scale must be strictly positive, with shape at most `1e12`.

For model m,

\[
 y\mid\beta_m,s,m\sim N(X_m\beta_m,sI),\quad
 \beta_m\mid s,m\sim N(\mu_m,sV_m),\quad
 s\mid m\sim IG(a_m,b_m).
\]

All models use the same physical complete-case sample over the union of outcome, forced and optional columns. `missing='raise'` refuses missing union cells; `missing='drop'` records all original rows and exact complete positions. No model receives extra observations merely because it omits a predictor. Resident numeric DataFrames retain numeric dtypes, original index identity, names, MultiIndex unused levels/sortorder and string storage/missing representation. Boolean, complex, nonrepresentable integer/float values and unsupported extended precision are refused before target algebra.

Proper priors make rank-deficient likelihood designs legitimate. No likelihood rank test or silent column deletion changes this prior-defined model. Model coordinates and proper priors must be transformed together for a within-model basis change. Arbitrary mixing of optional columns defines a different collection of subspaces and prior model odds; no basis parity across that changed model space is claimed.

## Exact weights, atoms and uncertainty

Each posterior/evidence uses the existing proper NIG core. Model posterior weights are normalized in log space from explicit prior model odds and exact marginal likelihood. Underflow of any positive normalized prior or posterior mass is a numerical refusal; it is never silently pruned. Inclusion probabilities and the full joint covariance of inclusion indicators come from the complete finite model posterior.

Coefficients omitted by a model are embedded as **exact zeros**, creating a genuine atom at zero in an optional coefficient's marginal posterior. Equal-tail limits use the generalized inverse of the complete Student-t/point-atom mixture CDF. They can both equal zero inside a jump. They are neither averages of component limits nor intervals for a fictitious single Student-t law. Variance limits invert the complete inverse-gamma mixture CDF. The quantile solver brackets by component quantiles and performs bounded bisection; an unresolved/nonrepresentable target is refused. Every returned continuous endpoint is checked against its actual CDF target; a large location cannot silently turn a narrow continuous interval into a point.

When moments exist,

\[
 \bar\mu=\sum_m w_m\mu_m,\quad
 C_{\beta}=\sum_m w_m C_m+
 \sum_m w_m(\mu_m-\bar\mu)(\mu_m-\bar\mu)' .
\]

The complete `[embedded coefficients, sigma_squared]` joint covariance includes between-model coefficient–variance blocks and all original-unit off-diagonal elements. Conditional within-model coefficient–variance covariance is zero when its absolute product moment exists; this does not eliminate the between-model block.

Moment availability is decided separately for each coefficient from **every positive-weight model in which it is present**, regardless of how small that weight is. With `n` common observations, the stable increments determining existence are:

| Quantity | Required positive increment |
|---|---|
| Included coefficient mean | `a_m + (n-1)/2` |
| Included coefficient second moment; sigma squared mean | `a_m + (n-2)/2` |
| Coefficient–sigma squared absolute product moment | `a_m + (n-3)/2` |
| Sigma squared second moment | `a_m + (n-4)/2` |

An omitted coordinate is a point mass with all finite moments. Unavailable analytic moments are `None` with explicit per-coordinate availability; independent sample covariance is not substituted. Full matrices are supplied only when every required marginal/product moment exists. A finite positive moment that cannot be represented in float64 is a numerical refusal, distinct from mathematical nonexistence. Variance-parameter standard deviation is evaluated before its final square, preserving representable small moments near the shape boundary. Student-t scale and finite covariance remain distinct fields.

## Joint prediction and independent draws

`bayes_bma_predict` retains complete query identity and positions. Conditional means share one model and coefficient vector across rows. Their full cross-row covariance includes within-model coefficient uncertainty and between-model means. New independent outcomes add `E[sigma_squared]` on the covariance diagonal. Mean/outcome intervals invert their actual marginal finite Student-t mixtures. Full parameter–query covariance includes coefficient–variance/model-selection effects when the required absolute moments exist.

`bayes_bma_draws(draws=..., seed=..., data=...)` returns bounded **independent** model/coefficient/variance draws, with optional joint query means/new outcomes. Each joint draw shares its model, coefficients and variance across all query rows. Locally seeded CPU Torch generators preserve the caller's global RNG, dtype and device defaults. A nonfinite seeded draw refuses the result; draws are never discarded. These are conjugate independent draws, not MCMC chains; no R-hat or ESS is manufactured.

## Portable state and capacity

`BMAResult`, `BMAPrediction` and `BMADraws` are immutable typed states. Public readers and complete or selectively excluded serializers semantically replay all saved equations, model priors/odds, exact sample/index identity, weights, interval targets, cross-covariance and seeded draws. A rehashed changed cache is refused. Restoration does not call a new fit or select a new subset.

Before allocation or expensive parent replay, guards validate complete schemas, float/type/shape primitives, resident source/query precision, seeds, descriptor geometry, all-model factors, full query matrices and full output. JSON decoding, deep copy and indentation each have pre-allocation workspace checks. Shallow Pydantic copy intentionally retains ordinary unchecked update semantics; every subsequent public reader/export refuses forged content. The encoded envelope is 32 MiB. Resident metadata and simultaneous copy/decoder/output buffers are charged separately against the current workspace and declared `max_bytes`.

Capacity limits are implementation resource contracts: at most 10,000 source/query rows, 256 models, 10,000 draws, and 2,000,000 planned complete array values. The default work bound is 100,000,000 conservative units and declared workspace bound 256 MiB, both additionally limited by the current global workspace. Query matrices and output can exhaust capacity long before the row ceiling. The exact expanded metadata geometry of recursive labels/names and all unused index levels/categories is streamed with O(depth) temporary storage before descriptor construction. Shared tuple objects are counted per portable occurrence; compact RangeIndex descriptors remain compact.

## Verification and outstanding gates

The source tests cover an independent observation-space NIG law, full model/inclusion/parameter/query covariance, exact atom quantiles, high-shape 100-digit evidence, stable heavy-tail moments, sample/index/dtype binding, strict structural-zero forgery, ambient RNG/defaults and measured parser/copy/indent admission. The standalone oracle reads actual exported state using NumPy/SciPy observation covariance and complete finite-mixture laws without importing a production estimator or replay.

`bayesian-bma-calibration-plan.json` prospectively fixes all 256 replications, priors, model odds, physical design, seeds, six randomized PIT targets, three inclusion residuals, failure denominator and wrong-law controls. Overall family-error upper bound .05 is allocated .025 to the six-target DKW family and .025 to the three-target bounded Hoeffding family. Initial pre-run draft and the prospective allocation correction are retained; no thresholds/targets/seeds change after results. Calibration is not a claim about all priors, sample sizes or model spaces.

The production model uses proper per-model priors, distinct from many packages' default g-prior/Occam-window/MC3 implementations. Original-author BMA package equivalence is not claimed. Installed native acceptance has its own source-pinned execution and persistence gate. Hierarchical hyperpriors, shrinkage mixtures, nonlinear/non-Gaussian BMA, BVAR, MC3/search and general Bayesian predictive scoring remain separate parent gates. Existing HMC and exact conditional hypothesis stages remain unchanged.

## Primary references

- Hoeting, Madigan, Raftery and Volinsky (1999), corrected *Bayesian Model Averaging: A Tutorial*, equations 1–3 and posterior variance decomposition: <https://sites.stat.washington.edu/www/research/online/hoeting1999.pdf>. The author page identifies the corrected paper: <https://sites.stat.washington.edu/raftery/Research/bma.html>.
- Proper conjugate Gaussian linear-model derivation: <https://www.ucm.es/data/cont/media/www/pag-105960/TheCLRM.pdf>.
- Murphy (2007), conjugate Gaussian analysis: <https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf>.

No third-party implementation or licensed external dataset is copied into runtime or tests. NumPy/SciPy/mpmath are development oracle dependencies only; runtime numerical algebra remains CPU Torch float64.
