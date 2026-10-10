# Fixed-K Gaussian regression mixtures — MARKET-682

`finite_mixture(data, y, x, components=2, sigma_min=..., ...)` fits a native
CPU float64 conditional Gaussian regression mixture. Declare `sigma_min` in
the outcome's units: it defines a **constrained likelihood**, rather than a
numerical variance repair. Components have separate regression coefficients
and scales and constant positive prior membership weights. This stage admits
complete real numeric independent rows and a fixed explicit K. It does not
estimate the number of classes or infer latent causal groups.

The native EM uses normalized log-sum-exp responsibilities, exact weighted QR
regressions, weights `N_k/n` and scale variance `RSS_k/N_k`. This denominator
is Gaussian maximum likelihood, not the residual-df OLS variance. Every
declared start, local seed, initialization, EM parameter iterate, failure and
Newton endpoint is retained. The highest converged constrained objective wins;
neither multiple starts nor monotone iterations prove a global maximum. An
active-bound winning solution remains the winner and has unavailable ordinary
interior inference. An inferior regular solution is never substituted.

The joint observed information uses Louis's complete-minus-missing identity in
the free chart `(all beta, all log sigma, K-1 mixing log-odds)`. It includes
cross-component and coefficient/scale/membership blocks. Full information is
checked after equilibration in its coordinate units; positive normalized
eigenvalue ratio greater than `max(128*q*eps,1e-10)` and a resolved joint score
are required for regular inference. This is an explicit numerical admission
boundary, not a universal statistical threshold. Very small weights (`<=1e-8`),
binding scale bounds or weak information make inference unavailable. Rank
deficient component designs and zero posterior masses fail explicitly.

The physical covariance contains **all K weights** and is correctly singular
along the simplex sum constraint. No ridge is added. Coefficient SE/z/p/CI use
an asymptotic normal reference with `df=None`, conditional on the stated K and
local labelled solution. Scale CIs transform log scales; probability CIs use
one-versus-rest log-odds. No weight-zero or scale-zero Wald p-value is reported:
those are boundary targets. K comparison does not use an ordinary chi-square
likelihood-ratio calibration.

`finite_mixture_restore(result_or_json)` validates the bounded versioned source
and state, exact source dtypes/index/positions, all EM updates, all numerical
endpoints, full information/Jacobian/covariance and every cached row target.
It does not train or launch the optimizer. Generated initializations are
recreated using a private seeded generator; explicit physical starts are
reproduced exactly. Intermediate Newton value histories and failure messages
are bounded optimizer provenance rather than regenerated optimizer paths.
Recorded objective endpoints and projected score are numerically checked.
Display labels use a recorded post-fit lexicographic parameter permutation;
all chart, covariance and membership arrays follow those labels. Mixture
density and moments are invariant under input label permutation.

`finite_mixture_predict(result, data, target=..., ...)` supports mean, variance,
component mean, prior membership, posterior membership, log density, density,
CDF and quantile. Posterior requires an explicit observed-y column. Density
and CDF require an explicit outcome/threshold argument. Unconditional targets
refuse an outcome argument, so a future y cannot enter them. Prior membership
is constant in this stage and distinct from the posterior conditioned on y.
New predictors retain the declared saved order and have no refitted transform.

Regular fits return the complete cross-query delta covariance `J V J'` and
target CIs. Quantile derivatives use the implicit CDF identity and a bounded
96-step solve. Target parameter uncertainty is separate from the conditional
mixture's aleatory variance; its fixed-parameter CDF/quantiles are not a
Bayesian distribution integrating unknown parameters. Nonregular fits permit
explicit `parameter_uncertainty=False` fixed-parameter predictions. Probability
CIs transform log-odds and variance CIs transform logs where resolved. Density
can underflow to zero; log density preserves the likelihood coordinate.

The result is a real `TableSet`, with parameter, full covariance, original-row
membership and complete start tables. Its `to_json()` contains the complete
typed source, sample, equations, free/physical parameters, iterates, uncertainty
and numerical cache. JSON and LaTeX exports have separate purposes; display
formatting never truncates the saved model.

## Resource and data admission

Resident pandas frames and bounded one-dimensional column mappings are admitted.
Dataset collection, iterators, GPU, case/survey weights and missing-row deletion
require separate stages. Nonexact integer conversion to float64 is refused.
The outer caps are 20,000 rows, 33 design columns, K<=4, 160 free parameters,
32 starts, 1,000 EM iterations and 200 Newton iterations. They are ceilings;
the conservative worst-case full fit/replay plan can refuse a smaller request.
Plans include every start/iteration, 40 line trials per Newton iteration,
chunked Louis score/information, QR, complete source/trajectories and serialized
results **before** numeric scans, frame copying or tensor construction. Default
work is 5 billion, maximum 200 billion scalar-operation units. These are
conservative algorithmic work estimates, not seconds or a process-RSS limit.
Workspace uses the shared configurable 512 MiB default and complete state has
a 64 MiB envelope; prediction has at most512 full joint target cells. Restore
re-admits the same geometry and state. Torch default dtype/device and global
RNG are preserved.

## Independent references and evidence boundaries

[De Veaux (1989)](https://www.sciencedirect.com/science/article/pii/0167947389900431)
provides the regression-mixture likelihood/EM target.
[Louis (1982)](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1982.tb01203.x)
provides the observed-information identity.
[Benaglia, Chauveau, Hunter and Young (2009)](https://www.jstatsoft.org/article/view/v032i06)
publish the original mixtools0.4.3 package and complete Section5 CO2 example.
The publisher's package archive SHA256 is
`f360e316100ee7e87e892e692f85a85771b38b332db0f7ede271cf4ecf0c22ec`;
its DESCRIPTION declares GPL>=2. The original all28-row CO2 RData SHA256 is
`6676876fff5b5bd0072c7b4c75aa9fbb389c44127d017d2284b6a66393f14a7e`.
Original-author GPL code/data are external development references and are not
copied into the Apache-licensed production package.

The unmodified original `regmixEM` returns its final pre-M-step responsibility
array alongside post-M-step parameters. At the paper's default epsilon=1e-8,
the maximum discrepancy from terminal probabilities is about1.1128e-5 on all
28 CO2 rows. The oracle retains and labels that reported array and separately
recomputes terminal probabilities using the reported author parameters; it
does not widen a tolerance to hide this convention. Matching tighter endpoint
fits and full numerical derivatives are additional independent checks.

Source tests, independent NumPy/finite-difference checks, actual native R author
output, frozen package and installed desktop persistence are separate gates.
This child completes this Gaussian fixed-K stage; MARKET-364 still requires
Poisson/binary mixtures, covariate membership, polytomous latent classes and
validated nonregular class comparison. MARKET-363 supervised algorithms are
separate and remain open. No blanket Stata/SPSS parity is implied.
