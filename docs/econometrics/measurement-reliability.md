# Measurement reliability: bounded method contracts

These eight descriptive procedures return `TableSet` objects. They are resident
CPU float64 Torch computations, separate from ModelSpec estimators. Existing
Cronbach alpha and exploratory factor analysis remain unchanged. There is no
general IRT, invariance, DIF or vendor-parity claim.

| API | Supported target | Key restrictions |
| --- | --- | --- |
| `polychoric(data, columns, categories=[x_order,y_order])` | Two-step bivariate latent-normal ordinal correlation | Exactly two variables; every declared level observed; interior optimum within `max_correlation` (default .98); no joint ML or cell smoothing |
| `polyserial(data, continuous, ordinal, categories=order)` | Two-step conditional-normal ordinal/continuous correlation | Continuous mean and n−1 SD plus marginal thresholds re-estimated; no joint ML |
| `omega_total(data, items)` | One-factor Gaussian ML model-implied congeneric sum-score reliability | >=3 identified continuous items; existing native ML extraction; interior uniqueness required; no automatic reversal, hierarchical or multifactor omega |
| `icc(data, raters, model="ICC2", average=False)` | Balanced complete Shrout–Fleiss ICC1/2/3, single or average rating | Subjects are rows, raters columns; ICC1 one-way absolute, ICC2 random two-way absolute, ICC3 fixed two-way consistency; negative estimates preserved |
| `cohen_kappa(data, raters, categories=order)` | Two-rater weighted/unweighted agreement | Exactly two raters; separate marginal distributions; declared category order |
| `fleiss_kappa(data, raters, categories=order)` | Exchangeable fixed-count multi-rater unweighted agreement | Complete units; pooled marginals; no weighted Fleiss extension |
| `krippendorff_alpha(data, raters, categories=order, metric="nominal")` | Finite-sample coincidence alpha for nominal, ordinal or interval judgments | Available observations by default; only >=2-judgment units contribute; ordinal distance uses empirical marginal mass; interval categories numeric |
| `gwet_ac(data, raters, categories=order)` | AC1 with identity agreement weights or AC2 with symmetric weights | Complete units; chance term uses the entire declared universe, including absent categories |

## Sample and uncertainty

`missing="drop"` removes an entire incomplete unit for all procedures except
Krippendorff alpha (`missing="available"`). `missing="raise"` rejects any missing
selected value. No procedure silently changes category orders or drops empty
ordinal categories. Metadata preserves input/used/dropped counts, missing cells,
original row positions and row labels. Numeric infinities are rejected.

Default `inference="none"` returns a point estimate and null inference fields.
With `inference="jackknife"`, each subject is removed once, keeping that subject's
raters/items together. All thresholds, continuous normalization and ML nuisance
parameters are re-estimated. Covariance is `(n-1)/n * sum((theta_i-mean(theta_i))²)`;
SE, df=n−1, a two-sided t test of coefficient zero and a symmetric t interval at
`level` are reported. The estimate is the original sample estimate, without
jackknife bias correction. No interval clipping occurs. This is an approximate
independent-subject sampling calculation, distinct from classical ICC F intervals,
fixed-threshold Hessians, Gwet's analytic SE and Krippendorff's pair bootstrap.
It does not establish nominal coverage for small or irregular samples. A failed
replicate or zero replicate variance stops inference, with its original position
and cause; request the point estimate separately when appropriate.

Agreement weights are identity, linear or quadratic **agreement** weights in
category-index distance, or a symmetric q×q matrix in [0,1] with unit diagonal.
These are not observation sampling weights. Arbitrary weights/device requests,
Dataset inputs and undeclared category values fail explicitly.

Omega rescales ML correlation loadings and uniquenesses for raw items, or uses
unit item variances with `standardized=True`. Its denominator is the sum of the
one-factor **implied** covariance, which can differ from empirical scale variance
when the model fits poorly. The loading vector, unique variances, implied matrix
and likelihood discrepancy remain available for inspection. Explicit `reverse`
items are multiplied by −1. One-factor support does not validate dimensionality.
ICC mean squares are reported after centering/dividing by a recorded scale to
avoid scale-dependent degeneracy; coefficient values are in original ratios.

## Resources and persistence

Inputs are limited to 100000 units, 64 columns and 64 categories. Named live
numeric/categorical/jackknife buffers are planned against the caller's workspace
budget before allocation. Caller input, Python result objects, BLAS workspace and
allocator overhead are excluded; this is not an RSS guarantee. Work is bounded
by `max_fits` (default1024, <=10001) and `max_work` (default100000000 conservative
fit/row operation proxy); complete jackknife work may require an explicit larger
budget. No truncation occurs. Optimizer iterations and ordinal tolerances/bounds
are recorded. Native Genz rectangle probabilities are checked for occupied-cell
positivity and total mass, and independently audited with adaptive conditional
normal integration in development tests; no new parallel bivariate filter exists.

`reliability_save(result, path=None)` returns a complete JSON artifact and can save
it to the caller's path. `reliability_load(artifact_or_path)` restores every table,
sample/category convention, diagnostics, covariance and jackknife estimate, with
SHA-256 integrity checks. Checksums detect accidental changes; they are not an
authentication mechanism. LaTeX tables remain available after restoration.

## References and verification boundaries

* [Fox/polycor original implementation](https://github.com/dmurdoch/polycor): two-step ordinal likelihoods and sample-SD polyserial convention. Our uncertainty refits nuisance parameters instead of conditioning on thresholds.
* [Revelle/psych omega documentation](https://www.personality-project.org/r/psych/help/omega.html): common/unique variance decomposition. Our target is explicitly the one-factor implied-covariance ratio.
* [Revelle/psych ICC documentation](https://www.personality-project.org/r/psych/help/ICC.html): Shrout–Fleiss conventions and the six-subject/four-rater published example. Projection-matrix ANOVA independently checks every supported variant.
* [Gwet's original irrCAC source](https://github.com/kgwet/irrCAC/blob/master/R/agree.coeff3.raw.r): AC1/AC2 chance term and declared universe. Distinct-rater pair enumeration independently checks weighted observed agreement.
* [Hayes and Krippendorff (2007) author fixture](https://afhayes.com/public/kalpha.pdf): 11 pairable units, 55 pairs and nominal alpha .7434; numeric coincidence matrices are checked. Our intervals use subject jackknife, separate from the author's bootstrap.
* [statsmodels inter-rater documentation](https://www.statsmodels.org/stable/stats.html#inter-rater-reliability-and-agreement): development-only Cohen and Fleiss references. SciPy adaptive integration and unconcentrated Gaussian likelihood optimization are development-only independent checks, absent from application runtime.

[Editable example](../examples/reliability_eight.py) runs all eight methods and
checks exact file-artifact restoration. Source tests, frozen source identity,
installed native Run and actual app restart are distinct acceptance gates.
No proprietary executable or public release is part of this method acceptance.
