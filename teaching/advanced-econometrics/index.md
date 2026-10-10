# Advanced Econometrics with OpenEconometrics

Twenty English labs develop statistical questions through explanations, equations, complete application code, computed results, figures and student exercises. Each supplies a descriptively named Excel workbook with fixed original synthetic observations and a variable dictionary. Resampling appears only where it is the subject of the lesson.

## Chapters

| Lab | Student handout | Prepared dataset | Runnable analysis |
|---|---|---|---|
| 01 | [The Frisch–Waugh–Lovell Identity](labs/01-partialling-out/index.md) | [Excel](labs/01-partialling-out/partialling_out.xlsx) | [Python](labs/01-partialling-out/lab.py) |
| 02 | [An Exact Sample Omitted-Variable Decomposition](labs/02-omitted-variable-bias/index.md) | [Excel](labs/02-omitted-variable-bias/omitted_variable_bias.xlsx) | [Python](labs/02-omitted-variable-bias/lab.py) |
| 03 | [Measurement Error and Attenuation](labs/03-measurement-error/index.md) | [Excel](labs/03-measurement-error/measurement_error.xlsx) | [Python](labs/03-measurement-error/lab.py) |
| 04 | [Collinearity, Coefficients and Stable Combinations](labs/04-collinearity/index.md) | [Excel](labs/04-collinearity/collinearity.xlsx) | [Python](labs/04-collinearity/lab.py) |
| 05 | [Leverage and Leave-One-Out Sensitivity](labs/05-leverage-influence/index.md) | [Excel](labs/05-leverage-influence/leverage_influence.xlsx) | [Python](labs/05-leverage-influence/lab.py) |
| 06 | [Known Variance Weights and WLS](labs/06-weighted-least-squares/index.md) | [Excel](labs/06-weighted-least-squares/weighted_least_squares.xlsx) | [Python](labs/06-weighted-least-squares/lab.py) |
| 07 | [Clustered Errors and the Sampling Unit](labs/07-clustered-inference/index.md) | [Excel](labs/07-clustered-inference/clustered_inference.xlsx) | [Python](labs/07-clustered-inference/lab.py) |
| 08 | [Serial Dependence and HAC Covariance](labs/08-hac-inference/index.md) | [Excel](labs/08-hac-inference/hac_inference.xlsx) | [Python](labs/08-hac-inference/lab.py) |
| 09 | [Joint Wald Restrictions with Robust Covariance](labs/09-joint-restrictions/index.md) | [Excel](labs/09-joint-restrictions/joint_restrictions.xlsx) | [Python](labs/09-joint-restrictions/lab.py) |
| 10 | [Delta-Method Inference for a Coefficient Ratio](labs/10-nonlinear-delta/index.md) | [Excel](labs/10-nonlinear-delta/nonlinear_delta.xlsx) | [Python](labs/10-nonlinear-delta/lab.py) |
| 11 | [Specification Changes and Common Samples](labs/11-common-sample/index.md) | [Excel](labs/11-common-sample/common_sample.xlsx) | [Python](labs/11-common-sample/lab.py) |
| 12 | [Centering a Polynomial without Changing Its Fit](labs/12-polynomial-centering/index.md) | [Excel](labs/12-polynomial-centering/polynomial_centering.xlsx) | [Python](labs/12-polynomial-centering/lab.py) |
| 13 | [Fixed Effects as Dummies and Within Variation](labs/13-fixed-effects-algebra/index.md) | [Excel](labs/13-fixed-effects-algebra/fixed_effects_algebra.xlsx) | [Python](labs/13-fixed-effects-algebra/lab.py) |
| 14 | [First Differences and Their Error Structure](labs/14-first-differences/index.md) | [Excel](labs/14-first-differences/first_differences.xlsx) | [Python](labs/14-first-differences/lab.py) |
| 15 | [Instrumental Variables and the Projection Formula](labs/15-iv-moment-condition/index.md) | [Excel](labs/15-iv-moment-condition/iv_moment_condition.xlsx) | [Python](labs/15-iv-moment-condition/lab.py) |
| 16 | [Weak Relevance and IV Sensitivity](labs/16-weak-instruments/index.md) | [Excel](labs/16-weak-instruments/weak_instruments.xlsx) | [Python](labs/16-weak-instruments/lab.py) |
| 17 | [Logit Probabilities, Odds and Marginal Effects](labs/17-logit-marginal-effects/index.md) | [Excel](labs/17-logit-marginal-effects/logit_marginal_effects.xlsx) | [Python](labs/17-logit-marginal-effects/lab.py) |
| 18 | [Poisson Rates with an Exposure Offset](labs/18-poisson-exposure/index.md) | [Excel](labs/18-poisson-exposure/poisson_exposure.xlsx) | [Python](labs/18-poisson-exposure/lab.py) |
| 19 | [Local Projections over Several Horizons](labs/19-local-projections/index.md) | [Excel](labs/19-local-projections/local_projections.xlsx) | [Python](labs/19-local-projections/lab.py) |
| 20 | [A Reproducible Treatment-Effect Report](labs/20-specification-report/index.md) | [Excel](labs/20-specification-report/specification_report.xlsx) | [Python](labs/20-specification-report/lab.py) |

## Use a chapter

Import the named workbook into OpenEconometrics without renaming it. Open the complete Python file in an empty Python document and run the whole file. The script loads the most recent import with that filename or stem. For ordinary Python, keep the workbook beside its script. An explicit `run_lab(data_path="/path/to/workbook.xlsx")` selects another location. Default execution displays results without writing exports.

Read the handout on GitHub or in the separate offline student edition. Students receive explanations, worked analyses and unanswered exercises. Teaching notes, exercise solutions, original data mechanisms and numerical evidence are indexed separately in [the instructor guide](INSTRUCTORS.md) and excluded from the student archive.

All observations, explanations and exercises are original synthetic teaching material under the repository's Apache-2.0 license. Results describe the supplied simulations. External readings provide conceptual background; their exercises and datasets are not reproduced.
