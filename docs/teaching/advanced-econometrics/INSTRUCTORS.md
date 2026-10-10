# Instructor guide · Advanced Econometrics with OpenEconometrics

Distribute the separate student edition to the class. This guide and the chapter companions contain worked answers and reproduction evidence.

## Chapter guides

| Lab | Instructor companion |
|---|---|
| 01 | [The Frisch–Waugh–Lovell Identity](instructors/01-partialling-out.md) |
| 02 | [An Exact Sample Omitted-Variable Decomposition](instructors/02-omitted-variable-bias.md) |
| 03 | [Measurement Error and Attenuation](instructors/03-measurement-error.md) |
| 04 | [Collinearity, Coefficients and Stable Combinations](instructors/04-collinearity.md) |
| 05 | [Leverage and Leave-One-Out Sensitivity](instructors/05-leverage-influence.md) |
| 06 | [Known Variance Weights and WLS](instructors/06-weighted-least-squares.md) |
| 07 | [Clustered Errors and the Sampling Unit](instructors/07-clustered-inference.md) |
| 08 | [Serial Dependence and HAC Covariance](instructors/08-hac-inference.md) |
| 09 | [Joint Wald Restrictions with Robust Covariance](instructors/09-joint-restrictions.md) |
| 10 | [Delta-Method Inference for a Coefficient Ratio](instructors/10-nonlinear-delta.md) |
| 11 | [Specification Changes and Common Samples](instructors/11-common-sample.md) |
| 12 | [Centering a Polynomial without Changing Its Fit](instructors/12-polynomial-centering.md) |
| 13 | [Fixed Effects as Dummies and Within Variation](instructors/13-fixed-effects-algebra.md) |
| 14 | [First Differences and Their Error Structure](instructors/14-first-differences.md) |
| 15 | [Instrumental Variables and the Projection Formula](instructors/15-iv-moment-condition.md) |
| 16 | [Weak Relevance and IV Sensitivity](instructors/16-weak-instruments.md) |
| 17 | [Logit Probabilities, Odds and Marginal Effects](instructors/17-logit-marginal-effects.md) |
| 18 | [Poisson Rates with an Exposure Offset](instructors/18-poisson-exposure.md) |
| 19 | [Local Projections over Several Horizons](instructors/19-local-projections.md) |
| 20 | [A Reproducible Treatment-Effect Report](instructors/20-specification-report.md) |

## Reproduce and inspect the evidence

Every workbook has a flat `Data` sheet with original rows, order and missing cells, and a `Dictionary` sheet with definitions and units. Never replace a blank measurement with zero merely to simplify import. Original mechanisms in `instructors/generators/` expose `make_data()` and the seed; these files are excluded from the student edition. The ordinary student workflow reads prepared observations and never regenerates them. Resampling lessons derive their draws from the supplied observations with stated seeds and replication counts.

Run focused verification with the locked application environment and numerical threads capped:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync python scripts/verify_course_series.py --course advanced-econometrics
```

The verifier compares each workbook with its original mechanism at `1e-13`, preserving identifiers, column order, row order and the missing mask. Fresh Python execution and exact native worker execution are compared with the complete saved reference at `1e-8`. Full native tables are compared, not only selected headline numbers. Model results, where present, preserve covariance, sample positions and declared inference. Explicit export and reopened application history are checked separately. Independent development references use SciPy and statsmodels only in verification; the delivered analyses use OpenEconometrics, Python, pandas and Torch.

Build the offline editions with `scripts/build_teaching_handouts.py --course advanced-econometrics` in the documented teaching renderer environment. Print the student's complete edition to `output/teaching/advanced-econometrics/student/advanced-econometrics-labs.pdf` and the instructor edition to `output/teaching/advanced-econometrics/instructor/instructor-guide.pdf`. Then run `scripts/package_teaching_handouts.py --course advanced-econometrics`. It checks local links, source-identical Excel inputs, audience separation, portable PDF links and ZIP readback.

The renderer reads saved reference values to draw figures; it does not refit an analysis. Regenerate all dependent material together when intentionally changing a mechanism or method. Numerical agreement on these examples is distinct from empirical validity, installed acceptance and public-release evidence.
