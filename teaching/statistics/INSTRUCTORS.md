# Instructor guide · Statistics with OpenEconometrics

Distribute the separate student edition to the class. This guide and the chapter companions contain worked answers and reproduction evidence.

## Chapter guides

| Lab | Instructor companion |
|---|---|
| 01 | [What Does One Row Represent?](instructors/01-measurement.md) |
| 02 | [Which Average Answers the Question?](instructors/02-center.md) |
| 03 | [How Variable Are Delivery Times?](instructors/03-spread.md) |
| 04 | [Which Denominator Belongs in a Probability?](instructors/04-conditional-probability.md) |
| 05 | [What Does a Quality Alert Tell Us?](instructors/05-bayes.md) |
| 06 | [How Many Orders Arrive in Ten Offers?](instructors/06-discrete-model.md) |
| 07 | [From Parcel Weights to Standard Scores](instructors/07-normal-model.md) |
| 08 | [Why Do Averages Vary Less Than Observations?](instructors/08-sampling-means.md) |
| 09 | [What Does a Confidence Interval Cover?](instructors/09-confidence-intervals.md) |
| 10 | [Is the Filling Process Centered on Its Target?](instructors/10-mean-test.md) |
| 11 | [How Uncertain Is an Opt-In Rate?](instructors/11-one-proportion.md) |
| 12 | [Comparing Two Delivery Methods](instructors/12-independent-means.md) |
| 13 | [The Same Workers Before and After](instructors/13-paired-means.md) |
| 14 | [Did Two Messages Produce Different Response Rates?](instructors/14-two-proportions.md) |
| 15 | [Do Study Hours and Scores Move Together?](instructors/15-correlation.md) |
| 16 | [Are Shopping Channels and Membership Independent?](instructors/16-categorical-independence.md) |
| 17 | [Comparing Three Training Formats](instructors/17-anova.md) |
| 18 | [Comparing Ordinal Customer Ratings](instructors/18-rank-comparisons.md) |
| 19 | [Resampling the Observed Baskets](instructors/19-bootstrap.md) |
| 20 | [From a Question to a Statistical Report](instructors/20-statistical-report.md) |

## Reproduce and inspect the evidence

Every workbook has a flat `Data` sheet with original rows, order and missing cells, and a `Dictionary` sheet with definitions and units. Never replace a blank measurement with zero merely to simplify import. Original mechanisms in `instructors/generators/` expose `make_data()` and the seed; these files are excluded from the student edition. The ordinary student workflow reads prepared observations and never regenerates them. Resampling lessons derive their draws from the supplied observations with stated seeds and replication counts.

Run focused verification with the locked application environment and numerical threads capped:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --no-sync python scripts/verify_course_series.py --course statistics
```

The verifier compares each workbook with its original mechanism at `1e-13`, preserving identifiers, column order, row order and the missing mask. Fresh Python execution and exact native worker execution are compared with the complete saved reference at `1e-8`. Full native tables are compared, not only selected headline numbers. Model results, where present, preserve covariance, sample positions and declared inference. Explicit export and reopened application history are checked separately. Independent development references use SciPy and statsmodels only in verification; the delivered analyses use OpenEconometrics, Python, pandas and Torch.

Build the offline editions with `scripts/build_teaching_handouts.py --course statistics` in the documented teaching renderer environment. Print the student's complete edition to `output/teaching/statistics/student/statistics-labs.pdf` and the instructor edition to `output/teaching/statistics/instructor/instructor-guide.pdf`. Then run `scripts/package_teaching_handouts.py --course statistics`. It checks local links, source-identical Excel inputs, audience separation, portable PDF links and ZIP readback.

The renderer reads saved reference values to draw figures; it does not refit an analysis. Regenerate all dependent material together when intentionally changing a mechanism or method. Numerical agreement on these examples is distinct from empirical validity, installed acceptance and public-release evidence.
