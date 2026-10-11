# Statistics with OpenEconometrics

Twenty English labs develop statistical questions through explanations, equations, complete application code, computed results, figures and student exercises. Each supplies a descriptively named Excel workbook with fixed original synthetic observations and a variable dictionary. Resampling appears only where it is the subject of the lesson.

## Chapters

| Lab | Student handout | Prepared dataset | Runnable analysis |
|---|---|---|---|
| 01 | [What Does One Row Represent?](labs/01-measurement/index.md) | [Excel](labs/01-measurement/household_measurements.xlsx) | [Python](labs/01-measurement/lab.py) |
| 02 | [Which Average Answers the Question?](labs/02-center/index.md) | [Excel](labs/02-center/shop_spending.xlsx) | [Python](labs/02-center/lab.py) |
| 03 | [How Variable Are Delivery Times?](labs/03-spread/index.md) | [Excel](labs/03-spread/delivery_variation.xlsx) | [Python](labs/03-spread/lab.py) |
| 04 | [Which Denominator Belongs in a Probability?](labs/04-conditional-probability/index.md) | [Excel](labs/04-conditional-probability/customer_events.xlsx) | [Python](labs/04-conditional-probability/lab.py) |
| 05 | [What Does a Quality Alert Tell Us?](labs/05-bayes/index.md) | [Excel](labs/05-bayes/quality_alerts.xlsx) | [Python](labs/05-bayes/lab.py) |
| 06 | [How Many Orders Arrive in Ten Offers?](labs/06-discrete-model/index.md) | [Excel](labs/06-discrete-model/ten_offer_orders.xlsx) | [Python](labs/06-discrete-model/lab.py) |
| 07 | [From Parcel Weights to Standard Scores](labs/07-normal-model/index.md) | [Excel](labs/07-normal-model/parcel_weights.xlsx) | [Python](labs/07-normal-model/lab.py) |
| 08 | [Why Do Averages Vary Less Than Observations?](labs/08-sampling-means/index.md) | [Excel](labs/08-sampling-means/spending_population.xlsx) | [Python](labs/08-sampling-means/lab.py) |
| 09 | [What Does a Confidence Interval Cover?](labs/09-confidence-intervals/index.md) | [Excel](labs/09-confidence-intervals/service_waits.xlsx) | [Python](labs/09-confidence-intervals/lab.py) |
| 10 | [Is the Filling Process Centered on Its Target?](labs/10-mean-test/index.md) | [Excel](labs/10-mean-test/filling_volumes.xlsx) | [Python](labs/10-mean-test/lab.py) |
| 11 | [How Uncertain Is an Opt-In Rate?](labs/11-one-proportion/index.md) | [Excel](labs/11-one-proportion/app_opt_in.xlsx) | [Python](labs/11-one-proportion/lab.py) |
| 12 | [Comparing Two Delivery Methods](labs/12-independent-means/index.md) | [Excel](labs/12-independent-means/delivery_methods.xlsx) | [Python](labs/12-independent-means/lab.py) |
| 13 | [The Same Workers Before and After](labs/13-paired-means/index.md) | [Excel](labs/13-paired-means/worker_before_after.xlsx) | [Python](labs/13-paired-means/lab.py) |
| 14 | [Did Two Messages Produce Different Response Rates?](labs/14-two-proportions/index.md) | [Excel](labs/14-two-proportions/message_responses.xlsx) | [Python](labs/14-two-proportions/lab.py) |
| 15 | [Do Study Hours and Scores Move Together?](labs/15-correlation/index.md) | [Excel](labs/15-correlation/study_hours_scores.xlsx) | [Python](labs/15-correlation/lab.py) |
| 16 | [Are Shopping Channels and Membership Independent?](labs/16-categorical-independence/index.md) | [Excel](labs/16-categorical-independence/shopping_channels.xlsx) | [Python](labs/16-categorical-independence/lab.py) |
| 17 | [Comparing Three Training Formats](labs/17-anova/index.md) | [Excel](labs/17-anova/training_formats.xlsx) | [Python](labs/17-anova/lab.py) |
| 18 | [Comparing Ordinal Customer Ratings](labs/18-rank-comparisons/index.md) | [Excel](labs/18-rank-comparisons/customer_ratings.xlsx) | [Python](labs/18-rank-comparisons/lab.py) |
| 19 | [Resampling the Observed Baskets](labs/19-bootstrap/index.md) | [Excel](labs/19-bootstrap/bootstrap_baskets.xlsx) | [Python](labs/19-bootstrap/lab.py) |
| 20 | [From a Question to a Statistical Report](labs/20-statistical-report/index.md) | [Excel](labs/20-statistical-report/campus_cafe_offer.xlsx) | [Python](labs/20-statistical-report/lab.py) |

## Use a chapter

Import the named workbook into OpenEconometrics without renaming it. Open the complete Python file in an empty Python document and run the whole file. The script loads the most recent import with that filename or stem. For ordinary Python, keep the workbook beside its script. An explicit `run_lab(data_path="/path/to/workbook.xlsx")` selects another location. Default execution displays results without writing exports.

Read the handout on GitHub or in the separate offline student edition. Students receive explanations, worked analyses and unanswered exercises. Teaching notes, exercise solutions, original data mechanisms and numerical evidence are indexed separately in [the instructor guide](INSTRUCTORS.md) and excluded from the student archive.

All observations, explanations and exercises are original synthetic teaching material under the repository's Apache-2.0 license. Results describe the supplied simulations. External readings provide conceptual background; their exercises and datasets are not reproduced.
