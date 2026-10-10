# Instructor companion · Lab 05

## Teaching sequence

Use the manufactured-component setting to avoid confusing the mathematical exercise with a medical diagnostic recommendation. Count the routes into an alert before presenting Bayes algebra.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Sensitivity divides 55 by 55 + 7, the defective components. Predictive value divides 55 by 55 + 71, the alerted components.

**B.** Directly calculate 55 / (55 + 71). It equals 0.436508, matching 0.436508 from the conditional-rate formula.

**C.** Predictive value becomes 0.10585. The sound-component pool becomes larger relative to the defect pool, so false alerts make up more of the alerted group.

**D.** It would reveal outcomes among alerts but not missed defects or sound non-alerts. Sensitivity and specificity require those uninspected groups too. Verification sampling must cover the relevant cells.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/05-bayes/quality_alerts.xlsx) for the published analysis. The [original generator](generators/05-bayes.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/05-bayes/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/05-bayes/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: confusion matrix partition, bayes matches alert denominator, lower base rate lowers ppv. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Sensitivity as predictive value; ignoring the base rate; silently transporting error rates across populations; evaluating false positives without their denominator.
