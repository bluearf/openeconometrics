# Instructor companion · Lab 09

## Teaching sequence

Point to the two native tables and distinguish the mean interval from the interval for mean−10. The chapter intentionally extracts the mean interval from statistics.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** There are 72 raw encounters, 2 missing reports and 70 observed waits. Degrees of freedom are 69 = 70−1.

**B.** The 95% interval is [10.0523, 11.1501] minutes; the 99% interval is [9.87237, 11.33] minutes. Both are centered on 10.6012 minutes.

**C.** Multiply the mean, SE and both interval endpoints by 60. Degrees of freedom and the confidence level remain unchanged.

**D.** No. It concerns the population mean. Coverage is a repeated-sampling property of the interval procedure, and an individual-wait interval is a separate object.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/09-confidence-intervals/service_waits.xlsx) for the published analysis. The [original generator](generators/09-confidence-intervals.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/09-confidence-intervals/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/09-confidence-intervals/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native se matches sample formula, df is n minus one, ci symmetric about mean, higher confidence wider. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Confidence as a posterior probability; individual coverage as mean coverage; a narrower 99% interval; reading difference-from-null endpoints as mean endpoints.
