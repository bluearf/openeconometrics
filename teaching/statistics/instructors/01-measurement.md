# Instructor companion · Lab 01

## Teaching sequence

Ask students to mark the observational unit and classify every column before running a summary. Compare household-weighted and person-weighted questions verbally before writing either formula.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** There are 180 raw households, 176 observed spending values and 4 missing reports. Their sum reconciles to the raw count.

**B.** IDs distinguish records. The difference between IDs 10 and 20 is not a measured household characteristic. Relabeling IDs would change their average without changing any household.

**C.** On the observed households, use $\sum C_i/\sum H_i$. The displayed 299.558 instead gives each household equal weight after calculating $C_i/H_i$. Weighting those ratios by household size recovers the person-weighted ratio.

**D.** No. It would assert four zero-spending observations, enlarge the denominator and reduce the mean. Preserve unknown values and investigate their missingness mechanism.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/01-measurement/household_measurements.xlsx) for the published analysis. The [original generator](generators/01-measurement.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/01-measurement/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/01-measurement/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: unique household ids, observed plus missing equals raw, native count matches retained rows, native mean matches direct sum. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Treating numeric identifiers as measured quantities; replacing missing values by zero; mixing household and person denominators; assuming observed cases represent nonrespondents.
