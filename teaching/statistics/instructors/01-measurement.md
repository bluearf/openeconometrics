# Instructor companion · Lab 01

## Teaching sequence

Ask students to mark the observational unit and classify every column before running a summary. Compare household-weighted and person-weighted questions verbally before writing either formula.

Before running, ask students to define the observational unit and predict which denominator each reported number uses. Connect the code to the two ratio formulas before interpreting the output.

## Worked exercise answers

**1.** There are 180 raw households, 176 observed spending values and 4 missing reports; 176 + 4 = 180. Every number in the Section 4 table uses 176, including the household-size row, because the summary is computed on the observed-spending frame. The per-member mean 299.558 also uses 176.

**2.** On the 176 observed households, total spending is 156,461.44 and total membership is 618 people, so $\sum C_i/\sum H_i = 253.17$ currency per person per month. This is the size-weighted mean of the household ratios $C_i/H_i$. It is lower than the equal-weight mean 299.558 because larger households tend to spend less per member, and person weighting gives those households more weight.

**3.** The mean size is 3.5111 for all 180 households and 3.5114 for the 176 observed households. The four excluded households have sizes 6, 1, 1 and 6, which average 3.5, almost exactly the overall mean. The two means would differ noticeably if missing reports were concentrated among large or among small households.

**4.** With North coded 0 and South coded 1, "mean region = 0.5" only restates that half of the households (90 of 180) are in the South. The codes have no distance, so the number is not an average location. Report counts and shares by region instead: 90 North and 90 South.

**5.** Filling the blanks with zero gives a mean of 869.23 over 180 households, about 19.75 below the observed-case mean. It asserts that four households spent nothing, which the data do not say, enlarges the denominator and pulls the mean down. Preserve unknown values and investigate their missingness mechanism; here all four blanks are in the South.

**6.** For example: "Among the 176 of 180 synthetic households that reported spending, mean monthly spending is 888.99 currency units per household; the four households without a report are excluded rather than counted as zero."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/01-measurement/household_measurements.xlsx) for the published analysis. The [original generator](generators/01-measurement.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/01-measurement/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/01-measurement/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: unique household ids, observed plus missing equals raw, native count matches retained rows, native mean matches direct sum. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Treating numeric identifiers as measured quantities; replacing missing values by zero; mixing household and person denominators; assuming observed cases represent nonrespondents.
