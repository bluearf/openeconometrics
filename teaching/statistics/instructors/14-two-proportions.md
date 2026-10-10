# Instructor companion · Lab 14

## Teaching sequence

Use a number line in probabilities and percentage points. Ask which denominator belongs to each calculation.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** 100 × -0.0366667 = -3.66667 percentage points.

**B.** The pooled null SE 0.0365936, rather than the unpooled interval SE 0.036563.

**C.** The data and procedure do not rule out zero; they also allow a range of unequal rates.

**D.** No. Equivalence requires a specified acceptable margin and an appropriate interval-based test.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/14-two-proportions/message_responses.xlsx) for the published analysis. The [original generator](generators/14-two-proportions.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/14-two-proportions/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/14-two-proportions/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native z uses pooled null, native interval se is unpooled, group counts partition, percentage point scale. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Relative percentages as percentage points; treating non-rejection as equality; mixing pooled and unpooled variance.
