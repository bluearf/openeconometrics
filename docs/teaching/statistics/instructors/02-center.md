# Instructor companion · Lab 02

## Teaching sequence

Start with two audiences: a shop accountant and a researcher describing a middle basket. Require students to choose a target before computing a statistic.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Use the arithmetic mean: total revenue is 240 × 32.439 currency. The median and geometric mean do not satisfy this additive identity.

**B.** The original mean is 32.439 and the sensitivity mean 31.2358. The 1.20318 gap reports the effect of removing one valid observation and changing the denominator from 240 to 239.

**C.** All three centers double: arithmetic means and order statistics scale linearly, while adding log(2) to every log spending value doubles the geometric mean after exponentiation.

**D.** A revenue total requires the additive arithmetic mean or direct sum. A separate description of a central customer's basket can use the median. Robustness does not authorize replacing the economic quantity of interest.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/02-center/shop_spending.xlsx) for the published analysis. The [original generator](generators/02-center.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/02-center/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/02-center/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: median matches order statistics, payroll identity, geometric below arithmetic, tail removal changes mean upward gap. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Calling the median the “correct” mean; deleting the valid largest basket; treating exponentiated mean logs as arithmetic spending; confusing sensitivity analysis with an alternative primary sample.
