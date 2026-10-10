# Instructor companion · Lab 19

## Teaching sequence

Keep the prepared sample fixed. Show one resampling index vector and repeated IDs before presenting the distribution.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Without replacement, selecting all 90 observations preserves the same mean every time. Replacement generates empirical sampling variation.

**B.** 2.86531 is a Monte Carlo estimate; 2.89412 is the exact conditional SE of the empirical bootstrap mean.

**C.** With finitely many replications there is Monte Carlo error; its conditional expectation equals 24.9277.

**D.** No. The empirical population contains only observed baskets and inherits their selection mechanism.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/19-bootstrap/bootstrap_baskets.xlsx) for the published analysis. The [original generator](generators/19-bootstrap.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/19-bootstrap/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/19-bootstrap/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: bootstrap replicates within observed support, bootstrap center close with monte carlo tolerance, bootstrap se close to empirical target, retains all baskets per resample. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Resampling as new data collection; without-replacement bootstrap; percentile intervals as universally reliable; confusing baskets with replication means.
