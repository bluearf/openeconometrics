# Instructor companion · Lab 08

## Teaching sequence

Use separate labels for population size N, sample size n and replications B. Students often confuse all three.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** Use denominator N=1000 for the exact finite-population variance. The sample SD of the 1600 replicated means uses its own sample convention; the two roles should not be mixed.

**B.** They are 15.6099 and 5.51892 currency. The former divided by sqrt(8) equals the latter.

**C.** Monte Carlo precision of the estimated sampling distribution improves. The theoretical standard error of each sample mean remains sigma/sqrt(n). The Monte Carlo standard error of the average of replicated means halves.

**D.** A finite population correction would multiply the with-replacement SE by $\sqrt{(N-n)/(N-1)}$ when sigma uses denominator N. This experiment samples with replacement, so that correction is not applied.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/08-sampling-means/spending_population.xlsx) for the published analysis. The [original generator](generators/08-sampling-means.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/08-sampling-means/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/08-sampling-means/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: means within population support, larger samples have less dispersion, mean close relative to monte carlo error. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** CLT as normality of raw data; population SD as the SE; increasing replications as increasing sample size; applying a finite-population correction to with-replacement draws.
