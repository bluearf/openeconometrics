# Instructor companion · Lab 07

## Teaching sequence

Keep the known population reference distinct from a sample-fitted z score. Ask students to translate points on both axes.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** The score is (1024−1000)/12 = 2. This calculation uses the known population mean and SD.

**B.** The model fraction is 0.158655 and the observed fraction 0.138. Their difference is sampling variation relative to this model, rather than an arithmetic error.

**C.** The divisor is the population SD of 12. The sample SD becomes 0.95949, equal to 11.5139 / 12, and varies across finite samples.

**D.** Evidence should address measurement quality, process stability, independence or dependence, distributional shape and uncertainty about the population parameters. The synthetic mechanism alone supplies none of that empirical evidence.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/07-normal-model/parcel_weights.xlsx) for the published analysis. The [original generator](generators/07-normal-model.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/07-normal-model/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/07-normal-model/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: linear standardization mean, linear standardization sd, tail complement. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Treating a z score as a probability; forcing sample moments to equal population moments; normality from standardization alone; extrapolating a synthetic quality model to production.
