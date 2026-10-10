# Instructor companion · Lab 06

## Teaching sequence

Have students distinguish the batch observational unit from the offer-level trials. Verify the support and tail boundary before calculating.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** The mean is 10 × 0.3 = 3 orders. Variance is 10 × 0.3 × 0.7 = 2.1 orders squared.

**B.** Sum $\binom{10}{k}0.3^k0.7^{10-k}$ for k=5,…,10. The result is 0.150268. Starting at six would answer a different question.

**C.** The mean becomes 20 × 0.3 = 6 orders and variance 20 × 0.3 × 0.7 = 4.2 orders squared. Standard deviation increases by sqrt(2), rather than doubling.

**D.** A shared shock can make outcomes within a batch dependent; changing success probabilities across batches adds heterogeneity. Either can create more dispersion than a constant-p independent binomial model.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/06-discrete-model/ten_offer_orders.xlsx) for the published analysis. The [original generator](generators/06-discrete-model.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/06-discrete-model/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/06-discrete-model/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: pmf sums to one, mean matches np, variance matches npq, counts in support. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Omitting the combinatorial factor; mixing counts and rates; using variance np instead of np(1−p); treating model and observed tail fractions as identical by definition.
