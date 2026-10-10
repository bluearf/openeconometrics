# Instructor companion · Lab 16

## Teaching sequence

Calculate one expected cell manually, then connect the complete sum to the native expected and test tables.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** (3−1)(2−1)=2.

**B.** Multiply the relevant row and column totals, then divide by 360; it is a count predicted under independence.

**C.** No. Relabeling only reorders cells and leaves the sum unchanged.

**D.** No. The table tests association; assignment and selection are separate issues.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/16-categorical-independence/shopping_channels.xlsx) for the published analysis. The [original generator](generators/16-categorical-independence.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/16-categorical-independence/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/16-categorical-independence/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native chi square matches cell formula, expected margins match observed, adequate expected counts, observed cells partition. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Using numerical category codes as measured distances; sparse-cell asymptotics without checking counts; association as causation.
