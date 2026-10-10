# Instructor companion · Lab 16

## Teaching sequence

Derive the central comparison before execution. Inspect the full coefficient table and retained sample, then trace the chapter-specific arithmetic check. Require students to distinguish a point-estimate identity from an inference or identification claim.

Before running, ask students to define the observational unit and predict the comparison. Connect the code to the equation before interpreting the output.

## Worked exercise answers

**A.** The structural effect 1.2, exogenous control effect and structural disturbance.

**B.** The weak first-stage F=2.11473 versus 289.966.

**C.** No. They retain conventional t inference and must be interpreted with that limitation.

**D.** No. Exclusion and relevance strength are distinct requirements.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/16-weak-instruments/weak_instruments.xlsx) for the published analysis. The [original generator](generators/16-weak-instruments.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/16-weak-instruments/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/16-weak-instruments/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: weak instrument relevance lower, same structural effect, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A fixed F cutoff is not a universal guarantee. Weak-IV-robust procedures such as inversion of suitable moment tests answer an additional question and are not implemented by these ordinary intervals.
