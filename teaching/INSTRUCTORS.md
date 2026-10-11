# Instructor guide

This material accompanies the twenty student chapters. Teaching notes, worked exercise answers and reproduction evidence are kept here and in the chapter-specific files. The student distribution contains none of these files.

## Chapter guides

| Lab | Instructor notes and worked answers |
|---|---|
| 01 | [The Economic Question Comes First](instructors/01-economic-question.md) |
| 02 | [A First Look at Wages](instructors/02-wage-distributions.md) |
| 03 | [What Changes Across Random Samples?](instructors/03-sampling.md) |
| 04 | [Does Class Size Predict Test Scores?](instructors/04-class-size.md) |
| 05 | [How Precise Is Our Estimate?](instructors/05-inference.md) |
| 06 | [What Changes When We Add Controls?](instructors/06-controls.md) |
| 07 | [Do Groups Have Different Wage Profiles?](instructors/07-interactions.md) |
| 08 | [Modeling Percentage Changes and Curvature](instructors/08-functional-form.md) |
| 09 | [Same Coefficient, Different Standard Error?](instructors/09-robust-uncertainty.md) |
| 10 | [Testing Several Claims Together](instructors/10-joint-tests.md) |
| 11 | [Comparing Firms to Themselves](instructors/11-panel-fixed-effects.md) |
| 12 | [Explaining Labor-Force Participation](instructors/12-binary-outcomes.md) |
| 13 | [Estimating Effects with an Instrument](instructors/13-instrumental-variables.md) |
| 14 | [Evaluating a Randomized Training Program](instructors/14-randomized-program.md) |
| 15 | [Before and After a Policy Reform](instructors/15-difference-in-differences.md) |
| 16 | [What Happens Around an Eligibility Cutoff?](instructors/16-regression-discontinuity.md) |
| 17 | [The Trap of Trending Series](instructors/17-trending-series.md) |
| 18 | [Accounting for Persistent Shocks](instructors/18-serial-correlation.md) |
| 19 | [Forecasting Inflation Without Looking Ahead](instructors/19-forecasting.md) |
| 20 | [From Research Question to Finished Report](instructors/20-research-report.md) |

## Reproduce and inspect the evidence

The student inputs are committed Excel workbooks with descriptive filenames. Their first sheet, **Data**, contains the original observations in their original order, including missing cells. **Dictionary** records variable definitions, units, coding and provenance. Students import the workbook and run the analysis. Lab 03 draws repeated samples as part of the sampling lesson.

Original generators are separate instructor files under `instructors/generators/`. Each exposes `make_data()` and its original constants. In the offline instructor edition, `generator.py` sits beside each prepared-data chapter. For example, `runpy.run_path("generator.py")["make_data"]()` reconstructs that chapter's observations. Lab 13's generator also accepts the original `relevance` and `direct_effect` scenario arguments. These tools are excluded from the student distribution.

Use a source checkout with its locked application dependencies:

```sh
uv sync --frozen --extra app
uv run python scripts/prepare_teaching_datasets.py
uv run python scripts/verify_teaching_labs.py
```

The dataset check compares every saved workbook with its original generator, including column order, row order, missing cells and numerical precision. `scripts/prepare_teaching_datasets.py --output /path/to/input-rows` writes lossless JSON rows when intentionally rebuilding workbooks. The complete verifier stages a supplied workbook for standalone execution and imports it through the application's dataset reader before running the exact student source in the native Python worker. It compares scientific results and chart data with the saved references, checks the complete fitted-model JSON and estimation sample, verifies explicit exports, and reopens saved history. It preserves model IDs and timestamps when checking a single export's roundtrip but excludes those volatile fields from comparisons across independent fits. Floating-point comparisons use numerical tolerances; a checksum identifies a particular serialized realization rather than proving identical floating-point bytes on every CPU.

Each chapter also includes independent numerical calculations appropriate to its topic: direct descriptive identities, sampling moments, matrix regression and sandwich covariance, restriction tests, within transformations, treatment contrasts, local-polynomial calculations or forecast recursion. The chapter guide states its exact scope. These checks verify bounded teaching examples and do not establish universal estimator parity or an empirical identification strategy.

The prepared-data entry point is `run_lab(output_dir=None, display_callback=None, data_path=None)`. Lab 03 retains `run_lab(output_dir=None, display_callback=None)`. Default execution prints or displays results without exporting files. After running a chapter, request an explicit export with `run_lab(output_dir="my-results")`. It writes a complete `reference.json` and a `table.tex` fragment to that directory. Full fitted states include covariance matrices, inference metadata and retained-row positions where the model's contract supports them. Native display previews are not substituted for complete states.

The optional renderer reads the checked reference results and produces figures; it does not refit models. When intentionally changing a generating mechanism, rebuild the supplied workbook and update the analysis, narrative, reference, figures and worked answers together. Do not silently change a seed to obtain a more attractive diagnostic result.

## Build the separate editions

```sh
uv venv /tmp/openecon-teaching-build
uv pip install --python /tmp/openecon-teaching-build/bin/python -r docs/teaching/requirements-build.txt
npm --prefix web ci
/tmp/openecon-teaching-build/bin/python scripts/build_teaching_handouts.py
```

Open `output/teaching/student/index.html` or `output/teaching/instructor/index.html`. Each has its own combined print edition. Math assets and figures are local. Optional external readings need internet access. Print the combined editions to separate PDFs. The student's HTML tree contains student handouts, prepared Excel inputs, runnable files, figures and LaTeX table downloads; full references, worked answers and reproduction notes remain in the instructor tree. The delivery verifier checks archive contents and prevents instructor files or headings from entering the student package.

Save the student PDF as `output/teaching/student/econometrics-labs.pdf` and the instructor PDF as `output/teaching/instructor/instructor-guide.pdf`, then run `/tmp/openecon-teaching-build/bin/python scripts/package_teaching_handouts.py`. This replaces preview-server PDF links with paths inside each extracted edition and checks local links, required chapter files, PDF signatures and archive readback before producing the two ZIP distributions. Run it after every change to the printed editions. Distribute the ZIP when students need the Excel, Python and LaTeX files, and have them extract it and open `index.html` for the navigable offline course.

The LaTeX table fragments use `booktabs`, `adjustbox`, `amsmath`, `amssymb` and `array` as needed. The renderer assembles a standalone `output/teaching/instructor/regression-tables.tex` for compilation. The existing user OLS document is independent of these new course files.

## Review and maintenance

Review the clarity of the economic comparison, notation, units, worked numerical interpretation and exercise answers, alongside numerical checks. A valid computation can answer a poorly chosen question; a precise coefficient can still rely on implausible causal assumptions. Preserve those distinctions in teaching and in future empirical adaptations.

Task status is recorded in [the curriculum task register](TASKS.md). The earlier hold after three pilots was superseded by the user's explicit request to complete the full series. The PR remains available for review before merge or publication.
