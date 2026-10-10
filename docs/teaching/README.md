# OpenEconometrics Teaching Labs

[All four teaching courses](COURSES.md) · Econometrics · Statistics · Microeconomics · Advanced Econometrics

Twenty English labs connect economic questions, mathematical reasoning and complete analyses in OpenEconometrics. Each student handout explains the comparison, defines the variables and units, develops the equations, walks through the code, interprets computed results and ends with exercises.

The course moves from data and sampling to regression, policy evaluation and time series. Nineteen chapters provide fixed Excel datasets with descriptive filenames. Lab 03 generates repeated random samples because simulation is the subject of that lesson. The original synthetic data mechanisms are documented in the instructor edition. Reported results describe those simulations; they are not empirical findings about actual workers, schools or policies.

## Course chapters

| Lab | Student handout | Runnable analysis | Excel dataset |
|---|---|---|---|
| 01 | [The Economic Question Comes First](labs/01-economic-question/README.md) | [Python](labs/01-economic-question/lab.py) | [Excel](labs/01-economic-question/training_and_wages.xlsx) |
| 02 | [A First Look at Wages](labs/02-wage-distributions/README.md) | [Python](labs/02-wage-distributions/lab.py) | [Excel](labs/02-wage-distributions/wage_distributions.xlsx) |
| 03 | [What Changes Across Random Samples?](labs/03-sampling/README.md) | [Python](labs/03-sampling/lab.py) | Sampling simulation |
| 04 | [Does Class Size Predict Test Scores?](labs/04-class-size/README.md) | [Python](labs/04-class-size/lab.py) | [Excel](labs/04-class-size/class_size.xlsx) |
| 05 | [How Precise Is Our Estimate?](labs/05-inference/README.md) | [Python](labs/05-inference/lab.py) | [Excel](labs/05-inference/wage_inference.xlsx) |
| 06 | [What Changes When We Add Controls?](labs/06-controls/README.md) | [Python](labs/06-controls/lab.py) | [Excel](labs/06-controls/wages_and_controls.xlsx) |
| 07 | [Do Groups Have Different Wage Profiles?](labs/07-interactions/README.md) | [Python](labs/07-interactions/lab.py) | [Excel](labs/07-interactions/wage_interactions.xlsx) |
| 08 | [Modeling Percentage Changes and Curvature](labs/08-functional-form/README.md) | [Python](labs/08-functional-form/lab.py) | [Excel](labs/08-functional-form/wage_profiles.xlsx) |
| 09 | [Same Coefficient, Different Standard Error?](labs/09-robust-uncertainty/README.md) | [Python](labs/09-robust-uncertainty/lab.py) | [Excel](labs/09-robust-uncertainty/household_spending.xlsx) |
| 10 | [Testing Several Claims Together](labs/10-joint-tests/README.md) | [Python](labs/10-joint-tests/lab.py) | [Excel](labs/10-joint-tests/productivity_joint_tests.xlsx) |
| 11 | [Comparing Firms to Themselves](labs/11-panel-fixed-effects/README.md) | [Python](labs/11-panel-fixed-effects/lab.py) | [Excel](labs/11-panel-fixed-effects/firm_panel.xlsx) |
| 12 | [Explaining Labor-Force Participation](labs/12-binary-outcomes/README.md) | [Python](labs/12-binary-outcomes/lab.py) | [Excel](labs/12-binary-outcomes/labor_force_participation.xlsx) |
| 13 | [Estimating Effects with an Instrument](labs/13-instrumental-variables/README.md) | [Python](labs/13-instrumental-variables/lab.py) | [Excel](labs/13-instrumental-variables/schooling_instruments.xlsx) |
| 14 | [Evaluating a Randomized Training Program](labs/14-randomized-program/README.md) | [Python](labs/14-randomized-program/lab.py) | [Excel](labs/14-randomized-program/training_program.xlsx) |
| 15 | [Before and After a Policy Reform](labs/15-difference-in-differences/README.md) | [Python](labs/15-difference-in-differences/lab.py) | [Excel](labs/15-difference-in-differences/policy_panel.xlsx) |
| 16 | [What Happens Around an Eligibility Cutoff?](labs/16-regression-discontinuity/README.md) | [Python](labs/16-regression-discontinuity/lab.py) | [Excel](labs/16-regression-discontinuity/scholarship_cutoff.xlsx) |
| 17 | [The Trap of Trending Series](labs/17-trending-series/README.md) | [Python](labs/17-trending-series/lab.py) | [Excel](labs/17-trending-series/trending_series.xlsx) |
| 18 | [Accounting for Persistent Shocks](labs/18-serial-correlation/README.md) | [Python](labs/18-serial-correlation/lab.py) | [Excel](labs/18-serial-correlation/advertising_and_sales.xlsx) |
| 19 | [Forecasting Inflation Without Looking Ahead](labs/19-forecasting/README.md) | [Python](labs/19-forecasting/lab.py) | [Excel](labs/19-forecasting/forecast_series.xlsx) |
| 20 | [From Research Question to Finished Report](labs/20-research-report/README.md) | [Python](labs/20-research-report/lab.py) | [Excel](labs/20-research-report/school_tutoring.xlsx) |

## Run a chapter in OpenEconometrics

Download the chapter's named Excel workbook and import it into OpenEconometrics without renaming it. The first sheet, **Data**, contains the supplied observations; **Dictionary** defines the variables, coding, units and missing cells. Then create an empty Python document, import the complete `lab.py`, and press **Run**. Source import appends to the current draft, so start with an empty document. Run the complete file before using snippets that refer to its variables or helper functions.

The script loads the most recently imported dataset with that workbook's filename or filename stem. For ordinary Python, keep the workbook beside `lab.py` and run the file. For an explicit path, import the function with `from lab import run_lab`, then call `run_lab(data_path="/path/to/class_size.xlsx")`. Analysis never silently generates replacement data. Lab 03 is the exception: run its Python file directly to create repeated samples from the stated population.

Read the Markdown handout on GitHub, or use the offline HTML/PDF student edition. The application's Python document executes the paired code; Markdown is read separately. In an ordinary Python session, use `print(model.summary())` where the handout uses the application's `display(model)`.

Each full file is independent of earlier chapters' session variables. The analyses use the library and chart package included in the application. The selected source revision is the reproducibility target; older published packages can have different method contracts.

## Editions

The **student edition** contains explanations, worked analyses and unanswered exercises. The **instructor edition** contains teaching notes, exercise answers, and reproduction and verification details. These are separate files and separate distribution archives. The instructor material is indexed in [the instructor guide](INSTRUCTORS.md); it is excluded from the student archive.

## Reading map

The topic sequence is compatible with Stock and Watson, *Introduction to Econometrics*, Global Edition, fourth edition, while using original explanations, questions and simulations. The [publisher's contents](https://www.pearson.com/en-gb/subject-catalog/p/introduction-to-econometrics-global-edition/P200000005500/9781292264523) provide the reading map. Lab numbers identify this course's chapters, not textbook exercise numbers. The material does not reproduce the textbook's text or datasets and implies no author or publisher endorsement.

All original course material follows the repository's Apache-2.0 license. References to empirical extensions require an explicit data source and suitable redistribution permission before data can be bundled.
