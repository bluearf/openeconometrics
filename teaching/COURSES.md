# OpenEconometrics teaching courses

These four English courses use original explanations, synthetic observations, equations, complete Python analyses, executed results, figures and exercises. The foundational Econometrics series has 20 chapters; the three additional series each add 20. Student editions exclude teaching notes, exercise answers, original generators and full verification references. Those materials are indexed in each instructor guide.

| Course | Chapters | Student entry | Instructor entry | Focus |
|---|---:|---|---|---|
| Econometrics | 20 | [Course](index.md) | [Guide](INSTRUCTORS.md) | Regression, policy evaluation and introductory time series |
| Statistics | 20 | [Course](statistics/index.md) | [Guide](statistics/INSTRUCTORS.md) | Data, probability, sampling, intervals, tests and reporting |
| Microeconomics | 20 | [Course](microeconomics/index.md) | [Guide](microeconomics/INSTRUCTORS.md) | Consumer choice, production, markets, strategic interaction and welfare |
| Advanced Econometrics | 20 | [Course](advanced-econometrics/index.md) | [Guide](advanced-econometrics/INSTRUCTORS.md) | Projection algebra, covariance, panel transformations, IV, likelihood and dynamic responses |

Every additional chapter supplies a descriptively named Excel workbook. Import it without renaming, open the corresponding complete `lab.py` in a new Python document and run the whole file. For ordinary Python, keep the workbook beside the script or select an explicit `data_path`. Students use the stored rows; original generation belongs to instructor preparation. The Statistics sampling/bootstrap chapters resample the supplied population/sample because resampling is the lesson.

Microeconomics chapters 01–19 evaluate stated model scenarios through Python arithmetic and native OpenEconometrics tables/charts. Their workbooks contain feasible plans, schedules or model inputs; they do not pretend to estimate consumer behavior from empirical observations. Chapter 20 fits a synthetic demand regression. Advanced chapters preserve complete fitted models, including full covariance, declared inference and retained sample positions. The exact assumptions and limits are stated in each handout.

## Offline editions

Build each additional series with `scripts/build_teaching_handouts.py --course statistics`, `--course microeconomics` or `--course advanced-econometrics` using the optional pinned [renderer dependencies](requirements-build.txt) and local KaTeX assets. Source Markdown, Excel, Python, figures, LaTeX and full instructor references are versioned here. Build outputs are written under `output/teaching/<course>/` and verification receipts under `artifacts/teaching/`. Download the separate student and instructor distributions from the [teaching release](https://github.com/bluearf/openeconometrics/releases/tag/teaching-v0.1.0).

Each student ZIP contains a navigable offline course, a combined reading PDF, 20 workbooks, runnable scripts, figures and LaTeX table fragments. Each instructor ZIP additionally contains worked answers, full references and original mechanisms. Follow the corresponding instructor guide to reproduce, print and verify an edition. Course registers record completed implementation, numerical verification and editorial checks. The user has authorized distribution; this does not imply a textbook author or publisher endorsement.

All new questions, scenarios, mechanisms, explanations and worked solutions are original Apache-2.0 material. Further-reading links identify conceptual background; no third-party textbook chapters, exercises, solutions or datasets are copied into these courses.
