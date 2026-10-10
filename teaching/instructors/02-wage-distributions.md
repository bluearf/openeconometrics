# Instructor companion · Lab 02

Use the distribution question to motivate each statistic. Ask students to distinguish payroll totals, a worker near the middle, dispersion and low-pay thresholds. The unusually high wage is deliberately generated; this prevents a convenient deletion rule from becoming the default response to a tail observation.

## Worked exercise answers

1. The coefficient of variation is $10.7744151503/19.0517717342=0.565534$. It is dimensionless under positive rescaling but is not sensible for every scale, notably quantities with an arbitrary zero or a mean near zero. It does not completely describe inequality.
2. The right tail persists across reasonable bin counts. Local gaps and apparent small modes may change. Require the same 795 workers and scale before discussing plotting differences.
3. Grade the actual threshold count from `(observed.hourly_wage < 12).mean()`. A strict inequality count is not an interpolated percentile. Ties would make the distinction even more visible.
4. Doubling only the maximum adds $114.8624748893/795=0.144481101$ to the mean, producing 19.196252835. The median does not change because the maximum remains above the middle order statistic.
5. Dividing wages by ten divides mean, median, SD and IQR by ten. Log wages fall by $\log 10$. Their standard deviation is unchanged by this additive shift. Rank order and upper-tail wage shares remain unchanged.
6. A payroll paragraph should justify the arithmetic mean and its additive interpretation. A middle-worker paragraph can emphasize median and quartiles. Both must state synthetic data, hourly units and 795 observed of 800 generated workers.

## Scientific and reproduction notes

Native `oe.describe` uses the stated Stata empirical percentile convention. The script checks its mean against a direct sum, SD against centered squared deviations, and quartiles against independently ordered observations. Both native histograms use every observed wage; missing reports are explicitly excluded once before both transformations. The mean-without-maximum calculation is sensitivity analysis, not a declared replacement sample.

The [full reference](../labs/02-wage-distributions/reference.json) records all summaries, binned figure data and independent checks. Native output order is table, plot, plot. `models` is empty because this is descriptive analysis. Default complete-file execution writes no exports; explicit `run_lab(output_dir="docs/teaching/labs/02-wage-distributions")` writes and reads back the full JSON plus LaTeX table. Neither the student packet nor the student handout contains this companion or the reference receipt.

**Prepared input:** [wage_distributions.xlsx](../labs/02-wage-distributions/wage_distributions.xlsx). Use the stored observations for the published analysis. The [original generator](generators/02-wage-distributions.py) supports new dataset editions; update results and answers when changing observations. Retain all 800 workers and the five missing wage reports in the workbook; select observed wages explicitly in the analysis.
