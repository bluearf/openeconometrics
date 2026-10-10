# Instructor companion · Lab 06

## Teaching notes and worked answers

1. The omitted-variable component becomes positive. The short education coefficient should tend to exceed the controlled one, provided the other stated assumptions continue to hold. The population result is about the generating relationship; one realized estimate still includes sampling variation.

2. The estimate is **0.081664**, HC1 SE **0.004295**, 95% interval **[0.073227, 0.090100]**, with 585 degrees of freedom. At fixed experience, one more education year corresponds to **8.509%** higher fitted geometric-mean hourly earnings; the transformed interval is **[7.598%, 9.428%]**. Students should say “in this simulated model,” not claim an established real-world causal return.

3. Use $100\{\exp(2\times0.081663847)-1\}=17.742\%$. Changes compound multiplicatively. Doubling the exact one-year percentage omits the compounding term; doubling the log-point change is valid.

4. The complete-case rule removes 12 workers before either primary fit. A 600-row versus 588-row contrast changes both the specification and the sample. Computing the 588-row short model separately isolates the specification change on that sample. Complete-case selection can itself change an empirical target; its harmlessness cannot be assumed for real missing data.

5. $0.028984729\times(-1.236866107)=-0.035850229$, equal to $0.045813618-0.081663847$ up to rounding. Additional controls identify a causal coefficient only when the resulting conditional-exogeneity and specification assumptions are credible. A mediator, collider or poorly measured control can change the target or introduce bias. The known simulation equation gives an assumption students can inspect; observational data rarely offer that privilege.

**Common errors to surface:** comparing changing samples; treating robust SEs as a remedy for omitted variables; reading 0.0817 as an 0.0817% change; interpreting $R^2$ as identification; and reporting the partial-regression SE without the full-model correction. A useful closing discussion asks what evidence about schooling assignment would be needed before interpreting a real-data adjusted association causally.

## Reproduction and verification

The [saved reference](../labs/06-controls/reference.json) preserves both complete fitted results, including covariance matrices, retained-row positions and inference metadata, plus the selected worker IDs, data hash and chart coordinates. The [LaTeX table](../labs/06-controls/table.tex) is exported directly from those fits. Its parent document needs `booktabs`, `adjustbox` and `array`.

Each execution checks the common sample, absence of hidden exclusions, independent matrix OLS coefficients and HC1 covariance, Student-$t$ confidence limits calculated by an independent numerical integral, the sample omitted-variable identity, the partial-regression identity and full result JSON roundtrip. All eight checks passed for the reference run. These checks establish reproducibility of this small teaching example, not universal estimator parity or an empirical causal finding.

**Further reading:** [Stock and Watson, fourth-edition contents](https://www.pearson.com/en-gb/subject-catalog/p/introduction-to-econometrics-global-edition/P200000005500/9781292264523); [OpenEconometrics OLS and inference](../../ols.md); [publication-table exports](../../publication-tables.md). Labs 07–10 develop interactions, nonlinear specifications and robust inference.

**Prepared input:** [wages_and_controls.xlsx](../labs/06-controls/wages_and_controls.xlsx). Use the stored observations for the published analysis. The [original generator](generators/06-controls.py) supports new dataset editions; update results and answers when changing observations. Keep all 600 raw workers and twelve missing experience reports. Both primary models use the same 588 complete cases.
