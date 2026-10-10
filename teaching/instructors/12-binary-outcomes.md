# Instructor notes · Lab 12

Use the [student handout](../labs/12-binary-outcomes/README.md) to distinguish index coefficients, odds ratios, finite probability changes and derivatives. Education is treated as numerical; the AME is not automatically an exact discrete one-year difference.

## Worked answers

1. The realized binary outcome is 0/1; probability is an expectation; classification adds a decision threshold and loss tradeoff.
2. LPM .051758 means 5.1758 percentage points per schooling year, conditional on other regressors in the linear projection. A proportional increase would divide by a specified baseline probability.
3. exp(.265383)=1.303931, approximately 30.393% higher odds. Probability effects vary with the index.
4. Require the student's two actual `response` predictions at education 14 and 15, fixed age 35 and one child. The 14-year value is .332805591. The finite change is profile-specific; AME averages derivatives across all 720 rows.
5. Index normalizations differ. Compare fitted response probabilities at identical profiles or comparable probability-scale effects.
6. AME averages a nonlinear derivative, while MEM evaluates it at averaged covariates. They generally differ; reweighting the evaluation distribution changes AME.
7. LPM uses HC1 and t inference. Logit/probit use observed-information covariance and normal inference. Logit matches the generating link; probit model-based SEs are not promised robust to that link misspecification.
8. Education AME .051299617, SE .005801590, CI [.039928709,.062670525], equivalent to 5.130 pp and [3.993,6.267] pp, evaluated over 720 synthetic adults.

## Reproduction and verification

`run_lab(output_dir="lab12-review")` explicitly writes the full [reference](../labs/12-binary-outcomes/reference.json) and table; the default run writes nothing. An independent Newton/IRLS calculation verifies logit coefficients and observed-information covariance. A direct probit likelihood gradient and Hessian check its optimum and information covariance. Matrix OLS/HC1 verifies the LPM. Independent differentiation of the aggregate AME reproduces its value and delta-method SE. Full JSON roundtrip, complete samples and unit-interval binary predictions are checked.

Initial maximum coefficient and covariance errors were below $10^{-13}$; probit's maximum score component was below $5\times10^{-9}$. Native displays are two `ResultBundle` models, a margins `DataFrame`, `PlotSpec`, and `Latex`. Keep model-based and robust covariance labels visible.

**Prepared input:** [labor_force_participation.xlsx](../labs/12-binary-outcomes/labor_force_participation.xlsx). Use the stored observations for the published analysis. The [original generator](generators/12-binary-outcomes.py) supports new dataset editions; update results and answers when changing observations.
