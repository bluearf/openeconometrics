# Instructor companion · Lab 08

Separate three operations: differentiating the polynomial, taking an exact finite difference, and exponentiating a log difference. Students often combine their approximations without noticing. Also distinguish exp(ElogW) from EW before introducing any retransformation correction.

## Worked exercise answers

1. Three additional education years give exact fitted geometric-mean change 19.686167908%, with transformed 95% interval [16.465684434%, 22.995703481%]. Use exp(3b)−1, not three times the one-year percentage.
2. At 10 years, derivative = 0.033825622, SE = 0.001617681, CI [0.030647270, 0.037003973]. At 30 years, derivative = 0.002573415, SE = 0.002240476, CI [−0.001828578, 0.006975408]. Contrast weights on experience_squared are 20 and 60 respectively.
3. Exact log change from 10 to 20 is 10b1 + 300b2; exponentiation gives 29.709312400%. Ten times the derivative at 10 gives a 33.825621875% first-order approximation, differing because curvature and exponential conversion both matter.
4. Let C = E−20. The linear-in-C coefficient becomes b1 + 40b2 = 0.018199519, the quadratic coefficient stays −0.000781305, and the intercept shifts by 20b1 + 400b2. Add 20 to the turning point expressed in C to recover 31.646869501 years. Fitted log wages are unchanged.
5. `quadratic.nlcom(lambda b: -b["experience"]/(2*b["experience_squared"]))` gives point 31.646869501, delta SE 1.605609884 and 95% CI [28.492234181, 34.801504821]. A denominator near zero makes a ratio unstable; a symmetric delta interval can then be misleading. Do not interpret its test against zero as the substantive purpose of estimating the turning point.
6. The known DGP mean/geometric correction is exp(sigma(E)^2/2). At 5 years it is 1.016331931; at 35 it is 1.046027860. It is experience-dependent, so one global multiplier does not provide the exact conditional mean correction throughout this heteroskedastic generator.

## Scientific and reproduction notes

The [full reference](../labs/08-functional-form/reference.json) includes both complete models and all local contrasts. The source independently checks matrix OLS, HC3, derivative contrast covariance, exact finite-difference algebra, known DGP retransformation ordering and full ResultBundle JSON readback. Model df = 496, all 500 workers retained. Native outputs: model, table, plot. Explicit regeneration uses `run_lab(output_dir="docs/teaching/labs/08-functional-form")`. No default exports. Student text correctly labels the analytic known DGP means as population teaching quantities rather than fitted forecasts or empirical wage evidence.

**Prepared input:** [wage_profiles.xlsx](../labs/08-functional-form/wage_profiles.xlsx). Use the stored observations for the published analysis. The [original generator](generators/08-functional-form.py) supports new dataset editions; update results and answers when changing observations.
