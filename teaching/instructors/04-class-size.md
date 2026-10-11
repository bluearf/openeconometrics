# Instructor companion · Lab 04

## Teaching notes and worked answers

Use the opening question to elicit sign predictions before discussing the teaching mechanism. The core lesson is interpretation, not memorizing the HC3 formula. That formula can be an optional board derivation in a first course.

**Activity 1:** A reduction of two students corresponds to $(-2)(-1.942695)=3.885391$ fitted points. The interval is $[-2(-1.657909),-2(-2.227481)]=[3.315819,4.454963]$. Reversing the endpoints after multiplying by a negative number is essential.

**Activity 2:** $705.722744-1.942695(23.933307)=659.227618$, up to rounding. The intercept normal equation makes residuals sum to zero, so the average fitted outcome equals the average observed outcome. This property alone says nothing about causality.

**Activity 3:** The coefficient, fitted values, observation count and $R^2$ stay the same. Standard errors, test statistics, $p$ values and confidence intervals can change. The two columns use identical sample positions, not merely identical observation counts.

**Activity 4:** Check that the student applies the threshold to the supplied rows, reports both subset sizes and uses HC3 in both fits. These are two conditional subsets of one fixed sample, not newly simulated independent samples. The restricted class-size range and smaller sample can change coefficients and precision. Neither slope must equal the structural value of -1.8. Do not interpret their difference as evidence that the teaching population has a different structural slope in each subset.

**Activity 5:** One accurate version is: “In this original simulation, the fitted mean score is 9.713 points higher at 20 students per class than at 25, with a 95% HC3 interval of 8.290–11.137.” Family income, prior achievement, school resources or selection could confound an observational comparison. Do not describe the artificial score scale as a known national test scale.

**Common mistakes to catch:** reading the intercept as a realistic prediction; confusing score points with percentages; saying a low $p$ value proves causality; describing $R^2$ as a policy effect; and claiming robust standard errors correct omitted-variable bias. Lab 06 examines what changes when controls are added.

## Reproduction and verification

The paired script independently recomputes the centered OLS coefficients and explicit HC3 sandwich from the supplied workbook data. It verifies both fits retain all 240 observations and the same sample positions, checks the standard errors against the covariance diagonal, and restores the complete fitted result from JSON. In the reference execution, maximum coefficient disagreement was $2.28\times10^{-13}$ and maximum HC3 covariance disagreement was $1.09\times10^{-13}$.

After running the full source, `result = run_lab(output_dir="class-size-output")` explicitly exports complete `reference.json` and `table.tex` files relative to the active workspace. Inspect `result["summary"]` and `result["checks"]`. Model IDs and timestamps can change on a rerun; the fixed workbook observations and numerical results should agree within floating-point tolerances. The checksum identifies the analyzed workbook rows. This validates the bounded example, not every OLS specification.

**Prepared input:** [class_size.xlsx](../labs/04-class-size/class_size.xlsx). Use the stored observations for the published analysis. The [original generator](generators/04-class-size.py) supports new dataset editions; update results and answers when changing observations.
