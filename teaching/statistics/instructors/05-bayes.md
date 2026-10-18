# Instructor companion · Lab 05

## Teaching sequence

Use the manufactured-component setting to avoid confusing the mathematical exercise with a medical diagnostic recommendation. Count the routes into an alert in the confusion matrix (55 true, 71 false) before presenting Bayes algebra, and name each rate's denominator as a row or a column of the matrix.

Before running, use the pause question in Section 4: could most alerts be false even when most defects are detected? Then connect the second snippet to the formula term by term, and use the `predictive_value` helper from Section 6 for exercises 3 and 4.

## Worked exercise answers

**1.** The negative predictive value is $867/874=0.9920$ and the share of non-alerted components that are defective is $7/874=0.0080$; both divide by the 874 components with no alert. The NPV informs whether a non-alerted component can ship without further inspection: about 1 in 125 such components is still defective.

**2.** Of 10,000 components at 1% prevalence, 100 are defective; the alert catches $100\times0.8871=88.7$ of them and misses 11.3. The other 9,900 are sound, and $9{,}900\times0.07569=749.4$ of them are falsely alerted. The PPV is $88.7/(88.7+749.4)=88.7/838.1=0.1058$, matching the formula.

**3.** At 20% prevalence the PPV is 0.7455. With more defective components in the pool, the true-alert route ($0.8871\times0.20=0.1774$) grows relative to the false-alert route ($0.07569\times0.80=0.0606$), so a larger share of alerts are real.

**4.** Raising specificity to 0.99 gives a PPV of 0.8543; raising sensitivity to 0.99 gives only 0.4637. When defects are rare, most alerts come from the large sound pool. Cutting the false-alert rate from 7.6% to 1% reduces false alerts from 71 to about 9.4 per 1,000 components, while raising sensitivity can add at most about 6.4 true alerts (from 55 to 61.4).

**5.** Only the alert column would be observed: 55 true alerts and 71 false alerts. The 7 missed defects and 867 correctly silent components would never be checked. The PPV ($55/126$) could still be computed, but sensitivity, specificity, prevalence and the NPV could not, because each needs the non-alerted components. A random verification sample of non-alerted components would fill the gap.

**6.** The 89% is the sensitivity, not the accuracy. Accuracy, the share of all components classified correctly, is $(55+867)/1{,}000=0.922$. Neither number is the probability that an alerted part is defective, which is the PPV, 0.437. A correct version: "The alert flags 88.7% of defective components, but only 43.7% of alerted components (55 of 126) are defective, because defects are rare (6.2% of components)."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/05-bayes/quality_alerts.xlsx) for the published analysis. The [original generator](generators/05-bayes.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/05-bayes/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/05-bayes/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: confusion matrix partition, bayes matches alert denominator, lower base rate lowers ppv. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Sensitivity as predictive value; ignoring the base rate; silently transporting error rates across populations; evaluating false positives without their denominator.
