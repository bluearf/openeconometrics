# Instructor companion · Lab 05

Keep the hypothesis explicit whenever students use the word significant. The worked sample rejects a zero slope but does not reject a slope of one, making the distinction between evidence against zero and precision around a practical benchmark concrete. The generator's slope is 0.6; the realized estimate of 0.733351 is not an arithmetic inconsistency.

## Worked exercise answers

1. The 95% width is 0.6648898713 and half-width 0.3324449356. Dividing the half-width by SE 0.1681304507 gives the stated t critical value. Width is twice critical value times SE.
2. `model.lincom({"education": 1}, constant=-0.5)` gives difference 0.233351344, SE 0.168130451, t = 1.387918385, p = 0.167398663, and difference interval [−0.099093592, 0.565796279]. The null is not rejected at 5%; 0.5 lies within the original slope interval. Do not describe this as proof that the coefficient equals 0.5.
3. Multiplying wage by 100 multiplies coefficient, SE and interval endpoints by 100. The zero-null t statistic and p value are unchanged. A nonzero null must also be rescaled if it represents the same economic benchmark.
4. One fewer year gives −0.733351344 currency/hour and interval [−1.065796279, −0.400906408]. Endpoint reversal is necessary. Its SE remains 0.168130451.
5. Accept clearly defined practical thresholds. Require attention to costs, hours, target population, alternative use of resources, and causal relevance. A numerical benchmark alone does not supply a welfare calculation.
6. Require synthetic labeling, explicit association wording and the estimate/uncertainty. Neither a small p value nor HC3 identifies a real policy effect.

## Scientific and reproduction notes

Three complete models differ only in alpha: 0.10, 0.05 and 0.01. All retain 140 workers. The script independently computes matrix OLS and HC3, obtains Student-t quantiles through Simpson integration/bisection, checks interval nesting and the nonzero-null contrast, and restores every full ResultBundle from JSON. Native outputs: model, table, plot. The [full reference](../labs/05-inference/reference.json) includes complete fitted covariance/sample state and interval figure data. Explicit export uses `run_lab(output_dir="docs/teaching/labs/05-inference")`; default runs write no artifacts. Student packets exclude this companion and reference JSON.

**Prepared input:** [wage_inference.xlsx](../labs/05-inference/wage_inference.xlsx). Use the stored observations for the published analysis. The [original generator](generators/05-inference.py) supports new dataset editions; update results and answers when changing observations.
