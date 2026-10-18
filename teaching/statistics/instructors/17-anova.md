# Instructor companion · Lab 17

## Teaching sequence

Start from the three format means and the within-format ranges, and ask whether a 3.56-point gap is large. Build the variance decomposition by hand from the group means and standard deviations before running anything, then connect $MS_B/MS_W$ to the F reference distribution.

Pose the pause question before students open the pairwise table: can the omnibus test reject when not every pair differs? Then read the Bonferroni table, and only after that the variance checks and Welch test, so the assumption discussion is tied to a result that students already understand.

## Worked exercise answers

**1.** Dividing each Bonferroni p value by 3 gives unadjusted p values of 0.270 for A–B, 0.0028 for A–C and 0.054 for B–C. Only A–C is below 0.05 with or without adjustment, so the adjustment changes no conclusion here. B–C is just above 0.05 even unadjusted; students who expected it to "become significant" without adjustment should see that it does not.

**2.** $SS_B-(G-1)MS_W = 292.86-2\times 30.69 = 231.47$. Epsilon squared is $231.47/4344.12 = 0.0533$ and omega squared is $231.47/(4344.12+30.69) = 0.0529$, both matching the `effect_sizes` table and both below η² = 0.0674. They subtract $(G-1)MS_W$, the amount $SS_B$ is expected to reach from sampling noise alone when the population means are equal, so they correct η²'s upward bias.

**3.** Adding a constant to one group leaves every within-group deviation unchanged, so $SS_W$ stays 4,051.27 and $MS_W$ stays 30.69. C's mean becomes 72.07, moving further from A and B, so $SS_B$ rises from 292.86 to 762.89 and F rises from 4.77 to 12.43 (p = 0.0000113, about 1.1 × 10⁻⁵). η² rises from 0.067 to 0.158; $SS_T$ becomes 4,814.16.

**4.** The Brown–Forsythe statistic is $SS_B/\sum_g(1-n_g/n)s_g^2$. With equal sizes, $1-n_g/n = 2/3$ for every group and the classical $MS_W$ equals $\sum_g s_g^2/3$, so the denominator is $(G-1)MS_W$ and the statistic equals F exactly (denominator 61.38 = 2 × 30.69). Its denominator degrees of freedom come from a Satterthwaite approximation that reaches 132 only when the three sample variances are equal; here they differ (20.54, 28.37 and 43.17 squared points), giving 120.72. With 30, 45 and 60 trainees the classical pooled variance weights each group by $n_g-1$, while Brown–Forsythe weights by $1-n_g/n$, giving smaller groups relatively more weight, so the statistics would differ. If the larger group also had the larger variance, the classical test would tend to be conservative; if the smaller group had the larger variance, it would tend to reject too often.

**5.** Five formats give $\binom{5}{2} = 10$ pairs. To stay under 0.05 after Bonferroni adjustment a pair needs an unadjusted p below 0.05/10 = 0.005. An unadjusted 0.008 becomes 10 × 0.008 = 0.08, which is not significant at 5%.

**6.** For example: "In these 135 synthetic trainees, the three format means are not all equal (F(2, 132) = 4.77, p = 0.010). After Bonferroni adjustment, format C's mean is 3.56 points higher than format A's (adjusted 95% interval 0.73 to 6.40), but C is not distinguishable from B, and formats explain only about 7% of the variation in scores." The original sentence wrongly treats a non-directional omnibus test as proof of a ranking and ignores that C and B do not differ after adjustment.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/17-anova/training_formats.xlsx) for the published analysis. The [original generator](generators/17-anova.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/17-anova/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/17-anova/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: sum of squares decomposition, native F matches mean square ratio, native eta squared matches ss ratio, between df plus within df. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Omnibus rejection as every pair differing; ignoring multiple comparisons; F as a directional effect; η² as causal attribution.
