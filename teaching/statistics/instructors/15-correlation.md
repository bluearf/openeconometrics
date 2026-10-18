# Instructor companion · Lab 15

## Teaching sequence

Show the scatterplot before any number and have students describe direction, form, spread and unusual points in words. Then identify the two blank scores and agree on the common sample of 158 before computing anything, and pose the hours-to-minutes pause question.

Build $r$ from the covariance and the two standard deviations so that its lack of units is visible, then separate three ideas students tend to merge: the strength of a correlation, the p value of a zero-correlation test, and a slope. Close by discussing two different mechanisms that could produce the same scatter.

## Worked exercise answers

**1.** The slope of score on hours is $0.652 \times 12.28/2.917 = 2.745$ points per weekly hour; the slope of hours on score is $0.652 \times 2.917/12.28 = 0.155$ hours per point. The reciprocal of 2.745 is 0.364, not 0.155: each regression minimizes errors in its own outcome, and the product of the two slopes is $r^2 = 0.425$, so they are reciprocals only when $|r| = 1$. In minutes, $r$ stays 0.652 because a positive rescaling cancels in the ratio, while the slope of score on study time becomes 2.745/60 = 0.0458 points per weekly minute.

**2.** With $n = 20$, $t = 0.652\sqrt{18/(1 - 0.425)} = 3.65$ on 18 degrees of freedom, two-sided p = 0.0018. With 158 students the 5% critical value is 1.975 on 156 degrees of freedom, so the smallest significant correlation is $r = 1.975/\sqrt{156 + 1.975^2} = 0.156$. The same strength gives very different p values at different sample sizes, and with 158 students a correlation of 0.16, which describes a weak association, would already be "significant." The p value tracks evidence against zero, not strength.

**3.** $z = 0.779$ and $SE = 0.0803$; $0.779 \pm 2.576 \times 0.0803 = 0.779 \pm 0.207$ gives 0.572 to 0.986 on the Fisher scale, and back-transforming gives 0.517 to 0.755. `oe.correlate(data, ["hours", "score"], method="pearson", pairwise=False, ci=True, alpha=0.01)` reports the same interval. It is wider because the critical value rises from 1.960 to 2.576. The lower endpoint is 0.135 below 0.652 and the upper is 0.104 above it, so the interval again extends farther toward zero.

**4.** Adding a student with 25 hours and 45 points gives $r = 0.514$ on 159 students, down from 0.652. The new row lies about 18 hours above the mean study time of 6.87 and about 13 hours beyond the largest observed value (11.93), and it has a below-average score, so its deviation product is large and negative; it also inflates the standard deviation of hours. A point far outside the range of $x$ has high leverage on $r$, which is why the scatterplot comes first.

**5.** `oe.correlate(data, ["hours", "score"], method="spearman", pairwise=False, ci=True)` gives 0.649 (interval 0.545 to 0.733), almost the same as Pearson's 0.652. Ranks discard the distances between values, so the close agreement is consistent with a roughly linear, monotone relationship without influential extreme points. The two would differ substantially for a monotone but strongly curved pattern (for example, scores that rise steeply and then level off), where Spearman's coefficient would be higher, or when one or two extreme points dominate the Pearson calculation.

**6.** For example: "Among 158 synthetic students with both measurements, weekly study hours and assessment scores have a Pearson correlation of 0.65 (95% interval 0.55 to 0.73); a straight-line fit of score on hours accounts for 42.5% of the sample variance in scores. The association is observational and does not show that studying more raises scores." Prior preparation (or motivation) could increase both hours and scores without any effect of studying itself.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/15-correlation/study_hours_scores.xlsx) for the published analysis. The [original generator](generators/15-correlation.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/15-correlation/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/15-correlation/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native correlation matches centered crossproduct, pair count matches common sample, correlation bounded. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Correlation as causation; r as a slope; pairwise denominators mixed across variables; zero correlation as independence without distributional assumptions.
