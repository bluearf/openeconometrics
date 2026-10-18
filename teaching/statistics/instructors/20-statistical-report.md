# Instructor companion · Lab 20

## Teaching sequence

Before anyone runs code, have students write the analysis plan: primary outcome and contrast direction, secondary outcome, the family for multiplicity, and the sample rule for blank spending cells. Then build the denominator table (240 randomized, 234 with spending, 6 missing, all in B) and pose the pause question about whether randomization still protects the retained spending sample.

Run the primary and secondary comparisons, apply Holm by hand, and finish with the missing-data sensitivity check (Exercise 2) before groups write their own result paragraph with raw and retained counts, contrast direction, interval, multiplicity convention and missingness limitation.

## Worked exercise answers

**1.** The return rates are $38/120 = 0.3167$ and $35/120 = 0.2917$. The unpooled standard error is $\sqrt{0.3167 \times 0.6833/120 + 0.2917 \times 0.7083/120} = 0.0594$, so the interval is $0.025 \pm 1.96 \times 0.0594$, or −0.091 to 0.141, matching the `prtest` output. The data are compatible with return rates anywhere from about 9 percentage points higher under the discount to about 14 points higher under the standard offer, so they say little about the discount's effect on returns.

**2.** Filling the six blanks with the lowest observed B spending (5.20) gives a B mean of 18.76, a difference of −1.44, Welch interval −2.99 to 0.10 and p = 0.067. Filling them with the highest observed B spending (39.70) gives a B mean of 20.48, a difference of −3.17, interval −4.82 to −1.52 and p = 0.0002. The primary conclusion at the 5% level does not survive the pessimistic case, so the complete-case result depends on the assumption that the six missing values are unrelated to spending. (Filling with zero, below every observed B value, gives −1.18 with p = 0.155.)

**3.** Bonferroni gives $2 \times 0.0046 = 0.0092$ for spending and $\min(1, 2 \times 0.674) = 1$ for returns; Holm gives 0.0092 and 0.674. Holm's $j$th adjusted value is $\max_{k \le j}(m-k+1)p_{(k)} \le \max_{k \le j} m\,p_{(k)} = m\,p_{(j)}$, the Bonferroni value, and capping at 1 keeps the inequality. Only the smallest p value receives the full multiplier $m$.

**4.** $-2.158/17.313 = -0.125$: mean spending under the discount is 12.5% above mean spending under the standard offer. The pooled standard deviation is $\sqrt{(119 \times 5.421^2 + 113 \times 6.072^2)/232} = 5.748$, so $d = -2.158/5.748 = -0.375$, matching `cohens_d` in the `effect_sizes` table (Hedges' g is −0.374).

**5.** For example: "The prespecified primary outcome, return visits, showed no clear difference between offers (31.7% under A and 29.2% under B; p = 0.674). Spending, a secondary outcome, was higher under the discount (Holm-adjusted p = 0.0092) and should be reported as a secondary finding." Switching the primary outcome after seeing that spending was significant is outcome switching: it turns a search across outcomes into an apparently planned test, so the nominal p value overstates the evidence.

**6.** Problems: n = 240 is wrong for spending, which uses 234 customers; "raised" states a causal effect without the assumption about the six missing values; return visits were lower under the discount (A − B = +0.025), so "increased" is wrong; "though not significantly" presents a non-significant difference as a trend; there is no interval and no label for raw versus adjusted p values. Rewrite: "Among the 234 customers with recorded spending, the discount group spent 2.16 currency units more on average (95% interval 0.67 to 3.64; p = 0.0046, Holm-adjusted 0.0092), assuming the six missing values are unrelated to spending. Return rates for all 240 customers were 31.7% under the standard offer and 29.2% under the discount (p = 0.674), giving no clear evidence about returns."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/20-statistical-report/campus_cafe_offer.xlsx) for the published analysis. The [original generator](generators/20-statistical-report.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/20-statistical-report/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/20-statistical-report/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: primary welch se, primary t uses same rows, sample accounting, holm does not reduce p, distinct customer ids. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Randomization as a cure for differential missingness; secondary outcomes sharing the primary denominator; switching the primary outcome after inspecting p values; adjusted and raw p values left unlabeled.
