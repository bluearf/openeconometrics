# Instructor companion · Lab 16

## Teaching sequence

Start from the counts table and have students compute the member share in each channel and overall before any test. Then calculate one expected cell by hand, show that every row has the same expected counts because each channel has 120 transactions, and connect the full sum of contributions to the native `expected` and `tests` tables.

Point out explicitly that the procedure output also prints a linear-by-linear test and ordinal measures that rely on the alphabetical order of the channels, and that they should be ignored here. Close by separating the whole-table test from pairwise follow-ups, and the association from any causal direction.

## Worked exercise answers

**1.** Store: $(93-70)^2/70 = 7.56$ and $(27-50)^2/50 = 10.58$. Web: $(72-70)^2/70 = 0.057$ and $(48-50)^2/50 = 0.080$. With the app cells, 8.93 + 12.50 + 7.56 + 10.58 + 0.057 + 0.080 = 39.70. Ranked: app members 12.50, store members 10.58, app non-members 8.93, store non-members 7.56, web members 0.080, web non-members 0.057. The two member cells for app and store drive the result: app has 25 more member transactions than independence predicts, store 23 fewer.

**2.** After renaming `Web` to `Online`, the alphabetical order becomes App, Online, Store. The Pearson statistic (39.70) and the likelihood-ratio statistic (40.76), with their df and p values, are unchanged, as is Cramér's V (0.332), because relabeling only reorders the rows. The linear-by-linear statistic jumps from 12.46 to 39.39 (df 1, p = 3.5 × 10⁻¹⁰), and the ordinal measures change too (for example gamma becomes −0.520). A statistic that depends on an arbitrary label order is not meaningful for nominal channels.

**3.** App minus store is 0.625 − 0.225 = 0.400. The unpooled standard error is $\sqrt{0.625 \times 0.375/120 + 0.225 \times 0.775/120} = 0.0584$, so the 95% interval is 0.400 ± 1.960 × 0.0584 = 0.286 to 0.514 (28.6 to 51.4 percentage points). With three planned pairwise comparisons and a Bonferroni adjustment, each interval uses the 1 − 0.05/6 normal quantile, 2.394, widening this one to 0.260 to 0.540. Each comparison then has to clear a stricter bar so that the chance of any false claim across the three stays near 5%.

**4.** Halving the counts (n = 180) halves the statistic: $X^2 = 19.85$ with 2 df, p = 4.9 × 10⁻⁵. Cramér's V stays 0.332, and the member shares and expected proportions are unchanged. $X^2$ and the p value depend on sample size; V and the shares describe the pattern itself.

**5.** Among the 150 member transactions, 50.0% were in the app, 18.0% in stores and 32.0% on the web. Among the 210 non-member transactions, 21.4% were in the app, 44.3% in stores and 34.3% on the web. "App transactions are more often by members" conditions on channel (row shares: 62.5% against 41.7% overall); "members prefer the app" conditions on membership (column shares: 50.0% of member transactions against 33.3% of all transactions). Both reflect the same departure from independence, and the chi-square statistic is the same whichever way the table is read, but they answer different questions and use different denominators.

**6.** For example: "In 360 synthetic transactions, the share made by loyalty members was 62.5% in the app, 40.0% on the web and 22.5% in stores, and channel and membership are not independent (Pearson $X^2 = 39.70$, 2 df, p < 0.001; Cramér's V = 0.33)." The data do not show that the app causes membership: members may choose the app, or a third factor such as shopping frequency may drive both. The unit is the transaction, not the customer, so frequent shoppers may account for several rows and the statement should not be phrased as a share of customers.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/16-categorical-independence/shopping_channels.xlsx) for the published analysis. The [original generator](generators/16-categorical-independence.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/16-categorical-independence/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/16-categorical-independence/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native chi square matches cell formula, expected margins match observed, adequate expected counts, observed cells partition. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Using numerical category codes as measured distances; sparse-cell asymptotics without checking counts; association as causation.
