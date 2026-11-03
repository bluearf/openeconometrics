# Instructor companion · Lab 16

## Teaching sequence

Start from the counts table and have students compute the purchase share for each visitor type and overall before any test. Then calculate one expected cell by hand, connect the full sum of contributions to the native `expected` and `tests` tables, and point out that the new-visitor row alone supplies about 115 of the 135.

The central lesson with real data is the contrast between the p value ($4\times10^{-30}$) and Cramér's V (0.105): with 12,330 sessions, a weak association is detected beyond doubt. Point out that the procedure output also prints a linear-by-linear test and ordinal measures that rely on the alphabetical order of the visitor types, and that they should be ignored here. Close by separating the whole-table test from pairwise follow-ups, the association from any causal direction, and sessions from visitors.

## Worked exercise answers

**1.** Contributions: new visitors with a purchase 97.49, new without 17.85, returning with a purchase 16.22, returning without 2.97, `Other` with a purchase 0.62, `Other` without 0.11. They sum to 135.25. The two new-visitor cells drive the result (115.3 together): new visitors bought in 422 sessions where independence predicts 262.1. The `Other` row contributes little because its counts (69 and 16) are within three of their expected values (71.8 and 13.2).

**2.** Renaming `Other` to `Unknown` moves it to the end of the alphabetical order (`New_Visitor`, `Returning_Visitor`, `Unknown`). Pearson's $X^2$ (135.25) and the likelihood-ratio statistic (121.11) are unchanged because they do not depend on row order. The linear-by-linear statistic changes from 135.22 to 120.12, and the ordinal measures change too. A test whose value depends on an arbitrary label is not answering a question about visitor types.

**3.** New minus returning is $0.2491-0.1393=0.1098$. With the unpooled standard error $\sqrt{0.2491(0.7509)/1{,}694+0.1393(0.8607)/10{,}551}=0.01104$, the 95% Wald interval is $0.1098\pm1.96\times0.01104=[0.0882,\,0.1314]$. For three planned pairwise comparisons with an overall 95% level, a Bonferroni adjustment uses $z=2.394$ (the $1-0.05/6$ quantile), giving $[0.0834,\,0.1362]$. Each interval widens; comparisons involving the 85 `Other` sessions are much wider still.

**4.** Halving every count gives $X^2=67.63$ with 2 df and $p=2.1\times10^{-15}$. Cramér's V stays 0.105 and the purchase shares are unchanged. $X^2$ and the p value scale with the sample size; V and the shares describe the pattern and do not.

**5.** November (2,998 sessions): purchase shares are 0.305 for new visitors (128 of 419), 0.136 for `Other` (3 of 22) and 0.246 for returning visitors (629 of 2,557); $X^2=8.34$ on 2 df, $p=0.015$, smallest expected count 5.58, just adequate. June (288 sessions): 0.233 for new (7 of 30), 0 for `Other` (0 of 1) and 0.086 for returning (22 of 257); $X^2=6.59$ on 2 df, but the smallest expected count is 0.10, so the chi-square approximation is not adequate and an exact test is needed. In every month with more than one new-visitor session, new visitors' purchase share is above returning visitors', so the association does not disappear within months, although its size varies.

**6.** For example: "In 12,330 browsing sessions on one retailer's website, sessions by returning visitors ended in a purchase less often (13.9%) than sessions by new visitors (24.9%). The association is statistically clear but weak (Cramér's V = 0.10). The unit is the session, the file has no visitor identifier, and the data do not show that returning to the site causes a lower chance of buying."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/16-categorical-independence/visitor_purchases.xlsx) for the published analysis. Its rows are published data, not a simulation: the [generator](generators/16-categorical-independence.py) rebuilds them from the publisher's file vendored in [sources/uci-online-shoppers](sources/uci-online-shoppers/index.md) after checking its SHA-256, and nothing is downloaded. The source folder records the citation, the CC BY 4.0 licence and the retrieval. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/16-categorical-independence/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. The [LaTeX table](../labs/16-categorical-independence/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native chi-square matches the cell formula, expected margins match observed, adequate expected counts, observed cells partition. The course verifier compares fresh execution with this saved state, executes the exact student source through a disposable application worker, imports the workbook through the normal reader and reopens saved history. It also checks that `ROLES` in `lab.py` equals the `roles` in `dataset.json`, that the vendored file matches its recorded SHA-256 and that the workbook's Dictionary carries the citation and licence. Development-only independent references recompute Pearson's statistic, its p value and the smallest expected count with SciPy. These checks concern this teaching example and its published data, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A tiny p value read as a strong association; using numerical category codes as measured distances; sparse-cell asymptotics without checking counts; association as causation; sessions as independent people.
