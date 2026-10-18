# Instructor companion · Lab 18

## Teaching sequence

Open with the count table by service and ask what "average rating" would assume about the distances between categories. Count wins and ties in a small tied example on the board, then show that pooled midranks give the same $U$ as the pair count.

Pose the pause question about half credit for ties before running. After running, read $\hat P$ before the p value, then discuss the tie-corrected variance and why the exact no-tie distribution is not used. Use the grouped bar chart to show how much the two services' distributions overlap before reading the test.

## Worked exercise answers

**1.** Both medians are 4. A's cumulative counts are 2, 7, 13 and 25 at ratings 1–4, and B's are 2, 5, 12 and 23, so the 20th and 21st ratings in each service are 4. Equal medians can coexist with $\hat P = 0.458$ because the rank comparison uses all 1,600 pairs: B has 17 of 40 customers at 5 or above against A's 15, and 12 below 4 against A's 13. The test responds to the whole ordering, so calling it a median test would need an equal-shape assumption that these tied distributions do not support.

**2.** $U_B = W_B - n_B(n_B+1)/2 = 1687 - 820 = 867$. Then $U_A + U_B = 733 + 867 = 1600 = 40 \times 40$. The probability that a random B customer outrates a random A customer, counting ties as half, is $867/1600 = 0.542$, which is $1 - 0.458$.

**3.** Without the tie correction, $\operatorname{Var}(U_A) = 1600 \times 81/12 = 10{,}800$, standard deviation 103.92, $z = -67/103.92 = -0.645$ and two-sided p = 0.519. The correction lowers the variance to 10,305.3, so it makes $|z|$ slightly larger (0.660) and p slightly smaller (0.509). Ties restrict the possible rank sums, which reduces their spread under the null; ignoring that overstates the variance.

**4.** With 574 wins and 318 ties, scoring ties as losses gives $574/1600 = 0.359$ and scoring them as wins gives $892/1600 = 0.558$. When the two services have the same distribution, $P(A>B) = P(B>A)$, so wins plus half the ties equal exactly 0.5 whatever the tie rate. Scoring ties as 0 or 1 would move the null value with the number of ties, and half credit is also what midranks imply.

**5.** The mean rating difference, B minus A, is $4.175 - 3.950 = 0.225$ before the recode and $4.25 - 3.95 = 0.30$ after it, because the only rating of 7 is a B customer. The rank-sum results do not change: $U_A = 733$, $z = -0.660$ and p = 0.509, since an order-preserving recode leaves every rank and midrank unchanged. The mean depends on the arbitrary spacing of the codes; the rank test does not.

**6.** For example: "Among 80 synthetic customers, a random service A customer outrated a random service B customer with estimated probability 0.458, counting ties as half; the tie-corrected rank-sum test does not reject equal rating distributions (two-sided p = 0.51). The data do not show a difference, but with 40 customers per service they cannot rule out a moderate one." A p value above 0.05 does not show that the services are rated the same.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/18-rank-comparisons/customer_ratings.xlsx) for the published analysis. The [original generator](generators/18-rank-comparisons.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/18-rank-comparisons/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/18-rank-comparisons/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native U matches pair enumeration, rank sum identity, native probability matches pair comparison, all ranks sum correctly. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Ignoring ties; a rank test as universally a median test; undocumented continuity correction; ordinal categories as cardinal distances.
