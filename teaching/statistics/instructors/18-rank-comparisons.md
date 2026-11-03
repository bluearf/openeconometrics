# Instructor companion · Lab 18

## Teaching sequence

Start from the score table and convert it to shares, because the colours differ in size (1,599 red, 4,898 white). Have students find both medians and quartiles before any test: they are identical (5, 6, 6). Then define $U$ as wins plus half ties, show that the midrank sum gives the same number, and run the native test.

Use the pause question in Section 3 to justify the half-credit rule. The central lesson with these data is that a decisive rank-sum test (z = −9.91) coexists with equal medians and quartiles, which is exactly the point of Divine et al. (2018). Keep the effect size in front of the p value: $\hat P=0.423$ and $r=-0.12$ describe a modest shift that only looks dramatic because of the sample size.

## Worked exercise answers

**1.** Red: cumulative counts are 10, 63, 744 and 1,382 at scores 3–6, so the 400th ordered score is a 5 and the 800th and 1,200th are 6s: the lower quartile is 5, the median 6 and the upper quartile 6. White: cumulative counts 20, 183, 1,640 and 3,838 at scores 3–6 give the same 5, 6 and 6. On a coarse seven-point scale the quartiles can only take a few values, so they land in the same categories even though white wines have fewer fives (29.7% against 42.6%) and more sevens and eights. The rank-sum statistic compares whole distributions pair by pair, so it detects that shift; calling it a median test would be wrong here, where the medians are equal and the test still rejects.

**2.** $U_W=16{,}518{,}039-4{,}898\times4{,}899/2=16{,}518{,}039-11{,}997{,}651=4{,}520{,}388$. Then $U_R+U_W=3{,}311{,}514+4{,}520{,}388=7{,}831{,}902=n_Rn_W$. A random white wine outscores a random red wine, counting ties as half, with estimated probability $4{,}520{,}388/7{,}831{,}902=0.5772$, the complement of 0.4228.

**3.** Without the tie correction, $\operatorname{Var}(U_R)=7{,}831{,}902\times6{,}498/12=4{,}240{,}974{,}933$, a standard deviation of 65,123, so $z=(3{,}311{,}514-3{,}915{,}951)/65{,}123=-9.28$ and the two-sided p value is $1.7\times10^{-20}$. The tie correction reduces the variance to $3.717\times10^9$, so it moves z further from zero (to −9.91). Ties remove possible rank arrangements and so shrink the null spread of the rank sum; ignoring them makes the test conservative.

**4.** Scoring ties as losses for red gives $2{,}020{,}689/7{,}831{,}902=0.2580$; scoring them as wins gives $(2{,}020{,}689+2{,}581{,}650)/7{,}831{,}902=0.5876$. The half weight gives 0.4228. With a third of all pairs tied, the convention moves the answer by more than 0.3. Half is the only weight that treats the colours symmetrically: it makes the red-wins and white-wins probabilities sum to 1 and equal 0.5 when the distributions are identical.

**5.** Before the recode, mean scores are 5.6360 (red) and 5.8779 (white), a difference of 0.2419. After recoding 8 as 10 and 9 as 12, they are 5.6585 and 5.9524, a difference of 0.2939. The rank-sum results do not change: $U_R=3{,}311{,}514$, z = −9.914 and p = $3.6\times10^{-23}$, because the recode keeps every wine in the same order. Mapping 8 to 12 instead would place eights above nines and change $U$; only an increasing recode is harmless.

**6.** Equal medians do not imply equal distributions. The test detected that white wines tend to outscore red wines: a random red wine scores higher only 42.3% of the time, counting ties as half, mainly because whites have more sevens and eights and fewer fives. For example: "A random red wine outscored a random white wine with estimated probability 0.423, counting ties as half (U = 3,311,514 of 7,831,902 pairs). The shift is modest, both colours have median 6, and with 6,497 wines it is far beyond chance (tie-corrected z = −9.91, p < 0.001)."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/18-rank-comparisons/wine_quality.xlsx) for the published analysis. Its rows are published data, not a simulation: the [generator](generators/18-rank-comparisons.py) stacks the publisher's red and white files vendored in [sources/uci-wine-quality](sources/uci-wine-quality/index.md) after checking their SHA-256 values, and nothing is downloaded. The source folder records the citation, the CC BY 4.0 licence and the retrieval. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/18-rank-comparisons/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. The [LaTeX table](../labs/18-rank-comparisons/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: native U matches the pair count from the score table, the rank-sum identity, native probability matches the pair count, all ranks sum correctly, counts cover every wine. The course verifier compares fresh execution with this saved state, executes the exact student source through a disposable application worker, imports the workbook through the normal reader and reopens saved history. It also checks that `ROLES` in `lab.py` equals the `roles` in `dataset.json`, that the vendored files match their recorded SHA-256 values and that the workbook's Dictionary carries the citation and licence. Development-only independent references recompute the midrank U, the tie-corrected p value and the probability of superiority with SciPy. These checks concern this teaching example and its published data, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A rank test read as a median test; a tiny p value read as a large difference; ignoring ties; undocumented continuity correction; ordinal scores as cardinal distances; a comparison of separately sampled wines read as a causal effect of colour.
