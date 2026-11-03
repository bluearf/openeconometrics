# Instructor companion · Lab 05

## Teaching sequence

The data are published public-health results, so frame the lab as the evaluation of a test, not as individual medical advice. Start with the overall confusion matrix and count the routes into a positive result (39 true, 16 false) before presenting the Bayes algebra, naming each rate's denominator as a row or a column of the matrix. Stress early that the unit is a swab pair and that RT-PCR is a reference standard, not certain truth.

Before running, use the pause question in Section 4: could most positives be false in the asymptomatic group even with specificity above 98%? Then split by symptom status (Section 6). The key move is the counterfactual 0.843: the asymptomatic test rates at the symptomatic prevalence. It shows that most, but not all, of the PPV gap is the base rate, because sensitivity also differs between groups. Exercise 3 shows that the split depends on which group's rates are held fixed.

## Worked exercise answers

**1.** Overall, the NPV is $1{,}025/1{,}043=0.9827$ and the share of negative results that RT-PCR found positive is $18/1{,}043=0.0173$. Without symptoms the NPV is $840/850=0.9882$ (miss share 0.0118); with symptoms it is $185/193=0.9585$ (miss share 0.0415). Each divides by the antigen-negative pairs of its group. The NPV informs whether a negative result can be accepted without confirmation: about 1 negative in 85 is RT-PCR positive without symptoms, but about 1 in 24 with symptoms, which is why the report suggests confirming negative results in symptomatic people.

**2.** Of 10,000 people at 1% prevalence, 100 are infected; the test detects $100\times0.4118=41.2$ of them and misses 58.8. The other 9,900 are uninfected, and $9{,}900\times(14/854)=9{,}900\times0.01639=162.3$ of them test positive. The PPV is $41.2/(41.2+162.3)=41.2/203.5=0.2024$, matching the formula.

**3.** With the symptomatic sensitivity (0.800) and specificity (0.9893) at the asymptomatic prevalence (17/871 = 0.0195), the PPV is 0.5982. Holding prevalence at the asymptomatic level, the better symptomatic test rates raise the PPV from 0.333 to 0.598; holding the symptomatic test rates, the higher prevalence raises it from 0.598 to 0.941. Section 6 held the asymptomatic test rates fixed instead and attributed 0.333 to 0.843 to prevalence and 0.843 to 0.941 to the test rates. The two splits differ because prevalence and test rates interact in the formula; neither order is the "true" decomposition.

**4.** At prevalence 0.0195, raising specificity to 0.995 gives a PPV of 0.6211; raising sensitivity to 0.80 gives only 0.4928. The false-positive route is $(1-\text{spec})(1-p)$ and comes from the 98% of pairs that are uninfected: cutting the false-positive rate from 1.64% to 0.5% shrinks that route from 0.0161 to 0.0049. Raising sensitivity can at most double the small true-positive route, from 0.0080 to 0.0156.

**5.** Removing 8 of the 14 asymptomatic false positives leaves 846 RT-PCR-negative pairs, of which 840 test negative: specificity $840/846=0.9929$. The PPV becomes $7/13=0.5385$ instead of 0.333. With only 7 true positives in the group, the PPV denominator is small, so a handful of false results moves it a great deal. The report kept the eight results because no user error was found; the exercise shows how much the asymptomatic PPV depends on that decision.

**6.** "98% specific" describes uninfected swabs: 840 of 854 RT-PCR-negative pairs tested negative. It is not the probability that a positive result is right, which is the PPV, 0.333 without symptoms. Overall accuracy, $(39+1{,}025)/1{,}098=0.969$, is a third, different number. A correct version: "Among people without symptoms, the antigen test was negative for 98.4% of RT-PCR-negative swabs, but only 7 of its 21 positive results (33%) were confirmed by RT-PCR, because infection was rare in that group (2.0% of swabs)."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/05-bayes/antigen_tests.xlsx) for the published analysis. Its rows come from published counts, not a simulation: the [generator](generators/05-bayes.py) expands the eight Table 2 counts vendored in [sources/mmwr-antigen-2021](sources/mmwr-antigen-2021/index.md) after checking the file's SHA-256, and nothing is downloaded. The source folder records the citation, the public-domain statement and the date the table was read. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/05-bayes/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. The [LaTeX table](../labs/05-bayes/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: confusion matrix partition, groups partition every cell, Bayes matches the positive denominator overall and in each group, lower base rate lowers PPV. The course verifier compares fresh execution with this saved state, executes the exact student source through a disposable application worker, imports the workbook through the normal reader and reopens saved history. It also checks that `ROLES` in `lab.py` equals the `roles` in `dataset.json`, that the vendored file matches its recorded SHA-256 and that the workbook's Dictionary carries the citation and licence. Development-only independent references recompute the overall and group PPVs from selected rows and check that the expanded rows reproduce the eight published counts. These checks concern this teaching example and its published table, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Sensitivity or specificity read as predictive value; ignoring the base rate; attributing the whole PPV gap to prevalence when sensitivity also differs; treating RT-PCR as certain truth; counting swab pairs as independent people.
