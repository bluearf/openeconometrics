# Instructor companion · Lab 08

## Teaching sequence

Begin with the affordability idea before any notation: a bundle chosen when another was affordable is revealed preferred to it. Have students cost bundle 3 at choice 1's prices by hand, then pose the pause question about choice 2 and bundle 1, which lands on the exact tie at 120. Build the full cost matrix, read it row by row, and only then introduce the strict reverse condition and its difference from standard WARP and from GARP.

Emphasize that the lab's violation count uses a strict reverse inequality (the two-bundle GARP condition), which is slightly weaker than standard WARP. On the supplied data both give zero violations; Exercise 2 shows where they diverge.

## Worked exercise answers

**1.** The four direct relations and their reverse margins $p^j\cdot x^i-m_j$ are: choice 1 over bundle 3, $140-110=30$; choice 2 over bundle 1, $104-100=4$; choice 2 over bundle 3, $136-110=26$; choice 4 over bundle 3, $138-110=28$. The closest to a violation is choice 2 over bundle 1: bundle 2 would have been affordable on occasion 1 with 4 more currency units.

**2.** With bundle 2 = (30, 15), the bundle-2 column of the cost matrix becomes 90, 120, 150 and 180. Direct comparisons rise to 5 (bundle 2 is now affordable on occasion 1 at 90). The lab reports 1 strict violation: on occasion 2 bundle 1 was affordable (120 of 120) while on occasion 1 bundle 2 was strictly affordable (90 of 100). Standard WARP reports 2 ordered pairs, because it also flags occasion 1 over bundle 2, whose reverse cost (bundle 1 at occasion 2's prices) equals the budget of 120 exactly. Either way the two choices contradict stable, monotone preferences.

**3.** The direct relations are 1→3, 2→1, 2→3 and 4→3. The only chain is 2→1→3, which gives 2→3, already a direct relation. Bundle 3 has no outgoing relation and no relation points to 2 or 4, so there is no cycle. No contradiction appears, and these data satisfy GARP, although the lab does not compute this.

**4.** The X budget shares are $40/100$, $48/120$, $44/110$ and $60/150$, all equal to 0.4. A constant expenditure share across different prices and budgets is what Cobb–Douglas preferences with $\alpha=0.4$ predict. Four choices still cannot prove it: other preferences whose indifference curves touch these four budget lines at the same points rationalize the same data, and choices at other prices could differ.

**5.** Bundle 3 is ranked below every other bundle (by direct relations from 1, 2 and 4, and by dominance against 2 and 4), and bundle 2 is ranked above bundle 1. The pairs 1–4 and 2–4 remain unranked: neither dominates the other and neither was affordable when the other was chosen (bundle 4 costs 114 at occasion 1's prices and 126 at occasion 2's; bundles 1 and 2 cost 160 and 152 at occasion 4's). For example, at prices (3, 2) with budget 126, bundles 1, 2 and 4 cost 120, 120 and 126. A choice of bundle 4 there would reveal it preferred to both 1 and 2.

**6.** For example: "The four synthetic choices, each spending its full budget, contain no strict two-bundle revealed-preference contradiction. They are consistent with maximizing some stable, monotone preference, but they do not prove utility maximization, because later choices could contradict them, and they do not identify a unique utility function."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/08-revealed-preference/revealed_preference.xlsx) for the published analysis. The [original generator](generators/08-revealed-preference.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/08-revealed-preference/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/08-revealed-preference/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: chosen bundles exhaust budget, no strict two choice cycle. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** No observed violation does not identify unique preferences or prove consistency in unobserved budgets. A full GARP analysis would additionally examine transitive revealed-preference chains.
