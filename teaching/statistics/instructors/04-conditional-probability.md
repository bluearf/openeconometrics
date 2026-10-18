# Instructor companion · Lab 04

## Teaching sequence

Draw the two-by-two table on the board and have students circle a different denominator for each conditional question. Stress early that the unit is a visit, so every share is visit-weighted.

Before running, use the pause question in Section 3: students name each denominator as a count of visits. Then show that the two headline conditionals (0.626 and 0.618) are close only because the margins 235 and 238 are nearly equal; exercise 2 supplies a reversed pair that is far apart. Leave the chi-square output to Lab 16.

## Worked exercise answers

**1.** Among the 365 non-member visits, 91 ended in a purchase: $91/365=0.2493$. Among the 235 member visits the share is 0.6255. The difference is 0.3762 (about 37.6 percentage points) and the ratio is $0.6255/0.2493=2.509$. The crosstab's risk table reports the inverse ratio, non-members to members, as 0.3986.

**2.** $P(M\mid\text{no purchase})=88/362=0.2431$ and $P(\text{no purchase}\mid M)=88/235=0.3745$. By the Section 4 identity their ratio is $P(M)/P(\text{no purchase})=235/362=0.649$. Member visits (235) are much less common than no-purchase visits (362), so the shared numerator 88 is divided by very different denominators.

**3.** Expected counts are row total × column total / 600: member and purchase $235\times238/600=93.22$; member and no purchase $235\times362/600=141.78$; non-member and purchase $365\times238/600=144.78$; non-member and no purchase $365\times362/600=220.22$. The rows sum to 235 and 365 and the columns to 238 and 362, as in the observed table. Observed minus expected is $147-93.22=53.78$ visits in the member-and-purchase cell, which equals $600\times0.08964$. In a two-by-two table all four gaps have this size, with signs $+,-,-,+$ (88 − 141.78, 91 − 144.78 and 274 − 220.22).

**4.** Group the visits by customer, decide for each member customer whether they ever purchased (or compute that customer's own purchase share), and average over customers so that each counts once. If a few frequent visitors are members who buy on most visits, they dominate the visit-weighted 0.626; a customer-weighted share could then be noticeably lower. It could also be higher if frequent member visitors buy less often. This file cannot answer the question because it has no customer identifier.

**5.** No. Customers choose to join, so members may already be more engaged or frequent buyers; the association does not identify the effect of membership. A suitable study would randomly offer free membership to some customers in a defined population, follow all of them for a fixed period, and compare purchase outcomes between the offered and not-offered groups at the customer level.

**6.** For example: "Of 235 member visits, 147 (62.6%) ended in a purchase. Of 238 purchase visits, 147 (61.8%) were by members." The original reasoning is wrong because $P(M\mid P)=P(P\mid M)P(M)/P(P)$; the two shares are close only because member visits and purchase visits are nearly equally common, not because one follows from the other.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/04-conditional-probability/customer_events.xlsx) for the published analysis. The [original generator](generators/04-conditional-probability.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/04-conditional-probability/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/04-conditional-probability/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: four cells partition visits, conditional times marginal equals joint, bayes reverses denominator. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Reversing a conditional without changing the denominator; assuming independence; confusing visit and customer weights; reading association as a membership treatment effect.
