# Instructor companion · Lab 07

## Teaching sequence

Start from the Lab 06 result that doubling $p_x$ cuts X from 16 to 8, and ask students to name the two forces behind the fall. Draw the original indifference curve, the old and new budget lines and the compensated tangency $C$ before any formula. Pose the pause question (signs and relative size of the two effects), derive $m^c$ from the expenditure function, then run the file.

Stress that the effects summing to the total is an identity; the meaningful check is equal utility at $A$ and $C$. Finish with the Hicks–Slutsky contrast in Exercises 1 and 4.

## Worked exercise answers

**1.** Slutsky income is the cost of $A=(16,36)$ at new prices: $6\times16+2\times36=168$. Then $x_S=0.4\times168/6=11.2$, so the Slutsky substitution effect is $11.2-16=-4.8$ and the income effect $8-11.2=-3.2$, still totaling −8. Compared with Hicks (−5.44 and −2.56), Slutsky assigns less to substitution and more to income. Slutsky compensation (168) exceeds Hicks compensation (158.34) because $A$ is affordable but not optimal at the new prices; the Slutsky bundle reaches utility 27.62, above the original 26.03.

**2.** $y_C=0.6\times158.34/2=47.50$, while $y_A=y_B=36$. The substitution effect on Y is $+11.50$ units (toward the relatively cheaper good) and the income effect is $-11.50$ units, for a total of 0. Lab 06's zero cross-price response is two equal and opposite effects, not an absence of any effect.

**3.** At old prices, the income giving the post-rise utility is $120\times(3/6)^{0.4}=120/1.3195=90.94$, a money loss of $120-90.94=29.06$ (the equivalent variation). The Hicks compensation implies a loss of 38.34 (the compensating variation). They differ because each values the same utility loss at different prices: at the new, higher price of X money buys less, so more is needed. With a large price change and a nonzero income effect the two measures need not agree.

**4.** At $p_x=3.3$, $x_B=0.4\times120/3.3=14.545$, a total of −1.455. Hicks: $m^c=120\times1.1^{0.4}=124.66$, $x_C=15.111$, substitution −0.889, income −0.565. Slutsky: $m^S=3.3\times16+72=124.8$, $x_S=15.127$, substitution −0.873, income −0.582. The gap between substitution effects is 0.017 units (about 1.9% of the Hicks value), against 0.644 units (about 11.8%) for the doubling. The gap shrinks about 39 times for a price change 10 times smaller, consistent with the two concepts agreeing to first order.

**5.** Total $=-5.44+2.00=-3.44$ units; X still falls when its price rises, so demand slopes downward. X would be a Giffen good if the positive income effect exceeded 5.44 units, making the total positive.

**6.** For example: "In this model, doubling the price of X from 3 to 6 lowers one synthetic Cobb–Douglas consumer's X from 16 to 8 units. Under Hicks compensation, which keeps the original utility attainable, −5.44 units are the substitution effect and −2.56 units the income effect. The split depends on the compensation concept (Slutsky gives −4.8 and −3.2) and describes the model, not observed people."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/07-substitution-income/substitution_income.xlsx) for the published analysis. The [original generator](generators/07-substitution-income.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/07-substitution-income/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/07-substitution-income/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: effects add to total, utility held fixed. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** For inferior goods the income effect can oppose substitution. This example’s signs follow from normal Cobb–Douglas demand. Compensated income is a hypothetical comparison, not an actual transfer received.
