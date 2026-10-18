# Instructor companion · Lab 15

## Teaching sequence

Start from the untaxed equilibrium in row 50 and ask who "pays" a tax that sellers remit before doing any algebra. Write the wedge condition, solve for the taxed quantity, and read both prices off the original schedules; then separate the legal remitter from the change in each side's price.

Before running, ask students what one row is (a potential unit of trade on stipulated schedules, not an observed transaction) and to verify the wedge at the taxed quantity. Connect the row search on `gap == tax` to the wedge equation, and require revenue to appear in the surplus accounting before the deadweight loss is computed.

## Worked exercise answers

**1.** On $Q_D=2(60-P)$ and $Q_S=2(P-10)$, $dQ_D/dP=-2$ and $dQ_S/dP=2$. At $(Q,P)=(50,35)$: $\varepsilon_D=-2\times35/50=-1.4$ and $\varepsilon_S=2\times35/50=1.4$. The buyers' share is $\varepsilon_S/(\varepsilon_S-\varepsilon_D)=1.4/2.8=0.5$, so buyers bear $0.5\times10=5$ per unit, matching the computed buyer burden.

**2.** With $t=20$: quantity 30, buyer price 45, seller net price 25, each burden 10, revenue $20\times30=600$, consumer and producer surplus 225 each, deadweight loss $\tfrac12\times20\times20=200$. Doubling the tax raises revenue by half (400 to 600) but quadruples the deadweight loss (50 to 200). The loss triangle has base and height both proportional to $t$, so it grows with $t^2$, while revenue $t(50-t)$ grows less than proportionally because the tax base shrinks.

**3.** With inverse supply $10+1.5q$, the untaxed point is $q=25$, $P=47.5$. With the 10-unit tax, $60-0.5q-(10+1.5q)=10$ gives $q=20$, buyer price 50 and seller net price 40. Buyers bear $50-47.5=2.5$ and sellers $47.5-40=7.5$ per unit. Revenue is 200, consumer surplus 100, producer surplus 300 and deadweight loss 25. Sellers bear three quarters because supply is now steeper (less responsive) than demand: $b_S/(b_D+b_S)=1.5/2=0.75$.

**4.** If buyers remit, sellers face buyers' willingness to pay net of the tax, $60-0.5q-10=50-0.5q$. Setting this equal to $10+0.5q$ gives $q=40$ and a price to sellers of 30. Buyers hand over $30+10=40$ per unit in total. Quantity, buyer cost and seller net price are identical to the seller-remitted case: the wedge, not the remitter, determines incidence.

**5.** Without the tax, $CS=PS=\tfrac12\times25\times50=625$. Consumer surplus falls by $625-400=225$: buyers pay $5\times40=200$ of the tax on the units still traded and lose $\tfrac12\times10\times5=25$ on units 41 to 50. Producer surplus falls by the same 225, split into 200 of tax and 25 of loss. Revenue is $200+200=400$, and the deadweight loss is $25+25=50$.

**6.** For example: "The tax lowered buyers' and sellers' combined surplus by 450 a day, from 1,250 to 800. Of that, 400 went to the government as revenue and is a transfer; the remaining 50 is a deadweight loss from the ten units no longer traded."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/15-tax-incidence/tax_incidence.xlsx) for the published analysis. The [original generator](generators/15-tax-incidence.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/15-tax-incidence/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/15-tax-incidence/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: tax wedge, government included. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Equal incidence follows from the equal slopes here. It is not a general half-and-half rule, and statutory remittance alone cannot identify the economic burden.
