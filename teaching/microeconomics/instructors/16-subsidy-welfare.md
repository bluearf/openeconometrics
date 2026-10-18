# Instructor companion · Lab 16

## Teaching sequence

Start with the two schedules as marginal benefit and marginal cost, and have students find the unsubsidized crossing (50 units, price 35, total surplus 1,250) before any policy appears. Then introduce the subsidy as a wedge between the buyer price and the seller receipt, and ask students to locate the row of `Data` where the supply price exceeds the demand price by 10 before running the file.

After running, separate the three questions the handout keeps apart: who gains (incidence from the price changes), what the government pays (a transfer) and what is lost (the triangle over the ten extra units). Close with the externality exercise so students see that the sign of the welfare effect depends on the no-externality benchmark.

## Worked exercise answers

**1.** With $s=6$, $Q_s=50+6=56$; the row `quantity = 56` shows a demand price of 32 and a supply price of 38, a wedge of 6. Consumer surplus is $\tfrac12(60-32)\times56=784$, producer surplus $\tfrac12(38-10)\times56=784$ and government expense $6\times56=336$ currency/day. Net total surplus is $784+784-336=1{,}232$, so the deadweight loss is $1{,}250-1{,}232=18$, which equals $\tfrac12\times6\times(56-50)$.

**2.** Without the subsidy, consumer and producer surplus are each $\tfrac12\times25\times50=625$. With the 10 subsidy each is 900, so each rises by 275. The gains plus the loss are $275+275+50=600$, exactly the government expense. The identity must hold because total surplus falls by the deadweight loss while the subsidy is a transfer: $(CS_s+PS_s-G)-(CS^*+PS^*)=-DWL$ rearranges to $G=\Delta CS+\Delta PS+DWL$. Geometrically, the 600 rectangle splits into the buyers' gain area, the sellers' gain area and the loss triangle.

**3.** The deadweight loss is 12.5 for $s=5$ ($Q=55$), 50 for $s=10$ ($Q=60$) and 200 for $s=20$ ($Q=70$) currency/day. Doubling the subsidy quadruples the loss, because $DWL=\tfrac12 s(Q_s-50)$ and $Q_s-50=s$ here, so $DWL=s^2/2$: both the height and the base of the triangle double.

**4.** With supply $10+1.5Q$ and no subsidy, $60-0.5Q=10+1.5Q$ gives $Q=25$ and $P=47.5$. With $s=10$, $10+1.5Q=70-0.5Q$ gives $Q=30$, a buyer price of 45 and a seller receipt of 55. Buyers gain 2.5 per unit (25% of the subsidy) and sellers 7.5 (75%), matching $b/(b+d)=0.5/2=0.25$ and $d/(b+d)=1.5/2=0.75$. The steeper, less responsive supply side captures more. (Total surplus falls from 625 to 600, a loss of 25.)

**5.** With a constant external benefit of 10, marginal social benefit is $70-0.5Q$ and the efficient quantity solves $70-0.5Q=10+0.5Q$, so $Q=60$: the 10 subsidy is exactly corrective. Social surplus $\int_0^Q(60-q)\,dq$ is 1,800 at 60 units against 1,750 at 50, so the subsidy raises welfare by 50 and leaves no deadweight loss. With an external benefit of 4, the efficient quantity is 54 and social surplus $\int_0^Q(54-q)\,dq$ is 1,458 there, 1,450 at 50 units (no policy, loss 8) and 1,440 at 60 units (subsidy, loss 18). The 10 subsidy overshoots and is worse than doing nothing; a subsidy of 4 would be corrective.

**6.** For example: "In this stipulated market, a subsidy of 10 per unit lowers the buyer price from 35 to 30 and raises the seller receipt from 35 to 40, so buyers and sellers each gain 275 currency units a day. Taxpayers pay 600 a day, so net total surplus falls by 50, from 1,250 to 1,200, because the subsidy pays for ten units whose cost exceeds their value. The subsidy is not good for everyone once its funding is counted, unless production has external benefits the schedules omit."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/16-subsidy-welfare/subsidy_welfare.xlsx) for the published analysis. The [original generator](generators/16-subsidy-welfare.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/16-subsidy-welfare/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/16-subsidy-welfare/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: subsidy wedge, overproduction, fiscal cost subtracted. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The conclusion depends on the no-externality benchmark. Positive spillovers could justify subsidizing some production, while costly public finance could make the fiscal burden larger.
