# Instructor companion · Lab 04

## Teaching sequence

Ask students to read one row as a point on a stipulated demand schedule, not an observed sale, and to compare the same 2-unit drop in quantity at a price of 1 (118 units) and at 58 (4 units) before introducing the elasticity formula.

Use the pause question before running: students who say a constant slope means a constant elasticity are confusing the two measures. Derive $dR/dP=Q(1+\varepsilon)$ with its two-term interpretation (gain on units still sold, loss on units no longer sold) before interpreting the table, then link the symmetric revenue hill to the inelastic and elastic halves of the line.

## Worked exercise answers

**1.** At $P=10$: $Q=100$, $\varepsilon=-20/100=-0.2$, inelastic, and $dR/dP=100\times0.8=80>0$, so a small price rise increases revenue. At $P=40$: $Q=40$, $\varepsilon=-80/40=-2$, elastic, and $dR/dP=40\times(-1)=-40$, so revenue falls. At $P=50$: $Q=20$, $\varepsilon=-100/20=-5$, elastic, and $dR/dP=20\times(-4)=-80$, so revenue falls.

**2.** From 20 to 21, revenue moves from 1,600 to $21\times78=1{,}638$, a rise of 38, against $dR/dP=40$ at 20. From 40 to 41, revenue moves from 1,600 to $41\times38=1{,}558$, a fall of 42, against $dR/dP=-40$ at 40. Both exact changes are 2 below the derivative because revenue is a downward-bending parabola: for a one-unit step the exact change is $dR/dP+\tfrac12(-4)(1)^2=dR/dP-2$.

**3.** Setting $-P/(60-P)=-3$ gives $P=180-3P$, so $P=45$. There $Q=30$ units per day, $R=45\times30=1{,}350$ currency units per day and $dR/dP=120-180=-60$, which equals $Q(1+\varepsilon)=30\times(-2)$. The point lies on the falling right side of the revenue hill, at the same height as the point at $P=15$.

**4.** The quantity changes by $(40-80)/60=-0.667$ of its average and the price by $(40-20)/30=0.667$ of its average, so the arc elasticity is $-1.0$. The point elasticities are $-0.5$ at 20 and $-2$ at 40; the arc measure summarizes the whole interval. An arc elasticity of exactly −1 matches the fact that revenue is 1,600 at both ends: with midpoint bases, the arc elasticity equals −1 precisely when revenue is the same at the two prices.

**5.** Profit is $\pi=(P-10)(120-2P)$, so $d\pi/dP=120-2P-2(P-10)=140-4P=0$ gives $P=35$, $Q=50$ units per day and $\pi=25\times50=1{,}250$ currency units per day. Revenue there is 1,750, below the maximum of 1,800, but profit at the revenue-maximizing price of 30 is only $20\times60=1{,}200$. A positive marginal cost pushes the best price above 30. Equivalently, marginal revenue in quantity terms, $60-Q$, equals marginal cost 10 at $Q=50$.

**6.** For example: "At a price of 20 currency units the point elasticity is −0.5, so a small price increase raises revenue. The elasticity changes along the schedule: revenue peaks at 1,800 currency units per day at a price of 30, where demand is unit elastic, and falls at higher prices. The calculation ignores production costs, so it does not identify the profit-maximizing price."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/04-elasticity-revenue/elasticity_revenue.xlsx) for the published analysis. The [original generator](generators/04-elasticity-revenue.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/04-elasticity-revenue/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/04-elasticity-revenue/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: interior revenue foc, elasticity uses local quantity. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A local elasticity need not predict a large price change. Revenue excludes production costs, so its maximum is not a general recommendation for a firm’s price.
