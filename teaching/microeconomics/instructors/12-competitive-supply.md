# Instructor companion · Lab 12

## Teaching sequence

Start by sorting the cost function into the part that varies with output and the part that does not, and ask students what the firm pays if it closes for the day. Derive the price-equals-marginal-cost condition from the profit derivative, then compare the best positive output with zero output so that the shutdown rule appears as a separate step rather than as a property of the first-order condition.

Before running, ask students what one row of the workbook is (a point on one firm's supply schedule, not an observed firm) and to predict output at prices 1, 2 and 12. Connect the `clip(lower=0)` in the code to the `max` in the supply rule before interpreting the table, and keep the three comparisons (P with MC, AVC and ATC) distinct.

## Worked exercise answers

**1.** Price 8: $q=5(8-2)=30$, revenue $240$, variable cost $2\times30+0.1\times900=150$, profit $240-150-100=-10$. Price 9: $q=35$, revenue $315$, variable cost $70+122.5=192.5$, profit $315-192.5-100=22.5$. Profit changes sign between prices 8 and 9.

**2.** $ATC(q)=100/q+2+0.1q$ is minimized where $-100/q^2+0.1=0$, so $q=\sqrt{1000}\approx31.62$ units per day. There $ATC=2+2\sqrt{10}\approx8.32$ currency per unit, and $MC=2+0.2\times31.62\approx8.32$, so $MC=ATC$ at the minimum of average total cost (marginal cost cuts average cost at its lowest point). At $P\approx8.32$ the supply rule gives $q=5(8.32-2)\approx31.62$ and profit is zero. If the fixed cost became avoidable, this is the lowest price at which the firm would choose to stay in the market: the long-run exit price.

**3.** Producing 5 units at price 1 gives revenue $5$, variable cost $10+2.5=12.5$ and profit $5-12.5-100=-107.5$, which is worse than the shutdown profit of $-100$. Because $AVC(q)=2+0.1q\ge2>1$ for every positive $q$, each unit produced costs more in variable cost than it earns, so any positive output adds to the loss.

**4.** The supply schedule is unchanged: the fixed cost does not enter marginal cost or average variable cost, so output is still $\max\{0,5(P-2)\}$ and the shutdown price is still 2. Profit at every price falls by 100; at price 12 it becomes $600-350-200=50$. Shutdown profit becomes $-200$.

**5.** Each firm supplies $\max\{0,5(P-2)\}$, so market supply is $Q^s(P)=40\times5(P-2)=200(P-2)$ for $P>2$ and zero otherwise. At price 12, market output is $200\times10=2000$ units per day. Horizontal summation multiplies quantities, not prices; the shutdown price for the market is still 2.

**6.** At price 6 the firm produces $q=20$, earns revenue $120$, pays variable cost $40+40=80$ and has profit $120-80-100=-60$. Shutting down gives $-100$. A corrected sentence: "At a price of 6 the firm loses 60 a day by producing 20 units, but it would lose 100 by closing, because the fixed cost of 100 is unavoidable in the short run and revenue more than covers the variable cost of 80; it should keep producing in the short run."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/12-competitive-supply/competitive_supply.xlsx) for the published analysis. The [original generator](generators/12-competitive-supply.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/12-competitive-supply/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/12-competitive-supply/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: no negative output, shutdown keeps fixed cost, interior foc. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A short-run supply rule does not establish long-run industry equilibrium or entry. Positive profit can attract entry if technology and market conditions permit it.
