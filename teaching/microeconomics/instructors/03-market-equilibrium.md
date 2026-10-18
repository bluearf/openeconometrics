# Instructor companion · Lab 03

## Teaching sequence

Ask students to say what one row is (a price scenario on two stipulated schedules, not a recorded trade) and to recover both slopes and intercepts from the first two rows before writing any equation. Then have them solve the clearing condition by hand, as the pause question asks, and check it against the workbook row at a price of 35.

Before discussing the shift, ask students to predict the new quantity. Most will say 70; use the decomposition at the old price (excess demand of 20, closed by a price rise of 5) to show where the other 10 units go. Connect the excess-demand chart to the formula $Z(P)=140-4P$ before interpreting the table.

## Worked exercise answers

**1.** At a price of 45 the workbook shows demand 30 and supply 70, so excess demand is $30-70=-40$ units per day; the formula gives $140-4\times45=-40$. That is a surplus of 40 units per day, and the price would tend to fall toward 35.

**2.** With $Q_S=2P$ and $Q_D=120-2P$, clearing requires $120=4P$, so $P=30$ and $Q=60$ units per day. The price falls by 5, the opposite direction from the demand shift, while the quantity rises by 10, the same direction. An outward supply shift lowers price; an outward demand shift raises it.

**3.** With supply fixed at 50, clearing requires $140-2P=50$, so $P=45$ and $Q=50$. The price rises by 10 and the quantity does not change. The whole 20-unit shift is choked off: the higher price moves buyers back along the new demand curve by $2\times10=20$ units, and sellers cannot respond.

**4.** The slope is $(60-50)/(40-35)=2$ units per day per currency unit, which is positive and equals the supply slope. Because only demand shifted, both equilibria lie on the unchanged supply curve, so the line traces out supply, not demand. Equilibrium pairs alone do not identify the demand slope; that would require a shift in supply with demand held fixed.

**5.** With $Q_D=140-2P$ and $Q_S=2P$, clearing requires $140=4P$, so $P=35$ and $Q=70$ units per day. The price is unchanged and the quantity rises by 20. The demand shift alone would raise the price by 5 and the supply shift alone would lower it by 5, so the price effects cancel, while each shift adds 10 units to the quantity traded.

**6.** For example: "Demand increased by 20 units per day at every price, which raised the equilibrium price from 35 to 40 currency units. The higher price reduced the quantity demanded along the new curve by 10 units, so the quantity traded rose from 50 to 60 units per day, an increase of 10, not 20."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/03-market-equilibrium/market_equilibrium.xlsx) for the published analysis. The [original generator](generators/03-market-equilibrium.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/03-market-equilibrium/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/03-market-equilibrium/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: market clears, shift increases price and quantity. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A static crossing does not describe adjustment speed, rationing institutions or uncertainty. Observed price/quantity pairs alone would not identify separate demand and supply slopes.
