# Instructor companion · Lab 14

## Teaching sequence

Find the unregulated equilibrium from the workbook row where the two price columns are equal, and draw the gain per unit, demand price minus supply price, before introducing the ceiling. Then separate three ideas explicitly: the shortage (a gap between desired quantities), the traded quantity (the short side) and the welfare loss (gains on units no longer traded).

Before running, ask students what one row is (a potential unit of trade on stipulated schedules, not an observed transaction) and what quantity trades at 25. Connect the `min` in the first code block to the short-side rule and the `area` helper to the integrals, and make students state the rationing assumption before they quote any consumer-surplus figure.

## Worked exercise answers

**1.** At $q^*=50$, $P^*=35$: $CS=\tfrac12(60-35)\times50=625$ and $PS=\tfrac12(35-10)\times50=625$, which add to 1,250, the maximum total surplus.

**2.** Under the ceiling with efficient rationing, consumer surplus rises from 625 to 825 (+200) and producer surplus falls from 625 to 225 (−400); the net change is −200. For consumers, the 30 units still traded now cost 10 less, a transfer of $10\times30=300$ from sellers, while the consumer surplus on units 31 to 50 at the old price, $\int_{30}^{50}(25-0.5q)\,dq=100$, is lost: $+300-100=+200$. For producers, $-300$ transfer and $-100$ lost on units 31 to 50: $-400$. The two lost pieces, 100 + 100, make up the deadweight loss of 200.

**3.** Each of the 70 buyers who want the good at 25 receives it with probability $30/70$. Their total surplus if all were served would be $\int_0^{70}(35-0.5q)\,dq=2450-1225=1225$, so expected consumer surplus is $\tfrac{3}{7}\times1225=525$. Total surplus is $525+225=750$, a loss of 500 relative to 1,250. It is lower than under efficient rationing because some units go to buyers whose willingness to pay is barely above 25 while higher-value buyers go without.

**4.** Ceiling 30: demanded 60, supplied 40, shortage 20, traded 40. The second block prints consumer surplus 800, producer surplus 400, total 1,200 and lost gains 50. Ceiling 40: the first block prints demanded 40, supplied 60 and traded 40, but this is wrong. A ceiling of 40 is above the equilibrium price of 35, so it does not bind; the price settles at 35, 50 units trade, there is no shortage and no loss. The code assumes the ceiling is the price actually charged, which is only true when it binds.

**5.** The 30 buyers who obtain units lose $5\times30=150$ in queuing time, so consumer surplus is $825-150=675$, total surplus is $675+225=900$ and the loss relative to 1,250 is 350. The queuing cost is a real resource cost, not a transfer: no one receives the time spent waiting.

**6.** For example: "The ceiling lowers the price from 35 to 25, but only 30 units are now traded instead of 50. Buyers who obtain those 30 units save 10 each, a transfer of 300 from sellers; the 20 units no longer traded remove 200 of gains from trade, and with efficient rationing consumer surplus rises by only 200 while producer surplus falls by 400."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/14-price-ceiling/price_ceiling.xlsx) for the published analysis. The [original generator](generators/14-price-ceiling.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/14-price-ceiling/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/14-price-ceiling/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: binding ceiling, quantity short side, surplus partition. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The consumer surplus estimate is conditional on rationing. A lower posted price does not prove every buyer benefits or that everyone can obtain the good.
