# Instructor companion · Lab 19

## Teaching sequence

Have students write one firm's profit with rivals' output held fixed and derive the best response before any symmetry is imposed; the "Pause before running" question asks them to solve the duopoly by hand. Emphasize the $2b$ term (a unit sold lowers the price on all of a firm's units) and the check that $n=1$ reproduces monopoly.

After running, read the scenario table column by column: per-firm output falls, total output rises, and the markup shrinks as $1/(n+1)$. Exercises 2 and 5 contrast the Cournot outcome with collusion and with price competition, so students see which conclusions depend on quantity competition.

## Worked exercise answers

**1.** With $n=3$, a firm's best response to two rivals producing $q$ each is $q=40-2q/2$, so $2q=40$ and $q=20$; total output is 60 and the price $100-60=40$, matching the three-firm row. Profit per firm is $(40-20)\times20=400$. Industry profit is 1,600 with one firm, $2\times711.11=1{,}422.22$ with two and $3\times400=1{,}200$ with three. It falls as firms are added because total output rises and the price falls toward cost; each firm ignores the price reduction its output imposes on rivals.

**2.** Splitting the monopoly output gives each firm 20 units at a price of 60, a profit of $(60-20)\times20=800$, more than the Cournot 711.11. If firm 1 deviates while firm 2 keeps 20, its best response is $40-20/2=30$; total output is 50, the price 50, and firm 1 earns $(50-20)\times30=900$ while firm 2's profit drops to $(50-20)\times20=600$. Each firm gains by breaking the agreement, so it is not self-enforcing in a one-shot game; sustaining it would need repeated interaction or enforcement.

**3.** The markup is $P^*-c=80/(n+1)$. It is at most 5 when $n+1\ge16$, so 15 firms are needed: with 15 firms the price is exactly 25 and each firm produces 5 units. With 14 firms the price is 25.33.

**4.** With two firms, $Q=53.33$, consumer surplus is $\tfrac12\times53.33^2=1{,}422.22$, total profit is 1,422.22 and total surplus 2,844.44. The competitive outcome ($Q=80$, $P=20$) yields $\tfrac12\times80\times80=3{,}200$, so the deadweight loss is 355.56, equal to $\tfrac12(46.67-20)(80-53.33)$. With ten firms, consumer surplus is 2,644.63, total profit 528.93, total surplus 3,173.55 and the deadweight loss 26.45 currency.

**5.** With identical constant costs, no capacity limits and a homogeneous product, Bertrand price competition drives the price to marginal cost, 20, even with two firms: any price above 20 can be undercut slightly to capture the whole market. Total output is 80, split 40 and 40, and profit is zero, compared with the Cournot duopoly price of 46.67 and profit of 711.11 per firm.

**6.** For example: "In the stipulated Cournot duopoly, each firm independently chooses 26.67 units as its best response to the other, and the price of 46.67 exceeds marginal cost of 20 without any agreement, because each firm restricts output to protect the price on its own units. Collusion would instead restrict output to the monopoly 40 units and raise the price to 60."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/19-cournot-competition/cournot_competition.xlsx) for the published analysis. The [original generator](generators/19-cournot-competition.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/19-cournot-competition/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/19-cournot-competition/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: symmetric best response, more firms lower price. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The number of firms is an external scenario input. Entry costs, asymmetric costs, capacity and repeated-game collusion can produce different outcomes.
