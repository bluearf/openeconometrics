# Instructor companion · Lab 13

## Teaching sequence

Begin with the price cut a monopolist must make to sell one more unit, using two adjacent workbook rows, before writing the marginal-revenue derivative. Derive MR = MC, read the price off the demand schedule, and only then introduce the competitive benchmark and the three welfare areas.

Before running, ask students what one row of the workbook is (a candidate market quantity on one demand schedule, not an observed sale) and to predict monopoly output and price. Connect the grid search in the first code block to the calculus and the closed-form block to the welfare formulas, and insist that students name each area as a transfer or a loss before adding anything up.

## Worked exercise answers

**1.** $TR(40)-TR(39)=2400-2379=21$ and $TR(41)-TR(40)=2419-2400=19$. The 40th unit adds 21 to revenue and costs 20, so it raises profit by 1; the 41st adds 19 and costs 20, so it lowers profit by 1. The discrete marginal revenues straddle marginal cost at 40, matching the continuous $MR=100-2Q=20$ at $Q=40$.

**2.** On $Q=100-P$, $dQ/dP=-1$, so at $(Q,P)=(40,60)$ the elasticity is $\varepsilon=-1\times60/40=-1.5$. Then $-1/\varepsilon=1/1.5\approx0.667$, equal to the markup fraction $(60-20)/60$. Demand is elastic at the monopoly point, as it must be when marginal cost is positive.

**3.** At $Q_C=80$ and $P_C=20$, consumer surplus is $\tfrac12(100-20)\times80=3200$ and variable producer surplus is $(20-20)\times80=0$, so total variable surplus is 3,200. Under monopoly it is $800+1600=2400$; the difference, 800, is the deadweight loss. The competitive consumer surplus splits as $3200=800\,(\text{monopoly CS})+1600\,(\text{transfer rectangle})+800\,(\text{DWL})$.

**4.** Output, price and the location of the profit peak do not change: still 40 units at a price of 60, because the fixed cost does not enter marginal revenue or marginal cost. Profit becomes $1600-1800=-200$. If the fixed cost is unavoidable in the short run, producing 40 units still beats producing nothing (which would give $-1800$). If it can be avoided by leaving the market, the firm should leave, since it cannot cover the fixed cost at any output.

**5.** With $MC=30$: $100-2Q=30$ gives $Q_M=35$ and $P_M=65$. Profit is $(65-30)\times35-200=1225-200=1025$. The competitive benchmark becomes $Q_C=70$, so $DWL=\tfrac12(70-35)(65-30)=612.5$; consumer surplus is $\tfrac12(100-65)\times35=612.5$. The price rises by 5 for a 10-unit rise in marginal cost: with linear demand of slope $-1$ and constant marginal cost, a monopolist passes through half of a cost change.

**6.** Profit is not a social cost. Of the 1,600 in variable producer surplus, all of it is a transfer from buyers relative to the competitive benchmark, and the fixed cost of 200 is paid in both outcomes. The social cost is the value of the trades that do not happen. A corrected sentence: "Monopoly pricing reduces total surplus by 800 a day, the value of the 40 units between 40 and 80 that buyers value above their cost of 20; the firm's 1,600 gain in variable producer surplus over the competitive benchmark, which leaves it a profit of 1,400 after the fixed cost, is a redistribution from consumers, not a loss to society."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/13-monopoly-welfare/monopoly_welfare.xlsx) for the published analysis. The [original generator](generators/13-monopoly-welfare.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/13-monopoly-welfare/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/13-monopoly-welfare/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: marginal revenue equals mc, lost trade triangle. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Price discrimination, dynamic innovation and entry can change these comparisons. The exercise is a static partial-equilibrium model, and producer surplus excludes the separately stated fixed cost.
