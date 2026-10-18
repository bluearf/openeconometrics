# Instructor companion · Lab 18

## Teaching sequence

Start by asking what "one more unit" means when both people consume the same provision level, and use the "Pause before running" question to contrast vertical summation of a public good with horizontal summation of a private good. Have students add the two benefit columns in the workbook by hand at a few levels (for example 10 and 30 units) before solving the Samuelson condition algebraically.

After running, stress that neither person's marginal benefit covers the cost at the efficient level, and then move from efficiency to financing: personalized prices (Exercise 1) and voluntary provision (Exercise 3) show why the efficiency rule is not a mechanism.

## Worked exercise answers

**1.** Personalized prices are 10 per unit for A and 15 for B. A pays $10\times30=300$ and receives gross benefit $\int_0^{30}(40-q)\,dq=1{,}200-450=750$, a net surplus of 450 currency. B pays $15\times30=450$ and receives $\int_0^{30}(30-0.5q)\,dq=900-225=675$, a net surplus of 225. The payments total 750, exactly the cost $25\times30$, and the net surpluses add to $450+225=675$, the lab's total.

**2.** Horizontal summation at a price of 25 adds A's quantity, $40-Q=25\Rightarrow Q=15$, to B's, $30-0.5Q=25\Rightarrow Q=10$, giving 25 units. That would be the market quantity for a rival good, where each person consumes their own units. Here both people consume the same units, so the relevant question is what one more shared unit is worth to both together, which requires adding valuations at the same quantity.

**3.** Alone, A provides units until $40-Q=25$, so 15 units. At 15 units B's marginal benefit is $30-0.5\times15=22.5<25$, so B adds nothing and free rides on A's provision. Net surplus at 15 units is $45\times15-0.75\times15^2=506.25$ currency, a loss of $675-506.25=168.75$ relative to efficient provision.

**4.** With $MC=16$, $70-1.5Q=16$ gives $Q^*=36$. There A's marginal benefit is 4 and B's is 12, summing to 16. Net surplus is $\int_0^{36}(54-1.5q)\,dq=1{,}944-972=972$ currency.

**5.** At 20 units, $(20+20)-25=15>0$: one more unit is worth 15 more than it costs, so provision should rise. At 35 units, $(5+12.5)-25=-7.5<0$: the last unit costs 7.5 more than it is worth to both people together, so provision should fall.

**6.** For example: "With stipulated valuations, efficient provision is 30 units, where the two neighbors' marginal benefits of 10 and 15 add up to the marginal cost of 25. This identifies the target, not a way to reach it: if each neighbor decides alone, each weighs only their own benefit against the full cost, so voluntary contributions are likely to fund fewer than 30 units unless valuations are revealed and payments are arranged."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/18-public-good/public_good.xlsx) for the published analysis. The [original generator](generators/18-public-good.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/18-public-good/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/18-public-good/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: samuelson condition, positive individual marginal benefits. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The optimality condition is not a mechanism guaranteeing voluntary financing. Information, distribution and free-riding require separate institutional analysis.
