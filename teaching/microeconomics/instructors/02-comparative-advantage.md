# Instructor companion · Lab 02

## Teaching sequence

Have students write each economy's labor constraint and its two intercepts before computing anything. Then separate the absolute comparison (hours for the same good across economies) from the opportunity-cost comparison (the two goods within one economy).

Use the pause question before running: students should answer from the ratios 2/4 and 6/3, not from the hours alone. After the run, ask them to place the stipulated price of 1 on the bar chart and explain why any price between 0.5 and 2 divides the same gain of 1.5 units of Y per unit of X production moved from B to A.

## Worked exercise answers

**1.** The opportunity cost of Y is $a_Y/a_X$: 4/2 = 2 units of X per unit of Y for A and 3/6 = 0.5 units of X per unit of Y for B, so B has the comparative advantage in Y. Each number is the reciprocal of that economy's opportunity cost of X (0.5 and 2). The economy with the lower cost of X therefore has the higher cost of Y, so with two goods and unequal costs neither economy can have the comparative advantage in both.

**2.** With 60 hours on each good, A makes 60/2 = 30 units of X and 60/4 = 15 units of Y; B makes 60/6 = 10 units of X and 60/3 = 20 units of Y. World output is 40 units of X and 35 units of Y. Under specialization it is 60 units of X and 40 units of Y, so specialization adds 20 units of X and 5 units of Y with the same 240 hours.

**3.** The price is 1 unit of Y per unit of X and the shipment is feasible (20 ≤ 60 and 20 ≤ 40). A consumes 40 units of X and 20 units of Y; alone, A could pair 40 units of X with only (120 − 80)/4 = 10 units of Y, so it gains 10 units of Y. B consumes 20 units of X and 20 units of Y; alone, 20 units of X would use all 120 of B's hours and leave 0 units of Y, so B gains 20 units of Y. These match $(p-0.5)T=10$ and $(2-p)T=20$ with $T=20$; both bundles lie outside their owners' own frontiers.

**4.** At 0.4, A refuses: exporting a unit of X would bring 0.4 units of Y, while shifting the same 2 hours into Y at home gives 0.5, so A makes its own Y. At 1.5, both trade: A gains 1.5 − 0.5 = 1.0 unit of Y per unit of X exported and B saves 2 − 1.5 = 0.5 units of Y per unit imported, which again sum to 1.5. At 2.5, B refuses: buying X would cost 2.5 units of Y, while making it at home costs 2, so B makes its own X.

**5.** With 5 hours per unit of Y in B, A has the absolute advantage in both goods (2 < 6 hours for X and 4 < 5 hours for Y). B's opportunity cost of X becomes 6/5 = 1.2 units of Y per unit of X, still above A's 0.5, so A keeps the comparative advantage in X and B in Y. The price 1 still lies between 0.5 and 1.2, so both gain: A's gain stays at 0.5 units of Y per unit exported, while B's saving falls to 0.2. Specialized B now makes 120/5 = 24 units of Y; both checks in `analyze` still pass.

**6.** For example: "A needs fewer hours than B to make a unit of X (2 hours against 6), an absolute advantage. A should export X because its opportunity cost of X, 0.5 units of Y per unit of X, is below B's 2 units of Y per unit of X, a comparative advantage; absolute advantage alone would not settle the pattern of trade."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/02-comparative-advantage/comparative_advantage.xlsx) for the published analysis. The [original generator](generators/02-comparative-advantage.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/02-comparative-advantage/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/02-comparative-advantage/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: price between opportunity costs, labor constraints bind. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Opportunity costs alone do not determine the equilibrium world price or distribution of gains. Transport costs, adjustment, multiple factors and preferences can change the conclusion.
