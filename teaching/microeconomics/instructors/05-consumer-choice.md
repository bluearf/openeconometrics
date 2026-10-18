# Instructor companion · Lab 05

## Teaching sequence

Ask students to read one row as a candidate bundle that uses up the whole budget, and to notice that the last four columns are constant and define the problem. Have them draw the budget line from its intercepts and compute the MRS at a few rows, for example 3.0 at $x=10$ and 0.33 at $x=30$, before stating the tangency condition.

Use the pause question before running: students who remember the Cobb–Douglas property will answer 48; ask them to derive it from tangency and the budget rather than assert it. After the run, use the chart to show that only the location of the peak carries meaning, not its height.

## Worked exercise answers

**1.** At $x=20$, $y=60-1.5\times20=30$ and $MRS=(0.4/0.6)\times(30/20)=1.0$ unit of Y per unit of X, below the price ratio 1.5. She would give up only 1 unit of Y for another unit of X, while the market charges 1.5, so she values X at the margin less than it costs. She should move toward less X, back toward 16 units. The index there is 25.51, below 26.03.

**2.** With $p_x=2$: $x^*=0.4\times120/2=24$ units of X and $y^*=0.6\times120/2=36$ units of Y. Spending is $2\times24=48$ on X and 72 on Y, unchanged, so the shares stay 0.4 and 0.6. Only the quantity of X changes, from 16 to 24; the quantity of Y does not respond to the price of X under Cobb–Douglas preferences. At the new bundle $MRS=(2/3)\times(36/24)=1=p_x/p_y$.

**3.** The budget line becomes $6x+4y=240$, which is the same line as $3x+2y=120$: the same intercepts (40 and 60) and the same slope (−1.5). The optimal bundle stays at 16 units of X and 36 units of Y. Spending doubles in currency, to 96 and 144, but the shares and the real purchasing opportunities are unchanged; demand depends only on relative prices and real income.

**4.** $\partial V/\partial x=0.4/x$ and $\partial V/\partial y=0.6/y$, so $MRS=(0.4/0.6)(y/x)$, identical to the MRS from $U$. The bundle that maximizes $V$ on the budget line is therefore the same, 16 units of X and 36 units of Y, with $V=\ln 26.0273=3.2591$. The two numbers differ but rank every pair of bundles the same way, so a utility level has no meaning on its own; only the ranking does.

**5.** With $U=x+y$ the MRS is 1 everywhere, while the market charges 1.5 units of Y per unit of X. Every unit of X is worth less to her than it costs, so she spends everything on Y: 0 units of X and 60 units of Y, with $U=60$, against $U=40$ for the all-X bundle. Her indifference curves are straight lines with slope −1 and the budget line has slope −1.5, so they never touch tangentially and the best bundle is a corner.

**6.** For example: "Her best affordable bundle, 16 units of X and 36 units of Y, ranks above every other bundle on her budget line. The index value 26.03 only orders her own bundles; it cannot be compared with a neighbor's index, because any increasing transformation of either index would describe the same preferences while changing the numbers."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/05-consumer-choice/consumer_choice.xlsx) for the published analysis. The [original generator](generators/05-consumer-choice.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/05-consumer-choice/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/05-consumer-choice/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: budget exhausted, tangency. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Tangency is not universal: perfect substitutes or quantity constraints can produce corner solutions. This utility specification imposes constant shares rather than estimating preferences.
