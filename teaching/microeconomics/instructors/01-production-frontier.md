# Instructor companion · Lab 01

## Teaching sequence

Ask students to say what one row is (a production plan on the frontier, not a sampled economy) and to check the frontier equation at the two intercepts and at $x=24$ by hand. Then compare the average cost of the first 24 units of X (16/24 Y per X) with that of the last 16 (4 Y per X) before introducing the derivative.

Use the pause question before running: students should predict that one more unit of X costs more Y at $x=35$ than at $x=5$ and explain it by resources that are increasingly specialized in Y. Connect the code to both derivative routes, the explicit formula and implicit differentiation, before interpreting the table.

## Worked exercise answers

**1.** The workbook gives $y(23)=65.4523$, $y(24)=64$ and $y(25)=62.4500$. Moving from 23 to 24 units of X costs 1.4523 units of Y; moving from 24 to 25 costs 1.5500. The marginal cost 1.5 lies between them. The frontier gets steeper as X rises, so the step before $x=24$ averages slopes flatter than the slope at 24 and the step after averages steeper ones. Their mean, the central difference $(65.4523-62.4500)/2=1.5011$, is very close to 1.5.

**2.** $OC_X(10)=20/\sqrt{1500}=0.5164$, $OC_X(32)=64/\sqrt{576}=64/24=2.6667$ and $OC_X(39)=78/\sqrt{79}=8.7757$ units of Y per unit of X. The cost rises slowly at first and then very fast near the X intercept, because the last resources moved into X are those best suited to Y.

**3.** $(0, 0)$: $0<1600$, feasible but inefficient (both goods could increase). $(20, 50)$: $20^2+25^2=1025<1600$; the frontier gives $y=69.28$ at $x=20$, so the plan is feasible but inefficient, with 19.28 more units of Y available. $(30, 60)$: $30^2+30^2=1800>1600$; the frontier gives only $y=52.92$ at $x=30$, so the plan is infeasible. None of the three is efficient.

**4.** Set $2x/\sqrt{1600-x^2}=2$, so $x^2=1600-x^2$ and $x=\sqrt{800}=28.28$ units of X, with $y=2\sqrt{800}=56.57$ units of Y. To the left of this plan the frontier is flatter than the intercept line (cost below 2); to the right it is steeper.

**5.** An outward shift lets the economy produce more of at least one good without less of the other, so the feasible set grows. Whether people are better off also depends on which plan is chosen, how the output is distributed, and what was given up to obtain the new resources or technology. Answers should name preferences or a social welfare criterion, the chosen allocation and its distribution, and the cost of the shift.

**6.** For example: "At the efficient plan with 24 units of X and 64 units of Y, the marginal opportunity cost of X is 1.5 units of Y per additional unit of X. This is a local rate, not the average cost of the 24 units already produced, which was 16 units of Y in total."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/01-production-frontier/production_frontier.xlsx) for the published analysis. The [original generator](generators/01-production-frontier.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/01-production-frontier/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/01-production-frontier/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: frontier identity, negative frontier slope. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A frontier cannot choose an allocation without preferences or values. A movement along it differs from an outward shift caused by new resources or technology.
