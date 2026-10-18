# Instructor companion · Lab 09

## Teaching sequence

Start from the schedule itself: have students read row 4 and explain why output doubles from 1 to 4 worker-days and again from 4 to 16. Define average and marginal product, compare the derivative at $L=4$ with the two neighboring whole-day differences, then derive the hiring rule $p\cdot MP_L=w$ and check its units. Pose the pause question about a fixed cost of 20 before running the file.

Use the figure to connect the rule to labor demand: the value-of-marginal-product curve crosses the wage line at $L=4$, and moving the wage line shows how the hiring choice responds.

## Worked exercise answers

**1.** At $w=4$, $2\times5/\sqrt{L}=4$ gives $\sqrt{L}=2.5$, so $L^*=6.25$, $q=25$ and $\pi=2\times25-4\times6.25-10=15$. On the schedule, the 6th worker-day adds 2.134 units worth 4.27, above the wage, and the 7th adds 1.963 units worth 3.93, below it. The best whole number is 6, with profit 14.99 (14.92 at 7 and 14.72 at 5). The grid cannot reach 6.25, so the best whole step falls just short of the continuous optimum and earns slightly less.

**2.** $p\cdot5/\sqrt{L}=w$ gives $L^*(w)=(5p/w)^2$. At $p=2$ this is $L^*=100/w^2$; at $p=3$ it is $225/w^2$. The higher price raises the value of every worker-day's marginal product by half, shifting labor demand outward; at $w=5$, $L^*$ rises from 4 to 9 worker-days (output 30).

**3.** Profit is 1.33 at 11 worker-days and −0.72 at 12, so 11 is the largest whole number with non-negative profit. Setting $20\sqrt{L}-5L-10=0$ and writing $s=\sqrt{L}$ gives $s^2-4s+2=0$, so $s=2+\sqrt2$ and $L=6+4\sqrt2=11.66$. The other root, $L=0.34$, lies below the schedule's first point.

**4.** $L^*$ stays at 4 in both cases, since $F$ does not enter $p\cdot MP_L=w$. Revenue minus labor cost at $L^*$ is $40-20=20$. With $F=20$, profit is 0: if the fixed cost is unavoidable, operating (0) beats shutting down (−20); if avoidable, the firm is indifferent. With $F=25$, profit is −5: if unavoidable, operating (−5) still beats shutting down (−25); if avoidable, the firm should close and earn 0.

**5.** $wL/(pq)=20/40=0.5$ and $MP_L/AP_L=2.5/5=0.5$. For $q=AL^{b}$, $MP_L/AP_L=b$, the elasticity of output with respect to labor. At the optimum $w=p\cdot MP_L$, so $wL/(pq)=MP_L\cdot L/q=b$. Here $b=0.5$, so the two coincide at every price and wage, not only at the lab's values.

**6.** $1.5\times5/\sqrt{L}=5$ gives $\sqrt{L}=1.5$, so $L=2.25$ worker-days and $q=15$. Each unit of output adds less to revenue than the price, so the marginal revenue product of labor is lower at every $L$ and reaches the wage at a smaller labor input.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/09-production-input/production_input.xlsx) for the published analysis. The [original generator](generators/09-production-input.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/09-production-input/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/09-production-input/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: value marginal product equals wage, analytic matches grid. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The first-order condition assumes an interior choice and price taking. It does not account for indivisible workers, adjustment costs or an uncertain output price.
