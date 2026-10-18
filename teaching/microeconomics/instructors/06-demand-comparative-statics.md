# Instructor companion · Lab 06

## Teaching sequence

Have students derive the tangency condition and the two Cobb–Douglas demands before running anything, and check units in $x^*=\alpha m/p_x$. Then pose the pause question: predict X and Y in `income_high` and `x_dearer` from the formulas alone. Run the file, compare each scenario with `base` one parameter at a time, and use the price rows to separate point, simple-percentage and midpoint elasticities. Close by asking which results are imposed by the functional form.

Point out that the two elasticities in the lab's table are analytic constants written into `analyze`, not values computed from the five rows.

## Worked exercise answers

**1.** In `x_dearer` the price of X rises from 3 to 6 (+100%) and X falls from 16 to 8 (−50%), so the simple ratio is −50/100 = −0.5. With midpoints, $\Delta x/\bar x=-8/12=-0.667$ and $\Delta p_x/\bar p_x=3/4.5=0.667$, giving −1. The simple ratio depends on which end is the base and is not an elasticity for a large change. The midpoint formula agrees with the point elasticity here because spending $p_x x$ is constant at 48: for $x=k/p$ the midpoint ratio is exactly −1.

**2.** Intercepts are $m/p_x$ and $m/p_y$, slope $-p_x/p_y$. `income_low`: 30 X, 45 Y, slope −1.5. `income_high`: 50 X, 75 Y, slope −1.5. `x_cheaper`: 60 X, 60 Y, slope −1. `x_dearer`: 20 X, 60 Y, slope −3. The income rows shift the line in parallel (base intercepts 40 and 60); the price rows rotate it around the fixed Y intercept of 60.

**3.** Y rises from 27 to 45 as income rises from 90 to 150, so $\Delta y/\Delta m=18/60=0.3$ units of Y per currency unit, which equals $(1-\alpha)/p_y=0.6/2$. The midpoint elasticity is $(18/36)/(60/120)=0.5/0.5=1$. Differentiating, $\partial y^*/\partial m\cdot m/y^*=(0.3)\cdot m/(0.3m)=1$.

**4.** With $\alpha=0.5$, $x^*=0.5\times120/3=20$ and $y^*=0.5\times120/2=30$; the budget check is $60+60=120$. `income_high` also gives 20 units of X, but its Y is 45 because the whole budget rose to 150. Here the budget is unchanged and spending is redirected from Y (72 to 60) toward X (48 to 60), so Y falls. The same X can come from a preference change or an income change; X alone cannot tell them apart.

**5.** With income 240, $p_x=6$ and $p_y=4$, $y^*=0.6\times240/4=36$, the base value, as homogeneity requires (X is also 16). Doubling only income and $p_x$ gives $y^*=0.6\times240/2=72$. X demand does not depend on $p_y$, so scaling $m$ and $p_x$ is a full homogeneity test for X. Y depends on $m$ and $p_y$, so leaving $p_y$ at 2 is a 100% income increase for Y, not a common rescaling.

**6.** For example: "In this Cobb–Douglas model, Y stays at 36 when the price of X doubles because the consumer always spends 60% of income on Y, so the uncompensated cross-price elasticity is zero. That zero reflects a substitution effect toward Y and an income effect away from Y that cancel exactly; it does not mean the goods are unrelated, and other preferences would generally give a nonzero cross-price response."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/06-demand-comparative-statics/demand_comparative_statics.xlsx) for the published analysis. The [original generator](generators/06-demand-comparative-statics.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/06-demand-comparative-statics/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/06-demand-comparative-statics/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: all budgets exhausted, homogeneous demand. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** These elasticities are imposed by Cobb–Douglas preferences. They do not establish actual consumer responses or substitution patterns across arbitrary goods.
