# Instructor companion · Lab 11

## Teaching sequence

Ask students to sketch the 60-unit isoquant and one isocost line before running anything, and to say in words why a bundle with MRTS above w/r cannot be the cheapest. Then derive the conditional demands together and check them on the base scenario by hand.

Before running, ask students to identify what each row of the workbook is (a scenario at which the model is solved, not an observed firm) and to predict K/L for every row. Connect the two formulas in the code to the derivation in Section 4 before interpreting the table, and keep the high-wage comparison at fixed output.

## Worked exercise answers

**1.** Output: $2\sqrt{10\times 90}=2\sqrt{900}=60$. At (10, 90), $MP_L=\sqrt{90/10}=3$ units of output per unit of labor and $MP_K=\sqrt{10/90}=1/3\approx0.333$ units of output per unit of capital, so $MRTS=3/(1/3)=9=K/L=w/r=9/1$. Equivalently $MP_L/w=3/9=1/3=MP_K/r$, so the last currency unit buys the same extra output from either input.

**2.** The old bundle at the new wage costs $9\times15+1\times60=135+60=195$ currency per day, which is 15 more than the minimum of 180. The 15 is the saving from substitution: by moving along the 60-unit isoquant from (15, 60) to (10, 90), the firm avoids part of the cost increase that keeping the old input mix would impose.

**3.** Base: $4\times15/120=60/120=0.5$. High wage: $9\times10/180=90/180=0.5$. With $q=2\sqrt{LK}$, each input's cost is $wL^c=rK^c=(q/2)\sqrt{wr}=C/2$, so the labor share is one half at any prices. A higher wage lowers labor use by exactly the proportion needed to keep the labor bill a constant share; this is a property of the equal-exponent Cobb–Douglas technology, not a general result.

**4.** With $w/r=4/4=1$, the predicted ratio is $K/L=1$. Then $L^c=30\sqrt{4/4}=30$, $K^c=30$, and minimum cost is $4\times30+4\times30=240$ (equivalently $60\sqrt{16}=240$). Minimum cost per unit of output doubles from 2 to 4.

**5.** $\partial C/\partial w=q\sqrt r/(2\sqrt w)=\tfrac q2\sqrt{r/w}=L^c$. At the base scenario this is $60\times1/(2\times2)=15$, equal to conditional labor demand (Shephard's lemma). For a small wage increase, the first-order cost increase is the amount of labor already in use times the wage change; the induced reallocation of inputs has no first-order effect on cost because the bundle was already cost-minimizing (the envelope argument).

**6.** With $q=\min\{4L,K\}$, producing 60 units efficiently requires $4L=K=60$, so $L=15$ and $K=60$ at any positive prices; extra units of either input alone add nothing. Cost is $4\times15+60=120$ at the base wage and $9\times15+60=195$ at the high wage, an increase of 75 rather than the 60 in Section 6. The fixed-proportions firm cannot substitute capital for labor, so it pays the full wage increase on its unchanged labor input; its cost at the high wage equals the old-bundle cost from Exercise 2.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/11-conditional-inputs/conditional_inputs.xlsx) for the published analysis. The [original generator](generators/11-conditional-inputs.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/11-conditional-inputs/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/11-conditional-inputs/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: production constraint, input tangency. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** The solution relies on interior substitutability. Fixed-proportion technology or adjustment constraints would invalidate the tangency calculation.
