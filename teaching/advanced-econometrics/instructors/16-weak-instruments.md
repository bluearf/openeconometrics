# Instructor companion · Lab 16

## Teaching sequence

Start from the indirect-least-squares form $\hat\beta_{IV}=\hat\gamma/\hat\pi$ and have students derive the error identity $\hat\beta_{IV}-\beta=\hat\delta/\hat\pi$ before running anything. Then ask what the weak scenario holds fixed ($y-1.2x$, hence $\hat\delta$) and what it changes (only $\hat\pi$), and have them predict the weak-scenario estimate from the strong one.

After running, read the two first-stage rows together (same standard error, F = t²), then the two 2SLS columns, and finish by checking that error × $\hat\pi$ is 0.0482 in both scenarios. Keep the conventional t interval separate from a weak-instrument-robust confidence set; Exercise 5 makes the contrast concrete.

## Worked exercise answers

**1.** `weak_x - x` equals $-0.82z$ for every unit (largest deviation about $4\times10^{-16}$), matching $0.08-0.9$. Because $z$ is a regressor, the OLS fit of `weak_x` on intercept, control and $z$ is the strong fit with the $z$ coefficient lowered by exactly 0.82 (0.8966 to 0.0766); the residuals, $s^2$ and $(X'X)^{-1}$ are unchanged, so the standard error on $z$ (0.0527), the intercept (0.0059) and the control coefficient (0.3682) are identical. Hence $F_w=F_s(\hat\pi_w/\hat\pi_s)^2=289.97/137.12=2.115$.

**2.** Classical reduced-form slopes: strong 1.1241, weak 0.1401 (standard error 0.1161 in both). Ratios: 1.1241/0.8966 = 1.2538 and 0.1401/0.0766 = 1.8298, the two IV slopes. The reduced forms differ by $-0.984=-1.2\times0.82$, because $y^w=y-1.2x+1.2x^w=y-0.984z$.

**3.** $\hat\delta=0.048218$. With $\hat\pi=0.0766/2=0.03828$, $\hat\beta=1.2+0.048218/0.03828=2.46$. The error equals 1.2 when $\hat\pi=0.048218/1.2=0.0402$.

**4.** OLS slopes (classical standard errors): strong 1.598 (0.048), weak 1.912 (0.061). Both exceed 1.2, consistent with positive correlation between $v$ and $u$. The strong 2SLS slope (1.254) removes most of that gap; the weak 2SLS slope (1.830) stays close to its OLS counterpart, illustrating the finite-sample pull of 2SLS toward OLS when the first stage is weak, because the first-stage sampling error (driven by $v$) is correlated with $\hat\delta$ (driven by $u$).

**5.** Strong scenario: the retained grid values run from 1.10 to 1.39 (exact boundaries 1.0957 and 1.3940), close to the t interval 1.106 to 1.402. Weak scenario: all 1,301 grid values are retained. Over $[-200,200]$ the smallest p value is 0.146, and as $|b_0|\to\infty$ the test statistic approaches the first-stage t statistic for $z$ (p = 0.147), so the set is the whole real line: the data carry essentially no information about $\beta$. The conventional interval 0.340 to 3.319 looks bounded and informative only because it relies on an approximation that fails here.

**6.** For example: "In the weak scenario the first stage cannot reject irrelevance (F = 2.11), so the 2SLS estimate of 1.83 and its conventional interval are not reliable. With the structural slope fixed at 1.2 in both designs, the larger estimate reflects estimation error amplified by the weak first stage, not a larger effect; a weak-instrument-robust confidence set is uninformative here."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/16-weak-instruments/weak_instruments.xlsx) for the published analysis. The [original generator](generators/16-weak-instruments.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/16-weak-instruments/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/16-weak-instruments/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: weak instrument relevance lower, same structural effect, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A fixed F cutoff is not a universal guarantee. Weak-IV-robust procedures such as inversion of suitable moment tests answer an additional question and are not implemented by these ordinary intervals.
