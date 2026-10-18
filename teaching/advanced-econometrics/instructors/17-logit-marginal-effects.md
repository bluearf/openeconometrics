# Instructor companion · Lab 17

## Teaching sequence

Write the index, the log-odds equation and the probability equation side by side before running, and ask which scale each of the three reported X numbers (0.952, 2.59, 0.190) lives on. Derive the score equations so students can predict that the average fitted probability will equal the observed rate.

After running, read the coefficient table, then compute the derivative $\beta_x p(1-p)$ at a few probabilities and connect its maximum $\beta_x/4$ to the steepest part of the plotted curve. Keep the AME, the marginal effect at the means and the discrete change distinct, and stress that the AME needs its own standard error.

## Worked exercise answers

**1.** Exponentiate the coefficient endpoints: $e^{0.6763}=1.967$ and $e^{1.2274}=3.412$, so the 95% interval for the odds ratio is 1.97 to 3.41. It is asymmetric around 2.59 because the exponential is convex; the interval is symmetric on the log-odds scale. Adding ±1.96 coefficient standard errors to the odds ratio mixes scales: the standard error 0.141 belongs to $\hat\beta_x$, not to $e^{\hat\beta_x}$.

**2.** Multiply the odds by 2.5905. From 0.05 (odds 0.0526) the new probability is 0.120: a change of 0.070 and a probability ratio of 2.40. From 0.50 (odds 1) it is 0.721: a change of 0.221 and a ratio of 1.44. The odds ratio is the same for both units, but the probability ratio approaches the odds ratio only when the event is rare, and the percentage-point change depends on the starting point.

**3.** The sample means are $\bar x=-0.0418$ and $\bar z=-0.0538$; the index there is −0.4636, the fitted probability 0.386 and the marginal effect $0.9519\times0.386\times0.614=0.226$. That exceeds the AME 0.190. Because the index is linear, the index at the means equals the mean index, so the MEM is $\hat\beta_x\Lambda'(\bar\eta)$ while the AME is $\hat\beta_x$ times the average of $\Lambda'(\hat\eta_i)$. $\Lambda'$ is bell-shaped and concave over the central range where most indices fall, so, as Jensen's inequality suggests, the average of $\Lambda'(\hat\eta_i)$ (0.200) lies below $\Lambda'(\bar\eta)$ (0.237).

**4.** The average of $\Lambda(\hat\eta_i+\hat\beta_x)-\Lambda(\hat\eta_i)$ is 0.193, slightly above the AME 0.190. The logistic curve is convex below a probability of 0.5 and concave above it. For units below 0.5 (65.5% of the sample) a forward one-unit step gains more than the derivative predicts (by 0.021 on average); for units above 0.5 it gains less (by 0.032 on average). The two nearly cancel here. They diverge more when $\beta_x$ times the step is large, so that the step spans more curvature, or when most units sit on one side of 0.5.

**5.** With $s_i=\hat p_i(1-\hat p_i)$, the gradient is $\partial\widehat{AME}/\partial\beta=\frac1n\sum_i\big[s_i e_x+\hat\beta_x s_i(1-2\hat p_i)w_i\big]$, where $e_x$ selects the X coefficient. Numerically it is (0.0243, 0.1588, 0.0236). Then $\widehat{SE}=\sqrt{g'\hat Vg}=0.0218$, and the 95% normal interval is 0.147 to 0.233. A finite-difference gradient gives the same values.

**6.** For example: "Holding Z fixed, each extra unit of X multiplies the odds of the event by 2.59 (log-odds coefficient 0.952). Averaged over the 400 units, it raises the event probability by 0.190, or 19.0 percentage points." The coefficient 0.952 is not a percentage, and an odds ratio is not a probability ratio.

## Reproduction and verification evidence

Use the [supplied workbook](../labs/17-logit-marginal-effects/logit_marginal_effects.xlsx) for the published analysis. The [original generator](generators/17-logit-marginal-effects.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/17-logit-marginal-effects/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/17-logit-marginal-effects/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: probabilities in support, intercept score zero, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Odds ratios are not risk ratios or percentage-point effects. A descriptive AME is not causal without an identification argument; uncertainty for the AME requires its own delta or resampling calculation.
