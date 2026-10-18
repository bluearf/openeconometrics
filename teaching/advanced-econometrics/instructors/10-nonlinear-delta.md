# Instructor companion · Lab 10

## Teaching sequence

Start with units: the ratio of the two slopes is measured in z units per x unit, and rescaling either regressor rescales it. Derive the first-order Taylor expansion and the gradient $(0,\ 1/\beta_2,\ -\beta_1/\beta_2^2)$ before running, and expand the quadratic form into its two variance parts and the covariance part.

Before running, ask students for the sign of the denominator derivative and the sign of the covariance part. After running, use the bar chart (0.0188, 0.0294, 0.0257) to show that the covariance part is positive and about 35% of the variance, then use Exercises 2–5 to show that delta-method inference depends on how the function or null is written.

## Worked exercise answers

**1.** Dropping the covariance part leaves $0.01884+0.02940=0.04824$, a standard error of **0.2196**. Reversing the sign of $\hat V_{12}$ gives $0.04824-0.02567=0.02257$, a standard error of **0.1502**. With the actual negative covariance, an error that raises $\hat\beta_1$ tends to lower $\hat\beta_2$, and both raise the ratio, so the errors reinforce and the variance is larger (0.2719). Ignoring the covariance understates the standard error; a positive covariance would have made the errors partly cancel.

**2.** For $\beta_2/\beta_1$, `oe.nlcom` gives 0.5762 with delta standard error 0.0903 (equal to $0.2719/1.7355^2$) and 95% interval [0.3984, 0.7540]. The reciprocals of the ratio's endpoints are [0.4403, 0.8334]. The delta interval is symmetric on whatever scale the function is written, and the reciprocal of a symmetric interval is not symmetric, so the two sets cannot coincide. First-order linearization is not invariant to nonlinear reparametrization.

**3.** $\log(\hat\beta_1/\hat\beta_2)=0.5513$ with delta standard error 0.1566 (the ratio's relative standard error) and interval [0.2427, 0.8599]. Exponentiating gives [1.2747, 2.3629]. The upper side is longer (0.627 above 1.7355 against 0.461 below), matching the right skew of a ratio, whose value grows quickly when the denominator comes out small. The symmetric delta interval [1.200, 2.271] places more weight below the estimate.

**4.** Not rejecting $\beta_1-\theta_0\beta_2=0$ at 5% means $(\hat\beta_1-\theta_0\hat\beta_2)^2\le t^2(\hat V_{11}-2\theta_0\hat V_{12}+\theta_0^2\hat V_{22})$, with $t^2=1.9700^2=3.881$. Rearranged: $(\hat\beta_2^2-t^2\hat V_{22})\theta_0^2-2(\hat\beta_1\hat\beta_2-t^2\hat V_{12})\theta_0+(\hat\beta_1^2-t^2\hat V_{11})\le0$, that is $0.5853\,\theta_0^2-2.1465\,\theta_0+1.7879\le0$. Because the leading coefficient is positive (the `z` slope is significantly nonzero), the solution is a bounded interval, **[1.2789, 2.3885]**. Like the log-scale interval it is shifted to the right of the delta interval [1.2000, 2.2711]. The Fieller set is built from an exactly linear test, so it stays valid when the denominator is imprecise; then it can become very wide or unbounded, which is an honest description of the uncertainty.

**5.** Delta method: $(1.7355-1)/0.2719=2.706$, $p=0.0073$. Linear restriction: $\hat\beta_1-\hat\beta_2=0.5737$ with standard error 0.1625, $t=3.531$, $p=0.0005$. Both reject at 5%, but the evidence differs by more than a factor of ten. Wald tests are not invariant to how a nonlinear null is written: the linear form needs no approximation of the function, while the ratio form relies on linearizing $\beta_1/\beta_2$. When a null can be written linearly, test it that way.

**6.** For example: "The ratio of the x slope to the z slope is 1.74 z units per x unit: in this model, a one-unit difference in x is associated with the same difference in y as about 1.74 units of z, holding the other regressor fixed. Its delta-method standard error, using the full HC3 covariance, is 0.27 (95% interval 1.20 to 2.27); dividing the two slope standard errors does not give the ratio's standard error. This is a conditional association in synthetic data, not a causal comparison of effectiveness."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/10-nonlinear-delta/nonlinear_delta.xlsx) for the published analysis. The [original generator](generators/10-nonlinear-delta.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/10-nonlinear-delta/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/10-nonlinear-delta/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: delta gradient matches native, ratio matches coefficients, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Delta approximations for ratios can fail near a zero denominator. The contrast does not become causal merely because it is calculated from controlled coefficients.
