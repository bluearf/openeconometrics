# Instructor companion · Lab 04

## Teaching sequence

Show the coefficient table first and ask what can be concluded about `x` and `z` separately. Then derive the variance of one coefficient through partialling out (the variance inflation factor) and the variance of the sum as $c'\hat Vc$. The reparametrization $\theta x_i+\beta_z(z_i-x_i)$ is the clearest single explanation of why the sum is precise; draw it on the chart.

Before running, ask whether the separate standard errors can be added. After running, keep three ideas apart: collinearity limits information about some combinations of coefficients, it does not bias them, and dropping a regressor to "fix" it changes the parameter being estimated.

## Worked exercise answers

**1.** From $\mathrm{SE}_{\text{sum}}^2=V_{xx}+V_{zz}+2V_{xz}$, $V_{xz}=(0.06457^2-2.06762^2-2.06274^2)/2=-4.2629$, which matches the off-diagonal element of `model.covariance_matrix`. The correlation between the two estimates is $-4.2629/(2.06762\times2.06274)=-0.9995$: almost perfectly negative.

**2.** `oe.lincom(model, {"x": 1.0, "z": -1.0})` gives $\hat\beta_x-\hat\beta_z=-1.7046$ with standard error 4.1299 and 95% interval −9.84 to 6.43. Its variance is $V_{xx}+V_{zz}-2V_{xz}$, so the negative covariance now *adds* to the variance: $\sqrt{2.06762^2+2.06274^2+2\times4.2629}=4.1299$. The difference uses the direction in which the regressors barely differ.

**3.** $1/(1-0.99956^2)=1132.4$. The classical standard error of the `x` coefficient is 2.1018, so with an uncorrelated `z` (same residual variance and same variation of `x`) it would be $2.1018/\sqrt{1132.4}=0.0625$. Equivalently, $\sqrt{\hat\sigma^2/\mathrm{SST}_x}=0.0625$ with $\hat\sigma^2=198.95/237$.

**4.** The short regression of `y` on `x` alone gives a slope of 2.0369 (HC3 SE 0.0641), against the sum 2.0318 (SE 0.0646). By the Lab 02 identity, the short slope equals $\hat\beta_x+\hat\beta_z\hat\pi_x$, and the slope of `z` on `x` is $\hat\pi_x=1.0027$, so the short slope is almost exactly $\hat\beta_x+\hat\beta_z$. When `z` is nearly a copy of `x`, omitting it hands its coefficient to `x`.

**5.** At $x=z=1$ the fitted mean is 4.159 (SE 0.091, interval 3.980 to 4.338). At $x=1,\ z=-1$ it is 0.423 (SE 4.127, interval −7.71 to 8.55). The first point lies on the line along which all observed rows fall; the second lies far off it, since no row has `z` differing from `x` by more than 0.12, so the prediction depends on the poorly estimated contrast.

**6.** For example: "In a regression of `y` on two nearly identical regressors (correlation 0.9996), neither coefficient is precisely estimated (HC3 standard errors about 2.07), but they are jointly highly significant ($F=498.96$ on 2 and 237 degrees of freedom) and their sum is 2.03 with standard error 0.065. The data show a strong combined association and cannot apportion it between the two regressors; dropping both would discard it."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/04-collinearity/collinearity.xlsx) for the published analysis. The [original generator](generators/04-collinearity.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/04-collinearity/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/04-collinearity/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: full rank not exact collinearity, sum variance uses covariance, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** High collinearity does not itself cause endogeneity, and deleting a substantively required control can introduce bias. Precision depends on the contrast of interest and where predictions are made.
