# Instructor companion · Lab 09

## Teaching sequence

Write the joint null as $R\beta=r$ and derive the Wald quadratic form before running anything. Then invert the $2\times2$ slope covariance on the board to obtain $W=(t_1^2+t_2^2-2\rho t_1t_2)/(1-\rho^2)$, so that students see exactly how the correlation between the estimates enters.

Before running, ask students how $W$ would relate to the separate $t$ statistics if the estimates were uncorrelated. After running, compute $\rho=-0.616$ from the covariance matrix, draw the tilted confidence ellipse against the rectangle of two separate intervals, and use Exercises 3 and 4 to show that the two can disagree in either direction.

## Worked exercise answers

**1.** The $\chi^2_2$ $p$ value of $W=510.21$ is about $1.6\times10^{-111}$; the $F(2,237)$ $p$ value of $F=255.11$ is $8.0\times10^{-60}$. The 5% critical values are 5.991 for $\chi^2_2$ and 3.034 for $F(2,237)$ (6.068 on the $W$ scale). The $F$ reference has heavier tails because its denominator allows for estimating the error scale, and far in the tail that small difference in shape becomes an enormous ratio of tail probabilities, even though the decisions agree. $W$ belongs with $\chi^2_2$ and $F=W/2$ with $F(2,237)$; pairing $W$ with an $F$ $p$ value, or $F$ with a $\chi^2$ one, misstates the evidence.

**2.** With $t_1=10.4706$, $t_2=9.3195$ and $\rho=-0.6159$: $t_1^2+t_2^2=196.49$, $-2\rho t_1t_2=+120.20$ and $1-\rho^2=0.6207$, so $W=316.68/0.6207=510.21$. If $\rho$ were zero, $W=196.49$ and $F=98.24$, the average of the two squared $t$ statistics. The negative correlation adds the cross term and shrinks the denominator: both estimates are positive, a combination that negatively correlated estimation errors rarely produce by chance, so the joint evidence is stronger than the separate statistics suggest.

**3.** For $\beta_1=1$ and $\beta_2=1$: the separate $t$ statistics are 2.00 ($p=0.047$) and −2.34 ($p=0.020$), so each separate test rejects at 5%. The joint test gives $F=2.979$ with $p=0.0528$ and does not reject. The estimate lies above 1 for `x` and below 1 for `z`, a "one up, one down" deviation along the long axis of the ellipse, where negatively correlated estimates often err together. The null point lies outside the rectangle but inside the ellipse.

**4.** For $\beta_1=1.05$ and $\beta_2=0.65$: the separate $t$ statistics are 1.57 ($p=0.117$) and 1.74 ($p=0.083$), so neither separate test rejects. The joint test gives $F=7.164$ with $p=0.00095$ and rejects. Both estimates lie above their null values, a "both up" deviation along the narrow axis of the ellipse. The null point lies inside the rectangle but outside the ellipse, the opposite disagreement from Exercise 3.

**5.** With classical covariance the slope standard errors are 0.1054 and 0.0871, their correlation is −0.641, and the joint $F$ is **305.23**, against 255.11 under HC3. With $R^2=0.72034$, $(R^2/2)/[(1-R^2)/237]=305.23$, identical to the classical Wald $F$: under homoskedastic covariance the Wald $F$ equals the comparison of restricted (intercept-only) and unrestricted residual sums of squares, which is what the $R^2$ formula computes. The HC3 Wald statistic is a quadratic form in the robust covariance and has no sum-of-squares or $R^2$ interpretation. Both reject decisively.

**6.** For example: "The joint test rejects the hypothesis that both slopes are zero (HC3 F = 255.1 with 2 and 237 degrees of freedom), so at least one of x and z is associated with y holding the other fixed. The separate t statistics, 10.5 and 9.3, indicate that each slope differs from zero. These are conditional associations in synthetic data, not causal effects."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/09-joint-restrictions/joint_restrictions.xlsx) for the published analysis. The [original generator](generators/09-joint-restrictions.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/09-joint-restrictions/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/09-joint-restrictions/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: F is wald over q, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** A joint rejection says at least one restricted direction differs; it does not prove each slope differs from zero or establish causal effects.
