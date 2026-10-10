# Instructor notes · Lab 10

Emphasize that a difference between significance labels is not a significance test of a difference. The simulation's highly correlated regressors make the sum precise while their decomposition remains much less precise. See the [student handout](../labs/10-joint-tests/README.md).

## Worked answers

1. With order intercept, coaching, software, tenure, joint-zero rows are (0,1,0,0) and (0,0,1,0); equality is (0,1,-1,0); zero sum is (0,1,1,0). Restriction counts are 2, 1 and 1.
2. Coaching p=.209441 evaluates one conditional coordinate. Joint F(2,376)=187.582202 evaluates both-zero together. Rejection implies the vector is inconsistent with both zero; it does not establish each coordinate separately.
3. The sum variance is $.631560^2+.620855^2+2(-.383124)$, approximately .018081, giving SE .134464. Use full precision for exact agreement.
4. F(1,376)=.641304, p=.423745. Fail to reject equality; do not claim equivalence or proof.
5. Writing M=(C+S)/2 and D=C-S gives C=M+D/2 and S=M-D/2. The M slope is the original sum, 2.585257, while the D slope is half the original difference, about -.498596. The former follows the well-supported shared-exposure direction; the gap remains much less precise. This reparameterization changes neither fitted values nor the information in the sample.
6. If software is recorded as hours divided by ten, its coefficient is ten times the hourly coefficient. Equality per hour becomes coaching slope = software-unit slope divided by ten; adding one hour each uses contrast (1,0.1).
7. Include the combined 2.585257-point estimate, SE .134464, CI [2.320861,2.849654], strong joint evidence, decomposition uncertainty, and original synthetic scope.

## Reproduction and verification

`run_lab(output_dir="lab10-review")` explicitly saves full reference and LaTeX table. [Saved reference](../labs/10-joint-tests/reference.json) includes the native test and contrast mappings. Independent matrix OLS/HC1 calculations verify coefficients and covariance; the two-row restriction matrix reproduces the native Wald F; a one-row contrast independently reproduces the sum and its SE. Sample positions, df and complete ResultBundle JSON roundtrip are checked. Initial coefficient/covariance disagreement was below $3\times10^{-13}$.

Native displays are `OLSResult`, `DataFrame`, `Latex`. A robust Wald F should not be replaced by a classical RSS formula without changing the stated inference convention.

**Prepared input:** [productivity_joint_tests.xlsx](../labs/10-joint-tests/productivity_joint_tests.xlsx). Use the stored observations for the published analysis. The [original generator](generators/10-joint-tests.py) supports new dataset editions; update results and answers when changing observations.
