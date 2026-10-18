# Instructor companion · Lab 11

## Teaching sequence

Before anyone runs the file, have students write the three regressions and the telescoping identity, and state what each bracket holds fixed. Then derive $b_{SC}-b_{LC}=\hat\gamma\hat\delta$ on the board from the orthogonality of the long-model residuals, and ask why the derivation needs a common sample.

After the run, read the Rows and residual df columns before any slope. Insist that the decomposition is a point-estimate identity with no standard error of its own, and that neither component is a causal or missingness-bias estimate.

## Worked exercise answers

**1.** The blank Z cells are in `row_id` 6, 36 and 66, with (`x`, `y`) = (1.0008, 5.0080), (−0.8636, −0.4386) and (−1.0965, −1.4367). Their residuals from the short fit on 240 rows are +1.439, −0.903 and −1.513, and their deviations from $\bar x=-0.111$ are +1.112, −0.753 and −0.986. Each product of residual and deviation is positive (1.60, 0.68, 1.49), so each row pulls the slope up; removing them lowers it. Dropping each alone lowers the slope by 0.0077, 0.0032 and 0.0071.

**2.** On the 237 complete rows, the slope of `z` on `x` is $\hat\delta=0.6256$. Then $\hat\gamma\hat\delta=0.7745\times0.6256=0.4846$, equal to the control component to machine precision. The product cannot reconcile $b_{SA}$ with $b_{LC}$: that total is 0.5027, and the gap of 0.0181 is the sample component. The identity holds only when both regressions use the same rows, because it relies on the long-model residuals being orthogonal to `x` over exactly those rows.

**3.** Without rows 1–3 the short slope is 1.6697, so the "sample component" is $1.6654-1.6697=-0.0043$: smaller and of the opposite sign. Removing an arbitrary set of three rows also moves the slope, so the sample component measures sensitivity to the row set, not bias caused by the missing controls.

**4.** With zeros in place of the blanks, the controlled X slope on 240 rows is 1.1972 (Z coefficient 0.7588), against 1.1627 on the complete rows. The fill asserts that three units have Z exactly zero, which the data do not say, so the estimate belongs to a model with three invented control values rather than to either sample in the lab.

**5.** On the common rows, $R^2$ rises from 0.5116 to 0.6283, a change of 0.1166. Comparing 0.6283 with the short $R^2$ on 240 rows (0.5180) gives 0.1103. The second comparison divides by different total sums of squares (1,128.85 on 240 rows and 1,103.41 on 237 rows), so it mixes the change of sample with the contribution of Z.

**6.** For example: "In 240 synthetic records, the HC3 slope of Y on X falls from 1.665 (all 240 rows) to 1.163 when Z is added on the 237 rows where Z is recorded, a drop of 0.503 or about 30%. Of that drop, 0.018 comes from losing three rows and 0.485 from adding Z on a fixed sample. The split is an arithmetic identity, not an estimate of missingness bias or of a causal effect of X."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/11-common-sample/common_sample.xlsx) for the published analysis. The [original generator](generators/11-common-sample.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/11-common-sample/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/11-common-sample/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: sample plus control decomposes total, three missing controls, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** An arithmetic decomposition does not establish that complete cases represent excluded rows. Missingness assumptions are required for a population interpretation.
