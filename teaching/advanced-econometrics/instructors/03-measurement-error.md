# Instructor companion · Lab 03

## Teaching sequence

Show the two slopes side by side first, then write the outcome and measurement equations and derive $\operatorname{plim}\hat b_w=\beta\lambda$ from the two population moments. Present the same result as a correlation between the recorded regressor and a composite error. Then move to the exact sample decomposition, so that students can explain why this sample gives 0.824 rather than 0.75 without appealing to "noise" vaguely.

Before running, ask for the population target. After running, separate three things: the attenuation (a bias), the HC3 interval (sampling variation around the biased target) and the zero-slope test (still valid in direction with one regressor and classical error). Point out that the summary label "population attenuation target" refers to the noisy slope's limit, 0.75, not to the attenuation factor 0.5.

## Worked exercise answers

**1.** $\widehat{\operatorname{Var}}(x)=1.0463$, $\widehat{\operatorname{Var}}(v)=0.9346$, $\widehat{\operatorname{Cov}}(x,v)=-0.0419$ and $\widehat{\operatorname{Var}}(w)=1.0463+0.9346+2(-0.0419)=1.8972$. Hence $\widehat{\operatorname{Var}}(x)/\widehat{\operatorname{Var}}(w)=0.5515$. The lab's reliability drops the covariance term from the denominator, $1.0463/1.9810=0.5282$, and $\hat\lambda=(\widehat{\operatorname{Var}}(x)+\widehat{\operatorname{Cov}}(x,v))/\widehat{\operatorname{Var}}(w)=1.0045/1.8972=0.5294$. All three estimate $\lambda=0.5$; they differ only because the sample covariance of `x` and the error is not zero. $\hat\lambda$ comes closest to the slope ratio 0.5293 because it is the exact sample attenuation factor up to the negligible remainder.

**2.** $0.8236/0.5=1.6472$ and $0.8236/0.5282=1.5593$, against a benchmark of 1.5559. The sample-reliability correction is close to the benchmark because it uses this sample's own variances; the population correction overshoots because this sample's attenuation factor is above 0.5. A researcher without the true regressor needs external information on $\lambda$: validation data, repeated measures, or an instrument. Dividing by a known constant divides the standard error by the same constant (0.0597/0.5 = 0.1194); if $\lambda$ is itself estimated, its uncertainty must be added.

**3.** The reverse slope of `observed_x` on `y` is 0.4472, so its inverse is 2.2360. The benchmark 1.5559 lies between 0.8236 and 2.2360. In the reverse regression `y` is the regressor and the outcome error $e$ plays the role of noise: $\operatorname{plim}$ of the inverse is $\beta+\operatorname{Var}(e)/[\beta\operatorname{Var}(x)]\ge\beta$. The two regressions bound $\beta$ under classical error in `x` alone.

**4.** Classical error implies slope 1 and intercept 0 in this regression, since $E[w\mid x]=x+E[v]=x$. The estimates are slope 0.960 (HC3 SE 0.045, 95% interval 0.872 to 1.048) and intercept −0.078 (interval −0.178 to 0.023); the mean recording error is −0.078. Both intervals contain the classical values, so the evidence is consistent with the assumptions, although it cannot prove independence from the outcome error. This check is possible only because the teaching file reveals true `x`.

**5.** Regressing `y_noisy` on true `x` gives slope 1.5159 (HC3 SE 0.0676, $R^2=0.560$), against the benchmark 1.5559 (SE 0.0500, $R^2=0.725$). The slope moves by $-0.0400=\widehat{\operatorname{Cov}}(x,v)/\widehat{\operatorname{Var}}(x)$, a sampling accident, not attenuation: in the population the added error is uncorrelated with the regressor and leaves the slope's limit unchanged. It enlarges the error variance, so the standard error rises and $R^2$ falls. Error in the regressor instead inflates the slope's denominator.

**6.** For example: "Using the recorded measure, the least-squares slope is 0.82 (HC3 95% interval 0.71 to 0.94). Under classical measurement error this slope estimates the true slope times the measure's reliability, about one half here, so the interval does not bound the slope on the true regressor; in this teaching file that slope is 1.56. Ruling out large effects would require a reliability correction based on external information."

## Reproduction and verification evidence

Use the [supplied workbook](../labs/03-measurement-error/measurement_error.xlsx) for the published analysis. The [original generator](generators/03-measurement-error.py) records the synthetic mechanism and seed. Students receive the fixed workbook; regeneration is an instructor preparation step. Changing observations requires changing the reference, figures, interpretation and answers together.

The [complete reference](../labs/03-measurement-error/reference.json) retains numerical summaries, full native procedure tables and their attributes, chart coordinates, and independent arithmetic checks. If fitted models are present, their full covariance, inference metadata and sample positions are retained too. The [LaTeX table](../labs/03-measurement-error/table.tex) is exported from the executed numerical table. Explicit `run_lab(output_dir="lab-results")` writes and reads back the complete state; default execution writes no exports.

The source checks: same units and sample, positive measurement variance, complete model json roundtrip, unique retained positions, covariance positive semidefinite. The course verifier compares fresh execution with this saved state and executes the exact student source through a disposable application worker, imports the workbook through the normal reader, and reopens saved history. Development-only independent references check the distributions and inferential conventions appropriate to this topic. These checks concern this original teaching example, rather than universal estimator equivalence or an installed-release acceptance claim.

**Misinterpretations to discuss.** Nonclassical error can bias estimates in other directions. HC3 does not correct measurement bias, and an estimated reliability from an unvalidated proxy is not automatically known.
