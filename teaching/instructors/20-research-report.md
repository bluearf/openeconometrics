# Instructor notes · Lab 20: From Research Question to Finished Report

The [student handout](../labs/20-research-report/index.md) and
[source](../labs/20-research-report/lab.py) form a coherent original
observational-school capstone. The numerical workflow is reproducible, but
the observed controls do not identify the known structural tutoring effect.
Latent motivation affects both tutoring and final achievement and remains
unavailable in the analyst's table.

## Teaching approach

Have students state a population, outcome unit, and statistical target before
they inspect estimates. Require a same-sample comparison when they discuss
adding controls. Ask them to distinguish the confidence interval for an
association, a conditional mean prediction, and an intervention effect.

The adjusted coefficient is more precise than the unadjusted one and farther
from the structural truth in this realization. That prevents “more controls
plus robust standard errors” from becoming an automatic certification rule.
The result paragraph should place the confounding limitation next to the
estimate rather than relegating it to an appendix.

## Worked exercise answers

1. The statistical target is the linear conditional association between
   weekly tutoring hours and school final-score points, holding measured
   baseline score and resources fixed. A causal interpretation would need
   suitable assignment/exogeneity, comparable measurement, an appropriate
   estimand and functional form, and no relevant omitted confounding.
   The generator explicitly violates observed-control sufficiency through
   motivation, so the fitted tutoring association is not an identified
   intervention effect.

2. Resource values are missing for school identifiers divisible by 13:
   **23 of 300 schools**. All three main models use **277 complete cases**.
   Their differences reflect specification changes on one sample. The fourth
   model is an unadjusted fit on all 300 schools; comparing it directly with
   the adjusted model mixes sample and specification changes. The main
   unadjusted coefficients are **1.7732277620** on the common sample and
   **1.7099116007** on all schools.

3. The adjusted association is **2.2079614065 final-score points per extra
   weekly tutoring hour**, HC3 SE **0.1628171337**, interval
   **[1.8874246799, 2.5284981331]**, t reference with **273** degrees of
   freedom. The known structural effect is **1.2**, below this interval.
   HC3 addresses heteroskedastic uncertainty for the fitted association;
   it does not eliminate latent motivation from the error.

4. Baseline achievement affects final score and is negatively related to
   tutoring assignment in the generator. Resources positively affect both.
   Their omitted contributions need not have the same sign. Adjustment can
   increase or decrease the tutoring coefficient. Without resources but with
   baseline score, the coefficient is **2.4998195328**; with both controls,
   it is **2.2079614065**. Motivation remains correlated with tutoring after
   adjustment, so neither control set restores the structural coefficient.

5. At prior score 60 and resource index zero, two tutoring hours predict
   **58.4524836839** and six predict **67.2843293098**. Their contrast is
   **8.8318456259**, equal to $4\times2.2079614065$. Mean intervals are
   **[57.347933, 59.557035]** and **[66.690614, 67.878045]**.
   The structural intervention contrast holding motivation fixed is
   $4\times1.2=4.8$, illustrating why the prediction difference is not a
   causal contrast. Interval endpoints cannot simply be subtracted because
   both predictions depend on shared estimated parameters; use the covariance
   of the appropriate linear contrast.

6. Residual plots can reveal curvature, unusual observations, or changing
   spread conditional on fitted values. They do not show whether the fitted
   tutoring coefficient absorbed latent motivation. Even an unremarkable
   cloud can coexist with endogeneity. The generator deliberately makes
   outcome noise grow with tutoring, providing a reason to use HC3 rather
   than treating equal variance as automatic.

7. An acceptable report states the synthetic observational scope, outcome
   and tutoring units, the 277-school complete-case sample, measured controls,
   HC3 interval, and sensitivity to resources. It describes a conditional
   association and documents remaining motivation confounding. A stronger
   policy design could randomly assign a tutoring offer and analyze the
   corresponding assignment effect with implementation and take-up evidence.
   Additional observed motivation measures could improve a selection argument
   but would not automatically reproduce random assignment.

**Prepared input:** [school_tutoring.xlsx](../labs/20-research-report/school_tutoring.xlsx). Use the stored observations for the published analysis. The [original generator](generators/20-research-report.py) supports new dataset editions; update results and answers when changing observations. Keep all 300 raw schools and 23 blank resource-index cells. The analysis selects 277 complete cases explicitly.

## Reproduction and verification evidence

Use `run_lab(output_dir="report-output")` for explicit exports. The
[full reference state](../labs/20-research-report/reference.json) retains all
four native fits, their complete covariance matrices and sample positions,
retained/excluded school identifiers, mean predictions, residual plot data,
and comparison rows. The publication table uses only the three common-sample
models. Each export preserves full scientific state rather than a bounded
display preview.

`manual_hc3` independently solves the four-column OLS problem, computes
leverage values, and forms the full HC3 sandwich. Coefficient discrepancies
are below $4\times10^{-14}$; full covariance discrepancies are below
$5\times10^{-13}$. The scenario contrast matches four times the slope
within $6\times10^{-15}$. Checks also verify the 300/277/23 sample accounting,
273 inference degrees of freedom, identical primary sample positions, and
exact full-JSON `ResultBundle` roundtrip.

Fresh source execution and Ruff passed. Actual `ConsoleSession` execution
completed without error and returned **model, table, table, plot**. Its
unique result global is `report_lab_result`. No exports are written by the
default run. These checks validate source-runtime calculation and persistence
serialization; they do not establish empirical causal validity, representativeness,
or behavior of a separate hosted or installed application.
