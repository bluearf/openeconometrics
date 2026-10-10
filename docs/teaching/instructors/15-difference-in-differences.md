# Lab 15 · Before and After a Policy Reform — Instructor guide

## Teaching notes and worked answers

Begin by asking whether “treated regions still have lower employment” rules
out an effective program. Let students commit to a prediction before showing
the four-cell table. Give them the data-generation assumptions before discussing
the diagnostic warning; otherwise they may mistake the known simulation design
for something an empirical pretest established.

1. Using full precision, treated change is 6.2682245504 and control change is
   3.7189051336; their difference is **2.5493194168**. Rounded table entries yield
   approximately the same value, with small rounding discrepancies.
2. An appropriate sentence is: “Treated-region employment improved by about
   2.55 percentage points more than employment in comparison regions.” Causal
   language additionally requires the stated assumptions. The post-only gap
   mixes permanent level differences with treatment exposure.
3. The interaction coefficient is **2.5493194168**. In the saturated regression,
   `treated` captures the baseline group gap and `post` the common average
   change. Full region effects span the time-invariant group indicator; full
   year effects span the post indicator, so those terms add no independent
   variation to the two-way fixed-effects model.
4. Changing covariance leaves the fitted coefficient unchanged. HC1 treats
   distinct region-year disturbances as uncorrelated; region clustering allows
   arbitrary within-region dependence. Our generator deliberately creates
   persistent within-region errors, so the clustered choice follows the
   design. Here HC1 gives SE **0.300** and interval **[1.958, 3.140]**, narrower
   than the region-clustered interval. This ordering is a feature of this sample,
   not a theorem that clustered standard errors are always larger.
5. The linear and joint-leads tests give different sample evidence, with the
   latter flagging a concern at 5%. Neither observes untreated post-policy
   outcomes. Non-rejection can reflect limited power; rejection can occur by
   chance. Neither result rules out a shock that begins exactly at treatment.
6. The added shock shifts the estimate by exactly **2.000** points and leaves
   its standard error and both pre-period diagnostics unchanged. Useful
   evidence would include policy timing and implementation records, other
   regional reforms, alternative unaffected comparison outcomes or groups,
   and mechanisms that distinguish the subsidy from the concurrent event.

## Reproduction and verification

The core check uses region-cluster scores after two-way demeaning, not the
native regression's covariance helper. The script also verifies all 360 sample
positions, no dropped rows, 59 inference degrees of freedom, and the exact
two-point contamination shift. These checks verify computation and sample
alignment; they do not certify a real-world research design.

For saved results, call `run_lab(output_dir="lab15-output")`. It writes full
model state to `reference.json` and an OpenEconometrics publication table to
`table.tex`. The latter is a LaTeX fragment requiring `booktabs` and `adjustbox`,
not a standalone document. The checked-in [reference results](../labs/15-difference-in-differences/reference.json)
and [LaTeX table](../labs/15-difference-in-differences/table.tex) retain the lab's actual fit and uncertainty.

For implementation details, see the [native DiD documentation](../../econometrics/teffects.md#2-difference-in-differences-didregress).

**Prepared input:** [policy_panel.xlsx](../labs/15-difference-in-differences/policy_panel.xlsx). Use the stored observations for the published analysis. The [original generator](generators/15-difference-in-differences.py) supports new dataset editions; update results and answers when changing observations. The employment_no_policy column is a labeled teaching oracle and is excluded from model fitting.

Run the complete lab with the supplied workbook before changing a specification. Compare the four group-period means, all model coefficients and the full covariance with the saved reference, then check the 360 retained observations and the 60 region clusters. Reopen an exported model state and compare predictions with the original fit. A matching coefficient alone would miss a wrong sample, cluster correction or saved-state interpretation. When trying a different covariance convention, keep the observations and treatment coding fixed and report the convention beside the interval.
