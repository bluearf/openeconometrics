# Instructor notes · Lab 13

The [student handout](../labs/13-instrumental-variables/README.md) separates three dimensions: identifying information, instrument validity and uncertainty. Avoid presenting a rule of thumb such as F>10 as a universal validity certificate.

## Worked answers

1. Schooling is endogenous because ability enters both schooling and the outcome error. Background is observed exogenous; offer is excluded and independently randomized in the valid simulation.
2. Relevance: offer shifts schooling conditional on background. Exclusion: no direct offer effect on earnings. Orthogonality: offer is uncorrelated with structural outcome errors. Income support or job placement could violate exclusion despite randomization.
3. Strong IV .083863408, robust SE .015602498, CI [.053229880,.114496936], small-sample t(697). Exact geometric-mean comparison 8.748%, transformed interval [5.467%,12.131%].
4. F assesses relevance. Exactly one excluded instrument for one endogenous variable leaves zero overidentifying restrictions.
5. Weak F=.562852; conventional coefficient .160457506, SE .289287833, nominal CI [-.407522514,.728437527]. Do not promise nominal coverage under weak identification or infer a zero structural effect from non-rejection.
6. .180348071-.083863408=.096484663, equal here to .15/1.554651235. It follows from the controlled exactly identified scenario; unknown direct effects cannot generally be repaired from an estimated first stage alone.
7. The structural residual uses observed schooling, not predicted schooling. First-stage estimation and the instrument-projected score structure determine IV covariance.
8. Require assignment records, exclusion argument, first-stage evidence, complete outcome follow-up, sample-selection assessment and appropriate weak-instrument sensitivity.

## Reproduction and verification

The [full reference](../labs/13-instrumental-variables/reference.json) retains strong, weak, exclusion-violating IV, OLS and first-stage models. The [prepared workbook](../labs/13-instrumental-variables/schooling_instruments.xlsx) stores the weak scenario in `schooling_weak`/`log_earnings_weak` and the exclusion-violating scenario in `schooling_invalid`/`log_earnings_invalid`. The analysis source loads these fixed values, retains the original five-column order and replaces schooling/earnings for each fit. The [original generator](generators/13-instrumental-variables.py) preserves `make_data(relevance=.12)` and `make_data(direct_effect=.15)` for rebuilding those exact scenarios. Default execution writes nothing; `run_lab(output_dir="lab13-review")` exports full reference and table.

Independent moment equations, observed-regressor structural residuals and robust sandwich calculations verify all three 2SLS coefficient vectors and covariances. Here $Z$ and $X$ have the same number of columns, so exact identification gives $\hat\beta=(Z'X)^{-1}Z'y$. The script solves these equations directly and uses $A=(Z'X)^{-1}Z'$ to form $\widehat V=\frac{n}{n-k}(A\operatorname{diag}(\hat u))(A\operatorname{diag}(\hat u))'$. This is algebraically the same 2SLS sandwich and avoids projected normal equations that amplify floating-point error in the deliberately weak example. It is specific to this exactly identified design. A separate 60-digit calculation checks the complete coefficient vectors and covariance matrices; the numerical tolerances remain unchanged.

A separate HC1 OLS first-stage restriction test reproduces native robust F. The source also checks common primary sample positions, declared t(697) and full JSON roundtrip. The original projected-normal-equation reference had condition number about $1.57\times10^7$ in the weak case; the direct moment system is about $9.82\times10^3$. The stored model results, workbook, seeds and student conclusions are unchanged.

Native displays are `ResultBundle`, `OLSResult`, `ResultBundle`, `ResultBundle`, `Latex`. The comparison's weak-case interval must remain labeled conventional; its covariance robustness does not imply weak-identification robustness.
