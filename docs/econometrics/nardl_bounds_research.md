# NARDL bounds calibration: research decision and acceptance specification

Research checked on 2026-10-05. This document specifies work that remains open;
it does not introduce a testing API or validated critical values.

## Current decision

OpenEcon estimates the NARDL model and reports its joint levels F statistic and
adjustment t statistic. Their `critical_values`, `p_value` and `decision` remain
`None`; `bounds_critical_values_validated=False` and
`stata_parity_validated=False` remain explicit. Automatic NARDL cointegration
inference is **not completed**. An experimental F-only result would leave
important degeneracy checks unresolved and is not being exposed as a public
decision.

The existing [multiplier bootstrap](nardl.md) simulates outcomes using the fitted
alternative, conditions on the observed explanatory paths and selected lags,
and refits the model to estimate pointwise response uncertainty. Its confidence
bands answer a different question from a test that removes the levels
relationship. Renaming those draws as bounds calibration would not impose the
required null hypothesis.

## What the original sources establish

### Original nonlinear ARDL framework

[Shin, Yu and Greenwood-Nimmo (2014)](https://ssrn.com/abstract=1807745),
sections 2.3 and 3, explain that the dependence between positive and negative
partial sums affects bounds distributions. Their discussion of a single raw
regressor places its effective dimension between the ordinary one- and
two-regressor calibrations, rather than establishing independent expanded
regressors.

Their simulation also includes a legitimate **conditional null bootstrap**:
the restricted equation (3.24) removes lagged levels while retaining current
positive and negative differences. It uses restricted coefficients,
unrestricted residuals, 500 draws, known initial conditions and a known
explanatory path. Thus holding the explanatory path fixed is not intrinsically
invalid. The existing alternative multiplier bootstrap is different because
it does not impose that null. The narrow original experiment does not by
itself validate a general three-test implementation with arbitrary controls,
serial dependence, symmetry constraints and deterministic cases.

The dated author manuscript was checked in a
[mirror of the original text](https://www.scribd.com/document/1043903146/ssrn-1807745);
the [author publication page](https://www.greenwoodeconomics.com/publications.html)
identifies the published chapter, doi:10.1007/978-1-4899-8008-3_9. The publisher
chapter and direct SSRN delivery were not accessible in this research session.

### Bootstrap ARDL and degeneracy

[McNown, Sam and Goh's author working paper](https://www.colorado.edu/economics/sites/default/files/attached-files/wp16-08.pdf),
sections 2 and 3.2, precedes their 2018 Applied Economics article,
doi:10.1080/00036846.2017.1366643. It motivates three tests: joint lagged levels,
lagged outcome adjustment, and lagged explanatory levels. Rejection of only
the first two can confuse a stationary outcome without an explanatory levels
relationship with cointegration. The simulation procedure estimates a
restricted outcome equation and a marginal explanatory system, resamples
innovation vectors, and recursively generates levels. That working-paper
algorithm uses the joint levels restrictions for its auxiliary t statistics;
it is not identical to the test-specific null procedure below.

### Conditional ARDL with test-specific nulls

[Bertelli, Vacca and Zoia (2022), author preprint](https://arxiv.org/pdf/2204.04939),
section 3, accompanies Economic Modelling 116, 105987,
doi:10.1016/j.econmod.2022.105987. It fits separate restricted outcome equations
for the three nulls and estimates a marginal explanatory model. It resamples
innovation vectors, recentres bootstrap samples, generates the explanatory
and outcome paths recursively, and refits unrestricted conditional equations.
Current explanatory differences remain in the conditional equation. Its
factorization requires the stated absence of outcome-to-regressor Granger
causality. The detailed implementation concerns deterministic cases II and
III. This is a linear ARDL paper, not a proof for nonlinear sign-generated
regressors or for all of OpenEcon's supported NARDL specifications.

[Palm, Smeekes and Urbain (2010), publisher manuscript](https://cris.maastrichtuniversity.nl/ws/files/72516574/smeekes_2010_a_sieve_bootstrap_test.pdf),
Econometric Theory 26, 647–681, doi:10.1017/S0266466609990053, provides an
asymptotic basis for a multivariate sieve bootstrap in a conditional error
correction model under its linear-process assumptions. Those assumptions and
its triangular integrated system do not automatically prove validity after
nonlinear positive/negative transformations.

## The three hypotheses to implement together

For an error correction equation written as

\[
\Delta y_t=d(t)+\rho y_{t-1}
+\sum_j\left(\theta_j^+x_{j,t-1}^+
+\theta_j^-x_{j,t-1}^-\right)
+\text{short-run differences}+\varepsilon_t,
\]

the prospective procedure must distinguish:

| Statistic | Null restriction | Relevant tail |
| --- | --- | --- |
| Joint levels F | \(\rho=0\) and every \(\theta_j^+=\theta_j^-=0\), including restricted deterministic terms when the chosen case requires them | Upper |
| Adjustment t | \(\rho=0\), against negative adjustment | Lower |
| Explanatory levels F | Every \(\theta_j^+=\theta_j^-=0\) | Upper |

Names should describe degeneracy rather than using “type 1/type 2”, whose
numbering varies between sources. An outcome adjustment without explanatory
levels, and explanatory levels without outcome adjustment, must be reported
separately. Failure to reject a null is not proof that its coefficients vanish.

The proposed implementation will select and record a **test-specific null**
contract. It will not silently combine one paper's common-null resampling with
another paper's interpretation. Intercepts, trends, current differences,
degrees of freedom and covariance formulas must be specified for each test.
Ordinary regression F reference distributions must not supply its p-values.

## Proposed bounded implementation, not an implemented method

The following is an OpenEcon research and engineering proposal. It is not a
claim that the cited papers establish this NARDL extension.

Start with one raw I(1) explanatory variable, deterministic case III, explicitly
fixed lag orders and an identified marginal model. Initially exclude extra
controls, imposed symmetry, lag reselection, I(2) paths, feedback and unexplained
serial correlation. Excluding these features from calibration does not remove
them from model estimation or multiplier inference.

For the first research experiment, use an explicitly declared raw-increment
model with independent identically distributed innovations. A later serially
dependent version requires a separately validated marginal or sieve model;
residual independence is not established by simply offering a wild bootstrap
switch. A marginal rank or integration assumption must be supplied and
recorded rather than inferred silently from a low-power preliminary test.

The joint experiment must bootstrap the **raw explanatory variable**, then
recreate its positive and negative partial sums from each generated path.
Independently bootstrapping the two partial sums violates their construction.
Fitting a linear VECM directly to the expanded partial sums is also not an
automatic substitute: for serial raw increments the conditional expectation of
their positive part need not be linear. This last observation is a mathematical
reason to validate the marginal construction, not a result claimed by the
linear ARDL papers.

For each of the three nulls, fit the appropriate restricted outcome DGP,
estimate and document the marginal raw-variable DGP, and use paired innovation
indices so the intended contemporaneous dependence is preserved. Conditional
residual orthogonalization, recentering and finite-sample scaling must be
derived for that construction. Rebuild the raw paths, signed differences,
partial sums, lag design and unrestricted test statistic in every replication.
Record the initial-condition rule; a fixed prefix and randomly selected
original blocks are distinct methods, not interchangeable implementation
details.

In levels parameterization the adjustment null is affine:
\(\sum_i\phi_i=1\). The present homogeneous symmetry constraint basis is not
enough to impose it. A null-imposing solver must support affine restrictions,
and a unit root required by the null cannot be rejected using the stationary
alternative guard in the multiplier bootstrap. A finite explosive or singular
draw needs an explicit policy and diagnosis; rejecting and replacing draws
until a convenient statistic is obtained changes the simulated distribution.

Use bounded float64 Torch batches, native stable recursion and QR/SVD refits.
Preserve the local random generator, original-sample fingerprint, exact
restriction matrix, null coefficients, lags and marginal-model metadata.
Return the chosen upper/lower-tail calculation, ordered-quantile convention,
replication count and Monte Carlo uncertainty. If the proposed p-value uses a
plus-one correction, document it as an implementation convention rather than
attributing that formula to a source that does not use it.

This work requires memory and work limits. It is not a streaming NARDL refit
or a promise of unrestricted row counts. The output will identify its narrow
calibrated class before any broader automatic decision is permitted.

## Acceptance gates before public inference

### Numerical equivalence

An independent NumPy QR/SVD oracle must reconstruct the restricted equations,
raw explanatory paths, signed partial sums, recursive outcomes, unrestricted
refits and all three test statistics. It must not call the implementation's
Torch helpers. Check each affine restriction directly and verify
\(\Delta x_t^+\geq0\), \(\Delta x_t^-\leq0\),
\(\Delta x_t^+\Delta x_t^-=0\) and
\(x_t=x_0+x_t^++x_t^-\).

Tests must also establish batch invariance, RNG isolation, row/time alignment,
hold-back and initial-condition handling; paired innovation dependence;
nonfinite/rank failures; and serialization of the actual null and marginal
construction. A numerical oracle establishes implementation correctness,
not inferential validity.

### Statistical validity in the declared class

Predeclare a simulation grid and acceptance criteria before looking at the
results. Include unrelated integrated paths, stationary outcomes without
explanatory levels, explanatory levels without negative outcome adjustment,
and stable asymmetric cointegrating alternatives. Vary sample length, lag
orders, drift, near-unit roots, innovation skewness and innovation correlation.
Serial dependence, heteroskedasticity, multiple regressors, marginal
cointegration ranks and mixed I(0)/I(1) inputs require new grids before being
added to the calibrated scope.

Use enough independent outer simulations to attach binomial confidence
intervals to rejection rates, and enough inner bootstrap draws to measure
tail precision. Report every predeclared cell, failed calls, seeds, runtime,
memory and sensitivity to replication counts. Assess all three tests and the
combined classification, including both degeneracies. A convenient F-only
size result or selected successful cell cannot close this gate. Simulation
evidence must be described as such; it does not replace an asymptotic argument
for a broader DGP class.

### Release contract

Only the experimentally and theoretically supported specification may obtain
critical values or a decision. Unsupported cases retain `None` with a clear
reason. Include the source method and version, separate-null restrictions,
marginal model and assumptions, initial conditions, tails, precision and
calibration scope in JSON and LaTeX output. Neither automatic Stata parity nor
general NARDL bounds coverage follows from coefficient agreement.

## Concrete next step

Build a **private research harness** for the bounded case III/raw I(1) design,
with three separate nulls and independent recursive oracles. Reproduce the
original single-regressor conditional F experiment as a distinct benchmark,
then test the proposed joint-sign reconstruction and degeneracy classification.
Until those gates pass, retain the existing public guard and mark the NARDL
cointegration-calibration item as partial.

## Private research harness implemented on 2026-10-05

The engineering portion of the preceding proposal now exists in
[`benchmarks/research/nardl_bounds.py`](../../benchmarks/research/nardl_bounds.py).
It is **not imported or exposed by the OpenEcon library** and does not change
public NARDL critical values, p-values or decisions. Its output is explicitly
marked private, with `inferential_validity_established=False`.

The three-test experiment supports one declared raw I(1) predictor, case III,
fixed outcome/positive/negative lag orders from 1 to 4, no controls or imposed
symmetry, and an assumed iid raw-increment process with fitted drift. Requiring
both explanatory lag orders to be at least one keeps current positive and
negative differences independently available under the levels restrictions.
The integration and iid assumptions are supplied research assumptions, not
conclusions from an automatic pretest. All fits use float64 CPU Torch, an
affine null-space solution with a profiled free intercept, scaled QR refits,
and classical unrestricted covariance for the joint F, lower-tail adjustment
t and explanatory-levels F statistics. Restricted fits impose respectively
\(\sum\phi=1,\sum\beta^+=\sum\beta^-=0\),
\(\sum\phi=1\), and \(\sum\beta^+=\sum\beta^-=0\).

For each null, paired conditional-outcome/raw-increment innovations use the
same sampled row indices. The raw explanatory path is rebuilt first, then
both signed partial sums are reconstructed from that path. Conditional
residuals must be orthogonal to the raw innovation pool through the retained
current-difference design. This checks the fitted construction; it does not
prove innovation independence. Outcomes are generated by a trusted native
TorchScript recurrence; no stationary-root screening or companion-power
approximation alters the required unit-root null. Every invalid numerical
draw invalidates its **entire** call; there is no rejection, replacement or
success-only denominator.

The research conventions are recorded, rather than silently equated with the
linear paper. `recenter="draw"` implements the per-resampled-vector centering
in [Bertelli, Vacca and Zoia's equations 21–22](https://arxiv.org/pdf/2204.04939);
`pool` retains the centered empirical pools. Initialization can use the original
fixed prefix or an original contiguous block of length equal to the largest
specified lag, retaining that block's original signed-level anchors. Optional
residual scaling uses \(\sqrt{n/df}\) for the conditional pool and
\(\sqrt{n/(n-1)}\) for the raw pool. Scaling, differing lag lengths and
plus-one Monte-Carlo tail counts are explicit research conventions, **not**
claims that the paper proves this nonlinear extension. Local RNG isolation,
sample/index fingerprints, exact affine constraints, residual df, tails,
discrete quantile ranks, Monte-Carlo uncertainty and tensor/work budgets are
included. Tensor estimates exclude private BLAS workspace, allocator overhead
and Python/JSON output objects; they are not resident-memory guarantees.

A separate `conditional_f_benchmark` holds observed explanatory paths and
the original outcome prefix fixed and uses restricted joint-null coefficients
with **unrestricted** residuals. It checks the mechanics discussed in the
original NARDL experiment. Empirical input fitting, centering, seeds and tail
conventions here do not reproduce the authors' original parameter grid or
published tables. The benchmark supplies no three-test classification and is
not a public F-only inference API.

Independent verification is in
[`tests/test_econ_nardl_bounds_research.py`](../../tests/test_econ_nardl_bounds_research.py).
Its NumPy full-parameter affine SVD and sequential recurrence do not call the
Torch fit/path helpers. Tests cover coefficients, covariance and all three
statistics; paired raw/sign reconstruction; separate null constraints;
fixed/block prefixes; per-draw/pool centering; residual scaling; multiple
lag shapes; batch invariance; local RNG; time alignment including signed and
unsigned integer extrema; rank/nonfinite guards; structured native failures;
and complete failed-call denominators. Passing these tests establishes
numerical behavior, not statistical validity.

The predeclared 15-cell grid and future acceptance requirements are in
[`benchmarks/research/nardl_bounds_grid.py`](../../benchmarks/research/nardl_bounds_grid.py).
It includes full levels-null paths, stationary outcomes without explanatory
levels, stable asymmetric alternatives, drift, skewness, structural shock
correlation, lag differences and weak adjustment. A levels-present/no-adjustment
cell generally generates an I(2) outcome with this raw I(1) predictor; it is
explicitly an **out-of-scope stress**, excluded from inferential calibration.
Original case III intercepts absorb the observed signed-level origin after
the declared burn-in, and that adjustment is recorded.

The initial smoke uses 12 outer samples and 99 inner draws per cell. Both the
fixed/pool/df and original-block/draw/no-scaling runs completed 180/180 whole
calls with no failures or replaced draws. Under the full-null T=80 cell each
had 2/12 combined rejections: the 95% Wilson interval is approximately
\([0.0470,0.4480]\), far too wide to establish nominal 5% size. At T=160 the
stationary/no-explanatory-levels cell had 12/12 adjustment and joint rejections,
but 0/12 explanatory-levels and combined rejections, whose interval is
\([0,0.2425]\). The strong asymmetric T=160 cell had 12/12 combined rejections,
interval \([0.7575,1]\); weak adjustment remains difficult. These are reported
smoke observations, not acceptance results. Complete outputs and further
one-setting-at-a-time/inner-count/independent-seed smoke checks are retained
under ignored `artifacts/completion-wave5-2026-10-05/nardl-bounds/`.

The prospective calibration calls for at least 500 outer samples and 999
inner draws, multiple nominal levels, independent seed repeats and declared
convention sensitivity; it also requires a nonlinear-bootstrap validity
argument or independently justified alternative. Serial dynamics, multiple
raw regressors, symmetry constraints, other deterministic cases and automated
lag selection remain outside this experiment. **NARDL automatic cointegration
calibration remains partial; no public bounds decision has been enabled.**
