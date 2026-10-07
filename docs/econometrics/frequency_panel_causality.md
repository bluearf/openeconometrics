# Frequency and heterogeneous panel predictive noncausality

These two test procedures use OpenEconometrics CPU float64 Torch QR and return
`TableSet` tables. They do not fit through a third-party estimator, infer
structural effects, collect a `Dataset`, choose lags, or silently drop rows/units.
Each records the complete ordered model-input sample hash, lag roles, covariance
divisor, reference distribution, declared assumptions and workspace estimate.
They support saved console tables and LaTeX. Neither has licensed Stata parity.

## Breitung–Candelon

```python
result = oe.bccaustest(data=df, y="output", x="rate", time="quarter",
                      lags=4, frequencies=[0.2, 0.8, 2.4])
result["tests"]
```

For a bivariate VAR(p), the null on the cause coefficients beta is
`sum(beta_j*cos(j*omega)) = sum(beta_j*sin(j*omega)) = 0`. The restrictions
are those in [Breitung and Candelon (2006)](https://doi.org/10.1016/j.jeconom.2005.02.004),
also equations (12)–(15), page 9 of their
[author preprint](https://edoc.hu-berlin.de/bitstreams/bf1d9a80-bccb-4ef1-8d16-121eeb144918/download).
Wald W divided by two equals the ordinary two-restriction F statistic. The
classical covariance divisor is `usable_periods - 2*p - int(constant)`, where
`usable_periods = raw_periods - p`. The paper's no-intercept `F(2,T-2p)` uses
regression T; an included intercept consumes one additional degree of freedom.
The F reference is approximate for dynamic regressions, not an exact finite-T law.

The validated contract is a declared stationary, fixed-lag bivariate model with
iid homoskedastic innovations, `p >= 3` and interior frequencies `0 < omega < pi`.
`integration_order=0` declares stationarity; other orders are rejected. Fitted
unstable roots and numerically rank-deficient near-endpoint restrictions are
rejected. Fitted stability does not prove DGP stationarity. Cointegrated variants
of the paper, endpoints, multivariate conditioning, trend/HAC/automatic lag choice
are outside this contract. Frequencies are radians per observation; the cycle
period `2*pi/omega` is reported in observation units. All decisions are pointwise;
selecting a frequency after looking at these rows changes the inference problem.

The tests, full two-equation coefficients and cosine/sine restrictions are saved
separately. Numerical evaluation orthonormalizes the restriction row space by SVD,
avoiding loss from squaring its condition number. The independent development
oracle evaluates the original restriction matrix directly and fits a constrained
model by an independent null-space projection, comparing ordinary F, W/2, df,
coefficients, standard errors and SciPy reference tails.

## Dumitrescu–Hurlin

```python
result = oe.dhcausality(data=df, y="output", x="investment",
                        panel="country", time="year", lags=2)
result["tests"]
```

The heterogeneous regressions include each unit's own intercept and K lags of
both variables. The null excludes the cause's first K coefficients in **every**
unit; rejection supports predictive causality in at least some units. It does
not label every unit causal. W_i uses `SSR_i/(usable_periods-2*K-1)` covariance
and is K times the ordinary unit F statistic. `Wbar = mean(W_i)` is an
intermediate statistic with no manufactured p-value. `Zbar` and `Ztilde` are
separately reported, with Ztilde the designated decision.

For U usable periods, the paper's approximate moments are

```
mean(W_i) = K*(U-2*K-1)/(U-2*K-3)
var(W_i)  = 2*K*(U-2*K-1)^2*(U-K-3)/((U-2*K-3)^2*(U-2*K-5))
Ztilde    = sqrt(N/var(W_i))*(Wbar-mean(W_i))
Zbar      = sqrt(N/(2*K))*(Wbar-K)
```

They replicate equations (5), (11), (13)–(16) of the
[authors' working paper](https://shs.hal.science/file/index/docid/224434/filename/Causality_WP.pdf)
and the corresponding [2012 published method](https://doi.org/10.1016/j.econmod.2012.02.014).
The paper's T is the regression sample after removing initial K lags. Using raw T
instead introduces a finite-sample correction error. The
[plm authors' implementation](https://github.com/ycroissant/plm/blob/master/R/test_granger.R)
lines 100–105 and 193–194 use the equivalent raw-T substitution. Its *statistics*
are compared in the independent NumPy oracle. `plm`/`xtgcause` report two-sided
normal probabilities; this API follows the paper's upper-tail rejection rule.
That convention difference is explicit in result notes and is not Stata parity.

Both references require cross-section independence and jointly stationary,
correctly specified dynamics. The declared options are
`integration_order=0`, `cross_section="independent"`; other scopes are rejected.
Innovations are iid homoskedastic Gaussian within units, with heterogeneous unit
variances allowed. For Zbar, time tends to infinity before the number of units.
Ztilde uses an **approximation** to finite-T moments: dynamic regressors prevent
an exact fixed-T reference, as discussed in
[Juodis, Karavias and Sarafidis (2021)](https://doi.org/10.1007/s00181-020-01970-9).
No guarantee across all stationary dynamic DGPs follows from the simulation grid.

The supported panel is complete, balanced, common fixed K with an intercept,
the same regular grid in all units, and raw T strictly larger than `3*K+5`.
No cross-dependent bootstrap, unit-specific lag selection, unbalanced extension,
trend, weights or unit roots are claimed. Missing values, gaps, repeated keys,
nonfinite inputs, misaligned clocks, singular unit designs and perfect fits fail
the whole call; no unit is omitted. Exact signed/unsigned period keys beyond
2^53 and regular monthly/business-day dates are preserved.

## Acceptance evidence

`tests/test_econ_frequency_panel_causality.py` evaluates original-paper restriction
and moment formulas independently with NumPy QR/projection and SciPy tails.
It also checks affine scale invariance, calendar/key sorting, complete sample
hashes, scope rejection, JSON metadata, tables and LaTeX.

`benchmarks/research/frequency_panel_causality.py` declares four cells per method,
at least 500 outer replications for each of two independent seeds and alpha .01/.05/.10.
BC includes a nonzero lag polynomial with an exact zero at omega=.8, preventing a
global-causality-only shortcut. DH varies N/T/K and includes half-causal panels.
Its heterogeneous DGP is inspired by the paper's benchmark and is fully specified
in the script; it is not represented as an exact published Monte Carlo table
reproduction with unspecified original initialization/x dynamics. Planned-call
denominators and both failure assignments are retained; failed draws are never
replaced. The declared gates at .05 are null Wilson95 upper <= .10 and power
Wilson95 lower >= .70, with zero failed calls. This is bounded size/power evidence,
not a proof of universal finite-sample validity. Source, frozen-runtime execution,
native editor Run, rendering and restart readback are separate receipts.

The initial 500-replication grid is retained as `size-power-pilot.json`. One DH
null cell (seed 90149, N=20, raw T=80, K=1) rejected 37/500 at .05, with
Wilson95 upper 0.100335, narrowly failing the unchanged 0.10 gate. The complete
grid was then extended to 1000 replications per seed and cell, retaining the
first 500 draws, both seeds, all DGPs and the original gates. This extension
was chosen after viewing the pilot; it is descriptive precision evidence,
not a separate independent confirmatory experiment or proof of exact size.
The pilot script's rejection-counter initialization was moved before the loop
while the pilot process was already running. There were no failed calls, so
this did not change its counts, but its benchmark-file hash recorded at the
end cannot establish the exact executed pilot program. The final 1000-run
script is frozen before execution; its source fingerprints are authoritative.
