# MARKET-112: scoped correction proposal, not a validated calibration

2026-10-07. The frozen negative calibration receipt (internal evidence excluded from this public snapshot)
and its 133 passing numerical tests remain unchanged. No new candidate has been
simulated, selected or published. Public NARDL critical values, p-values and
decisions remain disabled.

## Diagnosis supported by evidence

The outer DGP is reproduced independently as a difference ECM in all 15 cells.
Full-parameter NumPy SVD reproduces Torch QR's coefficients, covariance,
restrictions, all statistics and all 24 complete 999-draw tail counts. Raw/sign
paths obey their exact sum identity, including original-block anchors. Both
centering/scaling conventions fail the size screen. This rules out a detected
arithmetic, sign-path or restriction-matrix error. It does **not** prove that
finite-sample studentization contributes nothing to rejection inflation.

A maintained-class issue can be derived exactly in the failing fixed-lag case.
Write the signed-level vector as `z`, its centered increment covariance as
`Sigma_z`, and the conditional ECM as

```text
Delta y_t = c + rho*y_(t-1) + theta'z_(t-1) + delta'Delta z_t + e_t.
```

For standard-normal raw increments, `Sigma_z` has diagonal
`1/2-1/(2*pi)` and off-diagonal `1/(2*pi)`. Its eigenvalues are `1/2` and
`1/2-1/pi`, both positive. Although the signed increments come from one raw
innovation, they have no nonzero stationary linear combination.

When `rho=0`, the state transition is `A=I+N`, with only the first row of `N`
nonzero (`theta'`). Then `N^2=0` and `A^k=I+k*N`. Nonzero `theta` produces an
outcome impulse response that grows linearly with horizon; its leading variance
is `theta'Sigma_z*theta*T^3/3`. Thus the outcome is generally I(2), rather than
the maintained I(0)/I(1). Even the local sequence `theta=c/T` contributes order
`T` outcome variance, so small fitted level coefficients cannot simply be
ignored as harmless numerical error. This is an algebraic scope argument,
not a proof of the bootstrap's limiting distribution.

Accordingly, the adjustment null within this **full-rank signed-I(1)** class
must impose `rho=0` **and** `theta=0`. Its restriction matrix in the current
levels parameterization is the joint affine constraint, while its statistic
remains the separate lower-tail adjustment t statistic. Dropping the adjustment
test or replacing all inference with the overall F test would not resolve the
issue.

The explanatory null `theta=0` is harder: it permits a unit-root outcome
(`rho=0`) and a stationary outcome (fixed stable `rho<0`). A freely fitted root
in the boundary case can import a random local-to-unity nuisance into the
bootstrap. [Park's primary unit-root bootstrap analysis](https://www.ruf.rice.edu/~econ/papers/2003papers/04Park.pdf),
section 2.2, imposes the unit-root null. Its theorem does not establish validity
for this NARDL composite null. Likewise, the
[conditional ARDL paper](https://arxiv.org/pdf/2204.04939), equation 30 and
Appendix A, uses stationary or cointegrated explanatory systems, whose rank
structure differs from these full-rank signed partials.

## Gates before implementing a new candidate

1. Derive the complete maintained null parameter spaces, including integration
   rank, stable short-run dynamics and signed-partial drift. Preserve separate
   overall F, adjustment t and explanatory F targets; the adjustment restriction
   above is a necessary candidate correction, not a complete three-test solution.
2. Establish an explanatory-null bootstrap argument covering both stationary
   and unit-root outcome branches and relevant local-to-unity sequences. A
   persistence pretest, arbitrary root threshold, or maximum of two endpoint
   p-values needs its own validity argument; none is authorized as an automatic
   correction. If a nuisance supremum is proposed, show why it covers the
   **whole** null space, rather than only a convenient finite grid.
3. Derive joint raw/sign empirical-process validity, including the two opposing
   signed-partial drifts. Reconstructing signs correctly is necessary but does
   not replace this distributional proof. Independently reproduce the proposed
   null equations, statistics and composite-null treatment before calibration.
4. Only after these gates are met, freeze a new source-pinned protocol before
   observing any candidate outcomes. Proposed fresh seed streams are
   `68232739` and `145397399`. Keep the existing 15-cell grid, 500 outer samples,
   999 inner draws per null, alpha 1%/5%/10%, both declared conventions, full
   failure denominators and existing size/power gates. Add boundary/local-root
   cells; do not remove the stationary-outcome degeneracy cells or the failed
   auxiliary tests. Label the I(2) cell as excluded stress.
5. Run an independently implemented nonlinear size replication with distinct
   RNG streams. Passing Monte Carlo screens alone cannot stand in for the
   missing composite-null argument. Publish only the precise supported class
   after all gates pass, and reject other cases explicitly.

No credible complete correction has met these gates today. MARKET-112 remains
pending research. Neither inflated p-values, relaxed acceptance thresholds,
chosen favorable conventions nor linear bounds tables resolve the failure.
