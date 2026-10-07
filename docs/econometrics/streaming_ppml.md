# Full-source PPML with absorbed fixed effects

`ppmlhdfe` has a native CPU float64 replay path for `Dataset` inputs. It
uses every retained observation at every likelihood and covariance step;
the maximum 400 rows in a saved result serve only to display fitted values.
No third-party estimator or optimizer runs inside this path.

```python
import openecon as oe

data = oe.scan("trade.parquet")
result = oe.ppmlhdfe(
    data=data, y="exports", x=["tariff", "distance"],
    absorb=["exporter_year", "importer_year"],
    cluster=["exporter", "importer"],
)
print(result.to_latex())
```

The shared discovery pass fixes treatment coding, missing rows, physical
row identities and frequency/analytic/probability weight semantics. An
owned SQLite selection table removes all-zero outcome FE levels and,
when requested, singletons until the complete global sample stops
changing. Frequency weights represent replicated rows for the singleton
rule. Zero-level and singleton counts refer to physical dropped rows;
`nobs` is the frequency sum under frequency weights.

At each IRLS step the working weights are `w * mu` and the working
response is `eta - offset + (y - mu) / mu`. Sequential disk vectors and
global weighted FE projections partial the outcome and regressors out;
global TSQR solves the remaining least-squares problem. The fitted full
linear predictor follows the Frisch–Waugh–Lovell identity. Step halving
uses the full Poisson deviance. The information matrix and score rows
are recomputed at the final weights. An offset or strictly positive
exposure is supported, as are nonnegative noninteger outcomes.

Nonrobust covariance is the inverse partialled information. Robust
covariance applies `N/(N-K)`; one/two-way cluster covariance uses global
cluster score sums with inclusion–exclusion and
`G_min/(G_min-1) * (N-1)/(N-K)`. If a two-way score meat is indefinite,
its positive-semidefinite projection is performed in the original
regressor score units before applying the bread, matching the native
in-memory convention. Coefficient tests use normal tails.

Absorbed degrees of freedom exclude a dimension nested in **any**
declared cluster column. When all absorbed dimensions are nested, one
degree is counted for the common constant. Two-dimensional connected
component redundancy is exact; the existing pairwise convention for
three or more dimensions can conservatively overstate the degrees lost.

The result retains the complete likelihood, deviance, constant-plus-offset
null likelihood, pseudo-R², slope Wald test, dropped-row counts,
source/selection fingerprints and changing-weight projection diagnostics.
JSON and LaTeX use the same `ResultBundle` contract as the native dense fit.

Numerical RAM is bounded by a snapshotted buffer budget, including both
discovery metadata, SQL caches, numerical row blocks and global factors.
Owned disk storage grows with retained rows times the design width; an
upfront conservative reserve covers up to 20 state vectors plus
selection metadata. Source changes, inadequate RAM/disk, absorbed or
singular regressors, insufficient clusters/degrees of freedom,
nonfinite working weights and nonconvergence produce explicit errors.
The budget estimates numerical buffers, not total process RSS or free
disk guarantees under simultaneous external writes. CPU projections and
iteration are not a general GPU claim or a throughput guarantee.

This implements the existing OpenEconometrics PPML scope. Fixed-effect levels
with all-zero outcomes are removed. General separation involving
regressors and FE combinations is refused when predictor drift persists
despite a flat deviance; it is not silently dropped or certified by the
complete `ppmlhdfe` separation algorithms. Heterogeneous absorbed slopes,
saved FE coefficients and general new-data FE predictions are outside
this native contract. The null model remains constant plus offset, not
the FE-only null reported by the external Stata package.

Independent tests compare explicit NumPy dummy-variable Poisson
equations, full information/robust/two-way covariance and cluster-order
invariant DoF; native parity covers one/two/three FE dimensions, weights,
categorical predictors, offsets/exposures, missing rows, pruning,
frequency replication, unit/partition invariance, full rows beyond the
display limit and explicit resource/provenance failure domains.

References: [Correia, Guimarães and Zylkin, *ppmlhdfe*](https://arxiv.org/abs/1903.01690)
and the authors' [method/options documentation](https://scorreia.com/help/ppmlhdfe.html).
These references motivate IRLS and FE absorption; the wider separation
and postestimation options documented there are not claimed here.
