# Structural microeconometrics research stages

MARKET-190 planning deliverable, 7 October 2026. Implementation remains tracked
in [GitHub #78](https://github.com/bluearf/openecon/issues/78). No new demand,
BLP, production-function or markup estimator is advertised by this plan.

The decision table in `reports/OpenEcon ekonometri tam ekleme planı.md`
(lines 192–194) places demand systems and BLP outside the current implementation
scope (Hayır), and OP/LP/ACF after demonstrated demand (Sonra). Those decisions
remain in force. The order below is a proposed research dependency order, not
a delivery date or authorization to change those decisions. Every stage stays
deferred under #78 until a separate product decision and its fixture gate pass.

## Existing foundations and remaining work

| Foundation | Reuse | Missing specialized contract |
| --- | --- | --- |
| `econometrics/systems` SUR, nonlinear systems | Equation samples, restrictions, full cross-equation covariance | Economic share restrictions, adding-up singularity and elasticity gradients |
| `econometrics/gmm` and `iv` | Moment weighting, rank checks, instruments, uncertainty | BLP inversion/integration, supply moments and nested derivatives |
| `econometrics/discrete` | Choice-data validation and stable likelihood arithmetic | Random coefficients over markets, outside shares and price endogeneity |
| `econometrics/panel`, `selection`, `frontier` | Panel calendars, sample accounting and diagnostics | Proxy/timing/exit selection; frontier inefficiency is not OP/LP productivity |
| ResultBundle, Dataset and resource guards | Persisted state and explicit execution domains | Saved system elasticities, inversion state and productivity/markup uncertainty |

## Deferred milestones and acceptance gates

| Stage | Scope and sample/options | Independent acceptance before support |
| --- | --- | --- |
| D1: LES and exact AIDS | Positive prices/expenditure; complete goods vector; identify LES subsistence parameters; impose adding-up, homogeneity and symmetry; drop one share equation and reconstruct it | Restrictions numerically and analytically; full cross-equation covariance, dropped-equation invariance, Marshallian/Hicksian elasticities with delta covariance; zero/negative price and singular-instrument failures |
| D2: translog and QUAIDS | Separate exact price-index and linear-approximation options; nonhomothetic quadratic expenditure, integrability/curvature domain | QUAIDS zero-quadratic AIDS limit, finite-difference elasticity Jacobians and covariance; original author benchmark; censored zeros/demographics/endogenous expenditure each require a later separately validated option |
| B1: random-coefficient BLP demand | Market/product/firm IDs, outside share, prices, instruments, draws and integration weights; start with demand only, fixed seeded quadrature | Zero-random-variance logit limit; share inversion residual, derivatives, objective and full GMM covariance against original-author reference and a pinned independent PyBLP run; weak instruments, tiny shares, bad weights and failed contraction must be reported |
| B2: supply, micro moments and merger scenarios | Only after B1: ownership, supply instruments, marginal-cost convention; later micro moments and explicit Bertrand counterfactual | Full implicit derivatives through inversion, cluster covariance, integration sensitivity; cost/ownership consistency, equilibrium residual and solver failures; uncertainty from full demand/supply covariance or market bootstrap |
| P1: OP, LP then ACF | Distinct models with stated input-choice timing and firm/year ordering; investment versus intermediate-input proxy; survival correction for OP; gross-output versus value-added explicitly separate | Reproduce author fixture and a seeded independently coded timing DGP; rank/monotonicity/functional-dependence checks; lags never cross gaps/firms; bootstrap resamples complete firms and repeats all nuisance stages |
| P2: productivity and markups | Only after P1: saved productivity/state mapping, variable input and expenditure share; distinguish revenue and quantity elasticity; multi-product allocation later | Author markup replication; elasticities and markup denominator checks, full generated-regressor uncertainty, firm bootstrap and sample alignment; no causal competition interpretation from a fitted ratio alone |

## Author sources and fixture access register

The source register was checked on 7 October 2026. A public paper or a visible
replication landing page is not a downloaded, licensed, runnable fixture.

| Method | Primary source | Fixture/access status and gate |
| --- | --- | --- |
| LES | [Stone (1954), publisher DOI](https://doi.org/10.2307/2227743) | Publisher route exists but retrieval was unavailable; locate original tables/series and document reconstruction and data rights before D1 |
| AIDS | [Deaton and Muellbauer, author publication page](https://deaton.scholar.princeton.edu/publications/almost-ideal-demand-system), [AEA paper](https://www.aeaweb.org/aer/top20/70.3.312-326.pdf) | Paper accessible; author machine-readable replication and redistribution license not established; synthetic restrictions tests cannot substitute for the author gate |
| Translog | [Christensen, Jorgenson and Lau (1975), publisher record](https://www.jstor.org/stable/1804840) | Catalog route verified; original code/data and redistribution rights not established |
| QUAIDS | [Banks, Blundell and Lewbel (1997), UCL record](https://discovery.ucl.ac.uk/id/eprint/14165/), [author-hosted paper](https://www.ucl.ac.uk/~uctp39a/Banks-Blundell-Lewbel-1997.pdf) | Publication identifies UK household survey analysis; original survey access and replication rights need confirmation; no household data copied |
| BLP | [Berry, Levinsohn and Pakes, NBER](https://www.nber.org/papers/w4264), [PyBLP automobile reference](https://pyblp.readthedocs.io/en/stable/_notebooks/tutorial/blp.html) | Independent implementation supplies product/agent example paths; its tutorial changes the original price-income specification. Pin version, dataset SHA, integration draws, specification and [software license](https://github.com/jeffgortmaker/pyblp/blob/master/LICENSE.txt); establish data redistribution rights separately |
| OP | [Olley and Pakes, NBER](https://www.nber.org/papers/w3977) | Original plant data availability/rights not established; author replication or documented restricted-data runner required |
| LP | [Levinsohn and Petrin, NBER](https://www.nber.org/papers/w7819) | Original-author code/data access not established; verify timing and proxy normalization before fixture acceptance |
| ACF | [Ackerberg, Caves and Frazer, publisher](https://onlinelibrary.wiley.com/doi/10.3982/ECTA13408) | Publisher lists supplementary material; supplement download, license and runnable programs not yet verified |
| Markups | [De Loecker and Warzynski, NBER](https://www.nber.org/papers/w15198), [author replication deposit V1](https://www.openicpsr.org/openicpsr/project/112552/version/V1/view) | Deposit lists DLW_AER0747 and LICENSE.txt; binaries, detailed license and data restrictions not downloaded/verified. This is a concrete fixture candidate, not a passed replication |

## Execution, persistence and promotion

Initial implementation proposals use native CPU float64 Torch. Optional external
software is a development oracle or a separately named integration, never a
silent runtime substitution. CUDA stays unvalidated (MARKET-53 remains open).
No method becomes supported merely because its name is in this document.

For every method, write an option matrix before coding: retained sample and
missing/zero policy, category encoding, weights, covariance correction/df,
identification, convergence and each saved postestimation target. Begin with
explicit guarded in-memory input. A Dataset route needs its own complete replay
contract; never collect it implicitly. Plan market × product × draw buffers,
nested inversion/optimizer iteration budgets and firm-bootstrap fits up front;
record refusals instead of truncating draws, goods, markets or firms.

Save economic restrictions, omitted equation, full covariance, original units,
instrument and sample hashes, integration nodes/weights/seed, convergence and
residual diagnostics, elasticity Jacobians and productivity timing/lag mapping.
Restored predictions and uncertainty must use this state without re-estimation.

Promotion gates are separate: (1) method/option decision and licensed fixture;
(2) source numerical, negative and author replication tests; (3) saved/read-back
state and result contracts; (4) exact-source frozen runtime; (5) installed
synthetic native execution and persisted reopened output; (6) licensed vendor
comparison when claimed; (7) public release validation. Record source SHA,
oracle/data versions and hashes, expected full inference, tolerances and failures
at every gate. Closing MARKET-190 completes this plan only; #78 retains all
implementation and unpassed fixture/promotion gates.
