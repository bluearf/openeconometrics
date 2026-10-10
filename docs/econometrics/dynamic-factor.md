# Identified dynamic factors: AR(1), known finite prior

`dfactor` estimates the Gaussian model

\[
y_t=d+Lf_t+e_t,\quad e_t\sim N(0,\operatorname{diag}(h));\qquad
f_{t+1}=Af_t+u_t,\quad u_t\sim N(0,Q).
\]

The caller declares factor count and anchor measurement names. Anchor loading rows equal the identity, fixing factor rotation, scale, sign and labels. Other loadings and intercepts are free; factor innovation covariance is unrestricted positive definite. The factor AR(1) has spectral radius strictly below one. Measurement noise is diagonal, white, independent of factor innovations. At least `2*r+1` measurement series are required. Positive definite, numerically resolved observed information supplies the final local identification gate.

The initial factor prior is **known and finite**, before the first measurement: by default `a0=0`, `P0=I`. The likelihood conditions on this prior. A stable transition does not turn that conditional likelihood into a likelihood with an estimated stationary prior. This stage does not estimate a Lyapunov covariance or the initial prior.

```python
from openecon.econometrics.tsworkflows.dfm import (
    dfactor, dfactor_restore, dfactor_nowcast, dfactor_forecast,
)

result = dfactor(
    data, ["activity", "production", "prices"],
    factors=1, anchors=["activity"], time="month",
)
reopened = dfactor_restore(result.to_json())
missing = dfactor_nowcast(reopened, [(40, "production")])
future = dfactor_forecast(reopened, 3)
```

All resident calendar rows remain in their original order, including interior/all-missing dates and ragged tails. Each date's likelihood uses only its observed measurements. A completely missing date contributes zero likelihood and still performs its transition. A completely missing measurement series is unidentifiable and refused. Declared dates must follow a regular numeric or datetime calendar; typed row identities, numeric source dtypes, response order, original observations and masks are saved. Without an explicit `time` column, input row order declares equally spaced unit periods; the source index is a row identity, not an inferred calendar.

## Estimation and uncertainty

Deterministic starting regressions use observed overlaps with the declared anchors. They initialize parameters only. EM uses the shared Kalman/RTS conditional factor moments and cross-date lag-one covariance; it never substitutes those initial proxies for missing observations. For each measurement it solves its expected intercept/loading regression using only observed cells. Anchor loading rows remain fixed. Factor `A/Q` use every adjacent latent date, including missing observation dates.

An exact M-step candidate can violate stability. A declared finite backtracking line search blends physical covariances/loadings and accepts only stable candidates whose actual observed-data likelihood does not decrease. Every accepted parameter vector, likelihood and fraction is saved. No variance floor, jitter, covariance clipping, pseudoinverse or removed date repairs the model.

EM starts are followed by native Torch analytic-score BFGS likelihood polishing, including at most ten declared native Newton polish steps. The result reports both phases and each restart's convergence/failure. The best converged interior restart is selected. Convergence is verified with the remaining observed-information Newton step, `score' I^-1 score <= max(tolerance**2, 1e-9)`. No global-maximum claim is made. Full observed information, score, parameter-chart covariance, Jacobian, physical parameter covariance, SE, normal z/p/CI and full retained state/disturbance moments are saved. Positive definite but unresolved near-singular information is refused after diagonal normalization of the declared parameter-chart information. This is an information rank-resolution rule, distinct from the coordinate-unit symmetry/positivity checks of an input covariance.

Factor intervals condition on fitted coefficients. Future measurement and historical missing-cell predictions return full joint conditional covariance and, by default, an additional first-order delta covariance for the conditional mean. That derivative includes the fitted terminal/smoothed state's dependence on coefficients. Process and parameter variance remain separate tables; their sum supplies approximate Gaussian intervals. This is a delta approximation, not integration over a posterior distribution of covariance/dynamic parameters.

`dfactor_nowcast` conditions on all observations saved in that result. A target is an original positional date and an unobserved response name. Vintage/publication filtering has not been applied by this API. The forthcoming release-vintage layer must select the eligible information set before fitting or conditioning.

## Persistence and resources

The public schema is `openecon.dynamic-factor.result.v1`, with numerical state `openecon.dynamic-factor.ar1-known-prior.v1`. JSON contains the complete source, assumptions, equations, optimization paths, OIM, covariance and Kalman/RTS output. Nested checksums protect transport integrity. Restore additionally reproduces source likelihood, posterior, OIM, every saved constrained EM step and every successful ML restart's final score/information endpoint without an optimizer or estimator call. ML counters and scalar objective history are bounded optimizer provenance: their strict schema, monotonicity, endpoint/start objective, iteration bounds and gradient/concavity fields are validated, while their intermediate optimizer iterates are not recreated. Typed identity/dtype and strict boolean/integer schema checks apply before reconstruction.

Production runs on native CPU float64 Torch. It preserves caller default dtype/device and RNG state. Inputs are resident DataFrames or bounded resident mappings; Dataset collection, generators, weights and GPU estimation are unsupported. DataFrames are projected to declared responses/calendar before coercion. Nonexact integer measurements are refused instead of silently rounding them to float64.

The first estimator admits 1–3 factors, 3–96 measurements, 4–20,000 dates and at most 256 parameters, subject to the tighter full-fit work and workspace plan. EM iterations, restarts, finite line-search trials, likelihood optimization, full Hessian/autodiff, retained disturbances and complete numerical serialization are preadmitted before scanning/coercing source data. Default full-fit work is 20 billion units; the explicit maximum is 200 billion. The workspace default is 512 MiB and remains configurable through the shared workspace API. These are buffer/work estimates, not total process RSS or statistical assumptions. Full joint prediction covariance admits at most 512 target cells and 128 future dates, with additional derivative/work admission.

The internal immutable factor admission profile allows the existing shared proper-prior engine up to 96 measurements and 128 states, bounded per pass. The public `kalman`/`smooth` signatures and the existing `sspace` limit of eight measurements and sixteen states remain unchanged. Its covariance recursions are shared, not duplicated. Later models must additionally preadmit their stationary Lyapunov or serial-state geometry before using this internal profile.

## Scientific references and acceptance boundaries

The observed-data latent Gaussian EM derivation follows the [Bańbura–Modugno working paper](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp1189.pdf), with the explicitly different known finite-prior and identity-anchor restrictions above. The [2014 JAE original data archive](http://qed.econ.queensu.ca/jae/datasets/banbura002/) publishes 86 already transformed ECB series; its readme explicitly omits fifteen purchased Markit/DataStream series from the paper's 101-series panel. The archive contains two data/list text files and no method-author implementation or reference outputs. It does not establish unrestricted access to the omitted series or full paper replication.

The original public archive SHA256 is `7f6d163f3a06eb551f9530d746acd5498f6b9fb498ad57313a527595f0c79ad3`; `bm-data.txt` is `7a4c66ef0431d4ca6beca7846c680d5081e03a0e726c4b3924f175631de7f348`, and the variable list is `d7342d54ee2675f9322a3f1a35e01fd9664b81840ac43f40ef7e38d513b9aeb5`. Data redistribution is not inferred from the package-code license. The reference acceptance packet must retain the exact original readme and data permissions separately.

Independent development references are native R/KFAS 1.6.0 (`GPL >= 2`) and [dfms 1.0.1](https://github.com/ropensci/dfms/tree/137a9a8d5a608a3e95f4015956be8742e2a789b1) (`GPL-3`), pinned to commit `137a9a8d5a608a3e95f4015956be8742e2a789b1`. These are package-author implementations, distinct from original BM method-author replication. The `dfms` retained-public-data extract has different dimensions/provenance; it cannot replace the omitted original panel. Production includes no R, SciPy or Statsmodels estimation dependency and copies no GPL package algorithm code.

`scripts/verify_dynamic_factor_oracles.py --state <actual-saved-result.json>` derives precisely the saved source/prior/system and compares actual cached fields with native R/KFAS, optional native dfms, an independent NumPy covariance filter, primitive full Gaussian conditioning and finite-difference OIM at two step sizes. It never regenerates a fit to substitute for saved native results. `dfms` predicts its initial state once before its first observation; the bridge explicitly converts and validates that convention, refusing an inadmissible conversion. Its partial-observation likelihood uses the full-series Gaussian normalization; the bridge reports the exact constant correction to the observed-cell convention.

Example/source-oracle commands:

```sh
PYTHONPATH=src python examples/dynamic_factor.py --output /tmp/dynamic-factor-example
PYTHONPATH=src python scripts/verify_dynamic_factor_oracles.py \
  --state /tmp/dynamic-factor-example/dynamic-factor.json \
  --rscript /opt/homebrew/bin/Rscript \
  --r-library /tmp/openecon-kfas-native-library \
  --dfms-library /tmp/openecon-dfm-native-library \
  --dfms-source /tmp/openecon-dfm-author-reference \
  --report /tmp/dynamic-factor-native-cached-oracles.json
```

Source/analytical and package-author numerical acceptance are separate from original-author/vendor acceptance, frozen packaging and installed native lifecycle/persistence. MARKET-647 is this AR(1), white-idiosyncratic, known-prior stage. MARKET-217 remains open for wider factor VAR orders, serial idiosyncratic AR states, stationary/initial-prior extensions, release-vintage/no-lookahead admission, monthly/quarterly temporal aggregation, and model-based news/revision decomposition. Neither stage nor parent implies blanket Stata parity.


## Original MARKET-647 acceptance remains unresolved

The original full MARKET-647 clause requires stationary factor and idiosyncratic AR structure and a lawful original Bańbura–Modugno author fixture. This delivered conditional AR(1) factor model uses a known finite initial prior and iid measurement errors; it excludes estimated stationary initialization and serial idiosyncratic AR measurement states. The independent KFAS/dfms and full Gaussian-conditioning evidence establishes the declared conditional model only. It does not complete the original MARKET-647 clause. Exact original clause text and source/data/license boundaries are retained; MARKET-647 and parent MARKET-217 remain open for their full acceptance requirements even after this bounded code stage is delivered.
