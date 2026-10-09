# Conditional sampling and hierarchical selection

`loglinear_multinomial` fits a single fixed-grand-total multinomial table.
`loglinear_product_multinomial` conditions on the observed totals of explicitly
declared dimensions, fitting one joint conditional model over their strata.
These procedures fit the conditional likelihood directly with native CPU
float64 Torch; they do not reuse independent-Poisson covariance.

```python
import openecon as oe

fit = oe.loglinear_product_multinomial(
    cells, ["group", "response"], "count",
    levels={"group": ["A", "B"], "response": ["no", "yes"]},
    conditioning=["group"],
    design=response_yes_indicator, terms=["yes_log_odds"],
)
state = oe.summary_state(fit)
restored = oe.restore_summary(state)
comparison = oe.loglinear_sampling_compare(restricted_fit, restored)
```

Every declared cell is required, including zero-count cells and explicit
boolean structural zeros. Input order and design rows align before conversion
to the canonical Cartesian grid. Finite integer counts, typed categories,
known log offsets, positive stratum totals and active support are checked.
Sampling-zero cells can have positive fitted probability. Boundary MLEs,
zero-total strata and ill-conditioned information are refused, without
pseudocounts or covariance repairs. Fixed margins beyond the declared totals,
weights, cluster/survey corrections and Dataset/streaming inputs are excluded.

For stratum g, p_i = exp(offset_i + x_i beta) / sum_g exp(offset + X beta),
mu_i = N_g p_i. The full likelihood includes log(N_g!) minus sum log(y_i!).
The score is X'(y-mu); observed information is the sum of
N_g X_g'[diag(p_g)-p_g p_g']X_g. Cell-design columns constant within strata,
including an overall intercept, are unidentified and must be omitted by the
caller. Full rank is checked after removing stratum constants; structural
support is included in that check. Newton updates use an Armijo line search
and certify both a scaled score and a scale-invariant centered predictor step.

Outputs retain the full conditional information/covariance, parameter
estimate/SE/z/p/Normal Wald CI, probabilities, fitted count delta SE,
stratum totals, Pearson/deviance statistics and convergence trace. The count
mean derivative is mu_i times [x_i minus its probability-weighted stratum
mean]. Structural cells have fixed zero probability and no estimated SE.
Residual df is active cells minus strata minus identified parameters. Wald
and chi-square approximations are asymptotic; small counts can be inaccurate.
No finite-sample calibration or simultaneous intervals are claimed.

Saved nested LR checks the exact cells/counts/support/offsets/conditioning and
the whole fitted numerical state. It checks design-space inclusion modulo
stratum constants, rather than parameter names or counts alone. Recomputed
probabilities, means, likelihood, information, covariance and stationarity
must agree with the saved state even when an altered payload is rehashed.
The checksum establishes integrity, not cryptographic authenticity.

Cell input is bounded at 4096 rows, 2–6 dimensions, 2–16 levels per dimension,
and 1–128 identified parameters. Fit iterations <=2000; default 200. Planned
work defaults to 300 million structural units; max_bytes defaults to 128 MiB
and is intersected with the global workspace budget before allocations.
Resident selection, full design conversion, information and iteration buffers
are named; the estimate excludes caller inputs, Python result objects, BLAS
private memory and allocator overhead, and is not an RSS limit.

`loglinear_select` separately implements a deterministic greedy backward
AIC/BIC path for independent-Poisson hierarchical IPF models. It starts from
explicit generating margins, retains every main effect and removes only a
current maximal interaction. All eligible deletion candidates are fitted on
the same counts/support/offsets, recorded and compared. A failed candidate
refuses the search. There is no invisible skip or incomplete candidate set.
The best strict improvement wins, with canonical term order breaking ties.
The search stops at main effects or no criterion improvement.

BIC uses log(total Poisson count) times rank, an explicit person-count sample
convention; AIC uses twice rank. This convention is not an arbitrary-cell GLM
BIC rule. Selected-model SE/CI are retained from the fitted model and remain
unadjusted for selection. The greedy path is not a global exhaustive search,
and the candidate table does not give calibrated post-selection p-values.
The result contains every complete candidate summary, the entire decision
trace and the selected complete summary. Restore the selected summary with
`oe.restore_summary(result.attrs["state"]["selected_summary"])`.

Selection admits only 2–4 dimensions and an initial <=128-column hierarchy;
the worst-case complete candidate count is charged before the first fit.
Default max_models=64 (maximum 256), default work=300 million and bytes=128
MiB. Complete retained candidate states and the largest fit buffers are named
in the resource record. Cases beyond these bounds fail without truncation.

Independent tests use SciPy conditional-likelihood optimization and a separate
analytic Hessian, closed-form saturated multinomial and common-binomial odds,
structural support, offsets and permutation tests. Selection is compared with
an independently constructed dummy-design likelihood and full candidate path.
Complete tables and concatenated LaTeX roundtrip, state corruption, failed
convergence and memory/work/default-device behavior are checked separately
from frozen/native execution. These checks do not establish licensed vendor
parity or a public release.

Sources for the sampling distinction and workflow:

- [IBM GENLOG overview](https://www.ibm.com/docs/en/spss-statistics/30.0.0?topic=genlog-overview-command): separate Poisson/product-multinomial models and Newton design fitting.
- [Penn State STAT 504 sampling schemes](https://online.stat.psu.edu/stat504/Lesson01): unrestricted Poisson versus fixed-total multinomial sampling.
- [IBM HILOGLINEAR model selection](https://www.ibm.com/docs/en/spss-statistics/30.0.0?topic=hiloglinear-method-subcommand-command): backward model building. Our declared criterion is AIC/BIC; IBM's significance cutoff behavior is not claimed.
