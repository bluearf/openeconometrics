# Declared passive derivation after multiple imputation

`mi_passive(source, recipes, max_work=100_000_000)` accepts a complete saved `MIResult` and returns `MIPassiveResult`. Each named recipe declares exactly one operation:

- `affine`: `inputs`, matching real `coefficients` and a real `intercept`.
- `product`: two or more explicit `inputs`.
- `power`: exactly one input and an integer `exponent` from 1 to 8.

Recipes are in dependency order. Inputs can name source columns or earlier derived columns. Forward references, cycles, collisions, arbitrary expressions and executable code are refused. At most 16 total columns, 10,000 rows and 100 imputations are retained. The function plans numeric, serialized-result and recipe work before buffers and enforces a declared work limit. Numerical operations use CPU float64 Torch; nonfinite results and nonzero multiplicative terms/results that underflow to zero are refused without clipping.

Every source completion and physical row identity is retained. The original value of a derived column is present only when every declared input was originally observed; nulls propagate even through a zero affine coefficient. Completed derived cells are computed independently in each imputation. `dataset(1)` returns a new expanded table with the source index. Result JSON retains the source state/hash, expanded state, full formula dependency graph and operation/resource declarations. Restoration recomputes all derived cells; a recomputed checksum cannot legitimize a formula/state mismatch. Checksums detect corruption and are not authentication.

The saved `source_result_class` preserves the `MIDeltaResult` or `MIDiscreteResult` type and its extra scientific validation, including inside passive JSON restoration. Workspace planning includes source metadata and serialized copies. Dataset/table access, validated copying and restoration also respect the current configured budget.

This is deterministic **post-MI derivation**. It produces no new stochastic draws, does not feed derived variables into FCS, does not estimate their sample uncertainty and does not establish compatibility between the imputation and analysis models. Such analysis still requires an appropriate fit within every imputation followed by valid pooling.

The [MICE passive-imputation reference](https://amices.org/mice/reference/mice.impute.passive.html) describes consistency of deterministic relations inside an imputation workflow. This narrower helper preserves the relation after completion and does not reproduce MICE's within-sweep feedback. The [author's discussion of transformations](https://stefvanbuuren.name/fimd/sec-knowledge.html) explains why deterministic consistency alone is not substantive-model compatibility.
