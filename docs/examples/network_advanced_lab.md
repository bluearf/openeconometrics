# Advanced production-network laboratory

Open [`network_advanced_lab.py`](network_advanced_lab.py) in the desktop code panel and run the entire file. This revision requires the current source runtime, including the count-model restart diagnostics; the older packaged 0.3.42 runtime does not include these diagnostics. It does not install packages or export files. `network_lab_result` retains all tables, chart specifications, timings and validation proof for subsequent Python commands.

The default design contains **2,400 synthetic firms, 12 sectors, 8 countries, 6 latent ecosystems, 8 monthly directed count networks and 96 skills**. Actual edge counts are measured during execution. Integer interactions follow seeded heterogeneous Poisson draws on sparse candidates; a matched shock scenario thins exactly the same draws in May–June, followed by partial recovery. The observed monthly graphs retain every firm, including isolates. A separate bipartite firm/skill layer supports a projection onto the 96 skills.

| Analysis | Computation and check |
| --- | --- |
| Structural roles | Weighted PageRank, directed Leiden modularity, adjusted Rand index against synthetic planted groups, binary core numbers and brokerage estimated from 32 uniformly sampled sources. Brokerage uses undirected hop distances; interaction strengths are not treated as path costs. |
| Resilience | The same removal fractions for a fixed PageRank ranking and one seeded random ranking. Largest components are divided by the original firm count. The curve is descriptive, without sampling intervals. |
| Time | Directed edge turnover, persistence and earliest arrival with at most one directed hop per snapshot. Equirectangular coordinates are calculated once and attached to every frame, so browser layout workers do not run for the timeline. Positions remain fixed while shock exposure changes. |
| Count models | Fixed K=6 Poisson and degree-corrected block models, two independent starts, finite local optimization, with convergence, stopping reason and selected seed reported. March fits predict the same independently selected directed non-loop pairs in April. Both observations and predictions represent one month. The intersection of identified pairs supplies the same evaluation subset for both models and baselines. Positive and zero dyads have separate MAEs; event totals, count bias, Poisson negative log likelihood and presence Brier scores accompany aggregate MAE. Baselines predict zero or the full March off-diagonal count rate. Neither baseline uses April outcomes. |
| Dyadic regression | An equal-size random sample within each sector selects 120 firms independently of outcomes. April log-counts are regressed on March log-counts, fixed geographic affinity and same-sector indicators. Approximate coefficient-specific Freedman–Lane MRQAP uses 199 permutations, giving p-value resolution 0.005. Inference is conditional on the unweighted selected subgraph. Residual node exchangeability is assumed, not verified by the generator: p-values are exploratory and have no multiplicity adjustment or causal interpretation. Sample identities and the design condition number are retained in the proof. |
| Future links | January–April weak binary topology supplies Jaccard and resource-allocation scores. 6,000 uniform unique training nonedges are frozen before May–August outcomes are consulted. Prevalence, precision at 100, score-threshold-group average precision and tie-aware ROC AUC are reported. Undefined metrics remain missing. Scores do not predict direction, counts or calibrated probabilities. |
| Capacity shock | May control and shock use identical artificial terminals and terminal capacities. Integer interaction counts serve as assumed capacity proxies. Both maximum-flow results must pass capacity/conservation/min-cut certificates, and shock flow cannot exceed control flow. This is not an estimate of economic output or a causal loss. |
| Two-mode network | A sparse firm/skill incidence graph projects onto the skill side, avoiding the potentially dense firm-by-firm projection. Native Leiden communities color its chart. |

The panel shows **12 tables and 6 interactive charts**, ordered by execution. Charts include the full aggregate production graph, the eight-frame geographic timeline, paired monthly counts, resilience comparisons, block intensities and the skill projection. No production nodes or edges are silently clipped: explicit display limits and assertions check the full graph and every frame. Browser layout computation has separate time/work limits; a stopped layout is not a convergence certificate.

`network_lab_result["proof"]["block_holdout"]` preserves every restart's seed,
likelihood and membership-change trace as well as the shared evaluation mask
and the training-only baseline rate. A positive held-out count assigned zero
mean has infinite Poisson negative log likelihood: the saved score stays missing
and `impossible_positive_pairs` records the failure. Empty positive/zero subsets
also retain missing MAEs. Predicted means are never clipped to improve scores,
and nonconverged fits remain explicitly uncertified.

The default local-optimization cap is now 100 sweeps per restart, matching the
native fitter default. A completed no-move sweep remains the convergence
criterion; the cap and numerical tolerance are never relaxed based on held-out
scores. Small explicit overrides such as the four-sweep example below can still
return a correctly labelled nonconverged fit.

The source-runtime validation record (internal evidence excluded from this public snapshot)
captures the complete 2,400-node run and persisted readback of all 18 outputs.
The selected Poisson and degree-corrected fits converge in 11 and 7 sweeps,
respectively. Aggregate held-out MAE still favors the zero baseline; the record
retains the positive-event failures and calibration metrics. This validation
does not certify the installed desktop package or a global optimum.

All analytical graphs remain resident and sparse. Calculations run on CPU with Torch-backed float64 network kernels. A 512 MiB graph-operation budget covers owned buffers and resident snapshots under the SDK contract; it is **not** a cap on Python process RSS, charts, the editor or the operating system. Browser WebGL acceleration is separate from analytical GPU support. The example does not claim out-of-core processing, all-model GPU support, causal identification or real-world predictive accuracy.

For a smaller run, set the following in the panel before running the entire file:

```python
NETWORK_LAB_CONFIG = {
    "firms": 144, "mrqap_nodes": 36, "permutations": 19,
    "betweenness_samples": 8, "link_candidates": 200,
    "block_starts": 1, "block_iterations": 4, "skills": 24,
}
```

Or call `run_lab(config, display_callback=display)` after importing the module. To return to defaults, delete `NETWORK_LAB_CONFIG` or pass an empty dictionary to `run_lab`. The full-file entry does not parse inherited desktop command-line arguments or raise `SystemExit`.

Method context: [Dekker, Krackhardt and Snijders on MRQAP](https://www.stats.ox.ac.uk/~snijders/DekkerKrackhardtSnijders.pdf), [Karrer and Newman on degree-corrected block models](https://arxiv.org/abs/1008.3926), and [Liben-Nowell and Kleinberg on link prediction](https://people.csail.mit.edu/dln/papers/link/paper.pdf). The executable uses the library's documented implementation and inference scope; these references do not imply implementation equivalence to every method in the papers.
