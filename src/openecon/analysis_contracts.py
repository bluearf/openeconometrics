"""Lightweight analysis errors and capability metadata, without loading Torch."""
from __future__ import annotations

from typing import Any


class AnalysisError(ValueError):
    """An actionable failure that clients can render without parsing messages."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


_SUPPORTED_COVARIANCES = {
    "ols": ("nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster", "cluster_hc2", "cluster_hc3", "hac", "bootstrap", "jackknife"),
    "logit": ("nonrobust", "cluster"),
    "probit": ("nonrobust", "cluster"),
}
_MAX_DESIGN_BYTES = 256 * 1024 * 1024


def capabilities() -> dict[str, Any]:
    """Describe implemented statistical capabilities and current limits."""
    from openecon.econometrics.registry import all_estimators, describe
    from openecon.econometrics.streaming_registry import algorithms, conditions, estimators as replay_estimators
    from openecon.resources import workspace_budget_bytes
    from openecon.linear_ols.spec import describe as describe_ols
    from openecon.prediction_capabilities import describe as describe_prediction

    # Estimator families beyond the core three are catalogued by the registry.
    registered = [info for info in all_estimators() if not info.legacy]
    streamed = replay_estimators()
    prediction = describe_prediction([*_SUPPORTED_COVARIANCES, *(info.name for info in registered)])
    return {
        "schema_version": "3",
        "engine": "openecon",
        "precision": "float64",
        "execution": "local Torch CPU; bounded OLS/system factors support CUDA float64 and checked Metal preconditioning",
        "estimators": {
            name: {
                "covariances": list(covariances),
                "binary_outcome": name != "ols",
                "categorical_predictors": True,
                "intercept": True,
                "missing": ["raise", "drop"],
                "inference": "Student t" if name == "ols" else "Normal z",
                **(describe_ols() if name == "ols" else {}),
            }
            for name, covariances in _SUPPORTED_COVARIANCES.items()
        } | {info.name: describe(info) for info in registered},
        "families": ["core", *dict.fromkeys(info.family for info in registered)],
        "eager_only_estimators": [info.name for info in registered if info.name not in streamed],
        "cluster_dimensions": 4,
        "weights": True,
        "fixed_effect_absorption": any(info.role("absorb") for info in registered),
        "out_of_core_estimation": True,
        "streaming": {
            "estimators": ["ols", "logit", "probit", *streamed],
            "covariances": list(_SUPPORTED_COVARIANCES["ols"]),
            "covariances_by_estimator": {name: list(values) for name, values in _SUPPORTED_COVARIANCES.items()}
                | {info.name: list(info.covariances)
                   for info in registered if info.name in streamed},
            "formats": ["csv", "parquet"], "numeric_predictors_only": False,
            "row_limit": None, "max_parameters": 384,
            "algorithm": "native float64 global factors, likelihood/moment replays, disk group/risk-set/time-series kernels",
            "algorithms": {"ols": "centered float64 TSQR with bounded covariance replays",
                           "logit": "replayed exact likelihood Newton steps",
                           "probit": "replayed exact likelihood Newton steps",
                           **algorithms()},
            "conditions": conditions(),
            "devices": {"ols": ["cpu", "cuda", "mps"],
                        "linear_system_factors": ["cpu", "cuda", "mps"],
                        "likelihood": ["cpu"]},
            "metal_precision": "float32 QR preconditioner; original CPU float64 observations refine/certify the factor; unsafe blocks fall back to CPU Householder QR",
            "categorical_encoding": "bounded treatment coding",
            "cluster_aggregation": "OLS up to four dimensions; binary one; native adapters one or two according to their registered condition; fixed-memory SQLite spill",
            "max_prediction_sample": 400,
            "max_prediction_sample_scope": "stored fit/chart preview only; independent full Dataset evaluation has no shared row ceiling",
            "working_memory_budget_bytes": 128 * 1024 * 1024,
            "full_sample_positions_retained": False,
        },
        "workspace": {"model_budget_bytes": workspace_budget_bytes(),
                      "configuration": "OPENECON_WORKSPACE_MB",
                      "scope": "planned live model buffers, not total process RSS"},
        "input": {"automatic_large_csv_parquet": True, "eager_row_threshold": 100_000,
                  "eager_byte_threshold": 32 * 1024 * 1024, "streaming_row_ceiling": None,
                  "large_xlsx_dta": "isolated bounded conversion to owned Parquet; close Dataset to release scratch",
                  "reader_guards": {"parser_extra_bytes": 512*1024*1024,
                      "parser_total_rss_bytes": 1536*1024*1024, "transfer_bytes": 64*1024*1024,
                      "csv_cell_bytes": 1024*1024, "csv_record_bytes": 4*1024*1024,
                      "parquet_footer_bytes": 2*1024*1024, "directory_footer_bytes": 64*1024*1024,
                      "parquet_row_group_bytes": 256*1024*1024,
                      "conversion_disk_bytes": 8*1024*1024*1024, "timeout_seconds": 120,
                      "scope": "physical preflight and disposable parser RSS supervision; separate from parent model buffers"}},
        "dataset_preparation": {
            "operations": ["project", "filter", "map", "join", "reshape_long", "reshape_wide"],
            "execution": "replayed 8192-row local blocks; SQLite disk joins/wide; no cloud or whole-source collection",
            "replay": "complete input/output digests include typed values, indices, dtype/category schema and missing tags; callable code identity checked",
            "defaults": {"stage_buffer_bytes": 64*1024*1024, "sqlite_cache_bytes": 16*1024*1024,
                "stage_disk_bytes": 1024*1024*1024, "max_rows": 10_000_000, "max_work": 100_000_000,
                "max_stages": 16, "max_wide_levels": 1000},
            "allocation_scope": "per-stage owned buffers/cache/disk; arbitrary map/predicate allocations and total process RSS are caller-owned",
            "pandas_differences": [
                "null keys never match by default; nulls='equal' is explicit; bool/string keys remain distinct from numeric keys",
                "join validates m:1 by default and requires agreeing key dtype/category schemas; output index retains both source indices",
                "map requires exact declared dtypes/categories and unchanged original index/row count; no implicit cast",
                "long output is row-major with (original index, variable) identities and lossless equal value-column dtypes",
                "wide requires explicit levels and duplicate/unknown policies; no implicit aggregation or unbounded level discovery",
                "outer join/wide gaps promote integer/bool to nullable dtypes; retained missing tags require unique indices within each block",
                "empty disk inputs need a declared typed map when a lossless pandas schema cannot be inferred"],
        },
        "stata_parity_validated": False,
        "max_chart_predictions": 400,
        "postestimation": prediction,
        "network": {
            "representation": "sparse weighted edges; no dense node-by-node matrix",
            "input": ["DataFrame", "column dictionary", "row records", "Dataset"],
            "directed": True, "weighted": True, "isolates": True,
            "algorithms": ["degree", "strength", "pagerank", "weak_components", "strong_components", "shortest_paths",
                           "louvain", "leiden", "modularity", "betweenness", "closeness", "harmonic", "eigenvector",
                           "triangles", "clustering", "k_core", "density", "transitivity", "assortativity",
                           "hits", "katz", "shortest_path", "distances", "eccentricity", "distance_summary",
                           "bridges", "articulation_points", "k_shortest_paths", "strong_bridges",
                           "strong_articulation_points", "minimum_spanning_forest", "max_flow", "min_cut",
                           "min_cost_flow", "multicommodity_flow",
                           "maximum_matching", "bipartite", "bipartite_projection", "link_prediction",
                           "global_min_cut", "triad_census", "qap_correlation",
                           "qap_regression", "block_model", "poisson_block_model",
                           "degree_corrected_block_model", "network_snapshots", "graphlets", "weighted_assignment", "general_matching"],
            "extended_algorithms": ["hypergraph_incidence", "hyperdegree", "hypergraph_clique_projection", "hypergraph_star_projection",
                "graphlets", "k_shortest_paths", "strong_bridges", "strong_articulation_points", "weighted_assignment", "general_matching"],
            "matching_scope": "maximum cardinality bipartite matching; weak binary topology for directed input",
            "weighted_matching_scope": "undirected signed rewards; exact-rational Hungarian assignment with primal/dual certificate; general matching uses componentwise exact exponential subset DP (default 24 nodes/component), no implicit bipartite projection",
            "graphlet_scope": "exact induced 4/5-node directed or undirected classes and automorphism orbits; weakly connected or all disconnected classes; loops/multiedges rejected; bounded enumeration, no sampling",
            "link_prediction_scope": "five undirected binary scores on explicit candidate pairs only",
            "distance_scope": "exact bounded source/target output; no dense all-pairs adjacency",
            "k_paths_scope": "simple paths, exact stored binary64 costs; explicit k, hop, frontier, output-node and work limits; exponential worst-case prefix search",
            "strong_cuts_scope": "oriented simple arcs/vertices whose deletion increases SCC count; sparse iterative Kosaraju per deletion, quadratic structural worst-case work",
            "hypergraph": {"representation": "typed hyperedge IDs; sparse CPU float64 incidence O(V+H+I)",
                           "direct_algorithms": ["incidence", "incidence_matvec", "degree", "strength", "summary"],
                           "projections": ["explicit clique: sum/count/normalized", "explicit star: disjoint proxies"],
                           "direction": "separate nonempty tail/head membership; overlap allowed",
                           "weights": "finite nonnegative; zero retained in direct incidence",
                           "duplicates": "membership within role and hyperedge ID rejected; distinct IDs may be parallel",
                           "budgets": "memberships, candidate projection edges, owned-memory estimate and structural work"},
            "flow_scope": "max-flow/cut on positive capacities; bounded directed CPU float64 minimum-cost LP with primal/dual, Farkas or improving-ray certificates; single-commodity integral or fractional, multicommodity fractional shared capacities",
            "global_cut_scope": "undirected aggregate capacities; bounded sparse Stoer-Wagner contractions",
            "motif_scope": "exact induced three-node census: 16 directed classes or four undirected classes",
            "qap_scope": "Pearson correlation over all aligned dyads; explicit node-label Monte Carlo permutations",
            "mrqap_scope": "dyad OLS with explicit Freedman-Lane response-residual or DSP focal-residual node permutations; partial correlation and joint partial R-squared; selected none/Bonferroni/Holm/BH adjustment over all reported tests; approximate conditional inference",
            "block_model_scope": "fixed-K hard-label Bernoulli profile likelihood; sparse coordinate ascent, multiple starts and explicit convergence; no global optimum guarantee",
            "poisson_block_model_scope": "fixed-K off-diagonal independent Poisson dyads; exact stored integer count multiplicities, full likelihood including factorials; CPU sparse local ascent",
            "degree_corrected_block_model_scope": "fixed-K Poisson multigraph with full loop sample space; directed in/out node parameters or undirected stub degrees; raw undirected loop mean theta^2*omega/2; CPU sparse local ascent",
            "block_prediction_scope": "selected explicit node pairs: mean count and presence probability; bounded full output, exact typed IDs and explicit identification flags; no parameter-uncertainty intervals",
            "count_weight_authority": "validate stored coalesced binary64 integer counts, not original pre-conversion input; count totals and degree-corrected undirected stubs bounded by 2^53",
            "snapshot_scope": "ordered or general sparse snapshots: transitions, edge persistence, aggregation and strict time-respecting paths; all snapshots remain resident",
            "multilayer_scope": "resident single-aspect typed node-layer states; sparse CPU supra-matvec and weighted supra-PageRank; explicit layer/node projections and lossless NDJSON; no layer-balanced walk, multislice community model or out-of-core/GPU execution",
            "not_implemented": ["ERGM/SAOM",
                                "mixed-membership and arbitrary continuous-weight block models",
                                "motif census beyond five nodes and loop/multiedge motif classes", "embeddings/GNN",
                                "integral multicommodity flow",
                                "unbounded simple-path enumeration or general matching", "direct hypergraph spectral/community/path solvers"],
            "static_file_formats": ["graphml", "gexf", "pajek"],
            "multigraph": {
                "representation": "resident typed-edge-ID snapshot; nonnegative float64 weights including zero; no implicit coalescing",
                "native_algorithms": ["degree", "strength", "summary", "weighted_assignment", "general_matching"],
                "projection_reducers": ["sum", "min", "max", "mean", "count", "binary"],
                "implicit_projection": False,
                "projection_loss_policy": "zero-weight projected pairs and conflicting attributes raise by default; explicit drop policy recorded",
                "interchange": ["graphml", "gexf"],
                "interchange_scope": "required unique wire edge IDs; native metadata preserves typed edge and node IDs, attributes, zero edges and loops; Pajek edge-ID mode explicitly rejected",
                "device": "cpu", "out_of_core": False,
            },
            "betweenness_sampling": "explicit uniform source sampling; exact by default",
            "dynamic": {
                "representation": "resident graph/node/edge interval spells and timed scalar attributes; separate typed edge IDs",
                "interchange": ["gexf-1.2draft", "gexf-1.3"],
                "timeformats": ["double", "integer", "date", "dateTime"],
                "bounds": "missing is unbounded; 1.2draft supports open/closed, 1.3 inclusive only; dateTime exact to microseconds",
                "conversion": "explicit at(time) or window(overlap/cover) returns MultiNetwork; explicit reducer required for simple topology",
                "attribute_policy": "ambiguity/intermittence raises; explicit window drop for attributes and min/max for weights",
                "activity": "intersection of graph, edge and both endpoint node intervals",
                "temporal_paths": "existing ordered snapshots after explicit point projection; windows do not infer continuous paths",
                "budgets": ["max_memory_mb", "max_events", "max_work", "max_file_mb"],
                "default_max_events": 1_000_000, "max_events_per_record": 4096,
                "device": "cpu", "out_of_core": False,
                "unsupported": ["timestamps", "mixed direction", "hierarchy", "visualization extensions", "compact interval strings"],
            },
            "work_budgets": True,
            "batch_import": True,
            "out_of_core_graph": True,
            "out_of_core_analysis": True,
            "disk_native_algorithms": ["summary", "degree", "pagerank", "components_weak"],
            "disk_analysis_scope": "bounded edge scans with admitted O(V) Torch CPU state and complete labelled output; CPU float64 PageRank; unsupported methods require explicit guarded materialization",
            "disk_analysis_budgets": ["max_memory_mb", "batch_rows", "max_work", "max_scan_bytes"],
            "disk_graph_scope": "bounded identity/coalescing indexes, adjacency and transpose, scalar attributes, atomic completion and source-verified reopening",
            "storage_selection": "Dataset import selects resident or temporary disk snapshot after bounded disk construction; store path requests persistence without an engine option",
            "devices": {"pagerank": ["cpu", "cuda"], "disk_pagerank": ["cpu"], "other_algorithms": ["cpu"]},
            "default_memory_mb": 256,
            "memory_scope": "planned graph/import buffers; not total process RSS",
            "visualization": "bounded Canvas graph; same-origin worker force layout",
            "default_visual_nodes": 1000, "default_visual_edges": 5000,
            "hard_visual_nodes": 2000, "hard_visual_edges": 10000,
            "cuda_hardware_validated": False,
        },
        "max_design_matrix_bytes": _MAX_DESIGN_BYTES,
        "limitations": [
            "New spatial, regularized, state-space/ETS/MIDAS, matrix-GARCH and S/MM/panel-MMQR methods use explicitly budgeted resident tables; no Dataset/GPU route is implied by registration.",
            "Logit and probit support nonrobust and one-way cluster covariance only.",
            "Categorical predictors use explicit treatment coding with the first level omitted.",
            "OLS omits collinear terms. Singular or separated binary models are rejected.",
            "Dense estimation materializes a float64 design matrix; registered streaming routes retain bounded batches and small factors.",
            "Cluster inference uses CR1; few clusters can make inference unreliable.",
            "CUDA hardware was not available on this verification host. Metal factors preserve float64 inference through checked CPU refinement.",
            "Adjusted single-contrast inference is available for HC2/HC3; joint tests report conventional Wald inference.",
            "Complex survey designs and Stata ecosystem prefixes are separate features, not OLS weight equivalence.",
            "Registered streaming estimators have no total row ceiling; CSV/Parquet readers use bounded local batches. Supported options are explicit; disk capacity and repeated I/O remain constraints.",
            "Categorical expansion remains bounded by model width and category metadata; cluster scores spill to local disk.",
            "Binary estimation replays the source for likelihood steps and separation checks, so it can require many more passes than OLS.",
            "Common saved predict/margins cover 46 estimator names with explicit fitted-option and response-domain restrictions; 80 Dataset fitting routes do not imply 80 common prediction routes.",
            "Dataset predictions materialize complete indexed output on local disk; the 400 stored chart-preview rows do not limit this evaluation. Global AME/MEM and at grids retain the full saved covariance.",
            "Chunked helpers cover the listed native procedures. Rank correlations, repeated-measures ANOVA, MANOVA and general descriptive quantiles still require separate adapters.",
        ],
    }
