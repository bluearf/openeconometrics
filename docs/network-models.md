# Native network statistical models and learning

These are CPU float64, code-first APIs. They use Torch and owned algorithms;
SciPy, NetworkX and statsmodels are test oracles only. Run
[`examples/network_models.py`](examples/network_models.py) in the code panel.
Integer `1` and string `"1"` remain different actors/nodes, including saved models.

| API | First supported contract | Estimation / inference |
| --- | --- | --- |
| `oe.ergm` | Fixed loopless undirected binary labeled graphs; edges, two-stars, triangles | Explicit exact enumeration MLE or conditional MPLE. Exact Fisher covariance requires score convergence, nonsingular information and an interior support certificate. Dependent MPLE has no reported SE. |
| `oe.simulate_ergm` | Same terms and sample space | Private seeded single-dyad Gibbs chain; explicit burn-in, thinning, acceptance/change rate, extreme-graph fraction, lag-one ESS diagnostic. These diagnostics do not prove mixing. |
| `oe.saom` | Fully observed directed fixed actors; strictly increasing explicit wave times; constant actor opportunity rate; outdegree, reciprocity, transitive triplets | Exact finite-state CTMC transition likelihood. Actor chooses no change or one outgoing toggle by objective softmax. Rate and effects fit jointly; score, Hessian, convergence and boundary diagnostics. |
| `oe.simulate_saom` | Same actor choices and time contract | Seeded Gillespie actor opportunities with residual waiting times carried across waves. Ordered binary snapshots plus opportunity/change counts. |
| `oe.gaussian_block_model` | Explicit real observed dyads; zero/negative values allowed; fixed nonempty K; directed or undirected; no self loops | Constrained Gaussian block profile likelihood with a stated variance floor and multiple coordinate-ascent starts. Local label convergence, no global-optimum or parameter-SE claim. |
| `oe.mixed_membership_block_model` | Explicit Bernoulli dyads; fixed K, per-node simplex, symmetric B when undirected | Conditional marginal Bernoulli likelihood, private seeded mini-batch Adam and train-objective-only start selection. Fixed simplex estimates, not a Dirichlet posterior. Objective stabilization does not certify a score optimum. |
| `oe.network_embedding` | Explicit binary observed dyads, including explicitly supplied train zeros | Seeded mini-batch logistic dot-product factors. Directed source/target factors or tied undirected factors. Explicit zeros are the negative samples; absent dyads are missing. |
| `oe.network_gnn` | Numeric node features, supervised labels and explicit train/validation/test edge splits | One mean-GraphSAGE layer, ReLU and softmax; uniform neighbor sampling without replacement. Fixed epochs; inference uses full outgoing-neighbor means. |

## Identity, observation masks and leakage

The block/embedding APIs take an observations DataFrame with `source`, `target`,
`value`, and optional `split` (`train`, `validation`, `test`; default train).
Every eligible dyad occurs at most once (including reversed undirected pairs).
The supplied Network provides only the typed registry and directedness. Its
existing topology is never an implicit response, initialization or training mask.
Only train responses fit parameters. Validation/test responses are evaluated
against a baseline fitted on train values. No unobserved dyads become zeros.
Train-unobserved Gaussian block parameters are explicitly unidentified and their
prediction fails. Mixed memberships have permutation/factorization ambiguity;
unobserved train nodes retain their initialization and are reported.

GNN labels require `node`, `label`, `split`; features require `node` plus numeric
columns; edges require `source`, `target`, `split`. Training edges must have two
training endpoints. Normalization and class discovery use train nodes only.
Validation inference sees train+validation edges; test sees train+test edges.
Held-out labels/features/edges never influence gradients or fitted state. Split
membership is supplied explicitly; these APIs do not infer chronological meaning
from integer identifiers or construct a temporal split for the caller.

`EmbeddingResult.save_json/load_json` preserve typed node factors and predict
supplied typed dyads without fitting. `GNNResult.save_json/load_json` preserve
weights, feature names, classes and train normalization; `predict_graph` is
inductive on a supplied new graph/features. Input byte/geometry/finite-value
checks precede parsing/large allocations. GNN inference intentionally uses all
edges of that explicitly supplied inference graph.

## Supported geometry and devices

All routines admit owned buffers through the Network memory budget, and charge
explicit `max_work`, iterations and `timeout`; cancellation/Stop raises instead
of returning a partial successful fit. The resource plan covers modeled live
buffers, not total process RSS or third-party DataFrame memory. Resident graphs
remain resident. ERGM/SAOM deliberately enumerate small dense state spaces:
ERGM defaults to at most 65,536 states (six actors), SAOM to 256 (three actors).
Larger state spaces or unsupported terms/constraints fail explicitly; no MCMC
likelihood, moment estimator, missing-dyad imputation, actor-set reconciliation
or alternate scientific family is silently substituted.

Block and learning inputs default to at most 100,000 observed dyads/edges;
Gaussian coordinate sweeps have O(starts × sweeps × N × K × observations) work.
Mixed/embedding batches have observation-sized index state plus node parameters;
GNN batches hold sampled features plus parameters and bounded explicit neighbor
lists. Prediction and saved-model loading have independent output/byte limits.

| Device | Training / estimation | dtype | Transfer / VRAM |
| --- | --- | --- | --- |
| CPU | Implemented and tested | float64 | No accelerator transfer, VRAM zero |
| CUDA | Unsupported in this first model API | — | Explicit error for learning `device="cuda"`; no fallback |
| MPS / automatic device | Unsupported | — | Explicit error; no fallback |

No CUDA hardware proof or broad Stata/RSiena/Gephi parity is implied. SAOM accepts
one panel or independent panel replicates with identical actor/time contracts;
parameter recovery needs informative transitions, and boundary/flat information
has unavailable uncertainty. ERGM exact support checks enumerate supporting
planes in up to three integer statistics; all likelihood arithmetic is native
float64. Chain simulation is separate from deterministic likelihood fitting.

## Primary methodological references

* [Hunter et al., ERGM (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2743438/).
* [Snijders, SAOM overview](https://www.stats.ox.ac.uk/~snijders/SAOM_ARStat1.pdf), actor opportunity/choice equations 4–5.
* [Airoldi et al., Mixed Membership Stochastic Blockmodels (2008)](https://www.jmlr.org/papers/volume9/airoldi08a/airoldi08a.pdf). The implemented conditional simplex likelihood is explicitly narrower than Bayesian MMSB.
* [Hamilton et al., GraphSAGE (2017)](https://arxiv.org/abs/1706.02216). The implemented one-layer mean aggregator is explicitly narrower than the full framework.
