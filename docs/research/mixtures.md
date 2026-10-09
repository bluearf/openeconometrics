# MARKET-169: finite mixtures and latent classes scope decision

Status: planning deliverable; implementation deferred. Preserve the earlier
`Hayır` decisions for general finite mixtures and latent-class analysis.
Existing ZIP/ZINB and Markov switching retain their declared zero-inflation and
ordered-state semantics; neither is relabelled as this general mixture family.
Activation requires a defined latent-class use case, a regularity policy and
capacity to validate full likelihood/state/inference.

Proposed `oe.finite_mixture` and `oe.latent_class` names are unregistered drafts.
Use a versioned joint model/state contract rather than separate fits whose
likelihoods or standard errors are pasted together. General latent-class item
models also remain distinct from the deferred IRT workstream MARKET-170.

## Deferred phase matrix

| Phase | Domain and dependency | Acceptance gate |
| --- | --- | --- |
| MIX-1 | Small unweighted complete-data Gaussian regression mixtures with fixed K, constant mixing weights, bounded multi-start EM | Tiny direct joint-density/responsibility oracle; monotonic accepted iterations; start failure denominators; permutation-invariant density and predictions; collapse/singularity diagnostics |
| MIX-2 | Regular joint information/covariance and restored conditional/unconditional predictions; depends on MIX-1 | Full transformed parameter covariance against numerical derivatives and independent fit; boundary/collapse/weak-separation inference refusal; all labels/state preserved across restart |
| MIX-3 | Poisson and binary logistic regression components plus multinomial class-membership covariates; depends on MIX-2 | Component and mixing likelihood normalization, joint cross-block information, count/binary domain/identification checks, saved new-X/new-membership predictions |
| MIX-4 | Polytomous locally independent latent-class item model; optional membership covariates; depends on stable likelihood/state | Exact tiny category-pattern likelihood, item/class probabilities, missing-item marginalization only under declared assumptions, independent poLCA reference and row alignment |
| MIX-5 | K comparison, entropy and separately justified bootstrap class-comparison inference; depends on all relevant regularity gates | AIC/BIC sample/parameter counts, label-independent entropy, full failure denominators; no ordinary chi-square LR calibration across unidentified added classes |

## Likelihood, sample and regularity contract

For each retained row, evaluate `log Σ_k π_k(z_i) f_k(y_i | x_i, θ_k)`
with log-sum-exp and normalized mixing probabilities. Distinguish *prior* class
membership from *posterior* membership conditioned on observed y/items. New-X
unconditional prediction cannot use an unknown future y; component predictions
are conditional targets. This likelihood and label-invariant joint predictive
target follow the finite-mixture definitions in the Stan guide; the proposal
is a native frequentist optimizer, not a Stan production integration.
[Mixture and label-switching definitions](https://mc-stan.org/docs/stan-users-guide/finite-mixtures.html).

MIX-1 defaults to missing="raise", complete real numeric predictors/outcomes,
no weights and fixed explicit K. Any complete-case option retains original
positions and counts. Poisson requires nonnegative integer counts; logistic
requires declared 0/1 coding; LCA saves full item category order. Mixed data,
survey weights, random effects, arbitrary missing patterns and ordinal-response
constraints are refused until separately supported.

Specify seed derivation, every initialization, max starts/iterations,
likelihood tolerance and converged/failed start counts. Use the best admissible
converged objective; retain all attempted-start diagnostics. Stop on component
collapse, near-zero weight or deficient information. A Gaussian mixture may
have unbounded likelihood as a variance collapses: an explicit variance bound
defines a constrained estimator, not unconstrained ML. Record that bound and
whether it is active; ordinary interior SEs are unavailable at a binding bound.
Canonical display ordering only permutes labels after fitting and must also
permute state and full covariance; it cannot force two different local optima
to appear equal.

Joint information includes component, mixing and membership-covariate blocks;
per-component regression SEs omit mixture uncertainty and are not accepted.
Parameter transformations and covariance Jacobians are explicit. Regularity
diagnostics precede Wald inference. Standard class-count LR asymptotics do not
follow merely from a likelihood improvement. Entropy, posterior assignments,
AIC/BIC and validated class-comparison tests must retain their different roles.

For LCA, begin with local independence of categorical items conditional on
class, explicit finite category maps and fixed K. Conditional missing-item
marginalization needs an ignorable-missingness contract and saved item masks;
it is not automatic MAR validation. Direct item dependence is a later separate
model. Use the authors' published poLCA model/algorithm as an independent
development reference. IBM STATS LATENT CLASS remains an extension reference,
not evidence that an IBM core-native routine was executed.
[poLCA original paper](https://www.jstatsoft.org/article/view/v042i10).

## Persistence, resources and validation plan

Save K, component type/order, all joint parameters and their full covariance,
mixing maps, class/item/category maps, covariate transformations, bounds,
start diagnostics, entropy conventions and exact prediction targets. Posterior
membership arrays may be large: use bounded checksummed artifacts instead of
silently truncating persisted state. Restored density, component probabilities
and unconditional predictions must equal the saved fit without rerunning EM.

Budget resident data, n×K responsibilities, component design/optimizer state,
joint q² information and starts/iterations before allocation. CPU float64 is
the initial route; no Dataset/GPU capability is inherited from existing models.
Cancellation and exhausted budgets yield an incomplete explicit status and
preserve prior accepted results; failed starts cannot vanish from denominators.

Future acceptance uses fixed small K=1/2 synthetic matrices, direct exhaustive
densities, independent numerical gradients/Hessians, then matching reference
fits with frozen seed/start/constraint protocols. Test extreme logits/log
densities, empty items/classes, duplicate predictors, singleton categories,
separation, overlapping components, collapsing variances and label permutations.
Simulation/calibration experiments freeze counts and tolerances before results
and cannot claim global optimization from successful multi-start examples.
Licensed vendor, source, frozen and installed native evidence remain separate
unexecuted gates.

Open implementation: [GitHub #56](https://github.com/bluearf/openecon/issues/56).
