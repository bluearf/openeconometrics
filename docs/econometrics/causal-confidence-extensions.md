# Causal confidence sets, multiarm effects and individual-effect bounds

These eight procedures extend the existing causal-design family with bounded, explicitly declared designs. Their method contracts are documented separately:

- [Finite-grid exact Fisher confidence sets](causal-confidence-sets.md): MARKET-723–725.
- [Bernoulli and multiarm Neyman inference](causal-design-multiarm.md): MARKET-726–728.
- [Individual-effect distribution bounds](causal-effect-distribution-bounds.md): MARKET-729–730.

All return editable `TableSet` outputs with complete original sample identity, design and numerical state. `causal_design_save` and `causal_design_load` retain every table and typed value. The two discrete effect-bound profiles additionally check all attained coupling margins, flow conservation, reachable residual min-cut and quantile witnesses without rerunning optimization. Integrity hashes alone are not substituted for these semantic certificates.

The [runnable synthetic example](../examples/causal_confidence_eight.py) uses all eight methods and compares full save/load results exactly. The [predeclared stages](roadmaps/causal-confidence-eight.md) preserve each issue's scope and required proof.

Finite-grid sharp-null acceptance has exact coverage only if the true constant effect belongs to the supplied grid. Neyman normal inference is conservative/asymptotic under its stated design sequence conditions. Individual-effect bounds condition on the empirical marginal input laws and provide pointwise identification endpoints, without population sampling intervals or one jointly attained entire curve. These distinctions are part of the saved outputs; no vendor parity flag is enabled.
