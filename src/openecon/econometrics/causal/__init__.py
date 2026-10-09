"""Torch-free contracts for orthogonal and honest causal learning."""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_COMMON = (
    Option("learner", "str", "linear", choices=("linear", "lasso", "ridge", "forest")),
    Option("folds", "int", 5, minimum=2, maximum=20),
    Option("seed", "int", 1729, minimum=0),
    Option("penalty", "float", 0.01, minimum=0),
    Option("max_iterations", "int", 2000, minimum=1, maximum=100000),
    Option("tolerance", "float", 1e-8, minimum=1e-12, maximum=0.001),
    Option("trees", "int", 64, minimum=1, maximum=2000),
    Option("max_depth", "int", 5, minimum=0, maximum=12),
    Option("min_leaf", "int", 8, minimum=2),
    Option("mtry", "int", 0, minimum=0, doc="0 selects ceil(sqrt(p)); otherwise 1..p."),
    Option("split_candidates", "int", 16, minimum=1, maximum=256),
    Option("leaf_prior", "float", 0.5, minimum=0, doc="Declared symmetric Bernoulli pseudocount."),
    Option("overlap", "float", 0.01, minimum=1e-8, maximum=0.49),
    Option("max_work", "int", 200000000, minimum=1),
)
_TREATMENT = Role("treatment", required=True, doc="Binary numeric treatment coded exactly 0/1.")


def _info(name, title, roles=(), options=(), description=""):
    return EstimatorInfo(
        name=name,
        title=title,
        family="causal",
        entry=f"openecon.econometrics.causal.estimators:fit_{name}",
        function=name,
        covariances=("HC0", "HC1", "cluster"),
        default_covariance="HC0",
        categorical=False,
        intercept="always",
        predictors="optional",
        weights=("iweight",),
        roles=(_TREATMENT, *roles),
        options=(*_COMMON, *options),
        inference="z",
        description=description,
    )


ESTIMATORS = (
    _info(
        "dmlirm",
        "Cross-fitted IRM/AIPW treatment effect",
        options=(Option("estimand", "str", "ate", choices=("ate", "atet")),),
        description="Binary IRM ATE/ATET with orthogonal scores, train-only nuisance learners and row/whole-cluster folds.",
    ),
    _info(
        "dmliivm",
        "Cross-fitted interactive-IV LATE",
        roles=(Role("instrument", required=True, doc="Binary instrument coded exactly 0/1."),),
        options=(Option("min_first_stage", "float", 0.02, minimum=0, maximum=1),),
        description="Binary interactive-IV orthogonal reduced-form/first-stage ratio, LATE assumptions and relevance diagnostics.",
    ),
    _info(
        "dmlcate",
        "CATE best linear projection",
        roles=(
            Role(
                "basis",
                many=True,
                required=True,
                doc="Predeclared numeric CATE basis; intercept is automatic.",
            ),
        ),
        description="Cross-fitted AIPW pseudo-outcome projection on a fixed basis; complete joint influence covariance.",
    ),
    _info(
        "dmlgate",
        "GATE and honest sorted GATES",
        roles=(
            Role(
                "group",
                kind="label",
                doc="Predeclared group; omit for independently learned sorted GATES.",
            ),
        ),
        options=(Option("groups", "int", 4, minimum=2, maximum=20),),
        description="Declared-group GATE or three-way honest DR-forest ranking GATES with joint covariance and heterogeneity test.",
    ),
    _info(
        "causalforest",
        "Honest doubly robust CATE forest",
        options=(
            Option(
                "query",
                "json",
                None,
                doc="Optional m by p numeric queries; default is admitted input X.",
            ),
        ),
        description="Three disjoint nuisance/split/leaf samples; DR forest local averages, conditional score variance and saved CATE prediction.",
    ),
    _info(
        "policyvalue",
        "Doubly robust policy value",
        roles=(Role("policy", doc="Outcome-independent predeclared policy probability in [0,1]."),),
        options=(
            Option("cost", "float", 0.0, doc="Declared per-treated-unit cost in outcome units."),
            Option(
                "learned",
                "bool",
                False,
                doc="Learn positive net CATE on nuisance/split samples; evaluate only independent leaf sample.",
            ),
        ),
        description="Predeclared or independent learned policy value with joint treat-all/treat-none/paired-gain influence covariance.",
    ),
)

EXPORTS = {
    "causal_predict": "openecon.econometrics.causal.estimators:causal_predict",
}
