"""Mediation and Oaxaca-Blinder; no tensor imports in the catalogue."""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

ESTIMATORS = (
    EstimatorInfo(
        name="mediation",
        title="Mediation and conditional process",
        family="decomposition",
        function="mediation",
        entry="openecon.econometrics.decomposition.mediation:fit_mediation",
        covariances=("HC1", "cluster"),
        default_covariance="HC1",
        categorical=False,
        roles=(
            Role("mediators", many=True, required=True),
            Role("controls", many=True),
            Role("moderator"),
        ),
        weights=("aweight",),
        options=(
            Option("serial", "bool", False),
            Option("at", "list[float]", None),
            Option("moderate", "list[str]", ["a", "b", "direct"]),
            Option("treatment0", "float", 0.0),
            Option("treatment1", "float", 1.0),
            Option("interpretation", "str", "associational", choices=("associational", "causal")),
            Option("assumptions", "list[str]", []),
            Option("outcome_model", "str", "linear", choices=("linear", "logit", "probit")),
            Option("integration_draws", "int", 128, minimum=20, maximum=2000),
            Option("seed", "int", 0, minimum=0),
        ),
        description="Linear parallel/serial/moderated paths with joint-score delta inference; logit/probit outcomes use explicit Gaussian-mediator natural effects.",
    ),
    EstimatorInfo(
        name="oaxaca",
        title="Oaxaca-Blinder decomposition",
        family="decomposition",
        function="oaxaca",
        entry="openecon.econometrics.decomposition.oaxaca:fit_oaxaca",
        covariances=("bootstrap",),
        default_covariance="bootstrap",
        weights=("aweight",),
        roles=(Role("group", required=True, kind="label"),),
        categorical=True,
        intercept="always",
        options=(
            Option("groups", "json", None),
            Option("fold", "int", 2, choices=(2, 3)),
            Option(
                "reference",
                "str",
                "pooled",
                choices=("pooled", "neumark", "a", "b", "reimers", "cotton"),
            ),
            Option("reps", "int", 200, minimum=50, maximum=5000),
            Option("seed", "int", 0, minimum=0),
        ),
        description="Two/three-fold weighted mean-gap decomposition with sum-to-zero categorical normalization and stratified-pairs bootstrap.",
    ),
)
ESTIMATORS = ESTIMATORS + (
    EstimatorInfo(
        name="mediation_interaction", title="Mediation with exposure interaction", family="decomposition",
        entry="openecon.econometrics.decomposition.interaction:fit_mediation_interaction", function="mediation_interaction",
        covariances=("HC1", "cluster"), default_covariance="HC1", categorical=False, cluster_dimensions=1,
        roles=(Role("mediator", required=True), Role("controls", many=True)),
        options=(Option("treatment0", "float", 0.0), Option("treatment1", "float", 1.0),
                 Option("interpretation", "str", "associational", choices=("associational", "causal")),
                 Option("assumptions", "list[str]", [])),
        description="Linear continuous mediator/outcome with exposure-mediator interaction, five natural-effect contrasts and full cross-equation delta covariance; causal labeling requires declared assumptions.",
    ),
)
EXPORTS = {
    "oaxaca_details": "openecon.econometrics.decomposition.oaxaca:oaxaca_details",
    "mediation_binary": "openecon.econometrics.decomposition.binary:mediation_binary",
    "mediation_binary_restore": "openecon.econometrics.decomposition.binary:mediation_binary_restore",
}
ESTIMATORS = ESTIMATORS + (
    EstimatorInfo(
        name="fairlie", title="Fairlie binary decomposition", family="decomposition",
        entry="openecon.econometrics.decomposition.fairlie:fit_fairlie", function="fairlie",
        covariances=("bootstrap",), default_covariance="bootstrap", categorical=False,
        roles=(Role("group", required=True, kind="label"),), intercept="always",
        options=(Option("groups", "json", required=True),
                 Option("link", "str", "logit", choices=("logit", "probit")),
                 Option("reference", "str", "pooled", choices=("pooled", "a", "b")),
                 Option("reps", "int", 99, minimum=49, maximum=2000),
                 Option("matching_reps", "int", 20, minimum=1, maximum=1000),
                 Option("seed", "int", 0, minimum=0),
                 Option("max_work", "int", 100_000_000, minimum=1)),
        description="Ordered binary mean-gap decomposition by native logit/probit, probability-rank matching and randomized covariate switching with full stratified-bootstrap covariance; matching MC remainder retained. Numeric unweighted resident domain.",
    ),
)
