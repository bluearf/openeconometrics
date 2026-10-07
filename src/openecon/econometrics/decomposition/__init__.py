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
EXPORTS = {"oaxaca_details": "openecon.econometrics.decomposition.oaxaca:oaxaca_details"}
