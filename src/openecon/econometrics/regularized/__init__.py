"""Torch-free contracts for prediction and orthogonal PLR inference."""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_SOLVER = (
    Option(
        "standardize",
        "bool",
        True,
        doc="Divide centered predictors by training-sample RMS; never scale y.",
    ),
    Option("selection", "str", "cv", choices=("fixed", "cv", "plugin")),
    Option("penalty", "float", None, minimum=0),
    Option("lambda_path", "list[float]", None),
    Option("n_lambdas", "int", 30, minimum=2, maximum=200),
    Option("lambda_ratio", "float", 0.001, minimum=1e-8, maximum=0.999999),
    Option("folds", "int", 5, minimum=2, maximum=20),
    Option("seed", "int", 1729, minimum=0),
    Option("max_iterations", "int", 2000, minimum=1, maximum=100000),
    Option("tolerance", "float", 1e-8, minimum=1e-14, maximum=0.01),
    Option("plugin_c", "float", 1.1, minimum=1.0),
    Option("plugin_gamma", "float", 0.1, minimum=1e-8, maximum=0.999999),
    Option("plugin_iterations", "int", 15, minimum=1, maximum=100),
    Option(
        "max_work",
        "int",
        2000000000,
        minimum=1,
        doc="Upper bound on planned scalar solver work; refuse rather than truncate.",
    ),
)
_PREDICTION_SOLVER = _SOLVER + (
    Option(
        "penalty_factors", "json", None,
        doc="Finite nonnegative factors in x order or mapped by original predictor name; multiply L1/L2 literally for each encoded block. Zero forces identified controls. Fixed/CV only.",
    ),
    Option(
        "forced_controls", "list[str]", None,
        doc="Distinct names already present in x whose effective penalty factor is zero. Training-fold rank must identify this block. Fixed/CV only.",
    ),
)
_KERNEL = (
    Option(
        "bandwidth", "json", None, doc="Positive scalar or one positive bandwidth per predictor."
    ),
    Option(
        "bandwidth_path",
        "json",
        None,
        doc="Nonempty list of scalar/vector candidates for exact leave-one-out CV.",
    ),
    Option("selection", "str", "plugin", choices=("fixed", "cv", "plugin")),
    Option(
        "kernel", "str", "gaussian", choices=("gaussian", "epanechnikov", "uniform", "triangular")
    ),
    Option(
        "query",
        "json",
        None,
        doc="Optional finite m by p numeric query matrix; default is estimation sample.",
    ),
    Option("support", "str", "raise", choices=("raise", "extrapolate")),
    Option("min_effective", "float", 2.0, minimum=1.0),
    Option("max_work", "int", 200000000, minimum=1),
)
_GLM_OPTIONS = (
    Option("l1_ratio", "float", 0.5, minimum=0, maximum=1),
    Option("selection", "str", "cv", choices=("fixed", "cv")),
    Option("penalty", "float", None, minimum=0),
    Option("lambda_path", "json", None, doc="1..100 distinct nonnegative absolute penalties; fixed penalty must occur here."),
    Option("n_lambdas", "int", 20, minimum=2, maximum=100),
    Option("lambda_ratio", "float", 0.001, minimum=1e-8, maximum=0.999999),
    Option("folds", "int", 5, minimum=2, maximum=20),
    Option("seed", "int", 1729, minimum=0),
    Option("standardize", "bool", True),
    Option("penalty_factors", "json", None, doc="Literal nonnegative factors by original predictor; multiply both L1/L2 for its encoded block."),
    Option("forced_controls", "list[str]", None, doc="Declared predictor blocks with zero penalty; no automatic inclusion/selection inference."),
    Option("max_iterations", "int", 200, minimum=1, maximum=10000),
    Option("tolerance", "float", 1e-8, minimum=1e-12, maximum=0.001),
    Option("max_work", "int", 2000000000, minimum=1),
    Option("device", "str", "cpu", choices=("cpu",)),
)
_INFERENCE_OPTIONS = (
    _SOLVER[:1]
    + (Option("selection", "str", "plugin", choices=("fixed", "cv", "plugin")),)
    + _SOLVER[2:]
    + (
        Option("nuisance", "str", "lasso", choices=("lasso", "ridge", "elasticnet", "forest")),
        Option("l1_ratio", "float", 0.5, minimum=0, maximum=1),
        Option("forest_config", "json", None, doc="dmlplr nuisance='forest': explicit native trees/depth/min_leaf/mtry/split_candidates configuration."),
    )
)


def _penalized_info(
    name,
    title,
    entry,
    options=_PREDICTION_SOLVER,
    description="Squared-error penalized prediction. No coefficient CI; exact objective, KKT, path, tuning, scaling and original-unit parameters persisted.",
):
    return EstimatorInfo(
        name=name,
        title=title,
        family="regularized",
        entry=entry,
        function=name,
        covariances=("nonrobust",),
        default_covariance="nonrobust",
        categorical=True,
        weights=("fweight", "aweight", "pweight"),
        inference="none",
        options=options,
        description=description,
    )


def _local_info(name, title, entry):
    return EstimatorInfo(
        name=name,
        title=title,
        family="regularized",
        entry=entry,
        function=name,
        covariances=("nonrobust",),
        default_covariance="nonrobust",
        categorical=False,
        inference="none",
        intercept="always",
        options=_KERNEL,
        description="Continuous numeric product-kernel conditional-mean prediction, bandwidth selection and explicit support/boundary policy; no pointwise CI.",
    )


def _orthogonal_info(name, title, entry):
    return EstimatorInfo(
        name=name,
        title=title,
        family="regularized",
        entry=entry,
        function=name,
        covariances=("HC0", "HC1", "cluster") if name == "dmlplr" else ("HC0", "HC1"),
        default_covariance="HC0",
        categorical=False,
        roles=(
            Role(
                "treatment",
                required=True,
                doc="Single scalar treatment; only its PLR effect is an inferential target.",
            ),
        ),
        options=_INFERENCE_OPTIONS,
        description="PLR treatment effect with an orthogonal residual score and influence covariance. DML uses honest row/whole-cluster folds and can use native forest nuisances; in-sample inferential-lasso methods require IID approximate sparsity and valid selection rates.",
    )


ESTIMATORS = (
    EstimatorInfo(
        name="elasticnet_logit", title="Penalized Bernoulli prediction", family="regularized",
        entry="openecon.econometrics.regularized.glm:fit_elasticnet_logit", function="elasticnet_logit",
        covariances=("nonrobust",), default_covariance="nonrobust", outcome="binary", categorical=True,
        weights=("fweight", "aweight", "pweight"), inference="none", options=_GLM_OPTIONS,
        description="Native CPU float64 Bernoulli weighted-mean elastic-net prediction. Full loss/KKT paths, training-fold-only CV, saved category/scaling/factors and restored predictions. A/p weights target empirical prediction loss; no selected coefficient or survey inference, Dataset/GPU or blanket parity.",
    ),
    EstimatorInfo(
        name="elasticnet_poisson", title="Penalized Poisson prediction", family="regularized",
        entry="openecon.econometrics.regularized.glm:fit_elasticnet_poisson", function="elasticnet_poisson",
        covariances=("nonrobust",), default_covariance="nonrobust", outcome="count", categorical=True,
        weights=("fweight", "aweight", "pweight"), inference="none", options=_GLM_OPTIONS,
        description="Native CPU float64 Poisson weighted-mean elastic-net prediction. Full likelihood/KKT paths, training-fold-only CV, saved category/scaling/factors and restored response/link targets. A/p weights target empirical prediction loss; no coefficient or survey inference, Dataset/GPU or blanket parity.",
    ),
    _penalized_info(
        "pls", "Partial least squares (PLS1) prediction",
        "openecon.econometrics.regularized.pls:fit_pls",
        description="Scalar PLS1 prediction: fixed or train-fold-only component CV; numeric or typed categorical predictors and normalized f/a/p empirical weights. Identified forced controls are partialled before components. No coefficient/survey inference, multivariate/GLM PLS, feature penalties or Dataset/GPU route.",
        options=(Option("components", "int", None, minimum=1, maximum=100),
                 Option("selection", "str", "cv", choices=("fixed", "cv")),
                 Option("component_path", "json", None),
                 Option("max_components", "int", 10, minimum=1, maximum=100),
                 Option("standardize", "bool", True),
                 Option("folds", "int", 5, minimum=2, maximum=20),
                 Option("seed", "int", 1729, minimum=0),
                 Option("forced_controls", "list[str]", None, doc="Identified original predictor blocks partialled from outcome and other predictors before PLS components; recovered without selected coefficient inference."),
                 Option("max_work", "int", 2000000000, minimum=1)),
    ),
    _penalized_info(
        "ridge", "Ridge prediction", "openecon.econometrics.regularized.prediction:fit_ridge"
    ),
    _penalized_info(
        "lasso", "Lasso prediction", "openecon.econometrics.regularized.prediction:fit_lasso"
    ),
    _penalized_info(
        "elasticnet",
        "Elastic-net prediction",
        "openecon.econometrics.regularized.prediction:fit_elasticnet",
        options=_PREDICTION_SOLVER + (Option("l1_ratio", "float", 0.5, minimum=0, maximum=1),),
    ),
    _local_info(
        "kernelreg",
        "Nadaraya-Watson kernel prediction",
        "openecon.econometrics.regularized.prediction:fit_kernelreg",
    ),
    _local_info(
        "localreg",
        "Local-linear prediction",
        "openecon.econometrics.regularized.prediction:fit_localreg",
    ),
    _orthogonal_info(
        "postdouble",
        "Post-double-selection treatment effect",
        "openecon.econometrics.regularized.inference:fit_postdouble",
    ),
    _orthogonal_info(
        "partiallingout",
        "Partialling-out treatment effect",
        "openecon.econometrics.regularized.inference:fit_partiallingout",
    ),
    _orthogonal_info(
        "dmlplr",
        "Cross-fitted orthogonal PLR DML treatment effect",
        "openecon.econometrics.regularized.inference:fit_dmlplr",
    ),
)
EXPORTS = {
    "regularized_glm_predict": "openecon.econometrics.regularized.glm:regularized_glm_predict",
    "regularized_glm_path": "openecon.econometrics.regularized.glm:regularized_glm_path",
    "regularized_predict": "openecon.econometrics.regularized.prediction:regularized_predict",
    "regularized_table": "openecon.econometrics.regularized.prediction:regularized_table",
    "local_derivatives": "openecon.econometrics.regularized.derivatives:local_derivatives",
    "local_average_derivatives": "openecon.econometrics.regularized.derivatives:local_average_derivatives",
}
