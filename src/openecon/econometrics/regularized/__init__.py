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
_INFERENCE_OPTIONS = (
    _SOLVER[:1]
    + (Option("selection", "str", "plugin", choices=("fixed", "cv", "plugin")),)
    + _SOLVER[2:]
    + (
        Option("nuisance", "str", "lasso", choices=("lasso", "ridge", "elasticnet")),
        Option("l1_ratio", "float", 0.5, minimum=0, maximum=1),
    )
)


def _penalized_info(name, title, entry, options=_SOLVER):
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
        options=options,
        description="Squared-error penalized prediction. No coefficient CI; exact objective, KKT, path, tuning, scaling and original-unit parameters persisted.",
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
        covariances=("HC0", "HC1"),
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
        description="PLR treatment effect with an orthogonal residual score and influence covariance. IID approximate-sparsity/nuisance-rate assumptions required; only DML uses honest cross-fitting.",
    )


ESTIMATORS = (
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
        options=_SOLVER + (Option("l1_ratio", "float", 0.5, minimum=0, maximum=1),),
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
    "regularized_predict": "openecon.econometrics.regularized.prediction:regularized_predict",
    "regularized_table": "openecon.econometrics.regularized.prediction:regularized_table",
}
