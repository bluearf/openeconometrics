"""Resident CPU smoothing stages; static contracts, no numerical imports."""

from openecon.econometrics.registry import EstimatorInfo, Option

_WORK = (Option("max_work", "int", 2000000000, minimum=1),)
_KNOT = (
    Option("knots", "json", None),
    Option("n_knots", "int", 3, minimum=0, maximum=40),
    Option("boundary", "json", None),
)


def _info(
    name,
    title,
    module,
    options,
    inference="none",
    description="Resident CPU float64 numeric smoothing with persisted transformations and bounded work. No weights, Dataset or device route.",
):
    return EstimatorInfo(
        name=name,
        title=title,
        family="smoothing",
        entry=f"openecon.econometrics.smoothing.{module}:fit_{name}",
        function=name,
        covariances=("nonrobust",),
        default_covariance="nonrobust",
        categorical=False,
        intercept="always",
        inference=inference,
        options=options + _WORK,
        description=description,
    )


ESTIMATORS = (
    _info(
        "bspline_regress",
        "B-spline regression",
        "splines",
        _KNOT + (Option("degree", "int", 3, minimum=1, maximum=3),),
        "t",
        "Fixed/training-quantile Cox-de Boor spline basis, Gaussian iid OLS covariance and conditional t inference; reject outside boundary. Saved basis replay.",
    ),
    _info(
        "rcs_regress",
        "Restricted cubic spline regression",
        "splines",
        (Option("knots", "json", None), Option("n_knots", "int", 5, minimum=3, maximum=40)),
        "t",
        "Truncated-power natural cubic basis with linear tails and Gaussian iid OLS full covariance; basis and knots persisted. Conditional t inference.",
    ),
    _info(
        "gam_gaussian",
        "Penalized Gaussian additive splines",
        "splines",
        _KNOT
        + (
            Option("penalty", "float", 1.0, minimum=0),
            Option("penalty_path", "list[float]", None),
            Option("selection", "str", "fixed", choices=("fixed", "gcv")),
        ),
        description="Centered cubic additive B-splines with second-difference penalties, training GCV and full frequentist covariance conditional on smoothing. Approximate pointwise estimator mean intervals only, excluding smoothing bias/selection uncertainty; no coefficient p values.",
    ),
    _info(
        "loess",
        "Direct one-dimensional LOESS",
        "local",
        (
            Option("span", "float", 0.75, minimum=0.01, maximum=1),
            Option("span_path", "list[float]", None),
            Option("selection", "str", "fixed", choices=("fixed", "loo")),
            Option("degree", "int", 1, minimum=1, maximum=2),
        ),
        description="One-dimensional direct Gaussian tricube local polynomial prediction, exact leave-one-out span selection and rank/support guards; no inferential intervals.",
    ),
    _info(
        "fp_regress",
        "Fractional polynomial regression",
        "fractional",
        (Option("powers", "list[float]", [1.0]), Option("scale", "float", 1.0, minimum=1e-300)),
        "t",
        "One positive continuous FP variable with one/two explicit powers, repeated-power log transforms and fixed linear adjustment variables. Full OLS covariance and conditional t inference.",
    ),
    _info(
        "mfp_regress",
        "Single-variable closed-test fractional polynomial selection",
        "fractional",
        (
            Option("scale", "float", 1.0, minimum=1e-300),
            Option("select_alpha", "float", 0.05, minimum=1e-8, maximum=0.999999),
            Option("form_alpha", "float", 0.05, minimum=1e-8, maximum=0.999999),
        ),
        "t",
        "Gaussian single nonlinear variable with fixed linear adjustments: eight FP1/36 FP2 fits and approximate 4/3/2 df closed F tests. Selected-model conditional inference; no unconditional postselection or general multivariable cycling claim.",
    ),
    _info(
        "mars",
        "Bounded adaptive paired-hinge regression",
        "adaptive",
        (
            Option("max_terms", "int", 11, minimum=3, maximum=41),
            Option("max_degree", "int", 1, minimum=1, maximum=3),
            Option("min_span", "int", 3, minimum=2),
            Option("max_candidates", "int", 20, minimum=1, maximum=100),
            Option("gcv_penalty", "float", 3.0, minimum=0, maximum=10),
        ),
        description="Deterministic bounded training paired-hinge forward SSE and backward GCV pruning, explicit candidate-grid/min-span/order policy and saved interactions. Predictive coefficients only; no inferential p/CI.",
    ),
    _info(
        "npreg_mixed",
        "Mixed product-kernel conditional means",
        "local",
        (
            Option("variable_types", "json", required=True),
            Option("categories", "json", required=True),
            Option("bandwidth", "list[float]", required=True),
            Option("bandwidth_path", "json", None),
            Option("selection", "str", "fixed", choices=("fixed", "loo")),
            Option("min_effective", "float", 2.0, minimum=1),
        ),
        description="Gaussian numeric/Aitchison-Aitken unordered/Wang-van Ryzin ordered product kernels with declared category universes, stable weights and training LOO selection; mean prediction only, no CI.",
    ),
)
EXPORTS = {
    "smoothing_predict": "openecon.econometrics.smoothing.replay:smoothing_predict",
    "spline_derivative": "openecon.econometrics.smoothing.postestimation:spline_derivative",
    "fp_derivative": "openecon.econometrics.smoothing.postestimation:fp_derivative",
    "gam_derivative": "openecon.econometrics.smoothing.postestimation:gam_derivative",
    "mars_derivative": "openecon.econometrics.smoothing.postestimation:mars_derivative",
    "loess_derivative": "openecon.econometrics.smoothing.postestimation:loess_derivative",
    "kernel_derivative": "openecon.econometrics.smoothing.postestimation:kernel_derivative",
    "smoothing_margins": "openecon.econometrics.smoothing.postestimation:smoothing_margins",
    "smoothing_contrast": "openecon.econometrics.smoothing.postestimation:smoothing_contrast",
}
