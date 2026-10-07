"""quantile estimator family manifest (Torch-free).

Quantile regression (``qreg`` and its bootstrap relatives ``bsqreg``, ``sqreg``,
``iqreg``), Stata's robust regression ``rreg`` and nonlinear least squares
``nl``. Every estimator is implemented in OpenEconometrics on float64 tensors; see
``docs/econometrics/quantile.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option

_KERNELS = ("epanechnikov", "epan2", "biweight", "cosine", "gaussian", "parzen", "rectangle",
            "triangle")
_REPS = Option("reps", "int", 20, minimum=2, doc="Bootstrap replications (Stata's reps()).")
_SEED = Option("seed", "int", None, minimum=0, maximum=2 ** 63 - 1,
               doc="Seed of the bootstrap resampling; drawn and recorded when omitted.")

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="qreg", title="Quantile regression", family="quantile",
        entry="openecon.econometrics.quantile.qreg:fit_qreg", function="qreg",
        stata=("qreg",), covariances=("nonrobust", "robust", "cluster"),
        default_covariance="nonrobust", predictors="optional",
        weights=("aweight", "fweight", "pweight"), inference="t",
        options=(
            Option("quantile", "float", 0.5, doc="Quantile to estimate, strictly inside (0, 1)."),
            Option("density", "str", None, choices=("fitted", "residual", "kernel"),
                   doc="Density estimator (Stata's denmethod): fitted (default for nonrobust "
                       "and robust), residual (nonrobust only) or kernel (default for "
                       "cluster)."),
            Option("bandwidth", "str", "hsheather",
                   choices=("hsheather", "bofinger", "chamberlain"),
                   doc="Bandwidth rule of the sparsity/density estimator."),
            Option("kernel", "str", "epanechnikov", choices=_KERNELS,
                   doc="Kernel of density='kernel'; giving it selects that method."),
        ),
        description="Stata's qreg: linear quantile regression solved exactly by a Frisch-Newton "
                    "interior-point method. nonrobust is vce(iid) with a fitted, residual or "
                    "kernel sparsity estimate, robust the Hendricks-Koenker (fitted) or Powell "
                    "(kernel) sandwich, cluster the Parente-Santos Silva cluster-robust "
                    "covariance. t inference with N - K degrees of freedom.",
    ),
    EstimatorInfo(
        name="bsqreg", title="Bootstrapped quantile regression", family="quantile",
        entry="openecon.econometrics.quantile.sqreg:fit_bsqreg", function="bsqreg",
        stata=("bsqreg",), covariances=("bootstrap",), default_covariance="bootstrap",
        predictors="optional", inference="t",
        options=(
            Option("quantile", "float", 0.5, doc="Quantile to estimate, strictly inside (0, 1)."),
            _REPS, _SEED,
        ),
        description="Quantile regression with a pairs-bootstrap covariance (resampling whole "
                    "clusters when a cluster column is given).",
    ),
    EstimatorInfo(
        name="sqreg", title="Simultaneous quantile regression", family="quantile",
        entry="openecon.econometrics.quantile.sqreg:fit_sqreg", function="sqreg",
        stata=("sqreg",), covariances=("bootstrap",), default_covariance="bootstrap",
        predictors="optional", inference="t",
        options=(
            Option("quantiles", "list[float]", [0.25, 0.5, 0.75],
                   doc="Distinct quantiles, each strictly inside (0, 1); one equation each."),
            _REPS, _SEED,
        ),
        description="Several quantile regressions estimated on the same bootstrap samples, so "
                    "the joint covariance across quantiles is available for cross-quantile "
                    "tests.",
    ),
    EstimatorInfo(
        name="iqreg", title="Interquantile range regression", family="quantile",
        entry="openecon.econometrics.quantile.sqreg:fit_iqreg", function="iqreg",
        stata=("iqreg",), covariances=("bootstrap",), default_covariance="bootstrap",
        predictors="optional", inference="t",
        options=(
            Option("quantiles", "list[float]", [0.25, 0.75],
                   doc="The lower and the upper quantile (two values inside (0, 1))."),
            _REPS, _SEED,
        ),
        description="Difference between two quantile regressions (default: the interquartile "
                    "range) with bootstrap standard errors.",
    ),
    EstimatorInfo(
        name="rreg", title="Robust regression", family="quantile",
        entry="openecon.econometrics.quantile.rreg:fit_rreg", function="rreg",
        stata=("rreg",), covariances=("nonrobust",), default_covariance="nonrobust",
        inference="t",
        options=(
            Option("tune", "number", 7,
                   doc="Biweight tuning constant: c = 4.685 * tune / 7 scaled residuals."),
            Option("tolerance", "float", 0.01,
                   doc="Convergence: maximum change in the case weights between iterations."),
        ),
        description="Stata's rreg: Cook's distance screening, Huber iterations, then biweight "
                    "iterations; standard errors from the pseudovalues of Street, Carroll and "
                    "Ruppert (1988).",
    ),
    EstimatorInfo(
        name="nl", title="Nonlinear least squares", family="quantile",
        entry="openecon.econometrics.quantile.nl:fit_nl", function="nl",
        stata=("nl",), covariances=("nonrobust", "robust", "HC2", "HC3", "cluster"),
        default_covariance="nonrobust", predictors="optional", categorical=False,
        intercept="never", weights=("aweight", "fweight", "pweight"), inference="t",
        options=(
            Option("formula", "str", None, required=True,
                   doc="Regression function with parameters in braces, e.g. "
                       "'{b0} + {b1} * exp(-{b2} * x)'."),
            Option("start", "json", None,
                   doc="Starting values {parameter: value}; they override {b=value} in the "
                       "formula. Parameters without a start begin at 0."),
            Option("max_iterations", "int", 1000, minimum=1, doc="Iteration limit."),
            Option("tolerance", "float", 1e-8,
                   doc="Relative convergence tolerance on the parameters and the residual sum "
                       "of squares."),
        ),
        description="Stata's nl for a substitutable expression: Gauss-Newton with "
                    "Levenberg-Marquardt damping and an analytic Jacobian obtained by "
                    "differentiating the parsed formula.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {}
