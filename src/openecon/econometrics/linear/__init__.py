"""linear estimator family manifest (Torch-free).

Least-squares estimators of a linear conditional mean beyond plain OLS:
``areg`` (one absorbed fixed effect), ``reghdfe`` (any number of absorbed
fixed effects) and ``cnsreg`` (linear equality constraints). Plain OLS/WLS
is the ``ols`` estimator of ``openecon.linear_ols``; the Stata-named
``oe.regress`` and ``oe.newey`` published here are thin wrappers over it
(``EXPORTS``). This module imports only the registry so that spec
validation and ``oe.capabilities()`` never load the tensor runtime; the
estimators live in the sibling modules and load when a model is fitted.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

# robust is Stata's vce(robust): the entry functions map it to HC1 (N/(N-K) White sandwich).
_COVARIANCES = ("nonrobust", "HC1", "robust", "cluster")

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="areg", title="Linear regression with one absorbed fixed effect", family="linear",
        entry="openecon.econometrics.linear.areg:fit_areg", function="areg",
        stata=("areg",),
        covariances=_COVARIANCES, default_covariance="nonrobust",
        weights=("aweight", "fweight", "pweight"),
        intercept="always", cluster_dimensions=2, inference="t",
        roles=(Role("absorb", many=False, required=True, kind="label",
                    doc="Categorical variable whose levels are absorbed as fixed effects."),),
        description="Within estimator for one absorbed categorical variable with Stata's areg "
                    "degrees of freedom (the absorbed indicators count as parameters), a reported "
                    "constant and the F test of the absorbed effects.",
    ),
    EstimatorInfo(
        name="reghdfe", title="Linear regression with high-dimensional fixed effects",
        family="linear", entry="openecon.econometrics.linear.reghdfe:fit_reghdfe",
        function="reghdfe", stata=("reghdfe",),
        covariances=_COVARIANCES, default_covariance="nonrobust",
        weights=("aweight", "fweight", "pweight"),
        intercept="never", cluster_dimensions=2, inference="t",
        roles=(Role("absorb", many=True, required=True, kind="label",
                    doc="One or more categorical variables absorbed as fixed effects."),),
        options=(
            Option("drop_singletons", "bool", True,
                   doc="Iteratively drop observations that are alone in a level (reghdfe); "
                       "frequency-weighted rows count as their weight."),
            Option("tolerance", "float", 1e-8,
                   doc="Relative convergence tolerance of the alternating-projection demeaning."),
            Option("max_iterations", "int", 10000, minimum=1,
                   doc="Maximum number of demeaning sweeps."),
        ),
        description="Correia's reghdfe: any number of fixed-effect dimensions absorbed by "
                    "conjugate-gradient alternating projections, singleton removal, exact "
                    "degrees-of-freedom accounting for redundant and cluster-nested effects.",
    ),
    EstimatorInfo(
        name="cnsreg", title="Constrained linear regression", family="linear",
        entry="openecon.econometrics.linear.cnsreg:fit_cnsreg", function="cnsreg",
        stata=("cnsreg",),
        covariances=_COVARIANCES, default_covariance="nonrobust",
        weights=("aweight", "fweight", "pweight"),
        cluster_dimensions=2, inference="t",
        options=(
            Option("constraints", "json", required=True,
                   doc="List of {'terms': {term: coefficient, ...}, 'value': number} linear "
                       "equality constraints R b = r on the design terms (finite numbers)."),
        ),
        description="Least squares under linear equality constraints by exact reparameterization "
                    "(Stata's cnsreg): N-K+q residual degrees of freedom, coefficients fixed by "
                    "the constraints reported separately.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
# regress and newey are Stata-named wrappers that delegate to oe.ols (openecon.linear_ols).
EXPORTS: dict[str, str] = {
    "regress": "openecon.econometrics.linear.wrappers:regress",
    "newey": "openecon.econometrics.linear.wrappers:newey",
}
