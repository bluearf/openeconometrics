"""discrete estimator family manifest (Torch-free).

Discrete-choice models estimated by full maximum likelihood with analytic
scores and Hessians: ``ologit`` / ``oprobit`` (ordered outcomes with directly
estimated cutpoints), ``mlogit`` (multinomial logit, one equation per non-base
category), ``clogit`` (conditional / fixed-effects logit by the recursive
elementary-symmetric-function algorithm), ``hetprobit`` (probit with a
multiplicative variance equation) and ``biprobit`` (bivariate probit with a
bivariate normal distribution function implemented here). Binary ``logit`` and
``probit`` stay in the core. This module imports only the registry so that
spec validation and ``oe.capabilities()`` never load the tensor runtime; see
``docs/econometrics/discrete.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_ML = ("nonrobust", "opg", "robust", "cluster")
_ALL_WEIGHTS = ("fweight", "aweight", "pweight", "iweight")
_OFFSET = Role("offset", doc="Column added to the linear predictor with coefficient 1.")

ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="ologit", title="Ordered logistic regression", family="discrete",
        entry="openecon.econometrics.discrete.ordered:fit_ologit", function="ologit",
        stata=("ologit",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="ordered",
        intercept="never", roles=(_OFFSET,),
        description="Proportional-odds model Pr(y = j) = L(cut_j - x'b) - L(cut_{j-1} - x'b) "
                    "for an ordered outcome (numeric values or an ordered Categorical). No "
                    "constant: the J-1 cutpoints are estimated directly and reported as "
                    "/cut1../cut{J-1}. Newton-Raphson with analytic derivatives, LR or Wald "
                    "model test, McFadden pseudo R-squared, separation detection.",
    ),
    EstimatorInfo(
        name="oprobit", title="Ordered probit regression", family="discrete",
        entry="openecon.econometrics.discrete.ordered:fit_oprobit", function="oprobit",
        stata=("oprobit",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="ordered",
        intercept="never", roles=(_OFFSET,),
        description="Ordered probit Pr(y = j) = Phi(cut_j - x'b) - Phi(cut_{j-1} - x'b) with "
                    "directly estimated cutpoints /cut1../cut{J-1}; same estimator, tests and "
                    "options as ologit.",
    ),
    EstimatorInfo(
        name="mlogit", title="Multinomial logistic regression", family="discrete",
        entry="openecon.econometrics.discrete.mlogit:fit_mlogit", function="mlogit",
        stata=("mlogit",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="categorical",
        options=(
            Option("base", "json", None,
                   doc="Base outcome category (a value of the outcome). Default: the most "
                       "frequent category, as Stata's baseoutcome()."),
        ),
        description="Multinomial logit Pr(y = j) = exp(x'b_j) / sum_l exp(x'b_l) with b = 0 "
                    "for the base category: one equation per other category (terms "
                    "'<category>:<term>'). Newton-Raphson with the analytic k(J-1)-square "
                    "Hessian, LR or Wald model test, McFadden pseudo R-squared.",
    ),
    EstimatorInfo(
        name="clogit", title="Conditional (fixed-effects) logistic regression",
        family="discrete", entry="openecon.econometrics.discrete.clogit:fit_clogit",
        function="clogit", stata=("clogit", "xtlogit, fe"), covariances=_ML,
        default_covariance="nonrobust", weights=("fweight", "iweight", "pweight"),
        cluster_dimensions=2, inference="z", outcome="binary", intercept="never",
        roles=(
            Role("group", required=True, kind="label",
                 doc="Group (matched set, stratum or panel unit) whose effect is conditioned "
                     "out."),
            _OFFSET,
        ),
        description="Conditional logit for a 0/1 outcome within groups: the likelihood "
                    "conditions on the number of positives per group, computed by the "
                    "recursive elementary-symmetric-function algorithm with analytic score "
                    "and Hessian. Groups without outcome variation are dropped and counted; "
                    "weights apply to whole groups; robust means clustered on the group. "
                    "xtlogit, fe is clogit with group = the panel variable.",
    ),
    EstimatorInfo(
        name="hetprobit", title="Heteroskedastic probit regression", family="discrete",
        entry="openecon.econometrics.discrete.hetprobit:fit_hetprobit", function="hetprobit",
        stata=("hetprobit",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="binary",
        roles=(
            Role("het", many=True, required=True,
                 doc="Regressors z of the variance equation ln sigma = z'g (no constant)."),
        ),
        description="Probit with multiplicative heteroskedasticity, Pr(y = 1) = "
                    "Phi(x'b / exp(z'g)). Newton-Raphson with analytic derivatives from the "
                    "probit estimates; variance coefficients are the equation 'lnsigma'; "
                    "Wald model test and LR (or Wald) test of homoskedasticity.",
    ),
    EstimatorInfo(
        name="biprobit", title="Bivariate probit regression", family="discrete",
        entry="openecon.econometrics.discrete.biprobit:fit_biprobit", function="biprobit",
        stata=("biprobit",), covariances=_ML, default_covariance="nonrobust",
        weights=_ALL_WEIGHTS, cluster_dimensions=2, inference="z", outcome="binary",
        roles=(
            Role("outcome2", required=True,
                 doc="Second binary outcome (0/1); the model outcome is the first."),
            Role("predictors2", many=True,
                 doc="Regressors of the second equation (default: the predictors of the "
                     "first), giving the seemingly unrelated bivariate probit."),
        ),
        description="Two probit equations with correlated errors, Pr(y1, y2) = "
                    "Phi2(q1 x1'b1, q2 x2'b2; q1 q2 rho). Newton-Raphson with analytic score "
                    "and Hessian on (b1, b2, athrho); the bivariate normal distribution "
                    "function is Genz's algorithm implemented here. Reports rho with a "
                    "delta-method standard error, a Wald model test and the LR (or Wald) "
                    "test of rho = 0.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "mprobit": "openecon.econometrics.discrete.mprobit:mprobit",
    "mprobit_restore": "openecon.econometrics.discrete.mprobit:mprobit_restore",
    "mprobit_predict": "openecon.econometrics.discrete.mprobit_postestimation:mprobit_predict",
    "mprobit_margins": "openecon.econometrics.discrete.mprobit_postestimation:mprobit_margins",
    "nlogit": "openecon.econometrics.discrete.nested_logit:nlogit",
    "nlogit_restore": "openecon.econometrics.discrete.nested_logit:nlogit_restore",
    "nlogit_predict": "openecon.econometrics.discrete.nested_logit_postestimation:nlogit_predict",
    "nlogit_margins": "openecon.econometrics.discrete.nested_logit_postestimation:nlogit_margins",
    "rologit": "openecon.econometrics.discrete.rank_ordered:rologit",
    "rologit_restore": "openecon.econometrics.discrete.rank_ordered:rologit_restore",
    "rologit_predict": "openecon.econometrics.discrete.rank_ordered_postestimation:rologit_predict",
    "rologit_margins": "openecon.econometrics.discrete.rank_ordered_postestimation:rologit_margins",
}
