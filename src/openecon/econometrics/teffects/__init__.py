"""teffects estimator family manifest (Torch-free).

Treatment effects and policy evaluation: regression adjustment, inverse-
probability weighting and the doubly robust IPWRA and AIPW estimators with
stacked estimating-equation standard errors, nearest-neighbour and
propensity-score matching with Abadie-Imbens standard errors (Stata's
``teffects``), two-way fixed-effects difference in differences with the
parallel-trends and anticipation tests (``didregress`` / ``xtdidregress``),
dynamic event studies, Callaway-Sant'Anna group-time ATTs (``csdid``), and
sharp/fuzzy regression discontinuity with
robust bias-corrected inference and binned scatter data (community
``rdrobust`` / ``rdplot``). Everything is implemented in OpenEconometrics on float64
tensors; see ``docs/econometrics/teffects.md``.
"""

from openecon.econometrics.registry import EstimatorInfo, Option, Role

_PACKAGE = "openecon.econometrics.teffects"


ESTIMATORS: tuple[EstimatorInfo, ...] = (
    EstimatorInfo(
        name="teffects", title="Treatment-effects estimation", family="teffects",
        entry=f"{_PACKAGE}.estimators:fit_teffects", function="teffects",
        stata=("teffects ra", "teffects ipw", "teffects ipwra", "teffects aipw",
               "teffects nnmatch", "teffects psmatch", "teoverlap"),
        covariances=("robust", "cluster"), default_covariance="robust",
        roles=(
            Role("treatment", required=True, kind="label",
                 doc="Treatment variable: two levels, or more for ra/ipw/ipwra/aipw."),
            Role("tx", many=True, doc="Treatment-model covariates (default: the outcome "
                                      "covariates)."),
            Role("biasadj", many=True, doc="nnmatch/psmatch: covariates of the linear "
                                           "bias adjustment (Abadie-Imbens 2011)."),
            Role("ematch", many=True, kind="label", doc="nnmatch: exact-match strata; insufficient cell support is an error."),
        ),
        options=(
            Option("method", "str", "ra",
                   choices=("ra", "ipw", "ipwra", "aipw", "nnmatch", "psmatch"),
                   doc="Estimator: regression adjustment, inverse-probability weighting, IPW "
                       "regression adjustment, augmented IPW, covariate or propensity-score "
                       "matching."),
            Option("estimand", "str", "ate", choices=("ate", "atet", "pomeans"),
                   doc="Average treatment effect, ATE on the treated, or potential-outcome "
                       "means (pomeans: not for matching)."),
            Option("omodel", "str", "linear", choices=("linear", "logit", "probit", "poisson"),
                   doc="Outcome model of ra/ipwra/aipw."),
            Option("tmodel", "str", "logit", choices=("logit", "probit"),
                   doc="Treatment model of ipw/ipwra/aipw/psmatch (multinomial logit for "
                       "more than two levels)."),
            Option("control", "json", None, doc="Control level (default: the lowest level)."),
            Option("tlevel", "json", None, doc="ATET target population (default: first non-control level)."),
            Option("matching_vce", "str", "robust", choices=("robust", "iid"),
                   doc="nnmatch: heteroskedastic or homoskedastic Abadie-Imbens variance."),
            Option("pstolerance", "float", 1e-5, minimum=0.0, maximum=0.5,
                   doc="Overlap check: smallest admissible estimated treatment probability."),
            Option("neighbors", "int", 1, minimum=1,
                   doc="nnmatch/psmatch: number of matches per observation (ties are all "
                       "matched)."),
            Option("metric", "str", "mahalanobis",
                   choices=("mahalanobis", "ivariance", "euclidean"),
                   doc="nnmatch distance metric."),
            Option("caliper", "float", None, minimum=0.0,
                   doc="nnmatch/psmatch: largest admissible match distance (an observation "
                       "without a match within it is an error, as in Stata)."),
            Option("vce_neighbors", "int", 2, minimum=1,
                   doc="nnmatch/psmatch: neighbours of the Abadie-Imbens conditional variance "
                       "estimator (Stata's vce(robust, nn(#)))."),
        ),
        outcome="continuous", predictors="optional", weights=("fweight", "pweight"),
        inference="z",
        description="Stata's teffects: ra, ipw, ipwra and aipw solved as stacked estimating "
                    "equations with the M-estimation sandwich covariance (robust or cluster), "
                    "potential-outcome means and overlap diagnostics; nnmatch and psmatch "
                    "with Abadie-Imbens (2006, 2016) standard errors, bias adjustment and "
                    "caliper.",
    ),
    EstimatorInfo(
        name="didregress", title="Difference-in-differences regression", family="teffects",
        cluster_dimensions=2,
        entry=f"{_PACKAGE}.did:fit_didregress", function="didregress",
        stata=("didregress", "xtdidregress", "estat ptrends", "estat granger"),
        covariances=("robust", "cluster", "HC1", "nonrobust"), default_covariance="robust",
        roles=(Role("treatment", required=True,
                    doc="0/1 indicator: treated group in a treated period."),),
        predictors="optional", intercept="always", panel="required", time="required",
        weights=("aweight", "fweight", "pweight"), inference="t",
        description="Two-way fixed-effects difference in differences: group and time effects "
                    "absorbed, ATET with cluster-robust standard errors at the group level "
                    "(robust, the default), parallel-trends and anticipation (Granger) tests.",
    ),
    EstimatorInfo(
        name="eventstudy", title="Event-study regression", family="teffects",
        cluster_dimensions=2,
        entry=f"{_PACKAGE}.did:fit_eventstudy", function="eventstudy",
        stata=("eventdd", "reghdfe (event-time indicators)"),
        covariances=("robust", "cluster", "HC1", "nonrobust"), default_covariance="robust",
        roles=(Role("treatment_time", required=True,
                    doc="First treated period of the group; missing for never-treated groups."),),
        options=(
            Option("leads", "int", None, minimum=1,
                   doc="Pre-treatment periods shown; earlier ones are binned into the first."),
            Option("lags", "int", None, minimum=0,
                   doc="Post-treatment periods shown; later ones are binned into the last."),
            Option("reference", "int", -1, doc="Omitted reference relative period."),
        ),
        predictors="optional", intercept="always", panel="required", time="required",
        weights=("aweight", "fweight", "pweight"), inference="t",
        description="Dynamic two-way fixed-effects event study with relative-time indicators, "
                    "group-clustered standard errors, a joint pre-trend test and a plotting "
                    "table.",
    ),
    EstimatorInfo(
        name="rdrobust", title="Regression discontinuity (local polynomial)", family="teffects",
        entry=f"{_PACKAGE}.rd:fit_rdrobust", function="rdrobust",
        stata=("rdrobust", "rdbwselect"),
        covariances=("robust", "cluster"), default_covariance="robust",
        roles=(
            Role("running", required=True, doc="Running variable (score)."),
            Role("fuzzy", doc="Treatment received, for a fuzzy design."),
            Role("covariates", many=True, doc="Common-slope covariate adjustment on both sides."),
        ),
        options=(
            Option("cutoff", "float", 0.0, doc="RD cutoff; x >= cutoff is the treated side."),
            Option("p", "int", 1, minimum=0, maximum=8, doc="Order of the local polynomial."),
            Option("q", "int", None, minimum=1, maximum=9,
                   doc="Order of the bias-correction polynomial (default p + 1)."),
            Option("deriv", "int", 0, minimum=0, maximum=8, doc="Derivative order; 1 estimates a kink."),
            Option("masspoints", "str", "adjust", choices=("off", "check", "adjust")),
            Option("bwcheck", "int", None, minimum=1, doc="Minimum unique values in pilot bandwidths."),
            Option("kernel", "str", "triangular",
                   choices=("triangular", "epanechnikov", "uniform")),
            Option("bwselect", "str", "mserd",
                   choices=("mserd", "msetwo", "msesum", "msecomb1", "msecomb2", "cerrd",
                            "certwo", "cersum", "cercomb1", "cercomb2")),
            Option("h", "json", None, doc="Main bandwidth: a number or [left, right]."),
            Option("b", "json", None, doc="Bias bandwidth: a number or [left, right]."),
            Option("rho", "float", None, minimum=0.0, doc="b = h / rho when b is not given."),
            Option("vce", "str", "nn", choices=("nn", "hc0", "hc1", "hc2", "hc3"),
                   doc="Residuals of the variance estimator."),
            Option("nnmatch", "int", 3, minimum=1, doc="Neighbours of vce='nn'."),
            Option("scaleregul", "float", 1.0, minimum=0.0,
                   doc="Scale of the bandwidth regularization term."),
            Option("bwrestrict", "bool", True, doc="Cap bandwidths at the data range."),
            Option("sharpbw", "bool", False,
                   doc="Fuzzy designs: select the bandwidth from the outcome equation alone "
                       "(default: from the linearized ratio of the two jumps)."),
        ),
        predictors="none", categorical=False, intercept="always", inference="z", weights=("aweight",),
        description="Sharp and fuzzy RD by local polynomials with CCT MSE/CER-optimal "
                    "bandwidths; conventional, bias-corrected and robust bias-corrected rows; "
                    "nearest-neighbour or HC variances, cluster option.",
    ),
    EstimatorInfo(
        name="csdid", title="Callaway-Sant'Anna group-time ATT", family="teffects",
        entry=f"{_PACKAGE}.csdid:fit_csdid", function="csdid",
        stata=("csdid", "hdidregress", "xthdidregress"),
        covariances=("robust",), default_covariance="robust",
        roles=(Role("treatment_time", required=True,
                    doc="First treated period of the unit; missing for never-treated units."),),
        options=(
            Option("method", "str", "dr", choices=("dr", "ipw", "reg"),
                   doc="Doubly robust, normalized IPW or outcome regression (with covariates)."),
            Option("control", "str", "never", choices=("never", "notyet"),
                   doc="Control group: never-treated or not-yet-treated units."),
            Option("base", "str", "varying", choices=("varying", "universal"),
                   doc="Pre-treatment base period: t - 1 (varying) or g - 1 (universal)."),
            Option("sample", "str", "balanced", choices=("balanced", "unbalanced", "repeated_cross_section"),
                   doc="Balanced, pair-complete unbalanced panels or independent repeated cross-sections without covariates."),
            Option("anticipation", "int", 0, minimum=0, doc="Potentially affected observed periods before adoption."),
            Option("bootstrap_reps", "int", 0, minimum=0, doc="Shared normal IF multiplier draws; zero retains analytic inference."),
            Option("seed", "int", None, minimum=0, doc="Local multiplier RNG seed."),
            Option("uniform", "bool", False, doc="Separate simultaneous bands for ATT(g,t), dynamic, group and calendar families."),
        ),
        predictors="optional", intercept="always", panel="optional", time="required",
        inference="z",
        description="Heterogeneity-robust staggered DiD: group-time ATTs with outcome-"
                    "regression, IPW or doubly robust estimators, influence-function standard "
                    "errors, simple/dynamic/group/calendar aggregations and a pre-trend test.",
    ),
)

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "causal_evaluate": "openecon.econometrics.postest.causal_evaluation:causal_evaluate",
    "rdplot": f"{_PACKAGE}.rdplot:rdplot",
    "rddensity": f"{_PACKAGE}.density:rddensity",
    "bacon": "openecon.econometrics.teffects.modern_did:bacon",
}

ESTIMATORS = ESTIMATORS + (
    EstimatorInfo(
        name="heterodid", title="Heterogeneous difference in differences", family="teffects",
        entry="openecon.econometrics.teffects.modern_did:fit_heterodid", function="heterodid",
        predictors="none", categorical=False, panel="required", time="required", inference="z",
        covariances=("robust",), default_covariance="robust",
        roles=(Role("treatment", required=True, doc="Binary absorbing adoption indicator."),),
        options=(Option("model", "str", "bjs", choices=("bjs", "sunab")),
                 Option("event_min", "int", 0), Option("event_max", "int", None),
                 Option("max_work", "int", 200_000_000, minimum=1)),
        description="BJS untreated-sample imputation or Sun-Abraham saturated cohort/event interactions; explicit joint unit-cluster covariance on balanced complete panels.",
    ),
    EstimatorInfo(
        name="synthcontrol", title="Synthetic control and synthetic DiD", family="teffects",
        entry="openecon.econometrics.teffects.synthetic:fit_synthcontrol", function="synthcontrol",
        predictors="none", categorical=False, panel="required", time="required", inference="z",
        covariances=("nonrobust",), default_covariance="nonrobust",
        roles=(Role("treatment", required=True, doc="Binary common-date adoption indicator."),),
        options=(Option("model", "str", "sc", choices=("sc", "sdid")),
                 Option("ridge", "float", 0., minimum=0.),
                 Option("zeta_omega", "float", None, minimum=0.),
                 Option("zeta_lambda", "float", None, minimum=0.),
                 Option("placebo_reps", "int", 100, minimum=2, maximum=5000),
                 Option("seed", "int", 0, minimum=0),
                 Option("max_work", "int", 200_000_000, minimum=1)),
        description="Native simplex-constrained outcome-history control and regularized SDID with profiled intercepts; explicit original-control placebo uncertainty and saved weights.",
    ),
)
