"""S/MM regression and panel quantiles via moments (Torch-free manifest)."""

from openecon.econometrics.registry import EstimatorInfo, Option

_S_OPTIONS = (
    Option(
        "breakdown",
        "float",
        0.5,
        minimum=0.05,
        maximum=0.5,
        doc="Normalized bisquare scale equation target b; nominal asymptotic breakdown b.",
    ),
    Option(
        "scale_tune",
        "float",
        None,
        minimum=0.01,
        doc="Bisquare S tuning; omitted calibrates E rho(N(0,1))=breakdown.",
    ),
    Option(
        "starts",
        "int",
        500,
        minimum=1,
        maximum=100000,
        doc="Seeded elemental subsets for the approximate S minimum.",
    ),
    Option(
        "seed",
        "int",
        0,
        minimum=0,
        maximum=2**63 - 1,
        doc="Local Torch generator seed; the global RNG is untouched.",
    ),
    Option("max_iterations", "int", 200, minimum=1, maximum=10000),
    Option("tolerance", "float", 1e-8, minimum=1e-12, maximum=1e-3),
)

ESTIMATORS = (
    EstimatorInfo(
        name="sreg",
        title="Bisquare S regression",
        family="robust",
        entry="openecon.econometrics.robust.smm:fit_sreg",
        function="sreg",
        covariances=("nonrobust", "robust", "cluster"),
        default_covariance="robust",
        predictors="optional",
        inference="t",
        options=_S_OPTIONS,
        description="Seeded elemental-start S regression minimizing a bisquare M-scale. "
        "The finite randomized search does not certify a global minimum or breakdown.",
    ),
    EstimatorInfo(
        name="mmreg",
        title="Bisquare MM regression",
        family="robust",
        entry="openecon.econometrics.robust.smm:fit_mmreg",
        function="mmreg",
        covariances=("nonrobust", "robust", "cluster"),
        default_covariance="robust",
        predictors="optional",
        inference="t",
        options=(
            *_S_OPTIONS,
            Option(
                "efficiency_tune",
                "float",
                4.685061,
                minimum=0.01,
                doc="Final bisquare c, at least S c; default gives 95% Gaussian efficiency.",
            ),
        ),
        description="High-breakdown S start followed by efficient bisquare M estimation "
        "at the FIXED S scale, with objective descent and scale-aware sandwich.",
    ),
    EstimatorInfo(
        name="panel_mmqr",
        title="Panel quantiles via moments",
        family="robust",
        entry="openecon.econometrics.robust.panel_mmqr:fit_panel_mmqr",
        function="panel_mmqr",
        stata=("xtqreg (user-written; coefficient algorithm)",),
        covariances=("HC0", "robust", "cluster"),
        default_covariance="robust",
        panel="required",
        time="optional",
        intercept="never",
        inference="t",
        options=(
            Option(
                "quantiles",
                "list[float]",
                [0.25, 0.5, 0.75],
                doc="Distinct quantiles in [0.05,0.95], reported jointly.",
            ),
            Option(
                "density_bandwidth",
                "float",
                None,
                minimum=1e-12,
                doc="Gaussian KDE bandwidth for standardized residuals; default Silverman.",
            ),
        ),
        description="Machado-Santos Silva linear location/scale panel moments, same X in "
        "both equations, absorbed distributional individual effects, strictly "
        "positive fitted scale. Joint profile-moment covariance; robust clusters "
        "on the panel. Large-T assumptions and incidental-parameter bias apply.",
    ),
)

EXPORTS = {
    "panel_mmqr_bootstrap": "openecon.econometrics.robust.panel_resampling:panel_mmqr_bootstrap",
    "panel_mmqr_resampled_predict": "openecon.econometrics.robust.panel_resampling:panel_mmqr_resampled_predict",
}
