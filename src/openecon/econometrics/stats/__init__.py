"""Classical parametric statistics: manifest of the ``stats`` family (Torch-free).

These are test procedures, not model fits: they register no estimator and are
published as ``oe.<name>`` through ``EXPORTS``. Each returns a result table or a
``TableSet`` of tables (see ``docs/econometrics/stats.md``).
"""

from openecon.econometrics.registry import EstimatorInfo

ESTIMATORS: tuple[EstimatorInfo, ...] = ()

# Extra public functions exported as oe.<name>: {"name": "package.module:function"}.
EXPORTS: dict[str, str] = {
    "rm_contrast": "openecon.econometrics.stats.rm_contrast:rm_contrast",
    "ttest": "openecon.econometrics.stats.ttest:ttest",
    "sdtest": "openecon.econometrics.stats.ttest:sdtest",
    "oneway": "openecon.econometrics.stats.oneway:oneway",
    "anova": "openecon.econometrics.stats.anova:anova",
    "rm_anova": "openecon.econometrics.stats.rm_anova:rm_anova",
    "manova": "openecon.econometrics.stats.manova:manova",
    "manova_oneway": "openecon.econometrics.stats.manova_options:manova_oneway",
    "manova_summary": "openecon.econometrics.stats.manova_options:manova_summary",
    "manova_contrast": "openecon.econometrics.stats.manova_options:manova_contrast",
    "rm_mtest": "openecon.econometrics.stats.manova_options:rm_mtest",
    "manova_factorial": "openecon.econometrics.stats.manova_factorial:manova_factorial",
    "manova_factorial_summary": "openecon.econometrics.stats.manova_factorial:manova_factorial_summary",
    "rm_anova_fweight": "openecon.econometrics.stats.rm_moments:rm_anova_fweight",
    "rm_anova_summary": "openecon.econometrics.stats.rm_moments:rm_anova_summary",
    "correlate": "openecon.econometrics.stats.correlation:correlate",
    "pcorr": "openecon.econometrics.stats.correlation:pcorr",
    "describe": "openecon.econometrics.stats.descriptives:describe",
    "power_mean": "openecon.econometrics.stats.planning:power_mean",
    "power_proportion": "openecon.econometrics.stats.planning:power_proportion",
    "power_correlation": "openecon.econometrics.stats.planning:power_correlation",
    "power_tmean": "openecon.econometrics.stats.power_designs:power_tmean",
    "power_ttwomeans": "openecon.econometrics.stats.power_designs:power_ttwomeans",
    "power_tpaired": "openecon.econometrics.stats.power_designs:power_tpaired",
    "power_anova": "openecon.econometrics.stats.power_designs:power_anova",
    "power_regression": "openecon.econometrics.stats.power_designs:power_regression",
    "power_cluster_mean": "openecon.econometrics.stats.power_designs:power_cluster_mean",
    "power_gof": "openecon.econometrics.stats.power_designs:power_gof",
    "power_independence": "openecon.econometrics.stats.power_designs:power_independence",
    "precision_mean": "openecon.econometrics.stats.planning:precision_mean",
    "power_paired_mean": "openecon.econometrics.stats.planning_extended:power_paired_mean",
    "power_two_proportions": "openecon.econometrics.stats.planning_extended:power_two_proportions",
    "power_two_correlations": "openecon.econometrics.stats.planning_extended:power_two_correlations",
    "power_slope": "openecon.econometrics.stats.planning_extended:power_slope",
    "power_logrank": "openecon.econometrics.stats.planning_extended:power_logrank",
    "power_mcnemar": "openecon.econometrics.stats.planning_extended:power_mcnemar",
    "precision_mean_unknown": "openecon.econometrics.stats.planning_extended:precision_mean_unknown",
    "precision_variance": "openecon.econometrics.stats.planning_extended:precision_variance",
    "power_welch": "openecon.econometrics.stats.planning_heterogeneous:power_welch",
    "power_unbalanced_anova": "openecon.econometrics.stats.planning_heterogeneous:power_unbalanced_anova",
    "power_unequal_cluster_mean": "openecon.econometrics.stats.planning_heterogeneous:power_unequal_cluster_mean",
    "power_mcnemar_unconditional": "openecon.econometrics.stats.planning_discrete_survival:power_mcnemar_unconditional",
    "precision_twomeans_unknown": "openecon.econometrics.stats.planning_precision:precision_twomeans_unknown",
    "precision_binomial": "openecon.econometrics.stats.planning_precision:precision_binomial",
    "precision_poisson": "openecon.econometrics.stats.planning_precision:precision_poisson",
    "survival_accrual": "openecon.econometrics.stats.planning_discrete_survival:survival_accrual",
    "planning_scenarios": "openecon.econometrics.stats.planning_scenarios:planning_scenarios",
    "planning_plot": "openecon.econometrics.stats.planning_scenarios:planning_plot",
}
