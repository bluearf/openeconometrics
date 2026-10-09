"""Single-stage design-based survey procedures; lazy tensor runtime."""

ESTIMATORS = ()
EXPORTS = {
    "survey_four_stage_mean": "openecon.econometrics.survey.four_stage_targets:survey_four_stage_mean",
    "survey_four_stage_total": "openecon.econometrics.survey.four_stage_targets:survey_four_stage_total",
    "survey_four_stage_ratio": "openecon.econometrics.survey.four_stage_targets:survey_four_stage_ratio",
    "survey_four_stage_proportion": "openecon.econometrics.survey.four_stage_targets:survey_four_stage_proportion",
    "survey_four_stage_regress": "openecon.econometrics.survey.four_stage_regression:survey_four_stage_regress",
    "survey_four_stage_logit": "openecon.econometrics.survey.four_stage_regression:survey_four_stage_logit",
    "survey_four_stage_probit": "openecon.econometrics.survey.four_stage_regression:survey_four_stage_probit",
    "survey_four_stage_poisson": "openecon.econometrics.survey.four_stage_regression:survey_four_stage_poisson",
    "survey_fully_stratified_three_stage_mean": "openecon.econometrics.survey.fully_stratified_three_stage_targets:survey_fully_stratified_three_stage_mean",
    "survey_fully_stratified_three_stage_total": "openecon.econometrics.survey.fully_stratified_three_stage_targets:survey_fully_stratified_three_stage_total",
    "survey_fully_stratified_three_stage_ratio": "openecon.econometrics.survey.fully_stratified_three_stage_targets:survey_fully_stratified_three_stage_ratio",
    "survey_fully_stratified_three_stage_proportion": "openecon.econometrics.survey.fully_stratified_three_stage_targets:survey_fully_stratified_three_stage_proportion",
    "survey_fully_stratified_three_stage_regress": "openecon.econometrics.survey.fully_stratified_three_stage_regression:survey_fully_stratified_three_stage_regress",
    "survey_fully_stratified_three_stage_logit": "openecon.econometrics.survey.fully_stratified_three_stage_regression:survey_fully_stratified_three_stage_logit",
    "survey_fully_stratified_three_stage_probit": "openecon.econometrics.survey.fully_stratified_three_stage_regression:survey_fully_stratified_three_stage_probit",
    "survey_fully_stratified_three_stage_poisson": "openecon.econometrics.survey.fully_stratified_three_stage_regression:survey_fully_stratified_three_stage_poisson",
    "survey_stratified_three_stage_mean": "openecon.econometrics.survey.stratified_three_stage_targets:survey_stratified_three_stage_mean",
    "survey_stratified_three_stage_total": "openecon.econometrics.survey.stratified_three_stage_targets:survey_stratified_three_stage_total",
    "survey_stratified_three_stage_ratio": "openecon.econometrics.survey.stratified_three_stage_targets:survey_stratified_three_stage_ratio",
    "survey_stratified_three_stage_proportion": "openecon.econometrics.survey.stratified_three_stage_targets:survey_stratified_three_stage_proportion",
    "survey_stratified_three_stage_regress": "openecon.econometrics.survey.stratified_three_stage_regression:survey_stratified_three_stage_regress",
    "survey_stratified_three_stage_logit": "openecon.econometrics.survey.stratified_three_stage_regression:survey_stratified_three_stage_logit",
    "survey_stratified_three_stage_probit": "openecon.econometrics.survey.stratified_three_stage_regression:survey_stratified_three_stage_probit",
    "survey_stratified_three_stage_poisson": "openecon.econometrics.survey.stratified_three_stage_regression:survey_stratified_three_stage_poisson",

    **{name: f"openecon.econometrics.survey.three_stage_targets:{name}"
       for name in ("survey_three_stage_mean", "survey_three_stage_total", "survey_three_stage_ratio", "survey_three_stage_proportion")},
    **{name: f"openecon.econometrics.survey.three_stage_regression:{name}"
       for name in ("survey_three_stage_regress", "survey_three_stage_logit", "survey_three_stage_probit", "survey_three_stage_poisson")},
    **{name: f"openecon.econometrics.survey.two_stage_targets:{name}"
       for name in ("survey_two_stage_mean", "survey_two_stage_total", "survey_two_stage_ratio", "survey_two_stage_proportion")},
    **{name: f"openecon.econometrics.survey.two_stage_regression:{name}"
       for name in ("survey_two_stage_regress", "survey_two_stage_logit", "survey_two_stage_probit", "survey_two_stage_poisson")},
    "survey_margins_replicate": "openecon.econometrics.survey.replicate_margins:survey_margins_replicate",
    **{
        name: f"openecon.econometrics.survey.targets:{name}"
        for name in ("survey_mean", "survey_total", "survey_ratio", "survey_proportion")
    },
    **{
        name: f"openecon.econometrics.survey.replication:{name}"
        for name in ("survey_brr", "survey_fay", "survey_jackknife", "survey_bootstrap")
    },
    **{
        name: f"openecon.econometrics.survey.regression:{name}"
        for name in ("survey_regress", "survey_logit", "survey_probit", "survey_poisson")
    },
    **{
        name: f"openecon.econometrics.survey.regression_replication:{name}"
        for name in ("survey_regress_replicate", "survey_logit_replicate",
                     "survey_probit_replicate", "survey_poisson_replicate")
    },
    **{
        name: f"openecon.econometrics.survey.regression_postest:{name}"
        for name in ("survey_predict", "survey_margins", "survey_lincom", "survey_test")
    },
}
