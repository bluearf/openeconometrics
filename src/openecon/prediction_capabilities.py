"""Lightweight saved-prediction contract; importing this does not load kernels.

The inventories are checked against the actual numerical dispatch by tests.
Estimator-name coverage does not imply every fitted option has a response mean.
"""
from __future__ import annotations

from collections.abc import Iterable


DIRECT_ESTIMATORS = frozenset({
    "ols", "cnsreg", "ivregress", "logit", "probit", "glm", "poisson",
    "nbreg", "cloglog", "fracreg", "betareg", "gnbreg", "cpoisson", "cnbreg",
    "hetprobit", "tobit", "truncreg", "intreg", "ologit", "oprobit", "mlogit",
})
LINEAR_ESTIMATORS = frozenset({
    "areg", "reghdfe", "ivreghdfe", "xtreg", "xtivreg", "xtgls", "xtpcse",
    "xtfmb", "prais", "rreg", "qreg", "bsqreg", "iqreg", "sqreg", "xtgee",
    "ppmlhdfe", "mixed", "xtlogit", "xtprobit", "xtpoisson",
})
COUNT_ESTIMATORS = frozenset({"zip", "zinb", "hurdle", "tpoisson", "tnbreg"})
GROUP_ESTIMATORS = frozenset({'melogit', 'meprobit', 'mepoisson', 'menbreg'})
ADVANCED_ESTIMATORS = frozenset({"nl", "sureg", "mvreg", "reg3", "threshold", "biprobit",
    "heckman", "heckprobit", "churdle", "ivprobit", "ivtobit", "frontier"})
SAVED_ESTIMATORS = DIRECT_ESTIMATORS | LINEAR_ESTIMATORS | COUNT_ESTIMATORS | ADVANCED_ESTIMATORS | GROUP_ESTIMATORS


def describe(registered: Iterable[str]) -> dict:
    """Describe common saved prediction independently of fitting coverage."""
    names = frozenset(registered)
    supported = sorted(SAVED_ESTIMATORS & names)
    remaining = sorted(names - SAVED_ESTIMATORS)
    conditions = {
        name: {"predict_kinds": ["xb", "stdp", "response", "derivative"], "margins_kinds": ["xb", "response"],
               "response": 'conditional using saved fitted effect map; historical state absent is rejected',
               "response_inference": 'full nonrobust nuisance information within recorded geometry; otherwise point means only',
               "xb_excludes_group_effects": True}
        for name in ("areg", "reghdfe", "ivreghdfe", "ppmlhdfe")
    }
    conditions.update({
        name: {
            "fe": {"predict_kinds": ["xb", "stdp", "response"], "margins_kinds": ["xb", "response"],
                   "response": 'saved known-group conditional mean; full nuisance inference when recorded', "xb_excludes_group_effects": True},
            "be": {"predict_kinds": ["xb", "stdp"], "margins_kinds": ["xb"],
                   "response": False, "requires_supplied_panel_mean_covariates": True},
            "fd": {"supported": False, "reason": "raw levels omit the fitted difference transformation"},
            "re": {"response": "population linear mean; no group-conditioned BLUP"},
            **({"pooled": {"response": "pooled linear mean"},
                "mle": {"response": "population linear mean; no group-conditioned BLUP"}}
               if name == "xtreg" else {}),
        }
        for name in ("xtreg", "xtivreg")
    })
    conditions.update({
        name: {"model": ['pa', 're'], "response": "PA saved link or explicitly integrated normal RE likelihood; domains are separate",
               "fe": 'complete fitted alternatives and success count for xtlogit; other FE targets require their own adapter'}
        for name in ("xtlogit", "xtprobit", "xtpoisson")
    })
    conditions.update({
        **{name: {'response': 'normal population integral with full variance/covariance delta gradient',
                  'targets': ['population', 'conditional'], 'conditional': 'explicit effects held fixed',
                  'posterior': 'complete fitted group normal posterior' if name != 'menbreg' else False,
                  'margins_methods': ['ame']} for name in GROUP_ESTIMATORS},
        'clogit': {'response': 'exact complete-group conditional inclusion probability', 'success_count': 'fitted count', 'complete_alternative_set': 'exact retained row multiset'},
        "nl": {"response":"saved safe formula function", "mem":"global weighted raw numeric covariate means"},
        **{name:{"outcome":"required saved equation name for multiple equations", "response":"selected structural equation"}
           for name in ("sureg","mvreg","reg3")},
        "threshold": {"response":"saved regime mean; q <= threshold ties in lower regime", "boundary_effect":"undefined", "inference":"conditions on fitted thresholds"},
        "biprobit": {"outcome":["joint00","joint01","joint10","joint11","marginal1","marginal2","conditional1","conditional2"], "recursive":"requires explicit binary intervention path; unsupported"},
        "heckman": {"outcome":["mean","conditional","selection"],"methods":["ml","twostep"]},
        "heckprobit": {"outcome":["joint11","marginal1","selection","conditional1"]},
        "churdle": {"outcome":["mean","conditional","participation"],"models":["linear","exponential"]},
        **{name:{"ml":"structural response at supplied endogenous covariates", "twostep":"xb/stdp/latent only; missing structural response scale"}
           for name in ("ivprobit","ivtobit")},
        "frontier": {"outcome":["frontier","mean","u","te"],"posterior_requires_observed_outcome":["u","te"]},
        "mixed": {"response": "population linear mean; group-conditioned BLUP uses mixed_predict"},
        "rreg": {"response": "robust fitted linear location"},
        "qreg": {"response": "saved conditional quantile"},
        "bsqreg": {"response": "saved conditional quantile"},
        "iqreg": {"response": "saved interquantile difference"},
        "sqreg": {"response": "selected saved quantile equation",
                  "outcome": "required when several quantile equations exist"},
        "intreg": {"response": "latent unconditional normal mean", "conditional": False},
        "hetprobit": {"additional_kind": "sigma", "requires_both_equation_predictors": True},
        "zip": {"response": "(1-inflation probability) times latent count mean",
                "conditional": "positive-count conditional mean"},
        "zinb": {"response": "(1-inflation probability) times latent count mean",
                 "conditional": "positive-count conditional mean"},
        "hurdle": {"response": "participation probability times positive-count conditional mean",
                   "conditional": "positive-count conditional mean"},
        "tpoisson": {"response": "count mean conditional on exceeding fitted lower cutoff",
                     "row_cutoff": "nonnegative integer; maximum 1e12",
                     "mem": "global encoded cutoff must remain a valid integer"},
        "tnbreg": {"response": "count mean conditional on exceeding fitted lower cutoff",
                   "row_cutoff": "nonnegative integer; maximum 1e7 - 1",
                   "mem": "global encoded cutoff must remain a valid integer"},
    })
    return {
        "saved_prediction": {
            "estimators": supported, "estimator_count": len(supported),
            "remaining_estimators": remaining, "remaining_estimator_count": len(remaining),
            "coverage_scope": "estimator-name dispatch; fitted options and target domains remain explicit",
            "refits": False, "precision": "float64", "devices": ["cpu"],
            "full_saved_parameter_covariance": True,
            "default_kind": "response", "native_ols_default_kind": "xb",
            "conditions": conditions,
            "interval": {"mean": "pointwise delta method with saved normal/Student t inference",
                         "observation": False},
            "dataset": {
                "supported": True, "requires_explicit_evaluation_data": True,
                "row_limit": None, "maximum_batch_rows": 65536,
                "predict_output": "owned indexed temporary Parquet Dataset",
                "predict_materialization": "complete disk output before return; bounded memory",
                "preserves_input_index_and_missing_rows": True,
                "margins_methods": ["ame", "mem"], "maximum_at_combinations": 1000,
                "ame": "global weighted effects and gradients; full covariance applied once",
                "mem": "global weighted encoded design; one evaluation at its means",
                "at": "global evaluation per original-predictor grid cell",
                "source_integrity": "selected-column and index digest checked after every complete replay",
                "sample": "explicit eligible evaluation rows; never inferred from stored chart preview",
                "resource_scope": "bounded planned buffers; model width, source reader, I/O and local disk still matter",
                'complete_groups': {'estimators': ['clogit','mixed','melogit','meprobit','mepoisson','xtlogit'],
                                    'maximum_group_rows_default': 100000, 'input_output_disk_budget_default': 1073741824,
                                    'scratch_cleanup': True, 'unknown_groups': 'rejected'},
            },
            "domains_document": "docs/econometrics/prediction-domains.md",
        },
        "specialized_saved_targets": {
            "common_scalar_dispatch": False,
            "refits": False,
            "output": "owned indexed temporary Parquet Dataset",
            "survival_predict": {
                "stcox": ["relative_hazard", "survival", "cumulative_hazard", "hazard_jump", "quantile"],
                "streg": ["survival", "hazard", "cumulative_hazard", "relative_hazard", "mean", "quantile"],
                "conditions": "explicit time/stratum, saved PH/AFT/distribution and full matching Cox baseline; TVC unsupported",
                "uncertainty": "full saved parameter delta; Cox counting-process baseline variance excluded",
            },
            "cox_baseline": "full immutable Dataset risk replay; display thinning never used",
            "dynamic_predict": {
                "history": "explicit Dataset plus end or integer origin; source integrity rechecked",
                "targets": {
                    "arima": ["forecast_levels"], "arch": ["forecast_levels", "conditional_variance"],
                    "var": ["forecast_levels"], "vec": ["forecast_levels"],
                    "ucm": ["forecast_levels", "components_smoothed"],
                    "mswitch": ["probabilities_filtered", "probabilities_predicted", "probabilities_smoothed"],
                    "ardl": ["fitted_levels"], "nardl": ["fitted_levels"],
                    "ahreg": ["fitted_transformed"], "xtdpd": ["fitted_transformed"],
                },
                "uncertainty": "existing native family forecast/filter convention; fitted equations use full parameter covariance",
            },
            "causal_evaluate": {
                "estimators": ["teffects", "didregress", "eventstudy", "csdid", "rdrobust", "gmm"],
                "population": ["estimation", "fixed_evaluation"],
                "conditions": "identified original contrasts; explicit nuisance outcome or GMM function; exact RD cutoff/support",
                "standardization": "global value/gradient reduction; evaluation rows fixed, no transported effect claim",
            },
            "document": "docs/econometrics/specialized-prediction.md",
        },
        "dataset_helpers": {
            "procedures": {
                "ttest": "one-sample, paired, independent pooled/Welch, Levene and effect sizes",
                "sdtest": "one-sample/F and exact mean/median/trimmed-mean robust tests",
                "oneway": "ANOVA, homogeneity, Welch/Brown-Forsythe and native post hoc",
                "anova": "between-subject Type I/II/III factorial/ANCOVA and marginal means",
                "correlate": "Pearson and exact external Spearman/Kendall, pairwise/listwise, exact counts and Fisher intervals",
                "describe": "grouped native moments and exact external Stata/HAVERAGE percentiles",
                "rm_anova": "complete typed subject/cell validation on disk, contrast TSQR and sphericity corrections",
                "manova": "blocked multivariate QR/SSCP, native multivariate tests and Box M",
                "pcorr": "partial and semipartial through blocked TSQR",
                "pca": "correlation/covariance PCA",
                "factor": "pf/ipf/pcf/ml extraction, rotations and score coefficients",
                "factortest": "KMO, MSA/SMC and Bartlett sphericity",
                "pca_scores": "replayable lazy Dataset preserving index and missing rows",
                "factor_scores": "replayable lazy Dataset preserving index and missing rows",
            },
            "remaining": ["other multivariate and resampling procedures"],
            "exact_group_medians": "owned bounded-cache SQLite scratch storage",
            "source_integrity": "complete replay checks; lazy scores require exhausting a pass",
            "resource_scope": "bounded batches and guarded model/group/result geometry; not total process RSS",
            "document": "docs/econometrics/streaming_helpers.md",
        },
    }
