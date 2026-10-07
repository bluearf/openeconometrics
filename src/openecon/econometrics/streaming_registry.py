"""Torch-free dispatch metadata for verified native replay adapters.

An estimator appears here only after its algorithm consumes bounded passes.
Unsupported families never fall back to collecting a Dataset in RAM.
"""
from importlib import import_module

from openecon.analysis_contracts import AnalysisError

_LIKELIHOODS = ("glm", "poisson", "cloglog", "fracreg", "nbreg", "betareg", "hetprobit",
                "biprobit", "ologit", "oprobit", "mlogit", "tobit", "intreg", "truncreg",
                "heckman", "heckprobit", "ivprobit", "ivtobit", "cpoisson", "cnbreg",
                "tpoisson", "tnbreg", "zip", "zinb", "gnbreg", "hurdle", "churdle", "frontier", "streg")

_ADAPTERS = {
    "areg": "openecon.econometrics.streaming_linear:fit_streaming_linear",
    "reghdfe": "openecon.econometrics.streaming_hdfe:fit_streaming_hdfe",
    "ppmlhdfe": "openecon.econometrics.streaming_ppml:fit_streaming",
    "cnsreg": "openecon.econometrics.streaming_linear:fit_streaming_linear",
    "sureg": "openecon.econometrics.streaming_systems:fit_streaming_systems",
    "mvreg": "openecon.econometrics.streaming_systems:fit_streaming_systems",
    "reg3": "openecon.econometrics.streaming_systems:fit_streaming_systems",
    "gmm": "openecon.econometrics.streaming_gmm:fit_streaming_gmm",
    "xtreg": "openecon.econometrics.streaming_linear:fit_streaming_linear",
    "xtfmb": "openecon.econometrics.streaming_fmb:fit_streaming_fmb",
    "xtgee": "openecon.econometrics.streaming_gee:fit_streaming_gee",
    "xtgls": "openecon.econometrics.streaming_panelgls:fit_streaming_panelgls",
    "xtpcse": "openecon.econometrics.streaming_panelgls:fit_streaming_panelgls",
    "ahreg": "openecon.econometrics.streaming_dpanel:fit_streaming",
    "xtdpd": "openecon.econometrics.streaming_dynamic_gmm:fit_streaming",
    "teffects": "openecon.econometrics.streaming_teffects:fit_streaming",
    "var": "openecon.econometrics.streaming_var:fit_streaming_var",
    "vec": "openecon.econometrics.streaming_vec:fit_streaming_vec",
    "ardl": "openecon.econometrics.streaming_ardl:fit_streaming_ardl",
    "nardl": "openecon.econometrics.streaming_nardl:fit_streaming_nardl",
    "prais": "openecon.econometrics.streaming_prais:fit_streaming_prais",
    "arima": "openecon.econometrics.streaming_arima:fit_streaming_arima",
    "arch": "openecon.econometrics.streaming_arch:fit_streaming_arch",
    "ucm": "openecon.econometrics.streaming_ucm:fit_streaming_ucm",
    "mswitch": "openecon.econometrics.streaming_mswitch:fit_streaming_mswitch",
    "threshold": "openecon.econometrics.streaming_threshold:fit_streaming_threshold",
    "rdrobust": "openecon.econometrics.streaming_rd:fit_streaming",
    "didregress": "openecon.econometrics.streaming_did:fit_streaming",
    "eventstudy": "openecon.econometrics.streaming_did:fit_streaming",
    "csdid": "openecon.econometrics.streaming_csdid:fit_streaming",
    "clogit": "openecon.econometrics.streaming_conditional:fit_streaming",
    "mixed": "openecon.econometrics.streaming_mixed:fit_streaming_mixed",
    "stcox": "openecon.econometrics.streaming_cox:fit_streaming_cox",
    "nl": "openecon.econometrics.streaming_nonlinear:fit_streaming_nonlinear",
    "rreg": "openecon.econometrics.streaming_nonlinear:fit_streaming_nonlinear",
    "ivregress": "openecon.econometrics.streaming_linear:fit_streaming_linear",
    "xtivreg": "openecon.econometrics.streaming_fe_iv:fit_streaming_fe_iv",
    "ivreghdfe": "openecon.econometrics.streaming_fe_iv:fit_streaming_fe_iv",
    **{name: "openecon.econometrics.streaming_glmm:fit_streaming_glmm"
       for name in ("melogit", "meprobit", "mepoisson", "xtlogit", "xtprobit", "xtpoisson")},
    **{name: "openecon.econometrics.streaming_quantile:fit_streaming_quantile"
       for name in ("qreg", "bsqreg", "sqreg", "iqreg")},
    **{name: "openecon.econometrics.streaming_likelihood:fit_streaming" for name in _LIKELIHOODS},
}

_ALGORITHMS = {
    "areg": "SQLite group moments, global within TSQR and covariance replays",
    "reghdfe": "Disk row vectors, SQLite group projections, global conjugate gradients and exact singleton/connected-component degrees of freedom",
    "ppmlhdfe": "Full-source native IRLS, changing-weight disk FE projections, global deviance and actual-row covariance",
    "cnsreg": "Global TSQR with exact affine constraints and covariance replays",
    **{name: "Global joint TSQR; native system kernels on compressed factors"
       for name in ("sureg", "mvreg", "reg3")},
    "gmm": "Global analytic moment/Jacobian passes and LM; disk cluster/HAC scores",
    "xtreg": "SQLite panel moments; FE/BE/RE/FD/pooled/MLE global factors, exact variance components and row/period score replays",
    "xtfmb": "Disk-resident period TSQR, global coefficient covariance and exact period-distance HAC",
    "xtgee": "Global native GEE equations, disk panel sums, ordered AR1 state and preplanned patterned correlation factors",
    "xtgls": "Disk-ordered panel Prais state, global GLS factors and preplanned true cross-panel covariance",
    "xtpcse": "Disk ordered panel fits, exact period overlap moments and preplanned full PCSE covariance",
    "ahreg": "Exact disk panel lag windows, global first-difference IV TSQR and full panel/MA(1) inference",
    "xtdpd": "Disk-ordered guarded complete panels, global native difference/system GMM, Windmeijer, AR and Sargan/Hansen diagnostics",
    "teffects": "Global nuisance likelihoods and stacked equations; exact disk-indexed/tiled matching and Abadie-Imbens inference",
    "var": "Disk time ordering, cross-batch lags, global system TSQR and residual diagnostic passes",
    "vec": "Disk time ordering, global Johansen joint TSQR, reduced-rank fit and full residual diagnostics",
    "ardl": "Disk time ordering, one global maximum-lag TSQR, bounded exhaustive lag-grid search and actual-row HC covariance",
    "nardl": "Global ordered positive/negative partial sums, bounded constrained ARDL lag search and native symmetry inference",
    "prais": "One global adjacent-period joint QR, exact small-factor rho iterations and actual transformed-row HC covariance",
    "arima": "Stateful native polynomial filters, exact stationary presample correction, global analytic ARMA likelihood and full innovation-score covariance",
    "arch": "Compiled float64 finite variance/tangent state, full presample moments and likelihood, disk EGARCH sign patterns and native active-set Newton inference",
    "ucm": "Full-source exact diffuse Kalman likelihood, bounded compiled states and disk fixed-interval smoothing",
    "mswitch": "Stateful native Hamilton likelihood, full ergodic Fisher scores and disk Kim smoothing of expanded AR histories",
    "threshold": "All observed candidates screened on disk, global TSQR refinement and exact native full-sample bootstrap draw order",
    "rdrobust": "Disk-sorted tied running values, exact NN halos, full local polynomial TSQR and native bandwidth/bias inference",
    "didregress": "Disk group/adoption metadata, full-sample HDFE and native restricted-reference diagnostics",
    "eventstudy": "Disk group/adoption metadata, exact event periods and full-sample HDFE with pretrend inference",
    "csdid": "Disk paired cohort/period nuisance models, global ATT influence TSQR and estimated cohort-share aggregation covariance",
    "clogit": "Disk complete informative groups, exact conditional likelihood and original-unit one/two-way row score covariance",
    "mixed": "Disk group joint TSQR, one/two nested Gaussian levels with random slopes, exact profiled ML/REML and joint parameter sandwich",
    "stcox": "Disk global risk sets, Breslow/Efron sweeps, exactp symmetric recursion and event-dependent TVC with real-record covariance",
    "nl": "Global analytic SSR, Jacobian, GN/LM iterations and covariance replays",
    "rreg": "Global Cook screen, exact disk median/MAD, Huber/biweight IRLS and pseudovalue covariance",
    "ivregress": "Global joint TSQR and IV projection; real-row covariance and diagnostic score replays",
    "xtivreg": "Disk panel FE/BE projection, first differences or native G2SLS/EC2SLS variance components; global IV TSQR and actual-row scores",
    "ivreghdfe": "Disk multi-way FE projection, global 2SLS/LIML/GMM and actual-row covariance/identification diagnostics",
    **{name: "Disk group quadrature/mode state, bounded node blocks, global native integrated likelihood and panel covariance"
       for name in ("melogit", "meprobit", "mepoisson", "xtlogit", "xtprobit", "xtpoisson")},
    **{name: "Disk primal-dual Frisch-Newton; exact LP vertex certification and joint global bootstrap"
       for name in ("qreg", "bsqreg", "sqreg", "iqreg")},
    **{name: "Global native analytic likelihood, gradient/Hessian and score-covariance replays" for name in _LIKELIHOODS},
}


def estimators():
    return tuple(_ADAPTERS)


def algorithms():
    return dict(_ALGORITHMS)


def conditions():
    return {"xtreg": {"model": ["fe", "be", "re", "fd", "pooled", "mle"],
                      "covariance": ["nonrobust", "robust", "cluster", "driscoll_kraay"],
                      "driscoll_kraay": True, "driscoll_kraay_models": ["fe", "pooled"]},
            "xtdpd": {"complete_panel_budget": True, "difference_system": True,
                      "transformation": ["first_difference", "forward_orthogonal_deviation"],
                      "steps": [1, 2], "windmeijer": True, "maximum_instrument_candidates": 2000},
            "teffects": {"method": ["ra", "ipw", "ipwra", "aipw", "nnmatch", "psmatch"], "cluster_dimensions": 1,
                        "resident_only_options": ["tlevel", "ematch", "matching_vce=iid"],
                         "matching": "exact global neighbours including ties; disk-indexed propensity scores, bounded covariate tiles and native work budget",
                         "overlap": "global minimum, maximum and weighted mean; matching also retains exact disk quantiles"},
            "ivregress": {"method": ["2sls", "liml", "gmm", "fuller", "kclass"], "hac": True,
                          "hac_kernels": ["bartlett", "truncated", "parzen", "quadratic_spectral"],
                          "centered_gmm": True, "iterated_gmm": True, "cluster_dimensions": 2},
            "xtivreg": {"model": ["fe", "be", "fd", "re"], "ec2sls": True, "ec2sls_models": ["re"], "cluster_dimensions": 1},
            "ivreghdfe": {"method": ["2sls", "liml", "gmm"], "cluster_dimensions": 2},
            "glmm": {"normal_random_effect_dimensions": [1, 2], "integration": ["ghermite", "mcaghermite", "mvaghermite"],
                     "group_sandwich": True, "population_averaged": ["independent", "exchangeable", "ar1", "stationary", "nonstationary", "unstructured"]},
            "xtgee": {"corr": ["independent", "exchangeable", "ar1", "stationary", "nonstationary", "unstructured"], "panel_covariance": True,
                      "patterned_correlation_budget": True},
            "mixed": {"levels": 2, "nested_levels": True, "random_slopes": True, "random_intercept": True,
                      "random_intercept_optional": True, "top_nested_level_requires_intercept": True,
                      "covariance_structure": ["independent", "identity", "exchangeable", "unstructured"],
                      "method": ["ml", "reml"],
                      "covariance_by_method": {"ml": ["nonrobust", "robust", "cluster"],
                                               "reml": ["nonrobust"]}},
            "xtlogit": {"model": ["re", "fe", "pa"],
                        "covariance_by_model": {"re": ["nonrobust", "robust", "cluster"],
                                                "fe": ["nonrobust"],
                                                "pa": ["nonrobust", "robust"]},
                        "corr_by_model": {"pa": ["independent", "exchangeable", "ar1", "stationary", "nonstationary", "unstructured"]},
                        "corr_options_apply_to": "pa", "quadrature_options_apply_to": "re"},
            "xtprobit": {"model": ["re", "pa"], "fixed_effects": False,
                         "covariance_by_model": {"re": ["nonrobust", "robust", "cluster"],
                                                 "pa": ["nonrobust", "robust"]},
                         "corr_by_model": {"pa": ["independent", "exchangeable", "ar1", "stationary", "nonstationary", "unstructured"]},
                         "corr_options_apply_to": "pa", "quadrature_options_apply_to": "re"},
            "xtpoisson": {"model": ["re", "fe", "pa"],
                          "covariance_by_model": {"re": ["nonrobust", "robust", "cluster"],
                                                  "fe": ["nonrobust", "robust"],
                                                  "pa": ["nonrobust", "robust"]},
                          "corr_by_model": {"pa": ["independent", "exchangeable", "ar1", "stationary", "nonstationary", "unstructured"]},
                          "corr_options_apply_to": "pa", "quadrature_options_apply_to": "re"},
            "clogit": {"cluster_dimensions": 2, "whole_group_budget": True},
            "gmm": {"cluster_dimensions": 1,
                    "hac_kernels": ["bartlett", "truncated", "parzen", "quadratic_spectral"],
                    "quadratic_spectral_hac": True,
                    "quadratic_spectral_work": "exact bounded disk pair tiles; explicit all-lag work budget"},
            "nl": {"cluster_dimensions": 1},
            "quantile": {"cluster_dimensions": 1, "approximate_loss": False},
            "likelihoods": {"cluster_dimensions": 2,
                            "model_test": "full-sample restricted LR/null, pseudo R2 and ancillary references where defined; explicit unavailable reference and Wald fallback on failed secondary fits"},
            "glm": {"optimizer": ["ml", "irls"], "irls_information": "expected Fisher information"},
            "heckman": {"method": ["ml", "twostep"], "twostep_covariance": ["nonrobust"], "twostep_weights": ["unweighted", "fweight"]},
            "ivprobit": {"method": ["ml", "twostep"], "twostep_covariance": ["nonrobust"], "twostep_weights": ["unweighted", "fweight"]},
            "ivtobit": {"method": ["ml", "twostep"], "limits": "explicit scalar or global sample extrema",
                        "twostep_covariance": ["nonrobust"], "twostep_weights": ["unweighted", "fweight"]},
            "tobit": {"limits": "explicit scalar or global sample extrema"}, "truncreg": {"limits": "explicit scalar"},
            "streg": {"independent_records": True, "id": True, "subject_overlap_validation": "disk-sorted across all batches", "cluster_dimensions": 1},
            "stcox": {"ties": ["breslow", "efron", "exactp"], "tvc": True, "cluster_dimensions": 1,
                      "exactp": "unweighted, no TVC, nonrobust; bounded symmetric state and explicit computational-work budget",
                      "tvc_details": "identity/log event-dependent native design; bounded disk intervals and explicit computational-work budget"}}


def supports_spec(spec):
    """Used only to choose automatic replay for an already resident table.

    A Dataset always reaches the adapter, which reports its exact option
    error. Unsupported resident-table options keep the guarded dense path.
    """
    if spec.estimator not in _ADAPTERS:
        return False
    if spec.estimator == "xtreg":
        return spec.options.get("model", "fe") in {"fe", "be", "re", "fd", "pooled", "mle"}
    if spec.estimator == "teffects":
        from . import registry
        if (spec.options.get("tlevel") is not None or registry.role_columns(spec, "ematch") or
                spec.options.get("matching_vce", "robust") != "robust"):
            return False
        if spec.options.get("method", "ra") in {"nnmatch", "psmatch"} and spec.weights is not None:
            return False
        # Category count is discovered from data; all resident ATET requests keep
        # the native path, which supports the new multivalued target contract.
        if spec.options.get("estimand", "ate") == "atet":
            return False
        return spec.options.get("method", "ra") in {"ra", "ipw", "ipwra", "aipw", "nnmatch", "psmatch"}
    if spec.estimator == "csdid":
        defaults = {"sample": "balanced", "anticipation": 0, "bootstrap_reps": 0, "seed": None, "uniform": False}
        return spec.panel is not None and all(spec.options.get(name, default) == default for name, default in defaults.items())
    if spec.estimator == "stcox":
        return spec.options.get("ties", "breslow") in {"breslow", "efron", "exactp"}
    if spec.estimator == "ivregress":
        return spec.options.get("method", "2sls") in {"2sls", "liml", "gmm", "fuller", "kclass"}
    if spec.estimator == "xtivreg":
        return spec.options.get("model", "fe") in {"fe", "be", "fd", "re"}
    if spec.estimator == "ivreghdfe":
        return spec.options.get("method", "2sls") in {"2sls", "liml", "gmm"}
    if spec.estimator in {"xtlogit", "xtprobit", "xtpoisson"}:
        return spec.options.get("model", "re") != "pa" or spec.options.get("corr", "exchangeable") in {"independent", "exchangeable", "ar1", "stationary", "nonstationary", "unstructured"}
    if spec.estimator == "xtgee":
        return spec.options.get("corr", "exchangeable") in {"independent", "exchangeable", "ar1", "stationary", "nonstationary", "unstructured"}
    if spec.estimator == "mixed":
        from . import registry
        return len(registry.role_columns(spec, "group")) in {1, 2}
    if spec.estimator in _LIKELIHOODS:
        if spec.estimator in {"tobit", "truncreg", "ivtobit"}:
            if spec.estimator == "truncreg" and (spec.options.get("ll_at_min") or spec.options.get("ul_at_max")):
                return False
            if any(isinstance(spec.options.get(key), str) for key in ("ll", "ul")):
                return False
    if spec.estimator == "gmm":
        from . import registry
        if len(registry.cluster_columns(spec)) > 1:
            return False
    return True


def fit(spec, source):
    target = _ADAPTERS.get(spec.estimator)
    if spec.estimator == "xtreg" and spec.options.get("model", "fe") == "re":
        target = "openecon.econometrics.streaming_re:fit_streaming_re"
    if spec.estimator == "xtivreg" and spec.options.get("model", "fe") == "fd":
        target = "openecon.econometrics.streaming_fd_iv:fit_streaming_fd_iv"
    if spec.estimator == "xtivreg" and spec.options.get("model", "fe") == "re":
        target = "openecon.econometrics.streaming_re_iv:fit_streaming_re_iv"
    if spec.estimator in {"xtlogit", "xtprobit", "xtpoisson"} and spec.options.get("model", "re") == "pa":
        target = "openecon.econometrics.streaming_gee:fit_pa"
    if target is None:
        raise AnalysisError("streaming_unsupported", f"{spec.estimator} has no bounded Dataset kernel yet. "
                            "No full table is collected automatically; use a supported replay estimator or a table that fits the planned workspace.")
    module, name = target.split(":")
    from .core import kernel_call
    from openecon.engines.execution import execution_scope
    import torch
    with torch.no_grad(), torch.device("cpu"), execution_scope("auto") as trace:
        result = kernel_call(getattr(import_module(module), name), spec, source)
    execution = trace.metadata()
    result.provenance.update({"execution": execution, "device": execution["device"]})
    return result
