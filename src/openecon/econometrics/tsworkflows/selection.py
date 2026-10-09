"""Exhaustive selection over the bounded native additive innovations ETS family."""

from __future__ import annotations

from copy import deepcopy
import math

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame, _frame_hasher, make_spec
from openecon.econometrics.resident_cpu import resident_cpu
from openecon.econometrics.tsworkflows.common import ordered
from openecon.econometrics.tsworkflows.ets import ets

MODELS = ("ANN", "AAN", "AdN", "ANA", "AAA", "AdA")


@resident_cpu
def auto_ets(*, data, y, models=None, period=2, time=None, criterion="aicc",
             candidate_options=None, max_candidates=6, max_work=240000000,
             missing="raise", max_iterations=300, tolerance=1e-7, alpha=0.05):
    """Select native additive ETS by AIC/AICc/BIC on the same complete response sample.

    Every smoothing parameter, estimated initial-state parameter and sigma counts
    in K. Optional per-model fixed/initial choices are retained. All failed fits
    remain in the search history. The result keeps the chosen ordinary ETS spec;
    refitting that spec alone does not repeat selection. Use rolling(selection=...)
    to run this search anew inside each training origin.
    """
    from openecon.dataset import Dataset

    if isinstance(data, Dataset):
        raise AnalysisError("streaming_unsupported", "ETS selection requires a resident training series; no Dataset collection is performed.")
    names = list(MODELS) if models is None else models
    if not isinstance(names, (list, tuple)) or not 1 <= len(names) <= 6 or any(name not in MODELS for name in names) or len(set(names)) != len(names):
        raise AnalysisError("invalid_option", "Choose 1..6 distinct native additive ETS model codes.")
    names = list(names)
    if not isinstance(criterion, str) or criterion not in {"aic", "aicc", "bic"}:
        raise AnalysisError("invalid_option", "ETS selection criterion must be aic/aicc/bic.")
    if isinstance(period, bool) or not isinstance(period, int) or not 2 <= period <= 48:
        raise AnalysisError("invalid_option", "ETS period must be an integer in 2..48.")
    for value, name, maximum in ((max_candidates, "max_candidates", 6), (max_work, "max_work", 1000000000)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise AnalysisError("invalid_option", f"{name} must be an integer in 1..{maximum}.")
    if len(names) > max_candidates:
        raise AnalysisError("search_budget", "Declared ETS candidates exceed max_candidates before fitting.")
    options = {} if candidate_options is None else candidate_options
    if not isinstance(options, dict) or set(options)-set(names) or any(
        not isinstance(record, dict) or set(record)-{"fixed", "initial"}
        for record in options.values()
    ):
        raise AnalysisError("invalid_option", "candidate_options may declare fixed/initial choices only for requested ETS models.")
    base = make_spec("ets", outcome=y, predictors=[], intercept=False, time=time,
                     missing=missing, alpha=alpha, options={"model": names[0], "period": period,
                        "max_iterations": max_iterations, "tolerance": tolerance})
    common = ModelFrame(base, data)
    ordered(common)
    if common.n > 20000:
        raise AnalysisError("system_budget", "ETS selection admits at most 20000 resident response periods.")
    work = 0
    for name in names:
        trend, seasonal, damped = name not in {"ANN", "ANA"}, name.endswith("A"), "d" in name
        settings = options.get(name, {})
        # Conservative full information geometry, including the nonredundant
        # zero-sum seasonal initial state and sigma, even for invalid candidates.
        k = 1 + int(trend) + int(seasonal) + int(damped) + 1
        if settings.get("initial") is None:
            k += 1 + int(trend) + (period-1 if seasonal else 0)
        m = 1 + int(trend) + (period if seasonal else 0)
        work += common.n*m*k*k
    if work > max_work:
        raise AnalysisError("search_budget", f"ETS candidate information geometry needs {work} work units, exceeding max_work={max_work}.")
    common.workspace_plan("ETS retained response and complete candidate histories", {
        "response_and_origin_sample": common.n*64,
        "candidate_records_and_full_covariances": len(names)*(65536 + 48**2*64),
    })
    records, best, best_score = [], None, math.inf
    for name in names:
        row = {"model": name, "status": "failed", "settings": deepcopy(options.get(name, {}))}
        try:
            result = ets(data=common.original, y=y, time=time, model=name, period=period,
                         missing=missing, alpha=alpha, max_iterations=max_iterations,
                         tolerance=tolerance, **options.get(name, {}))
            if result.sample_positions != common.positions or result.nobs != common.n:
                raise AnalysisError("incomparable_sample", "Every ETS candidate must use the same ordered retained response positions.")
            k, n = len(result.coefficients), result.nobs
            ll = float(result.metrics["log_likelihood"])
            aic, bic = -2*ll + 2*k, -2*ll + math.log(n)*k
            aicc = aic + 2*k*(k+1)/(n-k-1) if n > k+1 else None
            score = {"aic": aic, "aicc": aicc, "bic": bic}[criterion]
            if score is None:
                raise AnalysisError("insufficient_aicc_sample", "ETS AICc is undefined when N <= K+1.")
            if not all(math.isfinite(value) for value in (ll, aic, bic, score)):
                raise AnalysisError("non_finite_criterion", "ETS likelihood/selection criteria must be finite.")
            row.update(status="ok", log_likelihood=ll, aic=aic, aicc=aicc, bic=bic,
                       criterion=float(score), nobs=n, parameters=k,
                       parameter_order=[c.term for c in result.coefficients],
                       initial_estimated=result.extra["initial_estimated"],
                       coefficients=[c.model_dump(mode="json") for c in result.coefficients],
                       covariance_matrix=result.covariance_matrix,
                       spec=result.spec.model_dump(mode="json"),
                       convergence=result.provenance.get("optimizer"))
            if float(score) < best_score:
                best, best_score = result, float(score)
        except AnalysisError as exc:
            if exc.code == "incomparable_sample":
                raise
            row.update(error_code=exc.code, error=str(exc))
        records.append(row)
    if best is None:
        error = AnalysisError("no_valid_model", "All declared additive ETS candidates failed; inspect error.candidates for the complete history.")
        error.candidates = records
        raise error
    best.extra["auto_selection"] = {
        "schema": 1, "family": "ets", "search": "exhaustive declared native additive innovations ETS candidates",
        "models": names, "period": period, "criterion": criterion,
        "criterion_value": best_score, "selected_model": best.extra["model"],
        "candidate_count": len(records), "candidates": records,
        "common_input_positions": common.positions,
        "common_response_positions": common.positions,
        "common_data_hash": _frame_hasher(common.original).hexdigest(),
        "parameter_count": "every free smoothing parameter, independent estimated initial state, and sigma",
        "tie_rule": "first declared candidate at exactly equal criterion",
        "work_units": work, "max_work": max_work,
        "work_contract": "sum of candidate N*state_width*K_upper_bound^2 information geometries; optimizer iteration operations are separately limited by max_iterations and are not included in work_units",
        "selection_uncertainty": "inference and innovation forecast intervals condition on the selected model; model-selection uncertainty excluded",
        "forecast_parameter_uncertainty": False,
        "refit": "the selected underlying ETS ModelSpec is retained; rolling(selection=...) repeats the search inside each origin",
    }
    best.provenance["workflow"] = "auto_ets; selected underlying native ETS specification retained"
    best.inference["model_selection_uncertainty"] = False
    best.warnings.append("Inference and conditional innovation forecasts exclude model-selection uncertainty; ETS forecast intervals also exclude estimated-parameter uncertainty.")
    return best
