"""Parametric survival regression: Stata's ``streg``.

Full maximum likelihood of the models in ``parametric`` (exponential, Weibull,
Gompertz, lognormal, loglogistic, generalized gamma) with right censoring and
delayed entry (left truncation) in the likelihood, by Newton-Raphson with
analytic score and Hessian (numerical per-observation derivatives for the
generalized gamma).

Metric: exponential and Weibull are fitted in the proportional-hazards metric
by default (``metric='aft'`` is Stata's ``time`` option); Gompertz is PH only;
lognormal, loglogistic and generalized gamma are AFT only. ``ancillary``
columns model the (first) ancillary parameter, ``a = z'g`` (Stata's
``ancillary()``); without them it is the single term ``/ln_p``, ``/gamma``,
``/lnsigma`` or ``/lngamma`` (and ``/kappa`` for the generalized gamma).
``strata`` (one column) adds stratum indicators to the main and the ancillary
equation, which is Stata's definition of ``strata()`` for streg.

Covariance: ``nonrobust`` (observed information), ``opg``, ``robust``
(N/(N-1) sandwich; with an ``id``, clustered on it as streg does) and
``cluster`` (G/(G-1)), all through ``core.ml_covariance``.

Log likelihood (Stata's scale): the likelihood is maximized on the time scale
(``parametric``), but the reported log likelihood is that of the log survival
time, ``ln L_Stata = ln L_time + sum_i w_i delta_i ln t_i``: the density of
``ln T`` is ``t f(t)``. Estimates and every likelihood-ratio statistic are
unaffected (the term does not depend on the parameters); ``log_likelihood``,
``aic``, ``bic`` and the null log likelihood are those Stata prints (verified on
the ``kva`` example of [ST] streg: ll = 5.6934189 for the Weibull model).

Model test (Stata's ``ml`` convention, verified on [ST] streg examples 8-9):
``tests['model']`` tests every main-equation coefficient except the constant -
including the stratum indicators of ``strata()`` - against the model with a
constant-only main equation and the ancillary equation as specified (an
``ancillary()`` / ``strata()`` term of the ancillary equation stays in the null
model). LR chi2 under ``nonrobust``, Wald chi2 otherwise.

Regressors of both equations are centred at their (weighted) means for the
iteration and mapped back exactly (``discrete.common``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis import fit
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    Design, ModelFrame, build_result, column_list, information_criteria, kernel_call,
)
from openecon.econometrics.discrete.common import (
    build_spec, centre_regressors, check_pweights, constant_shift, likelihood_covariance,
    likelihood_weights, maximize, model_test, optimizer_record, resolve_covariance,
)
from openecon.econometrics.survival.cox import _survival_columns
from openecon.econometrics.survival.data import (
    FLOAT, check_overlap, check_times, end_before_entry_note, safe_exp,
)
from openecon.econometrics.survival.parametric import (
    ANCILLARY, ParametricObjective, SurvivalData,
)
from openecon.engines.inference import critical_value
from openecon.models import ModelSpec, ResultBundle

_PH_ONLY = {"gompertz"}
_AFT_ONLY = {"lognormal", "loglogistic", "ggamma"}
_TITLES = {"exponential": "Exponential", "weibull": "Weibull", "gompertz": "Gompertz",
           "lognormal": "Lognormal", "loglogistic": "Loglogistic",
           "ggamma": "Generalized gamma"}


def resolve_metric(dist: str, metric: str | None) -> str:
    if metric is None:
        return "aft" if dist in _AFT_ONLY else "ph"
    if metric == "aft" and dist in _PH_ONLY:
        raise AnalysisError("invalid_spec", "The Gompertz model is parameterized only as a "
                            "proportional-hazards model (metric='ph').")
    if metric == "ph" and dist in _AFT_ONLY:
        raise AnalysisError("invalid_spec", f"The {dist} model is implemented only in the "
                            "accelerated failure-time metric (metric='aft').")
    return metric


def _strata_block(frame: ModelFrame, prefix: str) -> Design | None:
    names = frame.role("strata")
    if not names:
        return None
    return frame.design(names, intercept=False, categorical=names, prefix=prefix)


def _equation(frame: ModelFrame, predictors: Sequence[str], strata: Design | None,
              weights: Tensor | None, prefix: str = "") -> Design:
    design = frame.design(list(predictors), intercept=True, prefix=prefix)
    if strata is not None:
        block = _strata_block(frame, prefix)
        design = Design(torch.cat([design.x, block.x], dim=1), design.terms + block.terms,
                        {**design.categories, **block.categories}, True)
    return frame.drop_collinear(design, weights)


def _start(dist: str, metric: str, data: SurvivalData, w: Tensor) -> tuple[float, float, float]:
    """Constant-only starting values: main constant, ancillary constant, kappa."""
    events = float((w * data.delta).sum())
    exposure = float((w * (data.t - data.t0)).sum())
    if events <= 0:
        raise AnalysisError("no_failures", "streg: the sample has no failures.")
    rate = math.log(events / exposure)
    if dist in {"exponential", "weibull", "gompertz"}:
        return (rate if metric == "ph" else -rate), 0.0, 0.0
    mean = float((w * data.log_t).sum() / w.sum())
    spread = math.sqrt(max(float((w * (data.log_t - mean) ** 2).sum() / w.sum()), 1e-4))
    if dist == "loglogistic":
        spread *= math.sqrt(3) / math.pi
    return mean, math.log(spread), 0.3


# A non-converged run whose index moves by more than DIVERGED (exp(-10) ~ 5e-5 in hazard
# or time ratio) across one standard deviation of a regressor is a monotone likelihood.
DIVERGED = 10.0
# |kappa| of a non-converged generalized gamma beyond which kappa is running off.
KAPPA_DIVERGED = 5.0


def _fit(objective: ParametricObjective, start: Tensor, scale: float, command: str):
    result = maximize(objective, start, what=command, scale=scale)
    if not result.converged:
        theta = result.theta
        spread = objective.x.std(0) if objective.x.shape[0] > 1 else torch.zeros(objective.k)
        if bool(((theta[:objective.k] * spread).abs() > DIVERGED).any()):
            raise AnalysisError(
                "separation_detected", f"{command}: the likelihood keeps increasing as a "
                "main-equation coefficient diverges (a covariate, category or stratum without "
                "failures, or one that separates failures from survivors). Drop or recode it.")
        if objective.q and bool((theta[objective.k:objective.k + objective.q].abs() > 30).any()):
            raise AnalysisError("boundary_solution", f"{command}: the ancillary parameter runs "
                                "to the boundary of its space (the distribution does not fit "
                                "these data); try another distribution.")
        if objective.extra and abs(float(theta[-1])) > KAPPA_DIVERGED:
            raise AnalysisError("boundary_solution", f"{command}: the generalized gamma shape "
                                "kappa diverges (the likelihood has no interior maximum for "
                                "these data); fit a two-parameter distribution (weibull, "
                                "lognormal, loglogistic) instead.")
        raise AnalysisError("nonconvergence", f"{command} did not converge: "
                            f"{result.diagnostics.get('message')} Check the scale of the "
                            "covariates and whether the distribution suits the data.")
    return result


def fit_streg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``streg``: ``(spec, data) -> ResultBundle`` (see the module notes)."""
    command = "streg"
    frame = ModelFrame(spec, data)
    dist = frame.option("distribution")
    metric = resolve_metric(dist, frame.option("metric"))
    anc_names = frame.role("ancillary")
    if dist == "exponential" and anc_names:
        raise AnalysisError("invalid_spec", "The exponential model has no ancillary parameter.")
    check_pweights(frame)
    t, t0, delta = _survival_columns(frame)
    usable = check_times(t, t0, spec.outcome, (frame.role("entry") or [None])[0])
    if not bool(usable.all()):
        frame.restrict(usable, end_before_entry_note(int((~usable).sum())))
        t, t0, delta = _survival_columns(frame)
    ids = None
    if frame.role("id"):
        ids = frame.codes(frame.role("id")[0])
        check_overlap(t, t0 if t0 is not None else torch.zeros_like(t), ids[0])
    weights = likelihood_weights(frame)
    screen = weights.for_screen()
    strata = _strata_block(frame, "")
    main = _equation(frame, spec.predictors, strata, screen)
    anc = ANCILLARY.get(dist)
    ancillary = None if anc is None else _equation(frame, anc_names, strata, screen,
                                                   prefix=f"{anc}:")
    main_means = centre_regressors(main, weights)
    anc_means = None if ancillary is None else centre_regressors(ancillary, weights)
    offset_names = frame.role("offset")
    offset = frame.numeric(offset_names[0]) if offset_names else None
    survival = SurvivalData(t, t0, delta)
    w = weights.user
    k, q = len(main.terms), 0 if ancillary is None else len(ancillary.terms)
    size = k + q + (1 if dist == "ggamma" else 0)
    if weights.nobs <= size:
        raise AnalysisError("insufficient_observations", f"{command} needs more observations "
                            f"({weights.nobs}) than parameters ({size}).")
    main_const, anc_const, kappa = _start(dist, metric, survival, w)
    # Null model (Stata's constant-only model): constant-only main equation, ancillary
    # equation as specified (stratum indicators of the ancillary equation included).
    n_strata = 0 if strata is None else sum(term in main.terms for term in strata.terms)
    null_columns = [0]
    null_x = main.x[:, null_columns]
    null_obj = ParametricObjective(dist, metric, null_x, None if ancillary is None
                                   else ancillary.x, survival, w, offset)
    start = torch.zeros(null_obj.size, dtype=FLOAT)
    start[0] = main_const
    if ancillary is not None:
        start[len(null_columns)] = anc_const
    if dist == "ggamma":
        start[-1] = kappa
    null = _fit(null_obj, start, weights.scale, f"{command} (constant-only model)")
    objective = ParametricObjective(dist, metric, main.x, None if ancillary is None
                                    else ancillary.x, survival, w, offset)
    full_start = torch.zeros(size, dtype=FLOAT)
    full_start[null_columns] = null.theta[:len(null_columns)]
    full_start[k:] = null.theta[len(null_columns):]
    result = _fit(objective, full_start, weights.scale, command)
    theta = result.theta
    covariance, info = likelihood_covariance(
        frame, result.hessian, lambda: kernel_call(objective.score_rows, theta), weights,
        **_robust_overrides(frame, ids))
    if spec.covariance == "robust" and ids is not None:
        info.update({"covariance": "robust",
                     "correction": "sandwich clustered on the id variable: G/(G-1)"})
    # Undo the centring: a = a_c - m'b in each equation (block diagonal Jacobian).
    jacobian = torch.eye(size, dtype=FLOAT)
    jacobian[:k, :k] = constant_shift(main_means)
    if ancillary is not None:
        jacobian[k:k + q, k:k + q] = constant_shift(anc_means)
    params, covariance = jacobian @ theta, jacobian @ covariance @ jacobian.T
    terms = list(main.terms)
    equations: list[str | None] = [spec.outcome] * k
    if ancillary is not None:
        if anc_names:
            terms += ancillary.terms
            equations += [anc] * q
        elif q == 1:
            terms.append(f"/{anc}")
            equations.append(None)
        else:   # stratum indicators only
            terms += ancillary.terms
            equations += [anc] * q
    if dist == "ggamma":
        terms.append("/kappa")
        equations.append(None)
    # Every main-equation term except the constant is tested (stratum indicators too).
    slopes = [i for i, term in enumerate(main.terms) if term != "Intercept"]
    # Stata reports the log likelihood of ln t: add sum w delta ln t (see the module notes).
    log_time = float((w * delta * survival.log_t).sum())
    log_likelihood = result.value + log_time
    null_log_likelihood = null.value + log_time
    criteria = information_criteria(log_likelihood, size, weights.nobs)
    frequency = weights.frequency
    events = delta if frequency is None else delta * frequency
    exposure = (t - (0 if t0 is None else t0))
    exposure = exposure if frequency is None else exposure * frequency
    tests = {"model": model_test(frame, params, covariance, slopes, log_likelihood,
                                 null_log_likelihood,
                                 null_model="the constant-only model",
                                 what="the main-equation covariates")}
    extra = _extra(dist, metric, terms, params, covariance, k, slopes, anc, anc_names, spec.alpha)
    extra.update({"null_log_likelihood": null_log_likelihood, "strata_terms": n_strata,
                  "log_likelihood_scale": "log time (Stata): time-scale ll + sum w d ln t",
                  "derivatives": "numerical in kappa (five-point central differences), "
                                 "analytic in the indices" if dist == "ggamma"
                  else "analytic"})
    metrics = {"log_likelihood": log_likelihood, "aic": criteria["aic"], "bic": criteria["bic"],
               "df_model": len(slopes), "n_subjects": _subjects(frame, ids, frequency),
               "n_failures": float(events.sum()), "time_at_risk": float(exposure.sum())}
    title = f"{_TITLES[dist]} {'PH' if metric == 'ph' else 'AFT'} regression"
    provenance = {"distribution": dist, "metric": metric,
                  "derivatives": extra["derivatives"], "bic_n": "number of observations",
                  "log_likelihood_scale": "log survival time (Stata streg)"}
    return build_result(
        frame, terms=terms, params=params, covariance=covariance, use_t=False,
        equations=equations, metrics=metrics, nobs=weights.nobs, inference=info, tests=tests,
        extra=extra, title=title, categories={**main.categories}, provenance=provenance,
        solver="newton_observed_hessian",
        solver_diagnostics={"converged": True, "iterations": result.iterations},
        optimizer=optimizer_record(result))


def _robust_overrides(frame: ModelFrame, ids) -> dict[str, Any]:
    if frame.spec.covariance == "robust" and ids is not None:
        return {"kind": "cluster", "clusters": [ids], "cluster_names": frame.role("id")}
    return {}


def _subjects(frame: ModelFrame, ids, frequency: Tensor | None) -> float:
    if ids is None:
        return float(frequency.sum()) if frequency is not None else float(frame.n)
    codes, count = ids
    if frequency is None:
        return float(count)
    replicas = torch.zeros(count, dtype=FLOAT).scatter_reduce(0, codes, frequency, "amax",
                                                              include_self=False)
    return float(replicas.sum())


def _transform(value: float, se: float, z: float, function) -> dict[str, float]:
    """exp-type transform of a parameter with delta-method SE and transformed CI."""
    point, slope = function(value)
    return {"estimate": point, "std_error": abs(slope) * se,
            "ci_low": min(function(value - z * se)[0], function(value + z * se)[0]),
            "ci_high": max(function(value - z * se)[0], function(value + z * se)[0])}


def _extra(dist: str, metric: str, terms: list[str], params: Tensor, covariance: Tensor,
           k: int, slopes: list[int], anc: str | None, anc_names: list[str],
           alpha: float) -> dict[str, Any]:
    z = critical_value(alpha, None)
    se = covariance.diagonal().sqrt()
    ratio = "hazard_ratios" if metric == "ph" else "time_ratios"
    extra: dict[str, Any] = {"distribution": dist, "metric": metric, "ancillary": anc}

    def exp_(v: float) -> tuple[float, float]:
        return safe_exp(v), safe_exp(v)

    extra[ratio] = {terms[i]: {"ratio": safe_exp(float(params[i])),
                               "std_error": safe_exp(float(params[i])) * float(se[i]),
                               "ci_low": safe_exp(float(params[i] - z * se[i])),
                               "ci_high": safe_exp(float(params[i] + z * se[i]))}
                    for i in slopes}
    if anc is None or anc_names or f"/{anc}" not in terms:
        return extra
    j = terms.index(f"/{anc}")
    value, s = float(params[j]), float(se[j])
    if dist == "weibull":
        extra["p"] = _transform(value, s, z, exp_)
        extra["1/p"] = _transform(value, s, z, lambda v: (safe_exp(-v), -safe_exp(-v)))
    elif dist in {"lognormal", "ggamma"}:
        extra["sigma"] = _transform(value, s, z, exp_)
    elif dist == "loglogistic":
        extra["gamma"] = _transform(value, s, z, exp_)
    return extra


def streg(*, data: Any, time: str, x: Sequence[str] | None = None, failure: str | None = None,
          entry: str | None = None, id: str | None = None, distribution: str = "weibull",
          metric: str | None = None, ancillary: Sequence[str] | None = None,
          strata: str | None = None, covariance: str | None = None, cluster: str | None = None,
          weights: str | None = None, weight_type: str | None = None, offset: str | None = None,
          categorical: Sequence[str] | None = None, missing: str = "raise",
          alpha: float = 0.05) -> ResultBundle:
    """Parametric survival regression (Stata ``streg``).

    Model
        Survivor functions (``mu = x'b + offset``, ancillary ``a = z'g``):
        exponential PH ``exp(-e^mu t)`` / AFT ``exp(-e^-mu t)``; Weibull PH
        ``exp(-e^mu t^p)`` / AFT ``exp(-(e^-mu t)^p)`` with ``p = exp(/ln_p)``; Gompertz
        (PH) ``exp(-e^mu (e^(gamma t) - 1)/gamma)``; lognormal (AFT)
        ``1 - Phi((ln t - mu)/sigma)``, ``sigma = exp(/lnsigma)``; loglogistic (AFT)
        ``1/(1 + exp((ln t - mu)/gamma))``, ``gamma = exp(/lngamma)``; generalized gamma
        (AFT) with ``/lnsigma`` and ``/kappa`` (Weibull at kappa = 1, lognormal at 0).
        PH coefficients are log hazard ratios; AFT coefficients are log time ratios.

    Estimator
        Maximum likelihood of ``sum_i w_i [delta_i ln h(t_i) + ln S(t_i) - ln S(t0_i)]``
        (right censoring and delayed entry) by Newton-Raphson with analytic score and
        Hessian; regressors are centred internally and mapped back exactly.

    Parameters
        data: DataFrame (or mapping of columns / list of records).
        time, failure, entry, id: the survival data (see ``oe.stcox``).
        x: main-equation covariates (a constant is always included); None or [] fits the
            constant-only model (Stata's ``streg, distribution(...)``).
        distribution: 'exponential', 'weibull' (default), 'gompertz', 'lognormal',
            'loglogistic' or 'ggamma'.
        metric: None (PH for exponential/Weibull/Gompertz, AFT otherwise), 'ph' or 'aft'
            (Stata's ``time`` option).
        ancillary: covariates of the ancillary-parameter equation (Stata ``ancillary()``).
        strata: one column; stratum indicators enter the main and ancillary equations.
        covariance: 'nonrobust' (default; observed information), 'opg', 'robust' (N/(N-1)
            sandwich; clustered on ``id`` when given), 'cluster' (G/(G-1)).
        cluster, weights, weight_type ('fweight', 'iweight', 'pweight'), offset,
        categorical, missing ('raise' or 'drop'), alpha: as in the other estimators.

    Result
        z statistics. ``metrics``: log_likelihood (Stata's scale: the log likelihood of
        ln t, i.e. the time-scale value plus ``sum w delta ln t``), aic, bic (N =
        observations), df_model, n_subjects, n_failures, time_at_risk. ``tests['model']``:
        LR chi2 of every main-equation term except the constant (stratum indicators
        included) against the constant-only main equation with the ancillary equation as
        specified (Wald under robust/cluster), as Stata reports it. ``extra``:
        hazard_ratios (PH) or time_ratios (AFT) with delta-method SEs and exp-transformed
        intervals; p and 1/p (Weibull), sigma (lognormal, ggamma) or gamma (loglogistic)
        with their intervals; null_log_likelihood.

    Stata
        ``streg age drug, distribution(weibull)`` is ``oe.streg(data=df, time='t',
        failure='died', x=['age', 'drug'])``; add ``time`` -> ``metric='aft'``,
        ``ancillary(drug)`` -> ``ancillary=['drug']``.

    Example
        >>> import numpy as np, pandas as pd, openecon as oe
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"x": rng.normal(size=400)})
        >>> df["t"] = rng.weibull(1.5, 400) * np.exp(-0.5 * df.x / 1.5)
        >>> df["d"] = 1
        >>> print(oe.streg(data=df, time="t", failure="d", x=["x"]).summary())
    """
    spec = build_spec(
        "streg", outcome=time, predictors=column_list(x, "x"),
        categorical=column_list(categorical, "categorical"),
        covariance=resolve_covariance(covariance, cluster, weight_type), cluster=cluster,
        weights=weights, weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"failure": failure, "entry": entry, "id": id, "strata": strata,
                 "offset": offset, "ancillary": column_list(ancillary, "ancillary")},
        options={"distribution": distribution, "metric": metric})
    return fit(spec, data=data)
