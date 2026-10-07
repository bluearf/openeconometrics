"""Markov-switching dynamic regression and autoregression (Stata's ``mswitch dr`` / ``mswitch ar``;
EViews: Markov switching regression).

See ``mswitch_kernels`` for the model, the blocked Hamilton filter, the Kim smoother and
the analytic score. Estimation is exact maximum likelihood (conditional on the first P
observations for autoregressions) by BFGS with the analytic gradient (Fisher identity);
the Hessian for the observed-information covariance is the Ridders-extrapolated
numerical derivative of that analytic gradient (``engines.optimize.numerical_hessian``).
MS likelihoods have several local maxima: the optimizer starts from several
residual-quantile splits and keeps the highest maximum. States are labelled in
ascending order of the first switching coefficient (the constant by default).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    build_result, column_list, information_criteria, kernel_call, ml_covariance, table,
)
from openecon.econometrics.tsmodels import mswitch_kernels as mk
from openecon.econometrics.tsmodels.common import as_int, as_int_list, build_spec, ordered_frame
from openecon.engines.linalg import least_squares
from openecon.engines.optimize import maximize_bfgs, numerical_hessian
from openecon.models import ModelSpec, ResultBundle

MAX_EXPANDED = 512


def _prepare(spec: ModelSpec, data: Any):
    frame, last = ordered_frame(spec, data, what="Markov-switching estimation")
    design = frame.drop_collinear(frame.design())
    states = int(spec.options.get("states", 2))
    switch = spec.options.get("switch")
    switch = (["Intercept"] if spec.intercept else []) if switch is None else list(switch)
    unknown = [term for term in switch if term not in design.terms]
    if unknown:
        raise AnalysisError("invalid_option", f"switch names terms that are not in the model: "
                            f"{', '.join(unknown)} (terms: {', '.join(design.terms)}).")
    lags = sorted(set(spec.options.get("ar") or []))
    if any(lag_ < 1 for lag_ in lags):
        raise AnalysisError("invalid_option", "ar lags must be positive integers.")
    layout = mk.Layout(states, lags, sum(term in switch for term in design.terms),
                       sum(term not in switch for term in design.terms),
                       bool(spec.options.get("arswitch", False)),
                       bool(spec.options.get("varswitch", False)))
    if not (layout.switching or layout.varswitch or (layout.arswitch and lags)):
        raise AnalysisError("invalid_option", "Nothing switches between states: list terms in "
                            "switch, or set varswitch=True (or arswitch=True with ar lags).")
    if layout.expanded > MAX_EXPANDED:
        raise AnalysisError("model_too_large", f"{states} states with AR order {layout.order} give "
                            f"{layout.expanded} expanded states (limit {MAX_EXPANDED}).")
    switching = [i for i, term in enumerate(design.terms) if term in switch]
    common = [i for i, term in enumerate(design.terms) if term not in switch]
    y = frame.numeric(spec.outcome)
    return frame, design, layout, switching, common, y, last


def _terms(layout: mk.Layout, design, switching: list[int], common: list[int],
           outcome: str) -> tuple[list[str], list[str | None]]:
    terms, equations = [], []
    k = layout.states
    for s in range(k):
        for i in switching:
            terms.append(f"state{s + 1}:{design.terms[i]}")
            equations.append(f"state{s + 1}")
    for i in common:
        terms.append(design.terms[i])
        equations.append(None)
    for s in range(k if layout.arswitch else 1):
        for lag_ in layout.lags:
            prefix = f"state{s + 1}:" if layout.arswitch else ""
            terms.append(f"{prefix}L{lag_}.ar")
            equations.append(f"state{s + 1}" if layout.arswitch else None)
    if layout.varswitch:
        terms += [f"/lnsigma{s + 1}" for s in range(k)]
        equations += [None] * k
    else:
        terms.append("/lnsigma")
        equations.append(None)
    for a in range(k):
        for c in range(k - 1):
            terms.append(f"/lgt(p{a + 1}{c + 1})")
            equations.append(None)
    return terms, equations


def _starts(layout: mk.Layout, y: Tensor, xs: Tensor, xc: Tensor) -> list[Tensor]:
    """Starting vectors: OLS with the intercept-like offsets from residual quantile groups."""
    order, k = layout.order, layout.states
    rows = slice(order, y.shape[0])
    columns = [xs[rows], xc[rows]]
    columns += [y[order - lag_:y.shape[0] - lag_, None] for lag_ in layout.lags]
    design = torch.cat(columns, dim=1)
    fit = kernel_call(least_squares, design, y[rows].contiguous())
    beta = torch.zeros(design.shape[1], dtype=torch.float64)
    beta[fit.kept] = fit.beta
    resid = fit.resid
    sd = float(resid.std())
    if not sd > 0.0:
        raise AnalysisError("perfect_fit", "The linear model fits the outcome exactly; there is no "
                            "variation left for regime switching.")
    ks, kc = layout.switching, layout.common
    order_ = torch.argsort(resid)
    groups = torch.tensor_split(order_, k)
    offsets = torch.tensor([float(resid[g].mean()) for g in groups], dtype=torch.float64)
    spreads = torch.tensor([max(float(resid[g].std()), 0.1 * sd) if len(g) > 1 else sd
                            for g in groups], dtype=torch.float64)
    phi = beta[ks + kc:]
    starts = []
    for persistence, shrink in ((0.9, 1.0), (0.7, 0.5), (0.95, 1.5)):
        b = beta[:ks].repeat(k, 1)
        if ks:
            b[:, 0] = b[:, 0] + shrink * offsets           # the first switching term moves
        alpha = beta[ks:ks + kc]
        ar = phi.repeat(k) if layout.arswitch else phi
        lnsigma = torch.log(spreads) if layout.varswitch else torch.tensor([math.log(sd)])
        p = torch.full((k, k), (1 - persistence) / max(k - 1, 1), dtype=torch.float64)
        p.fill_diagonal_(persistence)
        logit = torch.log(p[:, :k - 1] / p[:, k - 1:])
        starts.append(torch.cat([b.reshape(-1), alpha, ar, lnsigma.to(torch.float64),
                                 logit.reshape(-1)]))
    return starts


def _relabel(layout: mk.Layout, theta: Tensor) -> tuple[Tensor, list[int]]:
    """Order states by the first switching coefficient (else by sigma); returns theta, order."""
    parts = layout.split(theta)
    k = layout.states
    key = parts["beta"][:, 0] if layout.switching else (
        parts["lnsigma"] if layout.varswitch else torch.arange(k, dtype=torch.float64))
    order = torch.argsort(key, stable=True).tolist()
    if order == list(range(k)):
        return theta, order
    p = mk.transition(parts["logit"])[order][:, order]
    logit = torch.log(p[:, :k - 1] / p[:, k - 1:])
    pieces = [parts["beta"][order].reshape(-1), parts["alpha"],
              (parts["phi"][order] if layout.arswitch else parts["phi"]).reshape(-1),
              parts["lnsigma"][order] if layout.varswitch else parts["lnsigma"], logit.reshape(-1)]
    return torch.cat(pieces), order


def _units(layout: mk.Layout, y: Tensor, xs: Tensor, xc: Tensor) -> tuple[float, Tensor]:
    """Scale of y and the diagonal map from standardized to reported parameters.

    The likelihood is maximized for y / s_y on regressors x_j / s_j (s_y the standard
    deviation of y, s_j the root mean square of column j), which makes BFGS and its
    absolute gradient tolerance independent of the units of the data. In reported units
    beta_j = s_y / s_j * beta*_j, lnsigma = lnsigma* + ln s_y; AR coefficients and the
    transition logits are unit free. Returns s_y and the column scales [ks + kc].
    """
    scale_y = float(y.std())
    if not scale_y > 0.0 or not math.isfinite(scale_y):
        raise AnalysisError("constant_outcome", "The outcome is constant; there is nothing to "
                            "explain by regime switching.")
    columns = torch.cat([xs, xc], dim=1)
    scales = columns.square().mean(0).sqrt()
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    return scale_y, scales


def _to_reported(layout: mk.Layout, scale_y: float, scales: Tensor) -> tuple[Tensor, Tensor]:
    """(multiplier, offset): theta = multiplier * theta* + offset (elementwise)."""
    sizes = layout.sizes()
    k, ks = layout.states, layout.switching
    multiplier = torch.ones(layout.size, dtype=torch.float64)
    offset = torch.zeros(layout.size, dtype=torch.float64)
    if ks:
        multiplier[:sizes["beta"]] = (scale_y / scales[:ks]).repeat(k)
    start = sizes["beta"]
    multiplier[start:start + sizes["alpha"]] = scale_y / scales[ks:]
    start += sizes["alpha"] + sizes["phi"]
    offset[start:start + sizes["lnsigma"]] = math.log(scale_y)
    return multiplier, offset


def fit_mswitch(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of the ``mswitch`` estimator (see the module docstring and ``mswitch``)."""
    frame, design, layout, switching, common, y_raw, _ = _prepare(spec, data)
    xs_raw, xc_raw = design.x[:, switching], design.x[:, common]
    order, k = layout.order, layout.states
    n = frame.n - order
    if n < 3 * layout.size + 5:
        raise AnalysisError("insufficient_observations", f"{n} observations are too few for a "
                            f"{k}-state model with {layout.size} parameters.")
    max_iterations = int(frame.option("max_iterations"))
    # Estimation runs in standardized units (see _units); results are mapped back below.
    scale_y, scales = _units(layout, y_raw, xs_raw, xc_raw)
    y = y_raw / scale_y
    xs, xc = xs_raw / scales[:layout.switching], xc_raw / scales[layout.switching:]
    multiplier, offset = _to_reported(layout, scale_y, scales)

    def objective(theta: Tensor):
        out = kernel_call(mk.score, layout, theta, y, xs, xc)
        return out["loglik"], out["gradient"]

    def gradient(theta: Tensor) -> Tensor:
        return kernel_call(mk.score, layout, theta, y, xs, xc)["gradient"]

    hessian_fn = lambda theta: numerical_hessian(gradient, theta, levels=1)  # noqa: E731
    # Screen the starting values with a short run, then finish from the best of them.
    screened, attempts = [], []
    for start in _starts(layout, y, xs, xc):
        try:
            result = kernel_call(maximize_bfgs, objective, start, max_iter=min(40, max_iterations),
                                 hessian_fn=hessian_fn, raise_on_failure=False)
        except AnalysisError as exc:                   # a start in a degenerate region
            attempts.append({"converged": False, "message": str(exc)[:120]})
            continue
        value = float(result.value)
        attempts.append({"converged": bool(result.converged), "log_likelihood": value,
                         "iterations": result.iterations})
        if math.isfinite(value):
            screened.append((value, len(screened), result))
    best = None
    for value, _, result in sorted(screened, key=lambda item: (-item[0], item[1])):
        if not result.converged:
            try:
                result = kernel_call(maximize_bfgs, objective, result.theta,
                                     max_iter=max_iterations, hessian_fn=hessian_fn,
                                     raise_on_failure=False)
            except AnalysisError:
                continue
        if result.converged:
            best = result
            break
    if best is None:
        raise AnalysisError("nonconvergence", "The likelihood maximization did not converge from "
                            "any starting value; simplify the model (fewer states or switching "
                            "terms) or raise max_iterations.")
    theta, state_order = _relabel(layout, best.theta)
    if order:
        frame.restrict(torch.arange(frame.n) >= order, f"The first {order} observation(s) are the "
                       "presample lags of the autoregression.")
    out = kernel_call(mk.score, layout, theta, y, xs, xc, need_smoothed=True)
    hessian = kernel_call(numerical_hessian, gradient, theta)
    scores = None
    if spec.covariance in ("robust", "opg"):
        scores = kernel_call(_per_period_scores, layout, theta, y, xs, xc)
    covariance, info = ml_covariance(frame, hessian=hessian, scores=scores)
    fitted = _fitted(layout, theta, y, xs, xc, out["predicted"]) * scale_y
    # Back to the units of the data: a diagonal reparameterization (see _units).
    theta = multiplier * theta + offset
    covariance = multiplier[:, None] * covariance * multiplier[None, :]
    terms, equations = _terms(layout, design, switching, common, spec.outcome)
    log_likelihood = float(out["loglik"]) - n * math.log(scale_y)
    p, pi = out["transition"], out["ergodic"]
    extra = _transition_summary(layout, theta, covariance, p, pi)
    metrics = {**information_criteria(log_likelihood, layout.size, n), "states": k}
    for s in range(k):
        metrics[f"duration_state{s + 1}"] = 1.0 / (1.0 - float(p[s, s])) if float(p[s, s]) < 1 \
            else None
    lag_state = layout.lag_state()
    smoothed_base = torch.zeros((n, k), dtype=torch.float64).index_add_(1, lag_state[0],
                                                                        out["smoothed"])
    extra.update({
        "state_order": "states sorted by their first switching coefficient"
        if layout.switching else "states sorted by sigma" if layout.varswitch else "as estimated",
        "starts": attempts, "switching_terms": [design.terms[i] for i in switching],
        "ar_lags": layout.lags, "arswitch": layout.arswitch, "varswitch": layout.varswitch,
        "smoothed_share": smoothed_base.mean(0).tolist(),
        "estimation_units": {"outcome_scale": scale_y, "column_scales": scales.tolist(),
                             "note": "maximized for y / outcome_scale on columns / column_scales; "
                                     "reported in the units of the data"},
    })
    info["correction"] = f"{info['correction']}; transition probabilities, sigmas and durations by " \
                         "the delta method in extra"
    return build_result(
        frame, terms=terms, params=theta, covariance=covariance,
        title=f"Markov-switching {'autoregression' if order else 'dynamic regression'} ({k} states)",
        equations=equations, use_t=False, metrics=metrics, fitted=fitted,
        solver="bfgs_hamilton_filter",
        solver_diagnostics={"converged": True, "iterations": best.iterations},
        optimizer={"method": "bfgs", "gradient": "analytic (Fisher identity with the Kim smoother)",
                   "hessian": "numerical derivative of the analytic gradient (Ridders)",
                   "starts": len(attempts)},
        inference=info, extra=extra, categories=design.categories,
        provenance={"likelihood": "exact (conditional on the first P observations for AR models); "
                                  "initial state from the ergodic distribution",
                    "fitted_values": "one-step predictions E[y_t | Y_(t-1)]",
                    "time_order": f"sorted by {spec.time}" if spec.time else "input row order"},
    )


def _per_period_scores(layout: mk.Layout, theta: Tensor, y: Tensor, xs: Tensor, xc: Tensor) -> Tensor:
    """Scores of log f(y_t | Y_(t-1)) by central differences (n x p)."""
    eps = torch.finfo(torch.float64).eps ** (1.0 / 3.0)
    columns = []
    for j in range(theta.shape[0]):
        h = eps * max(1.0, abs(float(theta[j])))
        up, down = theta.clone(), theta.clone()
        up[j] += h
        down[j] -= h
        columns.append((mk.score(layout, up, y, xs, xc)["per_period"]
                        - mk.score(layout, down, y, xs, xc)["per_period"]) / (2 * h))
    return torch.stack(columns, dim=1)


def _fitted(layout: mk.Layout, theta: Tensor, y: Tensor, xs: Tensor, xc: Tensor,
            predicted: Tensor) -> Tensor:
    """E[y_t | Y_(t-1)] = sum_S xi_(t|t-1)(S) (y_t - e_t(S))."""
    pieces = mk.densities(layout, theta, y, xs, xc)
    order = layout.order
    observed = y[order:]
    return (predicted * (observed[:, None] - pieces.residual)).sum(1)


def _transition_summary(layout: mk.Layout, theta: Tensor, covariance: Tensor, p: Tensor,
                        pi: Tensor) -> dict[str, Any]:
    k = layout.states
    sizes = layout.sizes()
    start = sum(size for name, size in sizes.items() if name != "logit")
    block = covariance[start:, start:]
    rows = []
    for a in range(k):
        jac = torch.zeros((k, k * (k - 1)), dtype=torch.float64)
        for b in range(k):
            for c in range(k - 1):
                jac[b, a * (k - 1) + c] = p[a, b] * ((1.0 if b == c else 0.0) - p[a, c])
        var = (jac @ block @ jac.T).diagonal().clamp_min(0.0)
        rows.append([{"probability": float(p[a, b]), "std_error": float(var[b].sqrt())}
                     for b in range(k)])
    parts = layout.split(theta)
    s_start = sizes["beta"] + sizes["alpha"] + sizes["phi"]
    sigmas = []
    for s in range(sizes["lnsigma"]):
        value = math.exp(float(parts["lnsigma"][s]))
        sigmas.append({"sigma": value,
                       "std_error": value * math.sqrt(max(float(covariance[s_start + s, s_start + s]), 0))})
    return {"transition_matrix": rows, "ergodic_probabilities": pi.tolist(), "sigma": sigmas,
            "transition_definition": "p_ab = P(s_t = b | s_(t-1) = a); lgt(p_ac) = log(p_ac / p_ak)"}


def mswitch_probabilities(result: ResultBundle, data: Any) -> pd.DataFrame:
    """Filtered and smoothed state probabilities of a fitted ``oe.mswitch`` model.

    Runs the Hamilton filter and the Kim smoother at the estimates over ``data`` and
    returns one row per estimation period: ``period``, ``observed``,
    ``filtered_state1..k`` = P(s_t = j | y_1..y_t), ``smoothed_state1..k`` =
    P(s_t = j | y_1..y_n) and ``predicted_state1..k`` = P(s_t = j | y_1..y_(t-1)).
    Stata: ``predict pr*, pr`` (filtered) and ``predict pr*, pr smethod(smooth)``.

    Example
    -------
    >>> probs = oe.mswitch_probabilities(fit, df)
    >>> probs[["period", "smoothed_state2"]].tail()
    """
    from openecon.dataset import Dataset
    if isinstance(data, Dataset):
        from openecon.econometrics.streaming_mswitch import mswitch_probabilities_streaming
        return mswitch_probabilities_streaming(result, data)
    if not isinstance(result, ResultBundle) or result.spec.estimator != "mswitch":
        raise AnalysisError("invalid_result", "mswitch_probabilities needs a result returned by "
                            "oe.mswitch.")
    frame, design, layout, switching, common, y, _ = _prepare(result.spec, data)
    terms, _ = _terms(layout, design, switching, common, result.spec.outcome)
    estimates = {c.term: c.estimate for c in result.coefficients}
    if terms != list(estimates):
        raise AnalysisError("invalid_data", "The data do not reproduce the terms of the fitted model.")
    theta = torch.tensor([estimates[term] for term in terms], dtype=torch.float64)
    xs, xc = design.x[:, switching], design.x[:, common]
    out = kernel_call(mk.score, layout, theta, y, xs, xc, need_smoothed=True)
    order, k = layout.order, layout.states
    lag_state = layout.lag_state()
    n = y.shape[0] - order

    def base(values: Tensor) -> Tensor:
        return torch.zeros((n, k), dtype=torch.float64).index_add_(1, lag_state[0], values)

    filtered, smoothed, predicted = base(out["filtered"]), base(out["smoothed"]), base(out["predicted"])
    if order:
        frame.restrict(torch.arange(frame.n) >= order)
    columns: dict[str, Any] = {
        "period": frame.series(result.spec.time).tolist() if result.spec.time else frame.positions,
        "observed": y[order:].tolist()}
    for name, values in (("filtered", filtered), ("smoothed", smoothed), ("predicted", predicted)):
        for s in range(k):
            columns[f"{name}_state{s + 1}"] = values[:, s].tolist()
    return table(columns, title=f"State probabilities of {result.spec.outcome}",
                 states=k, smoother="Kim (1994)")


def mswitch(*, data: Any, y: str, x: Any = None, time: str | None = None, states: int = 2,
            switch: Any = None, varswitch: bool = False, ar: Any = None, arswitch: bool = False,
            intercept: bool = True, covariance: str | None = None, categorical: Any = None,
            max_iterations: int = 500, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Markov-switching dynamic regression / autoregression (Stata's ``mswitch dr`` and ``mswitch ar``).

    Model
    -----
    ``y_t = mu_t(s_t) + sum_(i in ar) phi_i (y_(t-i) - mu_(t-i)(s_(t-i))) + e_t`` with
    ``mu_t(s) = x_t'beta_s + w_t'alpha`` (terms in ``switch`` have state-specific
    coefficients beta_s, the others common coefficients alpha), ``e_t ~ N(0, sigma^2)``
    (``varswitch=True``: sigma_s) and an unobserved Markov chain s_t with constant
    transition probabilities p_ab = P(s_t = b | s_(t-1) = a). ``ar=None`` is Stata's
    dynamic-regression model (``mswitch dr``); ``ar=p`` or a list of lags is ``mswitch
    ar`` (Hamilton 1989: the autoregression acts on deviations from the state means,
    so the density depends on (s_t, ..., s_(t-p))); ``arswitch=True`` makes phi
    state-specific.

    Estimator
    ---------
    Exact ML through the Hamilton filter (initial state from the ergodic distribution,
    AR models conditional on the first p observations), BFGS with the analytic gradient
    (Fisher identity with the Kim smoother), several starting values (residual-quantile
    splits, different persistence) with the highest maximum kept. States are relabelled
    in ascending order of the first switching coefficient (the constant by default).

    Parameters
    ----------
    data, y : table and outcome. x : regressors (common unless listed in ``switch``).
    time : optional time column (consecutive periods). states : number of regimes (2..6).
    switch : terms with state-specific coefficients, default ``["Intercept"]``.
    varswitch, ar, arswitch : see Model. intercept : include the constant.
    covariance : ``"nonrobust"`` (default; observed information, numerical Hessian of the
    analytic gradient), ``"robust"`` (Huber-White, N/(N-1)) or ``"opg"`` (per-period scores
    by central differences). categorical : regressors to expand. max_iterations : BFGS
    limit per start. missing : ``"raise"`` or ``"drop"`` (start/end only). alpha : level.

    Result
    ------
    Coefficients ``state1:Intercept``, ``state2:Intercept``, ..., common terms, ``L1.ar`` (or
    ``state1:L1.ar``), ``/lnsigma`` (or ``/lnsigma1``...), ``/lgt(p11)``, ``/lgt(p21)``, ...
    (log-odds of p_ac against the last state; for two states logit(p11), logit(p21)); z
    tests. ``metrics``: ``log_likelihood``, ``aic``, ``bic``, ``states``,
    ``duration_state1..k`` (expected durations 1/(1 - p_jj)). ``extra``:
    ``transition_matrix`` (probabilities with delta-method standard errors),
    ``ergodic_probabilities``, ``sigma`` (with standard errors), ``starts``.
    ``oe.mswitch_probabilities(result, data)`` returns filtered, predicted and smoothed
    state probabilities.

    Stata: ``mswitch dr y, switch(x) varswitch``; ``mswitch ar y, ar(1/2) arswitch``.
    EViews: Markov Switching Regression (switchtype=markov).

    Example
    -------
    >>> fit = oe.mswitch(data=df, y="fedfunds", time="quarter", states=2, varswitch=True)
    >>> fit.extra["transition_matrix"][0][0]["probability"]
    """
    if isinstance(ar, int) and not isinstance(ar, bool):
        ar = list(range(1, ar + 1)) if ar > 0 else None
    spec = build_spec(
        "mswitch", outcome=y, predictors=column_list(x, "x"), intercept=intercept,
        categorical=column_list(categorical, "categorical"), time=time, covariance=covariance,
        missing=missing, alpha=alpha,
        options={"states": as_int(states), "switch": None if switch is None else column_list(switch, "switch"),
                 "varswitch": varswitch, "ar": as_int_list(ar, "ar"), "arswitch": arswitch,
                 "max_iterations": as_int(max_iterations)},
    )
    from openecon.analysis import fit

    return fit(spec, data=data)
