"""General method of moments over moment expressions: Stata's ``gmm`` (EViews GMM).

Model
-----
Moment conditions ``E[z_ij u_j(x_i; b)] = 0`` for residual expressions
``u_j`` (``j = 1..M``, written in the ``nl`` formula grammar: ``{b0}`` parameters,
column names, ``+ - * / ^``, ``exp``, ``ln`` ...) and instruments ``z_ij`` (the
constant included unless ``instrument_constant=False``). ``L = sum_j L_j``
moments identify ``P <= L`` parameters.

Estimator (Methods and formulas of [R] gmm)
-------------------------------------------
``b`` minimizes ``Q(b) = g(b)'W g(b)`` with ``g = sum_i w_i m_i`` (see
``gmm_kernels``; Gauss-Newton with the analytic moment Jacobian ``G``).

* one step: ``W = winitial``: the identity (Stata's default) or ``unadjusted``,
  ``blockdiag (Z_j'Z_j)^-1``, which makes a linear model's first step 2SLS;
* two step (default): ``W = S(b_1)^-1`` with the moment covariance ``S`` of
  ``wmatrix`` at the first-step estimates;
* iterated (``igmm``): ``W`` recomputed from the latest estimates until the
  coefficients change by less than ``igmm_tolerance`` (relative).

``S`` (sum form, no small-sample factors): ``robust`` ``sum_i w_i^2 m_i m_i'``;
``cluster`` the outer products of the cluster sums; ``hac`` the kernel-weighted
autocovariances (``lags``, ``kernel``; in time order within panels when a
time column is given); ``unadjusted`` ``S_jk = s_jk Z_j'Z_k``,
``s_jk = u_j'u_k / N``. ``center`` removes the mean moment first.

Covariance
----------
``B = (G'WG)^-1`` with ``G`` and ``W`` of the final step. When the estimator
reweights (two-step or iterated) and the covariance type is the weight-matrix
type, the efficient form ``V = B`` is reported (as ``ivregress gmm`` does);
otherwise the sandwich ``V = B G'W S_hat W G B`` with ``S_hat`` of the
covariance type at the final estimates. Cluster covariances carry ``G/(G-1)``.
z inference. Hansen's ``J = g'Wg ~ chi2(L - P)`` after a two-step or iterated
estimator (the criterion at the final weight matrix ``S^-1``).

A linear moment ``y - {b0} - {b1}*x1 - {b2}*x2`` with ``winitial='unadjusted'``
reproduces ``ivregress gmm`` (estimates, covariance and J).

Numerics
--------
Every equation's instruments are replaced by the orthonormal basis
``Q_j = Z_j R_j^-1`` (``gmm_kernels.orthonormal_instruments``): all formulas above
except the identity weight are invariant to that change of basis, and ``S``
stays well conditioned when an instrument has a large offset; the identity
weight is applied exactly as ``||R'g_Q||^2``. ``(G'WG)^-1`` comes from the QR
factor of the whitened Jacobian, never from the normal equations. A moment
equation whose residuals vanish at the estimates (``sum w u_j^2`` at most
``1e-24`` times the smallest ``sum w c^2`` of its data columns) raises
``perfect_fit``: its moment covariance and every standard error would be zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import ModelFrame, build_result, column_list, kernel_call, make_spec
from openecon.econometrics.iv.common import Setting, check_pweights, moment_meat
from openecon.econometrics.quantile import formula as formulas
from openecon.econometrics.systems import gmm_kernels as gk
from openecon.econometrics.systems.common import check_system_role, estimation_weights
from openecon.engines.distributions import chi2_sf
from openecon.engines.linalg import least_squares
from openecon.models import ModelSpec, ResultBundle

_KIND = {"nonrobust": "unadjusted", "robust": "robust", "cluster": "cluster", "hac": "hac"}
# sum w u_j^2 below this fraction of the smallest sum w c^2 of the equation's data columns c
# is rounding: the moment equation fits exactly and its moment covariance is zero.
_EXACT_FIT = 1e-24
# Relative residual sum of squares of a whitened Jacobian column (given the earlier ones) at
# or below which a parameter is not identified (condition numbers up to about 1e12 pass).
_JACOBIAN_TOL = 1e-24


def _exact_fit_floors(parsed: Sequence[formulas.Formula], columns: dict[str, torch.Tensor],
                      weights: torch.Tensor | None) -> list[float]:
    floors = []
    for formula in parsed:
        sums = [float(columns[name].square().sum() if weights is None
                      else weights @ columns[name].square()) for name in formula.columns]
        positive = [value for value in sums if value > 0]
        floors.append(_EXACT_FIT * min(positive) if positive else 0.0)
    return floors


def _check_exact_fit(resid: torch.Tensor, weights: torch.Tensor | None, floors: Sequence[float],
                     labels: Sequence[str]) -> None:
    """Refuse a fit whose residuals vanish: S and every standard error would be zero."""
    squares = resid.square()
    sums = squares.sum(dim=0) if weights is None else weights @ squares
    for label, value, floor in zip(labels, sums.tolist(), floors, strict=True):
        if value <= floor:
            raise AnalysisError("perfect_fit", f"The residuals of moment equation {label} are "
                                "zero at the estimates (an exact fit), so the moment covariance "
                                "and every standard error are zero. Remove the regressor that "
                                "reproduces the outcome, or check for a constant outcome.")


def _moment_texts(value: Any) -> tuple[list[str], list[str]]:
    if isinstance(value, dict) and value:
        labels, texts = [str(key) for key in value], list(value.values())
    elif isinstance(value, (list, tuple)) and value:
        labels, texts = [str(i) for i in range(1, len(value) + 1)], list(value)
    else:
        raise AnalysisError("invalid_spec", "moments must be a non-empty list of residual "
                            "expressions such as ['y - {b0} - {b1}*x'] (or a dict label -> "
                            "expression).")
    if not all(isinstance(text, str) for text in texts):
        raise AnalysisError("invalid_spec", "Every moment must be a residual expression string.")
    if any(":" in label or not label.strip() for label in labels):
        raise AnalysisError("invalid_spec", "Moment labels must be non-empty and contain no ':'.")
    return labels, texts


def _instrument_lists(value: Any, count: int) -> list[list[str]]:
    if value is None:
        return [[] for _ in range(count)]
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return [list(value) for _ in range(count)]
    if isinstance(value, (list, tuple)) and len(value) == count and all(
            isinstance(item, (list, tuple)) and all(isinstance(n, str) for n in item)
            for item in value):
        return [list(item) for item in value]
    raise AnalysisError("invalid_spec", "instruments must be a list of column names (common to "
                        "every moment equation) or one list per equation.")


def parse_moments(value: Any) -> tuple[list[str], list[formulas.Formula]]:
    labels, texts = _moment_texts(value)
    return labels, [formulas.parse(text) for text in texts]


def moment_columns(parsed: Sequence[formulas.Formula], instruments: Sequence[Sequence[str]]
                   ) -> list[str]:
    names = [name for formula in parsed for name in formula.columns]
    names.extend(name for block in instruments for name in block)
    return list(dict.fromkeys(names))


def _parameters(frame: ModelFrame, parsed: Sequence[formulas.Formula]
                ) -> tuple[list[str], list[list[int]], torch.Tensor]:
    names: list[str] = []
    written: dict[str, float] = {}
    for formula in parsed:
        for name in formula.parameters:
            if name not in names:
                names.append(name)
        for name, value in formula.start.items():
            if name in written and written[name] != value:
                raise AnalysisError("invalid_start", f"Parameter '{name}' is given two different "
                                    "starting values in the moment expressions.")
            written[name] = value
    given = frame.spec.options.get("start") or {}
    if not isinstance(given, dict):
        raise AnalysisError("invalid_start", "start must map parameter names to numbers.")
    unknown = [name for name in given if name not in names]
    if unknown:
        raise AnalysisError("invalid_start", f"start names unknown parameters: "
                            f"{', '.join(map(str, unknown))}. Parameters: {', '.join(names)}.")
    values = {**written, **given}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value):
            raise AnalysisError("invalid_start", f"The starting value of '{name}' must be a "
                                "finite number.")
    index = [[names.index(name) for name in formula.parameters] for formula in parsed]
    start = torch.tensor([float(values.get(name, 0.0)) for name in names], dtype=torch.float64)
    return names, index, start


def _wmatrix(frame: ModelFrame) -> str:
    spec = frame.spec
    wmatrix = spec.options.get("wmatrix") or (
        spec.covariance if spec.covariance in {"cluster", "hac"} else "robust")
    if wmatrix == "cluster" and spec.covariance != "cluster":
        raise AnalysisError("invalid_spec", "wmatrix='cluster' needs the cluster column: pass "
                            "cluster=... (the covariance is then cluster as well).")
    hac = wmatrix == "hac" or spec.covariance == "hac"
    if hac and "lags" not in spec.options:
        raise AnalysisError("invalid_spec", "A HAC weight matrix or covariance needs lags (the "
                            "Newey-West lag length; 0 gives White's estimator).")
    if not hac and ("lags" in spec.options or "kernel" in spec.options):
        raise AnalysisError("invalid_spec", "lags and kernel apply only with a HAC weight "
                            "matrix or covariance.")
    if hac and spec.panel is not None and spec.time is None:
        raise AnalysisError("invalid_spec", "HAC autocovariances within panels need a time "
                            "column alongside the panel column.")
    return wmatrix


def _instruments(frame: ModelFrame, labels: Sequence[str], lists: Sequence[Sequence[str]],
                 constant: bool, weights) -> tuple[list[torch.Tensor], list[list[str]]]:
    blocks, terms = [], []
    for label, names in zip(labels, lists, strict=True):
        if not names and not constant:
            raise AnalysisError("invalid_spec", f"Moment equation {label} has no instruments; "
                                "give instruments or keep the constant.")
        design = frame.design(list(names), intercept=constant, prefix=f"{label}:")
        design = frame.drop_collinear(design, weights)
        blocks.append(design.x)
        terms.append(design.terms)
    return blocks, terms


def fit_gmm(spec: ModelSpec, data: Any, *, _frame=None, _replay=None) -> ResultBundle:
    """Entry point of ``gmm``: ``(spec, data) -> ResultBundle``."""
    labels, parsed = parse_moments(spec.options.get("moments"))
    lists = _instrument_lists(spec.options.get("instruments"), len(parsed))
    frame = ModelFrame(spec, data) if _frame is None else _frame
    check_system_role(frame, moment_columns(parsed, lists))
    check_pweights(frame)
    wmatrix = _wmatrix(frame)
    weights, nobs = estimation_weights(frame) if _replay is None else (None, _replay.sample.nobs)
    names, index, start = _parameters(frame, parsed)
    p = len(names)
    if _replay is not None:
        system, identity, instrument_terms = _replay.initialize(parsed, index, p)
        widths = system.widths
    else:
        planned_widths = [frame.design_width(list(names), intercept=bool(frame.option("instrument_constant")))
                      for names in lists]
        resource_plan = gk.gmm_workspace_plan(frame.n, p, planned_widths, len(parsed),
                                         resident_bytes=frame.resource_input_bytes,
                                         budget_bytes=frame.resource_budget_bytes,
                                         formula_tape_elements=max(frame.n * len(formula.tape) * (1 + len(formula.parameters))
                                                                   for formula in parsed))
        frame.resource_plans.append(resource_plan.record())
        blocks, instrument_terms = _instruments(frame, labels, lists,
                                            bool(frame.option("instrument_constant")), weights)
        widths = [block.shape[1] for block in blocks]
    moments_count = sum(widths)
    if moments_count < p:
        raise AnalysisError("underidentified", f"{moments_count} moment condition(s) cannot "
                            f"identify {p} parameters; add instruments.")
    if (_replay.sample.nrows if _replay is not None else frame.n) <= p:
        raise AnalysisError("insufficient_observations", f"The model has {p} parameters but only "
                            f"{frame.n} observation(s).")
    if _replay is None:
        columns = {name: frame.numeric(name) for formula in parsed for name in formula.columns}
        bases, identity = kernel_call(gk.orthonormal_instruments, blocks, weights)
        system = gk.MomentSystem(list(parsed), index, columns, bases, weights, frame.n, p,
                             resident_bytes=frame.resource_input_bytes, budget_bytes=frame.resource_budget_bytes)
    clusters = (frame.cluster_dimensions() if _replay is None else [(None, _replay.cluster_count())]) if spec.covariance == "cluster" else None
    setting = Setting(nobs, weights, spec.covariance, False, clusters=clusters,
                      cluster_names=registry.cluster_columns(spec) or None)
    center = bool(frame.option("center"))

    def moment_s(resid: torch.Tensor, kind: str) -> torch.Tensor:
        if _replay is not None:
            return _replay.moment_s(kind, center)
        if kind == "unadjusted":
            return gk.unadjusted_s(system, resid, nobs)
        rows = system.rows(resid)
        if center:
            mean = rows.mean(dim=0) if weights is None else weights @ rows / weights.sum()
            rows = rows - mean
        return moment_meat(frame, setting, rows, kind=kind)

    def weighting(s: torch.Tensor | None) -> gk.Weighting:
        try:
            return kernel_call(gk.Weighting, s, moments_count)
        except AnalysisError as exc:
            raise AnalysisError("singular_weight_matrix", "The moment covariance is singular "
                                "(for a cluster weight matrix: fewer clusters than moments), "
                                "so its inverse cannot weight the moments. Use fewer "
                                "instruments or another wmatrix.") from exc

    tolerance, max_iterations = float(frame.option("tolerance")), int(frame.option("max_iterations"))
    twostep, igmm = bool(frame.option("twostep")), bool(frame.option("igmm"))
    estimator = "igmm" if igmm else "twostep" if twostep else "onestep"
    winitial = frame.option("winitial")
    current = gk.Weighting(None, moments_count, whitener=identity) if winitial == "identity" \
        else weighting(gk.identity_blocks(system) if _replay is None else _replay.initial_s())
    step = kernel_call(gk.minimize, system, current, start, tolerance=tolerance,
                       max_iterations=max_iterations)
    floors = _exact_fit_floors(parsed, columns, weights) if _replay is None else None
    if _replay is None:
        _check_exact_fit(step.resid, weights, floors, labels)
    else:
        _replay.check_exact(labels)
    steps, iterations, converged = 1, step.iterations, True
    if estimator != "onestep":
        limit = int(frame.option("igmm_max_iterations")) if igmm else 1
        converged = not igmm
        for _ in range(limit):
            current = weighting(moment_s(step.resid, wmatrix))
            previous = step.theta
            step = kernel_call(gk.minimize, system, current, previous, tolerance=tolerance,
                               max_iterations=max_iterations)
            if _replay is None:
                _check_exact_fit(step.resid, weights, floors, labels)
            else:
                _replay.check_exact(labels)
            steps += 1
            iterations += step.iterations
            if igmm and gk.relative_change(step.theta, previous) <= float(
                    frame.option("igmm_tolerance")):
                converged = True
                break
        if not converged:
            raise AnalysisError("nonconvergence", f"Iterated GMM did not converge in {steps - 1} "
                                "reweighting steps; use the two-step estimator (igmm=False) or "
                                "raise igmm_max_iterations.")
    jac = step.jacobian
    wjac = current.apply(jac)
    white = current.whiten(jac)
    # (G'WG)^-1 = (G~'G~)^-1 from the QR factor of the whitened Jacobian G~ = C^-1 G
    # (never from the normal equations, whose condition number is squared).
    fit = kernel_call(least_squares, white, torch.zeros(white.shape[0], dtype=torch.float64),
                      drop_collinear=True, tol=_JACOBIAN_TOL)
    if fit.omitted:
        which = ", ".join(names[i] for i in fit.omitted)
        raise AnalysisError("not_identified", f"The moment Jacobian is rank deficient at the "
                            f"estimates: {which} cannot be identified from these moments and "
                            "instruments.")
    bread = fit.xtx_inv
    kind = _KIND[spec.covariance]
    efficient = estimator != "onestep" and kind == wmatrix
    factor = 1.0
    info: dict[str, Any] = {"covariance": spec.covariance}
    if spec.covariance == "cluster":
        count = clusters[0][1]
        factor = count / (count - 1)
        info.update({"cluster_count": count, "cluster_df": count - 1,
                     "small_sample_correction": factor})
    if efficient:
        covariance = bread * factor
        info["correction"] = "efficient GMM (G'WG)^-1 with W the final weight matrix" + (
            "; G/(G-1)" if factor != 1.0 else "")
    else:
        meat = wjac.T @ moment_s(step.resid, kind) @ wjac
        covariance = bread @ meat @ bread * factor
        info["correction"] = (f"GMM sandwich (G'WG)^-1 G'W S W G (G'WG)^-1 with the {kind} S at "
                              "the final estimates" + ("; G/(G-1)" if factor != 1.0 else ""))
    tests: dict[str, Any] = {}
    j_value = None
    if estimator != "onestep" and moments_count > p:
        j_value = step.criterion
        df = moments_count - p
        tests["hansen_j"] = {"statistic": j_value, "df": df, "p_value": chi2_sf(j_value, df),
                             "distribution": "chi2", "label": "Hansen's J test of "
                             "overidentifying restrictions"}
    elif estimator == "onestep" and moments_count > p:
        frame.warn("Hansen's J test needs an efficient weight matrix; it is not reported for "
                   "the one-step estimator.")
    metrics = {"j": j_value, "criterion": step.criterion / nobs if j_value is not None else None,
               "n_moments": moments_count, "n_parameters": p, "n_equations": len(parsed),
               "gmm_steps": steps, "iterations": iterations}
    info.update({"nobs": nobs, "estimator": estimator, "wmatrix": wmatrix, "winitial": winitial,
                 "centered": center, "df_inference": None})
    extra = {"moments": dict(zip(labels, [f.source for f in parsed], strict=True)),
             "instruments": dict(zip(labels, instrument_terms, strict=True)),
             "parameters": names, "start": dict(zip(names, start.tolist(), strict=True)),
             "estimator": estimator, "wmatrix": wmatrix, "winitial": winitial}
    return build_result(
        frame, terms=names, params=step.theta, covariance=covariance,
        title={"onestep": "GMM estimation (one-step)", "twostep": "GMM estimation (two-step)",
               "igmm": "GMM estimation (iterated)"}[estimator],
        use_t=False, metrics=metrics, nobs=nobs, inference=info, tests=tests, extra=extra,
        solver="gauss_newton_levenberg_marquardt_gmm",
        solver_diagnostics={"converged": True, "steps": steps, "iterations": iterations,
                            "jacobian": "analytic (forward-mode over the moment expressions)",
                            "criterion_history": step.history[-10:]},
        optimizer={"method": f"{estimator} GMM, Gauss-Newton with Levenberg-Marquardt damping",
                   "tolerance": tolerance, "max_iterations": max_iterations})


def gmm(*, data: Any, moments: Sequence[str] | dict[str, str],
        instruments: Sequence[str] | Sequence[Sequence[str]] | None = None,
        instrument_constant: bool = True, start: dict[str, float] | None = None,
        wmatrix: str | None = None, winitial: str = "identity", twostep: bool = True,
        igmm: bool = False, center: bool = False, covariance: str | None = None,
        cluster: str | None = None, lags: int | None = None, kernel: str | None = None,
        panel: str | None = None, time: str | None = None,
        categorical: Sequence[str] | None = None, weights: str | None = None,
        weight_type: str | None = None, tolerance: float = 1e-8, max_iterations: int = 500,
        igmm_tolerance: float = 1e-10, igmm_max_iterations: int = 1000,
        missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Generalized method of moments for moment expressions (Stata ``gmm``; EViews GMM).

    Model: moment conditions ``E[z_ij u_j(x_i; b)] = 0`` for one or more
    residual expressions ``u_j`` and their instruments ``z_ij``.

    Moment expressions use the grammar of ``oe.nl``: parameters in braces
    (``{b0}``, or ``{b0=1}`` with a starting value), column names, numbers,
    ``+ - * / ^``, parentheses and ``exp``, ``ln``/``log``, ``sqrt``, ``abs``,
    ``sin``, ``cos``, ``tan``, ``expit``, ``normal``, ``normalden``. A parameter
    used in several equations is one parameter. Expressions are parsed into a
    syntax tree and never executed as code; their derivatives are analytic.

    Estimation: ``b`` minimizes ``Q = g'Wg``, ``g = sum_i w_i m_i``,
    ``m_i = (z_i1 u_i1, ..., z_iM u_iM)``, by Gauss-Newton with
    Levenberg-Marquardt damping on the analytic Jacobian ``G = dg/db'``.

    - ``twostep=True`` (default): first step with ``winitial``, then
      ``W = S(b_1)^-1`` with ``S`` of type ``wmatrix``; ``twostep=False``:
      one-step estimator with ``winitial``; ``igmm=True``: iterate ``W``
      until the coefficients change by less than ``igmm_tolerance``.
    - ``winitial``: ``'identity'`` (the default, taken to be Stata's; see the
      uncertain conventions of docs/econometrics/systems.md) or ``'unadjusted'``
      (``blockdiag (Z_j'Z_j)^-1``: 2SLS for linear moments, as ``ivregress gmm``).
    - ``wmatrix``: ``'robust'`` (``sum w^2 m m'``, default), ``'cluster'``
      (needs ``cluster``), ``'hac'`` (needs ``lags``; ``kernel`` default
      Bartlett; ordered by ``time`` within ``panel`` when given) or
      ``'unadjusted'`` (``s_jk Z_j'Z_k``). ``center`` demeans the moments.

    Covariance (``covariance``; default: the weight-matrix type, as in
    Stata): with ``B = (G'WG)^-1``, the efficient form ``V = B`` when the
    estimator reweights and the covariance matches the weight matrix,
    otherwise the sandwich ``B G'W S W G B`` with ``S`` at the final
    estimates. ``'nonrobust'`` (Stata's ``unadjusted``), ``'robust'``,
    ``'cluster'`` (``G/(G-1)``), ``'hac'``. z statistics.

    Parameters
    ----------
    data : DataFrame, dict of columns or list of records.
    moments : residual expressions, e.g. ``["y - {b0} - {b1}*x1"]``, or a dict
        ``{label: expression}``; labels default to 1, 2, ...
    instruments : columns common to every equation, or one list per equation;
        the constant is added unless ``instrument_constant=False``.
    start : starting values ``{parameter: value}`` (default 0, or the
        ``{b=value}`` in the expressions).
    lags, kernel, panel, time : HAC settings. categorical : instruments to
        expand as dummies. weights, weight_type : ``'aweight'``, ``'fweight'``
        or ``'pweight'`` (pweights need a robust covariance).
    tolerance, max_iterations : Gauss-Newton convergence of each step.
    missing, alpha : as usual.

    Returns
    -------
    Errors: ``invalid_formula``, ``invalid_start``, ``underidentified``,
    ``not_identified`` (rank-deficient Jacobian, parameters named),
    ``singular_weight_matrix``, ``perfect_fit`` (residuals of a moment equation
    vanish), ``nonconvergence``.

    ResultBundle whose terms are the parameter names; ``metrics``: ``j``
    (Hansen's J, two-step and iterated estimators), ``criterion`` (Stata's
    ``e(Q)`` = J/N), ``n_moments``, ``n_parameters``, ``gmm_steps``,
    ``iterations``; ``tests['hansen_j']`` ``~ chi2(L - P)`` when
    overidentified; ``extra``: the moments, instrument terms per equation,
    starting values and settings.

    Stata: ``gmm (y - {b0} - {b1}*x1 - {b2}*x2), instruments(x1 z1 z2)
    winitial(unadjusted) wmatrix(robust)``.

    Example::

        import openecon as oe
        fit = oe.gmm(data=df, moments=["y - {b0} - {b1}*x1 - {b2}*x2"],
                     instruments=["x1", "z1", "z2"], winitial="unadjusted")
        print(fit.summary())
        print(fit.tests["hansen_j"])
    """
    from openecon.analysis import fit

    _, parsed = parse_moments(moments)
    lists = _instrument_lists(instruments, len(parsed))
    if not parsed[0].columns:
        raise AnalysisError("invalid_spec", "The first moment expression must use at least one "
                            "data column (it names the model's outcome).")
    if covariance == "unadjusted":
        covariance = "nonrobust"
    if covariance is None and not cluster:
        # Stata: the covariance defaults to the type of the weight matrix (robust).
        covariance = {"unadjusted": "nonrobust", "hac": "hac"}.get(wmatrix or "robust")
    if isinstance(moments, dict):
        moment_option: Any = dict(moments)
    else:
        moment_option = list(moments)
    instrument_option = None if instruments is None else (
        list(instruments) if all(isinstance(i, str) for i in instruments)
        else [list(block) for block in instruments])
    spec = make_spec(
        "gmm", outcome=parsed[0].columns[0], panel=panel, time=time, covariance=covariance,
        cluster=cluster, categorical=column_list(categorical, "categorical"), weights=weights,
        weight_type=weight_type, missing=missing, alpha=alpha,
        columns={"system": moment_columns(parsed, lists)},
        options={"moments": moment_option, "instruments": instrument_option,
                 "instrument_constant": None if instrument_constant else False,
                 "start": start, "wmatrix": wmatrix,
                 "winitial": None if winitial == "identity" else winitial,
                 "twostep": None if twostep else False, "igmm": igmm or None,
                 "center": center or None, "lags": lags, "kernel": kernel,
                 "tolerance": None if tolerance == 1e-8 else tolerance,
                 "max_iterations": None if max_iterations == 500 else max_iterations,
                 "igmm_tolerance": None if igmm_tolerance == 1e-10 else igmm_tolerance,
                 "igmm_max_iterations": None if igmm_max_iterations == 1000
                 else igmm_max_iterations})
    return fit(spec, data=data)
