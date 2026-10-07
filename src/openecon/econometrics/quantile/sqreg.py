"""Stata's bsqreg, sqreg and iqreg: quantile regressions with a bootstrap covariance.

Point estimates are those of ``qreg`` at each requested quantile. The covariance
is the pairs bootstrap of Stata's commands: ``reps`` samples of N observations
are drawn with replacement (whole clusters when a cluster column is given),
every quantile is re-estimated on each sample, and with the R successful
replicates b*_1..b*_R (all quantiles stacked)

    V = sum_r (b*_r - bbar*)(b*_r - bbar*)' / (R - 1),      bbar* = mean of the b*_r.

Estimating all quantiles on the same samples is what gives ``sqreg`` its
between-quantile covariance blocks and ``iqreg`` the variance of a difference.

A resample is represented by its multiplicity counts c_i: the distinct rows
drawn enter a weighted check loss with weights c_i, which has the same
minimizers as the replicated data, costs less and keeps the solver's vertices
non-degenerate. The generator is a ``torch.Generator`` seeded with ``seed``
(drawn and recorded when omitted), so results are reproducible. Coefficient
tests use Student t with N - K degrees of freedom, as Stata prints.
"""

from __future__ import annotations

import secrets
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, kernel_call, make_spec,
)
from openecon.econometrics.quantile import kernels
from openecon.econometrics.quantile.qreg import (
    check_quantile, quantile_sample, quantile_title, raw_deviations, require_variation,
    solver_record,
)
from openecon.engines.contracts import KernelError
from openecon.models import ModelSpec, ResultBundle


def quantile_label(tau: float) -> str:
    """Stata-style equation name of a quantile: 0.25 -> 'q25', 0.025 -> 'q2_5'."""
    return "q" + f"{100 * tau:.10g}".replace(".", "_")


def _quantile_list(frame: ModelFrame, *, exactly: int | None = None) -> list[float]:
    values = frame.option("quantiles")
    if not values:
        raise AnalysisError("invalid_quantile", "Give at least one quantile.")
    taus = [check_quantile(value, "quantiles") for value in values]
    if len(set(taus)) != len(taus):
        raise AnalysisError("invalid_quantile", "The quantiles must be distinct.")
    if exactly is not None and len(taus) != exactly:
        raise AnalysisError("invalid_quantile", f"Exactly {exactly} quantiles are needed, for "
                            "example quantiles=[0.25, 0.75].")
    return taus


def _bootstrap(frame: ModelFrame, x: Tensor, y: Tensor, taus: Sequence[float],
               ) -> tuple[Tensor, dict[str, Any]]:
    """Replicate coefficients [R, len(taus), k] of the pairs (or cluster) bootstrap."""
    spec = frame.spec
    reps = int(frame.option("reps"))
    seed = spec.options.get("seed")
    if seed is None:
        seed = secrets.randbelow(2 ** 31)
    generator = torch.Generator().manual_seed(int(seed))
    n, k = x.shape
    clusters = None
    if spec.cluster is not None:
        (codes, groups), = frame.cluster_dimensions()
        clusters = (codes, groups)
    draws = torch.empty((reps, len(taus), k), dtype=torch.float64)
    done = torch.zeros(reps, dtype=torch.bool)
    for rep in range(reps):
        if clusters is None:
            counts = torch.bincount(torch.randint(n, (n,), generator=generator), minlength=n)
        else:
            codes, groups = clusters
            counts = torch.bincount(torch.randint(groups, (groups,), generator=generator),
                                    minlength=groups)[codes]
        keep = counts > 0
        try:
            problem = kernels.prepare(x[keep], y[keep], counts[keep].to(torch.float64))
            for j, tau in enumerate(taus):
                draws[rep, j] = kernels.solve(problem, tau).beta
            done[rep] = True
        except KernelError:
            continue        # a resample without full rank: counted and reported below
    failed = reps - int(done.sum())
    if reps - failed < 2:
        raise AnalysisError("bootstrap_failed", "Fewer than two bootstrap samples could be "
                            "estimated: the regressors are collinear in almost every resample. "
                            "Drop sparse indicator variables or use more observations.")
    if failed:
        frame.warn(f"{failed} of {reps} bootstrap samples could not be estimated (collinear "
                   "regressors in the resample) and were left out of the covariance.")
    record = {"reps": reps, "reps_used": reps - failed, "seed": int(seed),
              "resampling": "clusters" if clusters is not None else "observations",
              "variance": "about the replicate mean, divisor R - 1"}
    if clusters is not None:
        record.update({"cluster_column": spec.cluster, "cluster_count": clusters[1]})
    return draws[done], record


def _covariance(draws: Tensor) -> Tensor:
    """sum_r (b_r - bbar)(b_r - bbar)' / (R - 1) for replicate rows [R, p]."""
    centered = draws - draws.mean(dim=0)
    return centered.T @ centered / (draws.shape[0] - 1)


def _inference(record: dict[str, Any], df: int) -> dict[str, Any]:
    unit = "cluster" if record["resampling"] == "clusters" else "pairs"
    info = {"covariance": "bootstrap", "df_inference": df,
            "correction": f"{unit} bootstrap, {record['reps_used']} replications, seed "
                          f"{record['seed']}; variance about the replicate mean / (R - 1)"}
    if record["resampling"] == "clusters":
        info.update({"cluster_count": record["cluster_count"],
                     "cluster_column": record["cluster_column"]})
    return info


def _point_fits(frame: ModelFrame, taus: Sequence[float]):
    """Shared setup: design, outcome, N and the full-sample fit at every quantile."""
    design, y, _, nobs = quantile_sample(frame)
    problem = kernel_call(kernels.prepare, design.x, y)
    fits, raws = [], []
    for tau in taus:
        fit = kernel_call(kernels.solve, problem, tau)
        raw = raw_deviations(y, tau, None)
        require_variation(fit, raw[1])
        fits.append(fit)
        raws.append(raw)
    if not all(fit.unique for fit in fits):
        frame.warn("A quantile-regression solution is not unique (several coefficient vectors "
                   "attain the minimum); one optimal vertex is reported.")
    return design, y, nobs, fits, raws


def _pseudo(fit: kernels.QuantileFit, raw: tuple[float, float]) -> float | None:
    return 1 - fit.objective / raw[1] if raw[1] > 0 else None


def fit_bsqreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``bsqreg``: one quantile, bootstrap covariance."""
    frame = ModelFrame(spec, data)
    tau = check_quantile(frame.option("quantile"))
    design, y, nobs, (fit,), (raw,) = _point_fits(frame, [tau])
    k = len(design.terms)
    draws, record = _bootstrap(frame, design.x, y, [tau])
    metrics = {"quantile": tau, "pseudo_r_squared": _pseudo(fit, raw), "sum_adev": fit.objective,
               "sum_rdev": raw[1], "raw_quantile": raw[0], "reps": record["reps_used"],
               "iterations": fit.iterations, "df_model": k - int(design.intercept),
               "df_resid": nobs - k}
    return build_result(
        frame, terms=design.terms, params=fit.beta, covariance=_covariance(draws[:, 0]),
        title=quantile_title(tau) + ", bootstrap standard errors", df_inference=nobs - k,
        df_resid=nobs - k, metrics=metrics, fitted=design.x @ fit.beta,
        solver="frisch_newton_interior_point", solver_diagnostics=solver_record(fit),
        inference=_inference(record, nobs - k), extra={"bootstrap": record},
        categories=design.categories, nobs=nobs, provenance={"bootstrap": record},
    )


def fit_sqreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``sqreg``: several quantiles with their joint bootstrap covariance."""
    frame = ModelFrame(spec, data)
    taus = _quantile_list(frame)
    labels = [quantile_label(tau) for tau in taus]
    if len(set(labels)) != len(labels):
        raise AnalysisError("invalid_quantile", "The quantiles are too close to be told apart.")
    design, y, nobs, fits, raws = _point_fits(frame, taus)
    k = len(design.terms)
    draws, record = _bootstrap(frame, design.x, y, taus)
    terms = [f"{label}:{term}" for label in labels for term in design.terms]
    equations = [label for label in labels for _ in design.terms]
    metrics: dict[str, Any] = {"reps": record["reps_used"]}
    for label, fit, raw in zip(labels, fits, raws, strict=True):
        metrics[f"pseudo_r_squared_{label}"] = _pseudo(fit, raw)
    metrics.update({"df_model": k - int(design.intercept), "df_resid": nobs - k})
    extra = {
        "quantiles": taus, "equations": labels, "bootstrap": record,
        "sum_adev": {label: fit.objective for label, fit in zip(labels, fits, strict=True)},
        "sum_rdev": {label: raw[1] for label, raw in zip(labels, raws, strict=True)},
        "raw_quantile": {label: raw[0] for label, raw in zip(labels, raws, strict=True)},
    }
    return build_result(
        frame, terms=terms, params=torch.cat([fit.beta for fit in fits]),
        covariance=_covariance(draws.reshape(draws.shape[0], -1)), equations=equations,
        title="Simultaneous quantile regression, bootstrap standard errors",
        df_inference=nobs - k, df_resid=nobs - k, metrics=metrics,
        solver="frisch_newton_interior_point",
        solver_diagnostics={label: solver_record(fit)
                            for label, fit in zip(labels, fits, strict=True)},
        inference=_inference(record, nobs - k), extra=extra, categories=design.categories,
        nobs=nobs, provenance={"bootstrap": record},
    )


def fit_iqreg(spec: ModelSpec, data: Any) -> ResultBundle:
    """Entry point of ``iqreg``: the difference between two quantile regressions."""
    frame = ModelFrame(spec, data)
    low, high = _quantile_list(frame, exactly=2)
    if not low < high:
        raise AnalysisError("invalid_quantile", "Give the lower quantile first, for example "
                            "quantiles=[0.25, 0.75].")
    design, y, nobs, fits, raws = _point_fits(frame, [low, high])
    k = len(design.terms)
    draws, record = _bootstrap(frame, design.x, y, [low, high])
    metrics = {"quantile_low": low, "quantile_high": high,
               "pseudo_r_squared_low": _pseudo(fits[0], raws[0]),
               "pseudo_r_squared_high": _pseudo(fits[1], raws[1]), "reps": record["reps_used"],
               "df_model": k - int(design.intercept), "df_resid": nobs - k}
    extra = {"quantiles": [low, high], "bootstrap": record,
             "coefficients_low": dict(zip(design.terms, fits[0].beta.tolist(), strict=True)),
             "coefficients_high": dict(zip(design.terms, fits[1].beta.tolist(), strict=True))}
    return build_result(
        frame, terms=design.terms, params=fits[1].beta - fits[0].beta,
        covariance=_covariance(draws[:, 1] - draws[:, 0]),
        title=f"{low:g}-{high:g} Interquantile regression, bootstrap standard errors",
        df_inference=nobs - k, df_resid=nobs - k, metrics=metrics,
        solver="frisch_newton_interior_point",
        solver_diagnostics={"low": solver_record(fits[0]), "high": solver_record(fits[1])},
        inference=_inference(record, nobs - k), extra=extra, categories=design.categories,
        nobs=nobs, provenance={"bootstrap": record},
    )


# ---- convenience ---------------------------------------------------------------------


def _spec(estimator: str, *, y: str, x: Any, cluster: str | None, categorical: Any,
          intercept: bool, missing: str, alpha: float, options: dict[str, Any]) -> ModelSpec:
    return make_spec(
        estimator, outcome=y, predictors=column_list(x, "x"), covariance="bootstrap",
        cluster=cluster, categorical=column_list(categorical, "categorical"),
        intercept=intercept, missing=missing, alpha=alpha, options=options,
    )


def _fractions(quantiles: Any, example: str) -> list[float]:
    """A list of quantiles from any non-string iterable (list, tuple, array)."""
    try:
        if isinstance(quantiles, (str, bytes)):
            raise TypeError
        values = [value.item() if hasattr(value, "item") else value for value in quantiles]
    except (TypeError, ValueError):
        raise AnalysisError("invalid_quantile", "quantiles must be a list of fractions, for "
                            f"example quantiles={example}.") from None
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               for value in values):
        raise AnalysisError("invalid_quantile", "quantiles must be a list of fractions, for "
                            f"example quantiles={example}.")
    return [float(value) for value in values]


def bsqreg(*, data: Any, y: str, x: Sequence[str] | None = None, quantile: float = 0.5,
           reps: int = 20, seed: int | None = None, cluster: str | None = None,
           categorical: Sequence[str] | None = None, intercept: bool = True,
           missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Quantile regression with bootstrap standard errors (Stata's ``bsqreg``).

    The coefficients are those of :func:`qreg` at ``quantile`` (the minimizer of
    sum_i rho_tau(y_i - x_i'b)). The covariance is the pairs bootstrap: ``reps``
    samples of N observations drawn with replacement, the quantile regression
    re-estimated on each,

        V = sum_r (b*_r - bbar*)(b*_r - bbar*)' / (R - 1).

    It makes no assumption about the error density, which is why Stata offers
    it next to ``qreg``'s analytic standard errors; with the default 20
    replications the standard errors are themselves noisy, so use more for
    reported results.

    Parameters: ``data``; ``y``; ``x`` regressors (``categorical`` names those to
    treatment-code, ``intercept=False`` drops the constant); ``quantile`` in
    (0, 1); ``reps`` >= 2 (default 20); ``seed`` a nonnegative integer (when
    omitted a seed is drawn and recorded in ``extra['bootstrap']['seed']``, so
    the result can be reproduced); ``cluster`` resamples whole clusters instead
    of observations (Stata: ``bootstrap, cluster(id): qreg``); ``missing``
    ``'raise'`` or ``'drop'``; ``alpha``. Weights are not allowed, as in Stata.

    Inference: Student t with N - K degrees of freedom. ``metrics``:
    ``quantile``, ``pseudo_r_squared``, ``sum_adev``, ``sum_rdev``,
    ``raw_quantile``, ``reps`` (replications used), ``iterations``.
    ``extra['bootstrap']`` records reps, seed and the resampling unit. A
    resample in which the regressors are collinear is skipped and reported in
    ``warnings``.

    Stata: ``bsqreg y x1 x2, quantile(.5) reps(200)``.

    Example::

        import openecon as oe
        fit = oe.bsqreg(data=df, y="price", x=["weight", "length"], reps=200, seed=1)
        print(fit.summary())
    """
    from openecon.analysis import fit

    spec = _spec("bsqreg", y=y, x=x, cluster=cluster, categorical=categorical,
                 intercept=intercept, missing=missing, alpha=alpha,
                 options={"quantile": quantile, "reps": reps, "seed": seed})
    return fit(spec, data=data)


def sqreg(*, data: Any, y: str, x: Sequence[str] | None = None,
          quantiles: Sequence[float] = (0.25, 0.5, 0.75), reps: int = 20,
          seed: int | None = None, cluster: str | None = None,
          categorical: Sequence[str] | None = None, intercept: bool = True,
          missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Simultaneous quantile regression (Stata's ``sqreg``).

    Estimates the quantile regressions Q_tau(y | x) = x'b(tau) for every tau in
    ``quantiles`` (each exactly as :func:`qreg`) and obtains ONE joint
    covariance matrix of all coefficients by bootstrapping: each of the ``reps``
    resamples is used for all quantiles, so ``covariance_matrix`` contains the
    between-quantile blocks needed to test, for example, whether a coefficient
    is the same at the 25th and the 75th percentile:

        V = sum_r (b*_r - bbar*)(b*_r - bbar*)' / (R - 1),   b* stacking all quantiles.

    Terms are named ``'<equation>:<term>'`` with one equation per quantile:
    ``q25:x1``, ``q25:Intercept``, ``q50:x1`` ... (0.025 gives ``q2_5``).

    Parameters: ``data``; ``y``; ``x``; ``quantiles`` distinct fractions in
    (0, 1) (default 0.25, 0.5, 0.75); ``reps`` >= 2 (default 20); ``seed``
    (drawn and recorded when omitted); ``cluster`` to resample whole clusters;
    ``categorical``; ``intercept``; ``missing``; ``alpha``. No weights, as in
    Stata.

    Inference: Student t with N - K degrees of freedom, K the number of
    coefficients per quantile. ``metrics``: ``reps`` and
    ``pseudo_r_squared_<equation>`` per quantile; ``extra``: the quantiles,
    equation names, ``sum_adev``/``sum_rdev``/``raw_quantile`` per equation and
    the bootstrap record.

    Stata: ``sqreg y x1 x2, quantiles(.25 .5 .75) reps(100)``; a cross-quantile
    test is ``test [q25]x1 = [q75]x1``.

    Example::

        import openecon as oe
        fit = oe.sqreg(data=df, y="price", x=["weight", "length"],
                       quantiles=[0.25, 0.5, 0.75], reps=100, seed=1)
        print(fit.summary())
    """
    from openecon.analysis import fit

    spec = _spec("sqreg", y=y, x=x, cluster=cluster, categorical=categorical,
                 intercept=intercept, missing=missing, alpha=alpha,
                 options={"quantiles": _fractions(quantiles, "[0.25, 0.5, 0.75]"), "reps": reps,
                          "seed": seed})
    return fit(spec, data=data)


def iqreg(*, data: Any, y: str, x: Sequence[str] | None = None,
          quantiles: Sequence[float] = (0.25, 0.75), reps: int = 20, seed: int | None = None,
          cluster: str | None = None, categorical: Sequence[str] | None = None,
          intercept: bool = True, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """Interquantile range regression (Stata's ``iqreg``).

    Reports the difference between two quantile regressions,

        b = b(tau_high) - b(tau_low),      default (0.25, 0.75): the interquartile range,

    so a coefficient says how a regressor changes the spread of the conditional
    distribution of ``y`` (zero slopes mean a pure location shift). Both
    regressions are estimated as :func:`qreg`; the covariance of the difference
    is the pairs bootstrap with both quantiles re-estimated on the same
    resamples, V = sum_r (d*_r - dbar*)(d*_r - dbar*)' / (R - 1).

    Parameters: ``data``; ``y``; ``x``; ``quantiles`` the lower and the upper
    quantile; ``reps`` >= 2 (default 20); ``seed``; ``cluster`` (resample
    clusters); ``categorical``; ``intercept``; ``missing``; ``alpha``. No
    weights, as in Stata.

    Inference: Student t with N - K degrees of freedom. ``metrics``:
    ``quantile_low``, ``quantile_high``, ``pseudo_r_squared_low``,
    ``pseudo_r_squared_high``, ``reps``. ``extra`` holds the coefficients of the
    two underlying regressions and the bootstrap record.

    Stata: ``iqreg y x1 x2, quantiles(.25 .75) reps(100)``.

    Example::

        import openecon as oe
        fit = oe.iqreg(data=df, y="price", x=["weight", "length"], reps=100, seed=1)
        print(fit.summary())
    """
    from openecon.analysis import fit

    spec = _spec("iqreg", y=y, x=x, cluster=cluster, categorical=categorical,
                 intercept=intercept, missing=missing, alpha=alpha,
                 options={"quantiles": _fractions(quantiles, "[0.25, 0.75]"), "reps": reps,
                          "seed": seed})
    return fit(spec, data=data)
