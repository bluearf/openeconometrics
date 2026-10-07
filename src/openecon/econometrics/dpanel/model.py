"""Assembly and estimation of the dynamic panel GMM model (analysis layer).

``read_options`` turns the spec into validated settings and the instrument
groups, ``assemble`` builds the level observations, the transformed and level
equations and the stacked instrument matrix, and ``estimate`` runs one- and
two-step GMM with every covariance and specification test. ``xtdpd.py``
turns the outcome into a ``ResultBundle``.

Conventions (see ``docs/econometrics/dpanel.md``), taken from xtabond2's Mata code:

- Regressors ``L1.y .. Lp.y``, ``x`` and optional period dummies (base: the
  first period with a level observation). The constant exists only in the
  level equation of system GMM (it differences out of the transformed
  equation), as in xtabond2; it is then also an IV-style instrument of the
  level equation. With a constant, x and y are centred at their level-equation
  means internally (exact reparametrization, mapped back by ``Model.report``).
- ``sigma2 = e1'e1 / (c wttot)``: one-step residuals of the transformed
  equation (of the level equation when h = 1 in system GMM), ``c = 2`` for
  first differences unless h = 1, else 1; ``wttot`` = transformed observations
  in system GMM with h > 1, reported observations otherwise.
- Sargan = minimized one-step criterion / sigma2; Hansen = minimized two-step
  criterion (also reported after one-step robust estimation).
- small: one-step nonrobust ``V wttot/(wttot-k)``, df ``N - k``; otherwise
  ``V G/(G-1) (N-1)/(N-k)``, df ``G - (constant)``; sigma2 times
  ``wttot/(wttot-k)``. The AR tests use the covariance before the factor.
- Difference-in-Sargan/Hansen tests for an instrument subset re-estimate the
  model without the subset with the corresponding block of the full model's
  moment matrix (``sum g_i g_i'``, or ``sigma2 sum Z_i'HZ_i`` after one-step
  nonrobust estimation), as xtabond2 does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import Design, ModelFrame, kernel_call, wald_test
from openecon.econometrics.dpanel import kernels, structure
from openecon.econometrics.dpanel.instruments import (
    GmmGroup, Instruments, IvGroup, Rows, build, parse_gmm, parse_iv,
)
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import chi2_sf
from openecon.models import ModelSpec


@dataclass
class Options:
    lags: int
    system: bool
    twostep: bool
    orthogonal: bool
    small: bool
    time_dummies: bool
    h: int
    artests: int
    robust: bool
    constant: bool
    gmm: list[GmmGroup]
    iv: list[IvGroup]
    columns: list[str] = field(default_factory=list)


def read_options(spec: ModelSpec) -> Options:
    """Validated settings and instrument groups of an ``xtdpd`` spec."""
    info = registry.get(spec.estimator)

    def option(name: str) -> Any:
        return spec.options[name] if name in spec.options else info.option(name).default

    system, lags = bool(option("system")), int(option("lags"))
    collapse = bool(option("collapse"))
    gmm_value, iv_value = option("gmm"), option("iv")
    if gmm_value is None:
        gmm_value = [{"columns": [spec.outcome], "lags": [2, None]}] if lags >= 1 else []
    gmm = parse_gmm(gmm_value, system=system, collapse=collapse)
    if iv_value is None:
        # Regressors not instrumented GMM-style are taken as strictly exogenous.
        styled = {name for group in gmm for name in group.columns}
        exogenous = [name for name in spec.predictors if name not in styled]
        iv_value = [{"columns": exogenous, "equation": "diff"}] if exogenous else []
    iv = parse_iv(iv_value, system=system)
    if not spec.predictors and lags == 0:
        raise AnalysisError("invalid_spec", "The model has no regressors: give x=[...] or "
                            "lags >= 1.")
    reserved = {spec.panel, spec.time}
    columns = []
    for group in [*gmm, *iv]:
        for name in group.columns:
            if name in reserved:
                raise AnalysisError("invalid_spec", f"The panel or time variable '{name}' cannot "
                                    "be an instrument.")
            columns.append(name)
    return Options(
        lags=lags, system=system, twostep=bool(option("twostep")),
        orthogonal=bool(option("orthogonal")), small=bool(option("small")),
        time_dummies=bool(option("time_dummies")), h=int(option("h")),
        artests=int(option("artests")), robust=spec.covariance == "robust",
        constant=bool(spec.intercept) and system, gmm=gmm, iv=iv,
        columns=list(dict.fromkeys(columns)),
    )


@dataclass
class Model:
    """The assembled estimation problem."""

    frame: ModelFrame
    grid: structure.Layout
    obs: structure.Levels
    transform: Any                       # Differences or OrthogonalDeviations
    x_levels: Tensor                     # [NL, k] regressors on the level observations
    y_levels: Tensor                     # [NL]
    terms: list[str]
    x: Tensor                            # [R, k] stacked regressors
    y: Tensor                            # [R]
    codes: Tensor                        # [R] panel of each stacked row
    instruments: Instruments
    n_t: int                             # transformed rows (the first n_t stacked rows)
    system: bool
    collinear_instruments: int = 0
    constant: int | None = None          # index of the Intercept in terms
    # With a constant, x and y are centred at their level-equation means (shift, y_shift) in the
    # level rows: beta_reported = J beta with J = I - e_c shift', plus y_shift on the constant.
    shift: Tensor | None = None
    y_shift: float = 0.0

    def report(self, beta: Tensor, covariance: Tensor) -> tuple[Tensor, Tensor]:
        """Coefficients and covariance of the uncentred regressors (exact linear map)."""
        if self.shift is None or self.constant is None:
            return beta, covariance
        jac = torch.eye(len(beta), dtype=torch.float64)
        jac[self.constant] -= self.shift
        reported = jac @ beta
        reported[self.constant] += self.y_shift
        return reported, jac @ covariance @ jac.T


def _period_label(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, (int, str)):
        return value
    return str(value)


def _dummies(frame: ModelFrame, obs: structure.Levels) -> tuple[Tensor, list[str]]:
    periods = torch.unique(obs.period).tolist()
    values = frame.sample[frame.spec.time]
    columns, names = [], []
    for period in periods[1:]:
        mask = obs.period == period
        first = int(obs.rows[mask.nonzero()[0, 0]])
        columns.append(mask.to(torch.float64))
        names.append(f"{frame.spec.time}[{_period_label(values.iloc[first])}]")
    if not columns:
        return torch.empty((obs.n, 0), dtype=torch.float64), []
    return torch.stack(columns, dim=1), names


def assemble(frame: ModelFrame, opts: Options) -> Model:
    spec = frame.spec
    codes, groups = frame.codes(spec.panel)
    time = frame.time_index()
    period = time - time.min()
    grid = kernel_call(structure.layout, codes, period, groups)
    obs = structure.levels(grid, opts.lags)
    if obs.n == 0:
        raise AnalysisError("insufficient_observations",
                            f"No observation has its {opts.lags} lagged dependent variable(s) "
                            "available: every panel is too short or has gaps.")
    transform = (structure.OrthogonalDeviations(obs) if opts.orthogonal
                 else structure.Differences(grid, obs))
    if transform.n == 0:
        what = ("a later observation" if opts.orthogonal
                else "an observation in the previous period")
        raise AnalysisError("insufficient_observations",
                            f"No level observation has {what}, so the transformed equation is "
                            "empty. Dynamic panel GMM needs panels with at least "
                            f"{opts.lags + 2} consecutive periods.")
    cache: dict[str, Tensor] = {}

    def column(name: str) -> Tensor:
        if name not in cache:
            cache[name] = frame.numeric(name)
        return cache[name]

    y0 = column(spec.outcome)
    pieces, terms = [], []
    if opts.constant:
        pieces.append(torch.ones((obs.n, 1), dtype=torch.float64))
        terms.append("Intercept")
    for lag in range(1, opts.lags + 1):
        pieces.append(y0[grid.rows_at(obs.codes, obs.period - lag)][:, None])
        terms.append(f"L{lag}.{spec.outcome}")
    for name in spec.predictors:
        pieces.append(column(name)[obs.rows][:, None])
        terms.append(name)
    dummies, dummy_names = (_dummies(frame, obs) if opts.time_dummies
                            else (torch.empty((obs.n, 0), dtype=torch.float64), []))
    pieces.append(dummies)
    terms.extend(dummy_names)
    x_levels = torch.cat(pieces, dim=1)
    y_levels = y0[obs.rows]
    x_t, y_t = transform.apply(x_levels), transform.apply(y_levels)
    if opts.system:
        x, y = torch.cat([x_t, x_levels]), torch.cat([y_t, y_levels])
        codes_stacked = torch.cat([transform.codes, obs.codes])
    else:
        x, y, codes_stacked = x_t, y_t, transform.codes
    shift, y_shift = None, 0.0
    if opts.constant:
        # Centre the level rows at their means (the constant is 0 in the transformed rows, so
        # those are untouched): the collinearity screen then partials out the constant as
        # xtabond2's _rmcoll does, and a regressor or outcome with a large level (y around
        # 1e8) neither looks like the constant nor costs digits in the GMM steps. The exact
        # map back to the uncentred coefficients is Model.report.
        shift = x_levels.mean(dim=0)
        shift[0] = 0.0
        y_shift = float(y_levels.mean())
        x_levels, y_levels = x_levels - shift, y_levels - y_shift
        x, y = torch.cat([x[:transform.n], x_levels]), torch.cat([y[:transform.n], y_levels])
    design = frame.drop_collinear(Design(x, terms, {}, False))
    if not design.terms:
        raise AnalysisError("invalid_spec", "Every regressor is collinear (or zero after the "
                            "transformation); nothing can be estimated.")
    kept = [terms.index(term) for term in design.terms]
    x_levels, x = x_levels[:, kept], x[:, kept]
    shift = None if shift is None else shift[kept]
    terms = design.terms
    rows = Rows(transform.codes, transform.period, transform.source, obs.codes, obs.period,
                opts.system)

    def lookup(name: str, cells_codes: Tensor, cells_period: Tensor) -> tuple[Tensor, Tensor]:
        return grid.values_at(column(name), cells_codes, cells_period)

    extra = []
    if dummy_names:
        extra.append(("time", "time dummies", "level" if opts.system else "diff", dummies))
    if opts.constant:
        extra.append(("constant", "_cons", "level", torch.ones((obs.n, 1), dtype=torch.float64)))
    instruments = build(rows=rows, gmm=opts.gmm, iv=opts.iv, lookup=lookup,
                        level_values=lambda name: column(name)[obs.rows],
                        transform=transform.apply, extra_iv=extra)
    kept_z, omitted_z = kernel_call(kernels.instrument_rank, instruments.z)
    if omitted_z:
        instruments.z = instruments.z[:, kept_z]
        instruments.groups = [instruments.groups[j] for j in kept_z]
        instruments.equations = [instruments.equations[j] for j in kept_z]
    return Model(frame, grid, obs, transform, x_levels, y_levels, terms, x, y, codes_stacked,
                 instruments, transform.n, opts.system, len(omitted_z),
                 terms.index("Intercept") if "Intercept" in terms else None, shift, y_shift)


def one_step_factor(model: Model, h: int) -> list[Tensor]:
    """Row blocks B with ``B'B = sum_i Z_i' H Z_i`` for xtabond2's h(1), h(2) or h(3).

    With N the map from level errors to the stacked rows (``[M; I]`` in system
    GMM), h(3) is ``H = N N'`` so ``B = N'Z = M'Z_T + Z_L``; h(2) drops the cross
    blocks (``B = [M'Z_T; Z_L]``) and h(1) is ``H = I`` (``B = Z``). With
    orthogonal deviations, h(3) uses xtabond2's full-span deviation matrix for
    the cross block (``structure.OrthogonalDeviations.grid_adjoint``).
    """
    z, n_t = model.instruments.z, model.n_t
    z_t, z_l = z[:n_t], z[n_t:]
    if h == 1:
        return [z]
    if model.system and h == 3 and isinstance(model.transform, structure.OrthogonalDeviations):
        # xtabond2 takes the transformed-level block from the deviation matrix of a panel
        # spanning all periods (F_g); F_g F_g' = I, so B = F_g'Z_T + Z_L on every grid cell.
        image, absent = model.transform.grid_adjoint(z_t, model.grid.span)
        return [image + z_l, absent]
    image = model.transform.adjoint(z_t)
    if not model.system:
        return [image]
    if h == 2:
        return [image, z_l]
    image += z_l
    return [image]


@dataclass
class Estimate:
    beta: Tensor
    covariance: Tensor
    info: dict[str, Any]
    tests: dict[str, Any]
    metrics: dict[str, Any]
    extra: dict[str, Any]
    df_inference: float | None
    fitted: Tensor
    observed: Tensor
    sample_rows: Tensor          # frame rows of the reported equation


def _overid(statistic: float, df: int, label: str, note: str | None = None) -> dict:
    if df <= 0:
        return {"statistic": None, "df": df, "p_value": None, "distribution": "chi2",
                "label": label, "note": note or "the model is exactly identified"}
    statistic = max(statistic, 0.0)
    return {"statistic": statistic, "df": df, "p_value": chi2_sf(statistic, df),
            "distribution": "chi2", "label": label}


def _subsets(model: Model) -> dict[str, tuple[str, list[int]]]:
    """Instrument subsets of the difference-in-Sargan/Hansen tests: name -> (label, columns)."""
    inst = model.instruments
    subsets: dict[str, tuple[str, list[int]]] = {}
    for key, label in inst.labels.items():
        if key == "constant":
            continue
        members = [j for j, owner in enumerate(inst.groups) if owner == key]
        if members:
            subsets[key] = (label, members)
    if model.system:
        level = [j for j, (key, eq) in enumerate(zip(inst.groups, inst.equations, strict=True))
                 if key.startswith("gmm") and eq == "level"]
        if level:
            subsets["level"] = ("GMM instruments for levels", level)
    return subsets


def _difference_tests(model: Model, a: Tensor, b: Tensor, blocks: list[Tensor], full: float,
                      k: int, kind: str, scale: float) -> dict[str, Any]:
    """xtabond2's difference-in-Sargan/Hansen tests for every instrument subset.

    The model is re-estimated without the subset with the matching block of the
    full model's moment matrix (``blocks`` B with B'B = sum_i g_i g_i' after
    robust or two-step estimation, = sum_i Z_i'HZ_i after one-step nonrobust,
    where ``scale`` = sigma2 turns the criterion into a Sargan statistic); the
    difference with the full statistic has as many degrees of freedom as the
    subset has columns.
    """
    tests: dict[str, Any] = {}
    total = a.shape[0]
    title = kind.capitalize()
    for name, (label, members) in _subsets(model).items():
        dropped = set(members)
        rest = [j for j in range(total) if j not in dropped]
        infeasible = {"statistic": None, "df": len(members), "p_value": None,
                      "distribution": "chi2", "label": f"Difference-in-{title}: {label}"}
        if len(rest) < k or not members:
            tests[f"diff_{kind}_{name}"] = {
                **infeasible, "note": "the model is not identified without this subset"}
            continue
        try:
            root, rank = kernels.weight_root([block[:, rest] for block in blocks])
            restricted = kernels.gmm_step(a[rest], b[rest], root, rank)
        except KernelError as exc:            # the remaining instruments do not identify it
            tests[f"diff_{kind}_{name}"] = {**infeasible, "note": str(exc)}
            continue
        statistic = restricted.criterion / scale
        tests[f"{kind}_excl_{name}"] = _overid(
            statistic, len(rest) - k, f"{title} test excluding {label}")
        tests[f"diff_{kind}_{name}"] = _overid(
            full - statistic, len(members), f"Difference-in-{title} (null: exogenous): {label}")
    return tests


def _panel_counts(codes: Tensor, groups: int) -> tuple[int, int, float, int]:
    counts = torch.bincount(codes, minlength=groups)
    counts = counts[counts > 0]
    return len(counts), int(counts.min()), float(counts.double().mean()), int(counts.max())


def _ar_inputs(model: Model, opts: Options, beta: Tensor, covariance: Tensor, p: Tensor,
               residuals: Tensor, sigma2: float) -> kernels.ArInputs:
    """Inputs of the Arellano-Bond tests (always on differences of the level residuals)."""
    z, n_t, groups = model.instruments.z, model.n_t, model.grid.groups
    differences = isinstance(model.transform, structure.Differences)
    diffs = model.transform if differences else structure.Differences(model.grid, model.obs)
    robust = opts.robust or opts.twostep
    cross = align = None
    if not robust:
        if opts.h == 1:
            # H = I: pair each transformed row with the differenced row of the same date.
            cross = z[:n_t]
            if differences:
                align = torch.arange(n_t, dtype=torch.int64)
            else:
                span = model.grid.span
                table = torch.full((groups * (span + 1),), -1, dtype=torch.int64)
                table[diffs.codes * (span + 1) + diffs.period] = torch.arange(
                    diffs.n, dtype=torch.int64)
                align = table[model.transform.codes * (span + 1) + model.transform.period]
        elif differences:
            cross = model.transform.adjoint(z[:n_t])
        else:   # xtabond2: covariance with the full-span deviation matrix F_g (see H)
            cross = model.transform.grid_adjoint(z[:n_t], model.grid.span)[0]
    resid_levels = model.y_levels - model.x_levels @ beta
    return kernels.ArInputs(
        resid_diff=diffs.apply(resid_levels), x_diff=diffs.apply(model.x_levels),
        codes_diff=diffs.codes, period_diff=diffs.period, groups=groups,
        span=model.grid.span, covariance=covariance, p=p,
        moments=kernels.panel_moments(z, residuals, model.codes, groups) if robust else None,
        sigma2=None if robust else sigma2, cross=cross, adjoint=diffs, align=align)


def estimate(model: Model, opts: Options) -> Estimate:
    frame, inst = model.frame, model.instruments
    x, y, z, codes, groups = model.x, model.y, inst.z, model.codes, model.grid.groups
    n_t, k, n_inst = model.n_t, x.shape[1], z.shape[1]
    differences = isinstance(model.transform, structure.Differences)
    if opts.system:
        sample_rows, sample_codes = model.obs.rows, model.obs.codes
    else:
        sample_rows, sample_codes = model.obs.rows[model.transform.source], model.transform.codes
    n_groups, smallest, average, largest = _panel_counts(sample_codes, groups)
    nobs = len(sample_rows)
    if n_inst < k:
        raise AnalysisError("underidentified",
                            f"{n_inst} instrument(s) for {k} coefficient(s): the order condition "
                            "fails. Add GMM lags or IV-style instruments, or remove regressors.")
    onestep_nonrobust = not opts.twostep and not opts.robust
    # xtabond2's sigma2 = e'e / (c wttot): transformed residuals (level residuals for h = 1 in
    # system GMM), c = 2 for first differences unless h = 1, wttot = transformed rows in system
    # GMM with h > 1 and the reported observations otherwise.
    level_errors = opts.system and opts.h == 1
    factor_c = 2.0 if differences and opts.h != 1 else 1.0
    wttot = n_t if (opts.system and opts.h > 1) else nobs

    def sigma2_of(resid: Tensor) -> float:
        part = resid[n_t:] if level_errors else resid[:n_t]
        return float(part @ part) / (factor_c * wttot)

    a, b = z.T @ x, z.T @ y
    blocks1 = one_step_factor(model, opts.h)
    root1, rank1 = kernel_call(kernels.weight_root, blocks1)
    one = kernel_call(kernels.gmm_step, a, b, root1, rank1)
    e1 = y - x @ one.beta
    g1 = kernels.panel_moments(z, e1, codes, groups)
    ssr_t = float(e1[:n_t] @ e1[:n_t])
    # Residuals at the rounding level of the transformed outcome: nothing left to explain.
    if ssr_t <= 1e-28 * max(float(y[:n_t] @ y[:n_t]), float(model.y_levels @ model.y_levels)):
        raise AnalysisError("perfect_fit", f"The regressors explain the transformed "
                            f"'{frame.spec.outcome}' exactly (or it does not vary within "
                            "panels), so standard errors and tests are undefined.")
    sigma2 = sigma2_of(e1)             # one-step sigma2 (Sargan, nonrobust one-step covariance)
    if not sigma2 > 0:
        raise AnalysisError("perfect_fit", "The residuals used to estimate the error variance "
                            "are all zero, so standard errors and tests are undefined.")
    # Instruments counted as xtabond2 does: columns less the rank deficiency of sum Z_i'HZ_i.
    n_effective = min(n_inst, rank1)
    p1 = kernels.projector(one, a)
    v1_robust = kernels.sandwich(p1, g1)
    warnings: list[str] = []
    two = None
    hansen_note = None
    if not onestep_nonrobust:
        if n_groups < 2:
            raise AnalysisError("insufficient_clusters", "The robust covariance and the two-step "
                                "weight matrix are built from per-panel moments and need at "
                                "least two panels; use the one-step nonrobust estimator.")
        root2, rank2 = kernel_call(kernels.weight_root, [g1])
        if rank2 < n_inst:
            warnings.append(f"The two-step weight matrix is singular (rank {rank2} of "
                            f"{n_inst}); a generalized inverse is used. Reduce the number of "
                            "instruments (collapse=True or fewer lags).")
        if rank2 < k:
            message = (f"The two-step weight matrix has rank {rank2} (at most the number of "
                       f"panels, {n_groups}) but {k} coefficients are estimated")
            if opts.twostep:
                raise AnalysisError("underidentified", f"{message}: two-step GMM is not "
                                    "identified. Use more panels or the one-step estimator.")
            hansen_note = f"{message}, so the two-step Hansen statistic is not available."
            warnings.append(hansen_note)
        else:
            two = kernel_call(kernels.gmm_step, a, b, root2, rank2)
    if rank1 < n_inst:
        warnings.append(f"The one-step weight matrix Z'HZ is singular (rank {rank1} of "
                        f"{n_inst}); a generalized inverse is used.")
    if n_effective > n_groups:
        warnings.append(f"The number of instruments ({n_effective}) exceeds the number of panels "
                        f"({n_groups}): the Hansen test is weakened and two-step estimates may "
                        "be biased. Consider collapse=True or capping the GMM lags.")
    info: dict[str, Any] = {"steps": 2 if opts.twostep else 1}
    if not opts.twostep:
        final, p_final = one, p1
        if opts.robust:
            covariance = v1_robust
            info["correction"] = "one-step robust sandwich clustered on the panel (no factor)"
        else:
            covariance = sigma2 * one.bread
            info["correction"] = (f"one-step: sigma2 (X'Z W1 Z'X)^-1, sigma2 = e'e / "
                                  f"({factor_c:g} N) from the "
                                  f"{'level' if level_errors else 'transformed'}-equation "
                                  "residuals (xtabond2)")
    else:
        final = two
        p_final = kernels.projector(two, a)
        if opts.robust:
            covariance, _ = kernels.windmeijer(z=z, x=x, codes=codes, groups=groups, a=a, b=b,
                                               moments_one=g1, two=two, v_one=v1_robust)
            info["correction"] = "two-step robust: Windmeijer (2005) finite-sample correction"
        else:
            covariance = two.bread
            info["correction"] = "two-step: (X'Z W2 Z'X)^-1"
    beta = final.beta
    e_final = y - x @ beta
    sigma2_final = sigma2_of(e_final) if opts.twostep else sigma2
    # AR tests use the covariance before the small-sample factor and sigma2 after it (xtabond2).
    covariance_ar, sigma2_ar = covariance, sigma2
    df_inference = None
    if opts.small:
        tmp = wttot / (wttot - k) if wttot > k else math.inf
        if onestep_nonrobust:
            if wttot <= k or nobs <= k:
                raise AnalysisError("insufficient_observations", "small=True needs more "
                                    "observations than coefficients.")
            factor, df_inference = tmp, nobs - k
            info["correction"] += f"; small: N/(N-k) with N = {wttot}"
        else:
            df_inference = n_groups - (1 if model.constant is not None else 0)
            if n_groups < 2 or nobs <= k or wttot <= k or df_inference < 1:
                raise AnalysisError("insufficient_observations", "small=True with robust or "
                                    "two-step covariance needs at least two panels and more "
                                    "observations than coefficients.")
            factor = n_groups / (n_groups - 1) * (nobs - 1) / (nobs - k)
            info["correction"] += "; small: G/(G-1) (N-1)/(N-k)"
        covariance = covariance * factor
        sigma2_ar *= tmp
        sigma2_final *= tmp
        info["small_sample_correction"] = factor
    info["df_inference"] = df_inference
    if opts.robust:
        info.update({"cluster_count": n_groups, "cluster_column": frame.spec.panel,
                     "cluster_df": n_groups - 1 if df_inference is None else df_inference})
    tests: dict[str, Any] = {}
    ar = _ar_inputs(model, opts, beta, covariance_ar, p_final, e_final, sigma2_ar)
    for order in range(1, opts.artests + 1):
        tests[f"ar{order}"] = kernels.arellano_bond(ar, order)
    over = n_effective - k
    sargan_label = "Sargan test of overidentifying restrictions (not robust)"
    sargan_value = one.criterion / sigma2
    tests["sargan"] = _overid(sargan_value, over, sargan_label)
    if two is not None:
        tests["hansen"] = _overid(two.criterion, over,
                                  "Hansen test of overidentifying restrictions (robust)")
        if over > 0:
            tests.update(_difference_tests(model, a, b, [g1], two.criterion, k, "hansen", 1.0))
    elif hansen_note is not None:
        tests["hansen"] = {"statistic": None, "df": over, "p_value": None, "distribution": "chi2",
                           "label": "Hansen test of overidentifying restrictions (robust)",
                           "note": hansen_note}
    elif over > 0:
        tests.update(_difference_tests(model, a, b, blocks1, sargan_value, k, "sargan", sigma2))
    beta, covariance = model.report(beta, covariance)
    slopes = [j for j in range(k) if j != model.constant]
    tests["model"] = wald_test(beta, covariance, slopes, df_resid=df_inference,
                               label="Wald test of all coefficients except the constant")
    metrics = {
        "n_groups": n_groups, "n_instruments": n_effective, "obs_per_group_min": smallest,
        "obs_per_group_avg": average, "obs_per_group_max": largest,
        "df_model": len(slopes), "sigma_e": math.sqrt(sigma2_final),
    }
    if df_inference is not None:
        metrics["df_resid"] = df_inference
    if opts.system:
        fitted = model.x_levels @ final.beta + model.y_shift
        observed = model.y_levels + model.y_shift
    else:
        fitted, observed = x @ final.beta, y
    extra = {
        "transformation": model.transform.name, "equations": "system" if opts.system else
        "difference", "steps": 2 if opts.twostep else 1, "h": opts.h,
        "n_obs_transformed": n_t, "n_obs_level": model.obs.n if opts.system else 0,
        "instrument_groups": {key: {"label": label,
                                    "columns": sum(1 for owner in inst.groups if owner == key)}
                              for key, label in inst.labels.items()},
        "instrument_columns": n_inst,
        "instruments_planned": inst.planned, "instruments_zero_dropped": inst.zero_columns,
        "instruments_collinear_dropped": model.collinear_instruments,
        "weight_matrix_rank": {"one_step": rank1, "two_step": two.rank if two else None},
        "sigma2_one_step": sigma2, "windmeijer": bool(opts.twostep and opts.robust),
        "constant": ("level equation" if model.constant is not None else
                     "differenced out" if not opts.system else "excluded"),
        "rows_lag_only": frame.n - nobs,
    }
    for message in warnings:
        frame.warn(message)
    return Estimate(beta, covariance, info, tests, metrics, extra, df_inference, fitted,
                    observed, sample_rows)
