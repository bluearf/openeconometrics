"""Sample structure, weights and covariance dispatch shared by the panel estimators.

A ``PanelSample`` wraps the sorted ``ModelFrame`` with dense panel codes, the
panel lengths T_i and the estimation weights under Stata's conventions:
analytic and sampling weights are rescaled to sum to the number of rows,
frequency weights replicate rows (N = sum of weights, T_i = sum of weights in
the panel) and, for the fixed-effects model, weights must be constant within
panel. The random-effects models (``re``, ``mle``) and the between model take
no weights, as in Stata.

``covariance`` maps the family's covariance names onto the shared
``core.linear_covariance`` conventions:

    nonrobust        classical, SSR / df_resid of the regression actually run
                     (for the GLS model: rmse_gls^2 (X*'X*)^-1, Stata's e(rmse))
    robust           xtreg's vce(robust) IS vce(cluster panelvar): CR1 on the panel,
                     G/(G-1) * (N-1)/(N-K), inference with G-1 degrees of freedom
                     (the between model uses HC1 on its panel-mean regression)
    cluster          CR1 on the declared cluster column(s) with K = slopes + constant;
                     for the fixed- and random-effects models every cluster column
                     must nest the panels (Stata refuses otherwise: cluster_not_nested)
    driscoll_kraay   Hoechle's xtscc: HAC of the cross-sectional score sums with the
                     Bartlett (or chosen) kernel, lags floor(4 (T/100)^(2/9)) by
                     default, times T/(T-1) * (N-1)/(N-K), inference with T-1 df
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics import registry
from openecon.econometrics.core import ModelFrame, kernel_call, linear_covariance
from openecon.econometrics.panel import kernels
from openecon.engines import covariance as cov
from openecon.engines.absorb import group_means

# Relative within sum of squares below which a regressor has no within variation
# (the same threshold as ModelFrame.drop_absorbed: a relative length of about 3e-7).
WITHIN_TOL = 1e-13

# A sum of squares that is pure rounding: residuals (and within deviations) carry an
# absolute error of about eps times the level of y, so they floor at roughly eps^2
# (~5e-32) times the UNCENTERED sum w y^2. The guard sits ~2e3 eps^2 above that floor;
# it is the linear family's exact-fit criterion, so both families refuse the same fits.
EXACT_FIT = 1e-28


def center_slopes(x: Tensor, weights: Tensor | None = None) -> Tensor:
    """Subtract the (weighted) grand mean m from every slope column of ``[1, X]`` in place.

    Regressing on [1, X - m] instead of [1, X] is an exact reparameterization
    (Intercept_c = Intercept + m'b; identical slopes, residuals and fitted
    values) in which the constant is orthogonal to the slopes, so a regressor
    with a large offset no longer makes the design numerically collinear with
    the constant and the sandwich covariances do not cancel catastrophically.
    This is what Stata's regress does by sweeping the constant first. Returns
    m; ``uncenter_coefficients`` and the ``means`` argument of ``covariance``
    map the estimates and their covariance back to the original constant.
    """
    slopes = x[:, 1:]
    if weights is None:
        means = slopes.mean(dim=0)
    else:
        means = (weights[:, None] * slopes).sum(dim=0) / weights.sum()
    slopes -= means
    return means


def uncenter_coefficients(beta: Tensor, means: Tensor) -> Tensor:
    """``(Intercept_c, b)`` in the original units: Intercept = Intercept_c - m'b."""
    out = beta.clone()
    out[0] -= means @ beta[1:]
    return out


def uncenter_covariance(v: Tensor, means: Tensor) -> Tensor:
    """V = S V_c S' with S = [[1, -m'], [0, I]], the covariance of ``uncenter_coefficients``.

    Exact for every covariance estimator: a sandwich is equivariant under linear maps.
    """
    transform = torch.eye(v.shape[0], dtype=torch.float64)
    transform[0, 1:] = -means
    return transform @ v @ transform.T


def require_residual_variation(ssr: float, tss: float, scale: float, outcome: str,
                               what: str) -> None:
    """Refuse a fit whose residuals are rounding noise: standard errors would be meaningless.

    ``tss`` is the (weighted) centered sum of squares of the dependent variable in
    the sample ``what`` describes and ``scale`` the uncentered ``sum w y^2`` of the
    outcome's level, against which rounding is judged (``EXACT_FIT``). Raises
    ``constant_outcome`` when the outcome does not vary and ``perfect_fit`` when
    the residual sum of squares is rounding noise.
    """
    if tss <= EXACT_FIT * scale:
        raise AnalysisError("constant_outcome",
                            f"The outcome '{outcome}' has no variation in {what}; nothing can be "
                            "estimated.")
    if ssr <= EXACT_FIT * scale:
        raise AnalysisError("perfect_fit",
                            f"The regressors explain '{outcome}' exactly in {what} (the residual "
                            "sum of squares is rounding noise), so standard errors are "
                            "undefined. Remove the regressor that reproduces the outcome.")


@dataclass
class PanelSample:
    """The sorted estimation sample with its panel structure and Stata-style weights."""

    frame: ModelFrame
    codes: Tensor                 # [N] dense panel codes, nondecreasing after sort_panel
    n: int                        # number of panels
    weights: Tensor | None        # estimation weights (aweights rescaled to sum N)
    frequency: Tensor | None      # row multiplicities under frequency weights
    sizes: Tensor                 # [n] T_i (sum of frequency weights per panel)
    nobs: int                     # N as the estimator counts it

    @property
    def spec(self):
        return self.frame.spec

    def structure(self) -> dict[str, float]:
        """Stata's header: number of groups and observations per group."""
        return {"n_groups": self.n, "t_min": float(self.sizes.min()),
                "t_avg": self.nobs / self.n, "t_max": float(self.sizes.max())}

    def first_rows(self) -> Tensor:
        """Row index of the first observation of every panel."""
        codes = self.codes
        starts = (codes[1:] != codes[:-1]).nonzero().flatten() + 1
        return torch.cat([torch.zeros(1, dtype=torch.int64), starts])

    def nests_panels(self, codes: Tensor) -> bool:
        """Whether the group codes of a column are constant inside every panel."""
        return bool(kernel_call(kernels.constant_within_panel, codes.to(torch.float64),
                                self.codes, self.n))

    def panel_level(self, name: str) -> tuple[Tensor, int]:
        """Codes of a column that must be constant within panel, one per panel."""
        codes, count = self.frame.codes(name)
        if not self.nests_panels(codes):
            raise AnalysisError("cluster_varies_within_panel",
                                f"Cluster column '{name}' must be constant within each panel "
                                "for the between estimator.")
        return codes[self.first_rows()], count


def panel_sample(frame: ModelFrame, *, constant_weights: bool) -> PanelSample:
    """Panel codes, lengths and validated weights of the (sorted, restricted) frame."""
    spec = frame.spec
    codes, n = frame.codes(spec.panel)
    weights = frame.weights()
    frequency = None
    if weights is not None:
        if spec.weight_type == "pweight" and spec.covariance == "nonrobust":
            raise AnalysisError("unsupported_covariance",
                                "Sampling weights (pweight) require a robust covariance; choose "
                                "covariance='robust', 'cluster' or 'driscoll_kraay'.")
        if constant_weights and not kernel_call(kernels.constant_within_panel, weights, codes, n):
            raise AnalysisError("weights_vary_within_panel",
                                "Weights must be constant within each panel for the "
                                "fixed-effects model, as in Stata's xtreg. Aggregate the "
                                "weight to the panel level or use model='pooled'.")
        if spec.weight_type == "fweight":
            frequency = weights
        else:
            # Stata rescales analytic weights to sum to the number of observations.
            weights = weights * (frame.n / float(weights.sum()))
    sizes = kernel_call(kernels.panel_sizes, codes, n, frequency)
    return PanelSample(frame, codes, n, weights, frequency, sizes, int(round(float(sizes.sum()))))


def covariance(sample: PanelSample, *, x: Tensor, resid: Tensor, bread: Tensor, nobs: int,
               k: int, df_resid: float, weights: Tensor | None, means: Tensor,
               nested: bool = False, robust: str = "panel", panel_level: bool = False,
               ) -> tuple[Tensor, dict[str, Any], float]:
    """Covariance of a least-squares panel fit under ``spec.covariance``.

    ``x``, ``resid`` and ``bread`` come from the regression on centered slopes
    (``center_slopes`` returned ``means``); the covariance is computed there,
    where the sandwich is well conditioned, and returned in the ORIGINAL units
    (``uncenter_covariance``). ``k`` counts the slopes and the constant of the
    regression actually run (the K of every small-sample factor). ``nested``
    enforces Stata's rule for the fixed- and random-effects models that every
    cluster column nests the panels (``cluster_not_nested`` otherwise).
    ``robust`` is "panel" (cluster on the panel) or "HC1" (the between
    regression). ``panel_level`` maps cluster columns to one code per panel
    (the between regression has one row per panel).

    One estimator is not invariant to the centering: the Cameron-Gelbach-Miller
    eigenvalue fix of an indefinite multiway meat. When it applies, the sandwich
    is recomputed on the uncentered design (for fe: Stata's transformed design
    with the grand means added back), which is what ``core.linear_covariance``
    receives from an estimator that does not center, so the repaired matrix
    does not depend on this family's internal parameterization.

    Returns the covariance, the inference record and the inference df.
    """
    frame, spec = sample.frame, sample.spec
    kind = spec.covariance
    if kind == "nonrobust":
        v, info = linear_covariance(frame, x=x, resid=resid, bread=bread, n=nobs, k=k,
                                    df_resid=df_resid, weights=weights, kind="nonrobust")
        return uncenter_covariance(v, means), info, df_resid
    if kind == "driscoll_kraay":
        v, info, df = driscoll_kraay(sample, x=x, resid=resid, bread=bread, nobs=nobs, k=k,
                                     weights=weights)
        return uncenter_covariance(v, means), info, df
    if kind == "robust" and robust == "HC1":
        v, info = linear_covariance(frame, x=x, resid=resid, bread=bread, n=nobs, k=k,
                                    df_resid=df_resid, weights=weights, kind="HC1")
        info["covariance"] = "robust"
        return uncenter_covariance(v, means), info, df_resid
    if kind == "robust":
        dimensions, names = [(sample.codes, sample.n)], [spec.panel]
        if sample.n < 30:
            frame.warn(f"Only {sample.n} panels; cluster-robust inference on the panel "
                       "variable may be unreliable.")
    else:
        names = registry.cluster_columns(spec)
        if panel_level:
            dimensions = [sample.panel_level(name) for name in names]
            for (_, count), name in zip(dimensions, names, strict=True):
                if count < 2:
                    raise AnalysisError("insufficient_clusters",
                                        "Cluster covariance requires at least two clusters.")
                if count < 30:
                    frame.warn(f"Only {count} clusters in '{name}'; cluster-robust inference "
                               "may be unreliable.")
        else:
            if nested:
                for name in names:
                    if not sample.nests_panels(frame.codes(name)[0]):
                        raise AnalysisError(
                            "cluster_not_nested",
                            f"Panels are not nested within the cluster column '{name}'. Stata's "
                            "xtreg, fe and xtreg, re require every panel to lie inside a single "
                            "cluster (also for each column of a multiway cluster). Cluster on a "
                            "column that is constant within panels, or use model='pooled' "
                            "(regress), which accepts any cluster column.")
            dimensions = frame.cluster_dimensions()
    v, info = linear_covariance(frame, x=x, resid=resid, bread=bread, n=nobs, k=k,
                                df_resid=df_resid, weights=weights, kind="cluster",
                                clusters=dimensions, cluster_names=names)
    if info.get("psd_adjusted"):
        # X = X_c S^-1: column j regains m_j times the (transformed) constant column.
        original = x.clone()
        original[:, 1:] += x[:, :1] * means
        v, info = linear_covariance(frame, x=original, resid=resid,
                                    bread=uncenter_covariance(bread, means), n=nobs, k=k,
                                    df_resid=df_resid, weights=weights, kind="cluster",
                                    clusters=dimensions, cluster_names=names)
    else:
        v = uncenter_covariance(v, means)
    if kind == "robust":
        info["covariance"] = "robust"
        info["correction"] = "vce(robust) = vce(cluster panel); " + info["correction"]
    info["k_small_sample"] = k
    return v, info, float(info["df_inference"])


@dataclass
class GLSFit:
    """Swamy-Arora variance components and the feasible GLS fit built on them."""

    beta: Tensor
    xtx_inv: Tensor               # (X*'X*)^-1 of the transformed regression
    resid: Tensor                 # transformed residuals
    xs: Tensor                    # transformed design (constant column 1 - theta_i)
    ssr: float                    # RSS* of the transformed regression (rmse^2 = RSS*/(N-K))
    means: Tensor                 # [n, k] panel means of the slopes
    ybar: Tensor                  # [n]
    sigma_e2: float
    sigma_u2: float
    t_bar: float
    theta: Tensor                 # [n]
    ssr_within: float
    df_within: float
    k_within: int
    ssr_between: float
    df_between: float
    pooled_resid: Tensor
    pooled_ssr: float
    method: str


def random_effects_gls(sample: PanelSample, design, y: Tensor, *, sa: bool,
                       scale: float | None = None) -> GLSFit:
    """Swamy-Arora components and the GLS regression (the re model; starting point of mle).

    sigma_e^2 comes from the within regression of the regressors with within
    variation (time-invariant and within-collinear columns are left out of it
    without being dropped from the model; a column counts as varying when its
    within sum of squares exceeds ``WITHIN_TOL`` times its grand-mean-centered
    sum of squares, so a large offset does not hide real within variation),
    sigma_u^2 from the unweighted between regression of the panel means with
    the harmonic-mean T_bar, or from the Baltagi-Chang formula with ``sa``.
    The models that call this take no weights (Stata's xtreg, re / mle).

    ``design`` is [1, X] with any centering of the slopes (the fit is
    equivariant). ``scale`` is the uncentered sum of squares of the outcome's
    level in the units of ``y`` (default ``sum y^2``), the yardstick of the
    ``no_within_variation`` checks; callers that pass a centered ``y`` supply it.
    """
    from openecon.engines import linalg

    frame, codes, n, sizes = sample.frame, sample.codes, sample.n, sample.sizes
    outcome = frame.spec.outcome
    scale = float(y.square().sum()) if scale is None else scale
    x = design.x[:, 1:]
    means = group_means(x, codes, n)
    ybar = group_means(y, codes, n)
    within_x, within_y = x - means[codes], y - ybar[codes]
    tss_within = float(within_y.square().sum())
    if tss_within <= EXACT_FIT * scale:
        raise AnalysisError("no_within_variation",
                            f"The outcome '{outcome}' has no within-panel variation, so the "
                            "idiosyncratic variance sigma_e^2 cannot be estimated. Use "
                            "model='be' or model='pooled'.")
    before = (x - x.mean(dim=0)).square().sum(dim=0)
    after = within_x.square().sum(dim=0)
    kept = (after > WITHIN_TOL * before).nonzero().flatten().tolist()
    if kept:
        inner, _ = kernel_call(linalg.collinear_columns, within_x[:, kept])
        kept = [kept[i] for i in inner]
    if kept:
        ssr_within = float(kernel_call(linalg.least_squares, within_x[:, kept], within_y).ssr)
    else:
        ssr_within = tss_within
    if ssr_within <= EXACT_FIT * scale:
        raise AnalysisError("no_within_variation",
                            f"The regressors explain the within-panel variation of '{outcome}' "
                            "exactly, so sigma_e^2 is zero and the random-effects transform is "
                            "undefined. Remove the regressor that reproduces the outcome.")
    df_within = sample.nobs - n - len(kept)
    zbar = torch.cat([torch.ones((n, 1), dtype=torch.float64), means], dim=1)
    between = kernel_call(linalg.least_squares, zbar, ybar)
    df_between = n - between.rank
    sigma_e2, sigma_u2, t_bar = kernel_call(kernels.swamy_arora, ssr_within, df_within,
                                            float(between.ssr), df_between, sizes)
    method = "swamy_arora_harmonic_mean"
    if sa:
        zkept = zbar[:, between.kept]
        weighted = kernel_call(linalg.least_squares, zkept, ybar, sizes, drop_collinear=False)
        sigma_u2 = kernel_call(kernels.baltagi_chang_sigma_u, float(weighted.ssr), df_between,
                               sigma_e2, zkept, sizes)
        method = "baltagi_chang_sa"
    if sigma_u2 == 0:
        frame.warn("The estimated sigma_u is zero: the random-effects GLS reduces to pooled OLS.")
    theta = kernel_call(kernels.gls_theta, sizes, sigma_u2, sigma_e2)
    shrink = theta[codes]
    xs = design.x - shrink[:, None] * zbar[codes]
    ys = y - shrink * ybar[codes]
    fit = kernel_call(linalg.least_squares, xs, ys, drop_collinear=False)
    pooled = kernel_call(linalg.least_squares, design.x, y, drop_collinear=False)
    return GLSFit(beta=fit.beta, xtx_inv=fit.xtx_inv, resid=fit.resid, xs=xs, ssr=float(fit.ssr),
                  means=means, ybar=ybar, sigma_e2=sigma_e2, sigma_u2=sigma_u2, t_bar=t_bar,
                  theta=theta, ssr_within=ssr_within, df_within=df_within, k_within=len(kept),
                  ssr_between=float(between.ssr), df_between=df_between,
                  pooled_resid=pooled.resid, pooled_ssr=float(pooled.ssr), method=method)


def driscoll_kraay(sample: PanelSample, *, x: Tensor, resid: Tensor, bread: Tensor, nobs: int,
                   k: int, weights: Tensor | None) -> tuple[Tensor, dict[str, Any], float]:
    """Driscoll-Kraay covariance with Hoechle's xtscc conventions (see module docstring)."""
    frame = sample.frame
    time = frame.time_index()
    periods = int(torch.unique(time).numel())
    if periods < 2:
        raise AnalysisError("insufficient_periods",
                            "Driscoll-Kraay covariance needs at least two time periods.")
    lags = frame.option("lags")
    lags = cov.newey_west_lags(periods) if lags is None else int(lags)
    kernel = frame.option("kernel")
    if kernel == "truncated" and lags >= int(time.max()-time.min()):
        # For a full rectangular window M = (sum_t score_t)(sum_t score_t)'.
        # Least squares' normal equations make this exactly zero. Floating
        # residual summation must not turn it into spurious positive inference.
        raise AnalysisError("invalid_covariance", "A truncated Driscoll-Kraay window covering every period gives zero least-squares score covariance. Use a shorter lag window or another kernel.")
    scale = resid if weights is None else resid * weights
    meat = kernel_call(cov.meat_driscoll_kraay, x * scale[:, None], time, lags, kernel)
    if nobs <= k:
        raise AnalysisError("insufficient_observations",
                            "The Driscoll-Kraay small-sample factor needs more observations "
                            "than parameters.")
    factor = periods / (periods - 1) * (nobs - 1) / (nobs - k)
    v = kernel_call(cov.sandwich, bread, meat) * factor
    info = {"covariance": "driscoll_kraay",
            "correction": "Driscoll-Kraay (xtscc): T/(T-1) * (N-1)/(N-K), K = slopes + constant",
            "small_sample_correction": factor, "lags": lags, "kernel": kernel,
            "periods": periods, "df_inference": periods - 1, "k_small_sample": k}
    return v, info, float(periods - 1)
