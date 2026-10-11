"""Linear mixed model with random effects and a structured residual covariance.

    y_i = X_i b + Z_i u_i + e_i,    u_i ~ N(0, G),    e_i ~ N(0, s2 D_i C_i D_i)

for independent groups i (one grouping column). ``G`` is the covariance of a
random intercept and/or random slopes (independent or unstructured,
log-Cholesky coordinates). The residual covariance combines

* ``C_i``: ``independent`` (I), ``ar1`` (rho^|t_a - t_b| for integer
  occasions t, nlme ``corAR1(form = ~ t | g)``) or ``exchangeable``
  (1 on the diagonal, rho off it, nlme ``corCompSymm``); correlation acts
  within blocks: the groups, or the cells of an optional ``within`` column
  nested in the groups (nlme ``form = ~ 1 | g/cell``);
* ``D_i = diag(delta_s)``: heteroskedastic standard-deviation ratios by an
  optional ``stratum`` column (nlme ``varIdent(form = ~ 1 | stratum)``), with
  ``delta = 1`` for the first stratum in sorted label order.

Estimation is by ML or REML with b profiled out by GLS, on dense padded
per-group matrices (bounded, planned workspace). Coordinates: log-Cholesky of
G, ln s, ln delta_s (non-reference strata), rho = tanh(eta) for ar1 and
rho = l + (1 - l) logistic(eta), l = -1/(m - 1) with m the largest block, for
exchangeable. Derivatives are exact Torch chain rule (``extended_common``).

Inference: ML reports the inverse of the full joint observed information of
(b, covariance parameters), cross terms retained; REML reports the
model-based GLS covariance ``(X'V^-1 X)^-1`` for b and the inverse observed
restricted information for the covariance parameters (block diagonal). The
conditional GLS fixed covariance is kept in ``extra`` for both. Reported
variances, covariances, per-stratum residual variances ``s2 delta_s^2`` and
rho carry delta-method standard errors. A random intercept with an
exchangeable residual correlation on the same blocks is not identified and
is refused; boundary estimates raise ``boundary_solution``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import (
    ModelFrame, build_result, column_list, information_criteria, make_spec, wald_test,
)
from openecon.econometrics.discrete.common import constant_shift
from openecon.econometrics.mixed import common
from openecon.econometrics.mixed.extended_common import (
    Objective, covariance_report, factor, factor_size, factor_start, maximize, optimizer_record,
)
from openecon.econometrics.mixed.small_sample import padded_positions
from openecon.engines.linalg import least_squares

RESIDUALS = ("independent", "ar1", "exchangeable")
MAX_GROUP = 256
MAX_LAG = 512
_LOG_2PI = math.log(2 * math.pi)


def _pad(values, group, position, count, m):
    out = torch.zeros((count, m, *values.shape[1:]), dtype=values.dtype)
    out[group, position] = values
    return out


class _Model:
    """Padded dense group geometry and the (restricted) Gaussian likelihood."""

    def __init__(self, frame: ModelFrame):
        spec = frame.spec
        self.frame = frame
        self.group = frame.role("group")[0]
        self.random = frame.role("random")
        occasion, stratum, within = (frame.role(name) for name in ("occasion", "stratum", "within"))
        self.residual = frame.option("residual")
        self.reml = frame.option("method") == "reml"
        self.structure = frame.option("covstructure")
        self.intercept_re = bool(frame.option("random_intercept"))
        if self.group in {spec.outcome, *spec.predictors, *self.random}:
            raise AnalysisError("invalid_spec", f"The grouping column '{self.group}' must differ "
                                "from the outcome, the regressors and the random slopes.")
        if spec.outcome in {*occasion, *stratum, *within}:
            raise AnalysisError("invalid_spec", "The outcome cannot define occasions, strata or "
                                "correlation blocks.")
        if not self.random and not self.intercept_re:
            raise AnalysisError("invalid_spec", "The model has no random effect.")
        if occasion and self.residual != "ar1":
            raise AnalysisError("invalid_spec", "occasion is used only by residual='ar1'.")
        if self.residual == "ar1" and not occasion:
            raise AnalysisError("invalid_spec", "residual='ar1' needs an integer occasion column.")
        if within and self.residual == "independent":
            raise AnalysisError("invalid_spec", "within defines residual correlation blocks; "
                                "use it with residual='ar1' or 'exchangeable'.")
        design = frame.drop_collinear(frame.design(), None)
        if not design.terms:
            raise AnalysisError("invalid_spec", "At least one fixed-effect term is required.")
        self.design = design
        self.y = frame.numeric(spec.outcome)
        common.require_variation(self.y, "mixedresid", spec.outcome)
        columns = [frame.numeric(name) for name in self.random]
        if self.intercept_re:
            columns.append(torch.ones(frame.n, dtype=torch.float64))
        self.z = torch.stack(columns, dim=1)
        self.z_names = [*self.random, *(["_cons"] if self.intercept_re else [])]
        common.require_independent_effects(self.z, self.z_names, None, "mixedresid")
        codes, count = frame.codes(self.group)
        if count < 3:
            raise AnalysisError("insufficient_groups", "At least three groups are required.")
        self.codes, self.count = codes, count
        self.levels = common.group_sizes(codes, count)
        self.position, self.m = padded_positions(codes, count)
        if self.m > MAX_GROUP:
            raise AnalysisError("work_budget", f"Groups are bounded to {MAX_GROUP} rows for the "
                                "dense residual covariance.")
        n, self.p = frame.n, len(design.terms)
        self.q = self.z.shape[1]
        self.size = factor_size(self.q, self.structure)
        # Residual correlation blocks: the groups, or within-cells nested in them.
        if within:
            block, _ = frame.codes([self.group, within[0]])
            if not common.nested(block, codes, int(block.max()) + 1):
                raise AnalysisError("invalid_spec", "within cells must nest in the groups.")
        else:
            block = codes
        block_sizes = torch.bincount(block)
        self.max_block = int(block_sizes.max())
        if self.residual == "exchangeable":
            if self.max_block < 2:
                raise AnalysisError("unidentified_covariance", "Exchangeable residual "
                                    "correlation needs blocks with at least two observations.")
            if self.intercept_re and not within:
                raise AnalysisError("unidentified_covariance", "A random intercept and an "
                                    "exchangeable residual correlation on the same groups are "
                                    "not separately identified; declare nested within cells, "
                                    "drop the random intercept, or use residual='ar1'.")
        self.within = within[0] if within else None
        self.occasion = occasion[0] if occasion else None
        self.stratum = stratum[0] if stratum else None
        mask = _pad(torch.ones(n, dtype=torch.float64), codes, self.position, count, self.m)
        self.mask = mask
        blocks = _pad(block + 1, codes, self.position, count, self.m)
        self.same_block = ((blocks[:, :, None] == blocks[:, None, :]) & (blocks[:, :, None] > 0))
        self.same_block = self.same_block.to(torch.float64)
        self.observed = mask[:, :, None] * mask[:, None, :]
        self.pad_eye = torch.diag_embed(1 - mask)
        self.eye = torch.diag_embed(mask)
        if self.residual == "ar1":
            series = frame.numeric(self.occasion)
            if not bool((series == series.round()).all()) or float(series.abs().max()) > 1e6:
                raise AnalysisError("invalid_occasion", "AR(1) occasions must be integers "
                                    "within +/-1,000,000.")
            times = _pad(series, codes, self.position, count, self.m)
            gap = (times[:, :, None] - times[:, None, :]).abs() * self.same_block
            off = gap[(self.same_block > 0) & ~torch.eye(self.m, dtype=torch.bool)]
            if bool((off == 0).any()):
                raise AnalysisError("duplicate_occasion", "Occasions must be unique within each "
                                    "residual correlation block.")
            if off.numel() == 0:
                raise AnalysisError("unidentified_covariance", "AR(1) needs repeated "
                                    "observations within blocks.")
            if float(off.max()) > MAX_LAG:
                raise AnalysisError("work_budget", f"Within-block occasion gaps are bounded to "
                                    f"{MAX_LAG}.")
            if math.gcd(*{int(v) for v in off.tolist()}) != 1:
                raise AnalysisError("unsupported_occasion_grid", "Observed within-block integer "
                                    "lags must have greatest common divisor one (the sign of a "
                                    "negative rho is otherwise unidentified).")
            self.gap = gap.to(torch.int64)
            self.max_gap = int(self.gap.max())
        if self.stratum:
            raw, k_strata = frame.codes(self.stratum)
            labels = frame.levels(self.stratum)
            if k_strata < 2:
                raise AnalysisError("invalid_spec", "stratum needs at least two levels.")
            order = sorted(range(k_strata), key=lambda i: (str(type(labels[i])), labels[i]))
            rank = torch.empty(k_strata, dtype=torch.int64)
            rank[torch.tensor(order)] = torch.arange(k_strata)
            strata = rank[raw]
            if int(torch.bincount(strata).min()) < 2:
                raise AnalysisError("insufficient_observations", "Every stratum needs at least "
                                    "two observations.")
            self.strata_labels = [labels[i] for i in order]
            self.strata = _pad(strata, codes, self.position, count, self.m)
        else:
            self.strata_labels = []
            self.strata = None
        self.n_strata = len(self.strata_labels)
        self.k = self.size + 1 + max(self.n_strata - 1, 0) + int(self.residual != "independent")
        if n <= self.p + self.k:
            raise AnalysisError("insufficient_observations", "More observations than "
                                "parameters are required.")
        frame.workspace_plan("dense random-effects plus residual covariance likelihood", {
            "group covariance, factor and derivative tape": 8 * count * self.m * self.m
                                                             * (self.k + self.p + 12),
            "group designs": 8 * count * self.m * (self.p + self.q + 4) * 4,
        })
        # Centre the slopes for conditioning (exact constant reparameterization).
        x = design.x.clone()
        self.means = torch.zeros(self.p, dtype=torch.float64)
        if design.intercept and self.p > 1:
            self.means[1:] = x[:, 1:].mean(0)
            x[:, 1:] -= self.means[1:]
        self.shift = constant_shift(self.means) if design.intercept else \
            torch.eye(self.p, dtype=torch.float64)
        self.xp = _pad(x, codes, self.position, count, self.m)
        self.yp = _pad(self.y, codes, self.position, count, self.m)
        self.zp = _pad(self.z, codes, self.position, count, self.m)
        self.n = n
        self.work = count * (self.m ** 3 + self.m ** 2 * (self.p + self.q + 2)) * 8

    # ---- parameters --------------------------------------------------------------

    def rho(self, theta):
        if self.residual == "ar1":
            return torch.tanh(theta[-1])
        lower = -1.0 / (self.max_block - 1)
        return lower + (1 - lower) * torch.sigmoid(theta[-1])

    def rho_coordinate(self, value: float) -> float:
        if self.residual == "ar1":
            return math.atanh(value)
        lower = -1.0 / (self.max_block - 1)
        fraction = (value - lower) / (1 - lower)
        return math.log(fraction / (1 - fraction))

    def deltas(self, theta):
        if not self.n_strata:
            return None
        start = self.size + 1
        return torch.cat([torch.ones(1, dtype=torch.float64),
                          torch.exp(theta[start:start + self.n_strata - 1])])

    def covariance(self, theta):
        lower = factor(theta[:self.size], self.q, self.structure)
        zl = self.zp @ lower
        v = (zl @ zl.transpose(1, 2)) * self.observed
        sigma2 = torch.exp(2 * theta[self.size])
        if self.residual == "independent":
            corr = self.eye
        elif self.residual == "ar1":
            rho = self.rho(theta)
            powers = torch.stack([rho ** k for k in range(self.max_gap + 1)])
            corr = powers[self.gap] * self.same_block
        else:
            rho = self.rho(theta)
            corr = rho * self.same_block + (1 - rho) * self.eye
        deltas = self.deltas(theta)
        if deltas is not None:
            scale = deltas[self.strata] * self.mask
            corr = corr * scale[:, :, None] * scale[:, None, :]
        return v + sigma2 * corr + self.pad_eye

    def components(self, theta, beta=None):
        v = self.covariance(theta)
        lower, info = torch.linalg.cholesky_ex(v)
        if bool((info != 0).any()):
            return None
        vx = torch.cholesky_solve(self.xp, lower)
        xvx = torch.einsum("sap,saq->pq", self.xp, vx)
        xvx = (xvx + xvx.T) / 2
        chol, info = torch.linalg.cholesky_ex(xvx)
        if int(info) != 0:
            return None
        if beta is None:
            vy = torch.cholesky_solve(self.yp[..., None], lower)[..., 0]
            beta = torch.cholesky_solve(torch.einsum("sap,sa->p", self.xp, vy)[:, None],
                                        chol)[:, 0]
        resid = (self.yp - torch.einsum("sap,p->sa", self.xp, beta)) * self.mask
        quad = (resid * torch.cholesky_solve(resid[..., None], lower)[..., 0]).sum()
        logdet = 2 * torch.log(torch.diagonal(lower, dim1=1, dim2=2)).sum()
        logdet_x = 2 * torch.log(chol.diagonal()).sum()
        ml = -0.5 * (self.n * _LOG_2PI + logdet + quad)
        reml = -0.5 * ((self.n - self.p) * _LOG_2PI + logdet + logdet_x + quad)
        return beta, chol, ml, reml

    def value(self, theta):
        pieces = self.components(theta)
        if pieces is None:
            return theta.sum() * math.nan
        return pieces[3] if self.reml else pieces[2]

    def joint_value(self, parameters):
        pieces = self.components(parameters[self.p:], parameters[:self.p])
        if pieces is None:
            return parameters.sum() * math.nan
        return pieces[2]

    def report(self, theta):
        parts = [covariance_report(theta[:self.size], self.q, self.structure)]
        sigma2 = torch.exp(2 * theta[self.size])
        deltas = self.deltas(theta)
        parts.append(sigma2.reshape(1) if deltas is None else sigma2 * deltas.square())
        if self.residual != "independent":
            parts.append(self.rho(theta).reshape(1))
        return torch.cat(parts)

    def terms(self) -> list[str]:
        names, group = self.z_names, self.group
        terms = [f"/var({name}[{group}])" for name in names]
        if self.structure == "unstructured":
            terms += [f"/cov({names[i]},{names[j]}[{group}])"
                      for i in range(self.q) for j in range(i)]
        if self.n_strata:
            terms += [f"/var(Residual[{self.stratum}={label}])" for label in self.strata_labels]
        else:
            terms.append("/var(Residual)")
        if self.residual != "independent":
            terms.append("/rho(Residual)")
        return terms

    def starts(self):
        fit = least_squares(self.design.x, self.y, drop_collinear=False)
        sd = float((fit.ssr / max(self.n - self.p, 1)) ** 0.5)
        if not math.isfinite(sd) or sd <= 0:
            raise AnalysisError("perfect_fit", "The regressors explain the outcome exactly.")
        rhos = [0.0] if self.residual == "independent" else [0.05, 0.4, -0.3, 0.7]
        starts = []
        for fraction in (0.3, 0.7):
            for rho in rhos:
                parts = [factor_start(self.q, self.structure, sd * fraction),
                         torch.tensor([math.log(sd * (1 - fraction / 2))], dtype=torch.float64),
                         torch.zeros(max(self.n_strata - 1, 0), dtype=torch.float64)]
                if self.residual != "independent":
                    if self.residual == "exchangeable" and rho <= -1.0 / (self.max_block - 1):
                        continue
                    parts.append(torch.tensor([self.rho_coordinate(rho)], dtype=torch.float64))
                starts.append(torch.cat(parts))
        return starts


def _check_boundary(model: _Model, theta) -> None:
    lower = factor(theta[:model.size], model.q, model.structure)
    g = lower @ lower.T
    sigma2 = float(torch.exp(2 * theta[model.size]))
    z2 = model.z.square().mean(0)
    relative = g.diagonal() * z2 / sigma2
    if bool((relative < 1e-8).any()) or not bool(torch.isfinite(g).all()):
        raise AnalysisError("boundary_solution", "A random-effect variance is estimated at zero "
                            "(the boundary of the parameter space); remove that random effect.")
    if model.structure == "unstructured" and model.q > 1:
        sd = g.diagonal().sqrt()
        if float(torch.linalg.eigvalsh(g / sd[:, None] / sd)[0]) < 1e-8:
            raise AnalysisError("boundary_solution", "The random effects are estimated as "
                                "perfectly correlated; use covstructure='independent'.")
    if model.residual != "independent":
        rho = float(model.rho(theta))
        lower_bound = -1.0 if model.residual == "ar1" else -1.0 / (model.max_block - 1)
        if rho > 1 - 1e-6 or rho < lower_bound + 1e-6:
            raise AnalysisError("boundary_solution", "The residual correlation is estimated at "
                                "the edge of its admissible range.")
    deltas = model.deltas(theta)
    if deltas is not None and (float(deltas.min()) < 1e-4 or float(deltas.max()) > 1e4):
        raise AnalysisError("boundary_solution", "A stratum residual variance is at zero.")


def _inverse(information, what):
    matrix = (information + information.T) / 2
    scale = matrix.diagonal().abs().rsqrt()
    chol, info = torch.linalg.cholesky_ex(matrix * scale[:, None] * scale)
    if int(info) or not bool(torch.isfinite(chol).all()) or float(chol.diagonal().min()) < 1e-6:
        raise AnalysisError("boundary_solution", f"The {what} is singular or indefinite; Wald "
                            "inference is unavailable.")
    return torch.cholesky_inverse(chol) * scale[:, None] * scale


def fit_mixedresid(spec, data):
    """Entry point of ``mixedresid``: ``(spec, data) -> ResultBundle``."""
    with torch.device("cpu"), torch.no_grad():
        frame = ModelFrame(spec, data)
        model = _Model(frame)
        objective = Objective(model.value, model.work, frame.option("max_work"))
        fit, theta_cov = maximize(objective, model.starts())
        theta = fit.theta
        _check_boundary(model, theta)
        beta_c, chol, ml, reml = model.components(theta)
        conditional = torch.cholesky_inverse(chol)
        p, k = model.p, model.k
        if model.reml:
            raw = torch.block_diag(conditional, theta_cov)
            convention = ("REML: model-based GLS (X'V^-1 X)^-1 for b; inverse observed restricted "
                          "information for the covariance parameters; block diagonal")
        else:
            objective.charge(2 * (p + k + 1))
            with torch.enable_grad():
                hessian = torch.autograd.functional.hessian(model.joint_value,
                                                            torch.cat([beta_c, theta]))
            raw = _inverse(-hessian, "joint observed information")
            convention = ("ML: inverse of the full joint observed information of b and the "
                          "covariance parameters; cross terms retained")
        with torch.enable_grad():
            jacobian = torch.autograd.functional.jacobian(model.report, theta)
        transform = torch.block_diag(model.shift, jacobian)
        covariance = transform @ raw @ transform.T
        covariance = (covariance + covariance.T) / 2
        beta = model.shift @ beta_c
        reported = model.report(theta)
        terms = [*model.design.terms, *model.terms()]
        log_likelihood = float(reml if model.reml else ml)
        criteria = information_criteria(log_likelihood, p + k, model.n)
        slopes = [i for i, term in enumerate(model.design.terms) if term != "Intercept"]
        tests = {"model": wald_test(beta, covariance[:p, :p], slopes,
                                    label="Wald chi2 test of the fixed slopes")} if slopes else {}
        lower = factor(theta[:model.size], model.q, model.structure)
        extra = {
            "method": "reml" if model.reml else "ml",
            "likelihood": "restricted (REML)" if model.reml else "full (ML)",
            "covstructure": model.structure, "residual": model.residual,
            "random_effects": {"group": model.group, "effects": model.z_names},
            "G": (lower @ lower.T).tolist(),
            "residual_blocks": model.within or model.group,
            "occasion": model.occasion, "stratum": model.stratum,
            "strata": [str(label) for label in model.strata_labels],
            "stratum_reference": str(model.strata_labels[0]) if model.strata_labels else None,
            "rho": float(model.rho(theta)) if model.residual != "independent" else None,
            "levels": [{"group": model.group, **model.levels}],
            "max_block_size": model.max_block,
            "conditional_gls_covariance": (model.shift @ conditional @ model.shift.T).tolist(),
            "likelihood_theta": theta.tolist(),
            "likelihood_covariance": theta_cov.tolist(),
            "theta_parameterization": "log-Cholesky of G, ln sigma, ln delta (non-reference "
                                      "strata), rho coordinate (tanh / shifted logistic)",
            "inference_convention": convention,
            "small_sample_df": "not provided (nonlinear residual coordinates); z inference",
            "prediction_domain": "fitted population mean X b only; no BLUP adapter",
        }
        method = "REML" if model.reml else "ML"
        bundle = build_result(
            frame, terms=terms, params=torch.cat([beta, reported]), covariance=covariance,
            title=f"Mixed-effects {method} regression with structured residuals",
            equations=[spec.outcome] * p + ["Random"] * (len(terms) - p), use_t=False,
            metrics={"log_likelihood": log_likelihood, "aic": criteria["aic"],
                     "bic": criteria["bic"], "n_groups": model.count,
                     "group_size_min": model.levels["size_min"],
                     "group_size_avg": model.levels["size_avg"],
                     "group_size_max": model.levels["size_max"]},
            fitted=model.design.x @ beta, solver="native dense profiled Gaussian likelihood",
            optimizer=optimizer_record(fit, objective),
            inference={"covariance": "nonrobust", "correction": convention},
            tests=tests, extra=extra, categories=model.design.categories,
        )
        positions = [p + i for i, term in enumerate(terms[p:]) if term.startswith("/var(")]
        common.variance_intervals(bundle, positions, spec.alpha)
        return bundle


def mixedresid(*, data: Any, y: str, x=None, group: str, random=None,
               random_intercept: bool = True, covstructure: str = "independent",
               residual: str = "ar1", occasion: str | None = None, stratum: str | None = None,
               within: str | None = None, method: str = "reml", categorical=None,
               intercept: bool = True, missing: str = "raise", alpha: float = 0.05,
               max_work: int = 2_000_000_000):
    """Linear mixed model with random effects plus AR(1)/exchangeable/heteroskedastic residuals.

    Model
        ``y_i = X_i b + Z_i u_i + e_i`` with ``u_i ~ N(0, G)`` (random intercept and/or
        slopes; ``covstructure`` independent or unstructured) and
        ``Var(e_i) = s2 D_i C_i D_i``: ``residual='ar1'`` uses ``rho^|t_a - t_b|`` on the
        integer ``occasion`` column (nlme ``corAR1(form = ~ occasion | group)``),
        ``'exchangeable'`` a common correlation (nlme ``corCompSymm``), ``'independent'``
        none. Correlation acts within groups or within the cells of a nested ``within``
        column. ``stratum`` adds one residual standard-deviation ratio per level (nlme
        ``varIdent(form = ~ 1 | stratum)``; the first sorted level is the reference).

    Estimator
        ML or REML (default, as nlme ``lme``) with b profiled out by GLS, on dense
        per-group covariance matrices; exact Torch derivatives; BFGS with a Newton
        polish. ML reports the full joint observed-information covariance of b and the
        covariance parameters; REML the GLS ``(X'V^-1 X)^-1`` for b and the restricted
        observed information for the covariance parameters. Inference is asymptotic z.

    Result
        Fixed effects, then ``/var(...)``, ``/cov(...)`` of the random effects,
        ``/var(Residual)`` (or one ``/var(Residual[stratum=level])`` per stratum) and
        ``/rho(Residual)``. ``extra``: G, rho, strata, the conditional GLS covariance,
        likelihood coordinates and their covariance.

    Stata / R
        ``mixed y x || id:, residuals(ar 1, t(time)) reml`` is
        ``oe.mixedresid(data=df, y='y', x=['x'], group='id', residual='ar1',
        occasion='time')``; ``lme(y ~ x, random = ~ 1 | id, correlation =
        corAR1(form = ~ time | id), weights = varIdent(form = ~ 1 | arm))`` adds
        ``stratum='arm'``.
    """
    from openecon.analysis import fit

    columns = {"group": group, "random": column_list(random or [], "random")}
    for name, value in (("occasion", occasion), ("stratum", stratum), ("within", within)):
        if value is not None:
            columns[name] = value
    spec = make_spec(
        "mixedresid", outcome=y, predictors=column_list(x or [], "x"), intercept=intercept,
        categorical=column_list(categorical, "categorical"), missing=missing, alpha=alpha,
        columns=columns,
        options={"random_intercept": None if random_intercept else False,
                 "covstructure": covstructure, "residual": residual, "method": method,
                 "max_work": max_work})
    return fit(spec, data=data)
