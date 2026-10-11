"""Small-study/publication-bias sensitivity: trim-and-fill and PET-PEESE.

Trim-and-fill follows Duval and Tweedie (2000, Biometrics 56:455-463;
JASA 95:89-98) with the L0/R0 estimators, side selection by the sign of the
sqrt(vi) meta-regression slope and iteration on the trimmed estimate exactly
as documented by the metafor author:
https://wviechtb.github.io/metafor/reference/trimfill.html. PET and PEESE are
the weighted least-squares regressions of effects on their standard error or
sampling variance with weights 1/vi and an unrestricted multiplicative
dispersion (Stanley and Doucouliagos 2014, Research Synthesis Methods 5:60-78).
None of these procedures establishes publication bias or recovers the
unbiased effect; they are model-dependent sensitivity analyses.
"""

from __future__ import annotations

import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.engines.distributions import t_isf, t_sf

from .common import level_check, sample
from .models import meta_pool, meta_regress

TRIMFILL_SOURCE = "https://wviechtb.github.io/metafor/reference/trimfill.html"
PET_SOURCE = "doi:10.1002/jrsm.1095"


def _nullable(records, columns=None):
    frame = table(records, columns=columns)
    return frame.astype(object).where(frame.notna(), None)


def _summary(fit, label, k):
    coef = fit["coefficients"].iloc[0]
    het = fit["heterogeneity"].iloc[0]
    q = fit["tests"].iloc[0]
    return {"fit": label, "k": k, "estimate": float(coef.estimate), "std_error": float(coef.std_error),
            "statistic": float(coef.statistic), "p_value": float(coef.p_value), "ci_low": float(coef.ci_low),
            "ci_high": float(coef.ci_high), "tau2": float(het.tau2), "I2_percent": float(het.I2_percent),
            "Q": float(q.statistic), "Q_df": int(q.df1), "Q_p_value": float(q.p_value)}


def _ranks(values):
    """1-based ranks of |values| with ties in order of appearance (R ties.method='first')."""
    order = torch.argsort(values.abs(), stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, len(values)+1)
    return ranks.to(torch.float64)


@torch.no_grad()
def meta_trimfill(*, data, yi: str = "yi", vi: str = "vi", study: str | None = None, method: str = "REML",
                  estimator: str = "L0", side: str | None = None, inference: str = "z", level: float = .95,
                  max_iterations: int = 100, dependence: str = "independent") -> TableSet:
    """Duval-Tweedie trim-and-fill (L0/R0) sensitivity analysis for an intercept-only pool.

    Side is declared or set by the sign of the sqrt(vi) moderator slope; iterates
    the trimmed common/DL/ML/REML estimate until the missing-study count repeats,
    mirrors the most extreme effects and refits all observed plus filled studies.
    3..1000 studies. Imputed studies are not data; asymmetry is not proof of bias.
    """
    if estimator not in ("L0", "R0"):
        raise AnalysisError("invalid_spec", "estimator must be 'L0' or 'R0'.")
    if side not in (None, "left", "right"):
        raise AnalysisError("invalid_spec", "side must be None (estimated), 'left' or 'right'.")
    if type(max_iterations) is not int or not 1 <= max_iterations <= 1000:
        raise AnalysisError("invalid_spec", "max_iterations must be an integer in 1..1000.")
    level = level_check(level)
    # <=1000 observed studies keeps every observed-plus-filled refit within the pooling limit.
    frame, values, ids, digest = sample(data, [yi, vi], study, dependence=dependence, minimum=3, maximum=1000)
    y, v = values[yi], values[vi]
    k = len(ids)
    settings = dict(method=method, inference=inference, level=level)
    names = {"yi": "__trimfill_yi__", "vi": "__trimfill_vi__", "se": "__trimfill_sei__"}

    def pool(effects, variances):
        table_ = pd.DataFrame({names["yi"]: effects.tolist(), names["vi"]: variances.tolist()})
        return meta_pool(data=table_, yi=names["yi"], vi=names["vi"], **settings)

    observed = pool(y, v)
    side_slope = None
    if side is None:
        reg = pd.DataFrame({names["yi"]: y.tolist(), names["vi"]: v.tolist(), names["se"]: v.sqrt().tolist()})
        fit = meta_regress(data=reg, yi=names["yi"], vi=names["vi"], moderators=[names["se"]], method=method, level=level)
        side_slope = float(fit["coefficients"].iloc[1].estimate)
        side = "right" if side_slope < 0 else "left"
        side_source = "estimated: sign of the sqrt(vi) meta-regression slope"
    else:
        side_source = "declared"
    sign = -1.0 if side == "right" else 1.0
    u = sign*y
    order = torch.argsort(u, stable=True)
    u, vs = u[order], v[order]
    sorted_ids = [ids[i] for i in order.tolist()]
    k0, previous, history, beta, se_k0 = 0, -1, [], 0.0, 0.0
    while k0 != previous:
        if len(history) >= max_iterations:
            raise AnalysisError("trimfill_not_converged", "Trim-and-fill did not reach a repeated missing-study count within max_iterations.")
        previous = k0
        if k-k0 < 2:
            raise AnalysisError("trimfill_undefined", "The trimmed sample has fewer than two studies; trim-and-fill is undefined.")
        fit = pool(u[:k-k0], vs[:k-k0])
        beta = float(fit["coefficients"].iloc[0].estimate)
        centered = u-beta
        signed = torch.sign(centered)*_ranks(centered)
        if estimator == "L0":
            sr = float(signed[signed > 0].sum())
            raw = (4*sr-k*(k+1))/(2*k-1)
            var_sr = (k*(k+1)*(2*k+1)+10*raw**3+27*raw**2+17*raw-18*k*raw**2-18*k*raw+6*k**2*raw)/24
            se_k0 = 4*math.sqrt(var_sr)/(2*k-1) if var_sr > 0 else 0.0
        else:
            negative = signed[signed < 0]
            if negative.numel() == 0:
                raise AnalysisError("trimfill_undefined", "R0 needs at least one effect below the trimmed estimate.")
            raw = (k-float((-negative).max()))-1
            se_k0 = math.sqrt(2*max(0.0, raw)+2)
        k0 = max(0, round(raw))
        se_k0 = max(0.0, se_k0)
        history.append({"iteration": len(history)+1, "trimmed_k": k-previous, "trimmed_estimate": sign*beta,
                        "k0_raw": raw, "k0": k0})
    filled = []
    if k0 > 0:
        mirrored = 2*beta-u[k-k0:]
        filled = [{"study": f"Filled {i+1}", "yi": float(sign*mirrored[i]), "vi": float(vs[k-k0+i]),
                   "mirrored_from": sorted_ids[k-k0+i]} for i in range(k0)]
        effects = torch.cat([y, sign*mirrored])
        variances = torch.cat([v, vs[k-k0:]])
        adjusted = pool(effects, variances)
    else:
        adjusted = observed
    summary = {"side": side, "side_source": side_source, "side_slope": side_slope, "estimator": estimator,
               "k_observed": k, "k0": k0, "se_k0": se_k0, "p_k0": 0.5**(k0+1) if estimator == "R0" else None,
               "k_total": k+k0, "iterations": len(history), "method": method, "inference": inference}
    return TableSet({"summary": _nullable([summary]),
                     "estimates": table([_summary(observed, "observed", k), _summary(adjusted, "trim-and-fill", k+k0)]),
                     "filled": _nullable(filled, ["study", "yi", "vi", "mirrored_from"]),
                     "iterations": table(history),
                     "adjusted_covariance": adjusted["covariance"]},
                    title="Trim-and-fill sensitivity analysis", procedure="meta_trimfill", sample_hash=digest,
                    level=level, method=method, inference=inference, source=TRIMFILL_SOURCE,
                    notes=["Filled studies are imputed mirror images, not observed data; their uncertainty is treated as observed.",
                           "Funnel asymmetry can arise from heterogeneity, chance or design and does not establish publication bias.",
                           "R0 p_k0 is 0.5^(k0+1) as in the metafor author's implementation."])


def _wls(y, v, regressor):
    """Weighted LS y ~ 1 + regressor, weights 1/v, unrestricted multiplicative dispersion."""
    k = len(y)
    se = v.sqrt()
    x = torch.stack([torch.ones(k, dtype=torch.float64), regressor], 1)
    scale = float(regressor.square().mean().sqrt())
    if not math.isfinite(scale) or scale <= 0:
        raise AnalysisError("singular_design", "The precision regressor is degenerate.")
    z = torch.stack([1/se, regressor/scale/se], 1)
    q, r = torch.linalg.qr(z)
    singular = torch.linalg.svdvals(z)
    if float(singular[-1]) <= float(singular[0])*1e-10:
        raise AnalysisError("singular_design", "Standard errors do not vary enough to identify the precision regression.")
    b = torch.linalg.solve_triangular(r, (q.T@(y/se))[:, None], upper=True).squeeze(1)
    residual = (y-x@torch.stack([b[0], b[1]/scale]))/se
    rss = float(residual.square().sum())
    dispersion = rss/(k-2)
    if dispersion <= 1e-20 or not math.isfinite(dispersion):
        raise AnalysisError("degenerate_inference", "The weighted residual dispersion is degenerate.")
    inverse = torch.linalg.solve_triangular(r, torch.eye(2, dtype=torch.float64), upper=True)
    unscale = torch.diag(torch.tensor([1., 1/scale], dtype=torch.float64))
    beta = unscale@b
    covariance = unscale@(inverse@inverse.T)@unscale*dispersion
    return beta, (covariance+covariance.T)/2, rss, dispersion


@torch.no_grad()
def meta_pet_peese(*, data, yi: str = "yi", vi: str = "vi", study: str | None = None, alpha: float = .05,
                   alternative: str = "two-sided", level: float = .95,
                   dependence: str = "independent") -> TableSet:
    """PET, PEESE and conditional PET-PEESE weighted precision regressions.

    PET: yi ~ 1 + sqrt(vi); PEESE: yi ~ 1 + vi; WLS weights 1/vi with
    unrestricted multiplicative dispersion and t(k-2) inference. The conditional
    estimate reports PEESE when the PET intercept test rejects at alpha (declared
    alternative), else PET; its interval ignores this selection step. >=10 studies.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise AnalysisError("invalid_spec", "alternative must be 'two-sided', 'greater' or 'less'.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("invalid_spec", "alpha must lie strictly between 0 and 1.")
    level = level_check(level)
    frame, values, ids, digest = sample(data, [yi, vi], study, dependence=dependence, minimum=10)
    y, v = values[yi], values[vi]
    if bool(((v < 1e-150) | (v > 1e150)).any()) or float(v.max()/v.min()) > 1e12:
        raise AnalysisError("invalid_variance", "Sampling variances must be positive, lie in [1e-150,1e150] and span <=1e12.")
    k = len(ids)
    df = k-2
    c = t_isf((1-level)/2, df)
    rows, covariances, fits = [], {}, {}
    for model, regressor, term in (("PET", v.sqrt(), "sei"), ("PEESE", v, "vi")):
        beta, covariance, rss, dispersion = _wls(y, v, regressor)
        se = covariance.diagonal().sqrt()
        for i, name in enumerate(("Intercept", term)):
            statistic = float(beta[i]/se[i])
            rows.append({"model": model, "term": name, "estimate": float(beta[i]), "std_error": float(se[i]),
                         "statistic": statistic, "df": df, "p_value": 2*t_sf(abs(statistic), df),
                         "ci_low": float(beta[i]-c*se[i]), "ci_high": float(beta[i]+c*se[i])})
        covariances[model] = table(covariance.tolist(), columns=["Intercept", term], index=["Intercept", term])
        fits[model] = {"model": model, "weighted_rss": rss, "dispersion": dispersion, "residual_df": df, "k": k}
    pet = rows[0]
    if alternative == "two-sided":
        p_pet = pet["p_value"]
    elif alternative == "greater":
        p_pet = t_sf(pet["statistic"], df)
    else:
        p_pet = t_sf(-pet["statistic"], df)
    selected = "PEESE" if p_pet < alpha else "PET"
    chosen = next(row for row in rows if row["model"] == selected and row["term"] == "Intercept")
    conditional = {"selected_model": selected, "rule": f"PEESE if PET intercept {alternative} p < alpha",
                   "alpha": float(alpha), "alternative": alternative, "pet_intercept_p_value": p_pet,
                   "estimate": chosen["estimate"], "std_error": chosen["std_error"], "ci_low": chosen["ci_low"],
                   "ci_high": chosen["ci_high"], "df": df}
    return TableSet({"coefficients": table(rows), "pet_covariance": covariances["PET"],
                     "peese_covariance": covariances["PEESE"], "conditional": table([conditional]),
                     "fit": table([fits["PET"], fits["PEESE"]])},
                    title="PET-PEESE precision-effect regressions", procedure="meta_pet_peese", sample_hash=digest,
                    level=level, n_studies=k, source=PET_SOURCE,
                    notes=["Weights are 1/vi with an unrestricted multiplicative dispersion (ordinary weighted least squares).",
                           "The conditional PET-PEESE interval does not account for the model-selection step.",
                           "Precision-effect associations do not establish publication bias or identify the true effect."])
