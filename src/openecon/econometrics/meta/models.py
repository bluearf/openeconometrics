"""Study-level common/random pooling, meta-regression and saved prediction."""
from __future__ import annotations

import json
import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.engines.distributions import chi2_sf, f_sf
from .common import SOURCES, checksum, critical, finite, level_check, probability, sample
from . import kernels


def _design(values, moderators, intercept, k):
    if type(intercept) is not bool:
        raise AnalysisError("invalid_spec", "intercept must be a boolean.")
    if not intercept and not moderators:
        raise AnalysisError("invalid_spec", "At least one predictor or an intercept is required.")
    parts = [torch.ones(k, dtype=torch.float64, device="cpu")] if intercept else []
    parts.extend(values[n] for n in moderators)
    original = torch.stack(parts, 1)
    normalized = original.clone()
    transform = torch.eye(len(parts), dtype=torch.float64, device="cpu")
    start = int(intercept)
    for i in range(start, len(parts)):
        center = float(original[:, i].mean()) if intercept else 0.
        scale = float((original[:, i]-center).square().mean().sqrt())
        if not math.isfinite(scale) or scale <= 0:
            raise AnalysisError("singular_design", "A moderator is constant or numerically unscaled.")
        normalized[:, i] = (original[:, i]-center)/scale
        transform[i, i] = 1/scale
        if intercept:
            transform[0, i] = -center/scale
    singular = torch.linalg.svdvals(normalized)
    if float(singular[-1]) <= float(singular[0])*1e-10:
        raise AnalysisError("singular_design", "Moderator columns are linearly dependent or ill-conditioned after scaling.")
    return original, normalized, transform


@torch.no_grad()
def meta_regress(*, data, yi: str = "yi", vi: str = "vi", moderators: list[str] | None = None,
                 study: str | None = None, intercept: bool = True, method: str = "REML",
                 inference: str = "z", level: float = .95, dependence: str = "independent") -> TableSet:
    """Fit independent study effects with known positive vi and numeric moderators.

    method: 'common', 'DL', 'ML', 'REML'. Inference: z, hksj (residual weighted
    RSS/(k-p) times GLS covariance), or modified_hksj (scale floored at 1).
    HK uses t(k-p) coefficients and F(m,k-p) moderator tests. Random-effects
    I² uses tau²/(tau²+typical sampling variance); common uses Cochran Q.
    ML/REML solves a bounded profiled variance likelihood on CPU float64.
    The design is centered/scaled, with full covariance transformed back.
    Known vi and independent study effects are caller assumptions. No implicit
    encoding, missing deletion, selection, multilevel or multivariate fitting.
    Returns coefficients, full covariance, studies, heterogeneity and tests,
    plus checksum-protected portable state for meta_predict. <=2000 studies,
    <=30 moderators; 128 MiB estimated workspace and 100 root iterations.
    """
    moderators = [] if moderators is None else moderators
    if not isinstance(moderators, list) or len(moderators) > 30:
        raise AnalysisError("invalid_spec", "moderators must be an explicit list of <=30 numeric columns.")
    if method not in ("common", "DL", "ML", "REML") or inference not in ("z", "hksj", "modified_hksj"):
        raise AnalysisError("invalid_spec", "Use common/DL/ML/REML and z/hksj/modified_hksj.")
    level = level_check(level)
    frame, values, ids, digest = sample(data, [yi, vi, *moderators], study, dependence=dependence, minimum=2)
    y, v = values[yi], values[vi]
    if bool(((v < 1e-150) | (v > 1e150)).any()) or float(v.max()/v.min()) > 1e12 or bool((y.abs() > 1e75).any()):
        raise AnalysisError("invalid_variance", "Sampling variances must lie in [1e-150,1e150], span <=1e12, and effects have magnitude <=1e75; rescale if needed.")
    original, x, transform = _design(values, moderators, intercept, len(ids))
    k, p = x.shape
    if k <= p:
        raise AnalysisError("insufficient_studies", "The model needs positive residual degrees of freedom (k>p).")
    result = kernels.fit(y, v, x, method)
    beta = transform@result["beta"]
    bread = transform@result["bread"]@transform.T
    df = k-p
    hk_scale = result["rss"]/df
    scale = 1. if inference == "z" else max(1., hk_scale) if inference == "modified_hksj" else hk_scale
    if not math.isfinite(scale) or scale <= 1e-20:
        raise AnalysisError("degenerate_inference", "The residual HK variance is degenerate; no HK standard errors can be reported.")
    covariance = (bread+bread.T)*(.5*scale)
    finite(covariance, "full coefficient covariance")
    if bool((covariance.diagonal() <= 0).any()):
        raise AnalysisError("numerical_failure", "Coefficient uncertainty is not positive; rescale moderators.")
    terms = [*(["Intercept"] if intercept else []), *moderators]
    if len(set(terms)) != len(terms):
        raise AnalysisError("invalid_spec", "A moderator cannot duplicate the Intercept term label.")
    c = critical(level, inference, df)
    se = covariance.diagonal().sqrt()
    stats = beta/se
    tau = result["tau2"]
    typical = result["typical_variance"]
    q = result["q0"]
    i2 = max(0., 100*(q-df)/q) if method == "common" and q > 0 else 0. if method == "common" else 100*tau/(tau+typical)
    h2 = q/df if method == "common" else 1+tau/typical
    tests = [{"test": "Cochran residual Q (vi only)", "statistic": q, "distribution": "chi2", "df1": df, "df2": None, "p_value": chi2_sf(q, df)}]
    subset = slice(int(intercept), p)
    m = p-int(intercept)
    if m:
        b, cv = beta[subset], covariance[subset, subset]
        wald = float(b@torch.linalg.solve(cv, b))
        test = {"test": "joint moderators", "statistic": wald if inference == "z" else wald/m,
                "distribution": "chi2" if inference == "z" else "F", "df1": m,
                "df2": None if inference == "z" else df,
                "p_value": chi2_sf(wald, m) if inference == "z" else f_sf(wald/m, m, df)}
        tests.append(test)
    state = {"schema": "openecon.meta.v1", "terms": terms, "moderators": moderators.copy(), "intercept": intercept,
             "beta": beta.tolist(), "covariance": covariance.tolist(), "tau2": tau, "method": method,
             "inference": inference, "residual_df": df, "level": level, "n_studies": k,
             "sample_hash": digest, "dependence": dependence, "device": "cpu", "dtype": "float64"}
    state["checksum"] = checksum(state)
    fit_values = original@beta
    uncertainty = torch.einsum("ip,pq,iq->i", original, covariance, original).clamp_min(0).sqrt()
    critical_z = critical(level, "z", None)
    studies = table({"study": ids, "yi": y.tolist(), "vi": v.tolist(), "weight": result["weights"].tolist(),
                     "weight_percent": (result["weights"]/result["weights"].sum()*100).tolist(),
                     "fitted": fit_values.tolist(), "residual": result["resid"].tolist(),
                     "ci_low": (y-critical_z*v.sqrt()).tolist(), "ci_high": (y+critical_z*v.sqrt()).tolist(),
                     "mean_ci_low": (fit_values-c*uncertainty).tolist(), "mean_ci_high": (fit_values+c*uncertainty).tolist()})
    coefficients = table({"term": terms, "estimate": beta.tolist(), "std_error": se.tolist(), "statistic": stats.tolist(),
                          "p_value": [probability(float(s), inference, df) for s in stats],
                          "ci_low": (beta-c*se).tolist(), "ci_high": (beta+c*se).tolist()})
    heterogeneity = table([{n: result[n] for n in ("tau2", "q0", "typical_variance", "loglik_ml", "loglik_reml")}
                          | {"residual_df": df, "I2_percent": i2, "H2": h2, "hk_scale_raw": hk_scale, "covariance_scale": scale}])
    return TableSet({"coefficients": coefficients,
                     "covariance": table(covariance.tolist(), columns=terms, index=terms),
                     "studies": studies, "heterogeneity": heterogeneity, "tests": table(tests)},
                    title="Independent-study meta-regression" if moderators else "Independent-study meta-analysis",
                    procedure="meta_regress" if moderators else "meta_pool", method=method, inference=inference,
                    level=level, n_studies=k, n_parameters=p, residual_df=df, sample_hash=digest,
                    prediction_state=state, dependence=dependence, source=SOURCES["model"],
                    variance_solver={"status": "converged", "iterations": result["iterations"], "evaluations": result["evaluations"],
                                     "algorithm": "profile score roots on bounded log grid, boundary compared" if method in ("ML", "REML") else "closed form"},
                    execution="CPU float64; in-memory; complete studies; <=128 MiB estimated workspace",
                    notes=["Known sampling variances and independent effects are assumed; associations of study moderators are not causal effects.",
                           "HK and approximate true-effect prediction use t(k-p); z inference uses normal limits. Tau² uncertainty is not integrated."])


def meta_pool(*, data, yi: str = "yi", vi: str = "vi", study: str | None = None,
              method: str = "REML", inference: str = "z", level: float = .95,
              dependence: str = "independent") -> TableSet:
    """Common/DL/ML/REML independent-study intercept-only pooling; see meta_regress.

    Returns full study weights, Q/I²/tau², coefficient covariance and portable
    prediction state. meta_predict supplies the approximate true-study PI.
    """
    return meta_regress(data=data, yi=yi, vi=vi, study=study, method=method, inference=inference,
                        level=level, dependence=dependence)


@torch.no_grad()
def meta_predict(result, *, data=None):
    """Predict conditional means and approximate latent true-study intervals.

    Accept a fitted TableSet, its prediction_state, or JSON-restored state.
    Full coefficient covariance is required and integrity checked. Moderator
    rows must contain the named numeric roles; an intercept-only prediction
    uses one implicit row when data=None. PI adds tau² to mean uncertainty,
    excludes future sampling error, and uses the fitted z or t(k-p) convention.
    Does not refit the model or alter input/state. <=2000 prediction rows.
    """
    state = result.attrs.get("prediction_state") if isinstance(result, TableSet) else result
    try:
        state = json.loads(json.dumps(state, allow_nan=False))
        expected_keys = {"schema", "terms", "moderators", "intercept", "beta", "covariance", "tau2", "method", "inference",
                         "residual_df", "level", "n_studies", "sample_hash", "dependence", "device", "dtype", "checksum"}
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ValueError
        digest = state.pop("checksum")
        if checksum(state) != digest or state["schema"] != "openecon.meta.v1" or state["dependence"] != "independent" or state["dtype"] != "float64" or state["device"] != "cpu":
            raise ValueError
        names, intercept = state["moderators"], state["intercept"]
        if not isinstance(names, list) or len(names) > 30 or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names) or type(intercept) is not bool:
            raise ValueError
        terms = [*(["Intercept"] if intercept else []), *names]
        if not terms or state["terms"] != terms or len(set(terms)) != len(terms) or state["method"] not in ("common", "DL", "ML", "REML") or state["inference"] not in ("z", "hksj", "modified_hksj"):
            raise ValueError
        if type(state["residual_df"]) is not int or type(state["n_studies"]) is not int or state["residual_df"] != state["n_studies"]-len(terms) or state["residual_df"] < 1:
            raise ValueError
        level_check(state["level"])
        if isinstance(state["tau2"], bool) or not math.isfinite(state["tau2"]) or state["tau2"] < 0 or (state["method"] == "common" and state["tau2"] != 0):
            raise ValueError
        beta = torch.tensor(state["beta"], dtype=torch.float64, device="cpu")
        covariance = torch.tensor(state["covariance"], dtype=torch.float64, device="cpu")
        p = len(terms)
        if beta.shape != (p,) or covariance.shape != (p, p) or not bool(torch.isfinite(beta).all()) or not bool(torch.isfinite(covariance).all()):
            raise ValueError
        if not torch.allclose(covariance, covariance.T, rtol=1e-12, atol=0) or int(torch.linalg.cholesky_ex(covariance)[1]) != 0:
            raise ValueError
    except (ValueError, TypeError, RuntimeError, OverflowError) as exc:
        raise AnalysisError("invalid_prediction_state", "Prediction requires intact versioned meta state with full positive-definite coefficient covariance.") from exc
    if names:
        frame, values, _, _ = sample(data, names)
        parts = [torch.ones(len(frame), dtype=torch.float64, device="cpu")] if intercept else []
        x = torch.stack([*parts, *[values[n] for n in names]], 1)
    else:
        if data is not None:
            raise AnalysisError("invalid_spec", "Intercept-only meta prediction uses data=None and returns one mean.")
        x = torch.ones((1, 1), dtype=torch.float64, device="cpu")
    mean = x@beta
    variance = torch.einsum("ip,pq,iq->i", x, covariance, x)
    finite(mean, "predictions")
    finite(variance, "prediction covariance")
    if bool((variance < 0).any()):
        raise AnalysisError("numerical_failure", "Conditional mean variance is negative; rescale prediction moderators.")
    se, pi_se = variance.sqrt(), (variance+state["tau2"]).sqrt()
    c = critical(state["level"], state["inference"], state["residual_df"])
    endpoints = torch.stack([mean-c*se, mean+c*se, mean-c*pi_se, mean+c*pi_se])
    finite(endpoints, "prediction intervals")
    return table({"estimate": mean.tolist(), "std_error": se.tolist(), "ci_low": endpoints[0].tolist(), "ci_high": endpoints[1].tolist(),
                  "pi_low": endpoints[2].tolist(), "pi_high": endpoints[3].tolist()},
                 procedure="meta_predict", sample_hash=state["sample_hash"], state_checksum=digest, level=state["level"],
                 inference=state["inference"], residual_df=state["residual_df"], source=SOURCES["prediction"],
                 interval_target="latent true effect; approximate; excludes future sampling variance and tau² estimation uncertainty")
