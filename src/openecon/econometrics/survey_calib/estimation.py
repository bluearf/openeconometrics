"""Calibrated survey totals, means and ratios with GREG-residual Taylor linearization.

For a target theta with row influence u_k (total: y_k; Hájek mean: (y_k - theta)/N^;
ratio: (y_k - theta x_k)/X^; times the domain indicator) evaluated at the calibrated
weights w_k = d_k g_k, the calibration-adjusted linearized value is

    z_k = w_k e_k,   e_k = u_k - x_k' B,   B = (sum_k d_k x_k x_k')^{-1} sum_k d_k x_k u_k,

the residual from the design-weighted regression of u on the calibration variables
(within by-groups), multiplied by the calibrated weight. This is the Deville–Särndal
(1992) g-weighted residual estimator and the convention of R survey's svyrecvar() for
calibrate() designs (``qr.resid(qr(X sqrt(d)), x / (g sqrt(d))) * g sqrt(d)``).
Uncalibrated targets use z_k = d_k u_k. The design covariance is the first-stage
with-replacement ultimate-cluster estimator with the declared FPC,

    V = sum_h (1 - f_h) m_h/(m_h - 1) sum_j (z_hj - zbar_h)(z_hj - zbar_h)',

over PSU totals z_hj (Stata svy Taylor; R svytotal with fpc= population PSU counts).
Inference uses Student t with all declared PSUs minus strata as degrees of freedom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from torch import Tensor

from openecon.analysis import _coerce_frame, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.engines.inference import critical_value, student_t_two_sided
from openecon.resources import plan_workspace, tensor_bytes
from openecon.survey import SurveyDesign, _scalar

from .calibration import FLOAT, MAX_ROWS, REFERENCE, Calibrated, calibrate, canonical_spec

MAX_TARGETS = 32
DEFF_REFERENCE = ("SRS without replacement of n units from N^ = sum of the (calibrated) weights: "
                  "V_srs = s_w^2(u) N^2 (N^ - n)/(N^ n), with s_w^2 the weighted variance of the target's "
                  "influence values (R survey svytotal/svymean/svyratio deff=TRUE)")
GREG = ("Deville-Särndal g-weighted residual linearization: z_k = w_k e_k, e_k the residual of the design-weighted "
        "least-squares regression of the influence on the calibration variables (R survey calibrate()/svyrecvar())")


def frame_of(data) -> pd.DataFrame:
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates or not len(frame) or len(frame) > MAX_ROWS:
        raise AnalysisError("invalid_data", f"Survey estimation needs 1..{MAX_ROWS:,} rows with unique column names.")
    return frame


def names(value, what: str, *, distinct: bool = True) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if (not isinstance(value, (list, tuple)) or not 1 <= len(value) <= MAX_TARGETS
            or any(not isinstance(v, str) or not v.strip() or len(v) > 200 for v in value)
            or distinct and len(set(value)) != len(value)):
        raise AnalysisError("invalid_survey_target", f"{what} needs 1..{MAX_TARGETS} distinct column names.")
    return list(value)


def options(alpha, missing, deff) -> float:
    value = _scalar(alpha)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1e-8 <= value <= 1 - 1e-8:
        raise AnalysisError("invalid_survey_option", "alpha must be a real number in (0, 1).")
    if not isinstance(missing, str) or missing not in ("raise", "drop"):
        raise AnalysisError("invalid_survey_option", "missing must be 'raise' or 'drop'.")
    if type(deff) is not bool:
        raise AnalysisError("invalid_survey_option", "deff must be True or False.")
    return float(value)


@dataclass
class Targets:
    kind: str
    labels: list[str]
    values: Tensor          # [n, k] numerators; zero outside the eligible rows
    denominators: Tensor    # [n, k]: ratio denominators, ones for means
    member: Tensor          # [n] 1.0 for eligible (in-domain, jointly complete) rows
    record: dict


def indicator(frame: pd.DataFrame, column: str) -> Tensor:
    series = frame[column]
    if bool(series.isna().any()):
        raise AnalysisError("invalid_survey_domain", "Domain membership must be a complete Boolean/0-1 column.")
    if is_bool_dtype(series.dtype):
        return torch.as_tensor(series.to_numpy(dtype="bool", copy=True))
    if not is_numeric_dtype(series.dtype):
        raise AnalysisError("invalid_survey_domain", "Domain membership must be a complete Boolean/0-1 column.")
    values = _numeric(series, column)
    if not bool(((values == 0) | (values == 1)).all()):
        raise AnalysisError("invalid_survey_domain", "Domain membership must be a complete Boolean/0-1 column.")
    return values == 1


def targets(frame: pd.DataFrame, kind: str, outcomes, denominators, domain, missing: str) -> Targets:
    """Eligible rows, numerators and denominators; every design row is retained."""
    outcomes = names(outcomes, "outcomes" if kind != "ratio" else "numerators", distinct=kind != "ratio")
    if kind == "ratio":
        denominators = names(denominators, "denominators", distinct=False)
        if len(denominators) != len(outcomes) or len(set(zip(outcomes, denominators))) != len(outcomes):
            raise AnalysisError("invalid_survey_target", "Declare one denominator per numerator, as distinct pairs.")
    elif denominators is not None:
        raise AnalysisError("invalid_survey_target", "Denominators are only used for ratio targets.")
    else:
        denominators = []
    if domain is not None and (not isinstance(domain, str) or not domain.strip()):
        raise AnalysisError("invalid_survey_domain", "domain must name a complete Boolean/0-1 indicator column.")
    roles = list(dict.fromkeys(outcomes + denominators + ([domain] if domain else [])))
    absent = [c for c in roles if c not in frame.columns]
    if absent:
        raise AnalysisError("missing_columns", f"Survey target columns are absent: {', '.join(absent)}.")
    n, k = len(frame), len(outcomes)
    plan_workspace("survey target linearization", {
        "targets, influence, residual and PSU buffers": tensor_bytes((n, k), itemsize=96),
    })
    member = indicator(frame, domain) if domain else torch.ones(n, dtype=torch.bool)
    variables = list(dict.fromkeys(outcomes + denominators))
    incomplete = torch.as_tensor(frame[variables].isna().any(axis=1).to_numpy(dtype="bool", copy=True)) & member
    if bool(incomplete.any()) and missing == "raise":
        raise AnalysisError("survey_missing", f"{int(incomplete.sum())} in-domain observation(s) have missing target "
                            "values; choose missing='drop' to exclude them from the estimand.")
    eligible = member & ~incomplete
    if not bool(eligible.any()):
        raise AnalysisError("empty_survey_domain", "No complete in-domain target observations remain.")
    rows = torch.nonzero(eligible).flatten()
    positions = rows.numpy()

    def block(columns):
        out = torch.zeros((n, len(columns)), dtype=FLOAT)
        for j, column in enumerate(columns):
            out[rows, j] = _numeric(frame[column].iloc[positions], column)
        return out

    values = block(outcomes)
    denominator = block(denominators) if kind == "ratio" else torch.ones((n, k), dtype=FLOAT)
    labels = [f"{y}/{x}" for y, x in zip(outcomes, denominators)] if kind == "ratio" else outcomes
    record = {"n_design": n, "n_domain": int(member.sum()), "n_used": int(eligible.sum()),
              "n_missing_excluded": int(incomplete.sum()), "domain": domain, "missing": missing,
              "outcomes": outcomes, "denominators": denominators or None}
    return Targets(kind, labels, values, denominator, eligible.to(FLOAT), record)


def linearize(t: Targets, w: Tensor) -> tuple[Tensor, Tensor]:
    """Estimate and row influence u_k at weights w (unscaled by w)."""
    effective = w * t.member
    numerator = effective @ t.values
    if t.kind == "total":
        theta, influence = numerator, t.values * t.member[:, None]
    else:
        denominator = effective @ t.denominators
        size = effective.abs() @ t.denominators.abs()
        if bool((size == 0).any()) or bool((denominator.abs() <= 1e-12 * size).any()):
            raise AnalysisError("survey_denominator", "A weighted mean/ratio denominator is zero or numerically cancelled.")
        theta = numerator / denominator
        influence = t.member[:, None] * (t.values - t.denominators * theta) / denominator
    if not bool(torch.isfinite(theta).all()) or not bool(torch.isfinite(influence).all()):
        raise AnalysisError("survey_numerical_failure", "Survey estimates overflowed; rescale the input explicitly.")
    return theta, influence


def resolve(calibration, frame: pd.DataFrame, d: Tensor) -> Calibrated | None:
    """Solve a calibration spec; a saved svycal result must reproduce on these rows."""
    if calibration is None:
        return None
    fit = calibrate(frame, d, canonical_spec(calibration))
    if isinstance(calibration, TableSet) and "weights" in calibration:
        saved = calibration["weights"]
        if (len(saved) != len(frame) or not {"design_weight", "g_weight"} <= set(saved.columns)
                or not torch.allclose(_numeric(saved["design_weight"], "design_weight"), d, rtol=1e-12, atol=0.0)
                or not torch.allclose(_numeric(saved["g_weight"], "g_weight"), fit.g, rtol=1e-8, atol=1e-10)):
            raise AnalysisError("calibration_changed", "The supplied svycal result does not reproduce on these rows "
                                "and design weights; rerun svycal on the same data and design.")
    return fit


def design_effects(t: Targets, w: Tensor, influence: Tensor, covariance: Tensor) -> list:
    """Variance ratios against the SRSWOR comparator of R survey's deff=TRUE (None if undefined)."""
    use = (t.member > 0) & (w != 0)
    n = int(use.sum())
    population = float(w[use].sum())
    if n < 2 or not population > n:
        return [None] * len(t.labels)
    weights, values = w[use], influence[use]
    mean = weights @ values / population
    spread = n / (n - 1) * (weights @ (values - mean) ** 2) / population
    reference = spread * population * (population - n) / n
    return [float(covariance[i, i] / reference[i]) if float(reference[i]) > 0 else None
            for i in range(len(t.labels))]


def inference_table(t: Targets, theta: Tensor, covariance: Tensor, df: int, alpha: float, effects, *, title: str,
                    **attrs):
    covariance = (covariance + covariance.T) / 2
    if not bool(torch.isfinite(covariance).all()):
        raise AnalysisError("survey_numerical_failure", "The design covariance overflowed; rescale the input.")
    critical = kernel_call(critical_value, alpha, df) if df > 0 else None
    rows, notes = [], []
    for i, label in enumerate(t.labels):
        estimate, variance = float(theta[i]), float(covariance[i, i])
        if variance < 0:
            # Horvitz-Thompson-form estimators are not guaranteed nonnegative; keep the raw value.
            notes.append(f"The variance estimate of '{label}' is negative ({variance:.6g}); no SE or interval.")
        se = math.sqrt(variance) if variance >= 0 else math.nan
        statistic = estimate / se if se > 0 and df > 0 else math.nan
        p = student_t_two_sided(statistic, df) if math.isfinite(statistic) else math.nan
        width = critical * se if critical is not None else 0.0
        rows.append([estimate, se, statistic, p, estimate - width, estimate + width]
                    + ([math.nan if effects[i] is None else effects[i]] if effects is not None else []))
    columns = ["estimate", "std_error", "statistic", "p_value", "ci_low", "ci_high"] + (["deff"] if effects else [])
    return table(rows, columns=columns, index=t.labels, title=title, target=t.kind, df=df, alpha=alpha,
                 covariance=covariance.tolist(), inference="design Student t (null 0); zero-SE or zero-df tests undefined",
                 **t.record, **({"deff_reference": DEFF_REFERENCE} if effects is not None else {}),
                 stata_parity_validated=False, notes=notes, **attrs)


def estimate(t: Targets, d: Tensor, fit: Calibrated | None, covariance_of, deff: bool):
    """Point estimates, calibration-adjusted linearized covariance and optional design effects."""
    w = d if fit is None else fit.weights
    theta, influence = linearize(t, w)
    residual = influence if fit is None else fit.residuals(influence)
    covariance = covariance_of(w[:, None] * residual)
    return theta, covariance, design_effects(t, w, influence, covariance) if deff else None


def calibration_record(fit: Calibrated | None) -> dict | None:
    return None if fit is None else {**fit.summary(), "spec": fit.spec, "linearization": GREG,
                                     "reference": REFERENCE}


# ---- single-stage SurveyDesign (with-replacement PSUs, FPC) --------------------------------


def taylor_geometry(frame: pd.DataFrame, design: SurveyDesign, validation):
    """PSU codes, stratum of each PSU and per-PSU variance factors (1 - f_h) m_h/(m_h - 1)."""
    keys = [design.strata, design.psu] if design.strata else [design.psu]
    psu = torch.as_tensor(frame.groupby(keys, sort=False).ngroup().to_numpy(dtype="int64", copy=True))
    strata = (torch.as_tensor(pd.factorize(frame[design.strata], sort=False)[0].astype("int64"))
              if design.strata else torch.zeros(len(frame), dtype=torch.int64))
    groups = int(psu.max()) + 1
    owner = torch.zeros(groups, dtype=torch.int64).index_copy_(0, psu, strata)
    counts = torch.bincount(owner).tolist()
    rows = torch.bincount(strata).tolist()
    if (groups != validation.n_psu or counts != [s.n_psu for s in validation.strata]
            or rows != [s.nobs for s in validation.strata]):
        raise AnalysisError("survey_design_identity", "PSU/stratum labels mix value types that the declaration keeps "
                            "distinct; recode each design column to one type and declare the design again.")
    factors = torch.tensor([0.0 if s.certainty or s.n_psu < 2 else
                            (1.0 - s.n_psu / s.population_psu if s.population_psu else 1.0) * s.n_psu / (s.n_psu - 1)
                            for s in validation.strata], dtype=FLOAT)
    return psu, owner, factors


def taylor_covariance(z: Tensor, psu: Tensor, owner: Tensor, factors: Tensor) -> Tensor:
    totals = torch.zeros((len(owner), z.shape[1]), dtype=FLOAT).index_add_(0, psu, z)
    strata = torch.zeros((len(factors), z.shape[1]), dtype=FLOAT).index_add_(0, owner, totals)
    counts = torch.bincount(owner, minlength=len(factors)).to(FLOAT)
    centred = totals - (strata / counts[:, None])[owner]
    return (centred * factors[owner][:, None]).T @ centred


def _calibrated(kind, data, design, outcomes, denominators, *, calibration, domain, missing, alpha, deff):
    if not isinstance(design, SurveyDesign):
        raise AnalysisError("invalid_survey_design", "Supply a validated single-stage SurveyDesign (oe.survey_design).")
    if calibration is None:
        raise AnalysisError("invalid_calibration_spec", "calibration is required: an svycal result or its arguments.")
    alpha = options(alpha, missing, deff)
    validation = design.revalidate(data)
    frame = frame_of(data)
    d = _numeric(frame[design.weights], design.weights)
    fit = resolve(calibration, frame, d)
    t = targets(frame, kind, outcomes, denominators, domain, missing)
    psu, owner, factors = taylor_geometry(frame, design, validation)
    theta, covariance, effects = estimate(t, d, fit, lambda z: taylor_covariance(z, psu, owner, factors), deff)
    return inference_table(
        t, theta, covariance, validation.design_df, alpha, effects, title=f"Calibrated survey {kind}",
        variance=("first-stage with-replacement PSU Taylor linearization with FPC (1 - m_h/N_h) and m_h/(m_h - 1); "
                  "certainty strata contribute zero"),
        df_convention="declared PSUs minus strata", calibration=calibration_record(fit),
        design={"weights": design.weights, "psu": design.psu, "strata": design.strata, "fpc": design.fpc,
                "n_psu": validation.n_psu, "n_strata": validation.n_strata,
                "design_input_sha256": validation.design_input_sha256})


def survey_calibrated_total(data, design, outcomes, *, calibration, domain=None, missing="raise", alpha=0.05,
                            deff=False):
    """Calibrated (GREG/raking/logit) survey totals with the calibration-adjusted Taylor variance.

    The total sum_k w_k y_k uses the calibrated weights w_k = d_k g_k of ``calibration``;
    its variance replaces y_k by the g-weighted residual w_k e_k of the design-weighted
    regression of y on the calibration variables (Deville and Särndal 1992; R survey
    ``svytotal`` on a ``calibrate()`` design) within the declared strata/PSU/FPC design.

    Parameters
    ----------
    data : table with every design row (domain rows are never subset).
    design : SurveyDesign declaring weights, PSUs, strata and FPC (revalidated).
    outcomes : column name or list of 1..32 numeric columns.
    calibration : svycal result, its ``attrs['calibration']`` or a mapping of svycal
        arguments (method, totals, categorical, population, bounds, by, max_iter, tol).
    domain : optional complete Boolean/0-1 column; out-of-domain rows keep their PSUs.
    missing : 'raise' (default) or 'drop' in-domain rows with missing outcomes.
    alpha : significance level of the design-t confidence intervals.
    deff : add the SRSWOR design effect of R survey's ``deff=TRUE``.

    Returns
    -------
    Table with estimate, std_error, statistic, p_value, ci_low, ci_high (and deff) per
    outcome; ``attrs`` holds the full covariance, df, sample counts, design and the
    calibration diagnostics/specification.
    """
    return _calibrated("total", data, design, outcomes, None, calibration=calibration, domain=domain,
                       missing=missing, alpha=alpha, deff=deff)


def survey_calibrated_mean(data, design, outcomes, *, calibration, domain=None, missing="raise", alpha=0.05,
                           deff=False):
    """Calibrated Hájek means sum_k w_k y_k / sum_k w_k with the GREG-residual Taylor variance.

    The mean's influence (y_k - mean)/N^ is residualized on the calibration variables
    and multiplied by the calibrated weight (R survey ``svymean`` on a calibrated design).

    Parameters
    ----------
    data : table with every design row.
    design : SurveyDesign declaring weights, PSUs, strata and FPC (revalidated).
    outcomes : column name or list of 1..32 numeric columns.
    calibration : svycal result, its ``attrs['calibration']`` or a mapping of svycal arguments.
    domain : optional complete Boolean/0-1 domain column.
    missing : 'raise' (default) or 'drop' in-domain rows with missing outcomes.
    alpha : significance level of the design-t confidence intervals.
    deff : add the SRSWOR design effect of R survey's ``deff=TRUE``.

    Returns
    -------
    Table of estimates and design-t inference; ``attrs`` as for ``survey_calibrated_total``.
    """
    return _calibrated("mean", data, design, outcomes, None, calibration=calibration, domain=domain,
                       missing=missing, alpha=alpha, deff=deff)


def survey_calibrated_ratio(data, design, numerators, denominators, *, calibration, domain=None, missing="raise",
                            alpha=0.05, deff=False):
    """Calibrated ratios of totals sum w y / sum w x with the GREG-residual Taylor variance.

    The ratio's influence (y_k - R x_k)/X^ is residualized on the calibration variables
    and multiplied by the calibrated weight (R survey ``svyratio`` on a calibrated design).

    Parameters
    ----------
    data : table with every design row.
    design : SurveyDesign declaring weights, PSUs, strata and FPC (revalidated).
    numerators, denominators : paired lists of numeric columns (one denominator each).
    calibration : svycal result, its ``attrs['calibration']`` or a mapping of svycal arguments.
    domain : optional complete Boolean/0-1 domain column.
    missing : 'raise' (default) or 'drop' in-domain rows missing a numerator or denominator.
    alpha : significance level of the design-t confidence intervals.
    deff : add the SRSWOR design effect of R survey's ``deff=TRUE``.

    Returns
    -------
    Table of ratios ('y/x') and design-t inference; ``attrs`` as for ``survey_calibrated_total``.
    """
    return _calibrated("ratio", data, design, numerators, denominators, calibration=calibration, domain=domain,
                       missing=missing, alpha=alpha, deff=deff)
