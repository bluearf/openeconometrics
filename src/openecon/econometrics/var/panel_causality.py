"""Dumitrescu--Hurlin (2012) heterogeneous panel predictive noncausality."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import torch

from openecon.analysis import _coerce_frame, _frame_hasher, _numeric
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call, table
from openecon.econometrics.unitroot.common import check_count, check_magnitude, json_label
from openecon.econometrics.var.common import check_alpha
from openecon.econometrics.var.toda_yamamoto import _ordered_sample
from openecon.engines.distributions import chi2_sf

SOURCE = "https://doi.org/10.1016/j.econmod.2012.02.014"
_MAX_WORK_BYTES = 512 * 1024**2
_MAX_QR_WORK = 2_000_000_000


def _panel_sample(data, y, x, panel, time, p):
    names = [y, x, panel, time]
    if not all(isinstance(name, str) and name for name in names) or len(set(names)) != 4:
        raise AnalysisError("invalid_spec", "y, x, panel and time must name distinct nonempty columns.")
    frame = _coerce_frame(data)
    if frame.columns.has_duplicates:
        raise AnalysisError("duplicate_columns", "Data must have unique column names.")
    absent = [name for name in names if name not in frame]
    if absent:
        raise AnalysisError("missing_columns", f"Required columns are absent: {', '.join(absent)}.")
    # Count copies of model inputs, codes, sorting keys and dataframe output
    # before selecting/resetting the model-input table.
    input_bytes = sum(int(frame[name].memory_usage(index=False, deep=True)) for name in names)
    if 4 * input_bytes + len(frame) * 48 > _MAX_WORK_BYTES:
        raise AnalysisError("work_budget_exceeded", "Panel model-input selection exceeds the "
                            "bounded 512 MiB workspace. No automatic sample collection or truncation.")
    frame = frame.loc[:, names].reset_index(drop=True)
    if not len(frame):
        raise AnalysisError("empty_data", "The panel contains no observations.")
    if bool(frame.isna().any().any()):
        raise AnalysisError("missing_values", "Panel causality requires complete observations; "
                            "no missing rows or units are silently dropped.")
    if bool(frame.duplicated([panel, time]).any()):
        raise AnalysisError("repeated_time_values", "Every panel/time pair must be unique.")
    try:
        codes, labels = pd.factorize(frame[panel], sort=True)
    except TypeError:
        raise AnalysisError("invalid_panel", "Panel labels must be mutually sortable scalar values.") from None
    counts = pd.Series(codes).value_counts()
    count, total = len(labels), int(counts.iloc[0])
    if count < 2:
        raise AnalysisError("insufficient_panels", "At least two panel units are required.")
    if not bool((counts == total).all()):
        raise AnalysisError("unbalanced_panel", "This validated route requires a balanced panel.")
    if total <= 3 * p + 5:
        raise AnalysisError("insufficient_observations", "Dumitrescu--Hurlin Ztilde requires "
                            "more than 3*lags+5 complete periods in every unit.")
    usable, m = total - p, 2 * p + 1
    work_bytes = 4 * input_bytes + 8 * count * (8 * usable * m + 4 * total * 2 + 8 * m * m)
    if work_bytes > _MAX_WORK_BYTES or count * usable * m * m > _MAX_QR_WORK:
        raise AnalysisError("work_budget_exceeded", "Panel causality exceeds its bounded "
                            "512 MiB numerical workspace or QR work budget; reduce lags or sample size.")
    order = pd.Series(codes).sort_values(kind="stable").index
    frame = frame.iloc[order].reset_index(drop=True)
    parts, clock = [], None
    for unit in range(count):
        part = _ordered_sample(frame.iloc[unit * total:(unit + 1) * total], [y, x], time)
        if clock is None:
            clock = part[time]
        elif not part[time].equals(clock):
            raise AnalysisError("unaligned_panel", "Every unit must share the same complete regular time grid.")
        part[panel] = labels[unit]
        parts.append(part.loc[:, names])
    chosen = pd.concat(parts, ignore_index=True)
    levels = torch.stack([_numeric(chosen[name], name) for name in (y, x)], dim=1)
    check_magnitude(levels, [y, x])
    return chosen, levels.reshape(count, total, 2), [json_label(v) for v in labels], work_bytes


def _wald_batch(levels: torch.Tensor, p: int):
    """QR covariance and partial-regression Wald, all units in one batch."""
    count, total, _ = levels.shape
    usable, m = total - p, 2 * p + 1
    z = torch.stack([levels[:, p - j:total - j, v]
                     for v in range(2) for j in range(1, p + 1)], dim=2)
    mean_z, mean_y = z.mean(1), levels[:, p:, 0].mean(1)
    z = torch.cat([z - mean_z[:, None], torch.ones((count, usable, 1), dtype=torch.float64)], dim=2)
    outcome = levels[:, p:, 0] - mean_y[:, None]
    scales = torch.linalg.vector_norm(z, dim=1)
    if bool((scales == 0).any()):
        raise AnalysisError("collinear_regressors", "At least one unit has a constant or unidentified lag series.")
    scaled = z / scales[:, None]
    q, r = torch.linalg.qr(scaled, mode="reduced")
    if bool((r.diagonal(dim1=1, dim2=2).abs() <= 1e-10).any()) \
            or bool((torch.linalg.cond(r) > 1e10).any()):
        raise AnalysisError("collinear_regressors", "A unit's lag design is rank deficient or numerically singular.")
    projected = q.transpose(1, 2) @ outcome[:, :, None]
    beta = torch.linalg.solve_triangular(r, projected, upper=True)[:, :, 0] / scales
    residual = outcome - (q @ projected)[:, :, 0]
    ssr = residual.square().sum(1)
    if bool((ssr <= 1e-20 * outcome.square().sum(1)).any()):
        raise AnalysisError("perfect_fit", "At least one unit is fitted exactly; its Wald inference is unidentified.")
    inverse = torch.linalg.solve_triangular(r, torch.eye(m, dtype=torch.float64).expand(count, m, m), upper=True)
    inverse = inverse / scales[:, :, None]
    bread = inverse @ inverse.transpose(1, 2)
    covariance = bread * (ssr / (usable - m))[:, None, None]
    tested = beta[:, p:2 * p, None]
    restricted_covariance = covariance[:, p:2 * p, p:2 * p]
    factor, info = torch.linalg.cholesky_ex(restricted_covariance)
    if bool((info != 0).any()):
        raise AnalysisError("singular_test_covariance", "At least one unit's restriction covariance is singular.")
    statistics = (tested.transpose(1, 2) @ torch.cholesky_solve(tested, factor))[:, 0, 0]
    if not bool(torch.isfinite(statistics).all()) or bool((statistics < 0).any()):
        raise AnalysisError("numerical_failure", "A unit produced nonfinite Wald inference.")
    # Return the original-level intercept and its covariance.
    beta[:, -1] += mean_y - (beta[:, :-1] * mean_z).sum(1)
    change = torch.eye(m, dtype=torch.float64).expand(count, m, m).clone()
    change[:, -1, :-1] = -mean_z
    covariance = change @ covariance @ change.transpose(1, 2)
    return statistics, beta, covariance, ssr / (usable - m)


def dhcausality(*, data: Any, y: str, x: str, panel: str, time: str,
                lags: int = 1, integration_order: int = 0,
                cross_section: str = "independent", alpha: float = 0.05) -> TableSet:
    """Dumitrescu--Hurlin homogeneous noncausality in a heterogeneous panel.

    Null: x's first lags coefficients are zero in y's equation in EVERY unit;
    alternative: at least some units have predictive causality. Slopes and
    intercepts differ by unit. Input must be complete, balanced and share a
    regular time grid, with more than 3*lags+5 periods. Only a common caller-
    chosen fixed lag order and intercept are supported. No units are skipped.

    W_i is the classical df-corrected Wald (lags*F_i), Wbar its mean.
    Zbar = sqrt(N/(2K))*(Wbar-K). Ztilde uses the paper's approximate finite-T
    moments, where paper T is USABLE periods after removing K initial lags.
    Both use upper standard-normal tails, the original paper's rejection
    rule; plm/xtgcause two-sided software conventions differ. Wbar carries no
    invented p-value. Ztilde is the designated decision and Zbar a diagnostic.

    Normal reference inference assumes jointly stationary correctly specified
    dynamics and cross-section independent iid homoskedastic Gaussian errors.
    integration_order=0 and cross_section='independent' declare these scopes;
    other scopes are rejected. Finite-T moments are approximations for dynamic
    regressors, not exact fixed-T validity. No lag/pretest selection, trend,
    weights, cross-dependent bootstrap, or structural causal claim is added.

    Bounded in-memory CPU float64 batched QR returns tests, individual tests and
    coefficients, with complete sample/hash/df/null provenance and LaTeX output.

    Parameters
    ----------
    data : table
        Complete balanced in-memory panel observations.
    y, x : str
        Effect and cause columns; direction is x to y in each unit.
    panel, time : str
        Unit identity and shared regular calendar or consecutive integer clock.
    lags : int
        Common caller-chosen fixed lag order K >= 1.
    integration_order : int
        Declared upper integration order; only stationary order 0 is validated.
    cross_section : str
        Only 'independent' normal-reference scope is validated.
    alpha : float
        Significance level for the designated Ztilde upper-tail decision.

    Returns
    -------
    TableSet
        Wbar/Zbar/Ztilde, individual Wald diagnostics, and unit coefficients.
    """
    p = check_count(lags, "lags", minimum=1)
    alpha = check_alpha(alpha)
    if check_count(integration_order, "integration_order") != 0:
        raise AnalysisError("unsupported_integration_order", "Dumitrescu--Hurlin's validated "
                            "normal-reference route supports declared stationary series only.")
    if cross_section != "independent":
        raise AnalysisError("unsupported_dependence", "Normal-reference panel causality needs "
                            "declared cross-section independence; dependent bootstrap is unavailable.")
    with torch.device("cpu"), torch.no_grad():
        chosen, levels, labels, work_bytes = _panel_sample(data, y, x, panel, time, p)
        count, total, _ = levels.shape
        usable, m = total - p, 2 * p + 1
        wi, beta, covariance, variance = _wald_batch(levels, p)
        wbar = float(wi.mean())
        zbar = math.sqrt(count / (2 * p)) * (wbar - p)
        # Equations (9)--(10) in the 2012 journal paper. Substitute usable
        # T = raw periods - K, not raw periods (plm implementation L100--105).
        mean_w = p * (usable - 2 * p - 1) / (usable - 2 * p - 3)
        variance_w = (2 * p * (usable - 2 * p - 1) ** 2 * (usable - p - 3)
                      / ((usable - 2 * p - 3) ** 2 * (usable - 2 * p - 5)))
        ztilde = math.sqrt(count / variance_w) * (wbar - mean_w)
        tests = [{"test": "Wbar", "statistic": wbar, "p_value": None,
                  "reject": None, "reference": "intermediate mean Wald"}]
        for name, statistic in (("Zbar", zbar), ("Ztilde", ztilde)):
            probability = .5 * math.erfc(statistic / math.sqrt(2))
            tests.append({"test": name, "statistic": statistic, "p_value": probability,
                          "reject": probability < alpha, "reference": "upper normal approximation"})
        individual = [{"unit": label, "statistic": float(wi[i]), "df": p,
                       "p_value": kernel_call(chi2_sf, float(wi[i]), p),
                       "observations": usable, "residual_df": usable - m,
                       "residual_variance": float(variance[i])}
                      for i, label in enumerate(labels)]
        terms = [f"L{j}.{name}" for name in (y, x) for j in range(1, p + 1)] + ["Intercept"]
        coefficients = [{"unit": label, "term": term, "estimate": float(beta[i, j]),
                         "std_error": math.sqrt(float(covariance[i, j, j]))}
                        for i, label in enumerate(labels) for j, term in enumerate(terms)]
        tests_table = table(tests, title="Dumitrescu--Hurlin panel tests")
        tests_table["p_value"] = pd.Series([row["p_value"] for row in tests], dtype=object)
        return TableSet({"tests": tests_table,
                         "individual": table(individual, title="Unit Wald tests", distribution="asymptotic chi2"),
                         "coefficients": table(coefficients, title="Heterogeneous unit coefficients")},
                        title="Dumitrescu--Hurlin panel noncausality", source=SOURCE,
                        effect=y, cause=x, panel=panel, time=time, lags=p, units=count,
                        periods=total, usable_periods=usable, residual_df=usable - m,
                        wbar=wbar, zbar=zbar, ztilde=ztilde,
                        observations=count * usable, regressors_per_unit=m, alpha=alpha,
                        mean_w_approximation=mean_w, variance_w_approximation=variance_w,
                        covariance="classical SSR/residual_df", covariance_divisor=usable - m,
                        distribution="normal approximation", tail="upper", decision_test="Ztilde",
                        integration_order=0, cross_section="independent", skipped_units=0,
                        sample_sha256=_frame_hasher(chosen).hexdigest(),
                        sample_order=f"sorted by {panel}, {time}", device="cpu", dtype="float64",
                        estimated_workspace_bytes=work_bytes,
                        null="No predictive causality from x to y in every unit",
                        alternative="Predictive causality from x to y in at least some units",
                        inference_scope="stationary common fixed-lag panel; independent iid Gaussian innovations",
                        notes=["Ztilde uses approximate moments; dynamic regressors prevent an exact fixed-T law.",
                               "Zbar needs large T first, then large N; Ztilde is the designated decision.",
                               "Upper tails follow the paper; plm/xtgcause use a two-sided software convention.",
                               "Panel rejection does not imply causality in every unit."])
