"""Formal observed-control benchmarks and max-coordinate OVB robustness.

Primary reference: Cinelli and Hazlett (2020), https://doi.org/10.1111/rssb.12348,
sections4.3/4.4 and supplementA.4/B.2. Author implementations:
https://github.com/carloscinelli/sensemakr/blob/master/R/ovb_bounds.R and
https://github.com/carloscinelli/sensemakr/blob/master/R/sensitivity_stats.R.
These are sensitivity summaries of an intact original classical OLS result,
not validation of empirical unobserved-confounding assumptions.
"""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Integral

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import common as c
from openecon.engines.distributions import t_ppf, t_sf
from openecon.engines.linalg import cholesky_solve, least_squares
from openecon.resources import plan_workspace

from . import common as m
from .ovb import _failure


def _grid(values, name, *, alpha=False):
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 64:
        raise AnalysisError("invalid_option", f"{name} requires a list of 1 to 64 numbers.")
    numbers = [c.check_number(v, name, minimum=0, maximum=1 if alpha else None) for v in values]
    if (alpha and any(v == 0 for v in numbers)) or any(
        a >= b for a, b in zip(numbers, numbers[1:])
    ):
        raise AnalysisError(
            "invalid_option", f"{name} must increase strictly; alpha lies in (0,1]."
        )
    return numbers


def _base(result, max_work):
    if not isinstance(result, TableSet) or result.attrs.get("procedure") != "ovb_sensitivity":
        raise AnalysisError("invalid_result", "Supply an intact original ovb_sensitivity TableSet.")
    state = result.attrs.get("state")
    if not isinstance(state, dict):
        raise AnalysisError("invalid_result", "The original OVB state is missing.")
    sample, settings = state.get("sample"), state.get("settings")
    n = sample.get("n") if isinstance(sample, dict) else None
    terms = state.get("terms")
    if (
        not isinstance(n, Integral)
        or isinstance(n, bool)
        or not 4 <= n <= 100000
        or not isinstance(terms, list)
        or not 2 <= len(terms) <= 64
        or not isinstance(settings, dict)
    ):
        raise AnalysisError(
            "invalid_result", "Original OVB sample or model dimensions are invalid."
        )
    p = len(terms)
    controls = settings.get("controls")
    if (
        not isinstance(controls, list)
        or terms != ["_cons", settings.get("treatment"), *controls]
        or state.get("df") != n - p
        or n - p < 2
        or settings.get("confounder_dimension") != 1
        or settings.get("covariance_type") != "classical_homoskedastic"
    ):
        raise AnalysisError(
            "invalid_result",
            "The input must preserve the original single-confounder classical OLS contract.",
        )
    base_cells = len(state.get("sensitivity_rows", []))
    integrity_work = 8 * n * p + 32 * base_cells + 8 * p * p
    m.work(integrity_work, max_work, "complete original OVB integrity and state retention")
    integrity_plan = plan_workspace(
        "original OVB postestimation integrity",
        {
            "complete_base_numeric_and_json_copies": 512 * n * p + 512 * base_cells + 128 * p * p,
        },
    )
    try:
        artifact = m.causal_design_save(result)
    except (TypeError, ValueError, KeyError) as exc:
        raise AnalysisError(
            "invalid_result", "The original OVB integrity record is invalid."
        ) from exc
    # Shape/type checks supplement the checksummed transport contract. No new
    # model is fitted or sample selected by robustness postestimation.
    arrays = [
        ("outcome", n, None),
        ("regressors", n, p - 1),
        ("standardized_design", n, p),
        ("covariance", p, p),
        ("scaled_covariance", p, p),
    ]
    for key, rows, columns in arrays:
        value = state.get(key)
        if (
            not isinstance(value, list)
            or len(value) != rows
            or (
                columns is not None
                and any(not isinstance(row, list) or len(row) != columns for row in value)
            )
        ):
            raise AnalysisError("invalid_result", f"Original OVB {key} dimensions are invalid.")
    if (
        not {"covariance", "coefficients"} <= result.keys()
        or list(result["covariance"].columns) != terms
        or list(result["covariance"].index) != terms
        or list(result["coefficients"].index) != terms
        or result["covariance"].to_numpy().tolist() != state["covariance"]
        or result["coefficients"].estimate.tolist() != state.get("coefficients")
    ):
        raise AnalysisError(
            "invalid_result", "Original OVB coefficient tables and scientific state disagree."
        )
    return state, artifact, integrity_work, integrity_plan.record()


def _finish(name, result, artifact, settings, tables, state, work, plan, integrity_plan):
    base = artifact["payload"]["attrs"]["state"]
    metadata = deepcopy(base["sample"])
    metadata.update(
        max_work=settings["max_work"],
        postestimation_work_units=work,
        postestimation_resource_plan=plan.record(),
        integrity_resource_plan=integrity_plan,
    )
    state.update(
        base_artifact=artifact,
        base_artifact_sha256=artifact["sha256"],
        base_state_sha256=artifact["payload"]["attrs"]["state_sha256"],
        base_tables_sha256=artifact["payload"]["attrs"]["tables_sha256"],
        terms=base["terms"],
        coefficients=base["coefficients"],
        covariance=base["covariance"],
        df=base["df"],
        original_sample_unchanged=True,
        original_table_dtypes={k: v["dtypes"] for k, v in artifact["payload"]["tables"].items()},
    )
    tables.update(
        coefficients=result["coefficients"].copy(deep=True),
        covariance=result["covariance"].copy(deep=True),
    )
    return m.result(
        name,
        tables,
        metadata,
        settings,
        state,
        notes=[
            "Original complete sample, coefficients and covariance retained from the integrity-checked base OVB artifact.",
            "Sensitivity assumptions are declared rather than inferred from unobserved data; vendor/whole-product parity is not claimed.",
        ],
    )


def _group_strength(beta, inverse, indices, ssr):
    coefficients = beta[indices]
    block = inverse[indices][:, indices]
    explained = float(coefficients @ cholesky_solve(block, coefficients))
    if not math.isfinite(explained) or explained < 0:
        _failure("Group explained variation is not representable as nonnegative float64.")
    if bool((coefficients != 0).any()) and explained == 0:
        _failure("Nonzero group explained variation underflows float64.")
    strength = explained / (explained + ssr)
    if not math.isfinite(strength) or strength >= 1 or (explained > 0 and strength == 0):
        _failure("The informative observed group partial R2 is not representable in [0,1).")
    return strength, explained


def _formal_bound(a, b, kd, ky):
    d = kd * (a / (1 - a))
    denominator = (1 - kd * a) * (1 - a)
    if not math.isfinite(d) or d >= 1 or denominator <= 0:
        raise AnalysisError(
            "outside_support",
            "A benchmark multiplier implies treatment strength outside [0,1); reduce kd.",
        )
    # Avoid squaring a before multiplying by kd: small positive observed
    # strength must not silently disappear from a formal bound.
    u_root = math.sqrt(kd) * a / math.sqrt(denominator)
    u = u_root * u_root
    if not math.isfinite(u) or u >= 1:
        raise AnalysisError(
            "outside_support",
            "A benchmark multiplier implies auxiliary strength outside [0,1); reduce kd.",
        )
    if kd > 0 and a > 0 and (d == 0 or u == 0):
        _failure("A nonzero formal benchmark strength is not representable.")
    y_root = (math.sqrt(ky) + u_root) / math.sqrt(1 - u) * math.sqrt(b / (1 - b))
    y = y_root * y_root
    if not math.isfinite(y) or y >= 1:
        raise AnalysisError(
            "outside_support",
            "The outcome strength bound is uninformative (>=1); reduce kd/ky. No clipping is performed.",
        )
    if b > 0 and (ky > 0 or u > 0) and y == 0:
        _failure("A nonzero outcome benchmark bound underflows float64.")
    return d, u, y


@m.procedure
def ovb_benchmark(
    result,
    *,
    groups,
    kd=(1.0,),
    ky=(1.0,),
    assumption=None,
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Formal sensitivity strength/bias bounds calibrated to observed controls.

    Declare ``assumption='residualized_benchmark'``. Groups are named nonempty
    subsets of the original controls. The hypothetical Z is residualized on
    all controls, and kd/ky bound its strengths relative to each omitted group
    in the corresponding reduced conditioning sets. Values implied outside
    informative [0,1) support are refused, not clipped. No confidence interval
    or worst-case SE is claimed at a strength-box corner.
    """
    m.options(device, weights, max_work)
    c.check_choice(assumption, "assumption", ("residualized_benchmark",))
    kd_grid, ky_grid = _grid(kd, "kd"), _grid(ky, "ky")
    if not isinstance(groups, dict) or not 1 <= len(groups) <= 32:
        raise AnalysisError(
            "invalid_spec", "groups requires 1 to 32 named nonempty control groups."
        )
    checked = {
        c.check_name(label, "group label"): c.name_list(names, "group controls")
        for label, names in groups.items()
    }
    base, artifact, integrity_work, integrity_plan = _base(result, max_work)
    controls, terms = base["settings"]["controls"], base["terms"]
    if any(not set(names) <= set(controls) for names in checked.values()):
        raise AnalysisError(
            "invalid_spec", "Every benchmark group must contain only original control columns."
        )
    n, p = base["sample"]["n"], len(terms)
    cells = len(checked) * len(kd_grid) * len(ky_grid)
    work = integrity_work + 4 * n * p * p + 16 * len(checked) * p**3 + 64 * cells
    m.work(work, max_work, "complete observed-group benchmark models and multiplier grid")
    plan = plan_workspace(
        "formal OVB observed-control benchmarks",
        {
            "retained_complete_original_artifact": integrity_plan["estimated_workspace_bytes"],
            "native_qr_and_selected_sample": 128 * n * p,
            "group_matrices_and_complete_bound_state": 256 * len(checked) * p * p + 256 * cells,
        },
    )
    x = torch.tensor(base["standardized_design"], dtype=torch.float64, device="cpu")
    d_columns = [0, *range(2, p)]
    d_fit = least_squares(x[:, d_columns], x[:, 1], drop_collinear=False)
    d_ssr = float(d_fit.ssr)
    if d_ssr <= 0:
        _failure("The original treatment has no residual variation conditional on controls.")
    y_ssr = base["scaled_residual_sum_of_squares"]
    y_beta = torch.tensor(base["scaled_coefficients"], dtype=torch.float64, device="cpu")
    y_inverse = torch.tensor(base["scaled_covariance"], dtype=torch.float64, device="cpu") / (
        y_ssr / base["df"]
    )
    estimate = base["coefficients"][1]
    se = math.sqrt(base["covariance"][1][1])
    observed, rows, group_state = [], [], []
    for label, names in checked.items():
        y_indices = [terms.index(name) for name in names]
        d_indices = [d_columns.index(index) for index in y_indices]
        a, qd = _group_strength(d_fit.beta, d_fit.xtx_inv, d_indices, d_ssr)
        b, qy = _group_strength(y_beta, y_inverse, y_indices, y_ssr)
        observed.append([label, len(names), a, b])
        group_state.append(
            dict(
                group=label,
                controls=names,
                treatment_indices=d_indices,
                outcome_indices=y_indices,
                treatment_explained_scaled_ss=qd,
                outcome_explained_scaled_ss=qy,
                treatment_reduced_scaled_ssr=d_ssr + qd,
                outcome_reduced_scaled_ssr=y_ssr + qy,
                observed_r2_treatment=a,
                observed_r2_outcome=b,
            )
        )
        for k_d in kd_grid:
            for k_y in ky_grid:
                rd, auxiliary, ry = _formal_bound(a, b, k_d, k_y)
                bias = (
                    se * math.sqrt(base["df"]) * math.sqrt(rd) * math.sqrt(ry) / math.sqrt(1 - rd)
                )
                lower, upper = estimate - bias, estimate + bias
                if (
                    not all(math.isfinite(v) for v in (bias, lower, upper))
                    or (rd > 0 and ry > 0 and bias == 0)
                    or (bias > 0 and (lower == estimate or upper == estimate))
                ):
                    _failure(
                        "A nonzero benchmark bias bound or endpoint shift is not representable."
                    )
                rows.append([label, k_d, k_y, rd, auxiliary, ry, bias, lower, upper])
    settings = dict(
        groups=checked,
        kd=kd_grid,
        ky=ky_grid,
        assumption=assumption,
        device=device,
        weights=None,
        max_work=max_work,
    )
    state = dict(
        target="formal observed-control partial-R2 strength and plug-in coefficient bounds",
        observed_groups=group_state,
        bound_rows=rows,
        treatment_regression_terms=[terms[i] for i in d_columns],
        treatment_regression_standardized_coefficients=d_fit.beta.tolist(),
        treatment_regression_standardized_covariance=(
            d_fit.xtx_inv * (d_ssr / (n - len(d_columns)))
        ).tolist(),
        treatment_regression_scaled_ssr=d_ssr,
        treatment_regression_df=n - len(d_columns),
        outcome_standardized_inverse=y_inverse.tolist(),
        benchmark_strength_covariance=None,
        confidence_interval=None,
        assumptions=[
            "Z is the part of the hypothetical omitted scalar confounder orthogonal to every observed control.",
            "kd bounds R2(D~Z|X-minus-group)/R2(D~group|X-minus-group).",
            "ky bounds R2(Y~Z|D,X-minus-group)/R2(Y~group|D,X-minus-group).",
            "Ratios and group choice are substantive sensitivity assumptions, not evidence that omitted confounding is bounded.",
        ],
        interpretation="Strength bounds imply a plug-in absolute coefficient-bias bound; no joint confidence or corner-SE/CI guarantee.",
    )
    return _finish(
        "ovb_benchmark",
        result,
        artifact,
        settings,
        {
            "benchmarks": m.frame(
                observed,
                columns=["group", "group_size", "r2_treatment_observed", "r2_outcome_observed"],
            ),
            "bounds": m.frame(
                rows,
                columns=[
                    "group",
                    "kd",
                    "ky",
                    "r2_treatment_bound",
                    "auxiliary_r2",
                    "r2_outcome_bound",
                    "bias_bound",
                    "estimate_lower",
                    "estimate_upper",
                ],
            ),
        },
        state,
        work,
        plan,
        integrity_plan,
    )


def _threshold(f, critical_f):
    if f <= critical_f:
        return 0.0, 0.0, 0.0, "already_nonrejected"
    h = f - critical_f
    diagonal = h / (0.5 * math.hypot(h, 2) + 0.5 * h)
    if critical_f > 0 and f > 1 / critical_f:
        scale = math.hypot(1, f)
        first, second = f / scale, critical_f / scale
        rho = (first - second) * (first + second)
        if rho >= 1:
            _failure("The max-coordinate robustness threshold rounds to the excluded endpoint one.")
        root_y = math.sqrt(rho) / (math.sqrt(1 - rho) * f)
        ry = root_y * root_y
        regime = "interior"
    else:
        rho, ry, regime = diagonal, diagonal, "diagonal"
    if (
        not all(math.isfinite(v) for v in (rho, ry, diagonal))
        or not 0 < rho < 1
        or not 0 < ry <= rho
        or diagonal >= 1
    ):
        _failure(
            "Robustness threshold or witness strengths are not representable in informative support."
        )
    return rho, diagonal, ry, regime


@m.procedure
def ovb_robustness(
    result, *, q=(0.5, 1.0), alpha=(0.05, 1.0), device="cpu", weights=None, max_work=100_000_000
):
    """Minimum max-coordinate confounder strength to change an OLS conclusion.

    The reference coefficient is (1-q)*the original estimate. At alpha<1 the
    threshold makes the adjusted two-sided t test nonreject that reference;
    alpha=1 gives point-estimate tipping. Both partial R2 strengths must be <=
    the returned robustness_value. A rare interior optimum is distinguished
    from equal-strength diagonal tipping. q/alpha are sensitivity queries,
    not data-adaptive confidence guarantees for the reference coefficient.
    """
    m.options(device, weights, max_work)
    qs, alphas = _grid(q, "q"), _grid(alpha, "alpha", alpha=True)
    base, artifact, integrity_work, integrity_plan = _base(result, max_work)
    cells = len(qs) * len(alphas)
    work = integrity_work + 128 * cells
    m.work(work, max_work, "complete robustness queries and reconstructed witnesses")
    plan = plan_workspace(
        "OVB max-coordinate robustness grid",
        {
            "retained_complete_original_artifact": integrity_plan["estimated_workspace_bytes"],
            "complete_query_and_witness_state": 1024 * cells,
            "retained_original_covariance": 128 * len(base["terms"]) ** 2,
        },
    )
    estimate, variance, df = base["coefficients"][1], base["covariance"][1][1], base["df"]
    se = math.sqrt(variance)
    orientation = 1 if estimate >= 0 else -1
    rows, witnesses = [], []
    for fraction in qs:
        reference = (1 - fraction) * estimate
        f = (abs(estimate / se) / math.sqrt(df)) * fraction
        if (
            not math.isfinite(reference)
            or not math.isfinite(f)
            or (fraction > 0 and estimate != 0 and (reference == estimate or f == 0))
        ):
            _failure("A nonzero requested proportional effect change is not representable.")
        for significance in alphas:
            critical = -t_ppf(significance / 2, df - 1) if significance < 1 else 0.0
            if not math.isfinite(critical) or (significance < 1 and critical <= 0):
                _failure("The positive Student t critical value is not representable.")
            rho, diagonal, ry, regime = _threshold(f, critical / math.sqrt(df - 1))
            rd = rho
            bias = se * math.sqrt(df) * math.sqrt(rd) * math.sqrt(ry) / math.sqrt(1 - rd)
            adjusted = estimate - orientation * bias
            adjusted_se = se * math.sqrt(1 - ry) / math.sqrt(1 - rd) * math.sqrt(df / (df - 1))
            adjusted_variance = adjusted_se * adjusted_se
            statistic = (adjusted - reference) / adjusted_se
            if (
                not all(
                    math.isfinite(v)
                    for v in (bias, adjusted, adjusted_se, adjusted_variance, statistic)
                )
                or adjusted_variance <= 0
                or (rho > 0 and (bias == 0 or adjusted == estimate))
            ):
                _failure(
                    "A robustness witness bias, coefficient or uncertainty is not representable."
                )
            p_value = 2 * t_sf(abs(statistic), df - 1)
            boundary_error = orientation * statistic - critical
            kind = "point_estimate" if significance == 1 else "significance"
            rows.append(
                [
                    fraction,
                    significance,
                    kind,
                    regime,
                    rho,
                    diagonal,
                    rd,
                    ry,
                    reference,
                    adjusted,
                    adjusted_se,
                    df - 1,
                    statistic,
                    p_value,
                    critical,
                    boundary_error,
                ]
            )
            witnesses.append(
                dict(
                    q=fraction,
                    alpha=significance,
                    kind=kind,
                    regime=regime,
                    max_strength=rho,
                    equal_strength_diagonal=diagonal,
                    r2_treatment=rd,
                    r2_outcome=ry,
                    reference_estimate=reference,
                    adjusted_estimate=adjusted,
                    variance=adjusted_variance,
                    standard_error=adjusted_se,
                    df=df - 1,
                    t_to_reference=statistic,
                    p_to_reference=p_value,
                    critical=critical,
                    signed_boundary_reconstruction_error=boundary_error,
                    scaled_reference_distance=f,
                )
            )
    settings = dict(
        q=qs,
        alpha=alphas,
        definition="minimum maximum of the two partial-R2 strengths",
        device=device,
        weights=None,
        max_work=max_work,
    )
    state = dict(
        target="max-coordinate OVB robustness threshold for a proportional reference effect",
        witnesses=witnesses,
        robustness_covariance=None,
        confidence_interval=None,
        adjusted_df=df - 1,
        inference="Two-sided classical Student t sensitivity query for one omitted scalar regressor; alpha1 is point tipping.",
        assumptions=[
            "Original classical homoskedastic OLS model and one omitted scalar regressor.",
            "The reference effect (1-q)*the original estimate is a sensitivity query, not a independently fixed inferential parameter.",
            "Student t inference is exact under independent homoskedastic normal linear-model errors; otherwise it is an approximation.",
        ],
        optimization="Minimize oriented adjusted t over both partial-R2 coordinates in [0,rho]; diagonal or interior outcome-coordinate branch.",
    )
    return _finish(
        "ovb_robustness",
        result,
        artifact,
        settings,
        {
            "robustness": m.frame(
                rows,
                columns=[
                    "q",
                    "alpha",
                    "kind",
                    "regime",
                    "robustness_value",
                    "diagonal_value",
                    "witness_r2_treatment",
                    "witness_r2_outcome",
                    "reference_estimate",
                    "adjusted_estimate",
                    "adjusted_std_error",
                    "adjusted_df",
                    "t_to_reference",
                    "p_to_reference",
                    "critical_value",
                    "boundary_error",
                ],
            ),
        },
        state,
        work,
        plan,
        integrity_plan,
    )
