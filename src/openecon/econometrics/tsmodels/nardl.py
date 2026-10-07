"""Asymmetric partial-sum ARDL, symmetry restrictions and dynamic multipliers.

Shin, Yu and Greenwood-Nimmo (2014), doi:10.1007/978-1-4899-8008-3_9.
All numerical transformations use float64 Torch; estimation shares ARDL's QR.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import column_list, kernel_call, table
from openecon.econometrics.tsmodels.ardl import _fit_ardl_frame, _lag_name
from openecon.econometrics.tsmodels.common import (
    as_int_list, build_spec, linear_restriction_test, ordered_frame,
)
from openecon.engines.inference import critical_value
from openecon.models import ModelSpec, ResultBundle


def _expand_orders(value: Any, original: list[str], asymmetric: set[str], *,
                   name: str) -> list[int] | None:
    if value is None:
        return None
    values = list(value)
    expanded_count = len(original) + len(asymmetric) + 1
    if name == "maxlags" and len(values) == 1:
        return values
    if len(values) == len(original) + 1:
        orders = [values[0]]
        for variable, order in zip(original, values[1:], strict=True):
            orders.extend([order, order] if variable in asymmetric else [order])
        return orders
    if len(values) == expanded_count:
        return values
    raise AnalysisError("invalid_lags", f"{name} needs an outcome lag followed by one lag per "
                        "original predictor, or per expanded positive/negative predictor.")


def _symmetry_rows(terms: list[str], pairs: dict[str, list[str]], orders: dict[str, int]):
    """Levels-space long-run and conditional-EC lagwise short-run restrictions."""
    restrictions = {}
    for variable, (positive, negative) in pairs.items():
        qp, qn = orders[positive], orders[negative]
        long = torch.zeros(len(terms), dtype=torch.float64)
        for sign, name, q in ((1.0, positive, qp), (-1.0, negative, qn)):
            for i in range(q + 1):
                long[terms.index(_lag_name(name, i))] = sign
        short = []
        # EViews' zero-lag convention: no independently adjustable SR dynamics.
        # For unequal orders gamma_0=b_0 still applies to the zero-order side.
        if max(qp, qn) > 0:
            for lag_ in range(max(qp, qn)):
                row = torch.zeros_like(long)
                for sign, name, q in ((1.0, positive, qp), (-1.0, negative, qn)):
                    if lag_ == 0:
                        row[terms.index(name)] = sign
                    else:
                        for i in range(lag_ + 1, q + 1):
                            row[terms.index(_lag_name(name, i))] = -sign
                short.append(row)
        restrictions[variable] = (long, short)
    return restrictions


def _structural_difference_zeros(basis: Tensor | None, terms: list[str],
                                 pair: list[str], orders: dict[str, int], steps: int) -> Tensor:
    """Prove a zero step contrast from homogeneous *coefficient* identities.

    The common AR filter cancels only if every positive/negative distributed
    coefficient through that horizon is identical for every allowed parameter.
    This tests fixed linear lag contrasts in the restriction null space, never
    a nonlinear gradient or its variance at the fitted parameter value.
    """
    fixed = torch.zeros(steps + 1, dtype=torch.bool)
    if basis is None:
        return fixed
    maximum = max(orders[name] for name in pair)
    equal = []
    for lag_ in range(maximum + 1):
        row = torch.zeros(len(terms), dtype=torch.float64)
        for sign, name in zip((1.0, -1.0), pair, strict=True):
            if lag_ <= orders[name]:
                row[terms.index(_lag_name(name, lag_))] = sign
        equal.append(bool(torch.linalg.vector_norm(row @ basis) <=
                          1e-12 * torch.linalg.vector_norm(row)))
    for horizon in range(steps + 1):
        fixed[horizon] = all(equal[:min(horizon, maximum) + 1])
    return fixed


def _symmetry_wald(beta, covariance, matrix, df, label, basis=None, free=None):
    if basis is not None:
        estimable = torch.linalg.vector_norm(matrix @ basis, dim=1) > (
            1e-12 * torch.linalg.vector_norm(matrix, dim=1).clamp_min(1))
        imposed = int((~estimable).sum())
        matrix = matrix[estimable]
        if not len(matrix):
            return {"statistic": None, "df": 0, "df2": df, "p_value": None,
                    "distribution": "F", "label": label, "imposed": True,
                    "note": "This symmetry is imposed or implied by the fitted restrictions; "
                    "no Wald test on a zero-variance contrast.", "imposed_rows": imposed}
    else:
        imposed = 0
    if free is not None:
        beta, covariance, matrix = free[0], free[1], matrix @ basis
    result = linear_restriction_test(beta, covariance, matrix,
                                     torch.zeros(len(matrix), dtype=torch.float64),
                                     df_resid=df, label=label)
    if imposed:
        result.update(imposed_rows=imposed, note="Tests only the remaining estimable symmetry "
                      "restrictions; other rows are imposed by the fitted model.")
    return result


def _symmetry(result: ResultBundle, pairs: dict[str, list[str]]) -> dict[str, Any]:
    """Linear restrictions on the levels coefficients, invariant to EC reporting."""
    levels = result.extra["levels_coefficients"]
    terms = list(levels)
    beta = torch.tensor(list(levels.values()), dtype=torch.float64)
    covariance = torch.tensor(result.extra["levels_covariance"], dtype=torch.float64)
    df = result.metrics["df_resid"]
    tests = {}
    constraints = result.extra.get("constraints")
    basis = torch.tensor(constraints["parameter_basis"], dtype=torch.float64) if constraints else None
    free = (torch.tensor(constraints["free_coefficients"], dtype=torch.float64),
            torch.tensor(constraints["free_covariance"], dtype=torch.float64)) if constraints else None
    long_joint, short_joint, lagwise_joint = [], [], []
    for variable, (row, short) in _symmetry_rows(terms, pairs, result.extra["lags"]).items():
        long_joint.append(row)
        tests[f"symmetry_long_run:{variable}"] = _symmetry_wald(
            beta, covariance, row[None], df,
            f"Long-run symmetry of {variable}: sum b_positive = sum b_negative", basis, free)
        if short:
            sr = torch.stack(short).sum(0)
            short_joint.append(sr)
            lagwise_joint.extend(short)
            tests[f"symmetry_short_run:{variable}"] = _symmetry_wald(
                beta, covariance, sr[None], df,
                f"Cumulative short-run symmetry of {variable} in the x_(t-1) EC form", basis, free)
            tests[f"symmetry_short_run_lagwise:{variable}"] = _symmetry_wald(
                beta, covariance, torch.stack(short), df,
                f"Lagwise short-run symmetry of {variable} in the x_(t-1) EC form", basis, free)
    for name, rows in (("long_run", long_joint), ("short_run", short_joint),
                       ("joint", long_joint + short_joint), ("short_run_lagwise", lagwise_joint),
                       ("joint_lagwise", long_joint + lagwise_joint)):
        if rows:
            tests[f"symmetry_{name}"] = _symmetry_wald(
                beta, covariance, torch.stack(rows), df,
                f"Joint {name.replace('_', ' ')} symmetry of asymmetric predictors", basis, free)
    return tests


def fit_nardl(spec: ModelSpec, data: Any) -> ResultBundle:
    frame, _ = ordered_frame(spec, data, what="NARDL partial-sum estimation")
    delta = frame.option("time_delta")
    if spec.time is not None and is_datetime64_any_dtype(frame.sample[spec.time].dtype):
        try:
            if delta is None:
                raise ValueError("An explicit datetime interval is required")
            step = pd.Timedelta(delta)
            if pd.isna(step) or step.value <= 0:
                raise ValueError("Interval must be positive")
            times = frame.sample[spec.time].dt.as_unit("ns").astype("int64")
            if not bool((times.diff().iloc[1:] == step.value).all()):
                raise AnalysisError("time_gaps", "Datetime observations must be consecutive at "
                                    "time_delta; fill gaps or use one uninterrupted stretch.")
        except AnalysisError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalysisError("invalid_time_delta", "Datetime time requires an explicit positive "
                                "fixed time_delta, e.g. '1D'; use integer period codes for months.") from exc
    elif delta is not None:
        raise AnalysisError("invalid_time_delta", "time_delta applies only to datetime time; "
                            "numeric time uses consecutive integer period codes.")
    origin_row = frame.positions[0]
    original = list(spec.predictors)
    clash = set(frame.role("exog")) & (set(original) | {spec.outcome})
    if clash:
        raise AnalysisError("invalid_spec", "exog must not repeat the outcome or x: "
                            + ", ".join(sorted(clash)) + ".")
    chosen = frame.option("asymmetric")
    asymmetric = original if chosen is None else list(chosen)
    if not asymmetric or len(set(asymmetric)) != len(asymmetric) or set(asymmetric) - set(original):
        raise AnalysisError("invalid_spec", "asymmetric must be a nonempty, unique subset of x.")
    if frame.option("trend") == "none":
        raise AnalysisError("invalid_option", "NARDL partial sums absorb initial levels into a "
                            "constant; use trend='constant' or 'trend'.")
    imposed = {}
    for kind in ("long_run", "short_run"):
        subset = list(frame.option(f"{kind}_symmetric") or [])
        if len(subset) != len(set(subset)) or set(subset) - set(asymmetric):
            raise AnalysisError("invalid_spec", f"{kind}_symmetric must be a unique subset of "
                                "the asymmetric predictors.")
        imposed[kind] = subset
    names, columns, pairs = [], [], {}
    inputs = set(frame.original.columns)
    for name in original:
        values = frame.numeric(name)
        if name not in asymmetric:
            names.append(name)
            columns.append(values)
            continue
        difference = torch.diff(values, prepend=values[:1])
        if not bool(torch.isfinite(difference).all()):
            raise AnalysisError("non_finite_design", f"Changes in '{name}' exceed float64 range; "
                                "rescale the series before fitting.")
        if not bool((difference > 0).any()) or not bool((difference < 0).any()):
            raise AnalysisError("unidentified_asymmetry", f"'{name}' needs both positive and "
                                "negative changes to identify separate effects.")
        pair = [f"{name}_positive", f"{name}_negative"]
        if inputs.intersection(pair):
            raise AnalysisError("duplicate_terms", f"Partial-sum names for '{name}' collide "
                                "with model inputs; rename those inputs.")
        pairs[name] = pair
        names.extend(pair)
        partials = [difference.clamp_min(0).cumsum(0), difference.clamp_max(0).cumsum(0)]
        if not all(bool(torch.isfinite(partial).all()) for partial in partials):
            raise AnalysisError("non_finite_design", f"Partial sums of '{name}' exceed float64 "
                                "range; rescale the series before fitting.")
        columns.extend(partials)
    if len(names) != len(set(names)):
        raise AnalysisError("duplicate_terms", "Partial-sum names collide; rename the predictors.")
    options = dict(spec.options)
    for key in ("lags", "maxlags"):
        if key in options:
            options[key] = _expand_orders(options[key], original, set(asymmetric), name=key)
    if options.get("lags") is not None and options["lags"][0] < 1:
        raise AnalysisError("invalid_lags", "NARDL requires at least one lag of the outcome.")
    frame.spec = spec.model_copy(update={"options": options})

    def restrictions(terms, orders):
        rows = _symmetry_rows(terms, pairs, dict(zip(names, orders[1:], strict=True)))
        chosen = [rows[variable][0] for variable in imposed["long_run"]]
        chosen += [row for variable in imposed["short_run"] for row in rows[variable][1]]
        return torch.stack(chosen) if chosen else torch.empty((0, len(terms)), dtype=torch.float64)

    result = _fit_ardl_frame(frame, names=names, x_all=torch.stack(columns, dim=1), retain_levels=True,
                            constraint_builder=restrictions if any(imposed.values()) else None)
    return _finish_nardl(result, spec, names, pairs, imposed, origin_row)


def _finish_nardl(result, spec, names, pairs, imposed, origin_row):
    """Shared reporting and symmetry tests after a full-source levels estimate."""
    result.spec = spec
    result.title = result.title.replace("ARDL", "NARDL", 1)
    result.extra["asymmetric_predictors"] = pairs
    result.extra["expanded_predictors"] = names
    result.extra["imposed_symmetry"] = {
        **imposed, "short_run_definition": "lagwise gamma_positive = gamma_negative in the "
        "conditional EC form with x_(t-1); absent lags are zero",
        "vacuous_short_run": [variable for variable in imposed["short_run"] if
        all(result.extra["lags"][name] == 0 for name in pairs[variable])],
    }
    result.extra["partial_sum_origin"] = {
        "definition": "sum positive/negative first differences; both partial sums start at zero",
        "row": origin_row,
    }
    result.extra["short_run_symmetry_definition"] = "sum gamma_positive = sum gamma_negative " \
        "in the conditional EC form with x_(t-1); no test when both distributed lag orders are zero"
    result.tests.update(_symmetry(result, pairs))
    # Partial sums are dependent generated regressors. Do not silently attach
    # ordinary ARDL tables with the expanded k as NARDL-calibrated critical values.
    for name in ("bounds_f", "bounds_t"):
        if name in result.tests:
            test = result.tests[name]
            test.update(critical_values=None, decision=None, p_value=None,
                        note="Statistic uses classical covariance. NARDL-specific critical values "
                        "are not independently calibrated; no cointegration decision is reported.")
    result.warnings.append("NARDL bounds statistics have no validated critical values or "
                           "automatic cointegration decision.")
    result.provenance.update(
        transformation="Shin-Yu-Greenwood-Nimmo positive/negative partial sums",
        transformed_design_terms=names, bounds_critical_values_validated=False,
        stata_parity_validated=False,
    )
    return result


def nardl(*, data: Any, y: str, x: Any, asymmetric: Any = None, time: str | None = None,
          time_delta: str | None = None,
          long_run_symmetric: Any = None, short_run_symmetric: Any = None,
          lags: Any = None, maxlags: Any = 4, ic: str = "aic", trend: str = "constant",
          restricted: bool = False, exog: Any = None, ec: bool = True,
          covariance: str | None = None, missing: str = "raise", alpha: float = 0.05) -> ResultBundle:
    """NARDL with partial sums, optional LR/SR symmetry and dynamic multipliers.

    Implements the Shin-Yu-Greenwood-Nimmo model; corresponds to Stata's
    community NARDL estimators and EViews' asymmetric ARDL specification.

    ``asymmetric`` defaults to every predictor; remaining x are symmetric.
    ``lags`` accepts [p, q1, ...] for the original x (each asymmetric q used twice)
    or one q per expanded predictor (positive, negative, then next x).
    ``maxlags`` follows the same convention, or one shared maximum.
    Datetime ``time`` requires a fixed ``time_delta``, e.g. ``'1D'``;
    calendar monthly/quarterly data should use consecutive integer period codes.
    Coefficient covariance: nonrobust / robust / HC1 / HC2 / HC3.
    Bounds statistics are provided without unvalidated critical values/decisions.
    Use ``nardl_multipliers(result)`` for responses and pointwise delta-method CIs.
    ``long_run_symmetric`` and ``short_run_symmetric`` select separate subsets of
    asymmetric predictors. Short-run symmetry imposes equality at every gamma
    lag in the conditional x_(t-1) EC form, padding absent lags with zero; it
    is stronger than the separately reported cumulative symmetry test.
    """
    spec = build_spec(
        "nardl", outcome=y, predictors=column_list(x, "x"), time=time,
        covariance=covariance, missing=missing, alpha=alpha,
        columns={"exog": column_list(exog, "exog")},
        options={"asymmetric": None if asymmetric is None else column_list(asymmetric, "asymmetric"),
                 "long_run_symmetric": None if long_run_symmetric is None else
                 column_list(long_run_symmetric, "long_run_symmetric"),
                 "short_run_symmetric": None if short_run_symmetric is None else
                 column_list(short_run_symmetric, "short_run_symmetric"),
                 "lags": as_int_list(lags, "lags"),
                 "maxlags": as_int_list(maxlags, "maxlags") if lags is None else None,
                 "ic": ic, "trend": trend, "restricted": restricted, "ec": ec,
                 "time_delta": time_delta},
    )
    from openecon.analysis import fit

    return fit(spec, data=data)


def _multiplier_path(beta: Tensor, terms: list[str], outcome: str,
                     orders: dict[str, int], names: list[str], steps: int) -> tuple[Tensor, Tensor]:
    """Permanent-unit-change response and analytic coefficient derivatives.

    m_h = sum_(l<=h) b_l + sum_(i=1..p) phi_i m_(h-i).
    Only horizons/lags are iterated; no loops over observations are needed.
    """
    p = orders[outcome]
    y_index = [terms.index(_lag_name(outcome, i)) for i in range(1, p + 1)]
    b_index = [[terms.index(_lag_name(name, i)) for i in range(orders[name] + 1)] for name in names]
    response = torch.zeros((steps + 1, len(names)), dtype=torch.float64)
    gradient = torch.zeros((steps + 1, len(names), len(terms)), dtype=torch.float64)
    for h in range(steps + 1):
        for j, indexes in enumerate(b_index):
            chosen = indexes[:h + 1]
            response[h, j] = beta[chosen].sum()
            gradient[h, j, chosen] = 1.0
        for i, index in enumerate(y_index, start=1):
            if h >= i:
                response[h] += beta[index] * response[h - i]
                gradient[h] += beta[index] * gradient[h - i]
                gradient[h, :, index] += response[h - i]
    return response, gradient


def nardl_multipliers(result: ResultBundle, *, steps: int = 40, alpha: float | None = None,
                     method: str = "delta", data: Any = None, replications: int = 999,
                     seed: int = 0, batch_size: int = 64):
    """Positive/negative multipliers and pointwise confidence intervals.

    ``negative`` is the derivative per +1 unit of the negative partial-sum
    series. An actual -1 change in x has response ``-negative``. ``difference``
    compares positive and negative derivatives (zero under symmetry).
    ``method='delta'`` is the default analytic delta method. ``'residual'`` and
    ``'wild'`` instead recursively regenerate the outcome and refit the selected,
    possibly constrained levels model. Bootstrap methods require the original
    ``data`` and report percentile intervals, not simultaneous bands. Predictors,
    initial conditions, lag selection and symmetry constraints are conditioned
    on. Residual resampling assumes iid innovations; Rademacher wild resampling
    permits conditional heteroskedasticity under its recorded assumptions.

    Example
    -------
    >>> paths = oe.nardl_multipliers(fit, steps=40)
    >>> display(paths)
    """
    if not isinstance(method, str) or method not in {"delta", "residual", "wild"}:
        raise AnalysisError("invalid_option", "method must be 'delta', 'residual' or 'wild'.")
    if method != "delta":
        from openecon.econometrics.tsmodels.nardl_bootstrap import bootstrap_multipliers

        return bootstrap_multipliers(result, data=data, steps=steps, alpha=alpha,
                                     method=method, replications=replications, seed=seed,
                                     batch_size=batch_size)
    if not isinstance(result, ResultBundle) or result.spec.estimator != "nardl":
        raise AnalysisError("invalid_result", "nardl_multipliers requires a fitted NARDL result.")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 0 <= steps <= 1000:
        raise AnalysisError("invalid_option", "steps must be an integer from 0 to 1000.")
    alpha = result.spec.alpha if alpha is None else alpha
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise AnalysisError("invalid_option", "alpha must be strictly between zero and one.")
    try:
        levels = result.extra["levels_coefficients"]
        terms = list(levels)
        beta = torch.tensor(list(levels.values()), dtype=torch.float64)
        covariance = torch.tensor(result.extra["levels_covariance"], dtype=torch.float64)
        orders = result.extra["lags"]
        pairs = result.extra["asymmetric_predictors"]
        if (not isinstance(orders, dict) or result.spec.outcome not in orders
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in orders.values())
                or orders[result.spec.outcome] < 1 or not isinstance(pairs, dict)
                or any(not isinstance(v, list) or len(v) != 2 or
                       any(not isinstance(name, str) for name in v) for v in pairs.values())):
            raise ValueError("Invalid saved orders or partial-sum pairs")
        names = [name for pair in pairs.values() for name in pair]
        if covariance.shape != (len(terms), len(terms)) or not pairs:
            raise ValueError("Invalid saved dimensions")
        if not bool(torch.isfinite(beta).all()) or not bool(torch.isfinite(covariance).all()):
            raise ValueError("Nonfinite saved parameters")
        scale = float(covariance.abs().max())
        if (not bool(torch.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12))
                or float(torch.linalg.eigvalsh(covariance).min()) < -1e-10 * max(scale, 1e-300)):
            raise ValueError("Invalid saved covariance")
        constraint_record = result.extra.get("constraints")
        basis, free_covariance = None, None
        if constraint_record is not None:
            basis = torch.tensor(constraint_record["parameter_basis"], dtype=torch.float64)
            free_covariance = torch.tensor(constraint_record["free_covariance"], dtype=torch.float64)
            if (basis.ndim != 2 or basis.shape[0] != len(terms) or
                    free_covariance.shape != (basis.shape[1], basis.shape[1]) or
                    not bool(torch.isfinite(basis).all()) or
                    not bool(torch.isfinite(free_covariance).all()) or
                    not bool(torch.allclose(basis.T @ basis, torch.eye(basis.shape[1], dtype=torch.float64),
                                           rtol=1e-10, atol=1e-12)) or
                    not bool(torch.allclose(basis @ free_covariance @ basis.T, covariance,
                                           rtol=1e-8, atol=1e-12))):
                raise ValueError("Invalid saved constraint covariance")
        response, gradient = _multiplier_path(beta, terms, result.spec.outcome, orders, names, steps)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise AnalysisError("invalid_result", "Saved NARDL levels parameters are incomplete or invalid.") from exc
    if not bool(torch.isfinite(response).all()) or not bool(torch.isfinite(gradient).all()):
        raise AnalysisError("numerical_failure", "Multiplier recursion diverged; reduce steps or "
                            "review the autoregressive stability.")
    df = result.inference.get("df_inference")
    if (result.inference.get("use_t") is not True or result.inference.get("distribution") != "t"
            or isinstance(df, bool) or not isinstance(df, (int, float)) or not 0 < df < float("inf")):
        raise AnalysisError("invalid_inference", "Saved NARDL multiplier inference needs recorded "
                            "Student-t inference and positive finite df_inference.")
    critical = kernel_call(critical_value, float(alpha), df)
    rows = []
    for j, variable in enumerate(pairs):
        estimates = torch.stack([response[:, 2*j], response[:, 2*j+1],
                                 response[:, 2*j] - response[:, 2*j+1]], dim=1)
        jac = torch.stack([gradient[:, 2*j], gradient[:, 2*j+1],
                           gradient[:, 2*j] - gradient[:, 2*j+1]], dim=1)
        if basis is None:
            variances = torch.einsum("hsk,kl,hsl->hs", jac, covariance, jac)
        else:
            projected = jac @ basis
            variances = torch.einsum("hsk,kl,hsl->hs", projected, free_covariance, projected)
        # A zero first-order derivative is not a structurally constant nonlinear
        # response. Only proved distributed-coefficient identities are exact.
        fixed = _structural_difference_zeros(basis, terms, pairs[variable], orders, steps)
        variances[fixed, 2], estimates[fixed, 2] = 0.0, 0.0
        if not bool(torch.isfinite(variances).all()):
            raise AnalysisError("numerical_failure", "Multiplier uncertainty overflowed; reduce "
                                "steps or review autoregressive stability.")
        se = variances.clamp_min(0).sqrt()
        for h in range(steps + 1):
            row = {"variable": variable, "horizon": h}
            for i, name in enumerate(("positive", "negative", "difference")):
                value, error = float(estimates[h, i]), float(se[h, i])
                row.update({name: value, f"{name}_std_error": error,
                            f"{name}_ci_low": value - critical * error,
                            f"{name}_ci_high": value + critical * error})
            rows.append(row)
    p = orders[result.spec.outcome]
    companion = torch.zeros((p, p), dtype=torch.float64)
    companion[0] = torch.tensor([levels[_lag_name(result.spec.outcome, i)] for i in range(1, p+1)],
                                dtype=torch.float64)
    if p > 1:
        companion[1:, :-1] = torch.eye(p-1, dtype=torch.float64)
    stable = bool((torch.linalg.eigvals(companion).abs() < 1).all())
    return table(rows, title="NARDL cumulative dynamic multipliers", confidence_level=1-float(alpha),
                 inference="pointwise delta method, conditional on selected lag orders",
                 negative_shock_response="A -1 shock in x has response -negative",
                 autoregressive_stable=stable, stata_parity_validated=False)
