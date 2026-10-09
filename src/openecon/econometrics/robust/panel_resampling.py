"""Whole-panel MM-QR pairs bootstrap and balanced split-panel jackknife."""

import math
from numbers import Integral

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.summary_state import saved_summary
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.postest.common import matched_frame, require_result
from openecon.engines.distributions import normal_isf
from openecon.resources import plan_workspace
from openecon.econometrics.resident_cpu import resident_cpu

from .panel_mmqr import fit_panel_mmqr


def _coefs(result):
    return torch.tensor([c.estimate for c in result.coefficients], dtype=torch.float64)


def _fits(spec, data, split):
    full = fit_panel_mmqr(spec, data)
    if not split:
        return _coefs(full), [full]
    times = sorted(data[spec.time].unique().tolist())
    mid = len(times) // 2
    halves = [
        fit_panel_mmqr(spec, data.loc[data[spec.time].isin(t)].copy())
        for t in (times[:mid], times[mid:])
    ]
    terms = [c.term for c in full.coefficients]
    if any([c.term for c in fit.coefficients] != terms for fit in halves):
        raise AnalysisError(
            "bootstrap_terms", "Full and half panels have different estimable terms."
        )
    return 2 * _coefs(full) - (_coefs(halves[0]) + _coefs(halves[1])) / 2, [full, *halves]


@resident_cpu
def panel_mmqr_bootstrap(result, *, data, reps=199, seed=0, split_panel=False, max_work=50_000_000):
    """Resample complete independent individuals and all MM-QR moment stages.

    Optional SPJ is 2*full-(first_half+second_half)/2. Requires a balanced,
    common, equally spaced even calendar T>=12 and prespecified homogeneous
    parameters with a leading 1/T bias expansion. It does not make fixed-T
    quantile inference valid. Every failed draw aborts with its index; none are
    dropped. Joint covariance is from the corrected draws, never a sum of the
    three separate covariance matrices. The output stores all three fitted
    states for explicitly combined conditional quantile means.
    """
    result = require_result(result, "panel MM-QR result")
    spec = result.spec
    if (
        spec.estimator != "panel_mmqr"
        or spec.weights
        or spec.categorical
        or spec.covariance == "cluster"
    ):
        raise AnalysisError(
            "unsupported_spec",
            "Use numeric unweighted panel_mmqr; bootstrap independence is at the individual, without coarser clusters.",
        )
    if not isinstance(split_panel, bool):
        raise AnalysisError("invalid_option", "split_panel must be boolean.")
    for value, name, lo, hi in (
        (reps, "reps", 49, 10_000),
        (seed, "seed", 0, 2**63 - 1),
        (max_work, "max_work", 1, 500_000_000),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or not lo <= value <= hi:
            raise AnalysisError("invalid_option", f"{name} must be an integer in {lo}..{hi}.")
    original, positions = matched_frame(result, data)
    sample = original.iloc[positions].copy()
    codes, levels = pd.factorize(sample[spec.panel], sort=False)
    groups = len(levels)
    if groups < 3:
        raise AnalysisError(
            "insufficient_panels", "Pairs inference needs at least three independent individuals."
        )
    if split_panel:
        if not spec.time or not pd.api.types.is_numeric_dtype(sample[spec.time]):
            raise AnalysisError("split_panel_domain", "SPJ requires a numeric common calendar.")
        times = sorted(sample[spec.time].unique().tolist())
        if len(times) < 12 or len(times) % 2 or not all(math.isfinite(float(t)) for t in times):
            raise AnalysisError("split_panel_domain", "SPJ needs an even calendar with T>=12.")
        differences = [float(b - a) for a, b in zip(times[:-1], times[1:], strict=True)]
        if min(differences) <= 0 or any(
            abs(d - differences[0]) > 1e-10 * max(1, abs(differences[0])) for d in differences
        ):
            raise AnalysisError("split_panel_domain", "SPJ needs equally spaced time periods.")
        if any(sorted(sample.loc[codes == i, spec.time].tolist()) != times for i in range(groups)):
            raise AnalysisError(
                "split_panel_domain", "SPJ needs the same complete calendar for every individual."
            )
    width = len(result.coefficients)
    largest = max(int((codes == i).sum()) for i in range(groups)) * groups
    work = reps * largest * max(len(spec.predictors), 1) ** 2 * (3 if split_panel else 1)
    if work > max_work:
        raise AnalysisError(
            "work_budget_exceeded", "Panel resampling exceeds max_work; reduce dimensions or draws."
        )
    plan = plan_workspace(
        "whole-panel MM-QR bootstrap retained states",
        {
            "resident_input_and_refit_buffers": largest * (len(sample.columns) + width + 20) * 128,
            "joint_draws_and_choices": reps * (width + groups) * 32,
            "joint_covariance": width**2 * 32,
        },
    )
    estimate, states = _fits(spec, sample, split_panel)
    terms = [c.term for c in states[0].coefficients]
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    choices = torch.randint(groups, (reps, groups), generator=rng)
    draws = []
    for draw_index, selection in enumerate(choices):
        pieces = []
        for label, selected in enumerate(selection.tolist()):
            block = sample.loc[codes == selected].copy()
            block[spec.panel] = label
            pieces.append(block)
        resample = pd.concat(pieces, ignore_index=True)
        try:
            value, fitted = _fits(spec, resample, split_panel)
            if [c.term for c in fitted[0].coefficients] != terms:
                raise AnalysisError("bootstrap_terms", "Resampled estimable terms changed.")
            draws.append(value)
        except AnalysisError as exc:
            raise AnalysisError(
                "bootstrap_draw_failed",
                f"Draw {draw_index} failed ({exc.code}); no draws were discarded.",
            ) from exc
    values = torch.stack(draws)
    centered = values - values.mean(0)
    covariance = centered.T @ centered / (reps - 1)
    se = covariance.diagonal().sqrt()
    if not bool(torch.isfinite(covariance).all()) or bool((se <= 0).any()):
        raise AnalysisError(
            "invalid_covariance",
            "Bootstrap joint covariance lacks positive finite marginal variance.",
        )
    critical = normal_isf(spec.alpha / 2)
    low, high = torch.quantile(
        values, torch.tensor([spec.alpha / 2, 1 - spec.alpha / 2], dtype=torch.float64), dim=0
    )
    rows = [
        dict(
            term=t,
            estimate=float(estimate[i]),
            std_error=float(se[i]),
            ci_low=float(estimate[i] - critical * se[i]),
            ci_high=float(estimate[i] + critical * se[i]),
            percentile_low=float(low[i]),
            percentile_high=float(high[i]),
        )
        for i, t in enumerate(terms)
    ]
    return saved_summary(
        TableSet(
            {
                "coefficients": table(rows),
                "covariance": pd.DataFrame(covariance.tolist(), index=terms, columns=terms),
            },
            method="panel_mmqr_split_panel_pairs" if split_panel else "panel_mmqr_pairs",
            source_result_id=result.id,
            bootstrap_draws=values.tolist(),
            panel_draw_indices=choices.tolist(),
            panel_levels=levels.tolist(),
            reps=int(reps),
            seed=int(seed),
            failed_draws=0,
            covariance_matrix=covariance.tolist(),
            terms=terms,
            original_sample_positions=positions,
            source_data_hash=result.provenance["data_hash"],
            fitted_states=[s.model_dump(mode="json") for s in states],
            combination_weights=[2.0, -0.5, -0.5] if split_panel else [1.0],
            correction="2*full-(first_half+second_half)/2" if split_panel else "none",
            assumptions="independent whole individuals; common strictly exogenous location-scale law; growing T"
            + (
                "; homogeneous time parameters and leading 1/T bias expansion"
                if split_panel
                else ""
            ),
            normal_interval="asymptotic bootstrap-normal; no fixed-T coverage guarantee",
            percentile_interpolation="linear empirical quantile",
            resource_plan=plan.record(),
            work_units=work,
        )
    )


@resident_cpu
def panel_mmqr_resampled_predict(output, *, data):
    """Saved original/corrected conditional means for previously fitted individuals.

    Combines each saved state's individual effects and quantile slopes using
    the same SPJ weights. Does not attach an invalid slope-only prediction SE;
    uncertainty in individual effects is not stored by the slope bootstrap.
    """
    from openecon.analysis import _coerce_frame, _numeric
    from openecon.models import ResultBundle

    if not isinstance(output, TableSet) or output.attrs.get("method") not in {
        "panel_mmqr_pairs",
        "panel_mmqr_split_panel_pairs",
    }:
        raise AnalysisError("invalid_result", "Pass panel_mmqr_bootstrap output.")
    states = [ResultBundle.model_validate(s) for s in output.attrs["fitted_states"]]
    weights = output.attrs["combination_weights"]
    source = _coerce_frame(data)
    spec = states[0].spec
    required = [spec.panel, *states[0].extra["location_terms"]]
    if any(c not in source for c in required) or source[required].isna().any().any():
        raise AnalysisError(
            "prediction_domain", "Provide complete known individual keys and numeric predictors."
        )
    if len(source) > 8192:
        raise AnalysisError(
            "work_budget_exceeded", "Prediction permits at most 8192 resident rows."
        )
    plan_workspace(
        "saved panel conditional means",
        {
            "design_and_quantile_rows": len(source)
            * (len(required) + len(states[0].extra["quantiles"]) + 10)
            * 64
        },
    )
    x = torch.stack(
        [_numeric(source[name], name) for name in states[0].extra["location_terms"]], dim=1
    )
    if not bool(torch.isfinite(x).all()):
        raise AnalysisError("prediction_domain", "Predictors must be finite.")
    prediction = torch.zeros((len(source), len(states[0].extra["quantiles"])), dtype=torch.float64)
    for factor, state in zip(weights, states, strict=True):
        effects = {e["panel"]: e for e in state.extra["individual_effects"]}
        if any(key not in effects for key in source[spec.panel]):
            raise AnalysisError(
                "unknown_panel", "Conditional prediction requires previously fitted individuals."
            )
        beta = torch.tensor(state.extra["location_coefficients"], dtype=torch.float64)
        gamma = torch.tensor(state.extra["scale_coefficients"], dtype=torch.float64)
        a = torch.tensor(
            [effects[key]["location"] for key in source[spec.panel]], dtype=torch.float64
        )
        d = torch.tensor([effects[key]["scale"] for key in source[spec.panel]], dtype=torch.float64)
        scale = d + x @ gamma
        if bool((scale <= 0).any()):
            raise AnalysisError(
                "nonpositive_scale", "A saved component predicts nonpositive scale."
            )
        qs = torch.tensor(state.extra["error_quantiles"], dtype=torch.float64)
        prediction += factor * ((a + x @ beta)[:, None] + scale[:, None] * qs)
    columns = {"panel": source[spec.panel].tolist()}
    columns.update(
        {
            f"q{tau:g}": prediction[:, i].tolist()
            for i, tau in enumerate(states[0].extra["quantiles"])
        }
    )
    return saved_summary(
        TableSet(
            {"conditional means": pd.DataFrame(columns)},
            refitted=False,
            correction=output.attrs["correction"],
            quantile_crossing_possible=len(states) > 1,
            uncertainty="not available for individual conditional means; coefficient bootstrap does not store individual-effect draws",
        )
    )
