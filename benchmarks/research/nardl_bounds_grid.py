"""Predeclared PRIVATE NARDL calibration smoke; no public test calibration.

The grid, denominator/failure rules and future acceptance gates are fixed in
source before inspecting its results. ``--outer 12 --inner 99`` is a numerical
smoke with very wide Monte-Carlo intervals, never a size/power validation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from benchmarks.research.nardl_bounds import (
    NULLS, ResearchFailure, generate_y, partial_sums, research_bootstrap,
)


@dataclass(frozen=True)
class Cell:
    name: str
    hypothesis_class: str
    total: int
    rho: float
    theta_positive: float = 0.
    theta_negative: float = 0.
    drift: float = 0.
    skew: bool = False
    structural_contemporaneous_correlation: float = 0.
    lags: tuple[int, int, int] = (1, 1, 1)
    lagged_outcome_difference: float = 0.
    near_unit: bool = False
    within_maintained_class: bool = True


# These fixed hypotheses/dynamics are declared before the initial smoke.
GRID = (
    Cell("null_T80", "all_levels_null", 80, 0.),
    Cell("null_T160", "all_levels_null", 160, 0.),
    Cell("null_drift", "all_levels_null", 80, 0., drift=.15),
    Cell("null_skew", "all_levels_null", 80, 0., skew=True),
    Cell("null_correlated_structural_shocks", "all_levels_null", 80, 0., structural_contemporaneous_correlation=.6),
    Cell("null_lags_221", "all_levels_null", 160, 0., lags=(2, 2, 1), lagged_outcome_difference=.25),
    Cell("stationary_y_T80", "no_explanatory_levels", 80, -.45),
    Cell("stationary_y_T160", "no_explanatory_levels", 160, -.45),
    Cell("stationary_y_nearunit", "no_explanatory_levels", 160, -.035, near_unit=True),
    Cell("stationary_y_skew_drift", "no_explanatory_levels", 80, -.45, drift=-.15, skew=True),
    Cell("asymmetric_T80", "asymmetric_alternative", 80, -.35, .28, .4375),
    Cell("asymmetric_T160", "asymmetric_alternative", 160, -.35, .28, .4375),
    Cell("asymmetric_nearunit", "asymmetric_alternative", 160, -.035, .028, .04375, near_unit=True),
    Cell("asymmetric_lags_221", "asymmetric_alternative", 160, -.35, .28, .4375,
         structural_contemporaneous_correlation=.6, lags=(2, 2, 1), lagged_outcome_difference=.25),
    Cell("levels_without_adjustment_I2_stress", "rho_null_levels_present_outside_I0_I1", 80, 0., .08, .125,
         within_maintained_class=False),
)

ACCEPTANCE = {
    "status": "not evaluated by smoke; nonlinear bootstrap validity remains unestablished",
    "scope": "case III, one declared raw I(1) iid-increment predictor, fixed lags, no controls or constraints",
    "prospective_outer_minimum": 500,
    "prospective_inner_minimum": 999,
    "prospective_alphas": [.01, .05, .1],
    "size_screen_at_05": "For every within-class true null, marginal rejection Wilson95 upper <= .08; interpret conservatism separately. Full-null all-three rejection upper <= .08. No invalid whole calls.",
    "strong_alternative_screen": "For asymmetric_T160 at .05, all-three rejection Wilson95 lower >= .7; no such gate for weak/near-unit cells.",
    "replication_sensitivity": "Independent seed repeats plus pool/draw recentering and fixed/block initial policies; no convention chosen after looking at best rejection rates.",
    "proof_gap": "Numerical and Monte-Carlo gates cannot replace a nonlinear-bootstrap validity argument, serial-innovation study or external replication.",
    "failure_denominator": "All planned outer calls remain in denominator; failures shown separately and rejection intervals bounded under both failure assignments.",
    "invalid_draw_replacement": 0,
}


def generate_sample(cell: Cell, seed: int, burn=64):
    """Conditional nonlinear ECM DGP; structural shock correlation is explicit.

    Raw innovation u has variance one. A correlated structural outcome shock
    is corr*u+sqrt(1-corr²)*v; the corr*u part is absorbed in CURRENT signed
    difference coefficients, leaving a conditional innovation orthogonal to u.
    This creates structural instantaneous correlation without outcome feedback.
    Positive/negative level sums both come from the same raw path.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = cell.total + burn
    if cell.skew:
        uniform = torch.rand(total, generator=generator, dtype=torch.float64, device="cpu")
        u = -torch.log(uniform.clamp_min(torch.finfo(torch.float64).tiny))-1
    else:
        u = torch.randn(total, generator=generator, dtype=torch.float64, device="cpu")
    v = torch.randn(total, generator=generator, dtype=torch.float64, device="cpu")
    correlation = cell.structural_contemporaneous_correlation
    conditional = math.sqrt(1-correlation**2)*v
    raw = torch.cat((torch.zeros(1, dtype=torch.float64, device="cpu"),
                     (cell.drift+u[1:]).cumsum(0)))
    xp, xn = partial_sums(raw)
    p, qp, qn = cell.lags
    if p not in {1, 2} or qp not in {1, 2} or qn not in {1, 2}:
        raise ValueError("Predeclared simulation DGP supports only these lag shapes.")
    phi = ([1+cell.rho] if p == 1 else
           [1+cell.rho+cell.lagged_outcome_difference, -cell.lagged_outcome_difference])
    beta = [.03-correlation*cell.drift, *phi]
    for q, delta, theta in ((qp, .25+correlation, cell.theta_positive),
                            (qn, .15+correlation, cell.theta_negative)):
        lagged_difference = .1 if q == 2 else 0.
        beta += [delta, theta-delta+lagged_difference]
        if q == 2:
            beta += [-lagged_difference]
    hold = max(cell.lags)
    result = generate_y(torch.zeros((1, hold), dtype=torch.float64, device="cpu"), xp, xn,
                        torch.tensor(beta, dtype=torch.float64, device="cpu"), p, qp, qn,
                        conditional[hold:][None, :])[0]
    return result[burn:].clone(), raw[burn:].clone(), {
        "burn_in": burn, "original_prefix": "zero outcome/raw initial levels before burn-in",
        "level_coefficients_full_dgp": beta, "raw_innovation": "centered Exp(1), variance 1" if cell.skew else "standard normal",
        "signed_level_anchors_at_observed_start": [float(xp[0, burn]), float(xn[0, burn])],
        "observed_levels_intercept": beta[0]+cell.theta_positive*float(xp[0, burn])+cell.theta_negative*float(xn[0, burn]),
        "conditional_outcome_innovation": "independent normal scaled sqrt(1-corr²)",
        "structural_outcome_innovation": "corr*raw_innovation+sqrt(1-corr²)*independent_normal",
        "conditional_coefficients_absorb_raw_current_correlation": True,
        "outcome_order_scope": "I(2) stress, excluded from inferential calibration" if not cell.within_maintained_class else "I(0)/I(1) maintained DGP",
    }


def wilson(hits, count):
    if count == 0:
        return [0., 1.]
    z, rate = 1.959963984540054, hits/count
    divisor = 1+z*z/count
    center = (rate+z*z/(2*count))/divisor
    half = z*math.sqrt(rate*(1-rate)/count+z*z/(4*count*count))/divisor
    return [max(0., center-half), min(1., center+half)]


def run_smoke(*, outer=12, inner=99, seed=18271, initial="fixed_prefix",
              recenter="pool", residual_scale="df", alpha=.05, batch_size=32):
    if (isinstance(outer, bool) or not isinstance(outer, int) or not 1 <= outer <= 100
            or isinstance(inner, bool) or not isinstance(inner, int) or not 19 <= inner <= 999
            or isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31):
        raise ValueError("Smoke is bounded to outer<=100 and 19<=inner<=999; not a full acceptance run.")
    planned = [asdict(cell) for cell in GRID]
    specification = {"grid": planned, "acceptance": ACCEPTANCE,
                     "outer": outer, "inner": inner, "seed": seed, "alpha": alpha,
                     "initial": initial, "recenter": recenter, "residual_scale": residual_scale,
                     "outer_seed_formula": "seed+100003*cell_index+1009*outer_index",
                     "inner_seed_formula": "outer_seed+524287"}
    predeclared = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    work = sum(3*inner*(cell.total-max(cell.lags))*(sum(cell.lags)+3)**2 for cell in GRID)*outer
    if work > 1_000_000_000:
        raise ValueError("Total smoke refit work exceeds the one-billion declared operation proxy.")
    start, cells = time.perf_counter(), []
    for ci, cell in enumerate(GRID):
        records, failures, hits = [], [], {null: 0 for null in NULLS}
        dgp = None
        all_three = 0
        for oi in range(outer):
            outer_seed = seed+100003*ci+1009*oi
            p, qp, qn = cell.lags
            try:
                y, x, dgp = generate_sample(cell, outer_seed)
                result = research_bootstrap(y, x, p=p, q_positive=qp, q_negative=qn,
                                            replications=inner, seed=outer_seed+524287,
                                            initial=initial, recenter=recenter,
                                            residual_scale=residual_scale, alpha=alpha,
                                            batch_size=batch_size, memory_mb=32)
            except (ResearchFailure, RuntimeError) as error:
                if isinstance(error, RuntimeError):
                    error = ResearchFailure("torch_numerical", "Native outer DGP or bootstrap call failed.",
                                            native_error=str(error))
                failures.append({"outer_index": oi, "outer_seed": outer_seed,
                                 "code": error.code, "message": str(error), "details": error.details})
                continue
            for null in NULLS:
                hits[null] += int(result["research_rejection_profile"][null])
            all_three += int(result["research_all_three_rejected"])
            records.append({"outer_index": oi, "outer_seed": outer_seed,
                            "sample_sha256": result["input_sha256"],
                            "index_stream_sha256": result["index_stream_sha256"],
                            "research_tail_probabilities": {null: result["tests"][null]["research_tail_probability_plus_one"] for null in NULLS},
                            "observed_statistics": result["observed_statistics"],
                            "all_three_rejected": result["research_all_three_rejected"]})
        failed = len(failures)
        rates = {}
        for null, count in {**hits, "all_three": all_three}.items():
            rates[null] = {"rejected_successful_calls": count, "planned_outer_denominator": outer,
                           "failed_outer_calls": failed, "rate_bounds_with_failed_calls": [count/outer, (count+failed)/outer],
                           "wilson95_no_failed_call_rejects": wilson(count, outer),
                           "wilson95_all_failed_calls_reject": wilson(count+failed, outer)}
        cells.append({"specification": asdict(cell), "dgp": dgp,
                      "completed_outer_calls": len(records), "failed_outer_calls": failed,
                      "invalid_draws_replaced": 0, "rates": rates, "failures": failures, "records": records})
    return {"private_research_only": True, "inferential_validity_established": False,
            "public_bounds_inference_unchanged": True, "acceptance_evaluated": False,
            "protocol": specification, "predeclared_protocol_sha256": predeclared,
            "planned_outer_calls": outer*len(GRID), "completed_outer_calls": sum(c["completed_outer_calls"] for c in cells),
            "failed_outer_calls": sum(c["failed_outer_calls"] for c in cells),
            "invalid_draws_replaced": 0, "refit_operation_proxy": work,
            "elapsed_seconds": time.perf_counter()-start, "cells": cells,
            "limitations": ["Smoke cannot establish size, power or nonlinear bootstrap validity",
                            "I(2) stress is excluded from maintained-class inferential calibration",
                            "No serial increment dynamics, symmetry constraints, controls, automatic lag selection, GPU or large-data validation",
                            "No convention selected based on best smoke results; protocol and failures retained"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer", type=int, default=12)
    parser.add_argument("--inner", type=int, default=99)
    parser.add_argument("--seed", type=int, default=18271)
    parser.add_argument("--initial", choices=["fixed_prefix", "original_block"], default="fixed_prefix")
    parser.add_argument("--recenter", choices=["pool", "draw"], default="pool")
    parser.add_argument("--residual-scale", choices=["none", "df"], default="df")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Never write anywhere implicitly. Caller chooses a research artifact path.
    with torch.device("cpu"), torch.inference_mode():
        result = run_smoke(outer=args.outer, inner=args.inner, seed=args.seed,
                           initial=args.initial, recenter=args.recenter, residual_scale=args.residual_scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+"\n")
    print(json.dumps({key: result[key] for key in ("planned_outer_calls", "completed_outer_calls", "failed_outer_calls", "elapsed_seconds", "predeclared_protocol_sha256")}))


if __name__ == "__main__":
    main()
