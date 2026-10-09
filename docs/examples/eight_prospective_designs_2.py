"""Conditional counts and random-width assurance; synthetic assumptions only."""

import importlib
import json
import math
import sys

import openecon as oe
import torch

previous_threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    designs = {
        "power_logrank": oe.power_logrank(0.7, power=0.8, event_fraction=0.6),
        "power_mcnemar": oe.power_mcnemar(0.7, power=0.8, discordance_fraction=0.3),
        "precision_mean_unknown": oe.precision_mean_unknown(sd=1.0, width=0.5, assurance=0.9),
        "precision_variance": oe.precision_variance(variance=1.0, width=1.0, assurance=0.9),
    }
    for result in designs.values():
        for frame in result.values():
            display(frame)  # noqa: F821 -- supplied by the OpenEconometrics console
    hr = oe.power_logrank(events=300, power=0.8, direction="lower")["plan"].iloc[0].hazard_ratio
    assert math.isclose(oe.power_logrank(hr, events=300)["plan"].iloc[0].power, 0.8, abs_tol=1e-9)
    r = oe.power_mcnemar(discordant_pairs=100, power=0.8)["plan"].iloc[0].probability
    assert math.isclose(
        oe.power_mcnemar(r, discordant_pairs=100)["plan"].iloc[0].power, 0.8, abs_tol=1e-9
    )
    for name, key, value in [
        ("precision_mean_unknown", "sd", 1.0),
        ("precision_variance", "variance", 1.0),
    ]:
        n = int(designs[name]["plan"].iloc[0].n)
        checked = getattr(oe, name)(**{key: value}, n=n, assurance=0.9)["plan"].iloc[0]
        assert math.isclose(checked.width_probability, 0.9, abs_tol=1e-9)
        assert checked.width <= designs[name]["plan"].iloc[0].target_width
    boundary = oe.power_mcnemar(0.7, discordant_pairs=5, alpha=0.0625)["plan"].iloc[0]
    assert boundary.lower_critical == 0 and boundary.upper_critical == 5
    assert boundary.actual_alpha == 0.0625
    module = importlib.import_module("openecon.econometrics.stats.planning_extended")
    forbidden = [name for name in ("scipy", "statsmodels", "linearmodels") if name in sys.modules]
    assert not forbidden
    proof = dict(
        exact_dyadic_rejection_boundary_verified=True,
        part=2,
        frozen=bool(getattr(sys, "frozen", False)),
        module_origin=module.__file__,
        third_party_estimation_imports=forbidden,
        all_solve_modes_verified=True,
        synthetic_assumptions_only=True,
        methods={name: result.attrs for name, result in designs.items()},
        plan_rows={name: result["plan"].to_dict("records") for name, result in designs.items()},
        output_rows=[len(frame) for result in designs.values() for frame in result.values()],
    )
    print("EIGHT_DESIGNS_OK " + json.dumps(proof, allow_nan=False, sort_keys=True))
finally:
    torch.set_num_threads(previous_threads)
