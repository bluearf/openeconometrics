"""Four bounded normal-law design contracts; synthetic scalar assumptions only."""

import importlib
import json
import math
import sys

import openecon as oe
import torch

previous_threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    specs = [
        ("power_paired_mean", dict(sd_before=1.0, sd_after=1.2, correlation=0.6), 0.5, "effect"),
        ("power_two_proportions", dict(p1=0.3, ratio=1.5), 0.5, "p2"),
        ("power_two_correlations", dict(rho1=0.2, ratio=1.5), 0.5, "rho2"),
        ("power_slope", dict(error_sd=1.3, design_variance=0.7), 0.5, "effect"),
    ]
    designs = {}
    for name, assumptions, effect, effect_key in specs:
        fn = getattr(oe, name)
        result = fn(**assumptions, **{effect_key: effect}, power=0.8)
        designs[name] = result
        for frame in result.values():
            display(frame)  # noqa: F821 -- supplied by the OpenEconometrics console
        n = int(result["plan"].iloc[0].n)
        checked = fn(**assumptions, **{effect_key: effect}, n=n)
        assert checked["plan"].iloc[0].power >= 0.8
        detectable = fn(**assumptions, n=200, power=0.8)
        checked = fn(
            **assumptions, **{effect_key: float(detectable["plan"].iloc[0][effect_key])}, n=200
        )
        assert math.isclose(checked["plan"].iloc[0].power, 0.8, abs_tol=1e-9)
    module = importlib.import_module("openecon.econometrics.stats.planning_extended")
    forbidden = [name for name in ("scipy", "statsmodels", "linearmodels") if name in sys.modules]
    assert not forbidden
    proof = dict(
        part=1,
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
