"""Four independent prospective designs; every saved table has <=50 rows.

Synthetic scalar assumptions only. No observed-data or retrospective-power claim.
"""
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
        "mean": oe.power_mean(.5, sd=1., sd2=1.4, ratio=1.5, power=.8),
        "proportion": oe.power_proportion(.5, .7, power=.8),
        "correlation": oe.power_correlation(0., .3, power=.8),
        "precision": oe.precision_mean(sd=.8, width=.5),
    }
    for name, result in designs.items():
        print(name, result["plan"].to_string(index=False))
        for frame in result.values():
            display(frame)
    # Exercise the other solve modes in the same installed console.
    solved = [
        oe.power_mean(sd=1., sd2=1.4, ratio=1.5, n=100, power=.8),
        oe.power_proportion(.5, n=100, power=.8),
        oe.power_correlation(0., n=100, power=.8),
    ]
    for result in solved:
        assert math.isclose(result["plan"].iloc[0].power, .8, abs_tol=1e-10)
    checked = [
        oe.power_mean(solved[0]["plan"].iloc[0].effect, sd=1., sd2=1.4, ratio=1.5, n=100),
        oe.power_proportion(.5, solved[1]["plan"].iloc[0].p1, n=100),
        oe.power_correlation(0., solved[2]["plan"].iloc[0].rho, n=100),
    ]
    for result in checked:
        assert math.isclose(result["plan"].iloc[0].power, .8, abs_tol=1e-10)
    assert oe.precision_mean(sd=.8, n=40)["plan"].iloc[0].width <= .5
    # Training-free scenario curve; these sizes are chosen prospectively.
    curve = oe.DataFrame([dict(n=n, power=oe.power_mean(.5, sd=1., n=n)["plan"].iloc[0].power)
                          for n in (10, 20, 30, 40, 50, 60, 80, 100)])
    display(curve)
    module = importlib.import_module("openecon.econometrics.stats.planning")
    forbidden = [name for name in ("scipy", "statsmodels", "linearmodels") if name in sys.modules]
    assert not forbidden
    proof = dict(
        frozen=bool(getattr(sys, "frozen", False)),
        module_origin=module.__file__, third_party_estimation_imports=forbidden,
        methods={name: result.attrs for name, result in designs.items()},
        plan_rows={name: result["plan"].to_dict(orient="records") for name, result in designs.items()},
        output_rows=[len(frame) for result in designs.values() for frame in result.values()] + [len(curve)],
        all_solve_modes_verified=True, synthetic_assumptions_only=True,
        exact_binomial_critical_counts_saved=True, approximate_correlation_explicit=True,
    )
    print("PROSPECTIVE_PLANNING_OK " + json.dumps(proof, allow_nan=False, sort_keys=True))
finally:
    torch.set_num_threads(previous_threads)
