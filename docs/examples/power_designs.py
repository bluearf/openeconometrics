"""Eight synthetic prospective designs; complete saved results and all solve modes.

Displays 16 complete tables, saves all 26 tables/settings and reproducible inputs.
Run in the code panel; no data/retrospective-power or public release claim.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import openecon as oe
import torch

INPUTS = {
    "power_tmean": dict(effect=.5, sd=1., power=.8),
    "power_ttwomeans": dict(effect=.5, sd=1., ratio=1.5, power=.8),
    "power_tpaired": dict(effect=.5, sd1=1., sd2=1.2, correlation=.6, power=.8),
    "power_anova": dict(effect=.25, groups=3, power=.8),
    "power_regression": dict(effect=.15, predictors=5, tested=2, power=.8),
    "power_cluster_mean": dict(effect=.5, sd=1., cluster_size=20, icc=.1, power=.8),
    "power_gof": dict(null_probabilities=[.25]*4, proposed_probabilities=[.4,.3,.2,.1], power=.8),
    "power_independence": dict(joint_probabilities=[[.35,.15],[.15,.35]], power=.8),
}
previous_threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    results, encoded, proof = {}, {}, {}
    for name, inputs in INPUTS.items():
        fn = getattr(oe, name)
        result = fn(**inputs)
        results[name] = result
        row = result["plan"].iloc[0]
        assert row.power >= .8 and int(row.n) > 2
        explicit = dict(inputs, n=int(row.n))
        explicit.pop("power")
        assert math.isclose(fn(**explicit)["plan"].iloc[0].power, row.power, abs_tol=1e-10)
        explicit["n"] -= 1
        assert fn(**explicit)["plan"].iloc[0].power < .8
        detectable = dict(inputs, n=max(200, int(row.n)), power=.8)
        detectable["strength" if name in ("power_gof", "power_independence") else "effect"] = None
        assert math.isclose(fn(**detectable)["plan"].iloc[0].power, .8, abs_tol=1e-9)
        payload = {"inputs": inputs, "attrs": result.attrs,
                   "tables": {key: frame.astype(object).where(frame.notna(), None).to_dict(orient="split")
                              for key, frame in result.items()}, "latex": result.to_latex()}
        content = json.dumps(payload, sort_keys=True, allow_nan=False)
        saved = json.loads(content)
        replay = fn(**saved["inputs"])
        assert replay.attrs == saved["attrs"]
        assert replay.to_latex() == saved["latex"]
        encoded[name] = content
        proof[name] = {"n": int(row.n), "power": float(row.power),
                       "table_rows": {key: len(frame) for key, frame in result.items()},
                       "sha256": hashlib.sha256(content.encode()).hexdigest(),
                       "inference": result.attrs["inference"]}
        if "display" in globals():
            globals()["display"](result["plan"])
            globals()["display"](result["scenarios"])
    if "POWER_DESIGN_RESULT_DIRECTORY" in globals():
        directory = Path(globals()["POWER_DESIGN_RESULT_DIRECTORY"])
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in encoded.items():
            (directory / (name + ".json")).write_text(content)
    forbidden = [name for name in ("scipy", "statsmodels", "linearmodels") if name in sys.modules]
    assert not forbidden
    print("POWER_DESIGN_ACCEPTANCE_OK " + json.dumps({"methods": proof, "all_solve_modes_verified": True,
        "full_input_replay_verified": True, "frozen": bool(getattr(sys, "frozen", False)),
        "saved_tables": 26, "displayed_tables": 16, "third_party_estimation_imports": forbidden},
        sort_keys=True, allow_nan=False))
finally:
    torch.set_num_threads(previous_threads)
