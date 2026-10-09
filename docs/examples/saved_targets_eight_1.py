"""Synthetic complete saved-target native acceptance example."""
import base64
import hashlib
import importlib
import json
import sys
import zlib

import numpy as np
import pandas as pd
import torch
import openecon as oe
from openecon.econometrics.core import TableSet, table
from openecon.models import ResultBundle

old_threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    PART = 1
    rng = np.random.default_rng(6419)
    n = 144
    frame = pd.DataFrame({"x": rng.normal(size=n), "g": np.resize(["a", "b", "c"], n),
        "z1": rng.normal(size=n), "z2": rng.normal(size=n), "z3": rng.normal(size=n)})
    v = rng.normal(size=(n, 2))
    frame["p1"] = frame.z1 + .3 * frame.z3 + .2 * frame.x + v[:, 0]
    frame["p2"] = frame.z2 + .2 * frame.z3 - .1 * frame.x + v[:, 1]
    frame["y"] = 1 + .4 * frame.x + .6 * (frame.g == "b") - .2 * (frame.g == "c") + rng.normal(size=n)
    frame.loc[:9, "y"] += 20
    fits = {name: getattr(oe, name)(data=frame, y="y", x=["x", "g"], categorical=["g"],
        starts=60, seed=73, missing="drop") for name in ("sreg", "mmreg")}
    iv = frame.copy()
    iv["y"] = 1 + .4 * iv.x + .8 * iv.p1 - .3 * iv.p2 + .5 * v[:, 0] + rng.normal(size=n)
    fits["ivcue"] = oe.ivcue(data=iv, y="y", x=["x"], endog=["p1", "p2"], instruments=["z1", "z2", "z3"], missing="drop")
    rng = np.random.default_rng(18)
    top = np.repeat(np.arange(6), 16)
    mid = np.tile(np.repeat(np.arange(2), 8), 6)
    low = np.tile(np.repeat(np.arange(2), 4), 12)
    x = rng.normal(size=len(top))
    y = (2 + .4 * x + rng.normal(size=6)[top] + rng.normal(size=12)[top * 2 + mid]
        + rng.normal(size=24)[top * 4 + mid * 2 + low] + rng.normal(size=len(top)) * .4)
    fits["mixedflex"] = oe.mixedflex(data=pd.DataFrame(dict(y=y, x=x, top=top, mid=mid, low=low)),
        y="y", x=["x"], group=["top", "mid", "low"], grouping="nested", missing="drop")
    models = {name: json.loads(fit.model_dump_json()) for name, fit in fits.items()}
    query = pd.DataFrame({"x": [-1.2, .3, 1.7, -.8], "g": ["c", "a", "b", "a"],
        "p1": [.4, -1., .8, -.2], "p2": [-.9, .3, 1.3, .7]},
        index=pd.Index(["same", "same", "last", "last"], name="source_row"))
    modules = {"sreg": ("robust.smm", "fit_sreg"), "mmreg": ("robust.smm", "fit_mmreg"),
        "ivcue": ("iv.cue", "fit_ivcue"), "mixedflex": ("mixed.flexible_lmm", "fit_mixedflex")}
    saved_functions = []
    def refuse_fit(*args, **kwargs):
        raise AssertionError("Restored target evaluation called an estimator")
    targets = {}
    try:
        for module_name, function in modules.values():
            module = importlib.import_module("openecon.econometrics." + module_name)
            saved_functions.append((module, function, getattr(module, function)))
            setattr(module, function, refuse_fit)
        for name, fit in fits.items():
            saved = ResultBundle.model_validate_json(fit.model_dump_json())
            assert json.loads(saved.model_dump_json()) == models[name]
            prediction = oe.predict(saved, query, interval="mean", alpha=.1)
            effects = oe.margins(saved, "x", data=query, at={"x": [-.5, .8]})
            terms = [c.term for c in saved.coefficients]
            targets[name] = TableSet({"means": prediction, "effects": effects,
                "jacobian": table(effects.attrs["delta_gradients"], columns=terms),
                "parameter_covariance": table(saved.covariance_matrix, columns=terms), "sample": table(query)},
                title="Saved " + name + " location/population target", estimator=name,
                prediction_metadata=prediction.attrs, effects_metadata=effects.attrs,
                target=prediction.attrs["response_definition"], refitted=False)
    finally:
        for module, function, original in saved_functions:
            setattr(module, function, original)
    visible_keys = ["means", "effects", "jacobian", "parameter_covariance", "sample"]
    complete_states, summary_hashes, frames = {}, {}, []
    for name, target in targets.items():
        # Current summary schema keeps index labels; retain names explicitly.
        target.attrs["table_index_names"] = {key: list(frame.index.names) for key, frame in target.items()}
        state = oe.summary_state(target)
        restored_target = oe.restore_summary(state)
        for key, names in restored_target.attrs["table_index_names"].items():
            restored_target[key].index.names = names
            assert restored_target[key].index.equals(target[key].index)
            assert restored_target[key].index.names == target[key].index.names
        assert oe.summary_state(restored_target) == state
        complete_states[name] = json.loads(state)
        summary_hashes[name] = hashlib.sha256(state.encode()).hexdigest()
        for key in visible_keys:
            frame = target[key]
            assert len(frame) <= 50 and len(frame.columns) <= 30
            assert all(len(str(cell)) <= 500 for row in frame.to_numpy() for cell in row)
            frames.append(dict(method=name, key=key, start=0, rows=len(frame), columns=list(frame.columns),
                index_names=list(frame.index.names)))
            display(frame)  # noqa: F821 -- actual native console display
    forbidden = [name for name in ("scipy", "statsmodels", "linearmodels") if name in sys.modules]
    assert not forbidden
    proof = dict(part=PART, frozen=bool(getattr(sys, "frozen", False)), methods=list(targets),
        summary_states=complete_states, summary_hashes=summary_hashes, model_states=models,
        output_contracts=frames, all_complete_state_restorations_equal=True,
        third_party_estimation_imports=forbidden, restored_models_used_without_refit=True)
    encoded = base64.b64encode(zlib.compress(json.dumps(proof, allow_nan=False, sort_keys=True).encode(), 9)).decode()
    assert len(encoded) < 63_000  # complete state must fit the existing console stdout limit
    print("SAVED_TARGETS_EIGHT_OK " + encoded)
finally:
    torch.set_num_threads(old_threads)
