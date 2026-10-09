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
from openecon.models import ResultBundle

old_threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    PART = 2
    n = 24
    keys = [f"unit-{i}" for i in range(n)]
    edges = [(keys[i], keys[j], weight) for i in range(n)
        for j, weight in (((i - 1) % n, 1.), ((i + 1) % n, 2.), ((i + 5) % n, .3))]
    weights = oe.spatial_weights(keys, edges)
    w = weights.dense().numpy()
    rng = np.random.default_rng(281)
    x, z = rng.normal(size=n), rng.normal(size=n)
    y = np.linalg.solve(np.eye(n) - .35 * w,
        1 + .8 * x - .4 * z + np.linalg.solve(np.eye(n) - .2 * w, rng.normal(size=n)))
    data = pd.DataFrame(dict(id=keys, y=y, x=x, z=z))
    we = np.roll(w, 2, axis=1)
    np.fill_diagonal(we, 0)
    we /= we.sum(axis=1)[:, None]
    error_weights = oe.spatial_weights(keys, [(keys[i], keys[j], we[i, j]) for i, j in zip(*np.nonzero(we), strict=True)])
    fits = {name: getattr(oe, name)(data, "y", ["x", "z"], key="id", spatial_weights=weights,
        **({"error_weights": error_weights} if name == "sac" else {})) for name in ("sar", "sem", "sac", "sdm")}
    models = {name: json.loads(fit.model_dump_json()) for name, fit in fits.items()}
    query = data[["id", "x", "z"]].iloc[::-1].copy()
    query["x"] = query.x * 1.3 + .4
    query.index = pd.Index(["duplicate"] * len(query), name="query_row")
    module = importlib.import_module("openecon.econometrics.spatial.estimators")
    original = module.fit_spatial
    def refuse_fit(*args, **kwargs):
        raise AssertionError("Restored spatial evaluation called an estimator")
    module.fit_spatial = refuse_fit
    targets = {}
    try:
        for name, fit in fits.items():
            saved = ResultBundle.model_validate_json(fit.model_dump_json())
            assert json.loads(saved.model_dump_json()) == models[name]
            targets[name] = oe.spatial_predict(saved, data=query, alpha=.1)
            assert targets[name].attrs["refitted"] is False
            assert targets[name]["means"].key.tolist() == query.id.tolist()
    finally:
        module.fit_spatial = original
    visible_keys = ["means", "jacobian", "mean_covariance"]
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
