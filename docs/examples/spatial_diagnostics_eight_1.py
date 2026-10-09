"""Synthetic native saved OLS spatial-diagnostic acceptance; complete states."""
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

PART = 1
METHODS = ['moran_normal', 'moran_gaussian_mc', 'lm_error']
old_threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    n = 28
    rng = np.random.default_rng(841)
    keys = [f"region-{i}" for i in range(n)]
    edges = [(keys[i], keys[j], weight) for i in range(n)
             for j, weight in (((i + 1) % n, 1.), ((i + 4) % n, .7), ((i + 9) % n, .2))]
    W = oe.spatial_weights(keys, edges)
    x, z, noise = rng.normal(size=(3, n))
    frame = pd.DataFrame(dict(region=keys, y=1.2 + .8*x - .6*z + noise, x=x, z=z),
                         index=pd.Index([f"pair-{i//2}" for i in range(n)], name="caller_row"))
    fitted = oe.ols(data=frame, y="y", x=["x", "z"], covariance="nonrobust")
    model_state = json.loads(fitted.model_dump_json())
    saved = ResultBundle.model_validate_json(fitted.model_dump_json())
    assert json.loads(saved.model_dump_json()) == model_state
    ols_module = importlib.import_module("openecon.linear_ols")
    original_fit = ols_module.fit_ols
    def refuse_fit(*args, **kwargs):
        raise AssertionError("Saved spatial diagnostic evaluation refitted original OLS")
    ols_module.fit_ols = refuse_fit
    summaries = {}
    try:
        for method in METHODS:
            summaries[method] = oe.spatial_diagnostics(saved, data=frame, key="region", spatial_weights=W,
                tests=[method], wx_predictors=["x", "z"], simulations=39, seed=273)
    finally:
        ols_module.fit_ols = original_fit
    complete_states, hashes, outputs = {}, {}, []
    for method, output in summaries.items():
        assert json.loads(output.attrs["source_model"]) == model_state
        encoded = oe.summary_state(output)
        state = json.loads(encoded)
        restored = oe.restore_summary(encoded)
        for name, names in output.attrs["table_index_names"].items():
            restored[name].index.names = names
        assert json.loads(oe.summary_state(restored)) == state
        assert all(restored[name].index.names == output[name].index.names for name in output)
        complete_states[method] = state
        hashes[method] = hashlib.sha256(json.dumps(state, allow_nan=False, sort_keys=True).encode()).hexdigest()
        for name, table in restored.items():
            if name == "settings" or len(table) == 0:
                continue
            assert len(table) <= 50 and len(table.columns) <= 30
            outputs.append((method, name, table))
    assert len(outputs) <= 20
    output_contracts = []
    for method, name, table in outputs:
        output_contracts.append(dict(method=method, key=name, rows=len(table), columns=list(table.columns),
                                     index_names=list(table.index.names)))
        if "display" in globals():
            globals()["display"](table)
    proof = dict(part=PART, methods=METHODS, frozen=bool(getattr(sys,"frozen",False)),
        complete_summary_states=complete_states, summary_hashes=hashes, source_model_state=model_state,
        complete_source_model_retained=True, restored_ols_used_without_original_refit=True,
        all_complete_state_restorations_equal=True, output_contracts=output_contracts,
        third_party_estimation_imports=any(name in sys.modules for name in ("scipy","statsmodels")),
        spatial_fit_or_new_graph_inferred=False, public_release_delivered=False)
    encoded_proof = base64.b64encode(zlib.compress(json.dumps(proof,allow_nan=False).encode(),9)).decode()
    assert len(encoded_proof) < 64_000
    print("SPATIAL_DIAGNOSTICS_EIGHT_OK " + encoded_proof)
finally:
    torch.set_num_threads(old_threads)
