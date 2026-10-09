"""Four finite transformation/entire-distribution contracts, synthetic rows."""
import hashlib
import importlib
import json
import sys

import pandas as pd
import torch
import openecon as oe

old_threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    data = pd.DataFrame({
        "growth": [-2., 1., 3., -1., 4., 2., -3.],
        "income": [2., 3., 1., -2., 3., -1., 4.],
        "jobs": [-1., 4., 2., 3., -3., 1., 2.],
    }, index=[10, 11, 12, 13, 14, 15, 16])
    groups = pd.DataFrame({
        "growth": [1., 2., 4., 5., 3., 2., 6., 7.],
        "income": [5., 3., 2., 1., 4., 7., 6., 8.],
        "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
    })
    designs = {
        "mean_sign_stepdown": oe.mean_sign_stepdown(data, ["growth", "income", "jobs"],
            symmetry_model="Independent rows; every true-null subvector jointly centrally symmetric about declared null", null_values={"growth":0., "income":0., "jobs":0.}),
        "mean_permutation_stepdown": oe.mean_permutation_stepdown(groups, ["growth", "income"], "group",
            exchangeability_model="Fixed group sizes; pooled true-null outcome subvectors exchangeable"),
        "simultaneous_dkw_band": oe.simultaneous_dkw_band(data, ["growth", "income", "jobs"], sampling_model="iid_marginals"),
        "simultaneous_quantile_ci": oe.simultaneous_quantile_ci(data, ["growth", "income"], [.01, .5], sampling_model="iid_marginals"),
    }
    states, complete_states, frames = {}, {}, []
    for name, result in designs.items():
        state = oe.summary_state(result)
        assert state == oe.summary_state(oe.restore_summary(state))
        states[name] = hashlib.sha256(state.encode()).hexdigest()
        complete_states[name] = json.loads(state)
        for key, frame in result.items():
            assert len(frame.columns) <= 30
            assert all(len(str(cell)) <= 500 for row in frame.to_numpy() for cell in row)
            for start in range(0, len(frame), 50):
                chunk = frame.iloc[start:start+50]
                frames.append(dict(method=name, key=key, start=start, rows=len(chunk), columns=list(frame.columns)))
                display(chunk)  # noqa: F821 -- native console display
    module = importlib.import_module("openecon.econometrics.postest.distribution_free_joint")
    forbidden = [name for name in ("scipy", "statsmodels", "linearmodels") if name in sys.modules]
    assert not forbidden
    proof = dict(part=1, frozen=bool(getattr(sys, "frozen", False)), module_origin=module.__file__,
        methods=list(designs), summary_hashes=states, summary_states=complete_states, output_contracts=frames,
        all_complete_state_restorations_equal=True, third_party_estimation_imports=forbidden,
        quantile_unbounded_endpoint_verified=bool(designs["simultaneous_quantile_ci"]["intervals"].lower_unbounded.any()),
        orbit_counts={name: result.attrs.get("orbit_count") for name, result in designs.items() if "stepdown" in name})
    print("DISTRIBUTION_FREE_EIGHT_OK " + json.dumps(proof, allow_nan=False, sort_keys=True))
finally:
    torch.set_num_threads(old_threads)
