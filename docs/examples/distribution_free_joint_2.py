"""Four conservative count/support confidence families, synthetic inputs."""
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
        "growth": [0., .2, .4, .6, .8, 1., .3, .7],
        "income": [1., .9, .2, .4, .6, 0., .5, .8],
    }, index=[21, 22, 23, 24, 25, 26, 27, 28])
    designs = {
        "simultaneous_proportion_ci": oe.simultaneous_proportion_ci([0, 7, 20], [20, 20, 20],
            sampling_model="binomial_marginals", labels=["rare", "common", "complete"]),
        "multinomial_region": oe.multinomial_region([4, 0, 9, 7], sampling_model="iid_multinomial",
            labels=["alpha", "zero", "gamma", "delta"]),
        "hoeffding_mean_ci": oe.hoeffding_mean_ci(data, ["growth", "income"],
            bounds=[(0., 1.), (0., 1.)], sampling_model="iid_bounded"),
        "empirical_bernstein_mean_ci": oe.empirical_bernstein_mean_ci(data, ["growth", "income"],
            bounds=[(0., 1.), (0., 1.)], sampling_model="iid_bounded"),
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
    module = importlib.import_module("openecon.econometrics.postest.bounded_joint")
    forbidden = [name for name in ("scipy", "statsmodels", "linearmodels") if name in sys.modules]
    assert not forbidden
    proof = dict(part=2, frozen=bool(getattr(sys, "frozen", False)), module_origin=module.__file__,
        methods=list(designs), summary_hashes=states, summary_states=complete_states, output_contracts=frames,
        all_complete_state_restorations_equal=True, third_party_estimation_imports=forbidden,
        zero_count_category_preserved=bool(designs["multinomial_region"]["region"]["count"].eq(0).any()),
        simplex_constraint_saved=designs["multinomial_region"].attrs["constraint"])
    print("DISTRIBUTION_FREE_EIGHT_OK " + json.dumps(proof, allow_nan=False, sort_keys=True))
finally:
    torch.set_num_threads(old_threads)
