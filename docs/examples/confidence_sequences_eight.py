"""Eight declared time-uniform confidence families; synthetic, offline, replayable.

Run in the OpenEconometrics code panel. Complete results are saved beneath the
current project in confidence_sequence_results; display previews are separate.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import openecon as oe
from openecon.resources import use_workspace_budget


try:
    display
except NameError:
    def display(value):
        print(value)


base = [.2, .8, .5, .3, .9, .1, .7, .4, .6, .25, .75, .45]
cases = [
    ("bernoulli_confidence_sequence", "iid_bernoulli", [int(v > .45) for v in base], {}),
    ("poisson_confidence_sequence", "iid_poisson", [int(v*6) for v in base], {}),
    ("normal_mean_confidence_sequence", "iid_normal", [3*v-1 for v in base], {"sd": [1.2, .8]}),
    ("student_mean_confidence_sequence", "iid_normal", [3*v-1 for v in base], {}),
    ("normal_variance_confidence_sequence", "iid_normal", [3*v-1 for v in base], {}),
    ("exponential_mean_confidence_sequence", "iid_exponential", [4*v for v in base], {}),
    ("uniform_endpoint_confidence_sequence", "iid_uniform_zero", base, {}),
    ("hoeffding_confidence_sequence", "iid_bounded", base, {"bounds": [(0., 1.), (0., 1.)]}),
]
out = Path("confidence_sequence_results")
out.mkdir(exist_ok=True)
files = {}
for name, model, values, options in cases:
    data = oe.DataFrame({"first": values, "second": list(reversed(values))},
                        index=[f"unit-{i//2}" for i in range(len(values))])
    original = data.copy(deep=True)
    with use_workspace_budget(64):
        result = getattr(oe, name)(data, ["first", "second"], sampling_model=model, **options)
    assert data.equals(original) and data.index.equals(original.index)
    state = oe.summary_state(result)
    restored = oe.restore_summary(state)
    assert oe.summary_state(restored) == state
    assert len(restored["intervals"]) == 24 and len(restored["sample"]) == 12
    assert restored.attrs["time_uniform"] and restored.attrs["fixed_family_required"]
    assert restored.attrs["device"] == "cpu" and not restored.attrs["dataset_support"]
    assert restored["sample"].position.tolist() == list(range(12))
    tex = result["intervals"].to_latex()+"\n"+result["sample"].to_latex()
    for suffix, value in (("json", state), ("tex", tex)):
        path = out/f"{name}.{suffix}"
        path.write_text(value, encoding="utf-8", newline="\n")
        raw = path.read_bytes()
        files[path.name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    display(result["intervals"])

receipt = {"stages": 8, "tables": 8, "full_states": 8, "files": files,
           "full_state_roundtrip": True, "complete_ordered_samples": True,
           "input_unchanged": True, "native_cpu_float64": True,
           "frozen": bool(getattr(sys, "frozen", False)),
           "scipy_imported": "scipy" in sys.modules,
           "sdk_file": str(Path(oe.__file__).resolve()),
           "source_path_injected": any(p.replace("\\", "/").endswith("/src") or "/src/openecon" in p.replace("\\", "/") for p in sys.path),
           "human_data_access": False, "public_release_delivered": False}
if receipt["frozen"]:
    assert importlib.util.find_spec("scipy") is None
    assert Path(oe.__file__).resolve().is_relative_to(Path(sys._MEIPASS).resolve())
    assert not receipt["scipy_imported"] and not receipt["source_path_injected"]
Path("confidence_sequence_acceptance.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
print("CONFIDENCE_SEQUENCES_ACCEPTANCE_OK "+json.dumps(receipt, sort_keys=True))
