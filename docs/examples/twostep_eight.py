"""Synthetic full-state acceptance for eight bounded TwoStep stages."""
import json
import sys
from pathlib import Path
import openecon as oe

display = globals().get("display", print)
rows = [{"x": group*6 + (i%5)*0.06, "z": group*2+(i%7)*0.04,
         "kind": "ABC"[group]} for group in range(3) for i in range(18)]
rows += [{"x": 31.0, "z": 21.0, "kind": "C"},
         {"x": None, "z": 1.0, "kind": "A"}]
data = oe.DataFrame(rows)
before = data.to_json()
common = dict(data=data, continuous=["x", "z"], categorical=["kind"],
              threshold=0.005, max_preclusters=64, missing="drop")
fixed = oe.twostep(**common, n_clusters=3)
auto = oe.twostep(**common, criterion="bic", max_clusters=8)
aic = oe.twostep(**common, criterion="aic", max_clusters=8)
noise = oe.twostep(**common, n_clusters=3, min_precluster_size=2)
random = oe.twostep(**common, n_clusters=3, order="random", seed=172)
restored = oe.twostep_load(oe.twostep_save(fixed))
query = oe.twostep_assign(restored, data=oe.DataFrame([
    {"x": 0.1, "z": .1, "kind": "A"}, {"x": 12.1, "z": 4.1, "kind": "C"},
    {"x": None, "z": .1, "kind": "A"}]), missing="drop")
cut = oe.twostep_cut(restored, 2)
profiles = oe.twostep_profiles(restored)
quality = oe.twostep_quality(noise)
stability = oe.twostep_stability([fixed, random, noise])
outputs = [fixed["assignments"], fixed["merges"], cut["assignments"],
           auto["criteria"], profiles["continuous_profiles"], query["assignments"],
           noise["assignments"], stability["pairs"]]
for frame in outputs:
    display(frame)
states = {"fits": {name: oe.twostep_save(value) for name, value in
                   [("fixed", fixed), ("bic", auto), ("aic", aic), ("noise", noise),
                    ("random", random), ("cut", cut)]},
          "helpers": {name: oe.summary_state(value) for name, value in
                      [("profiles", profiles), ("query", query), ("quality", quality), ("stability", stability)]}}
assert oe.twostep_save(restored) == oe.twostep_save(fixed)
assert data.to_json() == before
assert fixed.attrs["n_missing"] == 1 and noise.attrs["n_noise"] >= 1
assert query["assignments"]["cluster"].iloc[:2].tolist() == [1, 3]
Path("twostep_eight_states.json").write_text(json.dumps(states, indent=2, allow_nan=False))
frozen = bool(getattr(sys, "frozen", False))
bundled = not frozen or Path(sys._MEIPASS) in Path(oe.__file__).parents
assert bundled and "scipy" not in sys.modules and "statsmodels" not in sys.modules
print("TWOSTEP_EIGHT_QA:" + json.dumps({"frozen": frozen, "sdk_from_bundle": bundled,
    "stages": 8, "table_rows": [len(x) for x in outputs], "full_state_roundtrip": True,
    "input_unchanged": True, "native_only": True, "physical_missing_noise_saved": True}))
