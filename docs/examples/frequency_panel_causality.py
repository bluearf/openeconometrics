"""Run in an OpenEconometrics code panel. Synthetic stationary observations."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
import openecon as oe

generator = torch.Generator(device="cpu").manual_seed(71921)
units, periods = 12, 160
levels = torch.randn((units, periods + 100, 2), generator=generator, dtype=torch.float64)
for t in range(2, periods + 100):
    levels[:, t, 0] += .15 + .35 * levels[:, t - 1, 0] - .12 * levels[:, t - 2, 0] \
        + .35 * levels[:, t - 1, 1]
    levels[:, t, 1] += .1 + .4 * levels[:, t - 1, 1]
frame = pd.DataFrame(levels[:, -periods:].reshape(-1, 2).numpy(), columns=["output", "investment"])
frame["unit"] = [unit for unit in range(units) for _ in range(periods)]
frame["year"] = list(range(2000, 2000 + periods)) * units

frequency = oe.bccaustest(data=frame.iloc[:periods], y="output", x="investment",
                          time="year", lags=3, frequencies=[.2, .8, 2.4])
panel = oe.dhcausality(data=frame, y="output", x="investment", panel="unit", time="year", lags=2)
assert frequency.attrs["pointwise"] and frequency.attrs["residual_df"] == 150
assert frequency.attrs["cause"] == "investment" and frequency.attrs["effect"] == "output"
assert len(frequency["tests"]) == 3 and len(frequency["restrictions"]) == 9
assert panel.attrs["units"] == units and panel.attrs["usable_periods"] == periods - 2
assert panel.attrs["tail"] == "upper" and panel["tests"].loc[0, "p_value"] is None
assert panel["tests"].loc[2, "p_value"] < .01
assert len(panel["individual"]) == units and len(panel["coefficients"]) == 60
assert all(result.attrs["device"] == "cpu" and result.attrs["dtype"] == "float64"
           for result in (frequency, panel))

full_results = {}
for name, result in (("frequency", frequency), ("panel", panel)):
    full_results[name] = {"attrs": result.attrs,
                          "tables": {key: table.to_dict(orient="split") for key, table in result.items()},
                          "latex": result.to_latex()}
    encoded = json.dumps(full_results[name], sort_keys=True, allow_nan=False)
    assert json.loads(encoded) == full_results[name]
    assert "\\begin{table}" in full_results[name]["latex"]
    if "CAUSALITY_RESULT_DIRECTORY" in globals():
        result_root = Path(globals()["CAUSALITY_RESULT_DIRECTORY"])
        result_root.mkdir(parents=True, exist_ok=True)
        (result_root / (name + ".json")).write_text(encoded)
    if "display" in globals():
        for output in result.values():
            globals()["display"](output)

print("CAUSALITY_ACCEPTANCE_OK " + json.dumps({
    "methods": ["bccaustest", "dhcausality"], "tables": 6,
    "frequency_statistics": frequency["tests"].statistic.tolist(),
    "panel_statistics": panel["tests"].statistic.tolist(),
    "complete_frequency_coefficients": len(frequency["coefficients"]),
    "complete_panel_coefficients": len(panel["coefficients"]),
    "sample_hashes": [result.attrs["sample_sha256"] for result in (frequency, panel)],
    "full_results_sha256": {name: hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                            for name, payload in full_results.items()},
    "publication_latex": True, "metadata_json_roundtrip": True,
}))
