"""Complete synthetic saved proxy-SVAR workflow, source Python or console.

Writes and reads owned full JSON state files in the current working directory.
Printed/displayed responses are previews. No vendor or desktop proof claimed.
"""

import json
from pathlib import Path

import pandas as pd
import torch

import openecon as oe


generator = torch.Generator(device="cpu").manual_seed(78213)
shocks = torch.randn((160, 3), generator=generator, dtype=torch.float64)
impact_matrix = torch.tensor([[1, .3, .1], [.5, 1, -.2], [-.4, .2, .8]], dtype=torch.float64)
a = torch.tensor([[.35, .1, 0], [.05, .3, -.05], [0, .1, .25]], dtype=torch.float64)
innovations = shocks @ impact_matrix.T
levels = torch.zeros((160, 3), dtype=torch.float64)
for period in range(1, len(levels)):
    levels[period] = a @ levels[period-1] + innovations[period]
series = pd.DataFrame(levels.tolist(), columns=["output", "inflation", "rate"])
series["period"] = range(len(levels))
reduced_form = oe.var(data=series, y=["output", "inflation", "rate"], time="period",
                      lags=1, maxlag=1, irf_steps=1, irf_kinds=["simple"], lm_lags=0)
proxy = pd.DataFrame({"period": range(1, len(levels)),
                      "surprise": (shocks[1:, 0] + .3*torch.randn(len(levels)-1, generator=generator, dtype=torch.float64)).tolist()})
model_file = Path.cwd()/"next-eight-reduced-var.json"
model_file.write_text(reduced_form.model_dump_json())
restored_var = oe.ResultBundle.model_validate_json(model_file.read_text())
identified = oe.proxy_svar(
    restored_var, data=series, proxy_data=proxy.iloc[::-1], proxy="surprise", key="period",
    normalize="output", impact=1.0,
    source="Prespecified synthetic independent innovations; noisy target-shock proxy, seed 78213",
    exogeneity_assumed=True, steps=8,
)
state_file = Path.cwd()/"next-eight-proxy-state.json"
state_file.write_text(oe.summary_state(identified))
restored = oe.restore_summary(state_file.read_text())
replayed = oe.proxy_svar_irf(restored, steps=8)
assert replayed.irf.tolist() == identified["responses"].irf.tolist()
assert restored.attrs["proxy_state"]["sample_positions"] == restored_var.sample_positions
assert len(restored.attrs["proxy_state"]["residuals"]) == 159
assert restored.attrs["proxy_state"]["normalization"]["impact"] == 1.0
assert not restored.attrs["proxy_state"]["inference_available"]
proof = {
    "status": "passed", "synthetic": True, "seed": 78213,
    "equations": 3, "retained_rows": restored_var.nobs,
    "response_rows": len(replayed), "complete_json_readback": True,
    "source_var_refitted": False, "sampling_inference_available": False,
    "fevd_available": False, "vendor_execution": False,
    "desktop_validation": False,
}
print("PROXY_SVAR_OK " + json.dumps(proof, sort_keys=True))
show = globals().get("display")
if show:
    show(identified["impact"])
    show(replayed)
