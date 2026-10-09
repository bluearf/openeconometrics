"""Four native fractional-memory outputs and independently restorable state."""

import json
import math
from io import StringIO

import pandas as pd
import torch
import openecon as oe
from openecon.models import ResultBundle

generator = torch.Generator().manual_seed(203206)
innovations = torch.randn(768, dtype=torch.float64, generator=generator)
# Independent direct binomial/integration recurrence for a synthetic process.
integration = [1.]
for lag in range(1, 128):
    integration.append(integration[-1] * (lag - 1 + .25) / lag)
values = []
for position in range(len(innovations)):
    size = min(position + 1, len(integration))
    values.append(1.5 + sum(integration[j] * float(innovations[position - j]) for j in range(size)))

difference = oe.fracdiff(values, .25, terms=128)
diagnostic = oe.gph(values, bandwidth=32)
model = oe.arfima({"y": values, "t": list(range(len(values)))}, "y", terms=128, burn=8, time="t")
future = oe.forecast(model, 8)
restored_model = ResultBundle.model_validate_json(model.model_dump_json())
pd.testing.assert_frame_equal(future, oe.forecast(restored_model, 8))
assert model.provenance["stata_parity_validated"] is False
assert all(math.isfinite(c.std_error) and c.std_error > 0 for c in model.coefficients)

def envelope(frame):
    return {"table": json.loads(frame.to_json(orient="table", double_precision=15)), "attrs": frame.attrs}

states = {"difference": envelope(difference),
          "gph": {"tables": {name: envelope(frame) for name, frame in diagnostic.items()},
                  "attrs": diagnostic.attrs},
          "model": model.model_dump(mode="json"), "forecast": envelope(future)}
# Validate JSON, metadata and result restoration before publishing the example.
payload = json.dumps(states, allow_nan=False)
again = json.loads(payload)
reopened = ResultBundle.model_validate(again["model"])
pd.testing.assert_frame_equal(future, oe.forecast(reopened, 8))
table_copy = oe.DataFrame(pd.read_json(StringIO(json.dumps(again["forecast"]["table"])), orient="table"))
table_copy.attrs = again["forecast"]["attrs"]
assert table_copy.attrs["covariance_matrix"] == future.attrs["covariance_matrix"]
assert table_copy.to_latex()

if "display" not in globals():
    display = print
display(difference.iloc[:8])
display(diagnostic["estimate"])
display(model)
display(future)
print("FRACTIONAL_MEMORY_RECEIPT:" + json.dumps({
    "stages": 4, "rows": len(values), "d": next(c.estimate for c in model.coefficients if c.term == "d"),
    "covariance": model.covariance_matrix,
    "forecast_covariance": future.attrs["covariance_matrix"],
    "restored_equal": True, "stata_parity_validated": False,
}, allow_nan=False))
