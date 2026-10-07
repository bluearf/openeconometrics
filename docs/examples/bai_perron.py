"""Complete synthetic multiple-break result, suitable for the native console."""
import json
import sys

import openecon as oe
import torch

assert not {"scipy", "statsmodels", "linearmodels", "pyfixest", "arch", "rpy2"}.intersection(sys.modules)

torch.set_num_threads(1)
generator = torch.Generator().manual_seed(218)
n = 45
x = torch.randn(n, dtype=torch.float64, generator=generator)
y = 2 + .7*x + .5*torch.randn(n, dtype=torch.float64, generator=generator)
y[15:30] += 4
data = oe.DataFrame({"t": list(range(n)), "x": x.tolist(), "y": y.tolist()})
result = oe.bai_perron(data, "y", ["x"], time="t", max_breaks=2, replications=199, seed=41)
assert result.attrs["selected_breaks"] == 2
assert result["models"].break_indices.iloc[2] == [15, 30]
assert result.attrs["p_value"] == .005
assert len(result["sample"]) == n
for name, frame in result.items():
    assert not frame.empty
    print(name)
    display(frame)
print("BAI_PERRON_OK " + json.dumps({"frozen": getattr(sys, "frozen", False),
      "selected_breaks": result.attrs["selected_breaks"], "break_indices": [15, 30],
      "p_value": result.attrs["p_value"], "nobs": n,
      "table_rows": {name: len(frame) for name, frame in result.items()},
      "metadata": result.attrs}, sort_keys=True))
