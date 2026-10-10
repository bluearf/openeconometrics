"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 730019


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 260
    x = torch.randn(n, generator=g, dtype=torch.float64)
    e = torch.randn(n, generator=g, dtype=torch.float64)
    y = torch.zeros(n, dtype=torch.float64)
    for t in range(n):
        y[t] = (0.5 * y[t - 1] if t else 0) + 0.8 * x[t] + e[t]
    z = torch.cat([torch.zeros(1, dtype=torch.float64), y[:-1]])
    return pd.DataFrame(
        {
            "row_id": range(1, n + 1),
            "x": x.tolist(),
            "z": z.tolist(),
            "y": y.tolist(),
            "time": range(n),
        }
    )
