"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 730008


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 240
    x = torch.randn(n, generator=g, dtype=torch.float64)
    z = torch.randn(n, generator=g, dtype=torch.float64)
    e = torch.randn(n, generator=g, dtype=torch.float64)
    u = e.clone()
    for t in range(1, n):
        u[t] = 0.6 * u[t - 1] + e[t]
    return pd.DataFrame(
        {
            "row_id": range(1, n + 1),
            "x": x.tolist(),
            "z": z.tolist(),
            "y": (2 + 1.2 * x + 0.8 * z + u).tolist(),
            "time": range(n),
        }
    )
