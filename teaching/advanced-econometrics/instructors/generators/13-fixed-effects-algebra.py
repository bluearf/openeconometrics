"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 730013


def make_data():
    g = torch.Generator().manual_seed(SEED)
    G = 30
    T = 6
    n = G * T
    unit = torch.arange(G).repeat_interleave(T)
    time = torch.arange(T).repeat(G)
    a = torch.randn(G, generator=g, dtype=torch.float64)[unit]
    x = 0.8 * a + torch.randn(n, generator=g, dtype=torch.float64)
    z = torch.randn(n, generator=g, dtype=torch.float64)
    y = 2 + 1.2 * x + 0.8 * z + a + torch.randn(n, generator=g, dtype=torch.float64)
    return pd.DataFrame(
        {
            "row_id": range(1, n + 1),
            "x": x.tolist(),
            "z": z.tolist(),
            "y": y.tolist(),
            "unit": unit.tolist(),
            "time": time.tolist(),
        }
    )
