"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 730007


def make_data():
    g = torch.Generator().manual_seed(SEED)
    G = 40
    T = 6
    n = G * T
    group = torch.arange(G).repeat_interleave(T)
    x = torch.randn(n, generator=g, dtype=torch.float64)
    z = torch.randn(n, generator=g, dtype=torch.float64)
    u = torch.randn(G, generator=g, dtype=torch.float64)[group]
    y = 2 + 1.2 * x + 0.8 * z + u + torch.randn(n, generator=g, dtype=torch.float64)
    return pd.DataFrame(
        {
            "row_id": range(1, n + 1),
            "x": x.tolist(),
            "z": z.tolist(),
            "y": y.tolist(),
            "group": group.tolist(),
        }
    )
