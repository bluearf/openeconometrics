"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 730006


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 240
    x = torch.randn(n, generator=g, dtype=torch.float64)
    z = torch.randn(n, generator=g, dtype=torch.float64)
    v = (1 + x.abs()) ** 2
    y = 2 + 1.2 * x + 0.8 * z + v.sqrt() * torch.randn(n, generator=g, dtype=torch.float64)
    return pd.DataFrame(
        {
            "row_id": range(1, n + 1),
            "x": x.tolist(),
            "z": z.tolist(),
            "y": y.tolist(),
            "variance": v.tolist(),
            "weight": (1 / v).tolist(),
        }
    )
