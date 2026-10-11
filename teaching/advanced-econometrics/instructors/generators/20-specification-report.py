"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 730020


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 300
    x = torch.randn(n, generator=g, dtype=torch.float64)
    z = torch.randn(n, generator=g, dtype=torch.float64)
    t = (torch.rand(n, generator=g) < 0.5).double()
    inter = t * x
    y = (
        2
        + 0.7 * x
        + 0.4 * z
        + 1.2 * t
        + 0.5 * inter
        + (1 + 0.2 * x.abs()) * torch.randn(n, generator=g, dtype=torch.float64)
    )
    return pd.DataFrame(
        {
            "row_id": range(1, n + 1),
            "x": x.tolist(),
            "z": z.tolist(),
            "y": y.tolist(),
            "treat": t.tolist(),
            "interaction": inter.tolist(),
        }
    )
