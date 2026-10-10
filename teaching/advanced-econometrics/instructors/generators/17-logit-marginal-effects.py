"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 730017


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 400
    x = torch.randn(n, generator=g, dtype=torch.float64)
    z = torch.randn(n, generator=g, dtype=torch.float64)
    p = torch.sigmoid(-0.4 + 0.8 * x - 0.5 * z)
    y = (torch.rand(n, generator=g, dtype=torch.float64) < p).double()
    return pd.DataFrame(
        {"row_id": range(1, n + 1), "x": x.tolist(), "z": z.tolist(), "y": y.tolist()}
    )
