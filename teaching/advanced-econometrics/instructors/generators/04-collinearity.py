"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 730004


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 240
    x = torch.randn(n, generator=g, dtype=torch.float64)
    z = x + 0.03 * torch.randn(n, generator=g, dtype=torch.float64)
    y = 2 + x + z + torch.randn(n, generator=g, dtype=torch.float64)
    return pd.DataFrame(
        {"row_id": range(1, n + 1), "x": x.tolist(), "z": z.tolist(), "y": y.tolist()}
    )
