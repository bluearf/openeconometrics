"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 730010


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 240
    x = torch.randn(n, generator=g, dtype=torch.float64)
    z = 0.65 * x + torch.randn(n, generator=g, dtype=torch.float64)
    e = torch.randn(n, generator=g, dtype=torch.float64)
    y = 2 + 1.2 * x + 0.8 * z + (1 + 0.3 * x.abs()) * e
    return pd.DataFrame(
        {"row_id": range(1, n + 1), "x": x.tolist(), "z": z.tolist(), "y": y.tolist()}
    )
