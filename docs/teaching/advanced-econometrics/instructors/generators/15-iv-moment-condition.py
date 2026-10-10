"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 730015


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 320
    z = torch.randn(n, generator=g, dtype=torch.float64)
    c = torch.randn(n, generator=g, dtype=torch.float64)
    v = torch.randn(n, generator=g, dtype=torch.float64)
    u = 0.7 * v + torch.randn(n, generator=g, dtype=torch.float64)
    x = 0.9 * z + 0.4 * c + v
    y = 2 + 1.2 * x + 0.6 * c + u
    return pd.DataFrame(
        {
            "row_id": range(1, n + 1),
            "x": x.tolist(),
            "z": z.tolist(),
            "control": c.tolist(),
            "y": y.tolist(),
        }
    )
