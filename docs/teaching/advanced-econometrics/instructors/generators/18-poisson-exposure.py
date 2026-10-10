"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 730018


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 360
    x = torch.randn(n, generator=g, dtype=torch.float64)
    z = torch.randn(n, generator=g, dtype=torch.float64)
    ex = 0.5 + 2 * torch.rand(n, generator=g, dtype=torch.float64)
    mu = ex * torch.exp(0.3 + 0.4 * x - 0.2 * z)
    y = torch.poisson(mu, generator=g)
    return pd.DataFrame(
        {
            "row_id": range(1, n + 1),
            "x": x.tolist(),
            "z": z.tolist(),
            "y": y.tolist(),
            "exposure": ex.tolist(),
        }
    )
