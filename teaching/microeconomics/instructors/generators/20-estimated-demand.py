"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 720020


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 180
    lp = 1 + 0.8 * torch.rand(n, generator=g, dtype=torch.float64)
    li = 4 + 0.6 * torch.rand(n, generator=g, dtype=torch.float64)
    lq = 4 - 1.3 * lp + 0.5 * li + 0.15 * torch.randn(n, generator=g, dtype=torch.float64)
    return pd.DataFrame(
        {
            "market_id": range(1, n + 1),
            "price": lp.exp().tolist(),
            "income": li.exp().tolist(),
            "quantity": lq.exp().tolist(),
        }
    )
