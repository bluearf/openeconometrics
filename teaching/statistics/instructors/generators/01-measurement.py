"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710001


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 180
    size = torch.randint(1, 7, (n,), generator=g)
    spend = (400 + 140 * size + 70 * torch.randn(n, generator=g, dtype=torch.float64)).tolist()
    for i in [8, 38, 68, 98]:
        spend[i] = None
    return pd.DataFrame(
        {
            "household_id": range(1, n + 1),
            "region": ["North" if i % 2 else "South" for i in range(n)],
            "household_size": size.tolist(),
            "monthly_spending": spend,
        }
    )
