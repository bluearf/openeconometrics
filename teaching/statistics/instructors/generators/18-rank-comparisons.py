"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710018


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 80
    group = torch.arange(n) % 2
    v = (
        (3.8 + 0.65 * group + 1.4 * torch.randn(n, generator=g, dtype=torch.float64))
        .round()
        .clamp(1, 7)
    )
    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "service": ["A" if x == 0 else "B" for x in group.tolist()],
            "rating": v.tolist(),
        }
    )
