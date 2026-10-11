"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710020


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 240
    group = torch.arange(n) % 2
    v = (18 + 2 * group + 6 * torch.randn(n, generator=g, dtype=torch.float64)).tolist()
    r = (torch.rand(n, generator=g) < (0.3 + 0.06 * group)).long()
    for i in [5, 35, 65, 95, 125, 155]:
        v[i] = None
    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "offer": ["A" if x == 0 else "B" for x in group.tolist()],
            "spending": v,
            "returned": r.tolist(),
        }
    )
