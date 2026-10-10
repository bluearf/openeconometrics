"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710014


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 600
    group = torch.arange(n) % 2
    p = 0.22 + 0.07 * group
    v = (torch.rand(n, generator=g) < p).long()
    return pd.DataFrame(
        {
            "recipient_id": range(1, n + 1),
            "message": ["A" if x == 0 else "B" for x in group.tolist()],
            "response": v.tolist(),
        }
    )
