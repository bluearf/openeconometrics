"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710016


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 360
    c = torch.arange(n) % 3
    p = torch.tensor([0.25, 0.4, 0.6])[c]
    m = (torch.rand(n, generator=g) < p).long()
    return pd.DataFrame(
        {
            "transaction_id": range(1, n + 1),
            "channel": [["Store", "Web", "App"][x] for x in c.tolist()],
            "member": m.tolist(),
        }
    )
