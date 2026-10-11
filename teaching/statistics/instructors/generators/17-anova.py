"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710017


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 135
    group = torch.arange(n) % 3
    v = 65 + 3 * group + 5 * torch.randn(n, generator=g, dtype=torch.float64)
    return pd.DataFrame(
        {
            "trainee_id": range(1, n + 1),
            "format": [["A", "B", "C"][x] for x in group.tolist()],
            "score": v.tolist(),
        }
    )
