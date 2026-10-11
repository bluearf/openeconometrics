"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710013


def make_data():
    g = torch.Generator().manual_seed(SEED)
    b = 30 + 6 * torch.randn(60, generator=g, dtype=torch.float64)
    a = (b + 2.2 + 2.5 * torch.randn(60, generator=g, dtype=torch.float64)).tolist()
    a[8] = None
    return pd.DataFrame({"worker_id": range(1, 61), "before": b.tolist(), "after": a})
