"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710015


def make_data():
    g = torch.Generator().manual_seed(SEED)
    x = 2 + 10 * torch.rand(160, generator=g, dtype=torch.float64)
    y = (48 + 3 * x + 9 * torch.randn(160, generator=g, dtype=torch.float64)).tolist()
    y[20] = None
    y[80] = None
    return pd.DataFrame({"student_id": range(1, 161), "hours": x.tolist(), "score": y})
