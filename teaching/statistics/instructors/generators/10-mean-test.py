"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710010


def make_data():
    g = torch.Generator().manual_seed(SEED)
    v = 500.6 + 1.8 * torch.randn(80, generator=g, dtype=torch.float64)
    return pd.DataFrame({"container_id": range(1, 81), "milliliters": v.tolist()})
