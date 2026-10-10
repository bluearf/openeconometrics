"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710008


def make_data():
    g = torch.Generator().manual_seed(SEED)
    v = torch.exp(3 + 0.9 * torch.randn(1000, generator=g, dtype=torch.float64))
    return pd.DataFrame({"person_id": range(1, 1001), "spending": v.tolist()})
