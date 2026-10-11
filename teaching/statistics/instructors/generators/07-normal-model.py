"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710007


def make_data():
    g = torch.Generator().manual_seed(SEED)
    v = 1000 + 12 * torch.randn(500, generator=g, dtype=torch.float64)
    return pd.DataFrame({"parcel_id": range(1, 501), "grams": v.tolist()})
