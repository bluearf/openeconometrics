"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710002


def make_data():
    g = torch.Generator().manual_seed(SEED)
    v = torch.exp(3.2 + 0.75 * torch.randn(240, generator=g, dtype=torch.float64))
    v[0] = 320
    return pd.DataFrame({"basket_id": range(1, 241), "spending": v.tolist()})
