"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710019


def make_data():
    g = torch.Generator().manual_seed(SEED)
    v = torch.exp(3 + 0.8 * torch.randn(90, generator=g, dtype=torch.float64))
    return pd.DataFrame({"basket_id": range(1, 91), "spending": v.tolist()})
