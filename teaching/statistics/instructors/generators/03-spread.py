"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710003


def make_data():
    g = torch.Generator().manual_seed(SEED)
    v = 35 + 7 * torch.randn(160, generator=g, dtype=torch.float64)
    return pd.DataFrame({"delivery_id": range(1, 161), "minutes": v.tolist()})
