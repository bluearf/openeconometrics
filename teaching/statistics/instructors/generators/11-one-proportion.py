"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710011


def make_data():
    g = torch.Generator().manual_seed(SEED)
    v = (torch.rand(400, generator=g) < 0.38).long()
    return pd.DataFrame({"user_id": range(1, 401), "opt_in": v.tolist()})
