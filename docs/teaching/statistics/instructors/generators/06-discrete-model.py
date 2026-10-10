"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710006


def make_data():
    g = torch.Generator().manual_seed(SEED)
    v = (torch.rand((500, 10), generator=g) < 0.3).sum(1)
    return pd.DataFrame({"batch_id": range(1, 501), "orders": v.tolist()})
