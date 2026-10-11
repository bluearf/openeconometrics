"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 720004


def make_data():
    p = torch.arange(1, 60, dtype=torch.float64)
    return pd.DataFrame({"price": p.tolist(), "quantity": (120 - 2 * p).tolist()})
