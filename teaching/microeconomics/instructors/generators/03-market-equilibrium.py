"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 720003


def make_data():
    p = torch.arange(10, 61, dtype=torch.float64)
    return pd.DataFrame(
        {"price": p.tolist(), "demand": (120 - 2 * p).tolist(), "supply": (-20 + 2 * p).tolist()}
    )
