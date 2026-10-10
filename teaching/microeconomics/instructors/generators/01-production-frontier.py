"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 720001


def make_data():
    x = torch.arange(0, 41, dtype=torch.float64)
    return pd.DataFrame({"x": x.tolist(), "y": (2 * torch.sqrt(1600 - x * x)).tolist()})
