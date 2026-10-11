"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 720009


def make_data():
    labor = torch.arange(1, 101, dtype=torch.float64)
    return pd.DataFrame({"labor": labor.tolist(), "output": (10 * torch.sqrt(labor)).tolist()})
