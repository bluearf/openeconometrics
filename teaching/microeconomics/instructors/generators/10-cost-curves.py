"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 720010


def make_data():
    q = torch.arange(1, 101, dtype=torch.float64)
    return pd.DataFrame(
        {"quantity": q.tolist(), "total_cost": (100 + 2 * q + 0.1 * q * q).tolist()}
    )
