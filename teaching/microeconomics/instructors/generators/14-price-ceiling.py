"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 720014


def make_data():
    q = torch.arange(0, 101, dtype=torch.float64)
    return pd.DataFrame(
        {
            "quantity": q.tolist(),
            "demand_price": (60 - 0.5 * q).tolist(),
            "supply_price": (10 + 0.5 * q).tolist(),
        }
    )
