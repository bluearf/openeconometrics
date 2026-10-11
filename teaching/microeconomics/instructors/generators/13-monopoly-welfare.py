"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 720013


def make_data():
    q = torch.arange(0, 101, dtype=torch.float64)
    return pd.DataFrame(
        {
            "quantity": q.tolist(),
            "price": (100 - q).tolist(),
            "marginal_cost": 20.0,
            "fixed_cost": 200.0,
        }
    )
