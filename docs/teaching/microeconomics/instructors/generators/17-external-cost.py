"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 720017


def make_data():
    q = torch.arange(0, 101, dtype=torch.float64)
    return pd.DataFrame(
        {
            "quantity": q.tolist(),
            "demand_price": (60 - 0.5 * q).tolist(),
            "private_mc": (10 + 0.5 * q).tolist(),
            "external_mc": 10.0,
        }
    )
