"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 720018


def make_data():
    q = torch.arange(0, 41, dtype=torch.float64)
    return pd.DataFrame(
        {
            "quantity": q.tolist(),
            "mb_a": (40 - q).tolist(),
            "mb_b": (30 - 0.5 * q).tolist(),
            "mc": 25.0,
        }
    )
