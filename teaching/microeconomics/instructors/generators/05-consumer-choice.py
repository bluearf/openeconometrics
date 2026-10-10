"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 720005


def make_data():
    x = torch.linspace(0.5, 39.5, 79, dtype=torch.float64)
    return pd.DataFrame(
        {
            "x": x.tolist(),
            "y": ((120 - 3 * x) / 2).tolist(),
            "income": 120.0,
            "px": 3.0,
            "py": 2.0,
            "alpha": 0.4,
        }
    )
