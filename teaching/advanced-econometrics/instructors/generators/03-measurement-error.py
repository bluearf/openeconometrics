"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 730003


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 360
    x = torch.randn(n, generator=g, dtype=torch.float64)
    v = torch.randn(n, generator=g, dtype=torch.float64)
    e = torch.randn(n, generator=g, dtype=torch.float64)
    return pd.DataFrame(
        {
            "row_id": range(1, n + 1),
            "x": x.tolist(),
            "observed_x": (x + v).tolist(),
            "y": (2 + 1.5 * x + e).tolist(),
        }
    )
