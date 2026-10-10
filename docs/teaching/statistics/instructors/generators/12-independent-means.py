"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710012


def make_data():
    g = torch.Generator().manual_seed(SEED)
    a = 34 + 4 * torch.randn(65, generator=g, dtype=torch.float64)
    b = 38 + 9 * torch.randn(85, generator=g, dtype=torch.float64)
    return pd.DataFrame(
        {
            "delivery_id": range(1, 151),
            "method": ["A"] * 65 + ["B"] * 85,
            "minutes": torch.cat([a, b]).tolist(),
        }
    )
