"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710009


def make_data():
    g = torch.Generator().manual_seed(SEED)
    v = (10.5 + 2 * torch.randn(72, generator=g, dtype=torch.float64)).tolist()
    v[10] = None
    v[40] = None
    return pd.DataFrame({"encounter_id": range(1, 73), "minutes": v})
