"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710005


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 1000
    d = (torch.rand(n, generator=g) < 0.05).long()
    a = (torch.rand(n, generator=g) < (0.08 + 0.82 * d)).long()
    return pd.DataFrame(
        {"component_id": range(1, n + 1), "defect": d.tolist(), "alert": a.tolist()}
    )
