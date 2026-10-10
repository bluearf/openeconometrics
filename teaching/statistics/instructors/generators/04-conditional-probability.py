"""Original synthetic observations, for instructor preparation only."""

import torch
import pandas as pd

SEED = 710004


def make_data():
    g = torch.Generator().manual_seed(SEED)
    n = 600
    m = (torch.rand(n, generator=g) < 0.35).long()
    p = (torch.rand(n, generator=g) < (0.25 + 0.35 * m)).long()
    return pd.DataFrame({"visit_id": range(1, n + 1), "member": m.tolist(), "purchase": p.tolist()})
