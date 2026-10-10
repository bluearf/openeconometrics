"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
import pandas as pd
import torch
SEED = 17311

def make_data():
    generator = torch.Generator().manual_seed(SEED)
    shocks = torch.randn((240, 2), generator=generator, dtype=torch.float64)
    x = 100 + torch.cumsum(0.35 + shocks[:, 0], 0)
    y = 80 + torch.cumsum(0.25 + shocks[:, 1], 0)
    return pd.DataFrame({'quarter': list(range(1, 241)), 'index_x': x.tolist(), 'index_y': y.tolist()})
