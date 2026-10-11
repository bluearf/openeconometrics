"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
import pandas as pd
import torch
SEED = 18311
LAGS = 4

def make_data():
    generator = torch.Generator().manual_seed(SEED)
    innovations = torch.randn((300, 2), generator=generator, dtype=torch.float64)
    x, u = (torch.zeros(300, dtype=torch.float64), torch.zeros(300, dtype=torch.float64))
    for t in range(1, 300):
        x[t] = 0.85 * x[t - 1] + 0.5 * innovations[t, 0]
        u[t] = 0.75 * u[t - 1] + innovations[t, 1]
    x, u = (x[100:] + 5, u[100:])
    full = pd.DataFrame({'quarter': list(range(1, 201)), 'advertising': x.tolist(), 'sales': (10 + 1.4 * x + u).tolist()})
    return full.loc[~full.quarter.isin([40, 41, 95])].reset_index(drop=True)
