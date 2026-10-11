"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import pandas as pd
import torch
SEED = 11112026
FIRMS = 60
PERIODS = 6

def make_data():
    generator = torch.Generator().manual_seed(SEED)
    ability = 1.5 * torch.randn(FIRMS, generator=generator, dtype=torch.float64)
    shocks = torch.zeros((FIRMS, PERIODS), dtype=torch.float64)
    records = []
    for t in range(PERIODS):
        new = 2 * torch.randn(FIRMS, generator=generator, dtype=torch.float64)
        shocks[:, t] = new if t == 0 else 0.55 * shocks[:, t - 1] + new
        investment = 6 + 1.2 * ability + 0.5 * t + torch.randn(FIRMS, generator=generator, dtype=torch.float64)
        output = 40 + 2 * investment + 4 * ability + 0.8 * t + shocks[:, t]
        for i in range(FIRMS):
            records.append({'firm': i, 'year': 2015 + t, 'trend': t, 'investment': float(investment[i]), 'productivity': float(output[i])})
    return pd.DataFrame(records).sort_values(['firm', 'year']).reset_index(drop=True)
