"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
import pandas as pd
import torch
SEED = 19311
TRAIN_N = 160
HORIZON = 20

def make_data():
    generator = torch.Generator().manual_seed(SEED)
    errors = 0.45 * torch.randn(380, generator=generator, dtype=torch.float64)
    deviations = torch.zeros(380, dtype=torch.float64)
    for t in range(1, 380):
        deviations[t] = 0.75 * deviations[t - 1] + errors[t]
    return pd.DataFrame({'quarter': list(range(1, 181)), 'inflation': (2.5 + deviations[200:]).tolist()})
