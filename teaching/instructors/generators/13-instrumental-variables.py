"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import pandas as pd
import torch
SEED = 13132026

def make_data(relevance=1.6, direct_effect=0.0):
    generator = torch.Generator().manual_seed(SEED)
    n = 700
    offer = (torch.rand(n, generator=generator, dtype=torch.float64) < 0.5).double()
    background = torch.randn(n, generator=generator, dtype=torch.float64)
    ability = torch.randn(n, generator=generator, dtype=torch.float64)
    schooling = 12 + relevance * offer + 0.4 * background + 0.8 * ability + torch.randn(n, generator=generator, dtype=torch.float64)
    outcome = 1 + 0.08 * schooling + 0.06 * background + 0.25 * ability + 0.2 * torch.randn(n, generator=generator, dtype=torch.float64) + direct_effect * offer
    return pd.DataFrame({'person': range(n), 'offer': offer.tolist(), 'background': background.tolist(), 'schooling': schooling.tolist(), 'log_earnings': outcome.tolist()})
