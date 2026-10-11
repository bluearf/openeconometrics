"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import pandas as pd
import torch
SEED = 14142026

def make_data():
    generator = torch.Generator().manual_seed(SEED)
    n = 600
    ability = torch.randn(n, generator=generator, dtype=torch.float64)
    baseline = 30 + 5 * ability + 8 * torch.randn(n, generator=generator, dtype=torch.float64)
    uniform = torch.rand(n, generator=generator, dtype=torch.float64)
    take0 = (uniform < torch.sigmoid(-2 + 0.5 * ability)).double()
    take1 = (uniform < torch.sigmoid(1 + 0.5 * ability)).double()
    assignment = torch.zeros(n, dtype=torch.float64)
    assignment[torch.randperm(n, generator=generator)[:n // 2]] = 1
    participation = assignment * take1 + (1 - assignment) * take0
    untreated = 100 + 1.5 * baseline + 10 * ability + 20 * torch.randn(n, generator=generator, dtype=torch.float64)
    earnings = untreated + 30 * participation
    return pd.DataFrame({'applicant': range(n), 'offer': assignment.tolist(), 'baseline_score': baseline.tolist(), 'participated': participation.tolist(), 'earnings': earnings.tolist(), 'take_if_no_offer': take0.tolist(), 'take_if_offer': take1.tolist()})
