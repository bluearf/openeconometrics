"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import pandas as pd
import torch
SEED = 9092026

def make_data():
    generator = torch.Generator().manual_seed(SEED)
    income = 1 + 9 * torch.rand(420, generator=generator, dtype=torch.float64)
    spending = 120 + 45 * income + (12 + 10 * income) * torch.randn(420, generator=generator, dtype=torch.float64)
    return pd.DataFrame({'household': range(1, 421), 'income_thousands': income.tolist(), 'spending': spending.tolist()})
