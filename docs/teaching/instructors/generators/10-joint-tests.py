"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import pandas as pd
import torch
SEED = 10102026

def make_data():
    generator = torch.Generator().manual_seed(SEED)
    n = 380
    common = torch.randn(n, generator=generator, dtype=torch.float64)
    coaching = 10 + 2 * common + 0.3 * torch.randn(n, generator=generator, dtype=torch.float64)
    software = 8 + 2 * common + 0.3 * torch.randn(n, generator=generator, dtype=torch.float64)
    tenure = 2 + 8 * torch.rand(n, generator=generator, dtype=torch.float64)
    productivity = 50 + 1.2 * coaching + 1.2 * software + 0.5 * tenure + 5 * torch.randn(n, generator=generator, dtype=torch.float64)
    return pd.DataFrame({'worker': range(n), 'coaching': coaching.tolist(), 'software': software.tolist(), 'tenure': tenure.tolist(), 'productivity': productivity.tolist()})
