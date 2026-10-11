"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import pandas as pd
import torch
SEED = 12122026

def make_data():
    generator = torch.Generator().manual_seed(SEED)
    n = 720
    education = torch.randint(10, 19, (n,), generator=generator).double()
    children = torch.randint(0, 4, (n,), generator=generator).double()
    age = 20 + 40 * torch.rand(n, generator=generator, dtype=torch.float64)
    index = -0.6 + 0.22 * (education - 12) - 0.45 * children + 0.025 * (age - 35)
    participating = (torch.rand(n, generator=generator, dtype=torch.float64) < torch.sigmoid(index)).double()
    return pd.DataFrame({'person': range(n), 'education': education.tolist(), 'children': children.tolist(), 'age': age.tolist(), 'participating': participating.tolist()})
