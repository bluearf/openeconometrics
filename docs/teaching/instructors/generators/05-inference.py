"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import torch
import openecon as oe
SEED = 5052026

def make_data():
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    education = 10 + 8 * torch.rand(140, generator=generator, dtype=torch.float64)
    noise = (3 + 0.3 * (education - 10)) * torch.randn(140, generator=generator, dtype=torch.float64)
    return oe.DataFrame({'education': education.tolist(), 'hourly_wage': (25 + 0.6 * (education - 14) + noise).tolist()})
