"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import torch
import openecon as oe
SEED = 2022026

def make_data():
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    wage = torch.exp(2.8 + 0.5 * torch.randn(800, generator=generator, dtype=torch.float64))
    wage[-1] *= 12
    values = wage.tolist()
    for index in range(0, 800, 160):
        values[index] = None
    return oe.DataFrame({'worker_id': range(1, 801), 'hourly_wage': values})
