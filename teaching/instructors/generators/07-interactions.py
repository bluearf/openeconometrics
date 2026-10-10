"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import torch
import openecon as oe
SEED = 7072026

def make_data():
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    experience = 5 + 25 * torch.rand(480, generator=generator, dtype=torch.float64)
    sector = torch.cat((torch.zeros(240), torch.ones(240))).to(torch.float64)
    noise = (3 + 0.06 * experience) * torch.randn(480, generator=generator, dtype=torch.float64)
    return oe.DataFrame({'experience': experience.tolist(), 'sector': sector.tolist(), 'experience_centered': (experience - 15).tolist(), 'sector_experience': (sector * (experience - 15)).tolist(), 'sector_experience_raw': (sector * experience).tolist(), 'hourly_wage': (14 + 0.8 * experience + 3 * sector + 0.45 * sector * experience + noise).tolist()})
