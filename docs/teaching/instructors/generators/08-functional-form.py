"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import torch
import openecon as oe
SEED = 8082026

def make_data():
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    education = 10 + 8 * torch.rand(500, generator=generator, dtype=torch.float64)
    experience = 40 * torch.rand(500, generator=generator, dtype=torch.float64)
    sigma = 0.16 + 0.004 * experience
    log_wage = 1.6 + 0.07 * education + 0.05 * experience - 0.0008 * experience ** 2 + sigma * torch.randn(500, generator=generator, dtype=torch.float64)
    return oe.DataFrame({'education': education.tolist(), 'experience': experience.tolist(), 'experience_squared': experience.square().tolist(), 'log_wage': log_wage.tolist(), 'hourly_wage': log_wage.exp().tolist()})
