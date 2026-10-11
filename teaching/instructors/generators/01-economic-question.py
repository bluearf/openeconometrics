"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import torch
import openecon as oe
SEED = 1012026

def make_data():
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    skill = torch.cat((torch.zeros(300), torch.ones(300))).to(torch.float64)
    probability = torch.where(skill == 0, 0.8, 0.2)
    trained = (torch.rand(600, generator=generator, dtype=torch.float64) < probability).to(torch.float64)
    noise = 2 * torch.randn(600, generator=generator, dtype=torch.float64)
    return oe.DataFrame({'worker_id': range(1, 601), 'prior_skill': skill.tolist(), 'trained': trained.tolist(), 'hourly_wage': (12 + 18 * skill + 4 * trained + noise).tolist()})
