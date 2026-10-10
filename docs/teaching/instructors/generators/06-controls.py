"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import openecon as oe
import torch
SEED = 6112026
GENERATED_ROWS = 600
TRUE_EDUCATION_SLOPE = 0.08
TRUE_EXPERIENCE_SLOPE = 0.03

def make_data():
    """Generate original independent workers with correlated regressors."""
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    education = torch.randint(10, 19, (GENERATED_ROWS,), generator=generator).double()
    experience = (20 - 1.3 * (education - 14) + 6 * torch.randn(GENERATED_ROWS, generator=generator, dtype=torch.float64)).clamp(2, 40)
    noise = torch.randn(GENERATED_ROWS, generator=generator, dtype=torch.float64)
    sigma = 0.12 + 0.006 * experience
    log_earnings = 1.2 + TRUE_EDUCATION_SLOPE * education + TRUE_EXPERIENCE_SLOPE * experience + sigma * noise
    reported_experience = experience.clone()
    reported_experience[::50] = float('nan')
    return oe.DataFrame({'worker_id': list(range(1, GENERATED_ROWS + 1)), 'education': education.tolist(), 'experience': reported_experience.tolist(), 'log_earnings': log_earnings.tolist(), 'hourly_earnings': torch.exp(log_earnings).tolist()})
