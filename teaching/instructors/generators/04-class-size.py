"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
from __future__ import annotations
import torch
import openecon as oe
SEED = 4204
SCHOOLS = 240
TRUE_SLOPE = -1.8

def make_data(seed=SEED, schools=SCHOOLS):
    """Generate independent schools with an explicitly known teaching DGP."""
    if schools < 30:
        raise ValueError('Use at least 30 schools for this teaching example.')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    class_size = 18.0 + 12.0 * torch.rand(schools, generator=generator, dtype=torch.float64)
    noise_sd = 3.0 + 0.8 * (class_size - 18.0)
    error = noise_sd * torch.randn(schools, generator=generator, dtype=torch.float64)
    score = 660.0 + TRUE_SLOPE * (class_size - 24.0) + error
    return oe.DataFrame({'school_id': list(range(1, schools + 1)), 'class_size': class_size.tolist(), 'test_score': score.tolist()})
