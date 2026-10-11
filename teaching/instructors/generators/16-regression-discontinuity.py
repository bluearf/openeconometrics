"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
import pandas as pd
import torch
SEED = 16311
CUTOFF = 50.0
TRUE_EFFECT = 3.0

def make_data():
    generator = torch.Generator().manual_seed(SEED)
    score = 20 + 60 * torch.rand(600, generator=generator, dtype=torch.float64)
    centered = score - CUTOFF
    awarded = (score >= CUTOFF).to(torch.float64)
    credits = 24 + 0.2 * centered + 0.006 * centered.square() + TRUE_EFFECT * awarded + 2.5 * torch.randn(600, generator=generator, dtype=torch.float64)
    return pd.DataFrame({'student': list(range(600)), 'score': score.tolist(), 'awarded': awarded.to(torch.int64).tolist(), 'credits': credits.tolist()})
