"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
import pandas as pd
import torch
SEED = 20311
TRUE_EFFECT = 1.2

def make_data():
    generator = torch.Generator().manual_seed(SEED)
    noise = torch.randn((300, 5), generator=generator, dtype=torch.float64)
    prior = 60 + 8 * noise[:, 0]
    resources, motivation = (noise[:, 1], noise[:, 2])
    tutoring = (5 + 0.4 * resources - 0.04 * (prior - 60) + 0.8 * motivation + 1.2 * noise[:, 3]).clamp_min(0.1)
    score = 20 + 0.65 * prior + TRUE_EFFECT * tutoring + 2 * resources + 2 * motivation + (3 + 0.2 * tutoring) * noise[:, 4]
    frame = pd.DataFrame({'school': list(range(1, 301)), 'prior_score': prior.tolist(), 'resource_index': resources.tolist(), 'tutoring_hours': tutoring.tolist(), 'final_score': score.tolist()})
    frame.loc[frame.school % 13 == 0, 'resource_index'] = float('nan')
    return frame
