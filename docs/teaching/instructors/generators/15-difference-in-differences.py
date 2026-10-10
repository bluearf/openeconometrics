"""Instructor-only synthetic input generator. Student analyses use the supplied Excel file."""
import pandas as pd
import torch
SEED = 1515311
POLICY_YEAR = 2018
TRUE_EFFECT = 3.0
CONCURRENT_SHOCK = 2.0
N_REGIONS = 60
YEARS = list(range(2015, 2021))

def make_data():
    """Return 360 region-year observations; outcome units are percent points."""
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    dtype = torch.float64
    region_effect = 4.0 * torch.randn(N_REGIONS, generator=generator, dtype=dtype)
    common_shock = 0.4 * torch.randn(len(YEARS), generator=generator, dtype=dtype)
    regional_error = torch.empty((N_REGIONS, len(YEARS)), dtype=dtype)
    regional_error[:, 0] = 2.0 * torch.randn(N_REGIONS, generator=generator, dtype=dtype)
    for period in range(1, len(YEARS)):
        innovation = 2.0 * (1.0 - 0.65 ** 2) ** 0.5 * torch.randn(N_REGIONS, generator=generator, dtype=dtype)
        regional_error[:, period] = 0.65 * regional_error[:, period - 1] + innovation
    records = []
    for region in range(N_REGIONS):
        treated = int(region < N_REGIONS // 2)
        for period, year in enumerate(YEARS):
            post = int(year >= POLICY_YEAR)
            treated_post = treated * post
            untreated = 60.0 - 6.0 * treated + float(region_effect[region]) + 1.2 * period + float(common_shock[period]) + float(regional_error[region, period])
            records.append({'region': region, 'year': year, 'treated': treated, 'post': post, 'treated_post': treated_post, 'employment_rate': untreated + TRUE_EFFECT * treated_post, 'employment_no_policy': untreated})
    return pd.DataFrame.from_records(records)
