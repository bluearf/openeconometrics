"""Lab 13's independent exactly identified calculation retains full precision."""

from pathlib import Path
import runpy

import mpmath as mp
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "docs/teaching/labs/13-instrumental-variables"


@pytest.mark.parametrize("suffix", ["", "_weak", "_invalid"])
def test_exact_moment_reference_matches_60_digit_coefficients_and_covariance(suffix):
    lab = runpy.run_path(str(LAB / "lab.py"))
    supplied = lab["load_data"](LAB / "schooling_instruments.xlsx")
    data = supplied[["person", "offer", "background", "schooling", "log_earnings"]].copy()
    if suffix:
        data["schooling"] = supplied["schooling" + suffix]
        data["log_earnings"] = supplied["log_earnings" + suffix]
    x = [[1.0, float(row.background), float(row.schooling)] for row in data.itertuples()]
    z = [[1.0, float(row.background), float(row.offer)] for row in data.itertuples()]
    with mp.workdps(60):
        xx, zz, yy = mp.matrix(x), mp.matrix(z), mp.matrix(data.log_earnings.tolist())
        inverse = (zz.T * xx) ** -1
        beta = inverse * zz.T * yy
        residual = yy - xx * beta
        scores = mp.matrix([[zz[i, j] * residual[i] for j in range(3)] for i in range(len(data))])
        covariance = mp.mpf(len(data)) / (len(data) - 3) * inverse * scores.T * scores * inverse.T
        expected_beta = torch.tensor(list(beta), dtype=torch.float64)
        expected_covariance = torch.tensor(covariance.tolist(), dtype=torch.float64)
    manual_beta, manual_covariance = lab["manual_iv"](data)
    torch.testing.assert_close(manual_beta, expected_beta, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(manual_covariance, expected_covariance, rtol=1e-8, atol=1e-8)
    model = lab["oe"].ivregress(data=data, y="log_earnings", x=["background"],
                              endog=["schooling"], instruments=["offer"],
                              covariance="robust", small=True, missing="raise")
    torch.testing.assert_close(torch.tensor([row.estimate for row in model.coefficients], dtype=torch.float64),
                               expected_beta, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(torch.tensor(model.covariance_matrix, dtype=torch.float64),
                               expected_covariance, rtol=1e-8, atol=1e-8)
