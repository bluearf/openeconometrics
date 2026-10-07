import pytest
import torch

from openecon.econometrics.arima.diagnostics import archlm_test, jarque_bera_test, ljung_box_test
from openecon.econometrics.streaming_residual_diagnostics import diagnostics


@pytest.mark.parametrize("block", [1, 3, 71, 809])
def test_exact_all_rows_and_cross_block_lags(block):
    values = torch.randn(997, generator=torch.Generator().manual_seed(718), dtype=torch.float64)+3.4
    actual = diagnostics(lambda: (values[start:start+block] for start in range(0, len(values), block)), len(values), 17, fitted_parameters=3)
    expected = {"ljung_box": ljung_box_test(values, 17, fitted_parameters=3, label="Ljung-Box Q(17) test of the residuals"),
                "jarque_bera": jarque_bera_test(values, label="Jarque-Bera normality test of the residuals"),
                "arch_lm": archlm_test(values, 1, label="ARCH-LM(1) test of the residuals")}
    for name, record in expected.items():
        for key, value in record.items():
            assert actual[name][key] == (pytest.approx(value, rel=2e-10, abs=2e-11) if isinstance(value, (float, int)) else value)
