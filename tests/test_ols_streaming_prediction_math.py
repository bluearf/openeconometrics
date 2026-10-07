"""Stable streamed prediction and arbitrary coefficient-functional oracles."""
from __future__ import annotations

import pandas as pd
import pytest
import torch

from openecon.dataset import Dataset
from openecon.linear_ols.streaming import fit_streaming
from openecon.models import ModelSpec


def _fixture(n=93):
    generator = torch.Generator().manual_seed(612)
    x = torch.randn(n, 2, generator=generator, dtype=torch.float64)
    y = 2 + x[:, 0] - .5 * x[:, 1] + torch.randn(n, generator=generator, dtype=torch.float64)
    return pd.DataFrame({"x": x[:, 0].tolist(), "z": x[:, 1].tolist(), "y": y.tolist(),
                         "group": [i % 11 for i in range(n)], "time": list(range(n)),
                         "weight": [1 + i % 3 for i in range(n)]})


def _fit(frame, covariance="HC3", *, intercept=True, **options):
    source = Dataset.from_batches(lambda: (frame.iloc[start:start + 7]
                                          for start in range(0, len(frame), 7)),
                                  list(frame), row_count=len(frame))
    spec = ModelSpec(outcome="y", predictors=["x", "z"], covariance=covariance,
                     intercept=intercept, **options)
    return fit_streaming(spec, source)


@pytest.mark.parametrize("covariance", ["nonrobust", "HC3"])
@pytest.mark.parametrize("intercept", [False, True])
def test_arbitrary_functionals_are_linear_and_match_coefficient_state(covariance, intercept):
    fit = _fit(_fixture(), covariance, intercept=intercept)
    rows = torch.tensor([[0., 1., -.5], [2., .25, 1.], [1., -.75, 0.], [-.5, 2., .25]],
                        dtype=torch.float64)
    if not intercept:
        rows = rows[:, 1:]
    state = fit["state"]
    expected = rows @ fit["covariance"] @ rows.T
    torch.testing.assert_close(state["contrast_covariance"](rows), expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(state["prediction_variance"](rows), expected.diagonal(), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(state["prediction_leverage"](rows),
                               (rows @ fit["bread"] * rows).sum(dim=1), rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(state["prediction_fitted"](rows), rows @ fit["params"], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(state["contrast_covariance"](-2 * rows), 4 * expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(state["prediction_fitted"](-2 * rows), -2 * (rows @ fit["params"]),
                               rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("covariance", ["nonrobust", "HC2", "HC3", "cluster", "cluster_hc3",
                                        "hac", "jackknife", "bootstrap"])
def test_large_translation_preserves_all_arbitrary_functional_covariances(covariance):
    frame = _fixture()
    shifts = torch.tensor([0., 1e8, -2e8], dtype=torch.float64)
    shifted = frame.assign(x=frame.x + float(shifts[1]), z=frame.z + float(shifts[2]))
    # Account for the actual float64 rounding of the physical shifted data.
    reference = shifted.assign(x=shifted.x - float(shifts[1]), z=shifted.z - float(shifts[2]))
    options = {}
    if covariance.startswith("cluster"):
        options["cluster"] = "group"
    elif covariance == "hac":
        options.update(time="time", options={"lags": 3, "kernel": "bartlett"})
    elif covariance == "bootstrap":
        options.update(weights="weight", weight_type="fweight", options={"reps": 7, "seed": 72})
    actual, expected = _fit(shifted, covariance, **options), _fit(reference, covariance, **options)
    rows = torch.tensor([[0., 1., -.5], [2., .25, 1.], [1., -.75, 0.], [-.5, 2., .25]],
                        dtype=torch.float64)
    physical_rows = rows + rows[:, :1] * shifts
    for name in ("contrast_covariance", "prediction_variance", "prediction_leverage", "prediction_fitted"):
        torch.testing.assert_close(actual["state"][name](physical_rows), expected["state"][name](rows),
                                   rtol=1e-10, atol=1e-11)
    # These are genuine prediction rows (constant coefficient one), and retain
    # the previously correct centered prediction behavior as well.
    encoded_actual = actual["design"].encode(shifted)[:, actual["kept_indices"]]
    encoded_reference = expected["design"].encode(reference)[:, expected["kept_indices"]]
    for name in ("prediction_variance", "prediction_leverage", "prediction_fitted"):
        torch.testing.assert_close(actual["state"][name](encoded_actual),
                                   expected["state"][name](encoded_reference), rtol=1e-10, atol=1e-11)


def test_clipped_multiway_contrast_covariance_preserves_raw_projection():
    generator = torch.Generator().manual_seed(0)
    n = 37
    x = torch.randn(n, 2, generator=generator, dtype=torch.float64)
    frame = pd.DataFrame({"x": x[:, 0].tolist(), "z": x[:, 1].tolist(),
                          "y": (1 + x[:, 0] + torch.randn(n, generator=generator, dtype=torch.float64)).tolist(),
                          "g": [i % 3 for i in range(n)], "h": [(i // 3) % 3 for i in range(n)]})
    fit = _fit(frame, "cluster", cluster=["g", "h"])
    assert fit["inference"]["psd_projection"] == "raw coefficient coordinates; not basis invariant"
    rows = torch.tensor([[0., 1., -.5], [2., .25, 1.], [1., -.75, 0.]], dtype=torch.float64)
    expected = rows @ fit["covariance"] @ rows.T
    observed = fit["state"]["contrast_covariance"](rows)
    torch.testing.assert_close(observed, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(fit["state"]["prediction_variance"](rows), expected.diagonal(),
                               rtol=1e-10, atol=1e-12)
    assert float(torch.linalg.eigvalsh(observed).min()) >= -1e-12


def test_stable_contrast_basis_preserves_independent_large_unit_restrictions():
    frame = _fixture()
    shifted = frame.assign(x=frame.x + 1e8)
    fit = _fit(shifted)
    rows = torch.tensor([[1., 1e8, 0.], [0., 1., 0.]], dtype=torch.float64)
    # The first restriction estimates the mean at the large x offset and the
    # second estimates its slope. They are independent despite their units.
    assert int(torch.linalg.matrix_rank(rows)) == 1
    assert int(torch.linalg.matrix_rank(fit["state"]["contrast_basis"](rows))) == 2
