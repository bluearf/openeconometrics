"""Actual Metal precision, rejected preconditioners and public streaming OLS."""
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.dataset import Dataset
from openecon.engines.execution import execution_scope, qr_factor, validate_device


@pytest.mark.parametrize("name", ["not-a-device", "mps:9", "meta", "cuda:999"])
def test_invalid_or_unavailable_device(name):
    with pytest.raises(oe.AnalysisError):
        validate_device(name)


def test_cpu_execution_trace_and_nested_scope():
    block = torch.randn(35, 4, dtype=torch.float64)
    with execution_scope("cpu") as outer:
        with execution_scope("cpu") as inner:
            qr_factor(block)
        factor = qr_factor(block)
    assert outer.metadata()["factor_devices"] == {"cpu": 1}
    assert inner.metadata()["factor_devices"] == {"cpu": 1}
    torch.testing.assert_close(factor.T @ factor, block.T @ block)


def test_automatic_helper_inherits_device_and_whole_run_trace():
    block = torch.randn(20, 3, dtype=torch.float64)
    with execution_scope("cpu") as outer:
        with execution_scope("auto") as inner:
            qr_factor(block)
        qr_factor(block)
    assert inner is outer
    assert outer.metadata()["factor_devices"] == {"cpu": 2}


def test_public_replay_family_preserves_actual_acceleration_trace():
    x = torch.randn(91, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(871))
    frame = pd.DataFrame({"x": x[:, 0].numpy(), "y": (x[:, 0]+x[:, 1]).numpy()})
    result = oe.nl(data=Dataset.from_frame(frame), y="y",
                   formula="{a}+{b}*x", start={"a": 0., "b": 0.})
    assert result.provenance["execution"]["requested_device"] == "auto"
    assert result.provenance["execution"]["reporting_precision"] == "float64"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal unavailable")
@pytest.mark.parametrize("shape", [(100, 3), (12000, 18)])
def test_actual_metal_factor_preserves_original_double_observations(shape):
    block = torch.randn(*shape, dtype=torch.float64, generator=torch.Generator().manual_seed(373))
    with execution_scope("mps") as trace:
        factor = qr_factor(block)
    assert factor.dtype == torch.float64 and factor.device.type == "cpu"
    assert trace.metadata()["factor_devices"] == {"mps": 1}
    torch.testing.assert_close(factor.T @ factor, block.T @ block, rtol=2e-12, atol=2e-11)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal unavailable")
@pytest.mark.parametrize("kind", ["singular", "short", "bad-scale"])
def test_metal_inaccurate_or_singular_factor_uses_original_cpu_qr(kind):
    block = torch.randn(100, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(28))
    if kind == "singular":
        block[:, -1] = block[:, 0]
    elif kind == "short":
        block = block[:2]
    else:
        block[:, 0] *= 1e8
    with execution_scope("mps") as trace:
        factor = qr_factor(block)
    assert trace.metadata()["factor_devices"] == {"cpu": 1}
    assert trace.metadata()["fallbacks"]
    torch.testing.assert_close(factor, torch.linalg.qr(block, mode="r").R, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal unavailable")
@pytest.mark.parametrize("covariance", ["nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster"])
def test_actual_metal_ols_coefficients_and_covariances_match_cpu(covariance):
    x = torch.randn(1500, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    y = x @ torch.tensor([.5, -.2, 1.4], dtype=torch.float64) + torch.cos(x[:, 0])
    frame = pd.DataFrame(x.numpy(), columns=["x", "z", "w"])
    frame["y"], frame["g"] = y.numpy(), [i // 30 for i in range(len(frame))]
    kwargs = dict(y="y", x=["x", "z", "w"], covariance=covariance,
                  cluster="g" if covariance == "cluster" else None)
    source = Dataset.from_frame(frame)
    cpu, metal = oe.ols(data=source, device="cpu", **kwargs), oe.ols(data=source, device="mps", **kwargs)
    assert metal.provenance["reporting_precision"] == "float64"
    assert metal.provenance["factor_devices"]["mps"] > 0
    torch.testing.assert_close(torch.tensor(metal.covariance_matrix, dtype=torch.float64),
                               torch.tensor(cpu.covariance_matrix, dtype=torch.float64), rtol=1e-11, atol=1e-13)
    torch.testing.assert_close(torch.tensor([c.estimate for c in metal.coefficients], dtype=torch.float64),
                               torch.tensor([c.estimate for c in cpu.coefficients], dtype=torch.float64), rtol=1e-11, atol=1e-13)


def test_categorical_parquet_plan_grows_after_discovery(tmp_path):
    x = torch.arange(800, dtype=torch.float64)
    frame = pd.DataFrame({"x": x.numpy(), "y": (x + x.sin()).numpy(), "g": ["a", "b"] * 400})
    path = tmp_path / "categorical.parquet"
    frame.to_parquet(path, index=False)
    fitted = oe.ols(data=oe.scan(path), y="y", x=["x", "g"], categorical=["g"], device="cpu")
    assert fitted.provenance["streaming"]["batch_rows"] > 399
    assert fitted.nobs == len(frame)
