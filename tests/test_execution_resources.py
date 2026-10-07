"""Resource arithmetic and early refusal, without allocating large buffers."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import ModelFrame
from openecon.econometrics.mixed.gee_kernels import WorkingCorrelation, gee_workspace_plan
from openecon.econometrics.systems import common as systems
from openecon.econometrics.systems.gmm_kernels import MomentSystem, gmm_workspace_plan
from openecon.econometrics.systems.panelgls import panel_covariance_plan
from openecon.econometrics.systems.xtpcse import pcse_meat
from openecon.models import ModelSpec
from openecon.resources import plan_workspace, tensor_bytes, use_workspace_budget, workspace_budget_bytes


def test_exact_large_dimension_arithmetic_and_zero():
    assert tensor_bytes((10**20, 10**20)) == 8 * 10**40
    assert tensor_bytes((0, 10**20)) == 0
    with pytest.raises(AnalysisError, match="exceeding") as exc:
        plan_workspace("huge", {"matrix": tensor_bytes((10**20, 10**20))})
    assert exc.value.code == "workspace_limit"
    assert exc.value.resource_plan["buffers"]["matrix"] == 8 * 10**40


@pytest.mark.parametrize("value", [True, -1, 1.5, "2"])
def test_invalid_dimensions(value):
    with pytest.raises(AnalysisError):
        tensor_bytes((2, value))


@pytest.mark.parametrize("value", ["0", "-1", "nan", "1.5", "", " 2", "+2"])
def test_invalid_environment_budget(monkeypatch, value):
    monkeypatch.setenv("OPENECON_WORKSPACE_MB", value)
    with pytest.raises(AnalysisError) as exc:
        workspace_budget_bytes()
    assert exc.value.code == "invalid_resource_budget"


def test_budget_override_nesting_exception_and_exact_boundary(monkeypatch):
    monkeypatch.setenv("OPENECON_WORKSPACE_MB", "9")
    assert workspace_budget_bytes() == 9 * 1024**2
    with use_workspace_budget(2):
        assert plan_workspace("at boundary", {"block": 2 * 1024**2}).estimated_bytes == 2 * 1024**2
        with pytest.raises(RuntimeError), use_workspace_budget(3):
            assert workspace_budget_bytes() == 3 * 1024**2
            raise RuntimeError("fixture")
        assert workspace_budget_bytes() == 2 * 1024**2
    assert workspace_budget_bytes() == 9 * 1024**2


def _no_allocation(*args, **kwargs):
    raise AssertionError("Allocation happened before the resource refusal.")


@pytest.mark.parametrize("kind", ["stationary", "nonstationary", "unstructured"])
def test_gee_quadratic_period_guard_before_allocation(monkeypatch, kind):
    layout = SimpleNamespace(codes=range(20), n_panels=2, periods=10**6,
                             size_groups=[(10**6, None), (999_999, None)])
    monkeypatch.setattr(torch, "zeros", _no_allocation)
    with pytest.raises(AnalysisError) as exc:
        WorkingCorrelation(kind, layout)
    assert exc.value.code == "workspace_limit"
    assert "cached_inverse_generations" in exc.value.resource_plan["buffers"]


def test_gee_counts_all_distinct_cached_sizes_not_only_longest():
    layout = SimpleNamespace(codes=range(10), n_panels=3, periods=3,
                             size_groups=[(1, None), (2, None), (3, None)])
    plan = gee_workspace_plan("unstructured", layout, 2)
    assert dict(plan.buffers)["cached_inverse_generations"] == 16 * (1 + 4 + 9)
    assert "not a process-RSS limit" in plan.record()["scope"]


@pytest.mark.parametrize("independent,hetonly", [(False, False)])
def test_pcse_short_panel_group_square_guard_before_allocation(monkeypatch, independent, hetonly):
    layout = SimpleNamespace(codes=range(20), m=100_000, periods=2)
    x, residual = torch.ones((20, 2), dtype=torch.float64), torch.ones(20, dtype=torch.float64)
    monkeypatch.setattr(torch, "zeros", _no_allocation)
    with pytest.raises(AnalysisError) as exc:
        pcse_meat(x, residual, layout, hetonly=hetonly, independent=independent, pairwise=True)
    assert exc.value.code == "workspace_limit"
    assert exc.value.resource_plan["buffers"]["group_covariance_counts_and_factor_scratch"] == 64 * 100_000**2


def test_independent_panel_covariance_has_no_group_square():
    layout = SimpleNamespace(codes=range(20), m=100_000, periods=2)
    plan = panel_covariance_plan(layout, 2, correlated=False)
    assert "group_covariance_counts_and_factor_scratch" not in dict(plan.buffers)


def test_combined_gmm_guards_equation_jacobians():
    with use_workspace_budget(1), pytest.raises(AnalysisError) as exc:
        gmm_workspace_plan(400, 100, [2] * 3, 3)
    assert exc.value.code == "workspace_limit"
    assert exc.value.resource_plan["buffers"]["all_equation_parameter_jacobians"] == 24 * 400 * 100 * 3


def test_direct_moment_system_guard_before_evaluation():
    # No tensor buffer is needed to test the anticipated joint allocation.
    instruments = [SimpleNamespace(shape=(400, 2))] * 3
    with use_workspace_budget(1), pytest.raises(AnalysisError) as exc:
        MomentSystem([None] * 3, [], {}, instruments, None, 400, 100)
    assert exc.value.code == "workspace_limit"


def test_public_gmm_guard_precedes_instrument_build(monkeypatch):
    from openecon.econometrics.systems import gmm as gmm_module

    rng = np.random.default_rng(845)
    data = pd.DataFrame(rng.normal(size=(500, 4)), columns=["x", "y0", "y1", "y2"])
    moments = [f"y{i} - " + " - ".join(f"{{b{i}_{j}}}*x" for j in range(10)) for i in range(3)]
    monkeypatch.setattr(gmm_module, "_instruments", _no_allocation)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as exc:
        oe.gmm(data=data, moments=moments, instruments=["x"])
    assert exc.value.code == "workspace_limit"


def test_public_gee_guard_precedes_design_build(monkeypatch):
    rng = np.random.default_rng(846)
    n = 2 * 200
    data = pd.DataFrame({"id": np.repeat([0, 1], 200), "t": np.tile(np.arange(200), 2),
                         "x": rng.normal(size=n), "y": rng.normal(size=n)})
    monkeypatch.setattr(ModelFrame, "design", _no_allocation)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as exc:
        oe.xtgee(data=data, y="y", x=["x"], panel="id", time="t", corr="unstructured")
    assert exc.value.code == "workspace_limit"


def test_core_input_guard_before_selection(monkeypatch):
    frame = pd.DataFrame({"x": np.arange(20_000.), "y": np.arange(20_000.)})
    spec = ModelSpec(estimator="poisson", outcome="y", predictors=["x"])
    monkeypatch.setattr(pd.DataFrame, "reset_index", _no_allocation)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as exc:
        ModelFrame(spec, frame)
    assert exc.value.code == "workspace_limit"
    assert exc.value.resource_plan["operation"] == "model input selection"


def test_core_categorical_width_guard_before_dummy_allocation(monkeypatch):
    data = pd.DataFrame({"y": np.arange(12.),
                         "c": pd.Categorical(np.arange(12), categories=np.arange(10_000))})
    with use_workspace_budget(2):
        frame = ModelFrame(ModelSpec(estimator="poisson", outcome="y", predictors=["c"], categorical=["c"]), data)
        assert frame.design_width() == 10_000
        monkeypatch.setattr(torch, "zeros", _no_allocation)
        with pytest.raises(AnalysisError) as exc:
            frame.design()
        assert exc.value.code == "workspace_limit"
        assert exc.value.resource_plan["operation"] == "dense encoded design"


def test_combined_sur_plan_precedes_column_and_qr_allocation(monkeypatch):
    rng = np.random.default_rng(84)
    data = pd.DataFrame(rng.normal(size=(2000, 9)), columns=[f"x{i}" for i in range(9)])
    equations = [{"y": "x7", "x": [f"x{i}" for i in range(7)]},
                 {"y": "x8", "x": [f"x{i}" for i in range(7)]}]
    spec = ModelSpec(estimator="sureg", outcome="x7", options={"equations": equations},
                     columns={"system": list(data.columns)})
    with use_workspace_budget(1):
        frame = ModelFrame(spec, data)
        monkeypatch.setattr(systems, "_variable_block", _no_allocation)
        with pytest.raises(AnalysisError) as exc:
            systems.build_system(frame, first=[], rest=list(data.columns), constant=True, outcomes=["x7", "x8"])
        assert exc.value.code == "workspace_limit"


def test_existing_fit_and_saved_inference_unchanged_by_budget():
    rng = np.random.default_rng(883)
    n = 120
    data = pd.DataFrame({"id": np.repeat(np.arange(20), 6), "t": np.tile(np.arange(6), 20),
                         "x": rng.normal(size=n)})
    data["y"] = 1 + .3 * data.x + rng.normal(size=n)
    fits = []
    for budget in (4, 16):
        with use_workspace_budget(budget):
            fits.append(oe.xtpcse(data=data, y="y", x=["x"], panel="id", time="t", pairwise=True))
    assert [c.estimate for c in fits[0].coefficients] == [c.estimate for c in fits[1].coefficients]
    assert fits[0].covariance_matrix == fits[1].covariance_matrix
    assert fits[0].sample_positions == list(range(n))
    assert fits[0].provenance["resource_plans"][-1]["budget_bytes"] == 4 * 1024**2
