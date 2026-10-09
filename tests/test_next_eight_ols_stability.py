"""Independent fixed-design projection, whole-path calibration and saved state."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError


def sample():
    rng = np.random.default_rng(173)
    n = 83
    x = rng.normal(size=(n, 2))
    y = 1.2 + x @ np.array([.4, -.7]) + rng.normal(size=n)
    return pd.DataFrame({"t": np.arange(n), "y": y, "x": x[:, 0], "w": x[:, 1]})


@pytest.mark.parametrize("intercept", [True, False])
@pytest.mark.parametrize("seed", [17, 987])
def test_entire_paths_and_every_null_draw_against_numpy(intercept, seed):
    f = sample()
    result = oe.ols_cusum(f, "y", ["x", "w"], time="t", intercept=intercept,
                          replications=131, seed=seed, alpha=.1)
    x = f[["x", "w"]].to_numpy()
    y = f.y.to_numpy()
    if intercept:
        x, y = x - x.mean(0), y - y.mean()
        x = np.column_stack((np.ones(len(f)), x))
    q, _ = np.linalg.qr(x, mode="reduced")
    residual = y - q @ (q.T @ y)
    fraction = np.arange(1, len(f) + 1) / len(f)

    def paths(e):
        rss = (e * e).sum(axis=-1)[..., None]
        return np.cumsum(e, axis=-1) / np.sqrt(rss), np.cumsum(e * e, axis=-1) / rss - fraction

    a, b = paths(residual)
    assert_allclose(result["path"]["cusum"], a, atol=3e-15)
    assert_allclose(result["path"]["cusumsq"], b, atol=3e-15)
    # Only the declared local RNG is shared. Projection and all path/p-value
    # arithmetic use NumPy and the full prespecified design independently.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    maxima = []
    for start in range(0, 131, 32):
        errors = torch.randn(min(32, 131-start), len(f), dtype=torch.float64,
                             generator=generator).numpy()
        simulated = errors - (errors @ q) @ q.T
        first, squares = paths(simulated)
        maxima.extend(np.column_stack((abs(first).max(1), abs(squares).max(1))))
    maxima = np.array(maxima)
    assert_allclose(result["null_maxima"].to_numpy(float), maxima, atol=5e-15)
    observed = np.array([abs(a).max(), abs(b).max()])
    assert_allclose(result["tests"].statistic, observed, atol=3e-15)
    assert_allclose(result["tests"].critical, np.quantile(maxima, .9, axis=0, method="higher"), atol=5e-15)
    assert_allclose(result["tests"].p_value, (1+(maxima >= observed).sum(0))/132, atol=0)
    assert result.attrs["failed_draws"] == 0
    assert result.attrs["rank"] == x.shape[1]
    assert list(result["path"].period) == list(f.t)


def test_design_scale_shift_row_order_and_local_rng():
    f = sample()
    torch.manual_seed(839)
    before = torch.random.get_rng_state().clone()
    base = oe.ols_cusum(f, "y", ["x", "w"], time="t", replications=99)
    shifted = f.assign(y=f.y*3+20, x=f.x+40, w=f.w-30).sample(frac=1, random_state=33)
    other = oe.ols_cusum(shifted, "y", ["x", "w"], time="t", replications=99)
    assert torch.equal(before, torch.random.get_rng_state())
    assert_allclose(base["path"][["cusum", "cusumsq"]], other["path"][["cusum", "cusumsq"]], atol=2e-14)
    assert_allclose(base["tests"][["statistic", "p_value", "critical"]], other["tests"][["statistic", "p_value", "critical"]], atol=2e-14)
    assert base.attrs["supplied_data_hash"] != other.attrs["supplied_data_hash"]
    assert abs(base["path"].cusum.iloc[-1]) < 1e-14
    assert abs(base["path"].cusumsq.iloc[-1]) < 1e-14


def test_collinear_terms_are_declared_without_changing_the_design():
    f = sample().assign(duplicate=lambda d: d.x * 2)
    one = oe.ols_cusum(f, "y", ["x", "w"], replications=99)
    two = oe.ols_cusum(f, "y", ["x", "w", "duplicate"], replications=99)
    assert two.attrs["rank"] == 3
    assert "duplicate" in str(two.attrs["notes"])
    assert_allclose(one["path"][["cusum", "cusumsq"]], two["path"][["cusum", "cusumsq"]], atol=3e-15)


def test_complete_file_restoration_without_numeric_execution(tmp_path, monkeypatch):
    result = oe.ols_cusum(sample(), "y", ["x", "w"], time="t", replications=131)
    path = tmp_path / "cusum.json"
    path.write_text(oe.summary_state(result))
    def prohibited(*args, **kwargs):
        raise AssertionError("restoration must not fit or simulate")
    monkeypatch.setattr(torch, "randn", prohibited)
    monkeypatch.setattr(torch.linalg, "qr", prohibited)
    restored = oe.restore_summary(path.read_text())
    assert oe.summary_state(restored) == path.read_text()
    assert len(restored["path"]) == 83 and len(restored["null_maxima"]) == 131
    # JSON sorts named tables; concatenated LaTeX follows that restored order.
    latex = restored.to_latex()
    assert latex.count("\\end{table}") + latex.count("\\end{longtable}") == len(result)
    assert "\\end{longtable}" in restored["null_maxima"].to_latex()
    assert json.loads(path.read_text())["attrs"]["curve_family"].startswith("each whole time path")


@pytest.mark.parametrize("options", [dict(replications=True), dict(replications=98), dict(seed=-1),
                                      dict(seed=2**63), dict(alpha=float("nan")), dict(errors="hac"),
                                      dict(max_work=True), dict(intercept="yes")])
def test_invalid_options(options):
    with pytest.raises(AnalysisError):
        oe.ols_cusum(sample(), "y", ["x"], **options)


@pytest.mark.parametrize("change", ["missing", "gap", "duplicate", "perfect", "infinite"])
def test_sample_failures_are_not_rescreened(change):
    f = sample()
    if change == "missing":
        f.loc[1, "y"] = np.nan
    elif change == "gap":
        f = f.drop(index=1)
    elif change == "duplicate":
        f.loc[1, "t"] = f.loc[0, "t"]
    elif change == "perfect":
        f["y"] = 2 * f.x + 3
    else:
        f.loc[1, "x"] = np.inf
    with pytest.raises(AnalysisError):
        oe.ols_cusum(f, "y", ["x"], time="t", replications=99)


def test_admission_before_model_or_simulation_allocations(monkeypatch):
    import openecon.econometrics.unitroot.ols_stability as module
    def prohibited(*args, **kwargs):
        raise AssertionError("over-budget data must be refused before model allocations")
    monkeypatch.setattr(module, "_model", prohibited)
    with pytest.raises(AnalysisError, match="2048"):
        oe.ols_cusum(pd.DataFrame({"y": np.arange(2049), "x": np.arange(2049)}), "y", ["x"])
    with pytest.raises(AnalysisError, match="32"):
        oe.ols_cusum(sample(), "y", [str(i) for i in range(32)])


def test_work_gate_before_rng_and_foreign_default_device(monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("simulation allocation before admission")
    with monkeypatch.context() as m:
        m.setattr(torch, "randn", prohibited)
        with pytest.raises(AnalysisError, match="max_work"):
            oe.ols_cusum(sample(), "y", ["x"], replications=99, max_work=1)
    old = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        result = oe.ols_cusum(sample(), "y", ["x"], replications=99)
        assert result.attrs["device"] == "cpu"
        assert str(torch.get_default_device()) == "meta"
    finally:
        torch.set_default_device(old)
