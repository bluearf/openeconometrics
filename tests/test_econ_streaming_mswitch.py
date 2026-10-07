"""Whole-source regime recursion, independent HMM law and covariance contracts."""

from __future__ import annotations

import itertools
import math
import gc

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset, scan
from openecon.econometrics.core import make_spec
from openecon.econometrics.ordered_replay import OrderedReplay
from openecon.econometrics.streaming_mswitch import (
    _Replay,
    _disk,
    _kernel,
    fit_streaming_mswitch,
    mswitch_probabilities_streaming,
)
from openecon.econometrics.tsmodels import mswitch_kernels as mk
from openecon.econometrics.tsmodels.mswitch import fit_mswitch, mswitch_probabilities
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


def data(n=105):
    rng = np.random.default_rng(63021)
    states = np.zeros(n, dtype=int)
    for j in range(1, n):
        states[j] = states[j - 1] if rng.random() < 0.94 else 1 - states[j - 1]
    x = rng.normal(size=n)
    y = np.array([-1.7, 2.5])[states] + 0.55 * x + rng.normal(scale=0.55, size=n)
    return pd.DataFrame({"y": y, "x": x, "t": np.arange(n) + 2**54})


def source(df, rows=13):
    return Dataset.from_batches(
        lambda: (df.iloc[j : j + rows].copy() for j in range(0, len(df), rows)),
        list(df),
        row_count=len(df),
    )


def specification(**options):
    return make_spec(
        "mswitch",
        outcome="y",
        predictors=["x"],
        time="t",
        options={"max_iterations": 180, **options},
    )


@pytest.mark.parametrize(
    "states,lags,arswitch,varswitch,switch",
    [
        (2, [], False, False, None),
        (3, [], False, True, ["Intercept", "x"]),
        (2, [1], False, True, None),
        (2, [1, 2], True, False, ["x"]),
        (3, [1], True, True, ["Intercept", "x"]),
    ],
)
def test_full_fisher_score_expanded_ar_initial_distribution_matches_native(
    states, lags, arswitch, varswitch, switch
):
    df = data(101)
    spec = specification(
        states=states, ar=lags, arswitch=arswitch, varswitch=varswitch, switch=switch
    )
    with OrderedReplay(spec, source(df, 7)) as ordered, _disk() as (sql, path):
        replay = _Replay(ordered, sql, path, 3)
        theta = replay.starts()[0]
        actual = replay.score(theta)
        x = (
            torch.tensor(np.column_stack((np.ones(len(df)), df.x)))[
                :, [*replay.switching, *replay.common]
            ]
            / replay.scales
        )
        y = torch.tensor(df.y.to_numpy()) / replay.scale_y
        expected = mk.score(
            replay.layout,
            theta,
            y,
            x[:, : replay.layout.switching],
            x[:, replay.layout.switching :],
            need_smoothed=True,
        )
        torch.testing.assert_close(actual["loglik"], expected["loglik"], rtol=2e-12, atol=2e-12)
        torch.testing.assert_close(actual["gradient"], expected["gradient"], rtol=2e-11, atol=2e-11)
        indicator = torch.nn.functional.one_hot(replay.layout.lag_state()[0], states).to(
            torch.float64
        )
        torch.testing.assert_close(
            actual["share"], (expected["smoothed"] @ indicator).mean(0), rtol=2e-12, atol=2e-12
        )
        assert replay.n == len(df) - max(lags, default=0)
        assert max(n for _, n, *_ in replay.blocks()) <= 3


@pytest.mark.parametrize("states", [2, 3])
def test_native_fused_block_filter_and_smoother_against_exact_path_enumeration(states):
    rng = np.random.default_rng(578)
    n = 5
    p = rng.uniform(0.1, 1.0, size=(states, states))
    p /= p.sum(1, keepdims=True)
    system = np.eye(states) - p.T
    system[-1] = 1.0
    rhs = np.zeros(states)
    rhs[-1] = 1.0
    pi = np.linalg.solve(system, rhs)
    density = np.exp(rng.normal(size=(n, states)))
    law = []
    for path in itertools.product(range(states), repeat=n + 1):
        probability = pi[path[0]]
        for j in range(n):
            probability *= p[path[j], path[j + 1]] * density[j, path[j + 1]]
        law.append((path, probability))
    total = sum(value for _, value in law)
    smooth = np.array(
        [
            [
                sum(value for path, value in law if path[j + 1] == state) / total
                for state in range(states)
            ]
            for j in range(n)
        ]
    )
    a = torch.tensor(p)
    current = torch.tensor(pi)
    outputs = []
    for j in range(0, n, 2):
        out = _kernel().forward(torch.tensor(np.log(density[j : j + 2])), a, current)
        outputs.append(out[:3])
        current = out[3]
    predicted = torch.cat([row[0] for row in outputs])
    filtered = torch.cat([row[1] for row in outputs])
    ll = torch.cat([row[2] for row in outputs])
    assert float(ll.sum()) == pytest.approx(math.log(total), abs=2e-14)
    future = torch.zeros(states, dtype=torch.float64)
    futurepred = future.clone()
    chunks = []
    hasnext = False
    for j in [4, 2, 0]:
        out, future, futurepred = _kernel().backward(
            filtered[j : j + 2], predicted[j : j + 2], a, future, futurepred, hasnext
        )
        chunks.append(out)
        hasnext = True
    actual = torch.cat(list(reversed(chunks)))
    np.testing.assert_allclose(actual, smooth, rtol=2e-13, atol=2e-13)


def numpy_likelihood(layout, theta, y, xs, xc):
    """Independent sequential normal HMM with expanded lag-regime law."""
    k = layout.states
    order = layout.order
    K = k ** (order + 1)
    cursor = 0
    beta = theta[cursor : cursor + k * layout.switching].reshape(k, layout.switching)
    cursor += beta.size
    alpha = theta[cursor : cursor + layout.common]
    cursor += alpha.size
    phi = theta[cursor : cursor + (k if layout.arswitch else 1) * len(layout.lags)].reshape(
        k if layout.arswitch else 1, len(layout.lags)
    )
    cursor += phi.size
    logsigma = theta[cursor : cursor + (k if layout.varswitch else 1)]
    cursor += logsigma.size
    logits = np.column_stack((theta[cursor:].reshape(k, k - 1), np.zeros(k)))
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    matrix = np.eye(k) - p.T
    matrix[-1] = 1.0
    rhs = np.zeros(k)
    rhs[-1] = 1.0
    pi = np.linalg.solve(matrix, rhs)
    histories = np.array([[(code // k**lag) % k for code in range(K)] for lag in range(order + 1)])
    a = np.zeros((K, K))
    initial = np.empty(K)
    for previous in range(K):
        initial[previous] = pi[histories[-1, previous]]
        for lag in range(order, 0, -1):
            initial[previous] *= p[histories[lag, previous], histories[lag - 1, previous]]
        for next in range(K):
            if all(
                histories[lag - 1, previous] == histories[lag, next] for lag in range(1, order + 1)
            ):
                a[previous, next] = p[histories[0, previous], histories[0, next]]
    mean = xs @ beta.T + (xc @ alpha)[:, None]
    deviation = y[:, None] - mean
    current = initial / initial.sum()
    ll = 0.0
    per = []
    for j in range(order, len(y)):
        residual = deviation[j, histories[0]].copy()
        for position, lag in enumerate(layout.lags):
            residual -= (
                phi[histories[0] if layout.arswitch else np.zeros(K, dtype=int), position]
                * deviation[j - lag, histories[lag]]
            )
        sigma = np.exp(logsigma[histories[0] if layout.varswitch else np.zeros(K, dtype=int)])
        logdensity = -0.5 * math.log(2.0 * math.pi) - np.log(sigma) - 0.5 * (residual / sigma) ** 2
        peak = logdensity.max()
        weighted = (current @ a) * np.exp(logdensity - peak)
        scale = weighted.sum()
        contribution = math.log(scale) + peak
        ll += contribution
        per.append(contribution)
        current = weighted / scale
    return ll, np.array(per)


@pytest.mark.parametrize("arswitch,varswitch", [(False, False), (True, True)])
def test_complete_analytic_gradient_and_period_score_covariance_independent_numpy_difference(
    arswitch, varswitch
):
    df = data(80)
    spec = specification(
        ar=[1, 2], arswitch=arswitch, varswitch=varswitch, switch=["Intercept", "x"]
    )
    with OrderedReplay(spec, source(df)) as ordered, _disk() as (sql, path):
        replay = _Replay(ordered, sql, path, 5)
        theta = replay.starts()[0]
        actual = replay.score(theta)
        y = df.y.to_numpy() / replay.scale_y
        x = (
            np.column_stack((np.ones(len(df)), df.x))[:, [*replay.switching, *replay.common]]
            / replay.scales.numpy()
        )
        xs = x[:, : replay.layout.switching]
        xc = x[:, replay.layout.switching :]
        eps = np.finfo(float).eps ** (1.0 / 3.0)
        h = eps * np.maximum(abs(theta.numpy()), 1.0)
        gradients = []
        rows = []
        for j in range(len(theta)):
            up = theta.numpy().copy()
            down = up.copy()
            up[j] += h[j]
            down[j] -= h[j]
            fu, pu = numpy_likelihood(replay.layout, up, y, xs, xc)
            fd, pd_ = numpy_likelihood(replay.layout, down, y, xs, xc)
            gradients.append((fu - fd) / (2.0 * h[j]))
            rows.append((pu - pd_) / (2.0 * h[j]))
        np.testing.assert_allclose(actual["gradient"], gradients, rtol=4e-7, atol=6e-8)
        score = np.array(rows).T
        factor = replay.score_factor(theta)
        np.testing.assert_allclose(factor.T @ factor, score.T @ score, rtol=3e-8, atol=2e-8)


@pytest.mark.parametrize("covariance", ["nonrobust", "robust", "opg"])
def test_complete_fit_reported_covariance_and_disk_probability_export(
    covariance, tmp_path, monkeypatch
):
    df = data(104)
    spec = specification().model_copy(update={"covariance": covariance})
    actual = fit_streaming_mswitch(spec, source(df), batch_rows=8)
    expected = fit_mswitch(spec, df)
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        rtol=2e-4,
        atol=3e-5,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=4e-4, atol=3e-6
    )
    assert actual.metrics["log_likelihood"] == pytest.approx(
        expected.metrics["log_likelihood"], abs=2e-7
    )
    assert actual.nobs == 104 and actual.sample_positions == []
    assert actual.provenance["filter_state_reset"] == "only once at full series start"
    assert actual.inference["small_sample_correction"] == (
        104 / 103 if covariance == "robust" else None
    )
    restored = ResultBundle.model_validate_json(actual.model_dump_json())
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    output = mswitch_probabilities_streaming(restored, source(df, 4))
    probability = pd.concat(list(output.iter_batches(batch_rows=5)), ignore_index=True)
    oracle = mswitch_probabilities(restored, df)
    np.testing.assert_allclose(
        probability.iloc[:, 1:].to_numpy(dtype=float),
        oracle.iloc[:, 1:].to_numpy(dtype=float),
        rtol=3e-12,
        atol=3e-12,
    )
    assert probability.period.tolist() == oracle.period.tolist()
    for name in ["predicted", "filtered", "smoothed"]:
        np.testing.assert_allclose(
            probability[[f"{name}_state1", f"{name}_state2"]].sum(1), 1.0, atol=3e-14
        )
    assert output.row_count == 104
    del output
    gc.collect()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "fault,code",
    [
        ("nothing", "invalid_option"),
        ("unknown", "invalid_option"),
        ("stateshape", "model_too_large"),
        ("short", "insufficient_observations"),
        ("gap", "time_gaps"),
        ("duplicate", "duplicate_time"),
        ("constant", "constant_outcome"),
    ],
)
def test_explicit_domains_and_scratch_cleanup(fault, code, tmp_path, monkeypatch):
    df = data(68)
    spec = specification()
    if fault == "nothing":
        spec = specification(switch=[])
    if fault == "unknown":
        spec = specification(switch=["absent"])
    if fault == "stateshape":
        spec = specification(states=6, ar=[4])
    if fault == "short":
        df = df.iloc[:15].copy()
    if fault == "gap":
        df.loc[13:, "t"] += 1
    if fault == "duplicate":
        df.loc[13, "t"] = df.loc[12, "t"]
    if fault == "constant":
        df["y"] = 1.0
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with pytest.raises(AnalysisError) as caught:
        fit_streaming_mswitch(spec, source(df), batch_rows=3)
    assert caught.value.code == code
    assert not list(tmp_path.iterdir())


def test_actual_parquet_ar_presample_drop_categories_physical_counts_and_cpu_state(tmp_path):
    df = data(92)
    df["g"] = pd.Categorical(np.where(df.x > 0, "b", "a"), categories=["a", "b", "unused"])
    df["y"] += 0.15 * (df.g == "b")
    df.loc[0, "y"] = np.nan
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    path = tmp_path / "data.parquet"
    df.to_parquet(path, index=False)
    spec = make_spec(
        "mswitch",
        outcome="y",
        predictors=["x", "g"],
        categorical=["g"],
        time="t",
        missing="drop",
        options={"ar": [1], "max_iterations": 180},
    )
    with OrderedReplay(spec, scan(path)) as ordered, _disk() as (sql, scratch):
        replay = _Replay(ordered, sql, scratch, 4)
        assert replay.n == 90 and replay.ordered.original_count == 92
        assert any(term.startswith("g[") for term in replay.design.terms)
        theta = replay.starts()[0]
        before = torch.random.get_rng_state().clone()
        with torch.device("meta"):
            # Public fit wraps cpu; internal scientific replay is already cpu.
            with torch.device("cpu"):
                actual = replay.score(theta)
            assert torch.empty(0).device.type == "meta"
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        assert math.isfinite(float(actual["loglik"]))
    with use_workspace_budget(8), pytest.raises(AnalysisError) as caught:
        fit_streaming_mswitch(spec, scan(path))
    assert caught.value.code == "workspace_limit"


def test_source_changed_after_fit_and_public_meta_cpu_scope(tmp_path, monkeypatch):
    df = data(76)
    spec = specification()
    rng = torch.random.get_rng_state().clone()
    dtype = torch.get_default_dtype()
    with torch.device("meta"):
        fit = fit_streaming_mswitch(spec, source(df), batch_rows=6)
        assert torch.empty(0).device.type == "meta"
    assert fit.nobs == 76 and torch.get_default_dtype() == dtype
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        for j in range(0, len(df), 7):
            rows = df.iloc[j : j + 7].copy()
            if calls > 1:
                rows["y"] += 0.03
            yield rows

    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with pytest.raises(AnalysisError) as caught:
        fit_streaming_mswitch(
            spec, Dataset.from_batches(factory, list(df), row_count=len(df)), batch_rows=6
        )
    assert caught.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


def test_long_series_finite_history_spool_fixed_workspace_and_exact_full_likelihood():
    df = data(431)
    spec = specification(states=3, ar=[1], arswitch=True, varswitch=True)
    with OrderedReplay(spec, source(df, 17)) as ordered, _disk() as (sql, path):
        replay = _Replay(ordered, sql, path, 6)
        theta = replay.starts()[0]
        actual = replay.score(theta)
        y = torch.tensor(df.y.to_numpy()) / replay.scale_y
        x = torch.tensor(np.column_stack((np.ones(len(df)), df.x))) / replay.scales
        expected = mk.score(replay.layout, theta, y, x[:, :1], x[:, 1:])
        torch.testing.assert_close(actual["loglik"], expected["loglik"], rtol=3e-12, atol=3e-12)
        torch.testing.assert_close(actual["gradient"], expected["gradient"], rtol=3e-11, atol=3e-11)
        assert replay.n == 430 and len(replay.report) == 400
        assert max(n for _, n, *_ in replay.blocks()) <= 6
        assert replay.last_plan.estimated_bytes <= replay.last_plan.budget_bytes
