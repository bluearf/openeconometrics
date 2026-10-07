"""Native RD replay: exact local samples, ties, selectors and independent RBC algebra."""

import builtins
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import registry
from openecon.econometrics.streaming_rd import fit_streaming
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def fixture(n=780, seed=937, tied=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    if tied:
        x = np.round(x, 1)
    probability = np.clip(0.25 + 0.35 * (x >= 0) + 0.15 * x, 0.05, 0.95)
    d = rng.binomial(1, probability).astype(float)
    y = 0.5 + 0.4 * x + 0.3 * x * x + 1.1 * d + rng.normal(size=n) * 0.4
    return pd.DataFrame({"x": x, "y": y, "d": d, "g": np.arange(n) % 23})


def spec(*, fuzzy=False, cluster=False, **options):
    return ModelSpec(
        estimator="rdrobust",
        outcome="y",
        columns={"running": "x", **({"fuzzy": "d"} if fuzzy else {})},
        covariance="cluster" if cluster else None,
        cluster="g" if cluster else None,
        options=options,
    )


def compare(current, frame, batch_rows=67):
    expected = registry.load_entry(registry.get("rdrobust"))(current, frame)
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=batch_rows)
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        rtol=3e-7,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=3e-6, atol=2e-8
    )
    for key, value in expected.metrics.items():
        assert actual.metrics[key] == pytest.approx(value, rel=3e-6, abs=2e-8), (
            key,
            value,
            actual.metrics[key],
        )
    if current.columns.get("fuzzy"):
        for key, value in expected.extra["first_stage"].items():
            assert actual.extra["first_stage"][key] == pytest.approx(value, rel=3e-6, abs=2e-8)
    assert actual.nobs_original == len(frame)
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= batch_rows
    assert actual.sample_positions == []
    restored = ResultBundle.model_validate_json(actual.model_dump_json())
    assert restored.coefficients == actual.coefficients
    assert r"\begin{tabular}" in restored.to_latex()
    json.dumps(actual.model_dump(), allow_nan=False)
    return actual


@pytest.mark.parametrize("kernel", ["triangular", "epanechnikov", "uniform"])
@pytest.mark.parametrize("vce", ["nn", "hc0", "hc1", "hc2", "hc3"])
@pytest.mark.parametrize("fuzzy", [False, True])
def test_explicit_asymmetric_bandwidth_native_parity(kernel, vce, fuzzy):
    compare(spec(fuzzy=fuzzy, kernel=kernel, vce=vce, h=[0.57, 0.74], b=[0.83, 0.93]), fixture())


@pytest.mark.parametrize("vce", ["nn", "hc0", "hc1", "hc2", "hc3"])
@pytest.mark.parametrize("fuzzy", [False, True])
def test_clustered_tied_running_values_and_halo_batches(vce, fuzzy):
    compare(
        spec(fuzzy=fuzzy, cluster=True, vce=vce, h=[0.55, 0.75], b=[0.75, 0.95]),
        fixture(tied=True),
        batch_rows=7,
    )


@pytest.mark.parametrize("method", ["mserd", "msesum", "msetwo", "msecomb1", "msecomb2"])
@pytest.mark.parametrize("fuzzy", [False, True])
def test_complete_source_automatic_bandwidth_selection(method, fuzzy):
    compare(spec(fuzzy=fuzzy, bwselect=method), fixture(n=1300, seed=533), batch_rows=83)


@pytest.mark.parametrize("method", ["cerrd", "certwo", "cercomb2"])
def test_coverage_error_selectors_count_entire_side_clusters(method):
    compare(
        spec(fuzzy=True, cluster=True, bwselect=method, vce="hc1"),
        fixture(n=1100, seed=418),
        batch_rows=77,
    )


@pytest.mark.parametrize("order", [(0, 1), (1, 3), (2, 3)])
def test_polynomial_order_and_bias_equation(order):
    compare(spec(p=order[0], q=order[1], h=0.8, b=0.9, vce="hc0"), fixture(n=700, seed=857))


def test_independent_numpy_local_polynomial_rbc_formula():
    frame = fixture(n=530, seed=473)
    current = spec(h=[0.63, 0.79], b=[0.88, 0.94], vce="hc0")
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=31)
    estimates = []
    corrected = []
    variances = []
    robust = []
    for side, (h, b) in enumerate(zip([0.63, 0.79], [0.88, 0.94], strict=True)):
        mask = frame.x.to_numpy() >= 0 if side else frame.x.to_numpy() < 0
        mask &= np.abs(frame.x.to_numpy()) < max(h, b)
        x, y = frame.x.to_numpy()[mask], frame.y.to_numpy()[mask]
        rp = np.column_stack((np.ones(len(x)), x))
        rq = np.column_stack((np.ones(len(x)), x, x * x))
        wh = np.maximum(1 - np.abs(x / h), 0) / h
        wb = np.maximum(1 - np.abs(x / b), 0) / b
        ip = np.linalg.inv(rp.T @ (rp * wh[:, None]))
        iq = np.linalg.inv(rq.T @ (rq * wb[:, None]))
        bp = ip @ (rp.T @ (wh * y))
        bq = iq @ (rq.T @ (wb * y))
        rows = rp * wh[:, None]
        moment = rows.T @ (x / h) ** 2
        projection = wb * (rq @ iq[:, 2])
        qrows = rows - h * h * projection[:, None] * moment
        bc = ip @ (qrows.T @ y)
        res = y - rp @ bp
        resb = y - rq @ bq
        score = rows * res[:, None]
        scorer = qrows * resb[:, None]
        estimates.append(bp[0])
        corrected.append(bc[0])
        variances.append((ip @ score.T @ score @ ip)[0, 0])
        robust.append((ip @ scorer.T @ scorer @ ip)[0, 0])
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [estimates[1] - estimates[0], corrected[1] - corrected[0], corrected[1] - corrected[0]],
        atol=2e-10,
    )
    np.testing.assert_allclose(
        np.diag(actual.covariance_matrix), [sum(variances), sum(variances), sum(robust)], atol=2e-10
    )


def test_parameter_units_and_input_permutation_are_invariant():
    frame = fixture(n=570, seed=198)
    base = spec(h=0.7, b=0.85, vce="nn")
    expected = fit_streaming(base, Dataset.from_frame(frame), batch_rows=31)
    changed = frame.sample(frac=1, random_state=189).reset_index(drop=True)
    changed["x"] *= 1e70
    changed["y"] *= 1e60
    actual = fit_streaming(
        spec(h=0.7e70, b=0.85e70, vce="nn"), Dataset.from_frame(changed), batch_rows=71
    )
    np.testing.assert_allclose(
        np.array([c.estimate for c in actual.coefficients]) / 1e60,
        [c.estimate for c in expected.coefficients],
        rtol=3e-8,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        np.asarray(actual.covariance_matrix) / 1e120,
        expected.covariance_matrix,
        rtol=3e-7,
        atol=2e-9,
    )


def test_missing_sample_bounded_factory_no_estimator_dependency_and_meta_cpu(monkeypatch):
    frame = fixture(n=630, seed=893)
    frame.loc[9, "y"] = np.nan
    current = spec(h=0.6, b=0.8, vce="nn").model_copy(update={"missing": "drop"})
    expected = registry.load_entry(registry.get("rdrobust"))(current, frame)

    def chunks():
        for start in range(0, len(frame), 23):
            yield frame.iloc[start : start + 23].copy()

    source = Dataset.from_batches(chunks, frame.columns.tolist(), row_count=len(frame))
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"statsmodels", "linearmodels", "sklearn", "scipy"}:
            raise AssertionError("External estimator import attempted")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with torch.device("meta"):
        actual = fit_streaming(current, source, batch_rows=17)
        assert torch.empty(0).device.type == "meta"
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, atol=2e-9, rtol=3e-7
    )
    assert actual.dropped_rows == 1
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= 17


def test_source_and_budget_guards_clean_owned_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture(n=370)
    current = spec(h=0.6, b=0.8)
    with use_workspace_budget(1), pytest.raises(Exception):
        fit_streaming(current, Dataset.from_frame(frame), batch_rows=31)
    calls = 0

    def chunks():
        nonlocal calls
        calls += 1
        copy = frame.copy()
        if calls >= 8:
            copy.loc[33, "y"] += 1
        yield copy

    with pytest.raises(AnalysisError) as error:
        fit_streaming(
            current,
            Dataset.from_batches(chunks, frame.columns.tolist(), row_count=len(frame)),
            batch_rows=31,
        )
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "change", ["one_side", "too_few_distinct", "weak_jump", "bad_polynomial", "orphan_bias"]
)
def test_model_domain_failures_are_global(change):
    frame = fixture(n=400)
    current = spec(h=0.6, b=0.8)
    if change == "one_side":
        frame["x"] = frame.x.abs()
    elif change == "too_few_distinct":
        frame["x"] = np.where(frame.x < 0, -0.2, 0.2)
    elif change == "weak_jump":
        frame["d"] = 0.0
        current = spec(fuzzy=True, h=0.6, b=0.8)
    elif change == "bad_polynomial":
        current = current.model_copy(update={"options": {"h": 0.6, "p": 2, "q": 1}})
    else:
        current = current.model_copy(update={"options": {"b": 0.8}})
    with pytest.raises(AnalysisError) as error:
        fit_streaming(current, Dataset.from_frame(frame), batch_rows=37)
    assert error.value.code in {
        "invalid_cutoff",
        "insufficient_observations",
        "weak_first_stage",
        "invalid_spec",
    }


def test_nearest_neighbor_halo_is_planned_before_allocation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    with pytest.raises(Exception) as error:
        fit_streaming(spec(h=0.6, b=0.8, nnmatch=1000000), Dataset.from_frame(fixture(n=400)))
    assert getattr(error.value, "code", "") in {
        "workspace_limit",
    }
    assert not list(tmp_path.iterdir())


def test_unrepresentable_automatic_bandwidth_derivatives_refuse_precision():
    frame = fixture(n=1300, seed=533)
    frame["x"] *= 1e-150
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec(), Dataset.from_frame(frame), batch_rows=83)
    assert error.value.code in {"precision_unsupported", "singular_design"}
