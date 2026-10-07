"""Exact whole-group conditional replay: mixtures, weights, covariance and resource guards."""

import builtins
import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics import registry
from openecon.econometrics.streaming_conditional import fit_streaming
from openecon.models import ModelSpec, ResultBundle
from openecon.resources import use_workspace_budget


def fixture(groups=60, size=5, seed=428):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(groups), size)
    x, z = rng.normal(size=(2, len(g)))
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.6 * x + 0.2 * z + rng.normal(size=groups)[g]))))
    return pd.DataFrame(
        {
            "y": y,
            "g": g,
            "x": x,
            "z": z,
            "o": 0.1 * z,
            "w": g % 3 + 1,
            "c": g // 3,
            "h": np.arange(len(g)) % 7,
            "a": np.where(z > 0, "north", "south"),
        }
    )


def spec(kind="nonrobust", weight=None, categorical=False):
    return ModelSpec(
        estimator="clogit",
        outcome="y",
        predictors=["x", "z", *(["a"] if categorical else [])],
        intercept=False,
        covariance=kind,
        cluster=["c", "h"] if kind == "cluster" else None,
        columns={"group": "g", "offset": "o"},
        weights="w" if weight else None,
        weight_type=weight,
        categorical=["a"] if categorical else [],
    )


def compare(current, frame, batch_rows=37):
    expected = registry.load_entry(registry.get("clogit"))(current, frame)
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=batch_rows)
    assert [c.term for c in actual.coefficients] == [c.term for c in expected.coefficients]
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        rtol=3e-8,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=3e-7, atol=2e-9
    )
    for name, value in expected.metrics.items():
        assert actual.metrics[name] == pytest.approx(value, rel=3e-7, abs=2e-8)
    assert actual.nobs == expected.nobs
    assert actual.dropped_rows == expected.dropped_rows
    assert actual.extra["observations_dropped"] == expected.extra["observations_dropped"]
    assert actual.extra["group_sizes"] == expected.extra["group_sizes"]
    assert actual.sample_positions == []
    assert actual.provenance["streaming"]["maximum_batch_rows"] <= batch_rows
    restored = ResultBundle.model_validate_json(actual.model_dump_json())
    assert restored.coefficients == actual.coefficients
    assert r"\begin{tabular}" in restored.to_latex()
    json.dumps(actual.model_dump(), allow_nan=False)
    return actual


@pytest.mark.parametrize("kind", ["nonrobust", "opg", "robust", "cluster"])
@pytest.mark.parametrize("weight", [None, "fweight", "iweight", "pweight"])
def test_full_group_native_parity_across_covariance_and_weight_semantics(kind, weight):
    if kind in {"nonrobust", "opg"} and weight == "pweight":
        pytest.skip("Declared native pweight covariance domain excludes information and OPG.")
    compare(spec(kind, weight), fixture())


def test_categorical_missing_absorbed_collinearity_and_variable_group_sizes():
    frame = fixture(groups=80)
    frame = frame.drop(index=np.arange(3, len(frame), 17)).reset_index(drop=True)
    frame["constant_within"] = frame.g * 0.3
    frame["duplicate"] = 2 * frame.x
    frame.loc[5, "x"] = np.nan
    current = spec("cluster", categorical=True).model_copy(
        update={"predictors": ["x", "z", "a", "constant_within", "duplicate"], "missing": "drop"}
    )
    compare(current, frame, batch_rows=43)


@pytest.mark.parametrize("seed", [35, 642])
def test_single_positive_single_negative_and_multi_positive_group_kernels(seed):
    frame = fixture(groups=60, size=6, seed=seed)
    rng = np.random.default_rng(seed)
    y = []
    for group in range(60):
        values = np.zeros(6, dtype=int)
        values[rng.choice(6, [1, 5, 3][group % 3], replace=False)] = 1
        y.extend(values)
    frame["y"] = y
    compare(spec("robust"), frame, batch_rows=41)


def test_independent_enumerated_conditional_likelihood_at_fitted_parameters():
    frame = fixture(groups=30, size=5, seed=491)
    current = spec()
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=31)
    beta = np.array([c.estimate for c in actual.coefficients])
    likelihood = 0.0
    for _, group in frame.groupby("g"):
        y = group.y.to_numpy()
        m = int(y.sum())
        if m in {0, len(y)}:
            continue
        x = group[["x", "z"]].to_numpy()
        eta = x @ beta + group.o.to_numpy()
        sums = np.array(
            [eta[list(indices)].sum() for indices in itertools.combinations(range(len(y)), m)]
        )
        shift = sums.max()
        likelihood += float(y @ eta) - (shift + math.log(np.exp(sums - shift).sum()))
    assert actual.metrics["log_likelihood"] == pytest.approx(likelihood, rel=2e-10, abs=2e-9)


def test_cpu_context_no_external_numerical_solver_and_changed_source_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture(groups=40)

    def chunks():
        for start in range(0, len(frame), 17):
            yield frame.iloc[start : start + 17].copy()

    source = Dataset.from_batches(chunks, frame.columns.tolist(), row_count=len(frame))
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"scipy", "statsmodels", "linearmodels", "sklearn"}:
            raise AssertionError("External numerical solver import attempted")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    with torch.device("meta"):
        actual = fit_streaming(spec("robust"), source, batch_rows=17)
        assert torch.empty(0).device.type == "meta"
    assert actual.nobs > 0
    calls = 0

    def changed():
        nonlocal calls
        calls += 1
        copy = frame.copy()
        if calls >= 5:
            copy.loc[7, "z"] += 0.1
        yield copy

    with pytest.raises(AnalysisError) as error:
        fit_streaming(
            spec(),
            Dataset.from_batches(changed, frame.columns.tolist(), row_count=len(frame)),
            batch_rows=31,
        )
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())


def test_complete_group_workspace_guard_precedes_conditional_recursion(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = fixture(groups=4, size=2500)
    with pytest.raises(AnalysisError) as error:
        fit_streaming(spec(), Dataset.from_frame(frame), batch_rows=17)
    assert error.value.code == "conditional_group_workspace"
    assert not list(tmp_path.iterdir())
    with use_workspace_budget(1), pytest.raises(Exception):
        fit_streaming(spec(), Dataset.from_frame(frame), batch_rows=17)


@pytest.mark.parametrize("change", ["outcome", "constant", "group_weight", "separation"])
def test_unidentified_domains_never_return_silent_sampled_fits(change):
    frame = fixture(groups=45)
    current = spec("robust", "fweight")
    if change == "outcome":
        frame["y"] = 0
    elif change == "constant":
        frame["x"] = frame.g * 0.2
        frame["z"] = frame.g * 0.3
    elif change == "group_weight":
        frame.loc[0, "w"] = 99
    else:
        frame["x"] = frame.y
        frame["z"] = 0.0
    with pytest.raises(AnalysisError) as error:
        fit_streaming(current, Dataset.from_frame(frame), batch_rows=37)
    assert error.value.code in {
        "no_outcome_variation",
        "no_within_group_variation",
        "weights_not_constant_within_group",
        "separation_detected",
        "singular_information",
    }


def test_unit_scaling_group_offset_and_partition_permutation_invariance():
    frame = fixture(groups=60, seed=526)
    expected = fit_streaming(spec("robust"), Dataset.from_frame(frame), batch_rows=37)
    changed = frame.sample(frac=1, random_state=333).reset_index(drop=True)
    changed["x"] = (changed.x + changed.g * 0.4) * 1e60
    changed["z"] *= 1e-40
    actual = fit_streaming(spec("robust"), Dataset.from_frame(changed), batch_rows=41)
    units = np.array([1e60, 1e-40])
    np.testing.assert_allclose(
        np.array([c.estimate for c in actual.coefficients]) * units,
        [c.estimate for c in expected.coefficients],
        rtol=3e-8,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        np.asarray(actual.covariance_matrix) * units[:, None] * units[None, :],
        expected.covariance_matrix,
        rtol=3e-7,
        atol=2e-9,
    )


def test_complete_groups_crossing_small_reader_partitions_remain_exact():
    frame = fixture(groups=12, size=80, seed=295)
    current = spec("robust")
    expected = registry.load_entry(registry.get("clogit"))(current, frame)
    actual = fit_streaming(current, Dataset.from_frame(frame), batch_rows=17)
    np.testing.assert_allclose(
        actual.covariance_matrix, expected.covariance_matrix, rtol=3e-7, atol=2e-9
    )
    np.testing.assert_allclose(
        [c.estimate for c in actual.coefficients],
        [c.estimate for c in expected.coefficients],
        rtol=3e-8,
        atol=2e-9,
    )
    assert actual.provenance["streaming"]["reader_batch_rows"] == 17
    assert actual.provenance["solver_diagnostics"]["maximum_whole_group_block_rows"] == 80
    assert (
        actual.provenance["streaming"]["resource_plan"]["estimated_workspace_bytes"]
        <= actual.provenance["streaming"]["working_memory_budget_bytes"]
    )
