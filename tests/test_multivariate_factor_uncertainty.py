"""Independent full-vector PF/Procrustes bootstrap and admission contracts."""
from __future__ import annotations

import copy
from decimal import Decimal
import json

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import orthogonal_procrustes
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import factor_uncertainty as u
from openecon.econometrics.multivariate.factor import factor
from openecon.econometrics.postest import index_codec
from openecon.resources import use_workspace_budget


def fixture(distribution="normal", *, n=603, factors=2, seed=563):
    rng = np.random.default_rng(seed)
    if factors == 2:
        target = np.array([[.8, 0], [.75, .1], [.7, -.05], [0, .65], [.1, .6], [-.05, .55]])
    else:
        target = np.zeros((3 * factors, factors))
        for axis, strength in enumerate([.85, .72, .6, .5][:factors]):
            target[3 * axis:3 * axis + 3, axis] = [strength, strength - .04, strength - .08]
    p = len(target)
    if distribution == "normal":
        latent, errors = rng.normal(size=(n, factors)), rng.normal(size=(n, p))
    else:
        latent = (rng.lognormal(sigma=.5, size=(n, factors)) - np.exp(.125)) \
            / np.sqrt((np.exp(.25) - 1) * np.exp(.25))
        errors = rng.standard_t(8, size=(n, p)) / np.sqrt(8 / 6)
    values = latent @ target.T + errors * np.sqrt(1 - (target * target).sum(1))
    scales = np.array([2, 3, 1, 4, .5, 1.8]) if p == 6 else np.linspace(.5, 3, p)
    means = np.array([5, -2, 12, .5, 3, 8]) if p == 6 else np.arange(p) - 3
    frame = pd.DataFrame(values * scales + means, columns=[f"x{i + 1}" for i in range(p)])
    return frame, target


def numpy_fit(values, factors, anchors=None, target=None):
    # Independent NumPy sample moments, inverse/SMC and eigensystem. The target
    # oracle uses SciPy's constrained Procrustes solver, never production code.
    shifted = values - values[0]
    mean = shifted.mean(0)
    centred = shifted - mean
    correction = centred.mean(0)
    centred -= correction
    covariance = centred.T @ centred / (len(values) - 1)
    sd = np.sqrt(covariance.diagonal())
    correlation = covariance / np.outer(sd, sd)
    np.fill_diagonal(correlation, 1)
    smc = 1 - 1 / np.linalg.inv(correlation).diagonal()
    reduced = correlation.copy()
    np.fill_diagonal(reduced, smc)
    roots, vectors = np.linalg.eigh(reduced)
    roots, vectors = roots[::-1], vectors[:, ::-1]
    loading = vectors[:, :factors] * np.sqrt(roots[:factors])
    if target is None:
        anchors = list(np.abs(loading).argmax(0)) if anchors is None else anchors
        loading *= np.where(loading[anchors, np.arange(factors)] < 0, -1, 1)
    else:
        loading = loading @ orthogonal_procrustes(loading, target)[0]
    uniqueness = 1 - (loading * loading).sum(1)
    return np.r_[loading.ravel(), uniqueness], {
        "loadings": loading, "uniqueness": uniqueness, "eigenvalues": roots,
        "smc": smc, "correlation": correlation,
        "mean": values[0] + mean + correction, "std": sd,
    }


def numpy_draws(values, factors, *, replications, seed, anchors=None, target=None):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return np.array([numpy_fit(values[torch.randint(len(values), (len(values),), generator=generator).numpy()],
                              factors, anchors=anchors, target=target)[0]
                     for _ in range(replications)])


@pytest.mark.parametrize("distribution", ["normal", "skewed_heavy_tail"])
@pytest.mark.parametrize("rotated", [False, True])
def test_two_domains_complete_numpy_scipy_bootstrap_oracle(distribution, rotated):
    frame, target = fixture(distribution)
    options = {"target": target} if rotated else {"anchors": ["x1", "x4"]}
    result = u.factor_multifactor_bootstrap(frame, list(frame), factors=2,
                                          replications=59, confidence=.9, seed=701, **options)
    reference = numpy_draws(frame.to_numpy(), 2, replications=59, seed=701,
                            anchors=None if rotated else [0, 3], target=target if rotated else None)
    point, details = numpy_fit(frame.to_numpy(), 2, anchors=None if rotated else [0, 3],
                               target=target if rotated else None)
    np.testing.assert_allclose(result["replicates"], reference, atol=4e-12, rtol=4e-12)
    np.testing.assert_allclose(result["estimates"].estimate, point, atol=3e-12, rtol=3e-12)
    covariance = np.cov(reference, rowvar=False, ddof=1)
    np.testing.assert_allclose(result["covariance"], covariance, atol=3e-13, rtol=4e-12)
    np.testing.assert_allclose(result["estimates"].std_error, np.sqrt(covariance.diagonal()), atol=3e-12)
    np.testing.assert_allclose(result["estimates"][["ci_lower", "ci_upper"]],
                               np.quantile(reference, [.05, .95], axis=0, method="linear").T, atol=4e-12)
    np.testing.assert_allclose(result["estimates"].bootstrap_bias, reference.mean(0) - point, atol=4e-12)
    for table, key in [("point_loadings", "loadings"), ("point_correlation", "correlation")]:
        np.testing.assert_allclose(result[table], details[key], atol=3e-12)
    for table, key in [("point_uniqueness", "uniqueness"), ("point_eigenvalues", "eigenvalues"), ("point_smc", "smc")]:
        np.testing.assert_allclose(result[table].iloc[:, 0], details[key], atol=3e-12)
    np.testing.assert_allclose(result["descriptives"], np.column_stack((details["mean"], details["std"])), atol=3e-12)
    assert result["estimates"][["p_value", "df"]].isna().all().all()
    assert not result.attrs["p_values_available"] and not result.attrs["inference_df_available"]
    assert result.attrs["successful_replications"] == 59 and result.attrs["failed_replications"] == []
    assert result.attrs["covariance_divisor"] == 58 and result.attrs["precision"] == "float64"
    assert result.attrs["factors"] == 2 and result.attrs["fixed_count"]
    assert "finite fourth moments" in result.attrs["inferential_assumptions"]
    if rotated:
        np.testing.assert_array_equal(result["target"], target)
        assert result.attrs["sign_anchors"] is None and result.attrs["kaiser"] is False
    else:
        for axis, name in enumerate(["x1", "x4"]):
            assert (result["replicates"][f"loading:{name}:Factor{axis + 1}"] > 0).all()


@pytest.mark.parametrize("factors", [3, 4])
@pytest.mark.parametrize("rotated", [False, True])
def test_fixed_three_and_four_factors_full_oracle(factors, rotated):
    frame, target = fixture(n=1003, factors=factors)
    options = {"target": target} if rotated else {"anchors": [f"x{3 * axis + 1}" for axis in range(factors)]}
    result = u.factor_multifactor_bootstrap(frame, list(frame), factors=factors, replications=39, seed=31, **options)
    reference = numpy_draws(frame.to_numpy(), factors, replications=39, seed=31,
                            anchors=None if rotated else [3 * axis for axis in range(factors)],
                            target=target if rotated else None)
    np.testing.assert_allclose(result["replicates"], reference, atol=2e-11, rtol=2e-11)
    np.testing.assert_allclose(result["covariance"], np.cov(reference, rowvar=False), atol=2e-12)
    assert result["point_loadings"].shape == (3 * factors, factors)


def portable(result):
    payload = {"title": result.title, "attrs": result.attrs, "tables": {
        name: {"data": u._json_value(table.to_numpy().tolist()), "index": list(table.index),
               "columns": list(table.columns), "attrs": table.attrs,
               "index_name": table.index.name, "column_name": table.columns.name}
        for name, table in result.items()}}
    return json.loads(json.dumps(payload, allow_nan=False))


def restore(payload):
    tables = {}
    for name, spec in payload["tables"].items():
        table = pd.DataFrame(spec["data"], index=spec["index"], columns=spec["columns"], dtype="float64")
        table.attrs = spec["attrs"]
        table.index.name, table.columns.name = spec["index_name"], spec["column_name"]
        tables[name] = table
    return TableSet(tables, title=payload["title"], **payload["attrs"])


@pytest.mark.parametrize("rotated", [False, True])
def test_full_json_saved_state_replay_seed_and_rng_independence(rotated):
    frame, target = fixture(n=203)
    options = {"target": target} if rotated else {}
    rng_before = torch.random.get_rng_state().clone()
    first = u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39, seed=83, **options)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    identical = u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39, seed=83, **options)
    changed = u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39, seed=84, **options)
    assert first.attrs == identical.attrs and first.title == identical.title
    assert not first["replicates"].equals(changed["replicates"])
    for key in first:
        pd.testing.assert_frame_equal(first[key], identical[key])
    saved = restore(portable(first))
    assert saved.attrs == first.attrs and saved.title == first.title
    assert saved.attrs["state_content_sha256"] == u._state_digest(saved)
    for key in first:
        np.testing.assert_array_equal(saved[key], first[key])
    replay_options = {"target": saved.attrs["target"]} if rotated else {"anchors": saved.attrs["sign_anchors"]}
    replay = u.factor_multifactor_bootstrap(saved["sample"], list(frame), factors=saved.attrs["factors"],
                                           replications=saved.attrs["replications"], seed=saved.attrs["seed"],
                                           confidence=saved.attrs["confidence"], **replay_options)
    for key in ["estimates", "covariance", "replicates", "replicate_diagnostics", "point_loadings", "point_uniqueness"]:
        pd.testing.assert_frame_equal(replay[key], first[key])
    assert "tabular" in saved.to_latex()
    for mutation in ["target", "confidence", "draw", "sample", "title"]:
        altered = restore(portable(first))
        if mutation == "target":
            altered.attrs["target"] = [[123]]
        elif mutation == "confidence":
            altered.attrs["confidence"] = .8
        elif mutation == "draw":
            altered["replicates"].iloc[0, 0] += .001
        elif mutation == "sample":
            altered["sample"].iloc[0, 0] += .01
        else:
            altered.title = "changed"
        assert u._state_digest(altered) != first.attrs["state_content_sha256"]


@pytest.mark.parametrize("rotated", [False, True])
def test_listwise_accounting_lossless_identities_and_input_unchanged(rotated):
    frame, target = fixture(n=203)
    ids = [None, pd.NA, pd.NaT, float("nan"), b"bytes", Decimal("1.20"), ("tuple", 7), *range(196)]
    frame.index = pd.Index(ids, dtype=object, tupleize_cols=False, name=("sample", 3))
    frame.columns.name = "measurements"
    frame.iloc[9, 1] = np.nan
    frame.iloc[17, 4] = np.nan
    frame["unused"] = "text"
    before = frame.copy(deep=True)
    options = {"target": target} if rotated else {"anchors": ["x1", "x4"]}
    names = list(frame)[:6]
    result = u.factor_multifactor_bootstrap(frame, names, factors=2, replications=39, **options)
    expected = u.factor_multifactor_bootstrap(frame[names].dropna(), names, factors=2, replications=39, **options)
    for key in ["estimates", "covariance", "replicates", "replicate_diagnostics"]:
        pd.testing.assert_frame_equal(result[key], expected[key])
    pd.testing.assert_frame_equal(frame, before)
    positions = [i for i in range(203) if i not in [9, 17]]
    assert result.attrs["n"] == 201 and result.attrs["n_missing"] == 2 and result.attrs["physical_rows"] == 203
    assert result.attrs["sample_positions"] == positions and list(result["sample"].index) == positions
    assert result.attrs["original_index_encoded"] == [index_codec.encode(ids[i]) for i in positions]
    assert result.attrs["original_index_names_encoded"] == [index_codec.encode(("sample", 3))]
    assert result.attrs["source_column_index_names_encoded"] == [index_codec.encode("measurements")]
    assert len(set(result.attrs["original_index_encoded"][:4])) == 4
    json.dumps(result.attrs, allow_nan=False)
    with pytest.raises(AnalysisError, match="missing"):
        u.factor_multifactor_bootstrap(frame, names, factors=2, missing="raise", **options)


def test_multiindex_axis_names_mapping_records_and_labelled_target():
    frame, target = fixture(n=203)
    frame.index = pd.MultiIndex.from_arrays([np.repeat("same", len(frame)), np.arange(len(frame))], names=["first", "second"])
    labelled = pd.DataFrame(target, index=frame.columns, columns=["Factor1", "Factor2"])
    direct = u.factor_multifactor_bootstrap(frame, list(frame), factors=2, target=labelled, replications=39)
    assert direct.attrs["original_index_is_multi"]
    assert [index_codec.decode(value) for value in direct.attrs["original_index_names_encoded"]] == ["first", "second"]
    assert index_codec.decode(direct.attrs["original_index_encoded"][2]) == ("same", 2)
    mapping = frame.to_dict("list")
    mapping["unused"] = object()
    for source in [mapping, frame.to_dict("records")]:
        fitted = u.factor_multifactor_bootstrap(source, list(frame), factors=2, target=target, replications=39)
        pd.testing.assert_frame_equal(fitted["replicates"], direct["replicates"])
    with pytest.raises(AnalysisError, match="labelled"):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, target=labelled.iloc[::-1], replications=39)


def exact_correlation_sample(correlation, n=120):
    rng = np.random.default_rng(84)
    z = rng.normal(size=(n, len(correlation)))
    z -= z.mean(0)
    q = np.linalg.qr(z)[0]
    values = q @ np.linalg.cholesky(correlation).T * np.sqrt(n - 1)
    return pd.DataFrame(values, columns=[f"x{i + 1}" for i in range(len(correlation))])


def block_correlation(strengths):
    p = 3 * len(strengths)
    r = np.eye(p)
    for i, strength in enumerate(strengths):
        r[3 * i:3 * i + 3, 3 * i:3 * i + 3] = strength
    np.fill_diagonal(r, 1)
    return r


def test_target_repeated_internal_roots_and_gauge_invariance_not_unrotated_axis_claim():
    frame = exact_correlation_sample(block_correlation([.6, .6]))
    target = np.zeros((6, 2))
    target[:3, 0], target[3:, 1] = .7, .7
    with pytest.raises(AnalysisError, match="gap"):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39)
    result = u.factor_multifactor_bootstrap(frame, list(frame), factors=2, target=target, replications=39)
    reference = numpy_draws(frame.to_numpy(), 2, replications=39, seed=0, target=target)
    np.testing.assert_allclose(result["replicates"], reference, atol=5e-12)
    assert result.attrs["internal_repeated_roots_allowed"]
    assert not result.attrs["point_unrotated_axes_identified"]
    loading = torch.tensor([[.8, .1], [.6, -.05], [.4, .03], [.1, .7], [.05, .5], [-.1, .8]], dtype=torch.float64)
    roots = torch.tensor([2., 2., -.1, -.2, -.3, -.4], dtype=torch.float64)
    t = torch.as_tensor(target)
    baseline = u._orientation(loading, roots, list(frame), t, None)[0]
    for gauge in [np.array([[0, 1], [-1, 0]]), np.array([[-1, 0], [0, 1]]),
                  np.array([[np.cos(.71), -np.sin(.71)], [np.sin(.71), np.cos(.71)]])]:
        transformed = u._orientation(loading @ torch.as_tensor(gauge, dtype=torch.float64), roots, list(frame), t, None)[0]
        np.testing.assert_allclose(transformed, baseline, atol=8e-16)
        np.testing.assert_allclose(transformed @ transformed.T, loading @ loading.T, atol=1e-15)


def test_target_boundary_gap_and_unrotated_zero_anchor_refused():
    frame = exact_correlation_sample(block_correlation([.6, .6, .6]))
    with pytest.raises(AnalysisError, match="gap"):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, target=np.eye(9, 2), replications=39)
    frame = exact_correlation_sample(block_correlation([.7, .5]))
    with pytest.raises(AnalysisError, match="anchor"):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, anchors=["x4", "x1"], replications=39)
    frame = exact_correlation_sample(np.eye(6))
    with pytest.raises(AnalysisError, match="positive"):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39)


def test_procrustes_full_oracle_reflections_scaling_covariance_and_old_pf_agreement():
    frame, target = fixture(n=303)
    result = u.factor_multifactor_bootstrap(frame, list(frame), factors=2, target=target, replications=39)
    old = factor(frame, list(frame), method="pf", factors=2, rotate=None)
    a = old["loadings"].to_numpy()
    q, _ = orthogonal_procrustes(a, target)
    b = result["point_loadings"].to_numpy()
    np.testing.assert_allclose(b, a @ q, atol=2e-12)
    np.testing.assert_allclose(b @ b.T, a @ a.T, atol=2e-12)
    rotation = result["point_rotation_matrix"].to_numpy()
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(2), atol=2e-15)
    for transform in [np.diag([-1, 1]), np.array([[0, 1], [1, 0]])]:
        other = u.factor_multifactor_bootstrap(frame, list(frame), factors=2, target=target @ transform, replications=39)
        np.testing.assert_allclose(other["point_loadings"], b @ transform, atol=2e-12)
        np.testing.assert_allclose(other["replicates"].iloc[:, :12].to_numpy().reshape(39, 6, 2),
                                   result["replicates"].iloc[:, :12].to_numpy().reshape(39, 6, 2) @ transform, atol=3e-12)
    scaled = u.factor_multifactor_bootstrap(frame, list(frame), factors=2, target=target * 1e-60, replications=39)
    np.testing.assert_allclose(scaled["replicates"], result["replicates"], atol=2e-12)


@pytest.mark.parametrize("rotated", [False, True])
def test_large_origin_and_physical_units_match_independent_functional(rotated):
    frame, target = fixture(n=303)
    frame = frame * [1e-5, 1e3, .02, 2, .1, 10] + [1e4, -1e9, 1e6, 1e10, -1e7, 1e11]
    options = {"target": target} if rotated else {"anchors": ["x1", "x4"]}
    result = u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39, **options)
    reference = numpy_draws(frame.to_numpy(), 2, replications=39, seed=0,
                            anchors=None if rotated else [0, 3], target=target if rotated else None)
    np.testing.assert_allclose(result["replicates"], reference, atol=4e-12)


@pytest.mark.parametrize("fail_all", [False, True])
def test_collect_all_refit_failures_no_replacement_or_successful_subset(monkeypatch, fail_all):
    frame, _ = fixture(n=203)
    original = u._parameters
    calls = []

    def failing(*args, **kwargs):
        calls.append(len(calls) + 1)
        if len(calls) > 1 and (fail_all or len(calls) in [3, 9]):
            raise AnalysisError("unidentified_factor", "declared refit geometry failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(u, "_parameters", failing)
    with pytest.raises(AnalysisError, match="bootstrap refits failed") as caught:
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39)
    error = caught.value
    assert len(calls) == 40 and error.code == "bootstrap_failure" and error.replications_attempted == 39
    assert error.successful_replications == (0 if fail_all else 37)
    assert [failure["replication"] for failure in error.failures] == (list(range(1, 40)) if fail_all else [2, 8])


@pytest.mark.parametrize("options", [{"factors": 1}, {"factors": 5}, {"factors": True}, {"factors": 2.0},
                                    {"replications": 18}, {"replications": 2000}, {"replications": True},
                                    {"confidence": 0}, {"confidence": 1}, {"confidence": .999},
                                    {"seed": -1}, {"seed": True}, {"seed": 2**63}, {"missing": "pairwise"},
                                    {"anchors": "x1"}, {"anchors": ["x1"]}, {"anchors": ["x1", "absent"]}])
def test_option_refusal(options):
    frame, _ = fixture(n=203)
    with pytest.raises(AnalysisError):
        u.factor_multifactor_bootstrap(frame, list(frame), **({"factors": 2} | options))


@pytest.mark.parametrize("options", [{"weights": "w"}, {"cluster": "id"}, {"method": "ml"},
                                    {"rotate": "varimax"}, {"kaiser": True}, {"device": "mps"}])
def test_unsupported_domains_are_not_implicitly_accepted(options):
    frame, _ = fixture(n=203)
    with pytest.raises(TypeError):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, **options)


@pytest.mark.parametrize("target", [np.zeros((6, 2)), np.ones((6, 2)), np.ones((5, 2)),
                                   np.full((6, 2), np.nan), np.full((6, 2), np.inf),
                                   np.ones((6, 2), dtype=complex), np.ones((6, 2), dtype=bool),
                                   [[True, False]] * 6, [["1", "0"]] * 6,
                                   torch.ones((6, 2), dtype=torch.bool), torch.ones((6, 2), device="meta")])
def test_nonfinite_nonreal_wrong_shape_singular_and_non_cpu_target_refusal(target):
    frame, _ = fixture(n=203)
    with pytest.raises(AnalysisError):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, target=target, replications=39)


def test_near_rank_target_and_conflicting_anchor_refusal():
    frame, target = fixture(n=203)
    weak = target.copy()
    weak[:, 1] *= 1e-10
    with pytest.raises(AnalysisError, match="target"):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, target=weak, replications=39)
    with pytest.raises(AnalysisError, match="anchors"):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, target=target, anchors=["x1", "x4"], replications=39)


def test_singular_near_rank_nonreal_infinite_constant_and_tiny_sample_refusals():
    frame, _ = fixture(n=203)
    for bad in [frame.assign(x6=frame.x5), frame.assign(x6=frame.x5 + 1e-6 * frame.x6),
                frame.assign(x6=1.), frame.assign(x6=np.inf), frame.assign(x6=True),
                frame.assign(x6=frame.x6.astype(complex)), frame.assign(x6="text")]:
        with pytest.raises(AnalysisError):
            u.factor_multifactor_bootstrap(bad, list(frame), factors=2, replications=39)
    with pytest.raises(AnalysisError, match="complete iid rows"):
        u.factor_multifactor_bootstrap(frame.iloc[:19], list(frame), factors=2, replications=39)
    with pytest.raises(AnalysisError):
        u.factor_multifactor_bootstrap(frame, ["x1", "x1", "x2"], factors=2)
    with pytest.raises(AnalysisError, match="more analysed"):
        u.factor_multifactor_bootstrap(frame, list(frame)[:3], factors=3)


def test_boundary_uniqueness_refused_without_clipping(monkeypatch):
    frame, _ = fixture(n=203)
    original = u.ex.principal_factors

    def boundary(*args):
        fitted = original(*args)
        fitted.loadings[0, :] = 1.
        return fitted

    monkeypatch.setattr(u.ex, "principal_factors", boundary)
    with pytest.raises(AnalysisError, match="uniqueness"):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39)


def test_geometry_work_workspace_and_export_admission_precede_value_conversion(monkeypatch):
    def forbidden(*args):
        raise AssertionError("values converted before admission")

    monkeypatch.setattr(u.u, "_selected_input", forbidden)
    names = [f"x{i}" for i in range(6)]
    with pytest.raises(AnalysisError, match="10000"):
        u.factor_multifactor_bootstrap({name: range(10001) for name in names}, names, factors=2)
    many = [f"x{i}" for i in range(17)]
    with pytest.raises(AnalysisError, match="16 variables"):
        u.factor_multifactor_bootstrap({name: range(50) for name in many}, many, factors=2)
    many = many[:16]
    with pytest.raises(AnalysisError, match="work"):
        u.factor_multifactor_bootstrap({name: range(10000) for name in many}, many, factors=4, replications=1999)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        u.factor_multifactor_bootstrap({name: range(4000) for name in names}, names, factors=2)
    long = ["x" * 4097, *names[1:]]
    with pytest.raises(AnalysisError, match="4096"):
        u.factor_multifactor_bootstrap({name: range(50) for name in long}, long, factors=2)
    monkeypatch.setattr(u, "MAX_EXPORT_BYTES", 100000)
    with pytest.raises(AnalysisError, match="32 MiB"):
        u.factor_multifactor_bootstrap({name: range(4000) for name in names}, names, factors=2)


def test_lossless_identity_size_admission_and_unsupported_identity():
    frame, _ = fixture(n=203)
    frame.index = ["a" * 4097] * len(frame)
    with pytest.raises(AnalysisError, match="4096"):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39)
    frame.index = [object()] * len(frame)
    with pytest.raises(AnalysisError, match="row labels"):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39)
    frame, _ = fixture(n=603)
    frame.index = ["a" * 4000] * len(frame)
    with pytest.raises(AnalysisError, match="2 MiB"):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39)


def test_dataset_summary_meta_and_invalid_resident_rows_refused_before_read():
    frame, _ = fixture(n=203)
    calls = []

    def batches():
        calls.append(True)
        yield frame

    stream = Dataset.from_batches(batches, columns=list(frame), row_count=len(frame))
    with pytest.raises(AnalysisError, match="resident"):
        u.factor_multifactor_bootstrap(stream, list(frame), factors=2)
    assert not calls
    from openecon.econometrics.multivariate.summary import prepare
    summary = prepare(np.eye(6), n=203, columns=list(frame), matrix="correlation")
    with pytest.raises(AnalysisError, match="summary"):
        u.factor_multifactor_bootstrap(summary, list(frame), factors=2)
    with pytest.raises(AnalysisError, match="CPU"):
        u.factor_multifactor_bootstrap({name: torch.empty(203, device="meta") for name in frame}, list(frame), factors=2)
    with pytest.raises(AnalysisError, match="consistent lengths"):
        u.factor_multifactor_bootstrap({name: range(202 if name == "x1" else 203) for name in frame}, list(frame), factors=2)
    with pytest.raises(AnalysisError, match="row must"):
        u.factor_multifactor_bootstrap([list(range(6))] * 203, list(frame), factors=2)


def test_duplicate_anchors_are_valid_and_strict_complete_state_hash():
    frame, _ = fixture(n=303)
    result = u.factor_multifactor_bootstrap(frame, list(frame), factors=2, anchors=["x2", "x2"], replications=39)
    assert result.attrs["sign_anchors"] == ["x2", "x2"]
    for label in ["loading:x2:Factor1", "loading:x2:Factor2"]:
        assert (result["replicates"][label] > 0).all()
    changed = copy.deepcopy(result)
    changed["replicates"].columns.name = "changed axis"
    assert u._state_digest(changed) != result.attrs["state_content_sha256"]


def test_resident_cpu_does_not_inherit_or_change_unrelated_default_device():
    frame, _ = fixture(n=203)
    before = torch.get_default_device()
    try:
        torch.set_default_device("meta")
        result = u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39)
        assert torch.get_default_device().type == "meta"
        assert result.attrs["device"] == "cpu"
    finally:
        torch.set_default_device(before)


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("scale", [1e-160, 1e-161])
def test_subnormal_sample_variance_refuses_before_correlations_lose_unit_invariance(rotated, scale):
    frame, target = fixture(n=603)
    options = {"target": target} if rotated else {"anchors": ["x1", "x4"]}
    with pytest.raises(AnalysisError, match="sample variance underflows") as caught:
        u.factor_multifactor_bootstrap(frame * scale, list(frame), factors=2, replications=39, **options)
    assert caught.value.code == "numerical_failure"


@pytest.mark.parametrize("rotated", [False, True])
def test_small_representable_units_preserve_complete_dimensionless_factor_functional(rotated):
    frame, target = fixture(n=303)
    options = {"target": target} if rotated else {"anchors": ["x1", "x4"]}
    reference = u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39, **options)
    actual = u.factor_multifactor_bootstrap(frame * 1e-100, list(frame), factors=2, replications=39, **options)
    for name in ["estimates", "covariance", "replicates", "point_loadings", "point_uniqueness"]:
        np.testing.assert_allclose(actual[name], reference[name], atol=4e-12, rtol=4e-12, equal_nan=True)


@pytest.mark.parametrize("kind", ["deep_tuple", "wide_tuple", "huge_text", "huge_bytes", "huge_integer", "huge_decimal"])
def test_identity_traversal_admission_precedes_recursive_codec(monkeypatch, kind):
    if kind == "deep_tuple":
        value = "leaf"
        for _ in range(33):
            value = (value,)
    elif kind == "wide_tuple":
        value = tuple(range(u.MAX_IDENTITY_NODES + 1))
    elif kind == "huge_text":
        value = "x" * 100000
    elif kind == "huge_bytes":
        value = b"x" * 100000
    elif kind == "huge_integer":
        value = 1 << (4 * u.MAX_LABEL_BYTES + 1)
    else:
        value = Decimal("1" * 15000)

    def forbidden(*args):
        raise AssertionError("unbounded value reached the recursive codec")

    monkeypatch.setattr(u.index_codec, "encode", forbidden)
    with pytest.raises(AnalysisError) as caught:
        u._identity(value)
    assert caught.value.code == "export_limit"


def test_identity_codec_recursion_and_unsupported_types_surface_structured_errors(monkeypatch):
    def recurse(*args):
        raise RecursionError("declared recursive codec failure")

    monkeypatch.setattr(u.index_codec, "encode", recurse)
    with pytest.raises(AnalysisError) as caught:
        u._identity(("ordinary", 3))
    assert caught.value.code == "invalid_index"


def test_aggregate_identity_node_budget_precedes_codec(monkeypatch):
    frame, _ = fixture(n=203)
    monkeypatch.setattr(u, "MAX_IDENTITY_NODES", 10)
    with pytest.raises(AnalysisError, match="traversal"):
        u.factor_multifactor_bootstrap(frame, list(frame), factors=2, replications=39)
