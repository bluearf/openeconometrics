"""Independent complete fixed-query score bootstrap references and reuse gates."""
import copy
import json
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.multivariate.pca_score_uncertainty import pca_bootstrap_scores
from openecon.econometrics.multivariate.pca_uncertainty import pca_bootstrap
from openecon.econometrics.postest.index_codec import decode
from openecon.econometrics.summary_state import restore_summary, summary_state


def fixture(domain=0):
    rng = np.random.default_rng(216+domain)
    z = rng.normal(size=(340, 4)) if not domain else rng.standard_t(8, size=(340, 4))
    a = np.column_stack((4*z[:, 0]+.5*z[:, 1], 2*z[:, 0]+.7*z[:, 2],
                         3*z[:, 0]+.8*z[:, 3], z[:, 1]+.3*z[:, 2]))
    return pd.DataFrame(a+[7, -2, 12, 3], columns=list("abcd"))


def oracle(x, query, matrix, anchor):
    mean = x.mean(0)
    centred = x-mean
    cov = centred.T@centred/(len(x)-1)
    sd = np.sqrt(np.diag(cov))
    target = cov if matrix == "covariance" else cov/np.outer(sd, sd)
    _, v = np.linalg.eigh(target)
    axis = v[:, -1:]
    axis *= np.sign(axis[anchor, 0])
    scaled = query-mean
    if matrix == "correlation":
        scaled = scaled/sd
    return (scaled@axis).ravel()


@pytest.mark.parametrize("domain", [0, 1])
@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_every_score_and_complete_joint_inference_against_numpy(domain, matrix):
    source = fixture(domain)
    query = pd.DataFrame([[8, 0, 15, 3], [5, -3, 9, 1], [11, 2, 15, 4]], columns=list("abcd"))
    fit = pca_bootstrap(source, list("abcd"), components=1, matrix=matrix,
                        replications=99, confidence=.9, seed=7)
    out = pca_bootstrap_scores(fit, query)
    anchor = list("abcd").index(fit.attrs["sign_anchors"][0])
    x, z = source.to_numpy(), query.to_numpy()
    expected = np.vstack([oracle(x[indices.astype(int)], z, matrix, anchor)
                          for indices in fit["resample_indices"].to_numpy()])
    point = oracle(x, z, matrix, anchor)
    covariance = np.cov(expected, rowvar=False, ddof=1)
    bounds = np.quantile(expected, [.05, .95], axis=0, method="linear")
    np.testing.assert_allclose(out["replicates"], expected, rtol=2e-12, atol=3e-12)
    np.testing.assert_allclose(out["point_scores"].to_numpy().ravel(), point, rtol=2e-12, atol=3e-12)
    np.testing.assert_allclose(out["covariance"], covariance, rtol=2e-12, atol=3e-12)
    np.testing.assert_allclose(out["estimates"]["std_error"], np.sqrt(np.diag(covariance)), rtol=2e-12, atol=3e-12)
    np.testing.assert_allclose(out["estimates"][["ci_lower", "ci_upper"]], bounds.T, rtol=2e-12, atol=3e-12)
    assert np.max(abs(covariance-np.diag(np.diag(covariance)))) > .001
    assert out.attrs["familywise_intervals"] is False
    assert out["estimates"][["p_value", "df"]].isna().all().all()
    assert out.attrs["reestimate_means_every_draw"] is True
    assert out.attrs["restandardize_every_draw"] == (matrix == "correlation")
    assert "future-observation" in out.attrs["query_contract"]


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
def test_full_saved_source_json_reuse_and_query_missing_typed_identity(matrix):
    source = fixture()
    source.index = pd.MultiIndex.from_tuples([(i//20, f"person-{i}") for i in range(len(source))], names=["block", "person"])
    source.columns.name = "measurement"
    source.iloc[4, 0] = np.nan
    fit = pca_bootstrap(source, list("abcd"), components=1, matrix=matrix, replications=39, confidence=.9)
    restored = restore_summary(summary_state(fit))
    query = pd.DataFrame([[1, 2, 3, 4], [1, np.nan, 3, 4], [8, 2, 14, 4]], columns=list("abcd"),
                         index=pd.MultiIndex.from_tuples([(True, "duplicate"), (False, "missing"), (True, "duplicate")], names=["flag", "id"]))
    out = pca_bootstrap_scores(restored, query)
    assert out.attrs["query_positions"] == [0, 2]
    assert out["point_scores"].iloc[1].isna().all()
    assert out.attrs["n_missing_query"] == 1
    assert [decode(v) for v in out.attrs["query_index_codes"]] == list(query.index)
    assert [decode(v) for v in out.attrs["query_index_names"]] == ["flag", "id"]
    persisted = json.loads(summary_state(out))
    again = json.loads(summary_state(restore_summary(summary_state(out))))
    assert persisted == again
    assert all("fit__"+key in out for key in fit)
    np.testing.assert_array_equal(out["fit__resample_indices"], fit["resample_indices"])
    with pytest.raises(AnalysisError):
        pca_bootstrap_scores(restored, query, missing="raise")


def test_correlation_query_uses_each_draw_scale_and_affine_unit_invariance():
    source = fixture()
    query = pd.DataFrame([[10, 1, 15, 4], [2, -5, 8, 1]], columns=list("abcd"))
    fit = pca_bootstrap(source, list("abcd"), components=1, matrix="correlation", replications=39, confidence=.9)
    out = pca_bootstrap_scores(fit, query)
    means = fit["replicate_means"].to_numpy()
    axes = fit["replicates"].to_numpy()[:, 1:5]
    frozen = np.einsum("bqp,bp->bq", (query.to_numpy()[None]-means[:, None])/fit["descriptives"]["std_dev"].to_numpy(), axes)
    assert np.max(abs(frozen-out["replicates"].to_numpy())) > .01
    scale, shift = np.array([.001, 10, 100, .2]), np.array([1, -10, 23, 2])
    fit2 = pca_bootstrap(source*scale+shift, list("abcd"), components=1, matrix="correlation", replications=39, confidence=.9)
    out2 = pca_bootstrap_scores(fit2, query*scale+shift)
    np.testing.assert_allclose(out["replicates"], out2["replicates"], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("field", ["sample", "point_eigenvectors", "point_loadings", "point_matrix", "point_covariance",
                                    "replicates", "replicate_means", "replicate_standard_deviations", "covariance", "estimates", "resample_indices"])
def test_rehashed_or_reexported_numeric_tamper_refuses(field):
    fit = pca_bootstrap(fixture(), list("abcd"), components=1, matrix="correlation", replications=19, confidence=.8)
    corrupted = restore_summary(summary_state(fit))
    if field == "resample_indices":
        corrupted[field] = corrupted[field].astype(float)
    corrupted[field].iloc[0, 0] += .01
    corrupted = restore_summary(summary_state(corrupted))
    with pytest.raises(AnalysisError, match="canonical|invalid"):
        pca_bootstrap_scores(corrupted, fixture().iloc[:1])


@pytest.mark.parametrize("key,value", [("seed", 100), ("replications", 20), ("components", True), ("physical_rows", 341),
                                        ("sample_positions", [0]*340), ("source_index_nlevels", 10000),
                                        ("estimated_work", "unbounded"), ("sign_anchors", ["b"]),
                                        ("source_content_sha256", "0"*64)])
def test_semantic_saved_settings_tamper_refuses(key, value):
    fit = pca_bootstrap(fixture(), list("abcd"), components=1, matrix="correlation", replications=19, confidence=.8)
    fit.attrs[key] = copy.deepcopy(value)
    with pytest.raises(AnalysisError):
        pca_bootstrap_scores(fit, fixture().iloc[:1])


@pytest.mark.parametrize("kind", ["too_many", "boolean", "infinite", "missing_name", "empty"])
def test_query_admission(kind):
    fit = pca_bootstrap(fixture(), list("abcd"), components=1, matrix="correlation", replications=19, confidence=.8)
    query = fixture().iloc[:2].copy()
    if kind == "too_many":
        query = fixture().iloc[:129]
    elif kind == "boolean":
        query["a"] = True
    elif kind == "infinite":
        query.iloc[0, 0] = np.inf
    elif kind == "missing_name":
        query = query.drop(columns="d")
    else:
        query = query.iloc[:0]
    with pytest.raises(AnalysisError):
        pca_bootstrap_scores(fit, query)


def test_repeated_query_covariance_is_retained_without_inversion():
    fit = pca_bootstrap(fixture(), list("abcd"), components=1, matrix="correlation", replications=19, confidence=.8)
    query = pd.concat([fixture().iloc[:1]]*2)
    out = pca_bootstrap_scores(fit, query)
    cov = out["covariance"].to_numpy()
    np.testing.assert_allclose(cov, np.full((2, 2), cov[0, 0]), rtol=1e-14, atol=1e-14)
    assert np.linalg.matrix_rank(cov) == 1


def test_covariance_two_axes_query_joint_order_and_independent_complete_vectors():
    rng = np.random.default_rng(435)
    rotate, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    x = (rng.normal(size=(450, 4))*[10, 3, .7, .2])@rotate.T+[3, 5, -8, 2]
    source = pd.DataFrame(x, columns=list("abcd"))
    query = source.iloc[[9, 24, 44]].copy()+[1, 2, -3, 2]
    fit = pca_bootstrap(source, list("abcd"), components=2, matrix="covariance", replications=39, confidence=.9, seed=24)
    out = pca_bootstrap_scores(fit, query)
    expected = []
    for indices in fit["resample_indices"].to_numpy():
        draw = x[indices.astype(int)]
        cov = np.cov(draw, rowvar=False, ddof=1)
        _, axes = np.linalg.eigh(cov)
        axes = axes[:, ::-1][:, :2]
        for j, anchor in enumerate(fit.attrs["sign_anchors"]):
            axes[:, j] *= np.sign(axes[list("abcd").index(anchor), j])
        expected.append(((query.to_numpy()-draw.mean(0))@axes).ravel())
    expected = np.vstack(expected)
    np.testing.assert_allclose(out["replicates"], expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(out["covariance"], np.cov(expected, rowvar=False, ddof=1), rtol=2e-11, atol=2e-11)
    assert out.attrs["parameter_order"] == [[i, f"Comp{j+1}"] for i in range(3) for j in range(2)]
    assert np.max(abs(out["covariance"].to_numpy()[::2, 1::2])) > .001


@pytest.mark.parametrize("kind", ["oversized_draw_table", "extra_table", "extra_huge_metadata", "oversized_text_cell", "boolean_geometry", "infinite_df"])
def test_forged_state_refuses_before_generic_serialization(monkeypatch, kind):
    from openecon.econometrics.multivariate import pca_score_uncertainty as module
    fit = pca_bootstrap(fixture(), list("abcd"), components=1, matrix="correlation", replications=19, confidence=.8)
    if kind == "oversized_draw_table":
        fit["replicates"] = pd.concat([fit["replicates"]]*20, ignore_index=True)
    elif kind == "extra_table":
        fit["forged"] = pd.DataFrame([["not a fitted table"]])
    elif kind == "extra_huge_metadata":
        fit.attrs["forged"] = "x"*(4*1024**2+1)
    elif kind == "oversized_text_cell":
        fit["settings"].iloc[0, 1] = "x"*(4*1024**2+1)
    elif kind == "boolean_geometry":
        fit["point_eigenvectors"] = fit["point_eigenvectors"].astype(bool)
    else:
        fit["estimates"].iloc[0, -1] = np.inf
    def forbidden_serialization(*args, **kwargs):
        pytest.fail("Forged admission reached generic serialization")
    monkeypatch.setattr(module, "summary_state", forbidden_serialization)
    with pytest.raises(AnalysisError):
        pca_bootstrap_scores(fit, fixture().iloc[:1])


def independent_stable_two_axis_scores(values, query, matrix):
    # The functional is evaluated on represented input values. Subtracting a
    # rounded absolute mean at a large common level is a different arithmetic
    # target; preserve its origin and small centered offset independently.
    origin = values[0]
    shift = values-origin
    offset = shift.mean(0)
    centred = shift-offset
    correction = centred.mean(0)
    centred -= correction
    offset += correction
    covariance = centred.T@centred/(len(values)-1)
    sd = np.sqrt(covariance.diagonal())
    target = covariance if matrix == "covariance" else covariance/np.outer(sd, sd)
    _, axes = np.linalg.eigh(target)
    axes = axes[:, ::-1][:, :2].copy()
    axes *= np.sign(axes[[0, 3], [0, 1]])
    query_centred = (query-origin)-offset
    if matrix == "correlation":
        query_centred /= sd
    return (query_centred@axes).ravel(), origin, offset


@pytest.mark.parametrize("matrix", ["covariance", "correlation"])
@pytest.mark.parametrize("level", [0., 1e12])
def test_two_axis_fixed_query_large_level_stable_functional_full_inference_and_json(matrix, level):
    rng = np.random.default_rng(891)
    n = 603
    latent, errors = rng.normal(size=(n, 2)), rng.normal(size=(n, 6))
    loading = np.array([.95, .93, .9, .8, .78, .76])
    values = latent[:, [0, 0, 0, 1, 1, 1]]*loading+errors*np.sqrt(1-loading**2)
    values = values*[3, 2.5, 2, 1.5, 1.3, 1.1]+[5, -2, 12, .5, 9, -3]+level
    names = list("abcdef")
    source = pd.DataFrame(values, columns=names)
    query_values = np.array([[7., 0., 15., 2., 12., -2.], [4., -4., 10., 0., 8., -4.]])+level
    query = pd.DataFrame(query_values, columns=names,
                         index=pd.Index(["same", "same"], name="query identity"))
    fit = pca_bootstrap(source, names, components=2, matrix=matrix, anchors=["a", "d"],
                        replications=39, seed=913)
    result = pca_bootstrap_scores(restore_summary(summary_state(fit)), query)
    fitted = [independent_stable_two_axis_scores(values[index], query_values, matrix)
              for index in fit["resample_indices"].to_numpy()]
    expected = np.asarray([entry[0] for entry in fitted])
    point = independent_stable_two_axis_scores(values, query_values, matrix)[0]
    covariance = np.cov(expected, rowvar=False, ddof=1)
    np.testing.assert_allclose(result["replicates"], expected, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(result["point_scores"].to_numpy().ravel(), point, atol=3e-12)
    np.testing.assert_allclose(result["covariance"], covariance, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(result["estimates"].std_error, np.sqrt(covariance.diagonal()), atol=3e-12)
    np.testing.assert_allclose(result["estimates"][["ci_lower", "ci_upper"]],
                               np.quantile(expected, [.025, .975], axis=0, method="linear").T, atol=3e-12)
    np.testing.assert_allclose(result["estimates"].bootstrap_bias, expected.mean(0)-point, atol=3e-12)
    np.testing.assert_array_equal(result["replicate_centering_origins"], [entry[1] for entry in fitted])
    np.testing.assert_allclose(result["replicate_centering_offsets"], [entry[2] for entry in fitted], atol=3e-12)
    assert result.attrs["parameter_order"] == [[i, f"Comp{j+1}"] for i in range(2) for j in range(2)]
    assert result.attrs["parameter_dimension"] == 4
    assert np.max(np.abs(covariance[::2, 1::2])) > .001
    assert result.attrs["query_index_codes"][0] == result.attrs["query_index_codes"][1]
    assert result["estimates"][["df", "p_value"]].isna().all().all()
    assert json.loads(summary_state(result)) == json.loads(summary_state(restore_summary(summary_state(result))))
    if level:
        means = fit["replicate_means"].to_numpy()
        coefficients = fit["replicates"].to_numpy().reshape(39, 2, 13)[:, :, 1:7].transpose(0, 2, 1)
        naive = query_values[None]-means[:, None]
        if matrix == "correlation":
            naive /= fit["replicate_standard_deviations"].to_numpy()[:, None]
        naive = np.matmul(naive, coefficients).reshape(39, 4)
        assert np.max(np.abs(naive-expected)) > 1e-5


@pytest.mark.parametrize("kind", ["column_levels", "missing_count", "draw_index"])
def test_saved_json_numeric_boolean_aliases_are_not_typed_state_equality(kind):
    fit = pca_bootstrap(fixture(), list("abcd"), components=1,
                        replications=19, confidence=.8, seed=15)
    if kind == "column_levels":
        fit.attrs["source_column_nlevels"] = True
    elif kind == "missing_count":
        fit.attrs["n_missing"] = False
    else:
        indices = fit["resample_indices"].astype(object)
        i, j = np.argwhere(indices.to_numpy() == 1)[0]
        indices.iloc[i, j] = True
        fit["resample_indices"] = indices
    with pytest.raises(AnalysisError) as error:
        pca_bootstrap_scores(fit, fixture().iloc[:1])
    assert error.value.code == "invalid_pca_state"


@pytest.mark.parametrize("kind", ["title", "aggregate_axes"])
def test_full_metadata_budget_applies_before_serialization_across_all_tables(monkeypatch, kind):
    from openecon.econometrics.multivariate import pca_score_uncertainty as module
    fit = pca_bootstrap(fixture(), list("abcd"), components=1,
                        replications=19, confidence=.8)
    if kind == "title":
        fit.title = "x"*(4*1024**2+1)
    else:
        for key in ["point_matrix", "point_covariance"]:
            fit[key].index = ["x"*(3*1024**2), "b", "c", "d"]

    def forbidden(*args, **kwargs):
        pytest.fail("Unbounded title/aggregate table metadata reached serialization")

    monkeypatch.setattr(module, "summary_state", forbidden)
    with pytest.raises(AnalysisError) as error:
        pca_bootstrap_scores(fit, fixture().iloc[:1])
    assert error.value.code == "invalid_pca_state"


def test_oversized_decimal_query_identity_refuses_before_scalar_encoding(monkeypatch):
    from openecon.econometrics.multivariate import pca_uncertainty as module
    fit = pca_bootstrap(fixture(), list("abcd"), components=1,
                        replications=19, confidence=.8)
    query = fixture().iloc[:2].copy()
    query.index = pd.Index([Decimal((0, (1,)*20000, 0)), "ordinary"], dtype=object)
    original = module.encode

    def guarded(value):
        if isinstance(value, Decimal) and value.__sizeof__() > 4096:
            pytest.fail("Oversized Decimal identity reached scalar encoding")
        return original(value)

    monkeypatch.setattr(module, "encode", guarded)
    with pytest.raises(AnalysisError) as error:
        pca_bootstrap_scores(fit, query)
    assert error.value.code == "resource_limit"


def test_enormous_saved_integer_refuses_before_generic_serialization(monkeypatch):
    from openecon.econometrics.multivariate import pca_score_uncertainty as module
    fit = pca_bootstrap(fixture(), list("abcd"), components=1,
                        replications=19, confidence=.8)
    fit.attrs["resource_plan"] = {"claimed_bytes": 1 << 100000}

    def forbidden(*args, **kwargs):
        pytest.fail("Unbounded saved integer reached generic serialization")

    monkeypatch.setattr(module, "summary_state", forbidden)
    with pytest.raises(AnalysisError) as error:
        pca_bootstrap_scores(fit, fixture().iloc[:1])
    assert error.value.code == "invalid_pca_state"


@pytest.mark.parametrize('kind', ['grad', 'sparse'])
def test_unsupported_query_tensor_representation_refuses_before_source_replay(monkeypatch, kind):
    import torch
    import openecon.econometrics.multivariate.pca_score_uncertainty as module
    source = fixture()
    fit = pca_bootstrap(source, list('abcd'), components=1, replications=19, confidence=.9)
    query = {name: torch.tensor(source[name].iloc[:2].to_numpy()) for name in 'abcd'}
    query['a'] = query['a'].requires_grad_() if kind == 'grad' else query['a'].to_sparse()
    def forbidden(*args, **kwargs):
        raise AssertionError('Unsupported query representation reached saved source replay')
    monkeypatch.setattr(module, '_validated', forbidden)
    with pytest.raises(AnalysisError) as error:
        pca_bootstrap_scores(fit, query)
    assert error.value.code == 'invalid_data'


def test_workspace_refusal_precedes_any_saved_numeric_tensor_copy(monkeypatch):
    import torch
    from openecon.resources import use_workspace_budget
    source = fixture()
    fit = pca_bootstrap(source, list("abcd"), components=1,
                        replications=19, confidence=.8)

    def forbidden(*args, **kwargs):
        pytest.fail("Saved numerical state reached tensor allocation before workspace admission")

    monkeypatch.setattr(torch, "tensor", forbidden)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        pca_bootstrap_scores(fit, source.iloc[:2])
    assert error.value.code == "workspace_limit"
