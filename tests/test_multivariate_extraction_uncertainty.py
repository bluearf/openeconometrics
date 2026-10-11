"""Independent five-method bootstrap oracles and strict refusal contracts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import minimize

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.core import TableSet
from openecon.econometrics.multivariate import extraction_uncertainty as u
from openecon.resources import use_workspace_budget


METHODS = ("ml", "minres", "alpha", "image_covariance", "generalized_image")


def data(n=280, seed=1011):
    rng = np.random.default_rng(seed)
    loading = np.array([-.72, .81, .64, .75, .68])
    values = rng.normal(size=n)[:, None] * loading
    values += rng.normal(size=(n, 5)) * np.sqrt(1-loading**2)
    return pd.DataFrame(values * [2, 3, 1, 4, .7] + [5, -2, 12, .5, 3], columns=list("abcde"))


def correlation(values):
    shifted = values-values[0]
    centred = shifted-shifted.mean(0)
    covariance = centred.T @ centred/(len(values)-1)
    scale = np.sqrt(np.diag(covariance))
    result = covariance/np.outer(scale, scale)
    np.fill_diagonal(result, 1.)
    return result


def independent_parameters(values, method, anchor):
    """NumPy definitions and independently converged SciPy fits; no native fit."""
    r = correlation(values)
    p = len(r)
    inverse = np.linalg.inv(r)
    conditional = 1/np.diag(inverse)
    smc = 1-conditional
    reduced = r.copy()
    np.fill_diagonal(reduced, smc)
    roots, vectors = np.linalg.eigh(reduced)
    start = vectors[:, -1]*np.sqrt(roots[-1])
    if method == "ml":
        def objective(theta):
            free = np.exp(theta)
            psi = .005+free
            weights = 1/np.sqrt(psi)
            eig, vec = np.linalg.eigh(r*np.outer(weights, weights))
            tail, directions = eig[:-1], vec[:, :-1]
            loss = np.sum(tail-np.log(tail)-1)
            gradient = -(free/psi)*(directions**2 @ (tail-1))
            return loss, gradient

        psi0 = np.maximum((1-.5/p)*conditional, .01)
        fit = minimize(objective, np.log(psi0-.005), jac=True, method="L-BFGS-B",
                       options={"gtol": 1e-11, "ftol": 1e-15, "maxiter": 2000, "maxls": 100})
        assert fit.success and fit.status == 0, (fit.status, fit.message, fit.fun)
        assert np.max(np.abs(objective(fit.x)[1])) < 1e-6
        uniqueness = .005+np.exp(fit.x)
        weights = 1/np.sqrt(uniqueness)
        roots, vectors = np.linalg.eigh(r*np.outer(weights, weights))
        loading = np.sqrt(uniqueness)*vectors[:, -1]*np.sqrt(roots[-1]-1)
    elif method == "minres":
        def objective(a):
            residual = np.outer(a, a)-r
            np.fill_diagonal(residual, 0.)
            return np.sum(residual**2)/2, 2*residual @ a

        fit = minimize(objective, start, jac=True, method="L-BFGS-B",
                       options={"gtol": 1e-11, "ftol": 1e-15, "maxiter": 2000, "maxls": 100})
        assert fit.success and fit.status == 0, (fit.status, fit.message, fit.fun)
        assert np.max(np.abs(objective(fit.x)[1])) < 1e-7
        loading = fit.x
        uniqueness = 1-loading**2
    elif method == "alpha":
        h = smc.copy()
        for _ in range(2000):
            scaled = (r-np.eye(p))/np.sqrt(np.outer(h, h))+np.eye(p)
            roots, vectors = np.linalg.eigh(scaled)
            loading = np.sqrt(h)*vectors[:, -1]*np.sqrt(roots[-1])
            updated = loading**2
            if np.max(np.abs(updated-h)) <= 1e-12:
                break
            h = updated
        else:
            raise AssertionError("Independent alpha fixed-point iteration failed")
        uniqueness = 1-loading**2
    elif method == "image_covariance":
        coefficients = np.eye(p)-inverse*conditional[None, :]
        np.fill_diagonal(coefficients, 0.)
        image = coefficients.T @ r @ coefficients
        roots, vectors = np.linalg.eigh((image+image.T)/2)
        loading = vectors[:, -1]*np.sqrt(roots[-1])
        uniqueness = 1-loading**2
    else:
        s = np.sqrt(conditional)
        roots, vectors = np.linalg.eigh(r/np.outer(s, s))
        loading = s*vectors[:, -1]*(roots[-1]-1)/np.sqrt(roots[-1])
        uniqueness = 1-loading**2
    loading *= 1 if loading[anchor] > 0 else -1
    return np.r_[loading, uniqueness]


def independent_indices(n, replications, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return np.array([torch.randint(n, (n,), generator=generator).numpy() for _ in range(replications)])


def portable_pack(result):
    return {"summary": oe.summary_state(result), "latex": result.to_latex(),
            "table_order": list(result),
            "table_dtypes": {name: [str(dtype) for dtype in frame.dtypes]
                             for name, frame in result.items()}}


@pytest.mark.parametrize("method", ["gls", "generalized_image"])
def test_new_extraction_uses_cpu_under_ambient_non_cpu_default(method):
    frame = data(n=120)
    expected = oe.factor(frame, list(frame), method=method, factors=1)
    summary_options = dict(n=len(frame), columns=list(frame), matrix="covariance",
                           means=frame.mean(), sds=frame.std(ddof=1), method=method, factors=1)
    expected_summary = oe.factor_matrix(frame.cov(), **summary_options)
    query = frame.iloc[[7, 2, 7, 11]].copy()
    query.iloc[1, 0] = np.nan
    expected_scores = oe.factor_scores(expected, query)
    expected_summary_scores = oe.factor_scores(expected_summary, query)
    with torch.device("meta"):
        actual = oe.factor(frame, list(frame), method=method, factors=1)
        actual_summary = oe.factor_matrix(frame.cov(), **summary_options)
        actual_scores = oe.factor_scores(actual, query)
        actual_summary_scores = oe.factor_scores(actual_summary, query)
    for actual_fit, expected_fit in ((actual, expected), (actual_summary, expected_summary)):
        assert actual_fit.attrs == expected_fit.attrs
        assert actual_fit.attrs["device"] == "cpu"
        for name in expected_fit:
            pd.testing.assert_frame_equal(actual_fit[name], expected_fit[name])
    pd.testing.assert_frame_equal(actual_scores, expected_scores)
    pd.testing.assert_frame_equal(actual_summary_scores, expected_summary_scores)


@pytest.mark.parametrize("method", METHODS)
def test_fixed_anchor_orients_complete_point_tables_and_saved_scores(method):
    frame = data(n=140)
    native_options = {"tolerance": 1e-10} if method == "alpha" else {}
    native = oe.factor(frame, list(frame), method=method, factors=1, **native_options)
    assert native["loadings"].loc["a", "Factor1"] < 0
    result = oe.factor_bootstrap(frame, list(frame), method=method, replications=19,
                                 confidence=.9, seed=33, anchor="a")
    loading = result["point_loadings"].to_numpy()
    np.testing.assert_array_equal(loading[:, 0], result["estimates"].estimate.iloc[:5])
    np.testing.assert_array_equal(result["point_uniqueness"].iloc[:, 0],
                                  result["estimates"].estimate.iloc[5:])
    np.testing.assert_array_equal(loading, -native["loadings"].to_numpy())
    np.testing.assert_array_equal(result["point_score_coefficients"], -native["score_coefficients"])
    np.testing.assert_allclose(loading @ loading.T, native["loadings"] @ native["loadings"].T,
                               atol=1e-15, rtol=1e-15)
    attrs = result.attrs["point_fit_attrs"]
    assert attrs["sign_anchor"] == "a" and attrs["native_to_anchor_sign"] == -1
    assert attrs["sign_convention"] == "fixed anchor loading positive"
    assert attrs["rotate"] is None
    point_fit = TableSet({name.removeprefix("point_"): table.copy()
                          for name, table in result.items()
                          if name.startswith("point_") and name != "point_diagnostics"},
                         title="Saved anchored point fit", **attrs)
    restored = oe.restore_summary(oe.summary_state(point_fit))
    query = frame.iloc[[8, 1, 8, 12]].copy()
    query.iloc[1, 0] = np.nan
    reference = -oe.factor_scores(native, query)
    pd.testing.assert_frame_equal(oe.factor_scores(restored, query), reference)


@pytest.mark.parametrize("method", ["ml", "minres"])
def test_curvature_and_complete_bootstrap_ignore_ambient_inference_mode(method):
    frame = data(n=140)
    options = dict(method=method, replications=19, confidence=.9, seed=33, anchor="a")
    expected = oe.factor_bootstrap(frame, list(frame), **options)
    with torch.inference_mode():
        actual = oe.factor_bootstrap(frame, list(frame), **options)
    assert portable_pack(actual) == portable_pack(expected)


@pytest.mark.parametrize("method", ["ml", "minres"])
def test_regularity_uses_native_sscp_correlation_without_covariance_underflow(method):
    frame = data(n=120)*1e-160
    native = oe.factor(frame, list(frame), method=method, factors=1)
    point, _, fitted, diagnostics = u._parameters(frame, list(frame), "a", method)
    np.testing.assert_array_equal(point[:5], -native["loadings"].iloc[:, 0])
    np.testing.assert_array_equal(point[5:], native["uniqueness"].iloc[:, 0])
    assert diagnostics[4] > 1e-8
    assert diagnostics[5] <= (1e-6 if method == "ml" else 1e-7)
    np.testing.assert_array_equal(fitted["loadings"].iloc[:, 0], point[:5])


@pytest.mark.parametrize("method", METHODS)
def test_five_methods_every_draw_full_covariance_and_marginal_intervals(method):
    frame = data()
    result = oe.factor_bootstrap(frame, list(frame), method=method, replications=39,
                                 confidence=.9, seed=614, anchor="a")
    indices = independent_indices(len(frame), 39, 614)
    reference = np.array([independent_parameters(frame.to_numpy()[index], method, 0) for index in indices])
    point = independent_parameters(frame.to_numpy(), method, 0)
    tolerance = 3e-6 if method in ("ml", "minres") else 3e-9
    np.testing.assert_array_equal(result["replicate_indices"], indices)
    np.testing.assert_allclose(result["replicates"], reference, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(result["estimates"].estimate, point, atol=tolerance, rtol=tolerance)
    covariance = np.cov(reference, rowvar=False, ddof=1)
    np.testing.assert_allclose(result["covariance"], covariance, atol=tolerance*1e-3, rtol=tolerance*100)
    np.testing.assert_allclose(result["estimates"].std_error, np.sqrt(covariance.diagonal()), atol=tolerance)
    np.testing.assert_allclose(result["estimates"][["ci_lower", "ci_upper"]],
                               np.quantile(reference, [.05, .95], axis=0, method="linear").T,
                               atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(result["estimates"].bootstrap_bias, reference.mean(0)-point, atol=2*tolerance)
    assert result.attrs["successful_replications"] == 39
    assert result.attrs["failed_replications"] == []
    assert result.attrs["covariance_divisor"] == 38
    assert result.attrs["method"] == method and result.attrs["factors"] == 1
    assert (result["replicates"]["loading:a"] > 0).all()
    assert not result.attrs["p_values_available"] and not result.attrs["inference_df_available"]
    assert result["estimates"][["p_value", "df"]].isna().all().all()
    assert (result["replicate_diagnostics"].local_chart_margin > 1e-8).all()


@pytest.mark.parametrize("method", METHODS)
def test_optimizer_free_portable_restoration_retains_full_result(method, monkeypatch):
    result = oe.factor_bootstrap(data(n=140), list("abcde"), method=method,
                                 replications=39, confidence=.9, seed=33)
    payload = json.loads(json.dumps(portable_pack(result), allow_nan=False))

    def forbidden(*args, **kwargs):
        raise AssertionError("portable restoration attempted extraction or fitting")

    monkeypatch.setattr(u, "factor", forbidden)
    monkeypatch.setattr(u, "_parameters", forbidden)
    restored = oe.restore_summary(payload["summary"])
    for name, types in payload["table_dtypes"].items():
        restored[name] = restored[name].astype(dict(zip(restored[name].columns, types, strict=True)))
    restored = TableSet({name: restored[name] for name in payload["table_order"]},
                        title=restored.title, **restored.attrs)
    assert portable_pack(restored) == payload
    assert restored.attrs == result.attrs
    assert set(restored) == set(result)
    for name in result:
        # JSON null has either None or NaN as its frame representation; all
        # finite numbers, labels and frame attributes must remain exact.
        actual = restored[name].astype(object).where(restored[name].notna(), None)
        expected = result[name].astype(object).where(result[name].notna(), None)
        pd.testing.assert_frame_equal(actual, expected)


def test_seed_is_deterministic_and_global_rng_is_unchanged():
    frame = data(n=120)
    torch_before, numpy_before = torch.random.get_rng_state().clone(), np.random.get_state()
    one = oe.factor_bootstrap(frame, list(frame), method="generalized_image", replications=39, seed=43)
    two = oe.factor_bootstrap(frame, list(frame), method="generalized_image", replications=39, seed=43)
    other = oe.factor_bootstrap(frame, list(frame), method="generalized_image", replications=39, seed=44)
    assert torch.equal(torch_before, torch.random.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and numpy_before[2:] == numpy_after[2:]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    for name in one:
        pd.testing.assert_frame_equal(one[name], two[name])
    assert not one["replicate_indices"].equals(other["replicate_indices"])
    assert not one["replicates"].equals(other["replicates"])


def test_missing_physical_positions_duplicate_typed_identities_and_source_preserved():
    from openecon.econometrics.postest.index_codec import decode

    frame = data(n=120)
    frame.index = pd.Index([("cohort", 7), ("cohort", 7), 9, "9", None, *range(115)], dtype=object)
    frame.iloc[7, 1], frame.iloc[19, 3] = np.nan, np.nan
    frame["unused"] = "untouched"
    original = frame.copy(deep=True)
    result = oe.factor_bootstrap(frame, list("abcde"), method="image_covariance", replications=39, seed=12)
    clean = frame[list("abcde")].dropna()
    expected = oe.factor_bootstrap(clean, list("abcde"), method="image_covariance", replications=39,
                                   seed=12, anchor=result.attrs["sign_anchor"])
    for name in ("estimates", "covariance", "replicates", "replicate_indices", "replicate_diagnostics"):
        pd.testing.assert_frame_equal(result[name], expected[name])
    pd.testing.assert_frame_equal(frame, original)
    positions = np.flatnonzero(frame[list("abcde")].notna().all(axis=1)).tolist()
    assert result.attrs["sample_positions"] == positions
    assert list(result["sample"].index) == positions
    assert result.attrs["n"] == 118 and result.attrs["n_missing"] == 2
    identities = result.attrs["source_identities"][0]
    assert [decode(code) for code in identities[:5]] == [("cohort", 7), ("cohort", 7), 9, "9", None]
    assert identities[2] != identities[3]
    with pytest.raises(AnalysisError, match="missing"):
        oe.factor_bootstrap(frame, list("abcde"), method="image_covariance", replications=39, missing="raise")


def test_every_failed_refit_retains_full_denominator_indices_and_successes(monkeypatch):
    original, calls = u._parameters, []
    frame = data(n=90)

    def refuse_selected(*args, **kwargs):
        calls.append(len(calls))
        if len(calls) in (3, 9):
            raise AnalysisError("unidentified_factor", "declared singular extraction chart")
        return original(*args, **kwargs)

    monkeypatch.setattr(u, "_parameters", refuse_selected)
    with pytest.raises(AnalysisError, match="2 of 39") as caught:
        oe.factor_bootstrap(frame, list(frame), method="image_covariance", replications=39,
                            confidence=.9, seed=17, anchor="a")
    error = caught.value
    assert len(calls) == 40
    assert error.code == "bootstrap_failure"
    assert error.replications_attempted == 39 and error.successful_replications == 37
    assert [item["replication"] for item in error.failures] == [2, 8]
    indices = independent_indices(len(frame), 39, 17)
    np.testing.assert_array_equal(error.replicate_indices, indices)
    success_ids = [i for i in range(1, 40) if i not in (2, 8)]
    assert error.successful_positions == success_ids
    reference = [independent_parameters(frame.to_numpy()[indices[i-1]], "image_covariance", 0)
                 for i in success_ids]
    np.testing.assert_allclose(error.successful_vectors, reference, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("matrix", [torch.zeros((3, 3)), torch.diag(torch.tensor([1., 0., 1.])),
                                    torch.diag(torch.tensor([1., -1., 1.])), torch.full((3, 3), torch.nan)])
def test_zero_saddle_and_nonfinite_curvature_refused(matrix):
    with pytest.raises(AnalysisError, match="Hessian|local minimum"):
        u._positive_curvature(matrix.to(dtype=torch.float64))


@pytest.mark.parametrize("method", ["ml", "minres", "alpha"])
def test_regular_exact_one_factor_with_repeated_discarded_roots(method):
    # A deterministic raw sample with exact compound-symmetric correlation.
    n, p = 40, 5
    q, _ = np.linalg.qr(np.column_stack([np.ones(n), *[np.arange(n, dtype=float)**j for j in range(1, p+1)]]))
    r = .6*np.ones((p, p))+.4*np.eye(p)
    values = q[:, 1:p+1] @ np.linalg.cholesky(r).T*np.sqrt(n-1)
    point, _, fitted, diagnostics = u._parameters(pd.DataFrame(values, columns=list("abcde")),
                                                 list("abcde"), "a", method)
    np.testing.assert_allclose(point[:p], np.sqrt(.6), atol=3e-7)
    np.testing.assert_allclose(point[p:], .4, atol=3e-7)
    assert diagnostics[4] > 1e-8 and fitted.attrs["factors"] == 1


def test_alpha_fixed_point_margin_matches_independent_central_difference():
    frame = data(n=200)
    _, _, fitted, diagnostics = u._parameters(frame, list(frame), "a", "alpha")
    h = np.array(fitted.attrs["last_iteration_communalities"])
    r, p = correlation(frame.to_numpy()), len(h)

    def update(value):
        roots, vectors = np.linalg.eigh((r-np.eye(p))/np.sqrt(np.outer(value, value))+np.eye(p))
        return value*vectors[:, -1]**2*roots[-1]

    numeric = np.empty((p, p))
    for j in range(p):
        delta = np.zeros(p)
        delta[j] = 1e-6
        numeric[:, j] = (update(h+delta)-update(h-delta))/2e-6
    singular = np.linalg.svd(np.eye(p)-numeric, compute_uv=False)
    assert diagnostics[4] == pytest.approx(singular[-1]/max(singular[0], 1.), rel=2e-7, abs=2e-9)


@pytest.mark.parametrize("method", ["image_covariance", "generalized_image"])
def test_nonseparated_retained_image_root_refuses(method):
    r = torch.eye(5, dtype=torch.float64)
    fake = TableSet({"eigenvalues": pd.DataFrame({"eigenvalue": [2., 2., 1., .5, .2]})}, method=method)
    with pytest.raises(AnalysisError, match="identified|separated"):
        u._regularity(fake, r, torch.ones(5, dtype=torch.float64)*.5, torch.ones(5, dtype=torch.float64)*.75)


def test_admission_limits_precede_conversion_and_fitting(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("source conversion or fit occurred before admission")

    monkeypatch.setattr(u, "_selected_input", forbidden)
    monkeypatch.setattr(u, "_parameters", forbidden)
    with pytest.raises(AnalysisError, match="10000"):
        oe.factor_bootstrap({name: range(10001) for name in "abcde"}, list("abcde"), method="alpha")
    names = [f"v{i}" for i in range(17)]
    with pytest.raises(AnalysisError, match="16"):
        oe.factor_bootstrap({name: range(50) for name in names}, names, method="minres")
    names = names[:16]
    with pytest.raises(AnalysisError, match="work"):
        oe.factor_bootstrap({name: range(10000) for name in names}, names, method="ml", replications=1999)
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        oe.factor_bootstrap(data(n=100), list("abcde"), method="image_covariance")
    names = ["x"*4097, "b", "c", "d", "e"]
    with pytest.raises(AnalysisError, match="4096|label|name"):
        oe.factor_bootstrap({name: range(40) for name in names}, names, method="image_covariance")


@pytest.mark.parametrize("options", [{"method": "gls"}, {"method": "pcf"}, {"method": "bad"},
                                    {"replications": 18}, {"confidence": 1.}, {"seed": -1},
                                    {"anchor": "absent"}, {"missing": "pairwise"}])
def test_invalid_method_or_sampling_options_refuse(options):
    with pytest.raises(AnalysisError):
        oe.factor_bootstrap(data(n=50), list("abcde"), **{"method": "image_covariance", **options})


def test_dataset_refusal_never_reads_stream():
    calls = []

    def batches():
        calls.append(True)
        yield data()

    source = Dataset.from_batches(batches, columns=list("abcde"), row_count=280)
    with pytest.raises(AnalysisError, match="resident"):
        oe.factor_bootstrap(source, list("abcde"), method="alpha")
    assert calls == []
