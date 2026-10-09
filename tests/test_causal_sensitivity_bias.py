"""Independent vertex-support and Gaussian coverage checks for fixed bias boxes."""

import itertools
import json

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.causal_design.bias import bias_sensitivity
from openecon.econometrics.causal_design.common import causal_design_load, causal_design_save
from openecon.resources import use_workspace_budget


def sample():
    return pd.DataFrame({"b": [1.4, -.3, 2.0], "l": [.2, -1., .5],
                         "lo": [-.1, -.5, 0.], "hi": [.8, .1, .3]},
                        index=["same", "same", "last"])


def fit(data=None, covariance=None, **kwargs):
    return bias_sensitivity(sample() if data is None else data, "b", "l", "lo", "hi",
                            covariance=[[.3, .05, -.02], [.05, .2, .01], [-.02, .01, .4]]
                            if covariance is None else covariance,
                            restriction="fixed_external", **kwargs)


@pytest.mark.parametrize("seed", range(12))
def test_complete_covariance_and_box_vertices(seed):
    rng = np.random.default_rng(seed)
    data = sample()
    matrix = rng.normal(size=(3, 3))
    cov = matrix @ matrix.T + np.eye(3) * .1
    data["l"] = rng.normal(size=3)
    result = fit(data, cov.tolist())
    coef, b = data.l.to_numpy(), data.b.to_numpy()
    se = np.sqrt(coef @ cov @ coef)
    vertices = np.array(list(itertools.product(*zip(data.lo, data.hi))))
    support = vertices @ coef
    est = coef @ b
    z = norm.isf(.025)
    for row in result["sensitivity"].itertuples():
        lower, upper = row.scale * support.min(), row.scale * support.max()
        assert_allclose([row.bias_lower, row.bias_upper, row.identified_lower,
                         row.identified_upper, row.ci_lower, row.ci_upper],
                        [lower, upper, est-upper, est-lower, est-upper-z*se, est-lower+z*se],
                        rtol=3e-13, atol=3e-14)
    assert_allclose(result["contrast"].iloc[0][["estimate", "variance", "se", "p_zero_bias"]].astype(float),
                    [est, coef @ cov @ coef, se, 2*norm.sf(abs(est/se))], rtol=5e-13)
    assert result.attrs["state"]["covariance"] == cov.tolist()
    assert result.attrs["state"]["covariance_order"] == [0, 1, 2]
    assert result.attrs["unit_labels"][0] == result.attrs["unit_labels"][1]


def test_off_diagonal_covariance_changes_interval_width():
    positive = fit(covariance=[[1., .5, 0.], [.5, 1., 0.], [0., 0., 1.]])
    negative = fit(covariance=[[1., -.5, 0.], [-.5, 1., 0.], [0., 0., 1.]])
    assert positive["contrast"].iloc[0].se < negative["contrast"].iloc[0].se


def test_exact_gaussian_union_coverage_for_every_vertex_and_interior():
    result = fit(scales=[1.0], level=.9)
    row = result["sensitivity"].iloc[0]
    se = result["contrast"].iloc[0].se
    center = result["contrast"].iloc[0].estimate
    # Distribution of estimate error has mean l'bias. The union contains the
    # true target iff -upper_extent <= normal_error <= -lower_extent.
    for bias in itertools.product(*[(a, (a+b)/2, b) for a, b in zip(sample().lo, sample().hi)]):
        mean = np.dot(sample().l, bias)
        upper_error = center - row.ci_lower
        lower_error = center - row.ci_upper
        coverage = norm.cdf((upper_error-mean)/se) - norm.cdf((lower_error-mean)/se)
        assert coverage >= .9 - 1e-14


@pytest.mark.parametrize("sign", [1, -1])
def test_tipping_scale_reaches_zero_without_grid_rounding(sign):
    data = pd.DataFrame({"b": [sign*3.], "l": [1.], "lo": [-.6], "hi": [.4]})
    result = fit(data, [[.04]], scales=[0., 1.])
    state = result.attrs["state"]
    idtip, citip = state["identification_tipping_scale"], state["interval_tipping_scale"]
    row = fit(data, [[.04]], scales=[citip, idtip])["sensitivity"]
    assert abs(row.iloc[0]["ci_lower" if sign > 0 else "ci_upper"]) < 2e-15
    assert abs(row.iloc[1]["identified_lower" if sign > 0 else "identified_upper"]) < 2e-15


def test_no_finite_tipping_and_deterministic_covariance():
    data = pd.DataFrame({"b": [2.], "l": [1.], "lo": [-1.], "hi": [0.]})
    result = fit(data, [[0.]])
    assert result.attrs["state"]["identification_tipping_scale"] is None
    assert result.attrs["state"]["interval_tipping_scale"] is None
    assert result.attrs["state"]["standard_error"] == 0
    assert pd.isna(result["contrast"].iloc[0].z)
    data["b"] = 0.
    zero = fit(data, [[0.]])
    assert zero.attrs["state"]["interval_tipping_scale"] == 0


def test_genuinely_deterministic_singular_contrast():
    data = sample().iloc[:2].copy()
    data["l"] = [1., -1.]
    result = fit(data, [[1., 1.], [1., 1.]])
    assert result["contrast"].iloc[0].variance == 0


def test_complete_json_roundtrip_and_tamper_rejected():
    result = fit()
    artifact = json.loads(json.dumps(causal_design_save(result), allow_nan=False))
    restored = causal_design_load(artifact)
    assert causal_design_save(restored) == artifact
    for key in result:
        pd.testing.assert_frame_equal(result[key], restored[key], check_exact=True)
    artifact["payload"]["attrs"]["state"]["covariance"][0][1] += .1
    with pytest.raises(AnalysisError, match="checksum"):
        causal_design_load(artifact)


@pytest.mark.parametrize("covariance", [[], [[1.]], [[1., 2., 0.], [2., 1., 0.], [0., 0., 1.]],
                                      [[1., .2, 0.], [.1, 1., 0.], [0., 0., 1.]],
                                      [[True, 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                                      [[np.inf, 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
def test_invalid_covariance_is_not_repaired(covariance):
    with pytest.raises(AnalysisError):
        fit(covariance=covariance)


@pytest.mark.parametrize("kwargs", [{"device": "cuda"}, {"device": "mps"}, {"weights": "w"},
                                    {"missing": "drop"}, {"scales": [1., 0.]},
                                    {"scales": [0., 0.]}, {"scales": [-1.]},
                                    {"scales": [True]}, {"level": 1.}, {"max_work": 1}])
def test_invalid_contracts(kwargs):
    with pytest.raises(AnalysisError):
        fit(**kwargs)


def test_declaration_and_role_and_missing_guards():
    with pytest.raises(AnalysisError):
        bias_sensitivity(sample(), "b", "l", "lo", "hi", covariance=np.eye(3).tolist())
    data = sample()
    data.loc[data.index == "last", "hi"] = np.nan
    with pytest.raises(AnalysisError, match="missing"):
        fit(data)
    data = sample()
    data["lo"] = data.hi + 1
    with pytest.raises(AnalysisError, match="lower"):
        fit(data)
    data["l"] = 0.
    with pytest.raises(AnalysisError):
        fit(data)


def test_workspace_limit_rejects_full_result():
    data = pd.DataFrame({"b": np.ones(64), "l": np.ones(64), "lo": np.zeros(64), "hi": np.ones(64)})
    with use_workspace_budget(1), pytest.raises(AnalysisError):
        fit(data, np.eye(64).tolist())


def test_nonzero_variance_underflow_is_refused():
    data = pd.DataFrame({"b": [1.], "l": [1e-100], "lo": [0.], "hi": [1.]})
    with pytest.raises(AnalysisError, match="variance"):
        fit(data, [[1e-200]])


def test_tiny_representable_variance_is_retained():
    data = pd.DataFrame({"b": [0.], "l": [1.], "lo": [0.], "hi": [1.]})
    result = fit(data, [[1e-300]], scales=[0.])
    assert_allclose(result.attrs["state"]["contrast_variance"], 1e-300, rtol=1e-14, atol=0)


def test_singular_covariance_nonzero_active_variance_underflow_is_refused():
    data = pd.DataFrame({"b": [0., 0.], "l": [1e-200, 1.], "lo": [0., 0.], "hi": [0., 0.]})
    with pytest.raises(AnalysisError, match="underflow"):
        fit(data, [[1., 0.], [0., 0.]])


def test_positive_tipping_ratio_cannot_turn_into_false_zero():
    data = pd.DataFrame({"b": [1e-250], "l": [1.], "lo": [0.], "hi": [1e150]})
    with pytest.raises(AnalysisError, match="tipping"):
        fit(data, [[0.]], scales=[0., 1.])


@pytest.mark.parametrize("ordering", list(itertools.permutations(range(3))))
def test_small_nonzero_contrast_survives_large_cancellation(ordering):
    data = pd.DataFrame({"b": [1e16, -1e16, 1.], "l": [1., 1., 1.], "lo": [0., 0., 0.], "hi": [0., 0., 0.]})
    result = fit(data.iloc[list(ordering)], [[0., 0., 0.]] * 3)
    assert result.attrs["state"]["contrast_estimate"] == 1.
    assert result["sensitivity"].iloc[0].ci_lower == 1.
    assert result.attrs["state"]["identification_tipping_scale"] is None


@pytest.mark.parametrize("variance,lower,upper", [(1., 0., 0.), (0., -1., 1.)])
def test_nonzero_uncertainty_or_bias_endpoints_cannot_collapse(variance, lower, upper):
    data = pd.DataFrame({"b": [1e150], "l": [1.], "lo": [lower], "hi": [upper]})
    with pytest.raises(AnalysisError, match="endpoint shift"):
        fit(data, [[variance]], scales=[0., 1.])


def test_gaussian_endpoint_preserves_all_three_contributions_during_cancellation():
    import math
    from openecon.engines.distributions import normal_isf

    critical = normal_isf(.025)
    variance = (1e16 / critical) ** 2
    data = pd.DataFrame({"b": [1.], "l": [1.], "lo": [1e16], "hi": [1e16]})
    result = fit(data, [[variance]], scales=[1.])
    radius = result.attrs["state"]["critical"] * result.attrs["state"]["standard_error"]
    assert result["sensitivity"].iloc[0].ci_upper == math.fsum([1., -1e16, radius])


def test_standard_error_underflow_cannot_turn_positive_quadratic_into_deterministic():
    data = pd.DataFrame({"b": [0.], "l": [5e-324], "lo": [0.], "hi": [0.]})
    with pytest.raises(AnalysisError, match="variance"):
        fit(data, [[1e-300]], scales=[0.])
