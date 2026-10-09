"""Independent saved-VAR proxy moment, normalization and propagation oracles."""

import copy

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.models import ResultBundle
from openecon.resources import use_workspace_budget


def fixture(n=180, p=2, shuffled=False, missing_endpoints=False):
    rng = np.random.default_rng(93771)
    structural = rng.normal(size=(n, 3))
    b = np.array([[1, 0.3, 0.1], [0.6, 1, -0.2], [-0.4, 0.1, 0.8]])
    a1 = np.array([[0.35, 0.1, 0], [0.05, 0.3, -0.05], [0, 0.1, 0.25]])
    a2 = np.diag([0.1, -0.04, 0.08])
    levels = np.zeros((n, 3))
    u = structural @ b.T
    for t in range(2, n):
        levels[t] = a1@levels[t-1] + a2@levels[t-2] + u[t] + [0.1, 0.2, -0.1]
    d = pd.DataFrame(levels, columns=list("abc"))
    d["t"] = np.arange(n)
    if missing_endpoints:
        d.loc[[0, n-1], "a"] = np.nan
    if shuffled:
        d = d.sample(frac=1, random_state=1729).reset_index(drop=True)
    result = oe.var(data=d, y=list("abc"), time="t", lags=p, maxlag=p,
                    irf_steps=1, irf_kinds=["simple"], lm_lags=0,
                    missing="drop" if missing_endpoints else "raise")
    keys = d.iloc[result.sample_positions].t.to_numpy()
    proxy = pd.DataFrame({"t": keys, "z": structural[keys, 0] + .4*rng.normal(size=len(keys))})
    return d, result, proxy


def identify(d, result, proxy, **options):
    args = dict(data=d, proxy_data=proxy, proxy="z", key="t", normalize="a",
                source="Prespecified independent synthetic external-shock fixture",
                exogeneity_assumed=True)
    return oe.proxy_svar(result, **(args | options))


def independent_residuals(d, result):
    order = d.dropna(subset=list("abc")).sort_values("t")
    levels = order[list("abc")].to_numpy()
    p = result.extra["layout"]["lags"]
    n, k = len(levels)-p, levels.shape[1]
    design = np.column_stack([levels[p-j:len(levels)-j, variable] for variable in range(k) for j in range(1, p+1)] + [np.ones(n)])
    coefficients = np.array([c.estimate for c in result.coefficients]).reshape(k, -1)
    residuals = levels[p:] - design@coefficients.T
    a = np.stack([coefficients[:, np.arange(k)*p+lag] for lag in range(p)])
    return residuals, a, order.t.to_numpy()[p:]


@pytest.mark.parametrize("anchor,impact", [("a", 1), ("b", -2)])
@pytest.mark.parametrize("shuffled,missing", [(False, False), (True, False), (True, True)])
def test_complete_proxy_moments_explicit_normalization_and_numpy_irfs(anchor, impact, shuffled, missing, tmp_path, monkeypatch):
    d, var, instrument = fixture(shuffled=shuffled, missing_endpoints=missing)
    result = ResultBundle.model_validate_json(var.model_dump_json())
    monkeypatch.setattr("openecon.econometrics.var.estimators.fit_var", lambda *a, **k: pytest.fail("saved VAR must not refit"))
    output = identify(d, result, instrument.sample(frac=1, random_state=713), normalize=anchor, impact=impact, steps=5)
    u, a, keys = independent_residuals(d, result)
    z = instrument.set_index("t").loc[keys, "z"].to_numpy()
    joined = np.column_stack([z-z.mean(), u-u.mean(0)])
    moments = joined.T@joined/len(z)
    gamma = moments[0, 1:]
    vector = gamma/gamma[list("abc").index(anchor)]*impact
    state = output.attrs["proxy_state"]
    assert_allclose(state["residuals"], u, atol=2e-13)
    assert_allclose(state["complete_joint_centered_moments"], moments, atol=2e-13)
    assert_allclose(state["gamma"], gamma, atol=2e-13)
    assert_allclose(state["impact_vector"], vector, atol=2e-13)
    assert state["sample_keys"] == keys.tolist()
    assert state["sample_positions"] == result.sample_positions
    assert state["source_data_hash"] == result.provenance["data_hash"]
    assert state["instrument_strength_calibrated"] is False
    assert state["source_var_refitted"] is False
    assert state["normalization"]["impact"] == impact
    assert state["inference_available"] is state["fevd_available"] is False
    # Independently propagate a one-time innovation using a companion matrix.
    k, p = 3, 2
    companion = np.zeros((k*p, k*p))
    companion[:k] = np.hstack(a)
    companion[k:, :-k] = np.eye(k*(p-1))
    shock = np.r_[vector, np.zeros(k*(p-1))]
    for h in range(6):
        expected = (np.linalg.matrix_power(companion, h)@shock)[:k]
        assert_allclose(output["responses"].query("horizon == @h").irf, expected, atol=3e-13)
    path = tmp_path/"complete-proxy-state.json"
    path.write_text(oe.summary_state(output))
    restored = oe.restore_summary(path.read_text())
    monkeypatch.setattr("openecon.econometrics.structural.proxy.proxy_svar", lambda *a, **k: pytest.fail("restoration must not identify or fit"))
    replay = oe.proxy_svar_irf(restored, steps=5)
    assert_allclose(replay.irf, output["responses"].irf, atol=0)
    assert len(restored.attrs["proxy_state"]["residuals"]) == result.nobs
    assert "std_error" not in replay and "fevd" not in replay


def test_instrument_scaling_sign_shift_and_permutation_do_not_change_anchored_shock():
    d, result, instrument = fixture()
    original = identify(d, result, instrument, steps=2)
    shifted = instrument.assign(z=lambda df: -7*df.z + 90).sample(frac=1, random_state=718)
    altered = identify(d, result, shifted, steps=2)
    assert_allclose(original["responses"].irf, altered["responses"].irf, atol=1e-13)
    assert original.attrs["proxy_state"]["proxy_data_hash"] != altered.attrs["proxy_state"]["proxy_data_hash"]
    assert original.attrs["proxy_state"]["anchor_correlation"] == pytest.approx(-altered.attrs["proxy_state"]["anchor_correlation"], abs=1e-14)
    reordered = identify(d, result, instrument.iloc[::-1], steps=2)
    assert original.attrs["proxy_state"]["proxy_data_hash"] == reordered.attrs["proxy_state"]["proxy_data_hash"]


@pytest.mark.parametrize("mutation", ["dropped", "extra", "duplicate", "float_keys", "missing_proxy", "infinite_proxy"])
def test_proxy_exact_key_and_complete_sample_refusal(mutation):
    d, result, instrument = fixture()
    if mutation == "dropped":
        instrument = instrument.iloc[1:]
    elif mutation == "extra":
        instrument = pd.concat([instrument, pd.DataFrame({"t": [-1], "z": [1]})])
    elif mutation == "duplicate":
        instrument.iloc[1, 0] = instrument.iloc[0, 0]
    elif mutation == "float_keys":
        instrument["t"] = instrument.t.astype(float)
    else:
        instrument.loc[2, "z"] = np.nan if mutation == "missing_proxy" else np.inf
    with pytest.raises(AnalysisError):
        identify(d, result, instrument)


def test_source_hash_zero_anchor_unknown_normalization_and_assumption_refusals():
    d, result, instrument = fixture()
    changed = d.copy()
    changed.loc[4, "b"] += .1
    with pytest.raises(AnalysisError) as caught:
        identify(changed, result, instrument)
    assert caught.value.code == "sample_mismatch"
    with pytest.raises(AnalysisError) as caught:
        identify(d.iloc[::-1], result, instrument)
    assert caught.value.code == "sample_mismatch"
    for option in [{"normalize": "unknown"}, {"impact": 0}, {"exogeneity_assumed": False}, {"source": ""}, {"key": "other"}]:
        with pytest.raises(AnalysisError):
            identify(d, result, instrument, **option)
    u, _, _ = independent_residuals(d, result)
    rng = np.random.default_rng(719)
    z = rng.normal(size=len(u))
    block = np.column_stack([np.ones(len(u)), u[:, 0]])
    z -= block@np.linalg.lstsq(block, z, rcond=None)[0]
    with pytest.raises(AnalysisError) as caught:
        identify(d, result, instrument.assign(z=z))
    assert caught.value.code == "unidentified_normalization"
    with pytest.raises(AnalysisError) as caught:
        identify(d, result, instrument.assign(z=1))
    assert caught.value.code == "unidentified_normalization"


def test_resource_dataset_and_saved_schema_refusals(monkeypatch):
    d, result, instrument = fixture()
    source = Dataset.from_frame(d)
    monkeypatch.setattr(source, "iter_batches", lambda *a, **k: pytest.fail("Dataset must not read"))
    with pytest.raises(AnalysisError) as caught:
        identify(source, result, instrument)
    assert caught.value.code == "streaming_unsupported"
    with pytest.raises(AnalysisError) as caught:
        identify(d, result, instrument, max_work=1)
    assert caught.value.code == "work_limit"
    with pytest.raises(AnalysisError) as caught:
        identify(d, result, instrument, steps=201)
    assert caught.value.code == "invalid_option"
    with pytest.raises(AnalysisError) as caught:
        identify(d, result, instrument.assign(z=instrument.z*1e300))
    assert caught.value.code == "numerical_failure"
    large, large_result, large_instrument = fixture(n=1200)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as caught:
        identify(large, large_result, large_instrument)
    assert caught.value.code == "workspace_limit"
    output = identify(d, result, instrument)
    bad = copy.deepcopy(output)
    bad.attrs["proxy_state"]["a"][0][0] = [1]
    with pytest.raises(AnalysisError) as caught:
        oe.proxy_svar_irf(bad)
    assert caught.value.code == "invalid_state"
    with pytest.raises(AnalysisError) as caught:
        oe.proxy_svar_irf(output, steps=100, max_work=1)
    assert caught.value.code == "work_limit"


def test_explicit_cpu_domain_under_foreign_default_device():
    import torch

    d, result, instrument = fixture()
    with torch.device("meta"):
        output = identify(d, result, instrument, steps=2)
        replay = oe.proxy_svar_irf(output, steps=2)
    assert_allclose(replay.irf, output["responses"].irf, atol=0)
