"""Original-paper restrictions/moments evaluated with independent NumPy QR."""

import json
import math

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.stats import chi2, f, norm
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.var.frequency_causality import bccaustest
from openecon.econometrics.var.panel_causality import dhcausality


@pytest.fixture(autouse=True, scope="module")
def owned_test_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def sample(units=12, periods=120, seed=110):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(units, periods + 100, 2))
    for t in range(2, periods + 100):
        values[:, t, 0] += .1 + .35 * values[:, t-1, 0] - .12 * values[:, t-2, 0] \
            + .3 * values[:, t-1, 1] - .1 * values[:, t-2, 1]
        values[:, t, 1] += .2 + .4 * values[:, t-1, 1]
    frame = pd.DataFrame(values[:, -periods:].reshape(-1, 2), columns=["y", "x"])
    frame["unit"] = np.repeat(np.arange(units), periods)
    frame["t"] = np.tile(np.arange(periods), units)
    return frame


def independent_fit(frame, p, constant=True):
    """One equation QR using original uncentered inputs, independent of Torch."""
    a = frame[["y", "x"]].to_numpy()
    n = len(a)
    z = np.column_stack([a[p-j:n-j, v] for v in (0, 1) for j in range(1, p+1)])
    if constant:
        z = np.column_stack([z, np.ones(n-p)])
    q, r = np.linalg.qr(z)
    beta = np.linalg.solve(r, q.T @ a[p:])
    u = a[p:] - z @ beta
    ri = np.linalg.solve(r, np.eye(r.shape[0]))
    covariance = (u[:, 0] @ u[:, 0] / (len(u) - z.shape[1])) * ri @ ri.T
    return z, a[p:, 0], beta[:, 0], covariance, u[:, 0]


@pytest.mark.parametrize("p", [3, 4, 6])
@pytest.mark.parametrize("constant", [True, False])
def test_frequency_restrictions_match_paper_qr_and_constrained_projection(p, constant):
    frame = sample(units=1)
    omegas = [.03, .25, 1., 2., math.pi - .04]
    result = bccaustest(data=frame, y="y", x="x", lags=p,
                       frequencies=omegas, constant=constant, time="t")
    z, outcome, beta, covariance, residual = independent_fit(frame, p, constant)
    for i, row in result["tests"].iterrows():
        restriction = np.zeros((2, z.shape[1]))
        restriction[:, p:2*p] = [np.cos(np.arange(1, p+1) * row.frequency),
                                 np.sin(np.arange(1, p+1) * row.frequency)]
        rb = restriction @ beta
        wald = rb @ np.linalg.solve(restriction @ covariance @ restriction.T, rb)
        # SVD null-space constrained fit -> original ordinary F test. This
        # independent formulation does not use coefficient covariance at all.
        _, _, vh = np.linalg.svd(restriction, full_matrices=True)
        reduced = z @ vh[2:].T
        qr, _ = np.linalg.qr(reduced)
        ur = outcome - qr @ (qr.T @ outcome)
        df = len(outcome) - z.shape[1]
        ordinary_f = ((ur @ ur - residual @ residual)/2) / (residual @ residual / df)
        assert_allclose(row.statistic, wald/2, rtol=2e-9, atol=2e-10)
        assert_allclose(row.statistic, ordinary_f, rtol=2e-9, atol=2e-10)
        assert_allclose(row.p_value, f.sf(ordinary_f, 2, df), rtol=2e-8, atol=2e-10)
        assert row.df_den == df
        observed = result["restrictions"].query("frequency == @row.frequency")
        assert_allclose(observed[["cosine", "sine"]].to_numpy().T, restriction[:, p:2*p])
    assert_allclose(result["coefficients"].query("equation == 'y'")["estimate"], beta,
                    rtol=2e-9, atol=2e-10)
    assert_allclose(result["coefficients"].query("equation == 'y'")["std_error"],
                    covariance.diagonal()**.5, rtol=2e-9)
    json.dumps(result.attrs, allow_nan=False)
    assert len(result.to_latex()) > 1000


@pytest.mark.parametrize("p", [1, 2, 3, 5])
def test_dh_individual_wald_moments_and_software_tail_convention(p):
    frame = sample()
    result = dhcausality(data=frame, y="y", x="x", panel="unit", time="t", lags=p)
    expected = []
    for unit, block in frame.groupby("unit"):
        z, outcome, beta, covariance, u = independent_fit(block, p)
        b = beta[p:2*p]
        wi = b @ np.linalg.solve(covariance[p:2*p, p:2*p], b)
        # Original paper eq (6): partial regression quadratic form / SSR/df.
        qo, _ = np.linalg.qr(z[:, [*range(p), 2*p]])
        cause_resid = z[:, p:2*p] - qo @ (qo.T @ z[:, p:2*p])
        qc, _ = np.linalg.qr(cause_resid)
        projected_y = qc.T @ outcome
        alternative_w = (projected_y @ projected_y) / (u @ u / (len(u) - z.shape[1]))
        assert_allclose(wi, alternative_w, rtol=1e-10)
        row = result["individual"].iloc[unit]
        assert_allclose(row.statistic, wi, rtol=2e-10)
        assert_allclose(row.p_value, chi2.sf(wi, p), rtol=2e-8, atol=1e-12)
        reported = result["coefficients"].query("unit == @unit")
        assert_allclose(reported.estimate, beta, rtol=2e-10, atol=2e-10)
        assert_allclose(reported.std_error, covariance.diagonal()**.5, rtol=2e-10)
        expected.append(wi)
    total, count = 120, 12
    wbar = np.mean(expected)
    # Independent plm authors' formula (raw T, common K); same statistics.
    ztilde_plm = np.sqrt(count/(2*p) * (total-3*p-5)/(total-2*p-3)) * \
        ((total-3*p-3)/(total-3*p-1) * wbar - p)
    zbar = np.sqrt(count/(2*p)) * (wbar-p)
    assert_allclose(result["tests"].statistic, [wbar, zbar, ztilde_plm], rtol=2e-10)
    probabilities = result["tests"].p_value.tolist()
    assert probabilities[0] is None
    assert_allclose(probabilities[1:], norm.sf([zbar, ztilde_plm]), rtol=2e-8, atol=1e-14)
    assert result.attrs["tail"] == "upper"
    assert result.attrs["usable_periods"] == total-p
    assert result.attrs["decision_test"] == "Ztilde"
    assert result.attrs["skipped_units"] == 0
    json.dumps(result.attrs, allow_nan=False)
    json.dumps(result["tests"].to_dict("records"), allow_nan=False)
    assert len(result.to_latex()) > 1000


@pytest.mark.parametrize("kind", ["uint64", "int64", "month", "business"])
def test_exact_integer_and_calendar_order_hash_and_scale_invariance(kind):
    frame = sample()
    if kind == "uint64":
        clock = np.arange(2**64-200, 2**64-80, dtype=np.uint64)
    elif kind == "int64":
        clock = np.arange(2**53+37, 2**53+157, dtype=np.int64)
    else:
        clock = pd.date_range("2000-01-01", periods=120, freq="MS" if kind == "month" else "B")
    frame["t"] = np.tile(clock, 12)
    plain = dhcausality(data=frame, y="y", x="x", panel="unit", time="t", lags=2)
    reordered = dhcausality(data=frame.sample(frac=1, random_state=82), y="y", x="x",
                           panel="unit", time="t", lags=2)
    assert_allclose(plain["tests"].statistic, reordered["tests"].statistic, rtol=1e-12)
    assert plain.attrs["sample_sha256"] == reordered.attrs["sample_sha256"]
    scaled = frame.assign(y=frame.y * 1e4 + 3e6, x=-.03 * frame.x + 8e2)
    changed = dhcausality(data=scaled, y="y", x="x", panel="unit", time="t", lags=2)
    assert_allclose(plain["tests"].statistic, changed["tests"].statistic, rtol=1e-9)
    a = bccaustest(data=frame.iloc[:120], y="y", x="x", time="t", frequencies=[.15, 1, 2])
    b = bccaustest(data=scaled.iloc[:120].sample(frac=1, random_state=82), y="y", x="x",
                  time="t", frequencies=[.15, 1, 2])
    assert_allclose(a["tests"].statistic, b["tests"].statistic, rtol=1e-8)


@pytest.mark.parametrize("option", [
    {"lags": 2}, {"lags": True}, {"lags": 3.5}, {"integration_order": 1},
    {"integration_order": True}, {"constant": 1}, {"frequencies": []},
    {"frequencies": [0]}, {"frequencies": [math.pi]}, {"frequencies": [math.nan]},
    {"frequencies": [True]}, {"frequencies": [.5, .5]}, {"frequencies": [.5, "1"]},
    {"frequencies": [1e-13]}, {"x": "y"}, {"y": ["y"]}, {"time": "x"},
    {"alpha": math.nan}, {"alpha": True},
])
def test_frequency_unsupported_or_unidentified_specs_are_rejected(option):
    with pytest.raises(AnalysisError):
        bccaustest(**{"data": sample(units=1), "y": "y", "x": "x", "frequencies": [.5], **option})


@pytest.mark.parametrize("option", [
    {"lags": True}, {"lags": 0}, {"lags": 1.2}, {"integration_order": 1},
    {"cross_section": "dependent"}, {"cross_section": []}, {"x": "y"},
    {"panel": "t"}, {"time": "x"}, {"alpha": math.inf},
])
def test_panel_unsupported_scopes_are_rejected(option):
    with pytest.raises(AnalysisError):
        dhcausality(**{"data": sample(), "y": "y", "x": "x", "panel": "unit", "time": "t", **option})


@pytest.mark.parametrize("operation,code", [
    (lambda d: d.iloc[:-1], "unbalanced_panel"),
    (lambda d: d.assign(t=d.t * 2), "time_gaps"),
    (lambda d: d.assign(t=d.t + .1), "invalid_time"),
    (lambda d: pd.concat([d, d.iloc[:1]]), "repeated_time_values"),
    (lambda d: d.assign(t=d.t + (d.unit == 1).astype(int)), "unaligned_panel"),
    (lambda d: d.assign(x=np.nan), "missing_values"),
    (lambda d: d.assign(x=np.inf), "non_finite_values"),
    (lambda d: d.assign(x=d.y), "collinear_regressors"),
    (lambda d: d.assign(y=d.t.astype(float)), "perfect_fit"),
])
def test_no_skipped_units_missing_compression_or_time_misalignment(operation, code):
    with pytest.raises(AnalysisError) as error:
        dhcausality(data=operation(sample()), y="y", x="x", panel="unit", time="t")
    assert error.value.code == code


def test_too_short_one_unit_constant_series_and_unstable_var():
    with pytest.raises(AnalysisError, match="3\\*lags"):
        dhcausality(data=sample(periods=11), y="y", x="x", panel="unit", time="t", lags=2)
    with pytest.raises(AnalysisError) as error:
        dhcausality(data=sample(units=1), y="y", x="x", panel="unit", time="t")
    assert error.value.code == "insufficient_panels"
    for frame in (sample().assign(x=1.), sample().assign(unit=np.nan)):
        with pytest.raises(AnalysisError):
            dhcausality(data=frame, y="y", x="x", panel="unit", time="t")
    frame = sample(units=1)
    frame.y += 1.12 ** frame.t
    with pytest.raises(AnalysisError) as error:
        bccaustest(data=frame, y="y", x="x", frequencies=[.5])
    assert error.value.code == "unstable_var"


def test_budget_guard_before_design_and_external_engine_not_used(monkeypatch):
    from openecon.econometrics.var import panel_causality as implementation
    monkeypatch.setattr(implementation, "_MAX_WORK_BYTES", 1)
    monkeypatch.setattr(implementation, "_wald_batch", lambda *_: pytest.fail("Allocated QR"))
    with pytest.raises(AnalysisError) as error:
        dhcausality(data=sample(), y="y", x="x", panel="unit", time="t")
    assert error.value.code == "work_budget_exceeded"


def test_public_exports_and_editor_catalogue_discovery():
    import openecon as oe
    from pathlib import Path
    assert oe.bccaustest is bccaustest
    assert oe.dhcausality is dhcausality
    entries = json.loads((Path(__file__).resolve().parents[1] / "web/src/editor-api.json").read_text())
    by_name = {entry["name"]: entry for entry in entries}
    assert "bccaustest" in str(by_name)
    assert "dhcausality" in str(by_name)
