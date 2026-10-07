"""Explicit NumPy panel/period covariance, time and early-allocation oracles."""
import numpy as np
import pandas as pd
import pytest

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.streaming_panelgls import fit_streaming_panelgls
from openecon.econometrics.systems.xtgls import fit_xtgls
from openecon.econometrics.systems.xtpcse import fit_xtpcse
from openecon.resources import use_workspace_budget

from test_econ_streaming_panelgls import data, parity, spec


@pytest.mark.parametrize("estimator", ["xtgls", "xtpcse"])
@pytest.mark.parametrize("rhotype", ["regress", "freg", "tscorr", "dw", "theil", "nagar"])
def test_independent_numpy_gls_pcse_and_prais_winsten_oracle(estimator, rhotype):
    frame = data().sort_values(["g", "t"])
    s = spec(estimator, panels="correlated", corr="psar1", rhotype=rhotype)
    result = fit_streaming_panelgls(s, Dataset.from_frame(frame), batch_rows=13)
    x, y = np.column_stack((np.ones(len(frame)), frame.x)), frame.y.to_numpy()
    m, periods = frame.g.nunique(), frame.t.nunique()
    initial = np.linalg.lstsq(x, y, rcond=None)[0]
    errors = (y-x@initial).reshape(m, periods)
    cross = (errors[:, 1:]*errors[:, :-1]).sum(1)
    every = (errors**2).sum(1)
    if rhotype=="regress":
        rho = cross/(errors[:, :-1]**2).sum(1)
    elif rhotype=="freg":
        rho = cross/(errors[:, 1:]**2).sum(1)
    elif rhotype in {"tscorr", "theil"}:
        rho = cross/every
        if rhotype=="theil":
            rho *= (periods-x.shape[1])/periods
    else:
        rho = 1-((errors[:, 1:]-errors[:, :-1])**2).sum(1)/every/2
        if rhotype=="nagar":
            rho = (rho*periods**2+x.shape[1]**2)/(periods**2-x.shape[1]**2)
    np.testing.assert_allclose(result.extra["rho"], rho, atol=2e-14)
    d = np.column_stack((x, y)).reshape(m, periods, -1)
    transformed = d.copy()
    transformed[:, 1:] = d[:, 1:]-rho[:, None, None]*d[:, :-1]
    transformed[:, 0] *= np.sqrt(1-rho**2)[:, None]
    xs, ys = transformed[:, :, :2].reshape(-1, 2), transformed[:, :, 2].flatten()
    reference = np.linalg.lstsq(xs, ys, rcond=None)[0]
    error = (ys-xs@reference).reshape(m, periods)
    sigma = error@error.T/periods
    omega = np.kron(sigma, np.eye(periods))
    if estimator=="xtgls":
        inverse_x = np.linalg.solve(omega, xs)
        cov = np.linalg.inv(xs.T@inverse_x)
        beta = cov@(xs.T@np.linalg.solve(omega, ys))
    else:
        beta = reference
        bread = np.linalg.inv(xs.T@xs)
        cov = bread@xs.T@omega@xs@bread
    np.testing.assert_allclose([c.estimate for c in result.coefficients], beta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.covariance_matrix, cov, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pairwise", [False, True])
def test_independent_numpy_unbalanced_sigma_exact_overlap_denominators(pairwise):
    frame = data(unbalanced=True)
    s = spec("xtpcse", pairwise=pairwise)
    result = fit_streaming_panelgls(s, Dataset.from_frame(frame), batch_rows=13)
    x, y = np.column_stack((np.ones(len(frame)), frame.x)), frame.y.to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y-x@beta
    m = frame.g.nunique()
    sigma = np.zeros((m, m))
    common = set.intersection(*(set(frame.loc[frame.g==g, "t"]) for g in range(m)))
    for i in range(m):
        for j in range(m):
            left = {int(t): residual[row] for row, t in enumerate(frame.t) if frame.g.iloc[row]==i}
            right = {int(t): residual[row] for row, t in enumerate(frame.t) if frame.g.iloc[row]==j}
            overlap = set(left)&set(right) if pairwise else common
            sigma[i, j] = sum(left[t]*right[t] for t in overlap)/len(overlap) if overlap else 0
    meat = np.zeros((2, 2))
    for _, subset in frame.groupby("t"):
        ids = subset.index.to_numpy()
        g = subset.g.to_numpy()
        meat += x[ids].T@sigma[np.ix_(g, g)]@x[ids]
    bread = np.linalg.inv(x.T@x)
    np.testing.assert_allclose(result.extra["sigma_matrix"], sigma, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(result.covariance_matrix, bread@meat@bread, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("estimator", ["xtgls", "xtpcse"])
@pytest.mark.parametrize("time", ["large_integer", "datetime_timezone"])
def test_exact_integer_and_global_datetime_order_matches_dense(estimator, time):
    frame = data()
    if time=="large_integer":
        frame.t = frame.t.astype("int64")+2**60
    else:
        frame.t = pd.to_datetime("2024-01-01", utc=True)+pd.to_timedelta(frame.t, unit="D")
        frame.t = frame.t.dt.tz_convert("Europe/Istanbul")
    s = spec(estimator, panels="heteroskedastic", corr="psar1", rhotype="tscorr", hetonly=True)
    reference = fit_xtgls(s, frame) if estimator=="xtgls" else fit_xtpcse(s, frame)
    parity(reference, fit_streaming_panelgls(s, Dataset.from_frame(frame), batch_rows=13))


def test_many_groups_iid_no_group_tensor_or_full_panel_collection(monkeypatch):
    frame = data(groups=1003, periods=3)
    source = Dataset.from_frame(frame)
    def forbidden(*args, **kwargs):
        raise AssertionError("scalar panel path must not collect data or allocate a group covariance")
    monkeypatch.setattr(source, "collect", forbidden, raising=False)
    import openecon.econometrics.streaming_panelgls as native
    monkeypatch.setattr(native, "_sigma", forbidden)
    result = fit_streaming_panelgls(spec("xtgls"), source, batch_rows=13)
    assert result.metrics["n_groups"]==1003
    assert result.provenance["streaming"]["maximum_batch_rows"]<=13
    parity(fit_xtgls(spec("xtgls"), frame), result)


def test_quadratic_sigma_refuses_before_any_sigma_allocation(monkeypatch):
    frame = data(groups=500, periods=2)
    import openecon.econometrics.streaming_panelgls as native
    def forbidden(*args, **kwargs):
        raise AssertionError("Sigma must not be allocated before the quadratic resource plan")
    monkeypatch.setattr(native, "_sigma", forbidden)
    with use_workspace_budget(20), pytest.raises(AnalysisError) as error:
        fit_streaming_panelgls(spec("xtpcse"), Dataset.from_frame(frame), batch_rows=13)
    assert error.value.code == "workspace_limit"


@pytest.mark.parametrize("failure", ["duplicate", "time_gaps", "int64_wrap", "bool", "singletons_ar", "unbalanced_gls", "few_periods_gls", "no_common", "incompatible_options", "perfect", "nonconvergence"])
def test_time_structure_and_estimation_failures(failure):
    frame = data()
    s = spec("xtpcse", corr="ar1", rhotype="tscorr")
    if failure=="duplicate":
        frame.loc[0, ["g", "t"]] = frame.loc[1, ["g", "t"]]
        code = "repeated_time_values"
    elif failure=="time_gaps":
        frame.t *= 2
        code = "time_gaps"
    elif failure=="int64_wrap":
        frame.t = frame.t.astype("int64")
        frame.loc[frame.t==0, "t"] = np.iinfo(np.int64).max
        frame.loc[frame.t==1, "t"] = np.iinfo(np.int64).min
        code = "time_gaps"
    elif failure=="bool":
        frame.t = frame.t==0
        code = "invalid_time"
    elif failure=="singletons_ar":
        frame.g = np.arange(len(frame))
        s = spec("xtpcse", corr="psar1", rhotype="tscorr", hetonly=True)
        code = "insufficient_observations"
    elif failure=="unbalanced_gls":
        frame = data(unbalanced=True)
        s = spec("xtgls", panels="correlated")
        code = "unbalanced_panel"
    elif failure=="few_periods_gls":
        frame = data(periods=3)
        s = spec("xtgls", panels="correlated")
        code = "insufficient_periods"
    elif failure=="no_common":
        frame = frame[frame.g%2==frame.t%2]
        s = spec("xtpcse", hetonly=True)
        code = "no_common_periods"
    elif failure=="incompatible_options":
        s = spec("xtpcse", independent=True, hetonly=True)
        code = "invalid_spec"
    elif failure=="perfect":
        frame.y = 1+.6*frame.x
        s = spec("xtgls")
        code = "perfect_fit"
    else:
        s = spec("xtgls", panels="heteroskedastic", igls=True).model_copy(update={"options": {"panels": "heteroskedastic", "igls": True, "max_iterations": 1}})
        code = "nonconvergence"
    with pytest.raises(AnalysisError) as error:
        fit_streaming_panelgls(s, Dataset.from_frame(frame), batch_rows=13)
    assert error.value.code == code


def test_original_source_mutation_while_using_snapshot_is_rejected_and_cleaned(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame, changed = data(), False
    import openecon.econometrics.streaming_panelgls as native
    original_qr = native._qr
    def arm_after_snapshot(*args, **kwargs):
        nonlocal changed
        value = original_qr(*args, **kwargs)
        changed = True
        return value
    monkeypatch.setattr(native, "_qr", arm_after_snapshot)
    def reader():
        rows = frame.copy()
        if changed:
            rows.loc[0, "y"] += 1
        yield rows
    source = Dataset.from_batches(reader, frame.columns.tolist(), row_count=len(frame))
    with pytest.raises(AnalysisError) as error:
        fit_streaming_panelgls(spec("xtgls"), source, batch_rows=13)
    assert error.value.code == "source_changed"
    assert not list(tmp_path.iterdir())
