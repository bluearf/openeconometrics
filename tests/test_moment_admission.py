"""Domain, early allocation and semantic portable-state refusal cases."""
import copy

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.stats import manova_factorial as mf
from openecon.econometrics.stats import rm_moments as rm
from openecon.econometrics.stats.manova_options import _digest, manova_contrast, rm_mtest
from openecon.econometrics.summary_state import restore_summary, summary_state
from openecon.resources import use_workspace_budget
from test_manova_factorial_moments import domain as factorial_domain
from test_rm_moment_methods import domain as rm_domain


def factorial_fit(frame=None):
    return mf.manova_factorial(factorial_domain()[0] if frame is None else frame,
                               ["y1", "y2", "y3"], ["A", "B"], weights="w")


def rm_fit(frame=None, **kwargs):
    return rm.rm_anova_fweight(rm_domain()[0] if frame is None else frame,
                              "y", "id", ["A", "B"], weights="w", between=["group"], **kwargs)


@pytest.mark.parametrize("frequency", [-1, .5, True, float("inf"), 2**53+1])
def test_factorial_invalid_frequency_is_refused_even_for_zero_or_deleted_domains(frequency):
    frame = factorial_domain()[0]
    frame["w"] = frame.w.astype(object)
    frame.loc[0, "w"] = frequency
    with pytest.raises(AnalysisError):
        factorial_fit(frame)


def test_factorial_listwise_missing_zero_accounting_and_empty_cells():
    frame = factorial_domain()[0]
    frame.loc[0, "y1"] = np.nan
    fit = factorial_fit(frame)
    reference = factorial_fit(frame.dropna())
    np.testing.assert_allclose(fit["error_sscp"], reference["error_sscp"], rtol=1e-13)
    sample = fit.attrs["sample"]
    assert sample["n_missing"] == 1
    assert 0 not in sample["source_positions"]
    assert sample["physical_rows"]+sample["n_zero_weight"]+sample["n_missing"] == len(frame)
    with pytest.raises(AnalysisError):
        mf.manova_factorial(frame, ["y1", "y2", "y3"], ["A", "B"], weights="w", missing="raise")
    frame = factorial_domain()[0]
    frame.loc[(frame.A == "A") & (frame.B == "early"), "w"] = 0
    with pytest.raises(AnalysisError) as error:
        factorial_fit(frame)
    assert error.value.code == "empty_cells"


@pytest.mark.parametrize("fault", ["fraction_count", "bool_count", "huge_count", "rank_count", "indefinite", "labels", "cell_order"])
def test_summary_refuses_count_covariance_and_order_faults(fault):
    _, _, means, covariances, counts = factorial_domain()
    first = means.index[0]
    if fault == "fraction_count":
        counts[0] = 2.5
    elif fault == "bool_count":
        counts[0] = True
    elif fault == "huge_count":
        counts[0] = 2**54
    elif fault == "rank_count":
        counts[0] = 2
    elif fault == "indefinite":
        covariances[first] = np.diag([-1., 1., 1.])
    elif fault == "labels":
        covariances[first] = pd.DataFrame(covariances[first], index=["y3", "y2", "y1"], columns=means.columns)
    else:
        means = means.iloc[::-1]
    with pytest.raises(AnalysisError):
        mf.manova_factorial_summary(means, covariances, counts)


@pytest.mark.parametrize("fault", ["rowwise_frequency", "between_changes", "duplicate", "missing_id", "incomplete"])
def test_rm_profile_identity_and_domain_refusals(fault):
    frame = rm_domain()[0]
    if fault == "rowwise_frequency":
        frame.loc[0, "w"] += 1
    elif fault == "between_changes":
        frame.loc[0, "group"] = "dose2"
    elif fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif fault == "missing_id":
        frame.loc[0, "id"] = np.nan
    else:
        frame = frame.iloc[1:]
    with pytest.raises(AnalysisError):
        rm_fit(frame)


def test_rm_missing_whole_subject_with_order_and_weight_accounting():
    frame = rm_domain()[0]
    frame.loc[0, "y"] = np.nan
    frame = frame.drop(index=6).reset_index(drop=True)
    output = rm_fit(frame, missing="drop_subject")
    reference = rm_fit(frame.loc[~frame.id.isin([0, 1])])
    for key in ["within", "between", "sphericity"]:
        np.testing.assert_allclose(output[key].select_dtypes(include="number"), reference[key].select_dtypes(include="number"), rtol=3e-12, atol=1e-12)
    sample = output.attrs["sample"]
    assert sample["n_missing"] == 11
    assert sample["n_dropped_subjects"] == 2
    assert sample["expanded_subjects"] == sum(sample["frequencies"])
    assert len(sample["profile_positions"])*6+sample["n_missing"] == len(frame)
    restored = restore_summary(summary_state(output))
    rm_mtest(restored, [[0., 1, 0]], M=[[-1.], [1.], [0.], [0.], [0.], [0.]])


@pytest.mark.parametrize("fault", ["reorder", "duplicate"])
def test_saved_whole_deleted_subject_identities_keep_exact_source_order(fault):
    frame = rm_domain()[0]
    frame.loc[0, "y"] = np.nan
    frame = frame.drop(index=6).reset_index(drop=True)
    fit = restore_summary(summary_state(rm_fit(frame, missing="drop_subject")))
    state = copy.deepcopy(fit.attrs["rm_moment_state"])
    provenance = state["provenance"]
    if fault == "reorder":
        provenance["dropped_subjects"].reverse()
    else:
        provenance["dropped_subjects"].append(provenance["dropped_subjects"][0])
        provenance["n_dropped_subjects"] += 1
    state["sha256"] = _digest({name: value for name, value in state.items() if name != "sha256"})
    fit.attrs["rm_moment_state"] = state
    with pytest.raises(AnalysisError) as error:
        rm_mtest(fit, [[0., 1, 0]], M=[[-1.], [1.], [0.], [0.], [0.], [0.]])
    assert error.value.code == "invalid_state"


@pytest.mark.parametrize("scope", ["factorial", "rm"])
@pytest.mark.parametrize("fault", ["origin", "mean", "sscp", "count", "design", "positive_order", "remove"])
def test_saved_moments_semantically_revalidated_after_recomputed_digest(scope, fault):
    source = factorial_fit() if scope == "factorial" else rm_fit()
    fit = restore_summary(summary_state(source))
    key = "factorial_moment_state" if scope == "factorial" else "rm_moment_state"
    state = copy.deepcopy(fit.attrs[key])
    if fault == "remove":
        del fit.attrs[key]
    else:
        if fault == "origin":
            state["origin"][0] += .1
        elif fault == "mean":
            state["mean_offsets"][1][0] += .1
        elif fault == "sscp":
            state["cell_sscp"][0] = (np.asarray(state["cell_sscp"][0])*2).tolist()
        elif fault == "count":
            state["counts"][0] += 1
        elif fault == "design":
            state["design_columns"][1] = "renamed"
        else:
            state["provenance"]["positive_complete_positions"] = state["provenance"]["positive_complete_positions"][::-1]
        state["sha256"] = _digest({name: value for name, value in state.items() if name != "sha256"})
        fit.attrs[key] = state
    with pytest.raises(AnalysisError):
        if scope == "factorial":
            manova_contrast(fit, [[0., 1, 0, 0, 0, 0]], M=[[-1.], [1.], [0.]])
        else:
            rm_mtest(fit, [[0., 1, 0]], M=[[-1.], [1.], [0.], [0.], [0.], [0.]])


@pytest.mark.parametrize("scope", ["factorial", "rm"])
def test_workspace_and_foreign_device_refused_before_numeric_conversion(monkeypatch, scope):
    frame = factorial_domain()[0] if scope == "factorial" else rm_domain()[0]
    def no_matrix(*args, **kwargs):
        pytest.fail("Numeric conversion ran after an early admission refusal")
    monkeypatch.setattr(mf.mc, "matrix", no_matrix)
    with use_workspace_budget(1), pytest.raises(AnalysisError) as error:
        # Sized raw inputs can be refused without constructing a DataFrame.
        data = {name: list(frame[name])*200 for name in frame}
        (factorial_fit if scope == "factorial" else rm_fit)(data)
    assert error.value.code in {"workspace_limit", "resource_budget_exceeded"}
    data = {name: frame[name].tolist() for name in frame}
    data["y1" if scope == "factorial" else "y"] = torch.empty(len(frame), device="meta")
    with pytest.raises(AnalysisError) as error:
        (factorial_fit if scope == "factorial" else rm_fit)(data)
    assert error.value.code == "unsupported_device"


def test_metadata_labels_and_portable_output_are_bounded_before_tensor_fit(monkeypatch):
    frame = factorial_domain()[0]
    frame["A"] = frame.A.map({"A": "x"*5000, "B": "b"})
    with pytest.raises(AnalysisError) as error:
        factorial_fit(frame)
    assert error.value.code == "workspace_limit"
    _, _, means, covariances, counts = factorial_domain()
    long_names = ["y"*3000+str(i) for i in range(3)]
    means.columns = long_names
    # Large valid individual names multiply across every H/E cell and table.
    monkeypatch.setattr(mf, "MAX_METADATA_BYTES", 1024**2)
    # A direct declared model planning case avoids an enormous test result.
    levels = {"A": list(range(2)), "B": list(range(2))}
    cells = [(a, b) for a in range(2) for b in range(2)]
    moments = mf.Moments(["A", "B"], levels, cells, ["y"*3000+str(i) for i in range(32)],
                         torch.zeros(32, dtype=torch.float64), torch.zeros((4, 32), dtype=torch.float64),
                         [torch.eye(32, dtype=torch.float64)]*4, [100]*4, "summary", {"n_input_rows": 0})
    with pytest.raises(AnalysisError) as error:
        mf.MomentModel(moments)
    assert error.value.code == "workspace_limit"


@pytest.mark.parametrize("summary", [False, True])
def test_subnormal_sample_variance_and_coefficient_underflow_explicitly_refused(summary):
    frame, _, means, covariances, counts = factorial_domain()
    if summary:
        covariances = {key: value*1e-310 for key, value in covariances.items()}
        means[:] = 0
        with pytest.raises(AnalysisError) as error:
            mf.manova_factorial_summary(means, covariances, counts)
    else:
        frame[["y1", "y2", "y3"]] *= 1e-160
        with pytest.raises(AnalysisError) as error:
            factorial_fit(frame)
    assert error.value.code == "unresolved_precision"
    covariances = {key: np.eye(3)*1e-300 for key in means.index}
    means[:] = 0
    with pytest.raises(AnalysisError) as error:
        mf.manova_factorial_summary(means, covariances, [10**10]*6)
    assert error.value.code == "unresolved_precision"


def test_rm_old_scale_and_saved_coefficient_precision_have_explicit_refusals():
    frame = rm_domain()[0]
    frame["y"] += 1e12
    fit = rm_fit(frame)
    with pytest.raises(AnalysisError) as error:
        rm_mtest(fit, [[1., 0, 0]], M=[[-1.], [1.], [0.], [0.], [0.], [0.]])
    assert error.value.code == "unresolved_precision"
    frame = rm_domain()[0]
    levels = np.arange(frame.id.nunique())*1e12
    frame["y"] += frame.id.map(dict(enumerate(levels)))
    with pytest.raises(AnalysisError) as error:
        rm_fit(frame)
    assert error.value.code in {"unresolved_precision", "singular_error_matrix", "invalid_covariance"}


def test_exact_frequency_total_excludes_repeated_within_rows():
    # Three cells carry one whole subject frequency, not three sample subjects.
    frame = rm_domain(492, between=False)[0]
    frame = frame.loc[frame.A == 0].copy()
    frame["w"] = 2**45
    output = rm.rm_anova_fweight(frame, "y", "id", ["B"], weights="w")
    assert output.attrs["n_subjects"] == frame.id.nunique()*2**45
    assert output.attrs["n"] == output.attrs["n_subjects"]*3
