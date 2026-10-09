"""Domain, alignment, precision, state restoration and preallocation refusals."""
import builtins
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.stats import manova_options as mo
from openecon.econometrics.stats.manova import manova
from openecon.econometrics.stats.rm_anova import rm_anova
from openecon.econometrics.summary_state import restore_summary, summary_state


def frame():
    rng = np.random.default_rng(550)
    result = pd.DataFrame(rng.normal(size=(36, 3)), columns=["a", "b", "c"])
    result["group"] = np.repeat(["A", "B", "C"], 12)
    result["w"] = np.tile([0, 1, 2, 3], 9)
    return result


def call(data=None, **kwargs):
    return mo.manova_oneway(frame() if data is None else data, ["a", "b", "c"], "group", **kwargs)


def fails(code, thunk):
    with pytest.raises(AnalysisError) as caught:
        thunk()
    assert caught.value.code == code


@pytest.mark.parametrize("weight", [-1., .5, float("inf"), float("nan"), 2**53+1, True, complex(1)])
def test_frequency_refuses_invalid_complete_rows(weight):
    data = frame().astype({"w": object})
    data.loc[0, "w"] = weight
    with pytest.raises(AnalysisError):
        call(data, weights="w", missing="raise")


@pytest.mark.parametrize("weight", [-1., .5, float("inf"), np.int64(2**53+1)])
def test_numeric_frequency_values_are_checked_before_any_float64_count_rounding(weight):
    data = frame()
    data["w"] = data.w.astype(type(weight))
    data.loc[0, "w"] = weight
    fails("invalid_weights", lambda: call(data, weights="w"))


@pytest.mark.parametrize("kwargs", [{"weight_type": "aweight"}, {"weight_type": "pweight"},
                                    {"weight_type": "iweight"}, {"missing": "pairwise"},
                                    {"alpha": 0}, {"alpha": True}, {"alpha": float("nan")}])
def test_refuses_unsupported_weight_missing_and_alpha_contract(kwargs):
    with pytest.raises(AnalysisError):
        call(**kwargs)


def test_missing_and_zero_frequency_positions_are_persisted_and_revalidated():
    data = frame()
    data.loc[2, "a"], data.loc[5, "group"], data.loc[8, "w"] = np.nan, None, np.nan
    output = call(data, weights="w")
    sample = output.attrs["manova_contrast_state"]["oneway"]["sample"]
    assert sample["source_positions"] == [i for i in range(36) if i not in [2, 5, 8]]
    assert sample["positive_complete_positions"] == [i for i, w in enumerate(sample["frequencies"]) if w > 0]
    assert output.attrs["n_missing"] == 3
    assert sum(sample["frequencies"]) == output.attrs["n"]
    assert output.attrs["physical_rows"] == len(sample["positive_complete_positions"])
    fails("missing_values", lambda: call(data, weights="w", missing="raise"))
    restored = restore_summary(summary_state(output))
    mo.manova_contrast(restored, [[-1., 1, 0]], M=[[1.], [-1], [0]])
    restored.attrs["manova_contrast_state"]["oneway"]["sample"]["source_positions"][0] = 1
    state = restored.attrs["manova_contrast_state"]
    state["sha256"] = mo._digest({k: v for k, v in state.items() if k != "sha256"})
    fails("invalid_state", lambda: mo.manova_contrast(restored, [[-1., 1, 0]]))


@pytest.mark.parametrize("kind", ["counts", "covariance", "coefficients", "sample", "typed_order", "design_alias", "digest"])
def test_restored_sufficient_geometry_tampering_refused_even_with_recomputed_checksum(kind):
    output = restore_summary(summary_state(call(weights="w")))
    state = output.attrs["manova_contrast_state"]
    if kind == "counts":
        state["oneway"]["counts"][0] += 1
    elif kind == "covariance":
        state["residual_sscp"][0][0] = -1
    elif kind == "coefficients":
        state["coefficients"][0][0] += 1e-12
    elif kind == "sample":
        state["oneway"]["sample"]["frequencies"][1] += 1
    elif kind == "typed_order":
        state["typed_design_columns"] = state["typed_design_columns"][::-1]
    elif kind == "design_alias":
        state["design_columns"][0] = "renamed"
        state["typed_design_columns"] = mo._typed_names(state["design_columns"])
    else:
        state["sha256"] = "0"*64
    if kind != "digest":
        state["sha256"] = mo._digest({k: v for k, v in state.items() if k != "sha256"})
    with pytest.raises(AnalysisError):
        mo.manova_contrast(output, [[-1., 1, 0]])


@pytest.mark.parametrize("kwargs", [
    {"L": [[0., 0, 0]]}, {"L": [[1., 0, 0], [2., 0, 0]]},
    {"L": [[1., 0]]}, {"L": [[True, False, False]]},
    {"L": [[1+1j, 0, 0]]}, {"L": [[float("nan"), 0, 0]]},
    {"L": [[1., 0, 0]], "M": [[1., 1], [0., 0], [0., 0]]},
    {"L": [[1., 0, 0]], "M": [[1.], [0.]]},
    {"L": [[1., 0, 0]], "null": [[1., 2.]]},
    {"L": [[1., 0, 0]], "contrast_names": ["same", "same"]},
    {"L": [[1., 0, 0]], "transform_names": ["duplicate"]*3},
])
def test_invalid_hypothesis_rank_dimension_values_and_labels(kwargs):
    output = call()
    with pytest.raises(AnalysisError):
        mo.manova_contrast(output, **kwargs)


def test_labelled_matrix_order_null_labels_and_roundtrip_are_explicit():
    output = call()
    columns = output.attrs["manova_contrast_state"]["design_columns"]
    left = pd.DataFrame([[-1., 1, 0]], index=["dose vs control"], columns=columns)
    m = pd.DataFrame([[-1.], [1.], [0.]], index=["a", "b", "c"], columns=["b minus a"])
    null = pd.DataFrame([[.2]], index=left.index, columns=m.columns)
    joint = mo.manova_contrast(output, left, M=m, null=null)
    assert joint["target_order"].iloc[0].tolist() == ["target[1]", "dose vs control", "b minus a"]
    fails("invalid_contrast", lambda: mo.manova_contrast(output, left[columns[::-1]], M=m))
    fails("invalid_contrast", lambda: mo.manova_contrast(output, left, M=m.iloc[::-1]))
    fails("invalid_contrast", lambda: mo.manova_contrast(output, left, M=m, null=null.rename(index={left.index[0]: "wrong"})))
    restored = restore_summary(summary_state(joint))
    pd.testing.assert_frame_equal(joint["target_covariance"], restored["target_covariance"])
    assert restored.attrs["contrast_spec"] == joint.attrs["contrast_spec"]
    assert json.loads(summary_state(restored))["attrs"]["contrast_spec"]["M"] == m.to_numpy().tolist()


@pytest.mark.parametrize("counts", [[True, 4, 4], [2.5, 4, 4], [2., 4, 4], [1, 4, 4], [2**53, 4, 4]])
def test_summary_declared_counts_must_be_integer_representable_and_n_at_least_two(counts):
    means = pd.DataFrame(np.zeros((3, 2)), index=["A", "B", "C"], columns=["a", "b"])
    cov = {g: np.eye(2) for g in means.index}
    with pytest.raises(AnalysisError):
        mo.manova_summary(means, cov, counts)


@pytest.mark.parametrize("covariance,count", [([[1., 2], [2., 1]], 5),
                                               ([[1., .1], [.2, 1]], 5),
                                               ([[0., .1], [.1, 1]], 5),
                                               (np.eye(2), 2)])
def test_summary_covariance_must_be_psd_symmetric_and_sample_rank_compatible(covariance, count):
    means = pd.DataFrame(np.zeros((2, 2)), index=["A", "B"], columns=["a", "b"])
    with pytest.raises(AnalysisError):
        mo.manova_summary(means, {"A": covariance, "B": np.eye(2)}, [count, 5])


def test_summary_per_group_psd_is_valid_with_full_rank_pooled_error():
    means = pd.DataFrame([[0., 0], [1., .5]], index=["A", "B"], columns=["a", "b"])
    output = mo.manova_summary(means, {"A": [[1., 0], [0, 0]], "B": [[0., 0], [0, 2]]}, [2, 2])
    assert output.attrs["df_resid"] == 2
    assert np.isfinite(output["multivariate"].statistic).all()
    assert output.attrs["manova_contrast_state"]["oneway"]["sample"] is None
    fails("insufficient_observations", lambda: mo.manova_contrast(output, [[1., 0], [0, 1]]))
    # Rank-one hypothesis at nu=d retains a defined exact F law.
    mo.manova_contrast(output, [[-1., 1]])


def test_higher_rank_low_error_df_is_rejected_before_undefined_classical_f():
    means = pd.DataFrame(np.zeros((3, 2)), index=["A", "B", "C"], columns=["a", "b"])
    # 3 groups of 2 gives nu=3; nu=2 fixture is a resident singleton group.
    data = pd.DataFrame([[0., 0, "A"], [1., 0, "A"], [0., 0, "B"], [0., 1, "B"],
                         [1., 1, "C"]], columns=["a", "b", "group"])
    fails("insufficient_observations", lambda: mo.manova_oneway(data, ["a", "b"], "group"))
    # Declared summary and resident laws never silently emit null F/df/p values.
    output = mo.manova_summary(means, {g: np.eye(2) for g in means.index}, [3, 3, 3])
    assert output["multivariate"][["statistic", "df1", "df2", "p_value"]].notna().all().all()


def test_early_resident_row_work_and_joint_covariance_admission(monkeypatch):
    class SizedOnly:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __iter__(self):
            raise AssertionError("Must refuse before materialization")

    fails("workspace_limit", lambda: call({name: SizedOnly(100001) for name in ["a", "b", "c", "group"]}))
    names = [f"y{i}" for i in range(80)]
    fails("work_limit", lambda: mo.manova_oneway({name: SizedOnly(50000) for name in [*names, "g"]}, names, "g"))
    means = pd.DataFrame(np.zeros((3, 100)), index=["A", "B", "C"])
    means.columns = [f"y{i}" for i in range(100)]
    fails("workspace_limit", lambda: mo.manova_summary(means, {}, [200]*3))
    # Joint covariance allocation is admitted by shape before any caller tensor conversion.
    monkeypatch.setattr(mo, "MAX_TARGETS", 2)
    with pytest.raises(AnalysisError) as caught:
        mo.manova_contrast(manova(frame(), ["a", "b", "c"], ["group"]), [[1., 0, 0]])
    assert caught.value.code == "workspace_limit"


def test_input_roles_devices_boolean_outcomes_and_total_frequency_guards():
    fails("unsupported_data", lambda: call(Dataset.from_frame(frame())))
    fails("invalid_spec", lambda: call(weights="a"))
    fails("invalid_matrix", lambda: call(frame().assign(a=True)))
    fails("unsupported_device", lambda: call({"a": torch.empty(36, device="meta"),
                                               "b": list(range(36)), "c": list(range(36)),
                                               "group": list(frame().group)}))
    data = frame().astype({"w": object})
    data["w"] = [2**53]*len(data)
    fails("invalid_weights", lambda: call(data, weights="w"))
    fails("empty_sample", lambda: call(frame().assign(w=0), weights="w"))


def test_saved_native_design_metadata_order_and_typed_levels_are_revalidated():
    output = manova(frame(), ["a", "b", "c"], ["group"])
    state = deepcopy(output.attrs["manova_contrast_state"])
    state["design_metadata"]["typed_factor_levels"]["group"][0] = "not a scalar identity"
    state["sha256"] = mo._digest({k: v for k, v in state.items() if k != "sha256"})
    output.attrs["manova_contrast_state"] = state
    with pytest.raises(AnalysisError):
        mo.manova_contrast(output, [[1., 0, 0]])


def test_rm_saved_column_alias_and_covariance_refuse_rehashed_tampering():
    data = frame().iloc[:24].copy()
    long = pd.DataFrame([(i, t, row.group, row.a+t*.2+row.b*t)
                         for i, row in data.iterrows() for t in range(3)], columns=["id", "t", "g", "y"])
    fit = rm_anova(long, "y", "id", ["t"], between=["g"])
    # A bounded but forged column coordinate is not accepted merely because the
    # caller recomputed the hash; expected factorial order is reconstructed.
    state = fit.attrs["rm_contrast_state"]
    state["between_design_columns"][1] = "aliased"
    state["sha256"] = mo._digest({k: v for k, v in state.items() if k != "sha256"})
    fails("invalid_state", lambda: mo.rm_mtest(fit, [[1., 0]], M=[[-1.], [1.], [0.]]))


def test_output_float64_cpu_survives_caller_default_dtype_and_meta_context():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            output = call(weights="w")
            joint = mo.manova_contrast(output, [[-1., 1, 0]])
        assert output.attrs["precision"] == "float64"
        assert joint.attrs["device"] == "cpu"
        assert joint["target_covariance"].to_numpy().dtype == np.dtype("float64")
        fails("unsupported_device", lambda: mo.manova_contrast(output, torch.empty((1, 3), device="meta")))
    finally:
        torch.set_default_dtype(previous)


def test_new_runtime_procedures_do_not_require_scipy_or_statsmodels(monkeypatch):
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"scipy", "statsmodels"}:
            raise AssertionError("Production numerical inference must remain native Torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    fit = call(weights="w")
    mo.manova_contrast(fit, [[-1., 1, 0]], M=[[1.], [-1.], [0.]])
    means = pd.DataFrame([[0., 0], [1., .5]], index=["A", "B"], columns=["a", "b"])
    mo.manova_summary(means, {"A": np.eye(2), "B": np.eye(2)}, [5, 6])
