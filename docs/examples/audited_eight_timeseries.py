"""Native saved mixed-frequency, ETS-selection and state-space workflows.

Every fitted source bundle and complete output is written and read back. The
synthetic inputs establish native runtime/persistence, not vendor parity.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import torch

import openecon as oe
from openecon.econometrics.core import TableSet


def _model(destination, name, result):
    path = destination / (name + "-model.json")
    path.write_text(result.model_dump_json())
    restored = oe.ResultBundle.model_validate_json(path.read_text())
    assert restored.model_dump(mode="json") == result.model_dump(mode="json")
    record = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "estimator": result.spec.estimator,
        "nobs": result.nobs,
        "parameter_order": [value.term for value in result.coefficients],
        "full_json_readback": True,
    }
    if result.spec.estimator == "sspace":
        # Verify the algorithm's canonical digest against the complete saved
        # source file, retaining its UUID and timestamp. Only its digest field
        # is omitted, exactly as required by the documented state schema.
        payload = json.loads(path.read_text())
        state_digest = payload["extra"].pop("state_sha256")
        verified = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        ).hexdigest()
        assert verified == state_digest
        record.update(
            state_digest=state_digest,
            state_digest_verified_full_bundle=True,
            state_digest_omitted_field="extra.state_sha256",
            state_digest_encoding="sorted compact finite JSON",
        )
    return restored, record


def _output(destination, name, output):
    if isinstance(output, pd.DataFrame):
        output = TableSet({"values": output}, title=name, **output.attrs)
    path = destination / (name + ".json")
    path.write_text(oe.summary_state(output))
    restored = oe.restore_summary(path.read_text())
    assert oe.summary_state(restored) == path.read_text()
    (destination / (name + ".tex")).write_text(restored.to_latex())
    return {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tables": {key: len(value) for key, value in restored.items()},
        "full_json_readback": True,
    }


def _same_frame(left, right):
    assert left.columns.tolist() == right.columns.tolist()
    assert left.index.tolist() == right.index.tolist()
    assert left.to_numpy().tolist() == right.to_numpy().tolist()
    assert left.attrs == right.attrs


def run(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(87139)
    dtype = torch.float64
    high = pd.DataFrame(
        {
            "month": pd.date_range("2000-01-01", periods=230, freq="MS"),
            "value": torch.randn(230, generator=generator, dtype=dtype).tolist(),
        }
    )
    high["release"] = high["month"] + pd.Timedelta(days=12)
    low = pd.DataFrame(
        {
            "quarter": pd.date_range("2001-03-31", periods=64, freq="QE"),
            "control": torch.randn(64, generator=generator, dtype=dtype).tolist(),
        }
    )
    low["origin"] = low["quarter"] - pd.Timedelta(days=20)
    aligned = oe.midas_align(
        low=low,
        high=high,
        low_time="quarter",
        high_time="month",
        value="value",
        lags=4,
        frequency="MS",
        origin="origin",
        release_time="release",
    )
    lag_names = aligned.attrs["midas_alignment"]["lags"]
    X = torch.tensor(aligned[["control", *lag_names]].to_numpy(), dtype=dtype)
    aligned["y"] = (
        0.6
        + X @ torch.tensor([0.2, 0.7, -0.3, 0.2, -0.1], dtype=dtype)
        + 0.25 * torch.randn(64, generator=generator, dtype=dtype)
    ).tolist()
    umidas = oe.umidas(data=aligned, y="y", x=["control"], covariance="HC1")

    ets_data = pd.DataFrame(
        {
            "t": list(range(100)),
            "y": (
                2
                + 0.008 * torch.arange(100, dtype=dtype)
                + torch.tensor([-0.3, 0.1, 0.35, -0.15], dtype=dtype).repeat(25)
                + 0.35 * torch.randn(100, generator=generator, dtype=dtype)
            ).tolist(),
        }
    )
    models = ["ANN", "AAN", "AdN", "ANA", "AAA", "AdA"]
    settings = {}
    for model in models:
        fixed = {"alpha": 0.32}
        if model not in {"ANN", "ANA"}:
            fixed["beta"] = 0.08
        if model.endswith("A"):
            fixed["gamma"] = 0.12
        if "d" in model:
            fixed["phi"] = 0.91
        settings[model] = {"fixed": fixed}
    selected = oe.auto_ets(
        data=ets_data,
        y="y",
        time="t",
        models=models,
        period=4,
        criterion="aicc",
        candidate_options=settings,
    )
    assert all(row["status"] == "ok" for row in selected.extra["auto_selection"]["candidates"])

    n = 9
    measurements = torch.randn((n, 2), generator=generator, dtype=dtype)
    frame = pd.DataFrame(measurements.tolist(), columns=["a", "b"])
    frame["t"] = list(range(n))
    frame["u"] = torch.linspace(-0.5, 0.6, n, dtype=dtype).tolist()
    frame.loc[2, "a"] = float("nan")
    frame.loc[5, ["a", "b"]] = float("nan")
    system = {
        "T": [[0.6, 0.1], [0, 0.4]],
        "Z": [[1, 0.2], [-0.1, 1]],
        "Q": [[0.3, 0.04], [0.04, 0.2]],
        "H": [[0.25, 0.05], [0.05, 0.3]],
        "c": [0.02, -0.03],
        "d": [0.1, 0.2],
        "a0": [0.3, -0.2],
        "P0": [[0.7, 0.1], [0.1, 0.6]],
        "initialization": "known",
        "schedules": {
            "T": [[[0.5 + 0.02 * t, 0.1], [0, 0.4]] for t in range(n)],
            "d": [[0.1 + 0.03 * t, 0.2] for t in range(n)],
        },
        "exogenous": {"state": {"columns": ["u"], "coefficients": [[0.12], [-0.08]]}},
    }
    sspace = oe.sspace(data=frame, y="a", responses=["b"], time="t", system=system, missing="mask")
    ml_data = pd.DataFrame(
        {"t": list(range(12)), "y": [1.3, 2.1, None, -0.4, 1.5, 3.2, None, 0.8, 1.9, 0.2, 2.7, 1.2]}
    )
    ml_system = {
        "T": [[0]],
        "Q": [[0]],
        "P0": [[0]],
        "a0": [0],
        "Z": [[1]],
        "c": [0.2],
        "d": [0.5],
        "H": [[1]],
        "schedules": {
            "Z": [[[z]] for z in [1, 1.2, 0.7, 1.5, 0.9, 1.1, 1.8, 0.6, 1.3, 1.7, 0.8, 1.4]]
        },
        "parameters": [
            {"name": "mean", "matrix": "d", "row": 0},
            {"name": "state_intercept", "matrix": "c", "row": 0},
            {"name": "variance", "matrix": "H", "row": 0, "col": 0, "transform": "positive"},
        ],
    }
    sspace_ml = oe.sspace(
        data=ml_data, y="y", time="t", system=ml_system, missing="mask", tolerance=1e-9
    )

    # Source bundles are persisted before any postestimation summaries. Their
    # raw hashes and independently verified integrity digests remain auditable.
    saved, sources = {}, {}
    for name, result in {
        "umidas": umidas,
        "auto-ets": selected,
        "sspace": sspace,
        "sspace-ml": sspace_ml,
    }.items():
        saved[name], sources[name] = _model(destination, name, result)
    mean = oe.umidas_predict(saved["umidas"], data=aligned)
    _same_frame(mean, oe.umidas_predict(umidas, data=aligned))
    midas_windows = oe.rolling(saved["umidas"].spec, data=aligned, window=32, step=32)
    ets_forecast = oe.forecast(saved["auto-ets"], 8)
    _same_frame(ets_forecast, oe.forecast(selected, 8))
    ets_windows = oe.rolling(
        saved["auto-ets"].spec,
        data=ets_data,
        window=60,
        step=40,
        forecast_steps=2,
        evaluate=True,
        selection={
            "models": ["ANN", "AAN"],
            "candidate_options": {key: settings[key] for key in ["ANN", "AAN"]},
            "criterion": "bic",
        },
    )
    assert all(
        row["auto_selection"]["candidate_count"] == 2 for row in ets_windows.attrs["origins"]
    )
    smooth = oe.sspace_smooth(saved["sspace"])
    lag = oe.sspace_autocov(saved["sspace"])
    noise = oe.sspace_disturbances(saved["sspace"])
    for function, output in [
        (oe.sspace_smooth, smooth),
        (oe.sspace_autocov, lag),
        (oe.sspace_disturbances, noise),
    ]:
        _same_frame(output, function(sspace))
    future = {
        "schedules": {
            "T": deepcopy(system["schedules"]["T"][-3:]),
            "d": deepcopy(system["schedules"]["d"][-3:]),
        },
        "exogenous": {"u": [0.2, 0.4, -0.1]},
    }
    state_forecast = oe.forecast(saved["sspace"], 3, future=future)
    _same_frame(state_forecast, oe.forecast(sspace, 3, future=future))
    ml_coefficients = pd.DataFrame(
        [value.model_dump() for value in saved["sspace-ml"].coefficients]
    )
    ml_coefficients.attrs.update(
        parameter_order=[value.term for value in saved["sspace-ml"].coefficients],
        covariance_matrix=saved["sspace-ml"].covariance_matrix,
        log_likelihood=saved["sspace-ml"].metrics["log_likelihood"],
    )
    low_periods = ["2019", "2020", "2021"]
    high_periods = [f"{year}Q{quarter}" for year in range(2019, 2022) for quarter in range(1, 5)]
    denton = oe.denton_additive(
        [8, 10, 12],
        [1.5, 2, 2.2, 2.4, 2, 2.3, 2.5, 2.7, 2.5, 3, 3.2, 3.5],
        low_periods=low_periods,
        high_periods=high_periods,
        aggregation="sum",
    )
    outputs = {}
    for name, output in {
        "umidas-mean": mean,
        "umidas-rolling": midas_windows,
        "ets-forecast": ets_forecast,
        "ets-rolling-selection": ets_windows,
        "sspace-smooth": smooth,
        "sspace-lag-covariance": lag,
        "sspace-disturbances": noise,
        "sspace-forecast": state_forecast,
        "sspace-ml-coefficients": ml_coefficients,
        "denton-sum": denton,
    }.items():
        outputs[name] = _output(destination, name, output)
    proof = {
        "status": "passed",
        "synthetic": True,
        "seed": 87139,
        "device": "cpu",
        "precision": "float64",
        "sources": sources,
        "outputs": outputs,
        "source_refit_on_restoration": False,
        "vendor_execution": False,
        "desktop_validation": False,
        "ets_candidate_count": len(models),
        "ets_selected_model": selected.extra["model"],
        "ets_free_initial_and_sigma_counted": True,
        "rolling_ets_search_repeated_each_origin": True,
        "sspace_masked_calendar_retained": saved["sspace"].sample_positions == list(range(n)),
        "sspace_all_missing_periods": [
            i for i, value in enumerate(sspace.extra["observed_count"]) if value == 0
        ],
        "forecast_uncertainty": "conditional innovation uncertainty; parameter and selection uncertainty excluded",
    }
    (destination / "timeseries-receipt.json").write_text(
        json.dumps(proof, allow_nan=False, sort_keys=True)
    )
    return proof


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(
        json.dumps(
            run(sys.argv[1] if len(sys.argv) > 1 else "/tmp/openecon-audited-eight-timeseries"),
            sort_keys=True,
        )
    )
