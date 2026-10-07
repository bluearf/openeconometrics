"""Development-only reproducible numerical receipt for the five MARKET-129 routes."""

from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
import openecon as oe

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    module_spec = importlib.util.spec_from_file_location(
        "ts_reference", ROOT / "tests/test_tsworkflows_oracle.py"
    )
    reference = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(reference)
    rng = np.random.default_rng(129)
    observations = rng.normal(size=(22, 2))
    system = reference.fixed_system()
    state = oe.sspace(
        data={"a": observations[:, 0], "b": observations[:, 1]},
        y="a",
        responses=["b"],
        system=system,
    )
    density = reference.density_reference(observations, system)[0]
    filtered, covariances, _, _, _, _ = reference.kalman_reference(observations, system)
    data, _, _, _ = reference.aligned_midas()
    midas = oe.midas(data=data, y="y", x=["x"], covariance="HC1")
    series = rng.normal(size=120)
    for i in range(1, len(series)):
        series[i] += 0.7 * series[i - 1]
    selected = oe.auto_arima(data={"y": series}, y="y", d=0, max_p=1, max_q=0, constant=False)
    ets = oe.ets(data={"y": series}, y="y", fixed={"alpha": 0.3}, initial=[0.0])
    receipt = {
        "schema_version": 1,
        "scope": "source-level CPU float64 method/domain evidence; no actual Stata/CUDA/installed/public proof",
        "runtime": "native Torch; independent NumPy/SciPy/statsmodels development references only",
        "references": [
            "https://www.jstatsoft.org/article/view/v027i03",
            "https://otexts.com/fpp3/ets.html",
            "https://doi.org/10.1080/07474930600972467",
            "https://www.rba.gov.au/publications/rdp/2012/2012-08/appendix-a.html",
        ],
        "seeds": [129, 93],
        "max_error": {
            "joint_density_likelihood": abs(state.metrics["log_likelihood"] - density),
            "filtered_state": float(np.max(np.abs(np.array(state.extra["filtered"]) - filtered))),
            "full_filtered_covariance": float(
                np.max(np.abs(np.array(state.extra["filtered_covariance"]) - covariances))
            ),
        },
        "results": {
            name: result.model_dump(mode="json")
            for name, result in (
                ("sspace", state),
                ("midas_HC1", midas),
                ("auto_arima", selected),
                ("ets", ets),
            )
        },
        "forecasts": {
            name: {
                "rows": oe.forecast(result, 5).to_dict("records"),
                "attrs": oe.forecast(result, 5).attrs,
            }
            for name, result in (("sspace", state), ("auto_arima", selected), ("ets", ets))
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "src/openecon/econometrics/tsworkflows").glob("*.py"))
        },
    }
    for error in receipt["max_error"].values():
        assert error < 1e-9
    destination = ROOT / "reports/validation/market129_tsworkflows.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt, allow_nan=False, indent=2) + "\n")
    print(json.dumps({"receipt": str(destination), "max_error": receipt["max_error"]}))


if __name__ == "__main__":
    main()
