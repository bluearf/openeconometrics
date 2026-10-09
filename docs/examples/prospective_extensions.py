"""Eight prospective extensions with full JSON replay; synthetic assumptions only.

The code panel displays sixteen short plan/scenario tables. Every count law,
allocation row and setting is preserved in complete summary JSON, independently
of the console's preview limit. Set PROSPECTIVE_EXTENSION_RESULT_DIRECTORY to
an owned directory to retain those complete files.
"""

import hashlib
import json
from pathlib import Path
import sys

import openecon as oe
import torch

INPUTS = {
    "power_welch": dict(effect=0.6, sd1=1.0, sd2=1.8, n1=35, n2=50),
    "power_unbalanced_anova": dict(means=[0.0, 0.4, 0.9], sizes=[20, 30, 45], sd=1.2),
    "power_unequal_cluster_mean": dict(
        effect=0.5,
        sd1=1.0,
        sd2=1.3,
        sizes1=[10, 20, 15, 30],
        sizes2=[12, 25, 18, 35, 15],
        icc1=0.08,
        icc2=0.12,
        weighting="participant",
    ),
    "power_mcnemar_unconditional": dict(p10=0.3, p01=0.1, n=100),
    "precision_twomeans_unknown": dict(sd=1.2, n=30, ratio=1.5, assurance=0.9),
    "precision_binomial": dict(p=0.25, n=20, assurance=0.9),
    "precision_poisson": dict(rate=1.2, exposure_per_unit=0.5, n=10, assurance=0.9),
    "survival_accrual": dict(
        event_rate1=0.12,
        event_rate2=0.08,
        accrual=12.0,
        followup=6.0,
        dropout_rate1=0.02,
        dropout_rate2=0.01,
        allocation=0.4,
        n=200,
    ),
}

previous_threads = torch.get_num_threads()
torch.set_num_threads(1)
try:
    proof, encoded, displayed_tables = {}, {}, 0
    for name, inputs in INPUTS.items():
        result = getattr(oe, name)(**inputs)
        state = oe.summary_state(result)
        restored = oe.restore_summary(state)
        assert oe.summary_state(restored) == state
        replay = getattr(oe, name)(**json.loads(json.dumps(inputs)))
        assert oe.summary_state(replay) == state
        assert restored.to_latex() == result.to_latex(), name
        payload = json.dumps(
            {"inputs": inputs, "summary": json.loads(state), "latex": result.to_latex()},
            allow_nan=False,
            sort_keys=True,
        )
        encoded[name] = payload
        proof[name] = {
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "table_rows": {key: len(frame) for key, frame in result.items()},
            "plan": result["plan"]
            .astype(object)
            .where(result["plan"].notna(), None)
            .to_dict("records"),
        }
        if "display" in globals():
            globals()["display"](result["plan"])
            globals()["display"](result["scenarios"])
            displayed_tables += 2
    if "PROSPECTIVE_EXTENSION_RESULT_DIRECTORY" in globals():
        directory = Path(globals()["PROSPECTIVE_EXTENSION_RESULT_DIRECTORY"])
        directory.mkdir(parents=True, exist_ok=True)
        for name, payload in encoded.items():
            path = directory / (name + ".json")
            path.write_text(payload)
            assert path.read_text() == payload
    forbidden = [name for name in ("scipy", "statsmodels", "linearmodels") if name in sys.modules]
    assert not forbidden
    print(
        "PROSPECTIVE_EXTENSIONS_OK "
        + json.dumps(
            {
                "methods": proof,
                "full_input_replay_verified": True,
                "full_summary_restore_verified": True,
                "saved_tables": sum(
                    sum(1 for _ in json.loads(payload)["summary"]["tables"])
                    for payload in encoded.values()
                ),
                "displayed_tables": displayed_tables,
                "third_party_estimation_imports": forbidden,
                "frozen": bool(getattr(sys, "frozen", False)),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
finally:
    torch.set_num_threads(previous_threads)
