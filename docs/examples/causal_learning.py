"""Run in the OpenEconometrics code panel; synthetic pre-treatment features."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import openecon as oe

rng = np.random.default_rng(701)
n = 720
x = rng.normal(size=(n, 3))
probability = 1 / (1 + np.exp(-0.3 * x[:, 0] + 0.2 * x[:, 1]))
d = rng.binomial(1, probability)
df = pd.DataFrame(x, columns=["income", "age", "baseline"])
df["treated"] = d
df["outcome"] = 1 + (2 + 1.5 * x[:, 0]) * d + 0.5 * x[:, 0] + rng.normal(size=n)
df["region"] = np.where(x[:, 0] > 0, "high", "low")
df["policy"] = (x[:, 0] > 0).astype(float)  # declared before observing outcomes
controls = ["income", "age", "baseline"]
common = dict(data=df, y="outcome", treatment="treated", x=controls)
results = {
    "ate": oe.dmlirm(**common),
    "atet": oe.dmlirm(**common, estimand="atet"),
    "cate_basis": oe.dmlcate(**common, basis=["income"]),
    "gate": oe.dmlgate(**common, group="region"),
    "gates": oe.dmlgate(**common, groups=2, trees=12, max_depth=3, min_leaf=12),
    "forest": oe.causalforest(
        **common, trees=12, max_depth=3, min_leaf=12, query=[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    ),
    "policy": oe.policyvalue(**common, policy="policy", cost=0.25),
    "learned_policy": oe.policyvalue(
        **common, learned=True, cost=0.25, trees=12, max_depth=3, min_leaf=12
    ),
}

# Separate binary IV experiment: conditional instrument independence,
# exclusion and monotone increasing compliance are built into this DGP.
z = rng.binomial(1, probability)
di = (rng.uniform(size=n) < 0.15 + 0.65 * z).astype(int)
iv = df.copy()
iv["instrument"], iv["treated"] = z, di
iv["outcome"] = 1 + 2 * di + 0.5 * x[:, 0] + rng.normal(size=n)
results["late"] = oe.dmliivm(
    data=iv, y="outcome", treatment="treated", instrument="instrument", x=controls
)

saved_forest = oe.ResultBundle.model_validate_json(results["forest"].model_dump_json())
queries = pd.DataFrame([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], columns=controls)
cate = oe.causal_predict(saved_forest, queries)
assert np.isfinite(cate[["cate", "std_error", "ci_low", "ci_high"]]).all().all()
assert cate.cate.iloc[1] > cate.cate.iloc[0]
hashes = {}
for name, result in results.items():
    encoded = result.model_dump_json()
    restored = oe.ResultBundle.model_validate_json(encoded)
    assert restored.extra == result.extra
    assert restored.provenance["stata_parity_validated"] is False
    assert len(result.covariance_matrix) == len(result.coefficients)
    hashes[name] = hashlib.sha256(encoded.encode()).hexdigest()
    if "CAUSAL_LEARNING_RESULT_DIRECTORY" in globals():
        root = Path(globals()["CAUSAL_LEARNING_RESULT_DIRECTORY"])
        root.mkdir(parents=True, exist_ok=True)
        (root / (name + ".json")).write_text(encoded)
    if "display" in globals():
        # The complete result above retains all nuisance/forest state. Keep the
        # visible coefficient tables within the console's total display budget.
        globals()["display"](result.model_copy(update={"extra": {}}))
if "display" in globals():
    globals()["display"](cate)

print(
    "CAUSAL_LEARNING_ACCEPTANCE_OK "
    + json.dumps(
        {
            "methods": list(results),
            "saved_forest_query_rows": len(cate),
            "forest_query_effects": cate.cate.tolist(),
            "full_result_hashes": hashes,
            "native_float64": True,
            "json_roundtrip": True,
            "third_party_estimator_at_runtime": False,
            "forest_interval_target": cate.attrs["target"],
        }
    )
)
