"""Eight bounded missing-data stages, synthetic resident CPU data only.

Run in the editor or source environment. The five imputation algorithms retain
parameter/predictive uncertainty and never change observed cells. Finite MCMC
chains are diagnostics, not a convergence certificate or a missingness proof.
"""

import json
import math

import openecon as oe
import pandas as pd

# This deterministic fixture is generated without using any global RNG state.
base = pd.DataFrame(
    {
        "x": [math.sin(i * 0.63) + i / 80 for i in range(72)],
        "y": [1.4 + 0.5 * (math.sin(i * 0.63) + i / 80) + math.cos(i * 1.47) for i in range(72)],
    },
    index=pd.Index([f"row-{i // 2}" for i in range(72)], name="participant"),
)
missing = base.copy()
missing.loc[missing.index.isin(["row-3", "row-8", "row-14"]), "y"] = float("nan")
missing.iloc[25:30, 0] = float("nan")
monotone = base.copy()
monotone.iloc[-16:, 1] = float("nan")
mono = oe.mi_monotone(monotone, ["x", "y"], m=8, seed=521)
em = oe.mvnorm_em(missing, ["x", "y"])
mcar = oe.little_mcar(missing, ["x", "y"])
mvn = oe.mi_mvn(
    missing,
    ["x", "y"],
    m=8,
    seed=714,
    burn=80,
    thin=15,
    prior={"mean": [0.0, 1.0], "kappa": 0.1, "df": 4.0, "scale": [[1.0, 0.0], [0.0, 1.0]]},
)
normal = oe.mi_chained(
    missing,
    ["x", "y"],
    methods={"x": "normal", "y": "normal"},
    m=8,
    seed=902,
    burn=10,
    iterations=5,
)
pmm = oe.mi_chained(
    missing,
    ["x", "y"],
    methods={"x": "pmm", "y": "pmm"},
    m=8,
    seed=347,
    burn=10,
    iterations=5,
    donors=5,
)
binary = pd.DataFrame(
    {"x": base.x, "z": [float((i * 7) % 11 >= 5) for i in range(72)]}, index=base.index
)
binary.iloc[8:19, 1] = float("nan")
logit = oe.mi_chained(
    binary,
    ["x", "z"],
    methods={"z": "logit"},
    m=8,
    seed=1304,
    burn=3,
    iterations=2,
    logit_steps=400,
    logit_burn=200,
)
fits = [oe.ols(data=mono.dataset(i), y="y", x=["x"], device="cpu") for i in range(1, 9)]
pool = oe.mi_pool(
    fits,
    imputation_description="Monotone Gaussian posterior imputation with all 72 physical rows and fixed x predictor",
)
d1 = oe.mi_test(pool, restrictions=[[0.0, 1.0]], values=[0.0])
results = {
    "em": em,
    "mcar": mcar,
    "mvn": mvn,
    "monotone": mono,
    "normal": normal,
    "pmm": pmm,
    "logit": logit,
    "d1": d1,
}
states = {name: value.model_dump(mode="json") for name, value in results.items()}
for name, state in states.items():
    cls = (
        oe.MIDiagnosticResult
        if name in {"em", "mcar"}
        else oe.MIJointResult
        if name == "d1"
        else oe.MIResult
    )
    restored = cls.model_validate_json(json.dumps(state, allow_nan=False))
    assert restored.model_dump(mode="json") == state
    if name not in {"em", "mcar", "d1"}:
        assert restored.dataset(1).equals(results[name].dataset(1))
        assert restored.dataset(1).index.equals(base.index)
    # Eight separate displayed results. Full covariances, draws and model state
    # are retained in `states`; display previews never become inference inputs.
    rendered = (
        em.to_frame()["Gaussian means"]
        if name == "em"
        else mcar.to_frame()["Little MCAR mean homogeneity"]
        if name == "mcar"
        else restored.table
    )
    if "display" in globals():
        globals()["display"](rendered)
    else:
        print(name, rendered)
print(
    "MI_EIGHT_OK:"
    + json.dumps({"stages": len(states), "restored_equal": True, "rows": 72}, sort_keys=True)
)
