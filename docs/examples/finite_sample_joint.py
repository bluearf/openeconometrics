"""Synthetic declared Gaussian examples; never reads a human dataset."""
# ruff: noqa: F821 -- display is provided by the OpenEconometrics workspace.
import json
import pandas as pd
import torch
import openecon as oe

generator = torch.Generator().manual_seed(207208209)
x = torch.randn(48, 2, dtype=torch.float64, generator=generator)
w = .5 + torch.rand(48, dtype=torch.float64, generator=generator)
noise = torch.randn(48, dtype=torch.float64, generator=generator)
data = pd.DataFrame({"x": x[:,0].tolist(), "z": x[:,1].tolist(), "w": w.tolist(),
                     "y": (.3+x[:,0]-.4*x[:,1]+noise).tolist(),
                     "yw": (.3+x[:,0]-.4*x[:,1]+noise/w.sqrt()).tolist()})
model = oe.ols(data=data, y="y", x=["x", "z"], covariance="nonrobust")
restored = oe.ResultBundle.model_validate_json(model.model_dump_json())
ordinary = oe.ols_stepdown(restored, error_model="iid_gaussian", terms=["x", "z"], draws=20000, seed=207)
weighted_model = oe.ols(data=data, y="yw", x=["x", "z"], covariance="nonrobust", weights="w", weight_type="aweight")
weighted = oe.ols_stepdown(weighted_model, error_model="known_precision_gaussian", terms=["x", "z"], draws=20000, seed=207)
intervals = oe.simultaneous_t_ci([c.estimate for c in model.coefficients], model.covariance_matrix,
    df=45, labels=[c.term for c in model.coefficients], draws=20000, seed=208,
    family_description="Predeclared fixed-design Gaussian coefficient family",
    pivot_description="Known design correlation, one independent residual chi-square scale with 45 df")
region = oe.hotelling_region(data, ["x", "z"], sampling_model="iid_multivariate_normal")
tables = [ordinary, weighted, intervals, region]
for result in tables:
    display(result)
states = [{"schema": 1, "table": json.loads(result.to_json(orient="table", double_precision=15)),
           "attrs": result.attrs} for result in tables]
json.dumps(states, allow_nan=False)
print("FINITE_SAMPLE_RECEIPT:"+json.dumps({
    "procedures": ["ols_stepdown", "simultaneous_t_ci", "hotelling_region"],
    "table_rows": [len(t) for t in tables], "ols_df": ordinary.attrs["df"],
    "joint_t_df": intervals.attrs["df"], "hotelling_df": [region.attrs["df_numerator"],region.attrs["df_denominator"]],
    "full_metadata": states, "stata_parity_validated": False,
}))
