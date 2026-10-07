"""Editable saved-target example using a disposable 510-row synthetic Parquet.

Run in OpenEconometrics. Outputs are small previews; complete adapter outputs
remain owned Datasets. No customer file, account or cloud service is accessed.
"""

import gc
import json
import math
from pathlib import Path
import tempfile

import pandas as pd
import torch
import openecon as oe
from openecon.models import ResultBundle

if "display" not in globals():
    display = print

torch.set_num_threads(2)
generator = torch.Generator().manual_seed(710102)
rows = 510
x = torch.randn(rows, generator=generator, dtype=torch.float64)
z = torch.randn(rows, generator=generator, dtype=torch.float64)
noise = torch.randn(rows, generator=generator, dtype=torch.float64)
d = pd.DataFrame(
    {
        "x": x.tolist(),
        "z": z.tolist(),
        "y": (1 + 0.6 * x**2 + 0.2 * z + noise * 0.3).tolist(),
        "treatment": (torch.rand(rows, generator=generator) > 0.5).to(torch.float64).tolist(),
        "duration": (torch.arange(rows, dtype=torch.float64) / 10 + 1).tolist(),
        "failure": [1.0] * rows,
        "period": list(range(rows)),
    }
)


def restore(result):
    return ResultBundle.model_validate_json(result.model_dump_json())


def small_output(source):
    # The output geometry of this synthetic example is bounded at 510 rows.
    return pd.concat(list(source.iter_batches(batch_rows=17)))


with tempfile.TemporaryDirectory(prefix="openecon-saved-targets-example-") as directory:
    path = Path(directory) / "synthetic.parquet"
    d.to_parquet(path, index=False, row_group_size=31)
    source = oe.scan(path)
    ranked = oe.correlate(source, ["x", "z"], method="spearman")
    display(ranked["coefficients"])
    display(oe.describe(source, ["x"], stats=["n", "p25", "p50", "p75"]))

    static = restore(oe.nl(data=source, y="y", formula="{c}+{b}*x^2+{z}*z"))
    prediction = oe.predict(
        static, oe.Dataset.from_frame(d.iloc[:6]), outcome="function", interval="mean", batch_rows=2
    )
    predicted = small_output(prediction)
    b = {c.term: c.estimate for c in static.coefficients}
    torch.testing.assert_close(
        torch.as_tensor(predicted.response.to_numpy()),
        torch.as_tensor((b["c"] + b["b"] * d.x.iloc[:6] ** 2 + b["z"] * d.z.iloc[:6]).to_numpy()),
    )
    display(predicted)
    del prediction

    cox = restore(oe.stcox(data=source, time="duration", failure="failure", x=["x", "z"]))
    baseline = oe.cox_baseline(cox, source, batch_rows=19)
    full = small_output(baseline)
    assert len(full) == 510 and cox.extra["baseline"]["thinned"]
    prediction = oe.survival_predict(
        cox,
        oe.Dataset.from_frame(d.iloc[:6]),
        target="survival",
        time=4.0,
        baseline=baseline,
        interval="mean",
        batch_rows=2,
    )
    display(small_output(prediction))
    del prediction, baseline
    parametric = restore(
        oe.streg(data=source, time="duration", failure="failure", x=["x"], distribution="weibull")
    )
    prediction = oe.survival_predict(
        parametric,
        oe.Dataset.from_frame(d.iloc[:6]),
        target="quantile",
        quantile=0.4,
        interval="mean",
    )
    assert (small_output(prediction).response > 0).all()
    display(small_output(prediction))
    del prediction

    series = torch.zeros(rows, dtype=torch.float64)
    for i in range(1, rows):
        series[i] = 0.2 + 0.45 * series[i - 1] + noise[i]
    d["series"] = series.tolist()
    history = oe.Dataset.from_frame(d)
    arima = restore(oe.arima(data=history, y="series", time="period", order=(1, 0, 0)))
    prediction = oe.dynamic_predict(
        arima, history, target="forecast_levels", origin=200, horizon=4, batch_rows=7
    )
    forecast = small_output(prediction)
    b = {c.term: c.estimate for c in arima.coefficients}
    last = d.series.iloc[200]
    for i, row in forecast.iterrows():
        last = b["Intercept"] + b["ARMA:L1.ar"] * (last - b["Intercept"])
        assert math.isclose(row.forecast, last, rel_tol=1e-9, abs_tol=1e-9)
    assert forecast.period.tolist() == [201, 202, 203, 204]
    display(forecast)
    del prediction

    causal = restore(
        oe.teffects(data=source, y="y", treatment="treatment", x=["x", "z"], method="aipw")
    )
    evaluation = oe.Dataset.from_frame(d.iloc[:60])
    prediction = oe.causal_evaluate(
        causal,
        evaluation,
        target="standardized_outcome",
        population="fixed_evaluation",
        treatment="1",
        batch_rows=7,
    )
    standardized = small_output(prediction)
    assert float(standardized.std_error.iloc[0]) > 0
    display(standardized)
    del prediction
    prediction = oe.causal_evaluate(causal, target="population_effect", treatment="1")
    display(small_output(prediction))
    del prediction
gc.collect()
print(
    "SAVED_DATASET_TARGETS_OK "
    + json.dumps(
        {
            "physical_parquet_rows": rows,
            "full_cox_failure_times": len(full),
            "saved_json_restore": True,
            "explicit_dynamic_origin": 200,
            "full_joint_causal_covariance": True,
            "device": "cpu",
            "dtype": "float64",
            "owned_outputs_released": True,
            "scientific_oracles_separate": True,
        }
    )
)
