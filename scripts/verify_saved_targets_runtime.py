"""Verify saved targets in an owned packaged console, with synthetic data only.

Reuses the local-runtime lifecycle/cancellation harness. No installed user
application, human project or cloud account is accessed. Independent scientific
oracles remain in source tests; this checks packaging, execution and disk output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import verify_streaming_runtime as harness


MODEL = r"""
import gc
import math
import sys
from openecon.models import ResultBundle
from openecon.analysis_contracts import AnalysisError
estimator = __ESTIMATOR__
def collect(source):
    return pd.concat(list(source.iter_batches(batch_rows=17)))
def restore(result):
    return ResultBundle.model_validate_json(result.model_dump_json())
small = frame.iloc[:320].dropna().copy()
small['d'] = small.binary
small['t'] = list(range(len(small)))
extra = {}
if estimator == 'external_helpers':
    disk = oe.scan(fixture_path)
    for method in ['spearman', 'kendall']:
        actual = oe.correlate(data=disk, columns=['x','z'], method=method)
        expected = oe.correlate(data=frame, columns=['x','z'], method=method)
        torch.testing.assert_close(torch.as_tensor(actual['coefficients'].to_numpy(dtype=float)),
                                   torch.as_tensor(expected['coefficients'].to_numpy(dtype=float)))
    described = oe.describe(data=disk, columns=['x'], by='category', stats=['n','p25','p50','p75'])
    extra.update(physical_rows=n, exact_rank_methods=['spearman','kendall'], grouped_quantiles=True)
    model = oe.ols(data=oe.Dataset.from_frame(small), y='y', x=['x','z'])
elif estimator == 'static_targets':
    model = restore(oe.nl(data=oe.Dataset.from_frame(small), y='y', formula='{c}+{b}*x^2+{z}*z'))
    prediction = oe.predict(model, oe.Dataset.from_frame(small), outcome='function', interval='mean', batch_rows=7)
    out = collect(prediction)
    b = {c.term:c.estimate for c in model.coefficients}
    want = b['c']+b['b']*small.x**2+b['z']*small.z
    torch.testing.assert_close(torch.as_tensor(out.response.to_numpy()),torch.as_tensor(want.to_numpy()))
    assert (out.std_error>0).all()
    del prediction
    extra.update(saved_json=True, formula_function=True, full_delta=True)
elif estimator == 'survival_targets':
    small['failure'] = 1.
    small['duration'] = [1.+i/10. for i in range(len(small))]
    model = restore(oe.stcox(data=oe.Dataset.from_frame(small),time='duration',failure='failure',x=['x','z']))
    baseline = oe.cox_baseline(model,oe.Dataset.from_frame(small),batch_rows=11)
    table = collect(baseline)
    assert len(table)==len(small) and baseline.metadata['analysis']['full_step_function']
    prediction = oe.survival_predict(model,oe.Dataset.from_frame(small.iloc[:9]),target='survival',time=4.,baseline=baseline,interval='mean',batch_rows=3)
    out = collect(prediction)
    assert out.response.between(0,1).all() and (out.std_error>=0).all()
    del prediction,baseline
    parametric = restore(oe.streg(data=oe.Dataset.from_frame(small),time='duration',failure='failure',x=['x'],distribution='weibull'))
    prediction=oe.survival_predict(parametric,oe.Dataset.from_frame(small.iloc[:9]),target='quantile',quantile=.4,interval='mean')
    assert (collect(prediction).response>0).all()
    del prediction
    extra.update(full_failure_times=len(table),cox_and_streg=True,saved_json=True)
elif estimator == 'dynamic_targets':
    model=restore(oe.arima(data=oe.Dataset.from_frame(small),y='y',time='t',order=(1,0,0)))
    prediction=oe.dynamic_predict(model,oe.Dataset.from_frame(small),target='forecast_levels',origin=120,horizon=3,batch_rows=5)
    out=collect(prediction)
    assert out.period.tolist()==[121,122,123] and (out.std_error>0).all()
    assert prediction.metadata['analysis']['ordered_history_rows']==121
    del prediction
    extra.update(integer_origin_excludes_future=True,saved_json=True,native_forecast_uncertainty=True)
else:
    model=restore(oe.teffects(data=oe.Dataset.from_frame(small),y='y',treatment='d',x=['x','z'],method='aipw'))
    prediction=oe.causal_evaluate(model,oe.Dataset.from_frame(small),target='standardized_outcome',population='fixed_evaluation',treatment='1',batch_rows=7)
    out=collect(prediction)
    assert len(out)==1 and float(out.std_error.iloc[0])>0
    assert len(model.extra['evaluation_state']['covariance'])>len(model.coefficients)
    del prediction
    rd=restore(oe.rdrobust(data=oe.Dataset.from_frame(small),y='y',running='x',h=1.,b=1.4))
    prediction=oe.causal_evaluate(rd,target='cutoff_side',side='left',point=0.)
    assert float(collect(prediction).std_error.iloc[0])>0
    del prediction
    extra.update(full_nuisance_joint_state=True,global_fixed_population=True,rd_side_covariance=True)
gc.collect()
owned_scratch=Path(os.environ['OPENECON_SCRATCH_DIRECTORY'])
assert not list(owned_scratch.iterdir()), list(owned_scratch.iterdir())
if getattr(sys,'frozen',False):
    import importlib.util
    assert importlib.util.find_spec('scipy') is None
    assert importlib.util.find_spec('statsmodels') is None
display(model)
print('__OPENECON_STREAMING_CHECK__'+json.dumps({'estimator':estimator,'nobs':model.nobs,
    'passed':True,'normal_scratch_reclaimed':True,'source_scientific_oracles_separate':True,**extra}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    harness.MODEL = MODEL
    harness.COMBINATIONS = [
        (name, "native")
        for name in (
            "external_helpers",
            "static_targets",
            "survival_targets",
            "dynamic_targets",
            "causal_targets",
        )
    ]
    report = harness.verify(args.runtime, args.output)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
