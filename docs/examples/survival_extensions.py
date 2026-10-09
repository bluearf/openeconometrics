"""Eight synthetic iid survival stages: complete JSON, LaTeX and input replay.

Displays two complete result tables per method; all covariance/settings tables
are saved. Turnbull bounds concern identification, not sampling uncertainty.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import openecon as oe
import torch

METHODS=("stinterval_exponential","stinterval_weibull","stinterval_lognormal","stinterval_loglogistic",
         "turnbull","cumulative_incidence","cause_specific_hazard","cif_compare")
latent=[math.exp(-1.1+3.2*(i+.5)/48+.18*math.sin(i*1.71)) for i in range(48)]
lower,upper=[],[]
for i,t in enumerate(latent):
    if i%7==0:
        lower.append(t)
        upper.append(t)
    elif i%11==1:
        lower.append(0.)
        upper.append(t)
    elif i%9==2:
        lower.append(t)
        upper.append(None)
    else:
        lower.append(t*.85)
        upper.append(t*1.15)
time=[.5+.5*(i%16) for i in range(48)]
event=[0 if i%7==0 else 1 if i%3==0 else 2 for i in range(48)]
groups=["A" if i%2==0 else "B" for i in range(48)]
previous_threads=torch.get_num_threads()
torch.set_num_threads(1)
try:
    proof,encoded,results={}, {}, {}
    for name in METHODS:
        if name.startswith("stinterval_"):
            inputs=dict(lower=lower,upper=upper,times=[0,.25,.5,1,2,4,8])
            shown=("parameters","survival")
        elif name=="turnbull":
            inputs=dict(lower=[0,1,2,3,4,5,1,2,3,4],upper=[1,1,2,4,4,None,2,3,4,5],
                        times=[0,1,1.5,2,2.5,3,4,4.5,5,6])
            shown=("support","cdf_bounds")
        elif name=="cif_compare":
            inputs=dict(time=time,event=event,group=groups,cause=1,times=[1,2,4,6])
            shown=("contrast","joint")
        else:
            inputs=dict(time=time,event=event,causes=[1,2],times=[0,1,2,4,6])
            shown=("curve","risksets")
        result=getattr(oe,name)(**inputs)
        assert result.attrs["complete_inputs_saved"] and result.attrs["contract"]=="survival_ext_v1"
        assert result.attrs["device"]=="cpu"
        payload={"inputs":inputs,"attrs":result.attrs,
                 "tables":{key:frame.astype(object).where(frame.notna(),None).to_dict(orient="split") for key,frame in result.items()},
                 "latex":result.to_latex()}
        content=json.dumps(payload,sort_keys=True,allow_nan=False)
        saved=json.loads(content)
        replay=getattr(oe,name)(**saved["inputs"])
        assert replay.attrs==saved["attrs"] and replay.to_latex()==saved["latex"]
        assert {key:frame.astype(object).where(frame.notna(),None).to_dict(orient="split") for key,frame in replay.items()}==saved["tables"]
        if name.startswith("stinterval_"):
            restored=oe.interval_survival_predict(saved["attrs"],times=inputs["times"])
            assert restored.attrs["refit"] is False
            assert restored["survival"].equals(result["survival"])
            assert restored["survival_covariance"][["row","column"]].equals(result["survival_covariance"][["row","column"]])
            assert torch.allclose(torch.tensor(restored["survival_covariance"].covariance.tolist(),dtype=torch.float64,device="cpu"),
                                  torch.tensor(result["survival_covariance"].covariance.tolist(),dtype=torch.float64,device="cpu"),rtol=1e-12,atol=1e-15)
        encoded[name]=content
        results[name]=result
        proof[name]={"table_rows":{key:len(frame) for key,frame in result.items()},
                     "displayed_keys":list(shown),"sha256":hashlib.sha256(content.encode()).hexdigest(),
                     "contract":result.attrs["contract"]}
        if "display" in globals():
            for key in shown:
                globals()["display"](result[key])
    if "SURVIVAL_RESULT_DIRECTORY" in globals():
        directory=Path(globals()["SURVIVAL_RESULT_DIRECTORY"])
        directory.mkdir(parents=True,exist_ok=True)
        for name,content in encoded.items():
            (directory/(name+".json")).write_text(content)
    forbidden=[name for name in ("scipy","statsmodels","linearmodels") if name in sys.modules]
    assert not forbidden
    print("SURVIVAL_ACCEPTANCE_OK "+json.dumps({"methods":proof,"frozen":bool(getattr(sys,"frozen",False)),
          "full_input_replay_verified":True,"all_eight_contracts_verified":True,"all_four_saved_predictions_verified":True,
          "saved_tables":sum(len(result) for result in results.values()),"displayed_tables":16,
          "third_party_estimation_imports":forbidden},sort_keys=True,allow_nan=False))
finally:
    torch.set_num_threads(previous_threads)
