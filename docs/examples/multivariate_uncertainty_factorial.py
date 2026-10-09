"""Eight uncertainty and factorial moment contracts with complete saved output.

Source, frozen and native acceptance execute this exact file. Set
MULTIVARIATE_NEXT_RESULT_DIRECTORY to retain all tables/settings and fixtures.
"""
import hashlib
import json
from pathlib import Path
import random
import sys

import pandas as pd
import torch
import openecon as oe

torch.set_num_threads(2)
rng = random.Random(20261008568)
names = [f"x{j+1}" for j in range(6)]
outcomes = [f"y{j+1}" for j in range(3)]
spectral_rows, factorial_rows, rm_rows = [], [], []
group_labels = ["control", "early", "late"]
for i in range(603):
    f1, f2 = rng.gauss(0, 1.25), rng.gauss(0, .9)
    spectral_rows.append([
        .85*f1+.10*f2+rng.gauss(0,.65),
        .75*f1+.15*f2+rng.gauss(0,.70),
        .65*f1+.20*f2+rng.gauss(0,.75),
        .12*f1+.85*f2+rng.gauss(0,.65),
        .10*f1+.72*f2+rng.gauss(0,.70),
        .20*f1+.62*f2+rng.gauss(0,.75),
    ])
    a, b = ["A", "B"][i % 2], group_labels[i % 3]
    latent = rng.gauss(0, 1)
    values = [3+j+.3*(i % 2)+.2*(j+1)*(i % 3)+.15*(i % 2)*(i % 3)
              +.2*latent+rng.gauss(0,.8) for j in range(3)]
    factorial_rows.append([a,b,*values,i % 5])
    subject = rng.gauss(0,.8)
    for time in range(2):
        for condition in range(2):
            rm_rows.append([f"subject-{i:04}", b, time, condition,
                            4+subject+.2*(i % 3)+.4*time+.2*condition
                            +.15*(i % 3)*time+rng.gauss(0,.7),i % 4])

spectral = pd.DataFrame(spectral_rows, columns=names)
factorial = pd.DataFrame(factorial_rows, columns=["treatment","group",*outcomes,"w"])
rm_data = pd.DataFrame(rm_rows, columns=["subject","group","time","condition","response","w"])
target = [[.85,.10],[.75,.15],[.65,.20],[.12,.85],[.10,.72],[.20,.62]]
fits = {
    "covariance_pca": oe.pca_bootstrap(spectral,names,components=2,matrix="covariance",anchors=["x1","x4"],replications=199,seed=71),
    "correlation_pca": oe.pca_bootstrap(spectral,names,components=2,matrix="correlation",anchors=["x1","x4"],replications=199,seed=71),
    "multifactor_pf": oe.factor_multifactor_bootstrap(spectral,names,factors=2,anchors=["x1","x4"],replications=199,seed=72),
    "target_pf": oe.factor_multifactor_bootstrap(spectral,names,factors=2,target=target,replications=199,seed=72),
}


def moments(values, counts):
    x = torch.tensor(values,dtype=torch.float64)
    w = torch.tensor(counts,dtype=torch.float64)
    keep = w > 0
    x,w = x[keep],w[keep]
    n = int(w.sum())
    origin = x[0]
    offset = (w[:,None]*(x-origin)).sum(0)/n
    dev = (x-origin)-offset
    covariance = dev.T@(w[:,None]*dev)/(n-1)
    return (origin+offset).tolist(),covariance.tolist(),n


factorial_keys, factorial_means, factorial_covariances, factorial_counts = [],[],[],[]
for key, block in factorial.groupby(["treatment","group"],sort=True):
    mean,covariance,count = moments(block[outcomes].to_numpy(),block.w.to_numpy())
    factorial_keys.append(key)
    factorial_means.append(mean)
    factorial_covariances.append(covariance)
    factorial_counts.append(count)
factorial_summary = pd.DataFrame(factorial_means,columns=outcomes,
    index=pd.MultiIndex.from_tuples(factorial_keys,names=["treatment","group"]))
fits["frequency_factorial"] = oe.manova_factorial(factorial,outcomes,["treatment","group"],weights="w")
fits["summary_factorial"] = oe.manova_factorial_summary(factorial_summary,dict(zip(factorial_keys,factorial_covariances)),factorial_counts)

within_cells = [(0,0),(0,1),(1,0),(1,1)]
rm_means,rm_covariances,rm_counts,rm_groups = [],[],[],[]
for group,block in rm_data.groupby("group",sort=True):
    wide = block.pivot(index="subject",columns=["time","condition"],values="response").loc[:,within_cells]
    weights = block.groupby("subject",sort=True).w.first().loc[wide.index]
    mean,covariance,count = moments(wide.to_numpy(),weights.to_numpy())
    rm_groups.append(group)
    rm_means.append(mean)
    rm_covariances.append(covariance)
    rm_counts.append(count)
rm_summary = pd.DataFrame(rm_means,
    index=pd.Index(rm_groups,name="group"),
    columns=pd.MultiIndex.from_tuples(within_cells,names=["time","condition"]))
fits["frequency_rm"] = oe.rm_anova_fweight(rm_data,"response","subject",["time","condition"],weights="w",between=["group"])
fits["summary_rm"] = oe.rm_anova_summary(rm_summary,dict(zip(rm_groups,rm_covariances)),rm_counts,within=["time","condition"],between=["group"])
# The declaration order is also the eight-case proof order.
fits = {key:fits[key] for key in ("covariance_pca","correlation_pca","multifactor_pf","target_pf",
                                "frequency_rm","summary_rm","frequency_factorial","summary_factorial")}


def pack(result):
    # JSON preserves every numeric cell but has no dtype for an all-null column.
    # Retain rendering dtypes explicitly so unavailable p/df cells render exactly.
    return {"summary":oe.summary_state(result),"latex":result.to_latex(),
            "table_order":list(result),
            "table_dtypes":{name:[str(dtype) for dtype in frame.dtypes]
                            for name,frame in result.items()}}


def restore_pack(state):
    result = oe.restore_summary(state["summary"])
    assert json.loads(oe.summary_state(result)) == json.loads(state["summary"])
    for name,dtypes in state["table_dtypes"].items():
        frame = result[name]
        result[name] = frame.astype(dict(zip(frame.columns,dtypes)))
    result = type(result)({name:result[name] for name in state["table_order"]},
                          title=result.title,**result.attrs)
    assert json.loads(oe.summary_state(result)) == json.loads(state["summary"])
    assert result.to_latex() == state["latex"]
    return result


post,contrasts = {},{}
for name in ("frequency_rm","summary_rm","frequency_factorial","summary_factorial"):
    result = fits[name]
    if name.endswith("rm"):
        k = len(result.attrs["rm_contrast_state"]["between_design_columns"])
        M = [[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]]
        function = oe.rm_mtest
    else:
        k = len(result.attrs["manova_contrast_state"]["design_columns"])
        M = [[-1.,0.],[1.,-1.],[0.,1.]]
        function = oe.manova_contrast
    L = torch.eye(k,dtype=torch.float64)[1:3].tolist()
    null = [[0.,0.],[0.,0.]]
    contrasts[name] = {"L":L,"M":M,"null":null}
    post[name] = function(result,L,M=M,null=null)

encoded = {}
for name,result in fits.items():
    state = pack(result)
    restored = restore_pack(state)
    payload = {"fit":state,"post":pack(post[name]) if name in post else None}
    if payload["post"] is not None:
        restore_pack(payload["post"])
    encoded[name] = json.dumps(payload,sort_keys=True,allow_nan=False)

fixtures = {
    "spectral":spectral.to_dict(orient="split"),
    "factorial":factorial.to_dict(orient="split"),
    "rm_data":rm_data.to_dict(orient="split"),
    "target":target,"contrasts":contrasts,
    "factorial_summary":{"keys":factorial_keys,"means":factorial_means,"covariances":factorial_covariances,"counts":factorial_counts},
    "rm_summary":{"groups":rm_groups,"cells":within_cells,"means":rm_means,"covariances":rm_covariances,"counts":rm_counts},
}
if "MULTIVARIATE_NEXT_RESULT_DIRECTORY" in globals():
    directory = Path(globals()["MULTIVARIATE_NEXT_RESULT_DIRECTORY"])
    directory.mkdir(parents=True,exist_ok=True)
    for name,content in encoded.items():
        (directory/(name+".json")).write_text(content)
    (directory/"fixtures.json").write_text(json.dumps(fixtures,sort_keys=True,allow_nan=False))

if "display" in globals():
    for name,result in fits.items():
        globals()["display"](result["estimates"] if name in list(fits)[:4]
                             else result["within"] if name.endswith("rm") else result["multivariate"])

print("MULTIVARIATE_NEXT_EIGHT_OK "+json.dumps({
    "cases":list(fits),"frozen":bool(getattr(sys,"frozen",False)),
    "fixture_rows":len(spectral),"rm_long_rows":len(rm_data),
    "bootstrap_replications":199,"full_saved_roundtrip":True,
    "fit_table_counts":{name:len(result) for name,result in fits.items()},
    "hashes":{name:hashlib.sha256(content.encode()).hexdigest() for name,content in encoded.items()},
}))
