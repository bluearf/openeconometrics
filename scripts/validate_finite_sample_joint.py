"""Predeclared Gaussian joint-inference grid; development oracles only."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy import stats
import torch
import openecon as oe

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/evidence/finite-sample-joint-2026-10-07/protocol.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new receipt; previous outcomes are preserved.")
    protocol = json.loads(PROTOCOL.read_text())
    torch.set_num_threads(1)
    start = time.monotonic()
    cells = []
    for index, (n, rho, weighted, nulls) in enumerate(itertools.product(
        protocol["ols"]["n"], protocol["ols"]["rho"], protocol["ols"]["weighted"], protocol["ols"]["nulls"]
    )):
        config = protocol["ols"]
        rng = np.random.default_rng(207000+index)
        x = rng.multivariate_normal([0,0], [[1,rho],[rho,1]], size=n)
        weights = rng.uniform(.5,2,n) if weighted else np.ones(n)
        design = np.column_stack([np.ones(n),x])
        precision = np.linalg.inv(design.T@(weights[:,None]*design))
        truth = np.array([.3,1. if nulls=="partial" else 0.,0.])
        failures, rejections, false_rejections, max_error = [], 0, 0, 0.
        for rep in range(config["replications"]):
            y = design@truth+rng.normal(size=n)/np.sqrt(weights)
            data = pd.DataFrame({"y":y,"x":x[:,0],"z":x[:,1],"w":weights})
            try:
                model = oe.ols(data=data,y="y",x=["x","z"],covariance="nonrobust",
                    **({"weights":"w","weight_type":"aweight"} if weighted else {}))
                result = oe.ols_stepdown(model,terms=["x","z"],
                    error_model="known_precision_gaussian" if weighted else "iid_gaussian",
                    draws=config["null_draws"],seed=207100000+index*1000+rep)
                beta = precision@(design.T@(weights*y))
                residual = y-design@beta
                covariance = (residual@(weights*residual))/(n-3)*precision
                expected = beta[1:]/np.sqrt(np.diag(covariance)[1:])
                max_error=max(max_error,float(np.max(np.abs(result.statistic-expected))))
                true_positions=[0,1] if nulls=="full" else [1]
                rejections+=int(result.reject.iloc[true_positions].any())
                false_rejections+=int(nulls=="partial" and result.reject.iloc[0])
                assert max_error<1e-9
            except Exception as e:
                failures.append({"replication":rep,"code":getattr(e,"code",type(e).__name__)})
        rate=rejections/config["replications"]
        cells.append(dict(method="ols_stepdown",n=n,rho=rho,weighted=weighted,nulls=nulls,
            replications=config["replications"],rejections=rejections,false_rejections=false_rejections,
            rejection_rate=rate,failures=failures,max_statistic_error=max_error,
            passed=not failures and rate<=config["alpha"]+config["upper_size_tolerance"]))
        print(json.dumps({"method":"ols_stepdown","cell":index,"passed":cells[-1]["passed"]}),flush=True)
    for index,(df,p,rho,seed) in enumerate(itertools.product(
        protocol["joint_t"]["df"],protocol["joint_t"]["dimensions"],protocol["joint_t"]["rho"],protocol["joint_t"]["calibration_seeds"]
    )):
        config=protocol["joint_t"]
        correlation=np.full((p,p),rho)
        np.fill_diagonal(correlation,1.)
        failures=[]
        rate=None
        try:
            result=oe.simultaneous_t_ci([0.]*p,correlation,df=df,draws=config["null_draws"],seed=seed,
                family_description="Predeclared Gaussian fixed-design coefficient family",
                pivot_description="Known shape and common independent chi-square residual scale")
            rng=np.random.default_rng(208100000+index)
            noise=rng.multivariate_normal(np.zeros(p),correlation,size=config["independent_evaluations"])
            pivot=noise/np.sqrt(rng.chisquare(df,config["independent_evaluations"])/df)[:,None]
            rejections=int((np.max(np.abs(pivot),axis=1)>result.attrs["critical_value"]).sum())
            rate=rejections/config["independent_evaluations"]
        except Exception as e:
            failures.append(getattr(e,"code",type(e).__name__))
        cells.append(dict(method="simultaneous_t_ci",df=df,dimensions=p,rho=rho,seed=seed,
            independent_evaluations=config["independent_evaluations"],rejection_rate=rate,
            failures=failures,passed=not failures and abs(rate-config["alpha"])<=config["absolute_size_tolerance"]))
    print(json.dumps({"method":"simultaneous_t_ci","cells":48}),flush=True)
    for index,(n,p,rho) in enumerate(itertools.product(protocol["hotelling"]["n"],protocol["hotelling"]["dimensions"],protocol["hotelling"]["rho"])):
        config=protocol["hotelling"]
        rng=np.random.default_rng(209000+index)
        covariance=np.full((p,p),rho)
        np.fill_diagonal(covariance,1.)
        names=[f"x{i}" for i in range(p)]
        failures=[]
        rejections=projected_rejections=0
        max_error=0.
        for rep in range(config["replications"]):
            sample=rng.multivariate_normal(np.zeros(p),covariance,size=n)
            try:
                result=oe.hotelling_region(pd.DataFrame(sample,columns=names),names,sampling_model="iid_multivariate_normal")
                center=np.array(result.attrs["region_center"])
                scale=np.array(result.attrs["region_scale"])
                precision=np.array(result.attrs["normalized_region_precision"])
                statistic=(center/scale)@precision@(center/scale)
                sample_cov=np.atleast_2d(np.cov(sample,rowvar=False))
                independent=n*sample.mean(0)@np.linalg.inv(sample_cov)@sample.mean(0)
                radius=p*(n-1)/(n-p)*stats.f.isf(config["alpha"],p,n-p)
                max_error=max(max_error,abs(statistic-independent)/max(1.,abs(independent)))
                assert max_error<1e-9
                assert abs(result.attrs["region_radius_squared"]-radius)/radius<1e-10
                rejected=statistic>radius
                projected=((result.projected_ci_low>0)|(result.projected_ci_high<0)).any()
                assert not projected or rejected
                rejections+=int(rejected)
                projected_rejections+=int(projected)
            except Exception as e:
                failures.append({"replication":rep,"code":getattr(e,"code",type(e).__name__)})
        rate=rejections/config["replications"]
        cells.append(dict(method="hotelling_region",n=n,dimensions=p,rho=rho,
            replications=config["replications"],rejections=rejections,rejection_rate=rate,
            coordinate_projection_rejections=projected_rejections,failures=failures,max_quadratic_relative_error=max_error,
            passed=not failures and abs(rate-config["alpha"])<=config["absolute_size_tolerance"]))
        print(json.dumps({"method":"hotelling_region","cell":index,"passed":cells[-1]["passed"]}),flush=True)
    receipt=dict(status="passed" if all(c["passed"] for c in cells) else "failed",cells=cells,
        protocol=json.loads(PROTOCOL.read_text()),protocol_sha256=digest(PROTOCOL),
        source_sha256=digest(ROOT/"src/openecon/econometrics/postest/finite_sample.py"),
        validator_sha256=digest(__file__),elapsed_seconds=time.monotonic()-start,
        versions={"numpy":np.__version__,"pandas":pd.__version__,"torch":torch.__version__},
        failures_replaced=False,limits=protocol["limits"])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt,handle,indent=2)
        handle.write("\n")
    print(json.dumps({"status":receipt["status"],"cells":len(cells)}),flush=True)
    return 0 if receipt["status"]=="passed" else 1


if __name__=="__main__":
    raise SystemExit(main())
