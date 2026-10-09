"""Development-only seeded competing-risk sampling validation; no runtime oracle imports.

Independent exponential cause/censor races have closed-form CIF and cumulative
cause-hazard targets. Full public API calls measure coverage and covariance,
rather than treating a scalar coefficient match as uncertainty validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

import openecon as oe


def race(rng, n, rates):
    latent = rng.exponential(1/np.asarray(rates),size=(n,3))
    winner = latent.argmin(axis=1)
    return latent.min(axis=1).tolist(), np.where(winner==2,0,winner+1).tolist()


def summary(estimates, covariance, target, coverage, reps):
    estimates, covariance = np.asarray(estimates), np.asarray(covariance)
    empirical = np.cov(estimates,rowvar=False,ddof=1)
    mean_covariance = covariance.mean(axis=0)
    mean_variance = mean_covariance.diagonal()
    bias_z = np.abs(estimates.mean(axis=0)-target)/np.sqrt(empirical.diagonal()/reps)
    variance_ratio = mean_variance/empirical.diagonal()
    # All gates are declared, include Monte Carlo error, and are diagnostic
    # bounded sampling checks rather than exact finite-sample coverage claims.
    coverage_rate = np.asarray(coverage).mean(axis=0)
    coverage_gate = .02+4*np.sqrt(.95*.05/reps)
    variance_gate = .04+4*np.sqrt(2/(reps-1))
    assert np.all(bias_z < 5), bias_z
    assert np.all(np.abs(coverage_rate-.95) < coverage_gate), coverage_rate
    assert np.all(np.abs(variance_ratio-1) < variance_gate), variance_ratio
    normalized_error = np.abs(mean_covariance-empirical)/np.sqrt(np.outer(empirical.diagonal(),empirical.diagonal()))
    assert normalized_error.max() < variance_gate, normalized_error
    return dict(status="passed",replications=reps,target=target.tolist(),maximum_absolute_bias_MC_z=float(bias_z.max()),
                coverage=coverage_rate.tolist(),coverage_deviation_gate=float(coverage_gate),
                mean_covariance=mean_covariance.tolist(),empirical_covariance=empirical.tolist(),
                variance_ratio=variance_ratio.tolist(),covariance_normalized_error_gate=float(variance_gate),
                maximum_normalized_full_covariance_error=float(normalized_error.max()))


def validate(reps):
    root = Path(__file__).resolve().parents[1]
    source_hashes = {name:hashlib.sha256((root / "src/openecon/econometrics/survival_ext" / name).read_bytes()).hexdigest()
                     for name in ("common.py","competing.py")}
    rng = np.random.default_rng(163377)
    grid = np.asarray([.25,.5,1.,1.5])
    rates = [.7,.4,.3]
    total = sum(rates[:2])
    cif = np.column_stack([rates[j]/total*(-np.expm1(-total*grid)) for j in range(2)]).reshape(-1)
    hazard = np.column_stack([rates[j]*grid for j in range(2)]).reshape(-1)
    collected = {name:dict(estimates=[],covariance=[],coverage=[]) for name in ("cumulative_incidence","cause_specific_hazard","cif_compare")}
    other=[.5,.6,.3]
    # Public contrasts are explicitly group_2 minus group_1; labels A,B
    # therefore target the second rate less the first rate.
    diff=(other[0]-rates[0])/total*(-np.expm1(-total*grid))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for i in range(reps):
            time,event=race(rng,240,rates)
            for name,target in (("cumulative_incidence",cif),("cause_specific_hazard",hazard)):
                r=getattr(oe,name)(time,event,causes=[1,2],times=grid)
                frame=r["curve"]
                collected[name]["estimates"].append(frame.estimate.to_numpy())
                collected[name]["covariance"].append(r["covariance"].to_numpy())
                collected[name]["coverage"].append(((frame.ci_lower.to_numpy()<=target)&(target<=frame.ci_upper.to_numpy())).tolist())
            other_time,other_event=race(rng,240,other)
            r=oe.cif_compare(time+other_time,event+other_event,["A"]*240+["B"]*240,cause=1,times=grid)
            frame=r["contrast"]
            assert frame.group_1.tolist()==["A"]*len(grid) and frame.group_2.tolist()==["B"]*len(grid)
            collected["cif_compare"]["estimates"].append(frame.difference.to_numpy())
            collected["cif_compare"]["covariance"].append(r["covariance"].to_numpy())
            collected["cif_compare"]["coverage"].append(((frame.ci_lower.to_numpy()<=diff)&(diff<=frame.ci_upper.to_numpy())).tolist())
    finally:
        torch.set_num_threads(previous)
    results={name:summary(**collected[name],target=target,reps=reps)
             for name,target in (("cumulative_incidence",cif),("cause_specific_hazard",hazard),("cif_compare",diff))}
    assert source_hashes == {name:hashlib.sha256((root / "src/openecon/econometrics/survival_ext" / name).read_bytes()).hexdigest() for name in source_hashes}
    return dict(status="passed",seed=163377,source_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                source_modules_sha256=source_hashes,
                n_per_group=240,times=grid.tolist(),independent_exponential_race_rates=[rates,other],
                iid_independent_censoring=True,full_public_API_calls=3*reps,random_replications=reps,
                empirical_uncertainty=results,licensed_vendor_run=False,whole_product_parity=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications",type=int,default=800)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.replications < 400:
        parser.error("At least 400 replications are required for the declared gates.")
    result=validate(args.replications)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("status","full_public_API_calls","random_replications")}))
