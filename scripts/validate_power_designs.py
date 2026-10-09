"""Predeclared synthetic experiments, independent statistics, two fixed seeds.

32 cells x 1000 replications. Exact normal/t/F designs have gate 4 MC SE+.005;
Pearson asymptotic designs have gate 4 MC SE+.04. No adaptive retries/seed search.
The Pearson gate measures a bounded approximation, not finite-sample exactness.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats
import torch

import openecon as oe

SEEDS = (730272, 910279)
REPLICATIONS = 1000


def run():
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for regime in (0, 1):
            r = REPLICATIONS
            def record(name, inputs, rejection, approximate=False):
                expected = float(getattr(oe, name)(**inputs)["plan"].iloc[0].power)
                empirical = float(np.mean(rejection))
                se = math.sqrt(expected * (1 - expected) / r)
                gate = 4 * se + (.04 if approximate else .005)
                rows.append(dict(seed=seed, regime=regime, method=name, inputs=inputs,
                                 planned=r, completed=r, failed=0, expected_power=expected,
                                 rejection_rate=empirical, monte_carlo_se=se, absolute_error=abs(empirical-expected),
                                 predeclared_gate=gate, approximate_law=approximate,
                                 passed=abs(empirical-expected) <= gate))
            n, effect = (8,.4) if regime == 0 else (34,.5)
            x = rng.normal(effect,1,size=(r,n))
            t = x.mean(axis=1)/(x.std(axis=1,ddof=1)/np.sqrt(n))
            record("power_tmean",dict(effect=effect,sd=1.,n=n),np.abs(t)>stats.t.isf(.025,n-1))
            ratio, n = (.37,31) if regime == 0 else (2.3,43)
            n2 = math.ceil(ratio*n)
            x, y = rng.normal(0,1,size=(r,n)), rng.normal(.5,1,size=(r,n2))
            pooled = ((n-1)*x.var(axis=1,ddof=1)+(n2-1)*y.var(axis=1,ddof=1))/(n+n2-2)
            t = (y.mean(axis=1)-x.mean(axis=1))/np.sqrt(pooled*(1/n+1/n2))
            record("power_ttwomeans",dict(effect=.5,sd=1.,n=n,ratio=ratio),np.abs(t)>stats.t.isf(.025,n+n2-2))
            n, rho = (8,-.4) if regime == 0 else (30,.7)
            first, innovation = rng.normal(size=(r,n)),rng.normal(size=(r,n))
            second = .5 + 1.4*(rho*first+np.sqrt(1-rho**2)*innovation)
            difference = second-first
            t = difference.mean(axis=1)/(difference.std(axis=1,ddof=1)/np.sqrt(n))
            record("power_tpaired",dict(effect=.5,sd1=1.,sd2=1.4,correlation=rho,n=n),np.abs(t)>stats.t.isf(.025,n-1))
            k, n, f = (3,6,.1) if regime == 0 else (5,30,.25)
            means = np.arange(k,dtype=float)-np.mean(np.arange(k))
            means *= f/np.sqrt(np.mean(means**2))
            x = rng.normal(size=(r,k,n))+means[None,:,None]
            groups = x.mean(axis=2)
            between = n*np.sum((groups-groups.mean(axis=1)[:,None])**2,axis=1)/(k-1)
            within = np.sum((x-groups[:,:,None])**2,axis=(1,2))/(k*(n-1))
            record("power_anova",dict(effect=f,groups=k,n=n),between/within>stats.f.isf(.05,k-1,k*(n-1)))
            p,q,n,f2 = (3,1,25,.1) if regime == 0 else (8,3,100,.15)
            raw = rng.normal(size=(n,p))
            basis = np.linalg.qr(np.column_stack([np.ones(n),raw]))[0]
            y = rng.normal(size=(r,n))+np.sqrt(n*f2)*basis[:,1][None,:]
            fitted = y @ basis
            numerator = np.sum(fitted[:,1:q+1]**2,axis=1)/q
            residual = y-fitted @ basis.T
            denominator = np.sum(residual**2,axis=1)/(n-p-1)
            record("power_regression",dict(effect=f2,predictors=p,tested=q,n=n),numerator/denominator>stats.f.isf(.05,q,n-p-1))
            n,m,icc = (4,6,0.) if regime == 0 else (12,17,.3)
            shared = rng.normal(size=(r,2,n,1))*np.sqrt(icc)
            members = shared+rng.normal(size=(r,2,n,m))*np.sqrt(1-icc)
            members[:,1,:,:] += .5
            difference = members[:,1,:,:].mean(axis=(1,2))-members[:,0,:,:].mean(axis=(1,2))
            se = np.sqrt(2*(1+(m-1)*icc)/(n*m))
            record("power_cluster_mean",dict(effect=.5,sd=1.,cluster_size=m,icc=icc,n=n),np.abs(difference/se)>stats.norm.isf(.025))
            n,strength = (120,.7) if regime == 0 else (400,1.)
            null = np.array([.25]*4)
            proposed = np.array([.4,.3,.2,.1])
            cells = rng.multinomial(n,null+strength*(proposed-null),size=r)
            statistic = np.sum((cells-n*null)**2/(n*null),axis=1)
            record("power_gof",dict(null_probabilities=null.tolist(),proposed_probabilities=proposed.tolist(),strength=strength,n=n),statistic>stats.chi2.isf(.05,3),True)
            joint = np.array([[.35,.15],[.15,.35]])
            null = np.outer(joint.sum(axis=1),joint.sum(axis=0))
            counts = rng.multinomial(n,(null+strength*(joint-null)).ravel(),size=r).reshape(r,2,2)
            estimated = counts.sum(axis=2)[:,:,None]*counts.sum(axis=1)[:,None,:]/n
            assert np.all(estimated>0)
            statistic = np.sum((counts-estimated)**2/estimated,axis=(1,2))
            record("power_independence",dict(joint_probabilities=joint.tolist(),strength=strength,n=n),statistic>stats.chi2.isf(.05,1),True)
    return dict(status="passed" if all(row["passed"] for row in rows) else "failed",
                seeds=list(SEEDS), replications_per_cell=REPLICATIONS, planned_cells=32,
                completed_cells=len(rows), planned_replications=32000, completed_replications=32000,
                failed_replications=0, rows=rows,
                scope="iid normal t, fixed Gaussian ANOVA/F, known Gaussian cluster covariance; two bounded Pearson approximations",
                not_verified="nonnormal, Welch, estimated ICC, unequal clusters, rare cells or licensed vendor execution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    receipt = run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({key:receipt[key] for key in ("status","completed_cells","completed_replications","failed_replications")}))
    if receipt["status"] != "passed":
        raise SystemExit(1)
