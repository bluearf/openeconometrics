"""MANOVA via blocked TSQR and bounded cell covariance reductions."""
from __future__ import annotations

import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, kernel_call
from openecon.resources import plan_workspace
from . import common as c
from .anova import ANOVA_COLUMNS, anova_rows
from .manova import multivariate_tests
from .replay import Moments, Replay
from .streaming_anova import ReplayLinearModel


def _box(states,p):
    cells = len(states)
    if cells < 2 or any(state.n <= p for state in states):
        return None
    counts = torch.tensor([state.n for state in states],dtype=torch.float64)
    pooled = sum((state.sscp for state in states),torch.zeros((p,p),dtype=torch.float64))
    log_dets = []
    for state in states:
        sign,logdet = torch.linalg.slogdet(state.sscp/(state.n-1))
        if float(sign) <= 0 or not math.isfinite(float(logdet)):
            return None
        log_dets.append(float(logdet))
    total = int(counts.sum())
    df = counts-1
    sign,logpooled = torch.linalg.slogdet(pooled/(total-cells))
    if float(sign) <= 0:
        return None
    statistic = max(0.,(total-cells)*float(logpooled)-float((df*torch.tensor(log_dets,dtype=torch.float64)).sum()))
    c1 = (float((1/df).sum())-1/(total-cells))*(2*p*p+3*p-1)/(6*(p+1)*(cells-1))
    c2 = (float((1/df.square()).sum())-1/(total-cells)**2)*(p-1)*(p+2)/(6*(cells-1))
    df1 = p*(p+1)*(cells-1)/2
    chi2 = statistic*(1-c1)
    gap = c2-c1*c1
    f = df2 = None
    if gap > 0:
        df2 = (df1+2)/gap
        f = statistic*(1-c1-df1/df2)/df1
    elif gap < 0:
        df2 = (df1+2)/-gap
        bound = df2/(1-c1+2/df2)
        f = df2*statistic/(df1*(bound-statistic)) if bound > statistic else None
    return {"statistic":statistic,"f":f,"df1":df1,"df2":df2,
        "p_value":c.f_upper(f,df1,df2) if df2 else None,"chi2":chi2,
        "chi2_p_value":c.chi2_upper(chi2,df1)}


def manova(data,outcomes,factors,covariates,interactions,alpha,missing):
    sample = Replay(data,[*outcomes,*factors,*covariates],[*outcomes,*covariates],missing)
    model = ReplayLinearModel(sample,outcomes,factors,covariates,interactions,False)
    p = len(outcomes)
    if model.df_error < p:
        raise AnalysisError("insufficient_observations","MANOVA needs at least as many error degrees of freedom as outcomes.")
    if bool((model.error.diagonal() <= 1e-24*model.corrected_totals).any()):
        raise AnalysisError("perfect_fit","The model fits an outcome exactly, so the error matrix is singular.")
    multivariate,univariate = [],[]
    for term in model.terms:
        rows = kernel_call(multivariate_tests,model.hypothesis(term,3),model.error,model.df(term),model.df_error)
        multivariate.extend([[term.name,*row] for row in rows])
    for i,name in enumerate(outcomes):
        (rows,labels),_ = anova_rows(model,i,3,None,totals=(float(model.raw_totals[i]),float(model.corrected_totals[i])))
        univariate.extend([[name,label,*row] for label,row in zip(labels,rows,strict=True)])
    tables = {"multivariate":c.frame(multivariate,columns=["effect","test","value","statistic","df1","df2","p_value","partial_eta_squared","f_type"]),
        "univariate":c.frame(univariate,columns=["outcome","source",*ANOVA_COLUMNS])}
    notes = []
    cellplan = None
    if factors:
        states = {}
        for frame in sample.batches():
            for cell,block in frame.groupby(factors,observed=True,sort=False):
                if cell not in states:
                    if len(states) >= 2000:
                        raise AnalysisError("workspace_limit","MANOVA cell geometry exceeds its bounded workspace.")
                    cellplan = plan_workspace("MANOVA cell covariance",{
                        "TSQR_and_design":model.resource_plan.estimated_bytes,
                        "cell_moments":128*(p+1)**2*(len(states)+1)})
                    states[cell] = Moments(p)
                states[cell].add(c.matrix(block,outcomes,check_scale=False))
        box = _box(list(states.values()),p)
        if box is None:
            notes.append("Box's M is not computed: a cell has no more observations than outcomes or a singular covariance matrix.")
        else:
            tables["box_m"] = c.frame([list(box.values())],columns=list(box),index=["box_m"])
    return TableSet(tables,title=f"Multivariate analysis of variance of {', '.join(outcomes)}",
        n=model.n,n_missing=sample.dropped,df_resid=model.df_error,outcomes=outcomes,
        terms=[term.name for term in model.terms],ss_type=3,alpha=alpha,notes=notes,
        coding="sum-to-zero (effect) coding",missing="listwise",**sample.attrs(),
        full_source_collected=False,factorial_resource_plan=model.resource_plan.record(),
        cell_resource_plan=None if cellplan is None else cellplan.record(),**model.tsqr)
