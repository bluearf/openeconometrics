"""Exact grouped percentiles and centred float64 moments over Dataset blocks."""
from __future__ import annotations

import math

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.resources import plan_workspace
from . import common as c
from .replay import Moments, Replay
from .spill import Spill


def _buffers(sample, variables, groups, requested):
    columns = len(requested) + ("ci" in requested)
    return {
        "source_buffers": sample.plan.estimated_bytes,
        "moments_geometry": 4096*groups*variables,
        "requested_result_geometry": 128*groups*variables*columns,
        "requested_column_metadata": sum(len(name.encode("utf-8"))+64 for name in requested),
        "SQLite_page_cache": 2*1024**2,
    }


def _quantile(db, variable, group, n, percent, method):
    if not n:
        return None
    position = (n if method == "stata" else n+1)*percent/100
    if method == "stata":
        nearest = round(position)
        whole = abs(position-nearest) <= 1e-9*max(n,1)
        low = max(0,min(n-1,nearest-1 if whole else math.floor(position)))
        high = max(0,min(n-1,low+1 if whole else low))
        fraction = .5
    else:
        k = math.floor(position)
        low,high = max(0,min(n-1,k-1)),max(0,min(n-1,k))
        fraction = position-k if 1 <= k < n else 0.
    a = db.execute("SELECT v FROM vals WHERE j=? AND g=? ORDER BY v LIMIT 1 OFFSET ?",(variable,group,low)).fetchone()[0]
    b = db.execute("SELECT v FROM vals WHERE j=? AND g=? ORDER BY v LIMIT 1 OFFSET ?",(variable,group,high)).fetchone()[0]
    return (1-fraction)*a+fraction*b


def describe(data,names,by,requested,percentile_method,moments,listwise,alpha):
    from .descriptives import _percent
    sample = Replay(data,[*names,*([by] if by else [])],names,"drop")
    states, ids, available, powers = {}, {}, {}, {}
    plan = plan_workspace("descriptive result allocation", _buffers(sample,len(names),1,requested))
    with Spill() as db:
        db.execute("CREATE TABLE vals (j INTEGER,g INTEGER,v REAL)")
        for frame in sample.batches(listwise=False):
            if by:
                frame = frame.loc[frame[by].notna()]
            if listwise:
                frame = frame.dropna(subset=names)
            if not len(frame):
                continue
            if by:
                codes,labels = c.group_codes(frame,by)
            else:
                codes,labels = torch.zeros(len(frame),dtype=torch.int64),[None]
            for code,label in enumerate(labels):
                if label not in ids:
                    if (len(ids)+1)*len(names) > 8192 or sum(len(str(x)) for x in ids)+len(str(label)) > 1024**2:
                        raise AnalysisError("workspace_limit","Descriptive group/result geometry exceeds its bounded workspace.")
                    plan_workspace("descriptive group/result allocation",
                        _buffers(sample,len(names),len(ids)+1,requested))
                    ids[label] = len(ids)
                    available[label] = 0
                    for j in range(len(names)):
                        states[j,label] = Moments(1)
                        powers[j,label] = [0.,0.]
                selected = frame.iloc[(codes == code).nonzero().flatten().tolist()]
                available[label] += len(selected)
                for j,name in enumerate(names):
                    values = c.values(selected,name,allow_missing=True,check_scale=False)
                    values = values[~torch.isnan(values)]
                    states[j,label].add(values[:,None])
                    db.executemany("INSERT INTO vals VALUES (?,?,?)",((j,ids[label],v) for v in values.tolist()))
        if not ids:
            raise AnalysisError("empty_sample","No eligible descriptive observations remain.")
        plan = plan_workspace("external grouped descriptive statistics",
            _buffers(sample,len(names),len(ids),requested))
        # Standardized third/fourth moments avoid overflow at a large data level.
        for frame in sample.batches(listwise=False):
            if by:
                frame = frame.loc[frame[by].notna()]
            if listwise:
                frame = frame.dropna(subset=names)
            if not len(frame):
                continue
            if by:
                codes,labels = c.group_codes(frame,by)
            else:
                codes,labels = torch.zeros(len(frame),dtype=torch.int64),[None]
            for code,label in enumerate(labels):
                selected = frame.iloc[(codes == code).nonzero().flatten().tolist()]
                for j,name in enumerate(names):
                    state = states[j,label]
                    if state.n < 2 or float(state.sscp[0,0]) == 0:
                        continue
                    values = c.values(selected,name,allow_missing=True,check_scale=False)
                    values = values[~torch.isnan(values)]
                    z = ((values-state.anchor[0])-state.mean[0])/math.sqrt(float(state.sscp[0,0])/(state.n-1))
                    powers[j,label][0] += float(z.pow(3).sum())
                    powers[j,label][1] += float(z.pow(4).sum())
        db.execute("CREATE INDEX ordered_vals ON vals(j,g,v)")
        if by:
            categories = sample.category_schema[by]
            labels = (c.group_codes(pd.DataFrame({by:list(ids)}),by)[1] if categories is None
                      else [c._json_scalar(v) for v in categories if c._json_scalar(v) in ids])
        else:
            labels = [None]
        header = [v for name in requested for v in (["ci_low","ci_high"] if name == "ci" else [name])]
        rows = []
        for j,name in enumerate(names):
            for label in labels:
                state = states[j,label]
                n = state.n
                mean = float(state.location[0]) if n else None
                variance = float(state.sscp[0,0])/(n-1) if n > 1 else None
                sd = None if variance is None else math.sqrt(variance)
                se = None if sd is None else sd/math.sqrt(n)
                table = {"n":n,"missing":available[label]-n,"mean":mean,"variance":variance,
                    "std_dev":sd,"std_error":se,"sum":None if mean is None else mean*n,
                    "min":float(state.low[0]) if n else None,"max":float(state.high[0]) if n else None,
                    "range":float(state.high[0]-state.low[0]) if n else None,
                    "cv":sd/mean if sd is not None and mean else None}
                third,fourth = powers[j,label]
                spread = n > 1 and variance > 0
                if moments == "spss":
                    table["skewness"] = n*third/((n-1)*(n-2)) if spread and n > 2 else None
                    table["kurtosis"] = n*(n+1)*fourth/((n-1)*(n-2)*(n-3))-3*(n-1)**2/((n-2)*(n-3)) if spread and n > 3 else None
                    table["se_skewness"] = math.sqrt(6*n*(n-1)/((n-2)*(n+1)*(n+3))) if n > 2 else None
                    table["se_kurtosis"] = math.sqrt(4*(n*n-1)*table["se_skewness"]**2/((n-3)*(n+5))) if n > 3 else None
                else:
                    table["skewness"] = third/n*(n/(n-1))**1.5 if spread else None
                    table["kurtosis"] = fourth/n*(n/(n-1))**2 if spread else None
                    table["se_skewness"] = table["se_kurtosis"] = None
                for stat in requested:
                    percent = _percent(stat)
                    if percent is not None:
                        table[stat] = _quantile(db,j,ids[label],n,percent,percentile_method)
                if "iqr" in requested:
                    table["iqr"] = (_quantile(db,j,ids[label],n,75.,percentile_method)-_quantile(db,j,ids[label],n,25.,percentile_method)) if n else None
                row = []
                for stat in requested:
                    if stat == "ci":
                        half = c.t_critical(alpha,n-1)*se if n > 1 else None
                        row.extend([None,None] if half is None else [mean-half,mean+half])
                    else:
                        row.append(int(table[stat]) if stat in {"n","missing"} else c.finite(table.get(stat)))
                rows.append(row if by is None else [name,label,*row])
    attrs = {"percentile_method":percentile_method,"moments":moments,"alpha":alpha,
        "missing":"listwise" if listwise else "variable-wise","n":sum(available.values()),
        "full_source_collected":False,"exact_external_order":True,**sample.attrs(),
        "descriptive_resource_plan":plan.record()}
    return c.frame(rows,columns=header if by is None else ["variable",by,*header],
        index=names if by is None else None,**attrs)
