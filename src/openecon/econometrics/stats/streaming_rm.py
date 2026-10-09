"""Repeated measures: external subject/cell validation, small contrast SSCP."""
from __future__ import annotations

import itertools
import math
import sqlite3

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.postest.index_codec import decode, encode
from openecon.resources import plan_workspace, workspace_budget_bytes
from . import common as c
from .replay import Moments, Replay
from .rm_anova import _contrasts, _finish, _MAX_CELLS
from .spill import Spill
from .streaming_anova import ReplayLinearModel


def rm_anova(data,y,subject,within,between,alpha,missing):
    sample = Replay(data,[y,subject,*within,*between],[y],missing)
    levels = {name:set() for name in [*within,*between]}
    overall = Moments(1)
    for frame in sample.batches():
        overall.add(c.values(frame,y,check_scale=False)[:,None])
        for name in levels:
            levels[name].update(c.group_codes(frame,name)[1])
            if len(levels[name]) > 2000 or sum(len(str(v)) for v in levels[name]) > 1024**2:
                raise AnalysisError("workspace_limit","Repeated factor metadata exceeds its bounded workspace.")
    for name,values in levels.items():
        categories = sample.category_schema[name]
        levels[name] = (c.group_codes(pd.DataFrame({name:list(values)}),name)[1] if categories is None
                        else [c._json_scalar(v) for v in categories if c._json_scalar(v) in values])
        if len(levels[name]) < 2:
            raise AnalysisError("single_level",f"Factor '{name}' has only one level.")
    sizes = [len(levels[name]) for name in within]
    cells = math.prod(sizes)
    if cells > _MAX_CELLS:
        raise AnalysisError("design_too_large",f"Repeated measures support at most {_MAX_CELLS} within-subject cells.")
    total_ss = float(overall.sscp[0,0])
    if c.negligible(total_ss,float(overall.raw_ss[0]),1e-30):
        raise AnalysisError("zero_variance",f"'{y}' does not vary.")
    groups = math.prod(len(levels[name]) for name in between)
    geometry = plan_workspace("external repeated-measures reductions",{
        "source_blocks":sample.plan.estimated_bytes,"within_contrasts_and_SSCP":256*cells*cells,
        "cell_descriptives":1024*groups*cells,"SQLite_page_cache":2*1024**2})
    transforms,columns = _contrasts(sizes)
    positions = {name:{label:i for i,label in enumerate(labels)} for name,labels in levels.items()}
    names = []
    for i in range(cells):
        name = f"__openecon_rm_{i}__"
        while name in between:
            name = "_"+name
        names.append(name)
    stats = {}
    with Spill() as db:
        db.execute("CREATE TABLE cells(s TEXT,c INTEGER,y REAL,b TEXT,PRIMARY KEY(s,c)) WITHOUT ROWID")
        for frame in sample.batches():
            block_stats = {}
            for record in frame.itertuples(index=False,name=None):
                value,s,*labels = record
                codes = [positions[name][c._json_scalar(label)] for name,label in zip([*within,*between],labels,strict=True)]
                cell = 0
                for code,size in zip(codes[:len(within)],sizes,strict=True):
                    cell = cell*size+code
                group = tuple(codes[len(within):])
                try:
                    db.execute("INSERT INTO cells VALUES (?,?,?,?)",(encode(s),cell,float(value),encode(group)))
                except sqlite3.IntegrityError as exc:
                    raise AnalysisError("unbalanced_design","A subject has repeated observations in the same within-subject cell.") from exc
                key = (group,cell)
                if key not in stats:
                    stats[key] = Moments(1)
                block_stats.setdefault(key, []).append(value)
            for key, values in block_stats.items():
                stats[key].add(torch.tensor(values,dtype=torch.float64)[:,None])
        incomplete = db.execute("SELECT COUNT(*) FROM (SELECT s FROM cells GROUP BY s HAVING COUNT(*)!=?)",(cells,)).fetchone()[0]
        if incomplete:
            raise AnalysisError("unbalanced_design",f"{incomplete} subjects lack a complete within-subject alternative set of {cells} cells.")
        inconsistent = db.execute("SELECT COUNT(*) FROM (SELECT s FROM cells GROUP BY s HAVING MIN(b)!=MAX(b))").fetchone()[0]
        if inconsistent:
            raise AnalysisError("invalid_design","A between-subject factor changes within subjects.")
        subjects = db.execute("SELECT COUNT(*) FROM (SELECT s FROM cells GROUP BY s)").fetchone()[0]
        rows = min(4096,max(1,(workspace_budget_bytes()-geometry.estimated_bytes)//(128*max(cells,1))))
        def factory():
            batch,wide,previous,group = [],[],None,None
            def observation():
                value = (torch.tensor(wide,dtype=torch.float64)-float(overall.location[0]))@transforms
                value[0] += float(overall.location[0])*math.sqrt(cells)
                return [*value.tolist(),*[levels[name][code] for name,code in zip(between,group,strict=True)]]
            for s,cell,value,b in db.execute("SELECT s,c,y,b FROM cells ORDER BY s,c"):
                if previous is not None and s != previous:
                    batch.append(observation())
                    if len(batch) >= rows:
                        yield pd.DataFrame(batch,columns=[*names,*between])
                        batch = []
                    wide = []
                previous,group = s,decode(b)
                wide.append(value)
            if previous is not None:
                batch.append(observation())
            if batch:
                yield pd.DataFrame(batch,columns=[*names,*between])
        source = Dataset.from_batches(factory,[*names,*between],row_count=subjects)
        reduced = Replay(source,[*names,*between],names,"raise")
        model = ReplayLinearModel(reduced,names,between,[],"full",False,allow_intercept=True)
        described = []
        for group in itertools.product(*(range(len(levels[name])) for name in between)):
            for cell,combo in enumerate(itertools.product(*(levels[name] for name in within))):
                state = stats.get((group,cell))
                if state is not None:
                    described.append([*[levels[name][code] for name,code in zip(between,group,strict=True)],
                        *combo,state.n,float(state.location[0]),math.sqrt(float(state.sscp[0,0])/(state.n-1)) if state.n > 1 else None])
        result = _finish(model,y,within,between,cells,columns,subjects,total_ss,described,
            sample.n,sample.dropped,alpha,contrast_geometry=(transforms,levels))
        result.attrs.update(sample.attrs(),full_source_collected=False,
            subject_cell_validation="owned SQLite unique typed subject/cell records; complete subjects",
            repeated_resource_plan=geometry.record(),tsqr_resource_plan=model.resource_plan.record(),
            subject_matrix_collected=False,**model.tsqr)
    return result
