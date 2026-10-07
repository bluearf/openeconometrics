"""Bounded exact conditional tied failures and time-dependent Cox designs.

Exactp carries normalized elementary-symmetric sufficient-statistic moments
through each tied risk set (Howard/Gail recursion). Time-dependent covariates
are evaluated at the actual failure time by owned disk interval splitting,
never by substituting an independent observation likelihood. [ST] stcox:
https://www.stata.com/manuals/ststcox.pdf . Computational work is explicitly
budgeted in addition to the shared live-memory plan.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.distributions import chi2_sf
from openecon.engines.linalg import collinear_columns
from openecon.engines.optimize import information_inverse
from openecon.engines.streaming_ols import _TSQRTree
from openecon.streaming_design import numeric_values

from . import streaming_cox as core
from .core import information_criteria, kernel_call, lr_test, wald_test
from .discrete.common import maximize, optimizer_record
from .streaming_linear import _finite, _result
from .streaming_work import work_budget

# Bounds actual risk-membership/moment updates, not input-file row count.
# Exact conditional likelihood can be intrinsically very expensive on ties.
EXACT_WORK_LIMIT = 50_000_000
TVC_WORK_LIMIT = 100_000_000


def _plan(sample,width,*,recursion=0,extra_cache=False):
    return sample.plan_rows("Cox exact event-dependent disk kernels",{
        "Cox_global_factors_and_risk_moments":384*(width+1)**2,
        "Cox_SQLite_group_and_score_caches":(24 if extra_cache else 20)*1024**2,
        "Cox_bounded_baseline_metadata":2*1024**2,
        "Cox_conditional_recursion_live_states":64*(recursion+1)*(width+1)**2,
    },384*(width+16)).record()


class _Exact:
    """Singles use the global sweep; ties use one bounded moment recursion."""
    def __init__(self,store,sample):
        self.store,self.k,self.evaluations = store,store.width,0
        self.single = core._Objective(store,"breslow",single_only=True)
        self.work,self.max_states,self.tied = 0,0,0
        for stratum,time,dead in store.connection.execute("SELECT s,t,COUNT(*) FROM rows WHERE d=1 GROUP BY s,t HAVING COUNT(*)>1"):
            risk = store.connection.execute("SELECT COUNT(*) FROM rows WHERE s=? AND t0<? AND t>=?",(stratum,time,time)).fetchone()[0]
            states = min(dead,risk-dead)
            if states:
                self.work += risk*states*(self.k+1)**2
                self.max_states = max(self.max_states,states)
                self.tied += 1
        self.work_budget = work_budget("OPENECON_COX_EXACT_WORK_LIMIT",EXACT_WORK_LIMIT)
        if self.work > self.work_budget:
            raise AnalysisError("exact_too_large",f"Exact Cox ties need {self.work:,} native sufficient-statistic moment updates per likelihood evaluation (budget {self.work_budget:,}); explicitly raise OPENECON_COX_EXACT_WORK_LIMIT for more computation, or use Efron ties for an approximation.")
        self.resource = _plan(sample,self.k,recursion=self.max_states)
        store.rows = sample.rows

    def value(self,beta):
        return self(beta)[0]

    def _event(self,beta,stratum,time,dead,risk,top):
        width = min(dead,risk-dead)
        if not width:
            return torch.zeros((),dtype=torch.float64),torch.zeros(self.k,dtype=torch.float64),torch.zeros((self.k,self.k),dtype=torch.float64)
        sign = -1. if dead > risk-dead else 1.
        logs = torch.full((width+1,),-math.inf,dtype=torch.float64)
        logs[0] = 0.
        means = torch.zeros((width+1,self.k),dtype=torch.float64)
        covariance = torch.zeros((width+1,self.k,self.k),dtype=torch.float64)
        numerator,observed = core._Sum(),core._Sum((self.k,))
        position = 0
        query = "SELECT pos,s,t0,t,d,w,off,x,id,cl,eta FROM rows WHERE s=? AND t0<? AND t>=? ORDER BY pos"
        for records,x in self.store.blocks(query,(stratum,time,time)):
            eta = sign*(x@beta+torch.tensor([row[6] for row in records],dtype=torch.float64)-top)
            selected = torch.tensor([bool(row[4] and row[3] == time) for row in records],dtype=torch.bool)
            if sign < 0:
                selected = ~selected
            numerator.add(eta[selected].sum())
            observed.add(sign*x[selected].sum(0))
            for value,row in zip(eta,sign*x,strict=True):
                position += 1
                stop = min(position,width)+1
                old = logs[1:stop].clone()
                plus = value+logs[:stop-1]
                mixture = torch.sigmoid(plus-old)
                difference = means[:stop-1]+row-means[1:stop]
                previous = covariance[:stop-1].clone()
                current = covariance[1:stop].clone()
                covariance[1:stop] = ((1-mixture)[:,None,None]*current+mixture[:,None,None]*previous
                                     +(mixture*(1-mixture))[:,None,None]*difference[:,:,None]*difference[:,None,:])
                means[1:stop] = means[1:stop]+mixture[:,None]*difference
                logs[1:stop] = torch.logaddexp(old,plus)
        if position != risk:
            raise AnalysisError("cox_spill_failed","Exact Cox risk membership changed during its conditional recursion.")
        result = numerator.value-logs[width],observed.value-means[width],-covariance[width]
        _finite(*result)
        return result

    def __call__(self,beta):
        value,gradient,hessian = self.single(beta)
        totals = [core._Sum(),core._Sum((self.k,)),core._Sum((self.k,self.k))]
        for total,base in zip(totals,(value,gradient,hessian),strict=True):
            total.add(base)
        sql = self.store.connection
        for stratum,time,dead in sql.execute("SELECT s,t,COUNT(*) FROM rows WHERE d=1 GROUP BY s,t HAVING COUNT(*)>1 ORDER BY s,t"):
            risk = sql.execute("SELECT COUNT(*) FROM rows WHERE s=? AND t0<? AND t>=?",(stratum,time,time)).fetchone()[0]
            top = sql.execute("SELECT top FROM strata WHERE s=?",(stratum,)).fetchone()[0]
            for total,extra in zip(totals,self._event(beta,stratum,time,dead,risk,top),strict=True):
                total.add(extra)
        self.evaluations += 1
        result = tuple(total.value for total in totals)
        _finite(*result)
        return result


def _time_function(time,texp):
    value = time if texp == "identity" else math.log(time)
    if not math.isfinite(value):
        raise AnalysisError("cox_precision","The time-dependent Cox transformation is not finite; rescale failure times.")
    return value


def _tvc_digest(sql,width):
    digest = hashlib.sha256()
    for pos,value in sql.execute("SELECT pos,z FROM tvc ORDER BY pos"):
        core._unpack(value,width)
        digest.update(struct.pack("<q",pos))
        digest.update(value)
    return digest.hexdigest()


def _expand(sample,original,terms,transform,centre,stack):
    names = core._roles(sample.spec,"tvc")
    texp = core._Notes(sample.spec).option("texp")
    if texp not in {"identity","log"}:
        raise AnalysisError("unsupported_streaming_option","Cox tvc time transformation must be identity or log.")
    q,k = len(names),len(terms)+len(names)
    if k > 384:
        raise AnalysisError("model_too_wide","Cox main and tvc equations together exceed384 parameters.")
    sql = original.connection
    sql.execute("CREATE TABLE times(s INTEGER,t REAL,previous REAL,PRIMARY KEY(s,t)) WITHOUT ROWID")
    sql.execute("INSERT INTO times SELECT s,t,LAG(t) OVER (PARTITION BY s ORDER BY t) FROM (SELECT s,t FROM rows WHERE d=1 GROUP BY s,t)")
    memberships = sql.execute("SELECT SUM((SELECT COUNT(*) FROM times e WHERE e.s=r.s AND e.t>r.t0 AND e.t<=r.t)) FROM rows r").fetchone()[0]
    work = memberships*(k+1)**2
    limit = work_budget("OPENECON_COX_TVC_WORK_LIMIT",TVC_WORK_LIMIT)
    if work > limit:
        raise AnalysisError("tvc_work_limit",f"Cox tvc requires {memberships:,} exact event-risk memberships and about {work:,} native moment updates per sweep (budget {limit:,}); explicitly raise OPENECON_COX_TVC_WORK_LIMIT for more computation, or use fewer distinct failure times/covariates.")
    resource = _plan(sample,k,extra_cache=True)
    original.rows = sample.rows
    sql.execute("CREATE TABLE tvc(pos INTEGER PRIMARY KEY,z BLOB)")
    magnitude = torch.zeros(q,dtype=torch.float64)
    for batch in sample.batches():
        z = torch.stack([numeric_values(batch.frame[name],name) for name in names],1)
        _finite(z)
        magnitude = torch.maximum(magnitude,z.abs().amax(0))
        sql.executemany("INSERT INTO tvc VALUES (?,?)",((pos,core._pack(row)) for pos,row in zip(batch.positions.tolist(),z,strict=True)))
    magnitude = torch.where(magnitude > 0,magnitude,torch.ones_like(magnitude))
    time_scale = max(abs(_time_function(t,texp)) for (t,) in sql.execute("SELECT t FROM times"))
    time_scale = time_scale or 1.
    inverse = magnitude.reciprocal()/time_scale
    if not bool(torch.isfinite(inverse).all()) or bool((inverse <= 0).any()):
        raise AnalysisError("cox_precision","Tvc parameter units cannot be represented in float64; rescale covariates/time.")
    tvc_checksum = _tvc_digest(sql,q)
    split = core._Store(k,sample.rows)
    stack.callback(split.close)
    split.groups = original.groups
    split.connection.executemany("INSERT INTO strata VALUES (?,?,?,NULL)",sql.execute("SELECT key,s,label FROM strata ORDER BY s"))
    split.connection.execute("CREATE TABLE physical(pos INTEGER PRIMARY KEY,original INTEGER NOT NULL)")
    pending,mapping,position = [],[],0

    def flush():
        if pending:
            split.connection.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,NULL)",pending)
            split.connection.executemany("INSERT INTO physical VALUES (?,?)",mapping)
            pending.clear()
            mapping.clear()

    # Only one physical record and one bounded virtual-row block are live;
    # SQLite owns the event grid and every expanded interval on disk.
    for record in sql.execute("SELECT r.pos,r.s,r.t0,r.t,r.d,r.w,r.off,r.x,r.id,r.cl,z.z FROM rows r JOIN tvc z ON z.pos=r.pos ORDER BY r.pos"):
        pos,s,entry,exit_,dead,weight,off,xbuf,subject,cluster,zbuf = record
        x,z = core._unpack(xbuf,original.width),core._unpack(zbuf,q)/magnitude
        for event,previous in sql.execute("SELECT t,previous FROM times WHERE s=? AND t>? AND t<=? ORDER BY t",(s,entry,exit_)):
            g = _time_function(event,texp)/time_scale
            design = torch.cat((x,z*g))
            _finite(design)
            begin = max(entry,previous) if previous is not None else entry
            pending.append((position,s,begin,event,float(bool(dead and exit_ == event)),weight,off,core._pack(design),subject,cluster))
            mapping.append((position,pos))
            position += 1
            if len(pending) == split.rows:
                flush()
    flush()
    if position != memberships or _tvc_digest(sql,q) != tvc_checksum:
        raise AnalysisError("cox_spill_failed","Time-dependent Cox input or exact event memberships changed.")
    split.connection.execute("CREATE INDEX physical_order ON physical(original,pos)")
    split.finalize()
    rank = _TSQRTree()
    for records,x in split.blocks():
        weights = torch.tensor([row[5] for row in records],dtype=torch.float64)
        _,factor = torch.linalg.qr(x*weights.sqrt()[:,None],mode="r")
        rank.add(factor)
    _,omitted = collinear_columns(rank.finish())
    if omitted:
        raise AnalysisError("singular_information","The exact event-time Cox design contains collinear main/tvc effects; remove duplicate or unidentified interactions.")
    full_transform = torch.zeros((k,k),dtype=torch.float64)
    full_transform[:original.width,:original.width] = transform
    full_transform[original.width:,original.width:] = torch.diag(inverse)
    meta = {"tvc_rows":position,"tvc_work_per_sweep":work,"tvc_work_budget":limit,"tvc_spool_hash":tvc_checksum,
            "tvc_time_scale":time_scale,"tvc_covariate_scale":magnitude.tolist(),
            "tvc_design":"exact event-time interactions on owned disk intervals"}
    return split,terms+[f"tvc:{name}" for name in names],full_transform,resource,meta


def _physical_scores(split,original):
    # Sorting by the immutable physical mapping makes the collapse sequential:
    # no observation-sized tensor/dictionary is needed even for a long path.
    def grouped():
        previous,total = None,None
        for records,scores in core._score_blocks(split,physical_order=True):
            for record,score in zip(records,scores,strict=True):
                position = record[0]
                if position != previous:
                    if total is not None:
                        yield previous,total.value
                    previous,total = position,core._Sum((split.width,))
                total.add(score)
        if total is not None:
            yield previous,total.value

    iterator = grouped()
    current = next(iterator,None)
    zero = torch.zeros(split.width,dtype=torch.float64)
    # Zero-exposure records also count as physical observations/clusters under
    # the dense contract. Include their exact zero scores in group corrections.
    for records,_ in original.blocks():
        values = []
        for record in records:
            if current is not None and current[0] < record[0]:
                raise AnalysisError("cox_spill_failed","A time-dependent score lost its physical mapping.")
            if current is not None and current[0] == record[0]:
                values.append(current[1])
                current = next(iterator,None)
            else:
                values.append(zero)
        yield records,torch.stack(values)
    if current is not None:
        raise AnalysisError("cox_spill_failed","A time-dependent score has no original physical record.")


def fit_events(spec,source,batch_rows,stack):
    sample,notes,original,terms,transform,centre,resource,counts,subjects = core._prepare(spec,source,batch_rows,stack)
    ties,tvc = notes.option("ties"),bool(core._roles(spec,"tvc"))
    store,meta = original,{}
    main_terms = list(terms)
    if tvc:
        store,terms,transform,resource,meta = _expand(sample,original,terms,transform,centre,stack)
        objective = core._Objective(store,ties)
    else:
        objective = _Exact(store,sample)
        resource = objective.resource
        meta = {"exact_tied_events":objective.tied,"exact_recursion_max_states":objective.max_states,
                "exact_moment_updates_per_evaluation":objective.work,"exact_work_budget":objective.work_budget,
                "exact_method":"normalized elementary-symmetric sufficient-statistic covariance recursion"}
    k = len(terms)
    start = torch.zeros(k,dtype=torch.float64)
    scale = sample.optimization_scale if spec.weight_type == "iweight" else 1.
    fitted = maximize(objective,start,what="stcox",scale=scale)
    if not fitted.converged:
        raise AnalysisError("separation_detected" if bool((fitted.theta.abs()>10).any()) else "nonconvergence",
                            "Cox exact event likelihood did not converge; check identification, scales and monotone likelihood.")
    working = fitted.theta
    residual = objective if tvc else core._Objective(store,"breslow")
    shift = 0. if tvc else float(centre@working)
    residual(working,record=True,shift=shift,baseline=not tvc)
    covariance,info = core._covariance(sample,original,fitted.hessian,
                                      score_factory=(lambda:_physical_scores(store,original)) if tvc else None)
    extra_notes = []
    tests = {} if tvc else core._ph_tests(store,covariance,terms,notes.option("phtest"))
    baseline = None if tvc else core._baseline(store)
    harrell,concordance = None,None
    if tvc:
        extra_notes.extend(["Baseline functions depend on the time-varying covariate path and are not reported after tvc.",
                            "Schoenfeld-residual PH tests are not computed after tvc."])
    else:
        extra_notes.append("After exactp, PH tests and baseline functions use Peto-Breslow formulas at the exact estimates, as in Stata.")
    has_entry = bool(original.connection.execute("SELECT 1 FROM rows WHERE t0>0 LIMIT 1").fetchone())
    if notes.option("concordance") and not spec.weights and not tvc and not has_entry and subjects == sample.nrows:
        harrell = core._concordance(store)
        concordance = harrell["concordance"]
    elif notes.option("concordance"):
        extra_notes.append("Harrell's C is not computed with weights, delayed entry, multiple records per subject or tvc.")
    null,g0,h0 = objective(start)
    null = float(null)
    try:
        inverse = kernel_call(information_inverse,-h0)
        statistic = float(g0@inverse@g0)
        score = {"statistic":statistic,"df":k,"p_value":chi2_sf(statistic,k),"distribution":"chi2","label":"Score test of b = 0"}
    except AnalysisError:
        score = {"statistic":None,"df":k,"p_value":None,"distribution":"chi2","label":"Score test: information singular at b = 0"}
    beta,covariance = transform@working,transform@covariance@transform.T
    _finite(beta,covariance)
    tests.update({"model":lr_test(fitted.value,null,k,label="LR chi2 test against b = 0") if spec.covariance == "nonrobust"
                  else wald_test(beta,covariance,range(k),label="Wald chi2 test of the coefficients"),"score":score})
    criteria = information_criteria(fitted.value,k,sample.nobs)
    strata_names = core._roles(spec,"strata")
    labels = [json.loads(row[0]) for row in original.connection.execute("SELECT label FROM strata ORDER BY s LIMIT 400")]
    extra = {"ties":ties,"hazard_ratios":core.dense._hazard_ratios(terms,beta,covariance,spec.alpha),
             "covariate_means":dict(zip(main_terms,(centre/transform.diagonal()[:len(main_terms)]).tolist(),strict=True)),
             "strata":strata_names or None,"strata_levels":labels if strata_names else None,
             "n_strata":original.groups,"strata_levels_thinned":original.groups>400,
             "partial_likelihood":"exact partial (conditional logit for tied times)" if ties == "exactp" else ties.capitalize(),
             "notes":extra_notes,"replay_event_kernel":meta}
    if baseline is not None:
        extra["baseline"] = baseline
    if harrell is not None:
        extra["concordance"] = harrell
    if original.digest() != original.checksum or store.digest() != store.checksum:
        raise AnalysisError("cox_spill_failed","Immutable original or event-dependent Cox records changed.")
    for _ in sample.batches():
        pass
    for spilled in [original] if store is original else [original,store]:
        spilled.connection.commit()
    extra["replay_risk_sets"] = {"ordering":"global stratum event-risk memberships on owned disk",
                                 "risk_interval":"entry < event_time <= exit","event_rows_materialized":False,
                                 "event_rows_materialized_on_disk":tvc,
                                 "row_scores_materialized":False,"source_spool_hash":original.checksum,
                                 "disk_passes":original.passes+(store.passes if store is not original else 0),
                                 "scratch_bytes":original.path.stat().st_size+(store.path.stat().st_size if store is not original else 0)}
    result = _result(sample,terms=terms,beta=beta,covariance=(covariance+covariance.T)/2,info=info,
                     metrics={"log_likelihood":fitted.value,"log_likelihood_null":null,"aic":criteria["aic"],"bic":criteria["bic"],
                              "df_model":k,**counts,"concordance":concordance},notes=notes,predictions=[],tests=tests,
                     solver="newton_partial_likelihood",diagnostics={"converged":True,"iterations":fitted.iterations,
                     "likelihood_evaluations":objective.evaluations,"risk_set_storage":"owned disk and bounded exact event kernels"},
                     extra=extra,resource=resource,use_t=False,title="Cox proportional hazards regression",
                     provenance_extra={"ties":ties,"bic_n":"number of observations (records)","optimizer":optimizer_record(fitted),**meta})
    if tvc:
        for index,row in enumerate(result.coefficients):
            row.equation = "main" if index < len(main_terms) else "tvc"
    return result
