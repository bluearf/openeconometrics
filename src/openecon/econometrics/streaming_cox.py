"""Exact Cox risk sets on bounded replays and owned ordered disk records.

The risk interval is (entry, exit], including censoring/entry boundaries and
all tied failures in the same stratum. Breslow/Efron partial likelihood,
analytic derivatives and Lin--Wei real-record scores follow [ST] stcox:
https://www.stata.com/manuals/ststcox.pdf . Risk sets are global, never fitted
independently per input batch. Disk event prefixes supply score integrals;
neither the observation matrix nor event-by-parameter arrays are retained.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import tempfile

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import chi2_sf
from openecon.engines.linalg import collinear_columns
from openecon.engines.optimize import information_inverse
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _TSQRTree
from openecon.models import ModelSpec, ResultBundle
from openecon.streaming_design import _json_scalar, encode_cluster_labels, numeric_values

from . import registry
from .core import information_criteria, kernel_call, lr_test, wald_test
from .discrete.common import maximize, optimizer_record
from .replay_sample import ReplaySample
from .streaming_linear import _CrossMoments, _GroupMeans, _Notes, _finite, _result, _weights
from .survival import cox as dense
from .survival.data import check_times, end_before_entry_note, failure_indicator

SUPPORTED = frozenset({"stcox"})


def supports_spec(spec):
    return spec.estimator == "stcox" and spec.options.get("ties","breslow") in {"breslow","efron","exactp"}


def _roles(spec,name):
    return registry.role_columns(spec,name)


def _keys(frame,names):
    if not names:
        return [b""]*len(frame)
    parts = [encode_cluster_labels(frame[name]) for name in names]
    if len(parts) == 1:
        return parts[0]
    return [b"".join(len(value).to_bytes(8,"little")+value for value in row)
            for row in zip(*parts,strict=True)]


def _pack(values):
    return values.contiguous().numpy().tobytes()


def _unpack(buffer,width):
    if len(buffer) != width*8:
        raise AnalysisError("cox_spill_failed","Cox scratch records changed or were truncated.")
    value = (torch.frombuffer(bytearray(buffer),dtype=torch.float64) if width else
             torch.empty(0,dtype=torch.float64))
    _finite(value)
    return value


class _Sum:
    """Neumaier totals retain small active risks when larger risks leave."""
    def __init__(self,shape=()):
        self.total = torch.zeros(shape,dtype=torch.float64)
        self.error = torch.zeros_like(self.total)

    @property
    def value(self):
        return self.total+self.error

    def add(self,value):
        value = torch.as_tensor(value,dtype=torch.float64)
        updated = self.total+value
        self.error += torch.where(self.total.abs() >= value.abs(),(self.total-updated)+value,(value-updated)+self.total)
        self.total = updated


class _Store:
    def __init__(self,width,rows):
        self.width,self.rows,self.passes = width,rows,0
        self.connection,self.scratch = None,None
        try:
            self.scratch = tempfile.TemporaryDirectory(prefix="openecon-cox-",dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None)
            self.path = Path(self.scratch.name)/"risk.sqlite"
            self.connection = sqlite3.connect(self.path)
            for pragma in ("journal_mode=OFF","synchronous=OFF","temp_store=FILE","cache_size=-4096","mmap_size=0"):
                self.connection.execute("PRAGMA "+pragma)
            self.connection.execute("CREATE TABLE rows(pos INTEGER PRIMARY KEY,s INTEGER,t0 REAL,t REAL,d REAL,w REAL,off REAL,x BLOB,id BLOB,cl BLOB,eta REAL)")
            self.connection.execute("CREATE TABLE strata(key BLOB PRIMARY KEY,s INTEGER UNIQUE,label TEXT,top REAL)")
            self.connection.execute("CREATE TABLE events(s INTEGER,t REAL,c INTEGER,d REAL,s0 REAL,a REAL,b REAL,mean BLOB,pull BLOB,b1 BLOB,ap REAL,pp BLOB,increment REAL,alpha REAL,hazard REAL,kp REAL,PRIMARY KEY(s,t)) WITHOUT ROWID")
            self.path.chmod(0o600)
        except (OSError,sqlite3.Error) as exc:
            self.close()
            raise self.failure() from exc

    @staticmethod
    def failure():
        return AnalysisError("cox_spill_failed","Cox replay requires writable local scratch storage and sufficient disk space for ordered risk/score records.")

    def blocks(self,query="SELECT pos,s,t0,t,d,w,off,x,id,cl,eta FROM rows ORDER BY pos",parameters=()):
        self.passes += 1
        cursor = self.connection.execute(query,parameters)
        try:
            while records := cursor.fetchmany(self.rows):
                values = torch.stack([_unpack(row[7],self.width) for row in records])
                yield records,values
        finally:
            cursor.close()

    def labels(self,keys,frame,names):
        result = []
        for i,key in enumerate(keys):
            found = self.connection.execute("SELECT s FROM strata WHERE key=?",(key,)).fetchone()
            if found is None:
                code = self.groups
                self.groups += 1
                label = _json_scalar(frame.iloc[i][names[0]]) if len(names) == 1 else code if names else None
                self.connection.execute("INSERT INTO strata VALUES (?,?,?,NULL)",(key,code,json.dumps(label,ensure_ascii=False)))
            else:
                code = found[0]
            result.append(code)
        return result

    def finalize(self):
        for column in ("t0","t"):
            self.connection.execute(f"CREATE INDEX risk_{column} ON rows(s,{column},pos)")
        self.connection.execute("CREATE INDEX failures ON rows(s,t,d)")
        self.connection.execute("CREATE INDEX subjects ON rows(id,t0,t,pos)")
        self.connection.commit()
        self.checksum = self.digest()

    def digest(self):
        result = hashlib.sha256()
        for records,_ in self.blocks():
            for pos,s,t0,t,d,w,off,x,subject,cluster,_ in records:
                result.update(struct.pack("<qqddddd",pos,s,t0,t,d,w,off))
                for value in (x,subject,cluster):
                    result.update(len(value).to_bytes(8,"little"))
                    result.update(value)
        return result.hexdigest()

    def top(self,beta):
        self.connection.execute("UPDATE strata SET top=NULL")
        for records,x in self.blocks():
            eta = x@beta+torch.tensor([row[6] for row in records],dtype=torch.float64)
            _finite(eta)
            codes = torch.tensor([row[1] for row in records],dtype=torch.int64)
            unique,inverse = torch.unique(codes,return_inverse=True)
            values = torch.full((len(unique),),-math.inf,dtype=torch.float64).scatter_reduce_(0,inverse,eta,reduce="amax")
            self.connection.executemany("UPDATE strata SET top=CASE WHEN top IS NULL THEN ? ELSE MAX(top,?) END WHERE s=?",
                                       ((value,value,code) for code,value in zip(unique.tolist(),values.tolist(),strict=True)))
            self.connection.executemany("UPDATE rows SET eta=? WHERE pos=?",((float(value),record[0]) for value,record in zip(eta,records,strict=True)))

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None


class _Moving:
    """Consume native row blocks up to an event boundary from one ordered cursor."""
    def __init__(self,store,stratum,column):
        condition,parameters = (" WHERE s=?",(stratum,)) if stratum is not None else ("",())
        self.iterator = store.blocks(f"SELECT pos,s,t0,t,d,w,off,x,id,cl,eta FROM rows{condition} ORDER BY {column},pos",parameters)
        self.column = 2 if column == "t0" else 3
        self.records,self.values,self.index = [],None,0
        self.done = False

    def consume(self,time):
        while not self.done:
            if self.index == len(self.records):
                try:
                    self.records,self.values = next(self.iterator)
                    self.index = 0
                except StopIteration:
                    self.done = True
                    break
            stop = self.index
            while stop < len(self.records) and self.records[stop][self.column] < time:
                stop += 1
            if stop == self.index:
                break
            first,self.index = self.index,stop
            yield self.records[first:stop],self.values[first:stop]

    def close(self):
        self.iterator.close()


def _failure_parts(store,stratum,time,beta,top):
    d0,d1,d2,d,count,linear,gconst = _Sum(),_Sum((len(beta),)),_Sum((len(beta),len(beta))),_Sum(),0,_Sum(),_Sum((len(beta),))
    for records,x in store.blocks("SELECT pos,s,t0,t,d,w,off,x,id,cl,eta FROM rows WHERE s=? AND t=? AND d=1 ORDER BY pos",(stratum,time)):
        w = torch.tensor([row[5] for row in records],dtype=torch.float64)
        eta = x@beta+torch.tensor([row[6] for row in records],dtype=torch.float64)-top
        u = torch.exp(eta)
        if bool((u == 0).any()):
            raise AnalysisError("cox_precision","A failing record's relative risk underflows float64; rescale covariates/offset or reduce separation.")
        v = w*u
        _finite(v)
        d0.add(v.sum())
        d1.add((x*v[:,None]).sum(0))
        d2.add(x.T@(x*v[:,None]))
        d.add(w.sum())
        linear.add(w@eta)
        gconst.add((x*w[:,None]).sum(0))
        count += len(records)
    return float(d0.value),d1.value,d2.value,float(d.value),count,float(linear.value),gconst.value


class _Objective:
    def __init__(self,store,ties,*,single_only=False):
        self.store,self.ties,self.k = store,ties,store.width
        self.single_only = single_only
        self.evaluations = 0

    def value(self,beta):
        return self(beta)[0]

    def __call__(self,beta,*,record=False,shift=0.,baseline=True):
        store,sql,k = self.store,self.store.connection,self.k
        store.top(beta)
        if record:
            sql.execute("DELETE FROM events")
        value,gradient,hessian = _Sum(),_Sum((k,)),_Sum((k,k))
        for stratum,top in sql.execute("SELECT s,top FROM strata ORDER BY s"):
            entering,exiting = _Moving(store,stratum,"t0"),_Moving(store,stratum,"t")
            s0,s1,s2 = _Sum(),_Sum((k,)),_Sum((k,k))
            ap,pp,cumulative,kp = _Sum(),_Sum((k,)),_Sum(),_Sum()
            kp_dead = False
            try:
                times = sql.execute("SELECT t FROM rows WHERE s=? AND d=1 GROUP BY t ORDER BY t",(stratum,))
                for (time,) in times:
                    for cursor,sign in ((entering,1.),(exiting,-1.)):
                        for records,x in cursor.consume(time):
                            w = torch.tensor([row[5] for row in records],dtype=torch.float64)
                            eta = x@beta+torch.tensor([row[6] for row in records],dtype=torch.float64)-top
                            v = w*torch.exp(eta)
                            _finite(v)
                            s0.add(sign*v.sum())
                            s1.add(sign*(x*v[:,None]).sum(0))
                            s2.add(sign*(x.T@(x*v[:,None])))
                    d0,d1,d2,d,count,linear,gconst = _failure_parts(store,stratum,time,beta,top)
                    if self.single_only and count > 1:
                        continue
                    risk,first,second = float(s0.value),s1.value,s2.value
                    if not risk > 0 or not math.isfinite(risk) or risk < d0*(1-1e-12):
                        raise AnalysisError("cox_precision","Ordered Cox risk sums cannot resolve a positive risk set at this event; rescale or reduce extreme predictors.")
                    a,b,pull,b1,means = _Sum(),_Sum(),_Sum((k,)),_Sum((k,)),_Sum((k,))
                    term_gradient,term_hessian,term_log = _Sum((k,)),_Sum((k,k)),_Sum()
                    if self.ties == "breslow":
                        mean = first/risk
                        a.add(d/risk)
                        pull.add((d/risk)*mean)
                        means.add(mean)
                        term_log.add(d*math.log(risk))
                        term_gradient.add(d*mean)
                        term_hessian.add(d*(second/risk-mean[:,None]*mean[None,:]))
                    else:
                        for start in range(0,count,store.rows):
                            fraction = torch.arange(start,min(start+store.rows,count),dtype=torch.float64)/count
                            phi = risk-fraction*d0
                            if not bool((phi > 0).all()):
                                raise AnalysisError("cox_precision","An Efron denominator cannot be represented as positive float64.")
                            mean = (first[None,:]-fraction[:,None]*d1[None,:])/phi[:,None]
                            inverse = 1/phi
                            a.add(inverse.sum())
                            b.add((fraction*inverse).sum())
                            pull.add((mean*inverse[:,None]).sum(0))
                            b1.add((mean*(fraction*inverse)[:,None]).sum(0))
                            means.add(mean.sum(0)/count)
                            term_log.add(phi.log().sum())
                            term_gradient.add(mean.sum(0))
                            term_hessian.add(second*inverse.sum()-d2*(fraction*inverse).sum()-mean.T@mean)
                    value.add(linear-term_log.value)
                    gradient.add(gconst-term_gradient.value)
                    hessian.add(-term_hessian.value)
                    if record:
                        ap.add(a.value)
                        pp.add(pull.value)
                        logscale = top+shift
                        increment = math.exp(math.log(float(a.value))-logscale) if baseline else 0.
                        alpha = _kp(store,stratum,time,beta,top,risk,d,d0,logscale,count) if baseline else 0.
                        cumulative.add(increment)
                        kp_dead |= alpha == -math.inf
                        if not kp_dead:
                            kp.add(alpha)
                        sql.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                    (stratum,time,count,d,risk,float(a.value),float(b.value),_pack(means.value),_pack(pull.value),_pack(b1.value),
                                     float(ap.value),_pack(pp.value),increment,alpha,float(cumulative.value),-math.inf if kp_dead else float(kp.value)))
            finally:
                entering.close()
                exiting.close()
        self.evaluations += 1
        result = value.value,gradient.value,(hessian.value+hessian.value.T)/2
        _finite(*result)
        return result


def _kp(store,stratum,time,beta,top,risk,d,d0,logscale,count):
    if d0/risk >= 1-1e-12:
        return -math.inf
    psi = -d
    if count == 1:
        record = store.connection.execute("SELECT w,eta FROM rows WHERE s=? AND t=? AND d=1",(stratum,time)).fetchone()
        r = math.exp(record[1]-top)/risk
        psi = math.log1p(-record[0]*r)/r
    else:
        for _ in range(100):
            g,slope = _Sum(),_Sum()
            for records,_ in store.blocks("SELECT pos,s,t0,t,d,w,off,x,id,cl,eta FROM rows WHERE s=? AND t=? AND d=1 ORDER BY pos",(stratum,time)):
                w = torch.tensor([row[5] for row in records],dtype=torch.float64)
                r = torch.exp(torch.tensor([row[10]-top for row in records],dtype=torch.float64))/risk
                q = r/(-torch.expm1(r*psi))
                g.add((w*q).sum())
                slope.add((w*q.square()*torch.exp(r*psi)).sum())
            if not float(slope.value) > 0:
                raise AnalysisError("cox_precision","The Kalbfleisch–Prentice baseline cannot be resolved in float64.")
            step = (float(g.value)-1)/float(slope.value)
            psi -= step
            if abs(step) <= 1e-15*abs(psi):
                break
        else:
            raise AnalysisError("nonconvergence","The Kalbfleisch–Prentice baseline did not converge.")
    alpha = psi*math.exp(-math.log(risk)-logscale)
    if not alpha < 0 or not math.isfinite(alpha):
        raise AnalysisError("cox_precision","The Cox baseline survivor cannot be represented in float64; rescale predictors.")
    return alpha


def _filter(spec):
    counts = {"pass":-1,"excluded":0}
    sample = None
    def usable(frame):
        if counts["pass"] != sample.passes:
            counts.update({"pass":sample.passes,"excluded":0})
        positive = (numeric_values(frame[spec.weights],spec.weights) > 0 if spec.weights
                    else torch.ones(len(frame),dtype=torch.bool))
        result = torch.ones(len(frame),dtype=torch.bool)
        if bool(positive.any()):
            kept = frame.iloc[positive.nonzero().flatten().tolist()]
            time = numeric_values(kept[spec.outcome],spec.outcome)
            names = _roles(spec,"entry")
            entry = numeric_values(kept[names[0]],names[0]) if names else None
            selected = check_times(time,entry,spec.outcome,names[0] if names else None)
            result[positive] = selected
            counts["excluded"] += int((~selected).sum())
        return result
    def attach(value):
        nonlocal sample
        sample = value
    return usable,attach,counts


def _prepare(spec,source,batch_rows,stack):
    notes = _Notes(spec)
    ties = notes.option("ties")
    tvc = _roles(spec,"tvc")
    if ties not in {"breslow","efron","exactp"}:
        raise AnalysisError("unsupported_streaming_option","Cox replay supports Breslow, Efron and exactp ties.")
    if ties in {"efron","exactp"} and spec.weights:
        raise AnalysisError("unsupported_weights","Cox Efron/exactp ties do not accept user weights; use Breslow.")
    if ties == "exactp" and (tvc or spec.covariance != "nonrobust"):
        raise AnalysisError("invalid_spec","Cox exactp cannot be combined with tvc or robust/cluster covariance.")
    if spec.weight_type == "pweight" and spec.covariance == "nonrobust":
        raise AnalysisError("unsupported_covariance","Cox sampling weights require robust or cluster covariance.")
    if spec.covariance not in {"nonrobust","robust","cluster"}:
        raise AnalysisError("unsupported_covariance","Cox replay supports observed information, robust and cluster covariance.")
    if spec.covariance == "cluster" and len(registry.cluster_columns(spec)) != 1:
        raise AnalysisError("unsupported_cluster_dimensions","Cox replay supports one cluster column.")
    filter_rows,attach,filtered = _filter(spec)
    sample = ReplaySample(spec,source,batch_rows=batch_rows,row_filter=filter_rows)
    attach(sample)
    design = sample.add_design("x")
    sample.prepare()
    if filtered["excluded"]:
        notes.warn(end_before_entry_note(filtered["excluded"]))
    width = len(design.terms)
    if not width and not tvc:
        raise AnalysisError("no_covariates","No Cox covariate remains after global rank screening.")
    resource = sample.plan_rows("exact Cox disk risk-set sweep", {
        "Cox_global_factors_and_risk_moments":384*(width+len(tvc)+1)**2,
        "Cox_SQLite_group_and_score_caches":20*1024**2,
        "Cox_bounded_baseline_metadata":2*1024**2,
    },384*(width+len(tvc)+16)).record()
    means = _GroupMeans(width,0)
    stack.callback(means.close)
    strata_names = _roles(spec,"strata")
    for batch in sample.batches():
        means.add(_keys(batch.frame,strata_names),batch.designs["x"],_weights(sample,batch),[])
    means.finish()
    tree = _TSQRTree()
    centre,mass = _Sum((width,)),_Sum()
    for batch in sample.batches():
        weights = _weights(sample,batch)
        x = batch.designs["x"]
        within = x-means.lookup(_keys(batch.frame,strata_names))
        _,factor = torch.linalg.qr(within*weights.sqrt()[:,None],mode="r")
        tree.add(factor)
        centre.add((x*weights[:,None]).sum(0))
        mass.add(weights.sum())
    kept,omitted = collinear_columns(tree.finish()) if width else ([],[])
    if not kept and not tvc:
        raise AnalysisError("no_covariates","All Cox covariates are constant or collinear within strata.")
    sample.notes["omitted_terms"].extend(design.terms[index] for index in omitted)
    terms = [design.terms[index] for index in kept]
    centre = (centre.value/float(mass.value))[kept]
    transform = design.transform[kept][:,kept]
    store = _Store(len(kept),sample.rows)
    stack.callback(store.close)
    store.groups = 0
    id_names,cluster_names = _roles(spec,"id"),registry.cluster_columns(spec)
    failures,duration = _Sum(),_Sum()
    for batch in sample.batches():
        time = batch.numeric(spec.outcome)
        entry_names,failure_names,offset_names = _roles(spec,"entry"),_roles(spec,"failure"),_roles(spec,"offset")
        entry = batch.numeric(entry_names[0]) if entry_names else torch.zeros_like(time)
        delta = failure_indicator(batch.numeric(failure_names[0]) if failure_names else None,len(time),failure_names[0] if failure_names else None)
        offset = batch.numeric(offset_names[0]) if offset_names else torch.zeros_like(time)
        weights = _weights(sample,batch)
        x = batch.designs["x"][:,kept]-centre
        strata = store.labels(_keys(batch.frame,strata_names),batch.frame,strata_names)
        subjects,clusters = _keys(batch.frame,id_names),_keys(batch.frame,cluster_names)
        store.connection.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,NULL)",
            ((position,stratum,t0,t,d,w,off,_pack(row),subject,cluster)
             for position,stratum,t0,t,d,w,off,row,subject,cluster in zip(batch.positions.tolist(),strata,entry.tolist(),time.tolist(),delta.tolist(),weights.tolist(),offset.tolist(),x,subjects,clusters,strict=True)))
        count_weights = weights if spec.weight_type == "fweight" else torch.ones_like(weights)
        failures.add(count_weights@delta)
        duration.add(count_weights@(time-entry))
    store.finalize()
    if float(failures.value) <= 0:
        raise AnalysisError("no_failures","The retained Cox sample contains no failures.")
    if id_names:
        previous,stop = None,0.
        for subject,entry,time in store.connection.execute("SELECT id,t0,t FROM rows ORDER BY id,t0,t,pos"):
            if subject == previous and entry < stop:
                raise AnalysisError("overlapping_records","Records of one subject overlap; use disjoint (entry, exit] intervals.")
            previous,stop = subject,time
        subjects = store.connection.execute("SELECT COUNT(*) FROM (SELECT id FROM rows GROUP BY id)").fetchone()[0]
        n_subjects = (store.connection.execute("SELECT SUM(m) FROM (SELECT MAX(w) AS m FROM rows GROUP BY id)").fetchone()[0]
                      if spec.weight_type == "fweight" else subjects)
    else:
        subjects,n_subjects = sample.nrows,sample.nobs
    if strata_names and store.groups > 400:
        notes.warn("Stratum labels and baseline rows are retained as a bounded sample; all strata enter estimation.")
    return sample,notes,store,terms,transform,centre,resource,{"n_failures":float(failures.value),"time_at_risk":float(duration.value),"n_subjects":float(n_subjects)},subjects


def _prefix(store,stratum,time):
    found = store.connection.execute("SELECT ap,pp FROM events WHERE s=? AND t<=? ORDER BY t DESC LIMIT 1",(stratum,time)).fetchone()
    return (found[0],_unpack(found[1],store.width)) if found else (0.,torch.zeros(store.width,dtype=torch.float64))


def _score_blocks(store,*,physical_order=False):
    # Indexed SQL joins perform prefix/event lookups in the ordered spill;
    # each resident score block is then evaluated in native vector arithmetic.
    mapping = " JOIN physical p ON p.pos=r.pos" if physical_order else ""
    position = "p.original" if physical_order else "r.pos"
    order = "p.original,r.pos" if physical_order else "r.pos"
    query = f"""SELECT {position},r.s,r.t0,r.t,r.d,r.w,r.off,r.x,r.id,r.cl,r.eta,
      COALESCE(hi.ap,0)-COALESCE(lo.ap,0),hi.pp,lo.pp,s.top,e.mean,COALESCE(e.b,0),e.b1
      FROM rows r{mapping} JOIN strata s ON s.s=r.s
      LEFT JOIN events hi ON hi.s=r.s AND hi.t=(SELECT MAX(t) FROM events WHERE s=r.s AND t<=r.t)
      LEFT JOIN events lo ON lo.s=r.s AND lo.t=(SELECT MAX(t) FROM events WHERE s=r.s AND t<=r.t0)
      LEFT JOIN events e ON e.s=r.s AND e.t=r.t ORDER BY {order}"""
    store.passes += 1
    cursor = store.connection.execute(query)
    zero = torch.zeros(store.width,dtype=torch.float64)
    try:
        while full := cursor.fetchmany(store.rows):
            records = [row[:11] for row in full]
            x = torch.stack([_unpack(row[7],store.width) for row in full])
            scalar = torch.tensor([[row[10]-row[14],row[11],row[16],row[4]] for row in full],dtype=torch.float64)
            vectors = [torch.stack([_unpack(row[index],store.width) if row[index] is not None else zero for row in full])
                       for index in (12,13,15,17)]
            u,a,b,delta = scalar[:,0].exp(),scalar[:,1],scalar[:,2],scalar[:,3]
            high,low,mean,b1 = vectors
            scores = -u[:,None]*(x*a[:,None]-(high-low))
            scores += delta[:,None]*(x-mean+u[:,None]*(x*b[:,None]-b1))
            _finite(scores)
            yield records,scores
    finally:
        cursor.close()


def _covariance(sample,store,hessian,*,score_factory=None):
    spec,k = sample.spec,hessian.shape[0]
    bread = kernel_call(information_inverse,-hessian)
    info = {"covariance":spec.covariance,"df_inference":None,"df_resid":None,
            "correction":"observed information (inverse negative Hessian)"}
    if spec.covariance == "nonrobust":
        return bread,info
    clustered = spec.covariance == "cluster" or bool(_roles(spec,"id"))
    meat = _Sum((k,k))
    acc = ClusterAccumulator(k) if clustered else None
    total = _Sum((k,))
    try:
        for records,scores in (score_factory() if score_factory else _score_blocks(store)):
            weights = torch.tensor([row[5] for row in records],dtype=torch.float64)
            total.add((scores*weights[:,None]).sum(0))
            if clustered:
                acc.add([row[9] if spec.covariance == "cluster" else row[8] for row in records],scores*weights[:,None])
            else:
                score_weights = weights if spec.weight_type == "fweight" else weights.square()
                meat.add(scores.T@(scores*score_weights[:,None]))
        factor = sample.nobs/(sample.nobs-1)
        matrix = meat.value
        if clustered:
            matrix,groups = acc.finish()
            factor = groups/(groups-1)
            names = registry.cluster_columns(spec) if spec.covariance == "cluster" else _roles(spec,"id")
            info.update({"cluster_count":groups,"cluster_df":groups-1,"cluster_columns":names,
                         "cluster_column":names[0],"cluster_spill":acc.diagnostics})
        info.update({"small_sample_correction":factor,
                     "correction":"Lin–Wei real-record scores: G/(G-1)" if clustered else "Lin–Wei real-record scores: N/(N-1)",
                     "weighted_score_sum":total.value.tolist()})
        covariance = bread@(matrix*factor)@bread
        _finite(covariance)
        return (covariance+covariance.T)/2,info
    finally:
        if acc is not None:
            acc.close()


def _baseline(store):
    sql = store.connection
    count = sql.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    selected = set(torch.linspace(0,count-1,min(count,400),dtype=torch.float64).round().to(torch.int64).tolist())
    result = {"stratum":[],"time":[],"cumulative_hazard":[],"survivor":[],"survivor_kp":[],"n_times":count,"thinned":count>400}
    for index,(stratum,time,hazard,kp) in enumerate(sql.execute("SELECT s,t,hazard,kp FROM events ORDER BY s,t")):
        if index not in selected:
            continue
        label = json.loads(sql.execute("SELECT label FROM strata WHERE s=?",(stratum,)).fetchone()[0])
        result["stratum"].append(label if label is not None else stratum)
        result["time"].append(time)
        result["cumulative_hazard"].append(hazard)
        result["survivor"].append(math.exp(-hazard))
        result["survivor_kp"].append(math.exp(kp))
    return result


def _gtable(store,kind):
    sql = store.connection
    sql.execute("CREATE TABLE g(t REAL PRIMARY KEY,value REAL)")
    if kind in {"identity","log"}:
        sql.executemany("INSERT INTO g VALUES (?,?)",((time,time if kind == "identity" else math.log(time))
                          for (time,) in sql.execute("SELECT t FROM rows WHERE d=1 GROUP BY t ORDER BY t")))
    elif kind == "rank":
        previous = 0
        for time,count in sql.execute("SELECT t,COUNT(*) FROM rows WHERE d=1 GROUP BY t ORDER BY t"):
            value = previous+(count+1)/2
            sql.execute("INSERT INTO g VALUES (?,?)",(time,value))
            previous += count
    else:
        entering,exiting = _Moving(store,None,"t0"),_Moving(store,None,"t")
        risk,survivor = _Sum(),1.
        try:
            for time,dead in sql.execute("SELECT t,SUM(w) FROM rows WHERE d=1 GROUP BY t ORDER BY t"):
                for cursor,sign in ((entering,1.),(exiting,-1.)):
                    for records,_ in cursor.consume(time):
                        risk.add(sign*math.fsum(row[5] for row in records))
                count = float(risk.value)
                if not count > 0 or dead > count*(1+1e-12):
                    raise AnalysisError("cox_precision","The global Kaplan–Meier PH transform cannot resolve the risk count.")
                survivor *= max(0.,1-dead/count)
                sql.execute("INSERT INTO g VALUES (?,?)",(time,1-survivor))
        finally:
            entering.close()
            exiting.close()


def _ph_tests(store,covariance,terms,kind):
    _gtable(store,kind)
    sql,k = store.connection,store.width
    anchor = sql.execute("SELECT value FROM g ORDER BY t LIMIT 1").fetchone()[0]
    scale = max((abs(value-anchor) for (value,) in sql.execute("SELECT value FROM g")),default=0.)
    if not math.isfinite(scale):
        raise AnalysisError("cox_precision","The PH time-function span exceeds finite float64 precision.")
    if scale == 0:
        return {"ph_global":{"statistic":None,"df":k,"p_value":None,"distribution":"chi2",
                              "label":"PH test: g(t) does not vary over the failures","time_function":kind}}
    query = "SELECT r.pos,r.s,r.t0,r.t,r.d,r.w,r.off,r.x,r.id,r.cl,r.eta,e.mean,g.value FROM rows r JOIN events e ON r.s=e.s AND r.t=e.t JOIN g ON r.t=g.t WHERE r.d=1 ORDER BY r.pos"
    mass = _Sum()
    moments = _CrossMoments(k+1)
    for records,x in store.blocks(query):
        w = torch.tensor([row[5] for row in records],dtype=torch.float64)
        schoenfeld = x-torch.stack([_unpack(row[11],k) for row in records])
        g = torch.tensor([(row[12]-anchor)/scale for row in records],dtype=torch.float64)
        moments.add(torch.cat((g[:,None],schoenfeld@covariance),1),w)
        mass.add(w.sum())
    d = float(mass.value)
    centre = float(moments.anchor[0]+moments.mean[0])
    spread = float(moments.m2.value[0,0])
    if not spread > 0:
        return {"ph_global":{"statistic":None,"df":k,"p_value":None,"distribution":"chi2",
                              "label":"PH test: g(t) does not vary over the failures","time_function":kind}}
    u = _Sum((k,))
    for records,x in store.blocks(query):
        w = torch.tensor([row[5] for row in records],dtype=torch.float64)
        schoenfeld = x-torch.stack([_unpack(row[11],k) for row in records])
        g = torch.tensor([(row[12]-anchor)/scale-centre for row in records],dtype=torch.float64)
        u.add((schoenfeld*(w*g)[:,None]).sum(0))
    vu = covariance@u.value
    tests = {}
    for j,term in enumerate(terms):
        statistic = d*float(vu[j])**2/(float(covariance[j,j])*spread)
        variance = float(moments.m2.value[j+1,j+1])
        rho = float(moments.m2.value[0,j+1])/math.sqrt(variance*spread) if variance > 0 else None
        tests[f"ph_{term}"] = {"statistic":statistic,"df":1,"p_value":chi2_sf(statistic,1),
                               "distribution":"chi2","rho":rho,"time_function":kind,
                               "label":f"PH test (scaled Schoenfeld residuals) for {term}"}
    statistic = float(u.value@vu)*d/spread
    if not math.isfinite(statistic) or any(not math.isfinite(test["statistic"]) for test in tests.values()):
        raise AnalysisError("cox_precision","The PH statistic cannot be represented in finite float64.")
    tests["ph_global"] = {"statistic":statistic,"df":k,"p_value":chi2_sf(statistic,k),
                          "distribution":"chi2","time_function":kind,
                          "label":"Global test of proportional hazards (Grambsch–Therneau)"}
    return tests


class _Fenwick:
    """Exact integer rank counts, fixed node LRU and O(distinct ranks) disk."""
    def __init__(self,sql,size):
        self.sql,self.size,self.cache = sql,size,OrderedDict()
        sql.execute("CREATE TABLE fenwick(i INTEGER PRIMARY KEY,n INTEGER)")

    def get(self,index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        record = self.sql.execute("SELECT n FROM fenwick WHERE i=?",(index,)).fetchone()
        return record[0] if record else 0

    def put(self,index,value):
        self.cache[index] = value
        self.cache.move_to_end(index)
        if len(self.cache) > 8192:
            old,count = self.cache.popitem(last=False)
            self.sql.execute("INSERT OR REPLACE INTO fenwick VALUES (?,?)",(old,count))

    def add(self,index):
        while index <= self.size:
            self.put(index,self.get(index)+1)
            index += index & -index

    def prefix(self,index):
        value = 0
        while index:
            value += self.get(index)
            index -= index & -index
        return value

    def clear(self):
        self.cache.clear()
        self.sql.execute("DELETE FROM fenwick")


def _concordance(store):
    sql = store.connection
    sql.execute("CREATE TABLE ranks(eta REAL PRIMARY KEY,r INTEGER UNIQUE)")
    sql.execute("INSERT INTO ranks SELECT eta,ROW_NUMBER() OVER(ORDER BY eta) FROM (SELECT DISTINCT eta FROM rows)")
    sql.execute("CREATE INDEX eta_ranks ON rows(s,t,d,eta)")
    size = sql.execute("SELECT COUNT(*) FROM ranks").fetchone()[0]
    tree = _Fenwick(sql,size)
    pairs,tied,concordant = 0,0,0
    for (stratum,) in sql.execute("SELECT s FROM strata ORDER BY s"):
        tree.clear()
        active = 0
        for (time,) in sql.execute("SELECT t FROM rows WHERE s=? GROUP BY t ORDER BY t DESC",(stratum,)):
            query = "SELECT r.r FROM rows x JOIN ranks r ON x.eta=r.eta WHERE x.s=? AND x.t=? AND x.d=? ORDER BY x.pos"
            for (rank,) in sql.execute(query,(stratum,time,0)):
                tree.add(rank)
                active += 1
            for (rank,) in sql.execute(query,(stratum,time,1)):
                lower,equal = tree.prefix(rank-1),tree.prefix(rank)
                pairs += active
                concordant += lower
                tied += equal-lower
            # Same-time failures do not form comparable pairs with each other.
            for (rank,) in sql.execute(query,(stratum,time,1)):
                tree.add(rank)
                active += 1
    value = (concordant+tied/2)/pairs if pairs else None
    return {"concordance":value,"somers_d":None if value is None else 2*value-1,
            "pairs":pairs,"concordant":concordant,"tied_predictions":tied,"discordant":pairs-concordant-tied}


def fit_streaming_cox(spec: ModelSpec,source: Dataset,*,batch_rows=None) -> ResultBundle:
    """Fit Cox by exact global ordered risk sets, without a dense fallback."""
    if spec.estimator != "stcox":
        raise AnalysisError("unsupported_streaming_estimator","This replay adapter fits stcox.")
    try:
        with torch.no_grad(),torch.device("cpu"),ExitStack() as stack:
            return _fit(spec,source,batch_rows,stack)
    except KernelError as exc:
        raise AnalysisError(exc.code,str(exc)) from exc
    except (sqlite3.Error,OSError) as exc:
        raise _Store.failure() from exc
    except (OverflowError,torch.linalg.LinAlgError) as exc:
        raise AnalysisError("cox_precision","Cox numerical quantities exceed representable float64; rescale predictors/offset/time.") from exc


def _fit(spec,source,batch_rows,stack):
    if _roles(spec,"tvc") or spec.options.get("ties") == "exactp":
        from .streaming_cox_events import fit_events
        return fit_events(spec,source,batch_rows,stack)
    sample,notes,store,terms,transform,centre,resource,counts,subjects = _prepare(spec,source,batch_rows,stack)
    k = len(terms)
    objective = _Objective(store,notes.option("ties"))
    start = torch.zeros(k,dtype=torch.float64)
    scale = sample.optimization_scale if spec.weight_type == "iweight" else 1.
    fitted = maximize(objective,start,what="stcox",scale=scale)
    if not fitted.converged:
        if bool((fitted.theta.abs() > 10).any()):
            raise AnalysisError("separation_detected","Cox partial likelihood increases toward divergent coefficients; a covariate separates failures from survivors.")
        raise AnalysisError("nonconvergence","Cox replay did not converge; check covariate scales and identification.")
    working = fitted.theta
    shift = float(centre@working)
    _,_,hessian = objective(working,record=True,shift=shift)
    covariance,info = _covariance(sample,store,hessian)
    tests = _ph_tests(store,covariance,terms,notes.option("phtest"))
    # Computing the null likelihood overwrites eta/top state, so do it after
    # all fitted-score/baseline diagnostics; fitted concordance uses eta below.
    concordance = None
    extra_notes = []
    baseline = _baseline(store)
    has_entry = bool(store.connection.execute("SELECT 1 FROM rows WHERE t0>0 LIMIT 1").fetchone())
    if notes.option("concordance") and not spec.weights and not has_entry and subjects == sample.nrows:
        harrell = _concordance(store)
        concordance = harrell["concordance"]
    else:
        harrell = None
        if notes.option("concordance"):
            extra_notes.append("Harrell's C is not computed with weights, delayed entry or multiple records per subject.")
    null,gradient0,hessian0 = objective(start)
    null = float(null)
    try:
        inverse = kernel_call(information_inverse,-hessian0)
        statistic = float(gradient0@inverse@gradient0)
        score = {"statistic":statistic,"df":k,"p_value":chi2_sf(statistic,k),"distribution":"chi2","label":"Score test of b = 0"}
    except AnalysisError:
        score = {"statistic":None,"df":k,"p_value":None,"distribution":"chi2","label":"Score test: information singular at b = 0"}
    beta,covariance = transform@working,transform@covariance@transform.T
    tests.update({"model":lr_test(fitted.value,null,k,label="LR chi2 test against b = 0") if spec.covariance == "nonrobust"
                  else wald_test(beta,covariance,range(k),label="Wald chi2 test of the coefficients"),"score":score})
    criteria = information_criteria(fitted.value,k,sample.nobs)
    strata_names = _roles(spec,"strata")
    labels = [json.loads(row[0]) for row in store.connection.execute("SELECT label FROM strata ORDER BY s LIMIT 400")]
    extra = {"ties":notes.option("ties"),"hazard_ratios":dense._hazard_ratios(terms,beta,covariance,spec.alpha),
             "covariate_means":dict(zip(terms,(centre/transform.diagonal()).tolist(),strict=True)),
             "strata":strata_names or None,"strata_levels":labels if strata_names else None,
             "n_strata":store.groups,"strata_levels_thinned":store.groups>400,"baseline":baseline,
             "partial_likelihood":"Breslow" if notes.option("ties") == "breslow" else "Efron"}
    if harrell is not None:
        extra["concordance"] = harrell
    if extra_notes:
        extra["notes"] = extra_notes
    if store.digest() != store.checksum:
        raise AnalysisError("cox_spill_failed","Immutable Cox risk-set input records changed.")
    for _ in sample.batches():
        pass  # Verify raw data, declared roles and retained positions again.
    store.connection.commit()
    extra["replay_risk_sets"] = {"ordering":"stratum-major global entry and exit cursors",
                                 "risk_interval":"entry < event_time <= exit",
                                 "event_rows_materialized":False,"row_scores_materialized":False,
                                 "source_spool_hash":store.checksum,"disk_passes":store.passes,
                                 "scratch_bytes":store.path.stat().st_size}
    diagnostics_record = {"converged":True,"iterations":fitted.iterations,"likelihood_evaluations":objective.evaluations,
                          "risk_set_storage":"owned disk ordered sweeps and event prefixes"}
    return _result(sample,terms=terms,beta=beta,covariance=(covariance+covariance.T)/2,info=info,
                   metrics={"log_likelihood":fitted.value,"log_likelihood_null":null,"aic":criteria["aic"],"bic":criteria["bic"],
                            "df_model":k,**counts,"concordance":concordance},notes=notes,predictions=[],tests=tests,
                   solver="newton_partial_likelihood",diagnostics=diagnostics_record,extra=extra,resource=resource,
                   use_t=False,title="Cox proportional hazards regression",provenance_extra={"ties":notes.option("ties"),
                   "bic_n":"number of observations (records)","optimizer":optimizer_record(fitted)})
