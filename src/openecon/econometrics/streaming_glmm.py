"""Disk-grouped native normal/gamma random effects and conditional panel models.

Group likelihoods are integrated jointly. Observation blocks and a fixed node
grid are resident; ordered rows, adaptive group locations and scales are on
owned disk. Even a single panel larger than RAM is replayed, never collected.
Adaptive quadrature formulas follow OpenEconometrics's native GLMM kernel and [ME]
melogit / [XT] xtpoisson, https://www.stata.com/manuals/memelogit.pdf and
https://www.stata.com/manuals/xtxtpoisson.pdf . No external fitting library.
"""
from __future__ import annotations

from contextlib import ExitStack
import hashlib
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
from openecon.engines.distributions import gauss_hermite
from openecon.engines.linalg import collinear_columns
from openecon.engines.optimize import information_inverse
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _TSQRTree
from openecon.models import ModelSpec, ResultBundle
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import information_criteria, kernel_call, lr_test, wald_test
from .discrete.common import maximize
from .mixed import common, glmm, xt
from .mixed.glmm_kernels import PooledObjective, family_pieces
from .replay_sample import ReplaySample
from .streaming_cox import _Sum, _pack, _unpack
from .streaming_likelihood import _offset
from .streaming_linear import _Notes, _finite, _result
from .streaming_work import work_budget

SUPPORTED = frozenset({"melogit","meprobit","mepoisson","xtlogit","xtprobit","xtpoisson"})
_FAMILY = {"melogit":"logit","meprobit":"probit","mepoisson":"poisson",
           "xtlogit":"logit","xtprobit":"probit","xtpoisson":"poisson"}
CONDITIONAL_WORK_LIMIT = 50_000_000


def supports_spec(spec):
    return spec.estimator in SUPPORTED and spec.options.get("model","re") != "pa"


def _role(spec,name):
    return registry.role_columns(spec,name)


def _key_chunks(names):
    start = 0
    while start < len(names):
        stop,payload = start,0
        while stop < len(names) and stop-start < 500:
            amount = len(names[stop])+128
            if stop > start and payload+amount > 1024**2:
                break
            payload += amount
            stop += 1
        yield names[start:stop]
        start = stop


class _Groups:
    def __init__(self,width,random_width,rows):
        self.width,self.q,self.rows,self.passes = width,random_width,rows,0
        self.connection,self.scratch = None,None
        try:
            self.scratch = tempfile.TemporaryDirectory(prefix="openecon-glmm-",dir=os.environ.get("OPENECON_SCRATCH_DIRECTORY") or None)
            self.path = Path(self.scratch.name)/"groups.sqlite"
            self.connection = sqlite3.connect(self.path)
            for pragma in ("journal_mode=OFF","synchronous=OFF","temp_store=FILE","cache_size=-4096","mmap_size=0"):
                self.connection.execute("PRAGMA "+pragma)
            self.connection.execute("CREATE TABLE groups(id INTEGER PRIMARY KEY,key BLOB UNIQUE,n INTEGER DEFAULT 0,y REAL DEFAULT 0,cluster BLOB,mu BLOB,tau BLOB)")
            self.connection.execute("CREATE TABLE rows(pos INTEGER PRIMARY KEY,g INTEGER,y REAL,off REAL,x BLOB,z BLOB)")
            self.path.chmod(0o600)
        except (OSError,sqlite3.Error) as exc:
            self.close()
            raise self.failure() from exc

    @staticmethod
    def failure():
        return AnalysisError("mixed_spill_failed","Grouped replay requires writable local scratch storage and sufficient disk space for ordered rows/adaptive states.")

    def codes(self,keys,clusters):
        sql = self.connection
        mu,tau = _pack(torch.zeros(self.q,dtype=torch.float64)),_pack(torch.eye(self.q,dtype=torch.float64).flatten())
        unique = dict(zip(keys,clusters,strict=True))
        sql.executemany("INSERT OR IGNORE INTO groups(key,cluster,mu,tau) VALUES (?,?,?,?)",((key,value,mu,tau) for key,value in unique.items()))
        mapping = {}
        names = list(unique)
        for part in _key_chunks(names):
            query = "SELECT key,id,cluster FROM groups WHERE key IN ("+",".join("?" for _ in part)+")"
            mapping.update((key,(code,cluster)) for key,code,cluster in sql.execute(query,part))
        if any(mapping[key][1] != cluster for key,cluster in zip(keys,clusters,strict=True)):
            raise AnalysisError("cluster_not_nested","Each random-effect panel must be wholly nested in one covariance cluster.")
        return [mapping[key][0] for key in keys]

    def blocks(self,group=None):
        query = "SELECT pos,g,y,off,x,z FROM rows"+(" WHERE g=?" if group is not None else "")+" ORDER BY pos"
        cursor = self.connection.execute(query,(group,) if group is not None else ())
        self.passes += 1
        try:
            while records := cursor.fetchmany(self.rows):
                x = torch.stack([_unpack(row[4],self.width) for row in records])
                z = torch.stack([_unpack(row[5],self.q) for row in records])
                y,offset = torch.tensor([[row[2],row[3]] for row in records],dtype=torch.float64).T
                yield records,x,y,z,offset
        finally:
            cursor.close()

    def finalize(self):
        self.connection.execute("CREATE INDEX grouped ON rows(g,pos)")
        self.connection.commit()
        self.checksum = self.digest()

    def digest(self):
        result = hashlib.sha256()
        for records,*_ in self.blocks():
            for pos,group,y,offset,x,z in records:
                result.update(struct.pack("<qqdd",pos,group,y,offset))
                result.update(x)
                result.update(z)
        # Group keys/nesting/counts are immutable; adaptive locations are not.
        for group,key,n,y,cluster in self.connection.execute("SELECT id,key,n,y,cluster FROM groups ORDER BY id"):
            result.update(struct.pack("<qqd",group,n,y))
            for value in (key,cluster):
                result.update(len(value).to_bytes(8,"little"))
                result.update(value)
        return result.hexdigest()

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None


def _plan(sample,p,q,nodes,states=0):
    return sample.plan_rows("grouped mixed likelihood replay",{
        "mixed_global_information_and_optimizer":768*(p+q+1)**2,
        "mixed_owned_SQLite_and_covariance_caches":14*1024**2,
        "mixed_node_sufficient_statistics":128*nodes*(p+q+4),
        "mixed_conditional_live_state":64*(states+1)*(p+1)**2,
    },8*(48*(p+q+4)+16*nodes)).record()


def _validate_y(y,family):
    if family == "poisson":
        if bool(((y < 0)|(y != y.round())).any()):
            raise AnalysisError("invalid_count_outcome","A panel count outcome must be a nonnegative integer.")
    elif bool(((y != 0)&(y != 1)).any()):
        raise AnalysisError("invalid_binary_outcome","A panel binary outcome must be0/1.")


def _prepare(spec,source,batch_rows,stack,*,row_filter=None):
    notes,family = _Notes(spec),_FAMILY[spec.estimator]
    is_xt = spec.estimator.startswith("xt")
    model = notes.option("model") if is_xt else "re"
    if is_xt:
        # The native validator needs only role/option/spec metadata here.
        facade = type("Metadata",(),{"spec":spec,"option":notes.option})()
        xt._validate(facade,family,model,spec.estimator)
    if spec.weights:
        raise AnalysisError("unsupported_weights","These GLMM/panel likelihood contracts do not support user weights.")
    if spec.covariance not in {"nonrobust","robust","cluster"}:
        raise AnalysisError("unsupported_covariance","Grouped mixed replay supports observed-information, group robust and enclosing-cluster covariance.")
    if spec.covariance == "cluster" and len(registry.cluster_columns(spec)) != 1:
        raise AnalysisError("unsupported_cluster_dimensions","Grouped mixed replay supports one enclosing cluster.")
    group = spec.panel if is_xt else _role(spec,"group")[0]
    random = [] if is_xt else _role(spec,"random")
    if len(random)>1 or group == spec.outcome or group in [*spec.predictors,*random]:
        raise AnalysisError("invalid_spec","GLMM needs a distinct group column and at most one independent random slope.")
    if model == "pa":
        raise AnalysisError("unsupported_streaming_option","Population-averaged XT models use the dedicated GEE replay adapter.")
    sample = ReplaySample(spec,source,batch_rows=batch_rows,row_filter=row_filter)
    design = sample.add_design("x",intercept=True if model == "fe" else None)
    sample.prepare()
    p,q = len(design.terms),len(random)+1
    if not p:
        raise AnalysisError("no_covariates","No fixed-effect term remains in the mixed model.")
    points = notes.option("intpoints")
    if model == "re" and (type(points) is not int or not 1 <= points <=64 or notes.option("intmethod") not in {"ghermite","mcaghermite","mvaghermite"}):
        raise AnalysisError("invalid_spec","GLMM quadrature needs1..64 points and a supported native integration method.")
    nodes = points**q if model == "re" else 1
    resource = _plan(sample,p,q,nodes)
    store = _Groups(p,q,sample.rows)
    stack.callback(store.close)
    check_periods = bool(spec.time and is_xt and (model == "fe" or (family == "poisson" and not notes.option("normal"))))
    if check_periods:
        store.connection.execute("CREATE TABLE periods(g INTEGER,t BLOB,PRIMARY KEY(g,t)) WITHOUT ROWID")
    tree = _TSQRTree()
    low,high = math.inf,-math.inf
    for batch in sample.batches():
        y = batch.numeric(spec.outcome)
        _validate_y(y,family)
        low,high = min(low,float(y.min())),max(high,float(y.max()))
        offset = _offset(batch,spec)
        offset = torch.zeros_like(y) if offset is None else offset
        z = torch.stack([*[batch.numeric(name) for name in random],torch.ones_like(y)],1)
        _,factor = torch.linalg.qr(z,mode="r")
        tree.add(factor)
        keys = encode_cluster_labels(batch.frame[group])
        clusters = encode_cluster_labels(batch.frame[registry.cluster_columns(spec)[0]]) if spec.covariance == "cluster" else keys
        codes = store.codes(keys,clusters)
        if check_periods:
            periods = encode_cluster_labels(batch.frame[spec.time])
            try:
                store.connection.executemany("INSERT INTO periods VALUES (?,?)",zip(codes,periods,strict=True))
            except sqlite3.IntegrityError as exc:
                raise AnalysisError("repeated_time_values","Time values are repeated within a panel.") from exc
        ids = torch.tensor(codes,dtype=torch.int64)
        unique,inverse = torch.unique(ids,return_inverse=True)
        count = torch.zeros(len(unique),dtype=torch.int64).scatter_add_(0,inverse,torch.ones_like(inverse))
        total = torch.zeros(len(unique),dtype=torch.float64).scatter_add_(0,inverse,y)
        store.connection.executemany("UPDATE groups SET n=n+?,y=y+? WHERE id=?",zip(count.tolist(),total.tolist(),unique.tolist(),strict=True))
        store.connection.executemany("INSERT INTO rows VALUES (?,?,?,?,?,?)",((pos,g,value,off,_pack(x),_pack(random_row))
            for pos,g,value,off,x,random_row in zip(batch.positions.tolist(),codes,y.tolist(),offset.tolist(),batch.designs["x"],z,strict=True)))
    if low == high and model != "fe":
        raise AnalysisError("constant_outcome","The complete mixed-model outcome does not vary.")
    _,omitted = collinear_columns(tree.finish())
    if omitted:
        raise AnalysisError("collinear_random_effects","A constant or collinear random slope duplicates the random intercept.")
    store.finalize()
    groups = store.connection.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
    if groups < 2 and model != "fe":
        raise AnalysisError("insufficient_groups","Mixed/panel likelihood requires at least two observed groups.")
    return sample,notes,store,design,group,random,family,model,resource


class _Pooled:
    def __init__(self,store,family):
        self.store,self.family,self.size = store,family,store.width

    def __call__(self,theta):
        sums = [_Sum(),_Sum((self.size,)),_Sum((self.size,self.size))]
        for _,x,y,_,offset in self.store.blocks():
            result = PooledObjective(x,y,self.family,offset)(theta)
            if not bool(torch.isfinite(result[0])):
                return _rejected(self.size)
            for total,value in zip(sums,result,strict=True):
                total.add(value)
        return tuple(total.value for total in sums)

    def value(self,theta):
        return self(theta)[0]


def _rejected(size):
    return torch.tensor(-math.inf,dtype=torch.float64),torch.zeros(size,dtype=torch.float64),-torch.eye(size,dtype=torch.float64)


class _Quadrature:
    def __init__(self,store,family,points,method):
        self.store,self.family,self.method = store,family,method
        self.p,self.q,self.size = store.width,store.q,store.width+store.q
        nodes,weights = gauss_hermite(points)
        self.grid = nodes[:,None] if self.q == 1 else torch.cartesian_prod(nodes,nodes)
        log_w = weights.log() if self.q == 1 else (weights.log()[:,None]+weights.log()[None,:]).flatten()
        self.node_const = log_w+self.grid.square().sum(1)+.5*self.q*(math.log(2)-math.log(2*math.pi))
        self.adaptations,self.evaluations,self.maximum_mode_iterations = 0,0,0

    def _nodes(self,mu,tau):
        v = mu[None,:]+math.sqrt(2)*(self.grid@tau.T)
        base = self.node_const+tau.diagonal().log().sum()-.5*v.square().sum(1)
        return v,base

    def _posterior(self,theta,group,mu,tau,*,derivatives=False):
        v,base = self._nodes(mu,tau)
        u = v*theta[self.p:].exp()
        nodes = len(v)
        likelihood,gradient = _Sum((nodes,)),_Sum((nodes,self.size))
        for _,x,y,z,offset in self.store.blocks(group):
            eta = (x@theta[:self.p]+offset)[:,None]+z@u.T
            log_f,score,_ = family_pieces(self.family,y,eta)
            likelihood.add(log_f.sum(0))
            if derivatives:
                gradient.add(torch.cat((score.T@x,(score.T@z)*u),1))
        ell = base+likelihood.value
        log_likelihood = torch.logsumexp(ell,0)
        weights = (ell-log_likelihood).exp()
        _finite(weights,log_likelihood)
        return weights,v,u,log_likelihood,gradient.value

    def _mode(self,theta,group):
        v = torch.zeros(self.q,dtype=torch.float64)
        eye = torch.eye(self.q,dtype=torch.float64)
        sigma = theta[self.p:].exp()
        for iteration in range(100):
            gradient,hessian = _Sum((self.q,)),_Sum((self.q,self.q))
            for _,x,y,z,offset in self.store.blocks(group):
                zs = z*sigma
                eta = x@theta[:self.p]+offset+zs@v
                _,score,curvature = family_pieces(self.family,y,eta)
                gradient.add(zs.T@score)
                hessian.add(zs.T@(zs*curvature[:,None]))
            information = eye-hessian.value
            chol,info = torch.linalg.cholesky_ex(information)
            if bool(info != 0):
                raise AnalysisError("integration_precision","A group's posterior curvature is not positive definite.")
            step = torch.cholesky_solve((gradient.value-v)[:,None],chol).flatten()
            _finite(step)
            largest = float(step.abs().max())
            v += step*min(1.,2./max(largest,1e-300))
            if largest < 1e-10:
                break
        else:
            raise AnalysisError("integration_nonconvergence","A group's posterior mode did not converge; rescale predictors or simplify random effects.")
        self.maximum_mode_iterations = max(self.maximum_mode_iterations,iteration+1)
        hessian = _Sum((self.q,self.q))
        for _,x,y,z,offset in self.store.blocks(group):
            zs = z*sigma
            curvature = family_pieces(self.family,y,x@theta[:self.p]+offset+zs@v)[2]
            hessian.add(zs.T@(zs*curvature[:,None]))
        factor = torch.linalg.cholesky(eye-hessian.value)
        tau = torch.linalg.cholesky(torch.cholesky_inverse(factor))
        _finite(v,tau)
        return v,tau

    def adapt(self,theta):
        if self.method == "ghermite":
            return
        sql = self.store.connection
        for (group,) in sql.execute("SELECT id FROM groups ORDER BY id"):
            mu,tau = self._mode(theta,group)
            if self.method == "mvaghermite":
                for _ in range(50):
                    weights,v,*_ = self._posterior(theta,group,mu,tau)
                    mean = weights@v
                    centered = v-mean
                    covariance = centered.T@(centered*weights[:,None])
                    chol,info = torch.linalg.cholesky_ex(covariance)
                    if bool(info != 0):
                        break
                    change = max(float((mean-mu).abs().max()),float((chol-tau).abs().max()))
                    mu,tau = mean,chol
                    if change < 1e-9:
                        break
            sql.execute("UPDATE groups SET mu=?,tau=? WHERE id=?",(_pack(mu),_pack(tau.flatten()),group))
        self.adaptations += 1

    def group(self,theta,group,mu,tau,*,hessian=True):
        weights,_,u,value,node_gradient = self._posterior(theta,group,mu,tau,derivatives=True)
        gradient = weights@node_gradient
        if not hessian:
            return value,gradient,None
        information = _Sum((self.size,self.size))
        for _,x,y,z,offset in self.store.blocks(group):
            eta = (x@theta[:self.p]+offset)[:,None]+z@u.T
            curvature = family_pieces(self.family,y,eta)[2]
            block = torch.zeros((self.size,self.size),dtype=torch.float64)
            block[:self.p,:self.p] = x.T@(x*(curvature@weights)[:,None])
            for d in range(self.q):
                hu = curvature@(weights*u[:,d])
                block[:self.p,self.p+d] = x.T@(z[:,d]*hu)
                block[self.p+d,:self.p] = block[:self.p,self.p+d]
                for e in range(d,self.q):
                    block[self.p+d,self.p+e] = (z[:,d]*z[:,e]*(curvature@(weights*u[:,d]*u[:,e]))).sum()
                    block[self.p+e,self.p+d] = block[self.p+d,self.p+e]
            information.add(block)
        out = information.value+node_gradient.T@(node_gradient*weights[:,None])-gradient[:,None]*gradient[None,:]
        out[self.p:,self.p:] += torch.diag(gradient[self.p:])
        _finite(value,gradient,out)
        return value,gradient,(out+out.T)/2

    def groups(self,theta,*,hessian=True):
        for group,key,cluster,mu,tau in self.store.connection.execute("SELECT id,key,cluster,mu,tau FROM groups ORDER BY id"):
            yield group,key,cluster,self.group(theta,group,_unpack(mu,self.q),_unpack(tau,self.q**2).reshape(self.q,self.q),hessian=hessian)

    def __call__(self,theta):
        totals = [_Sum(),_Sum((self.size,)),_Sum((self.size,self.size))]
        try:
            for *_,result in self.groups(theta):
                for total,value in zip(totals,result,strict=True):
                    total.add(value)
        except AnalysisError as exc:
            if exc.code != "numerical_failure":
                raise
            return _rejected(self.size)
        self.evaluations += 1
        return tuple(total.value for total in totals)

    def value(self,theta):
        return self(theta)[0]


def _group_covariance(sample,store,objective,theta,hessian,group):
    bread = kernel_call(information_inverse,-hessian)
    info = {"covariance":sample.spec.covariance,"df_inference":None,"df_resid":None,
            "correction":"observed information (OIM)"}
    if sample.spec.covariance == "nonrobust":
        return bread,info
    acc = ClusterAccumulator(len(theta),scratch_directory=Path(os.environ["OPENECON_SCRATCH_DIRECTORY"]) if os.environ.get("OPENECON_SCRATCH_DIRECTORY") else None)
    try:
        pending,scores = [],[]
        for _,key,cluster,(_,score,_) in objective.groups(theta,hessian=False):
            pending.append(cluster if sample.spec.covariance == "cluster" else key)
            scores.append(score)
            if len(pending) == store.rows:
                acc.add(pending,torch.stack(scores))
                pending,scores = [],[]
        if pending:
            acc.add(pending,torch.stack(scores))
        meat,groups = acc.finish()
        correction = groups/(groups-1)
        covariance = bread@(meat*correction)@bread
        names = registry.cluster_columns(sample.spec) if sample.spec.covariance == "cluster" else [group]
        info.update({"cluster_column":names[0],"cluster_columns":names,"cluster_count":groups,"cluster_df":groups-1,
                     "small_sample_correction":correction,"correction":"integrated group scores clustered with G/(G-1)",
                     "cluster_spill":acc.diagnostics})
        _finite(covariance)
        return (covariance+covariance.T)/2,info
    finally:
        acc.close()


def _normal_fit(sample,notes,store,design,group,random,family):
    p,q = store.width,store.q
    pooled = maximize(_Pooled(store,family),torch.zeros(p,dtype=torch.float64),what="GLMM pooled start")
    if not pooled.converged:
        raise AnalysisError("separation_detected" if family != "poisson" else "nonconvergence","The pooled starting model does not have a finite converged fit.")
    points,method = notes.option("intpoints"),notes.option("intmethod")
    objective = _Quadrature(store,family,points,method)
    best,best_value = None,-math.inf
    for sd in (.25,.5,1.,2.):
        theta = torch.cat((pooled.theta,torch.full((q,),math.log(sd),dtype=torch.float64)))
        objective.adapt(theta)
        value = float(objective.value(theta))
        if value > best_value:
            best,best_value = theta,value
    if best is None:
        raise AnalysisError("invalid_start","No finite integrated likelihood exists at the GLMM starting grid.")
    theta,iterations = best,0
    for outer in range(1,31):
        objective.adapt(theta)
        fitted = maximize(objective,theta,what=sample.spec.estimator)
        iterations += fitted.iterations
        moved = float(((fitted.theta-theta).abs()/theta.abs().clamp_min(1)).max())
        theta = fitted.theta
        if not fitted.converged or moved < 1e-8 or method == "ghermite":
            break
    if bool((theta[p:].exp()<1e-4).any()):
        raise AnalysisError("boundary_solution","A random-effect standard deviation is at the zero-variance boundary; fit a pooled model or remove that effect.")
    if not fitted.converged:
        raise AnalysisError("nonconvergence","The integrated random-effects likelihood did not converge.")
    covariance,info = _group_covariance(sample,store,objective,theta,fitted.hessian,group)
    transform = torch.eye(p+q,dtype=torch.float64)
    transform[:p,:p] = design.transform
    return transform@theta,transform@covariance@transform.T,info,fitted,pooled,objective,outer,iterations


class _PanelCount:
    def __init__(self,store,*,gamma):
        self.store,self.gamma,self.p = store,gamma,store.width
        self.size,self.evaluations = self.p+int(gamma),0

    def group(self,theta,group,total,*,hessian=True):
        if self.gamma:
            sums = [_Sum(),_Sum(),_Sum((self.p,)),_Sum((self.p,)),_Sum((self.p,self.p))]
            for _,x,y,_,offset in self.store.blocks(group):
                eta = x@theta[:self.p]+offset
                lam = eta.exp()
                values = (lam.sum(),(y*eta-torch.lgamma(y+1)).sum(),x.T@y,x.T@lam,x.T@(x*lam[:,None]))
                for acc,value in zip(sums,values,strict=True):
                    acc.add(value)
            big_l,constant,yx,s,second = (acc.value for acc in sums)
            a = (-theta[self.p]).exp()
            k = (total+a)/(big_l+a)
            d_a = torch.special.digamma(total+a)-torch.special.digamma(a)-torch.log1p(big_l/a)+(big_l-total)/(big_l+a)
            value = constant+torch.lgamma(total+a)-torch.lgamma(a)-a*torch.log1p(big_l/a)-total*torch.log(big_l+a)
            gradient = torch.cat((yx-k*s,(-a*d_a).reshape(1)))
            out = None
            if hessian:
                out = torch.zeros((self.size,self.size),dtype=torch.float64)
                out[:self.p,:self.p] = -k*second+k/(big_l+a)*s[:,None]*s[None,:]
                d_aa = (torch.special.polygamma(1,total+a)-torch.special.polygamma(1,a)
                        +big_l/(a*(a+big_l))-(big_l-total)/(big_l+a).square())
                out[self.p,self.p] = a.square()*d_aa+a*d_a
                cross = -a*(total-big_l)/(big_l+a).square()*s
                out[:self.p,self.p] = cross
                out[self.p,:self.p] = cross
        else:
            top = -math.inf
            for _,x,_,_,offset in self.store.blocks(group):
                top = max(top,float((x@theta+offset).max()))
            sums = [_Sum(),_Sum(),_Sum((self.p,)),_Sum((self.p,)),_Sum((self.p,self.p))]
            for _,x,y,_,offset in self.store.blocks(group):
                eta = x@theta+offset-top
                lam = eta.exp()
                values = (lam.sum(),(y*eta-torch.lgamma(y+1)).sum(),x.T@y,x.T@lam,x.T@(x*lam[:,None]))
                for acc,value in zip(sums,values,strict=True):
                    acc.add(value)
            big_l,constant,yx,s,second = (acc.value for acc in sums)
            mean = s/big_l
            value = torch.lgamma(total+1)+constant-total*big_l.log()
            gradient = yx-total*mean
            out = -total*(second/big_l-mean[:,None]*mean[None,:]) if hessian else None
        _finite(value,gradient,*([out] if out is not None else []))
        return value,gradient,(out+out.T)/2 if out is not None else None

    def groups(self,theta,*,hessian=True):
        for group,key,cluster,total in self.store.connection.execute("SELECT id,key,cluster,y FROM groups ORDER BY id"):
            yield group,key,cluster,self.group(theta,group,torch.tensor(total,dtype=torch.float64),hessian=hessian)

    def __call__(self,theta):
        totals = [_Sum(),_Sum((self.size,)),_Sum((self.size,self.size))]
        try:
            for *_,values in self.groups(theta):
                for total,value in zip(totals,values,strict=True):
                    total.add(value)
        except AnalysisError as exc:
            if exc.code != "numerical_failure":
                raise
            return _rejected(self.size)
        self.evaluations += 1
        return tuple(total.value for total in totals)

    def value(self,theta):
        return self(theta)[0]


class _PanelLogit:
    def __init__(self,store,sample):
        self.store,self.size,self.evaluations = store,store.width,0
        work,maximum = 0,0
        for n,y in store.connection.execute("SELECT n,y FROM groups"):
            states = min(int(y),n-int(y))
            work += n*states*(self.size+1)**2
            maximum = max(maximum,states)
        self.work_budget = work_budget("OPENECON_PANEL_LOGIT_WORK_LIMIT",CONDITIONAL_WORK_LIMIT)
        if work > self.work_budget:
            raise AnalysisError("conditional_work_limit",f"Conditional panel logit needs {work:,} sufficient-statistic moment updates per evaluation (budget {self.work_budget:,}); explicitly raise OPENECON_PANEL_LOGIT_WORK_LIMIT for more computation, or shorten panels/reduce regressors.")
        self.resource = _plan(sample,self.size,store.q,1,maximum)
        store.rows = sample.rows
        self.work,self.maximum_states = work,maximum

    def group(self,theta,group,n,total,*,hessian=True):
        width = min(total,n-total)
        sign = -1. if total > n-total else 1.
        top = -math.inf
        for _,x,_,_,offset in self.store.blocks(group):
            top = max(top,float((x@theta+offset).max()))
        logs = torch.full((width+1,),-math.inf,dtype=torch.float64)
        logs[0] = 0.
        means = torch.zeros((width+1,self.size),dtype=torch.float64)
        covariance = torch.zeros((width+1,self.size,self.size),dtype=torch.float64)
        numerator,observed,position = _Sum(),_Sum((self.size,)),0
        for _,x,y,_,offset in self.store.blocks(group):
            eta = sign*(x@theta+offset-top)
            selected = (y == 1) if sign > 0 else (y == 0)
            numerator.add(eta[selected].sum())
            observed.add(sign*x[selected].sum(0))
            for value,row in zip(eta,sign*x,strict=True):
                position += 1
                stop = min(position,width)+1
                old = logs[1:stop].clone()
                plus = value+logs[:stop-1]
                mixture = torch.sigmoid(plus-old)
                difference = means[:stop-1]+row-means[1:stop]
                previous,current = covariance[:stop-1].clone(),covariance[1:stop].clone()
                covariance[1:stop] = ((1-mixture)[:,None,None]*current+mixture[:,None,None]*previous
                                     +(mixture*(1-mixture))[:,None,None]*difference[:,:,None]*difference[:,None,:])
                means[1:stop] = means[1:stop]+mixture[:,None]*difference
                logs[1:stop] = torch.logaddexp(old,plus)
        value,gradient,hess = numerator.value-logs[width],observed.value-means[width],-covariance[width]
        _finite(value,gradient,hess)
        return value,gradient,hess if hessian else None

    def groups(self,theta,*,hessian=True):
        for group,key,cluster,n,y in self.store.connection.execute("SELECT id,key,cluster,n,y FROM groups ORDER BY id"):
            yield group,key,cluster,self.group(theta,group,n,int(y),hessian=hessian)

    def __call__(self,theta):
        totals = [_Sum(),_Sum((self.size,)),_Sum((self.size,self.size))]
        for *_,values in self.groups(theta):
            for total,value in zip(totals,values,strict=True):
                total.add(value)
        self.evaluations += 1
        return tuple(total.value for total in totals)

    def value(self,theta):
        return self(theta)[0]


def _conditional_prepare(spec,source,batch_rows,stack,prepared):
    sample,_,original,_,group,_,family,_,_ = prepared
    sql = original.connection
    condition = "y>0 AND y<n" if family == "logit" else "y>0"
    informative = sql.execute(f"SELECT COUNT(*) FROM groups WHERE {condition}").fetchone()[0]
    total = sql.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
    if not informative:
        raise AnalysisError("no_outcome_variation","No panel carries conditional outcome information.")

    def filter_rows(frame):
        keys = encode_cluster_labels(frame[group])
        present = set()
        for part in _key_chunks(keys):
            query = f"SELECT key FROM groups WHERE {condition} AND key IN ("+",".join("?" for _ in part)+")"
            present.update(row[0] for row in sql.execute(query,part))
        return torch.tensor([key in present for key in keys],dtype=torch.bool)

    retained = _prepare(spec,source,batch_rows,stack,row_filter=filter_rows)
    sample,notes,store,design,group,random,family,model,resource = retained
    notes.warn(f"{total-informative} group(s) dropped because of no conditional outcome variation.") if total != informative else None
    store.connection.execute("ALTER TABLE groups ADD COLUMN center BLOB")
    rank = _TSQRTree()
    for gid,n in store.connection.execute("SELECT id,n FROM groups ORDER BY id"):
        total_x = _Sum((store.width,))
        for _,x,*_ in store.blocks(gid):
            total_x.add(x.sum(0))
        mean = total_x.value/n
        store.connection.execute("UPDATE groups SET center=? WHERE id=?",(_pack(mean),gid))
        for _,x,*_ in store.blocks(gid):
            _,factor = torch.linalg.qr(x-mean,mode="r")
            rank.add(factor)
    kept,omitted = collinear_columns(rank.finish())
    if not kept:
        raise AnalysisError("no_within_group_variation","No predictor varies within informative panels.")
    sample.notes["omitted_terms"].extend(design.terms[index] for index in omitted)
    design.terms = [design.terms[index] for index in kept]
    design.transform = design.transform[kept][:,kept]
    for gid,buffer in store.connection.execute("SELECT id,center FROM groups ORDER BY id"):
        mean = _unpack(buffer,store.width)
        for records,x,*_ in store.blocks(gid):
            transformed = (x-mean)[:,kept]
            store.connection.executemany("UPDATE rows SET x=? WHERE pos=?",((_pack(value),record[0]) for record,value in zip(records,transformed,strict=True)))
    store.width = len(kept)
    store.checksum = store.digest()
    return (*retained[:-1],resource),total-informative


def _structure(store):
    groups,minimum,average,maximum = store.connection.execute("SELECT COUNT(*),MIN(n),AVG(n),MAX(n) FROM groups").fetchone()
    return {"n_groups":groups,"group_size_min":minimum,"group_size_avg":average,"group_size_max":maximum}


def _bundle(sample,notes,store,terms,theta,covariance,info,metrics,tests,extra,resource,optimizer,*,solver,equations=None):
    _finite(theta,covariance)
    if store.digest() != store.checksum:
        raise AnalysisError("mixed_spill_failed","Immutable grouped input records changed during estimation.")
    for _ in sample.batches():
        pass
    store.connection.commit()
    extra["replay_groups"] = {"observations_collected":False,"panels_collected":False,"adaptive_states":"owned disk",
                              "spool_hash":store.checksum,"disk_passes":store.passes,"scratch_bytes":store.path.stat().st_size}
    result = _result(sample,terms=terms,beta=theta,covariance=(covariance+covariance.T)/2,info=info,
                     metrics={**metrics,**_structure(store)},notes=notes,predictions=[],tests=tests,solver=solver,
                     diagnostics={"converged":True,"iterations":optimizer["iterations"]},extra=extra,resource=resource,
                     use_t=False,title=registry.get(sample.spec.estimator).title,
                     provenance_extra={"optimizer":optimizer,"group_storage":"owned ordered disk records","derivatives":"native analytic group likelihood"})
    if equations:
        for row,equation in zip(result.coefficients,equations,strict=True):
            row.equation = equation
    if sample.spec.estimator in {'melogit','meprobit','mepoisson'} or '/lnsig2u' in terms:
        from .postest.group_state import capture_replayed_contract
        result.extra['group_state'] = capture_replayed_contract(result, sample)
    return result


def _fit(spec,source,batch_rows,stack):
    prepared = _prepare(spec,source,batch_rows,stack)
    sample,notes,store,design,group,random,family,model,resource = prepared
    if model == "fe":
        prepared,dropped = _conditional_prepare(spec,source,batch_rows,stack,prepared)
        sample,notes,store,design,group,random,family,model,resource = prepared
        objective = _PanelLogit(store,sample) if family == "logit" else _PanelCount(store,gamma=False)
        resource = objective.resource if family == "logit" else _plan(sample,store.width,store.q,1)
        fitted = maximize(objective,torch.zeros(store.width,dtype=torch.float64),what=spec.estimator+" conditional")
        if not fitted.converged:
            raise AnalysisError("separation_detected" if family == "logit" else "nonconvergence","The conditional panel likelihood did not converge.")
        covariance,info = _group_covariance(sample,store,objective,fitted.theta,fitted.hessian,group)
        theta,covariance = design.transform@fitted.theta,design.transform@covariance@design.transform.T
        null = float(objective.value(torch.zeros(store.width,dtype=torch.float64)))
        metrics = {"log_likelihood":fitted.value,**information_criteria(fitted.value,len(theta),sample.nobs),"n_groups_dropped":dropped}
        if family == "logit":
            metrics["pseudo_r_squared"] = 1-fitted.value/null if null < 0 else None
        tests = {"model":lr_test(fitted.value,null,len(theta),label="LR chi2 test against b = 0") if family == "logit" else
                         wald_test(theta,covariance,range(len(theta)),label="Wald chi2 test of coefficients")}
        extra = {"model":"fe","null_log_likelihood":null,"likelihood":"conditional on panel totals","constant":"absorbed by panel effects"}
        if family == "logit":
            extra["conditional_recursion"] = {"maximum_states":objective.maximum_states,
                                              "work_per_evaluation":objective.work,"work_budget":objective.work_budget}
        return _bundle(sample,notes,store,design.terms,theta,covariance,info,metrics,tests,extra,resource,
                       common.optimizer_summary(fitted),solver="newton_observed_hessian")
    normal = spec.estimator != "xtpoisson" or notes.option("normal")
    p,q = store.width,store.q
    if normal:
        theta,covariance,info,fitted,pooled,objective,outer,iterations = _normal_fit(sample,notes,store,design,group,random,family)
        pooled_value = pooled.value
        optimizer = common.optimizer_summary(fitted)|{"iterations":iterations,"outer_iterations":outer}
        integration = {"method":notes.option("intmethod"),"points":notes.option("intpoints"),"adaptations":objective.adaptations}
        extra = {"family":family,"integration":integration,"pooled_log_likelihood":pooled_value}
        if spec.estimator.startswith("me"):
            variances = (2*theta[p:]).exp()
            jacobian = torch.eye(p+q,dtype=torch.float64)
            jacobian[p:,p:] = torch.diag(2*variances)
            reported = torch.cat((theta[:p],variances))
            reported_covariance = jacobian@covariance@jacobian.T
            names = [*random,"_cons"]
            terms = [*design.terms,*[f"/var({name}[{group}])" for name in names]]
            extra.update({"group":group,"random_effects":names,"ln_sd":{name:{"estimate":float(theta[p+d]),"std_error":float(covariance[p+d,p+d].sqrt())}
                         for d,name in enumerate(names)}})
            ancillary,equations = "lr_vs_pooled",[spec.outcome]*p+[group]*q
        else:
            jacobian = torch.eye(p+1,dtype=torch.float64)
            jacobian[p,p] = 2.
            reported,reported_covariance = torch.cat((theta[:p],2*theta[p:])),jacobian@covariance@jacobian.T
            terms = [*design.terms,"/lnsig2u"]
            value,se = float(reported[p]),float(reported_covariance[p,p].sqrt())
            sigma = xt._ancillary_summary(value,se,spec.alpha,lambda v:math.exp(v/2),lambda v:.5*math.exp(v/2))
            extra.update({"model":"re","sigma_u":sigma})
            ancillary,equations = ("rho" if family in xt._LATENT else "sigma_u"),[spec.outcome]*p+[None]
            if family in xt._LATENT:
                c = xt._LATENT[family]
                def rho(v):
                    return math.exp(v)/(math.exp(v)+c)
                extra["rho"] = xt._ancillary_summary(value,se,spec.alpha,rho,lambda v:rho(v)*(1-rho(v)))
        solver = "newton_adaptive_quadrature"
    else:
        objective = _PanelCount(store,gamma=True)
        pooled = maximize(_Pooled(store,family),torch.zeros(p,dtype=torch.float64),what="panel Poisson pooled")
        if not pooled.converged:
            raise AnalysisError("nonconvergence","The pooled Poisson starting likelihood did not converge.")
        best,best_value = None,-math.inf
        for alpha in (.1,.5,1.,2.):
            start = torch.cat((pooled.theta,torch.tensor([math.log(alpha)],dtype=torch.float64)))
            current = float(objective.value(start))
            if current > best_value:
                best,best_value = start,current
        if best is None:
            raise AnalysisError("invalid_start","Panel gamma-Poisson has no finite starting likelihood.")
        fitted = maximize(objective,best,what="xtpoisson gamma")
        if float(fitted.theta[p]) < math.log(1e-6):
            raise AnalysisError("boundary_solution","Panel gamma variance is estimated at zero; fit pooled Poisson.")
        if not fitted.converged:
            raise AnalysisError("nonconvergence","Panel gamma-Poisson likelihood did not converge.")
        covariance,info = _group_covariance(sample,store,objective,fitted.theta,fitted.hessian,group)
        transform = torch.eye(p+1,dtype=torch.float64)
        transform[:p,:p] = design.transform
        reported,reported_covariance = transform@fitted.theta,transform@covariance@transform.T
        terms = [*design.terms,"/lnalpha"]
        value,se = float(reported[p]),float(reported_covariance[p,p].sqrt())
        extra = {"model":"re","distribution":"gamma","alpha":xt._ancillary_summary(value,se,spec.alpha,math.exp,math.exp),
                 "pooled_log_likelihood":pooled.value}
        pooled_value = pooled.value
        optimizer,solver,ancillary,equations = common.optimizer_summary(fitted),"newton_observed_hessian","alpha",[spec.outcome]*p+[None]
    slopes = [i for i,term in enumerate(design.terms) if term != "Intercept"]
    tests = {"model":wald_test(reported,reported_covariance,slopes,label="Wald chi2 test of slopes")}
    if spec.covariance == "nonrobust":
        tests[ancillary] = common.boundary_lr(2*(fitted.value-pooled_value),q if normal else 1,"pooled "+family)
    metrics = {"log_likelihood":fitted.value,**information_criteria(fitted.value,len(reported),sample.nobs)}
    if "sigma_u" in extra:
        metrics["sigma_u"] = extra["sigma_u"]["estimate"]
    if "rho" in extra:
        metrics["rho"] = extra["rho"]["estimate"]
    if "alpha" in extra:
        metrics["alpha"] = extra["alpha"]["estimate"]
    if spec.estimator.startswith("me") and family in glmm._LATENT and q == 1:
        metrics["icc"] = float(reported[p]/(reported[p]+glmm._LATENT[family]))
    result = _bundle(sample,notes,store,terms,reported,reported_covariance,info,metrics,tests,extra,resource,optimizer,
                     solver=solver,equations=equations)
    if spec.estimator.startswith("me"):
        common.log_intervals(result,range(p,p+q),[2*float(covariance[p+d,p+d].sqrt()) for d in range(q)],spec.alpha)
        if any(not math.isfinite(value) for row in result.coefficients[p:] for value in (row.ci_low,row.ci_high)):
            raise AnalysisError("mixed_precision","A random-effect variance interval exceeds finite float64; simplify the random-effects specification.")
    return result


def fit_streaming_glmm(spec:ModelSpec,source:Dataset,*,batch_rows=None)->ResultBundle:
    if spec.estimator not in SUPPORTED:
        raise AnalysisError("unsupported_streaming_estimator","This adapter fits mixed binary/count and XT panel likelihoods.")
    try:
        with torch.no_grad(),torch.device("cpu"),ExitStack() as stack:
            return _fit(spec,source,batch_rows,stack)
    except KernelError as exc:
        raise AnalysisError(exc.code,str(exc)) from exc
    except (sqlite3.Error,OSError) as exc:
        raise _Groups.failure() from exc
    except (OverflowError,torch.linalg.LinAlgError) as exc:
        raise AnalysisError("mixed_precision","Mixed/panel numerical quantities cannot be represented in float64; rescale covariates or simplify the specification.") from exc
