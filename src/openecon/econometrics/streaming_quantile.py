"""Exact full-source regression quantiles with owned disk state.

Frisch--Newton/Mehrotra iterations and LP vertex/face certificates follow the
native dense solver, Portnoy and Koenker (1997), and [R] qreg:
https://www.stata.com/manuals/rqreg.pdf . Only row blocks and small global
factors are resident. SQLite supplies external order statistics and bootstrap
multiplicities; it never fits a model. Source changes are checked on replays.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import secrets
import shutil
import sqlite3
import tempfile

import torch
from torch import Tensor

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.engines.contracts import KernelError
from openecon.engines.distributions import normal_ppf
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.models import ModelSpec, ResultBundle
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.streaming_design import encode_cluster_labels

from . import registry
from .core import kernel_call
from .quantile import kernels as qr
from .quantile import qreg as single
from .quantile import sqreg as multiple
from .replay_sample import ReplaySample
from .streaming_linear import _Notes, _finite, _result, _weights

SUPPORTED = frozenset({"qreg", "bsqreg", "sqreg", "iqreg"})


@contextmanager
def _scratch():
    location = os.environ.get("OPENECON_SCRATCH_DIRECTORY")
    try:
        with tempfile.TemporaryDirectory(prefix="openecon-quantile-", dir=location or None) as name:
            os.chmod(name, 0o700)
            connection = sqlite3.connect(str(Path(name)/"order.sqlite"))
            try:
                os.chmod(Path(name)/"order.sqlite", 0o600)
                connection.execute("PRAGMA cache_size=-4096")
                connection.execute("PRAGMA temp_store=FILE")
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("CREATE TABLE ordered (value REAL NOT NULL, weight REAL NOT NULL, row_id INTEGER NOT NULL)")
                connection.execute("CREATE INDEX ordered_values ON ordered(value,row_id)")
                connection.execute("CREATE TABLE labels (label BLOB PRIMARY KEY, code INTEGER NOT NULL UNIQUE)")
                connection.execute("CREATE TABLE counts (row_id INTEGER PRIMARY KEY, multiplicity INTEGER NOT NULL)")
                yield Path(name), connection
            finally:
                connection.close()
    except (OSError, sqlite3.Error) as exc:
        raise AnalysisError("scratch_storage", "Quantile replay requires writable local scratch storage with space for observation state and external sorts.") from exc


class _Rows:
    """A fixed-width owned binary file; reads/writes never map the full file."""
    def __init__(self, path, width, rows):
        self.path, self.width, self.rows, self.n = path, width, rows, 0
        self.file = open(path, "w+b")
        os.chmod(path, 0o600)

    def append(self, matrix):
        if matrix.ndim != 2 or matrix.shape[1] != self.width or matrix.dtype != torch.float64:
            raise AnalysisError("invalid_replay_state", "Invalid quantile scratch record.")
        _finite(matrix)
        self.file.seek(0, 2)
        self.file.write(matrix.contiguous().numpy().tobytes())
        self.n += len(matrix)

    def blocks(self):
        self.file.flush()
        if os.fstat(self.file.fileno()).st_size != self.n*self.width*8:
            raise AnalysisError("scratch_storage", "Quantile scratch records changed or were truncated.")
        for first in range(0, self.n, self.rows):
            count = min(self.rows, self.n-first)
            self.file.seek(first*self.width*8)
            data = bytearray(self.file.read(count*self.width*8))
            if len(data) != count*self.width*8:
                raise AnalysisError("scratch_storage", "Quantile scratch records were truncated.")
            yield first, torch.frombuffer(data, dtype=torch.float64).reshape(count, self.width)

    def put(self, first, matrix):
        _finite(matrix)
        self.file.seek(first*self.width*8)
        self.file.write(matrix.contiguous().numpy().tobytes())

    def select(self, positions):
        rows = []
        for position in positions.tolist():
            self.file.seek(int(position)*self.width*8)
            data = bytearray(self.file.read(self.width*8))
            if len(data) != self.width*8:
                raise AnalysisError("scratch_storage", "Quantile basis records were truncated.")
            rows.append(torch.frombuffer(data, dtype=torch.float64))
        return torch.stack(rows)

    def clear(self):
        self.file.seek(0)
        self.file.truncate()
        self.n = 0

    def digest(self):
        self.file.flush()
        result = hashlib.sha256()
        self.file.seek(0)
        while value := self.file.read(1024*1024):
            result.update(value)
        return result.hexdigest()

    def close(self):
        self.file.close()


def _sum(shape=()):
    return _CompensatedSum(shape)


def _psi(resid,tau):
    values = torch.full_like(resid,tau)
    values[resid < 0] = tau-1.
    return values


@dataclass
class _Fit:
    beta: Tensor
    gamma: Tensor
    basis: Tensor
    objective: float
    iterations: int
    pivots: int
    gap: float
    unique: bool
    exact_fit: bool


@dataclass
class _Vertex:
    basis: Tensor
    gamma: Tensor
    factor: tuple
    lam: Tensor
    violation: float
    ties: int


# State columns: a,s,z,w,exact weighted residual,zero,psi,virtual,edge g.
_A, _S, _Z, _W, _RESID, _ZERO, _PSI, _VIRTUAL, _EDGE = range(9)


class _Problem:
    """Disk Q/outcome/weights plus full-source Newton and LP reductions."""
    def __init__(self, root, sql, base, k, *, bootstrap=False):
        self.sql, self.base, self.k = sql, base, k
        self.data = _Rows(root/"problem.bin", k+3, base.rows)
        self.state = _Rows(root/"state.bin", 9, base.rows)
        self.rows = base.rows
        self.bootstrap = bootstrap
        self.disk_passes = 0

    def close(self):
        self.data.close()
        self.state.close()

    def prepare(self):
        self.data.clear()
        self.state.clear()
        tree, mass, level = _TSQRTree(), _sum(), _sum()
        self.n = 0
        for _, raw in self.base.blocks():
            x, y = raw[:, :self.k], raw[:, self.k]
            w = raw[:, self.k+1]
            if self.bootstrap:
                keys = raw[:, self.k+3].to(torch.int64) if self.cluster_bootstrap else torch.arange(self.nbase, self.nbase+len(raw))
                if self.cluster_bootstrap:
                    found = {key:row[0] for key in set(keys.tolist())
                             if (row := self.sql.execute("SELECT multiplicity FROM counts WHERE row_id=?", (key,)).fetchone()) is not None}
                else:
                    first, last = int(keys.min()), int(keys.max())
                    found = dict(self.sql.execute("SELECT row_id,multiplicity FROM counts WHERE row_id BETWEEN ? AND ?", (first, last)))
                w = torch.tensor([found.get(int(key), 0) for key in keys.tolist()], dtype=torch.float64)
                self.nbase += len(raw)
                keep = w > 0
                x, y, w = x[keep], y[keep], w[keep]
            if not len(y):
                continue
            block = torch.cat((x*w[:, None], (y*w)[:, None]), 1)
            _finite(block)
            _, factor = torch.linalg.qr(block, mode="r")
            tree.add(factor)
            self.n += len(y)
            mass.add(w.sum())
            level.add((y*w).abs().sum())
            self.data.append(torch.cat((x, y[:, None], w[:, None], torch.zeros_like(y[:, None])), 1))
        if self.n <= self.k:
            raise KernelError("insufficient_observations", "A quantile sample needs more distinct observations than parameters.")
        factor = tree.finish()
        scale = torch.linalg.vector_norm(factor[:, :self.k], dim=0)
        if bool((scale <= 0).any()):
            raise KernelError("singular_design", "The quantile sample contains an all-zero design column.")
        smallq, triangle = torch.linalg.qr(factor[:, :self.k]/scale, mode="reduced")
        if float(triangle.diagonal().abs().min()) <= 1e-11:
            raise KernelError("singular_design", "The quantile sample is rank deficient.")
        self.map = torch.linalg.solve_triangular(triangle, torch.eye(self.k, dtype=torch.float64), upper=True)/scale[:, None]
        self.ols = smallq.T@factor[:, self.k]
        self.mass, self.level = float(mass.value), float(level.value)
        spread = _sum()
        for first, raw in self.data.blocks():
            q = (raw[:, :self.k]*raw[:, self.k+1, None])@self.map
            ys = raw[:, self.k]*raw[:, self.k+1]-q@self.ols
            norms = torch.linalg.vector_norm(q, dim=1)
            self.data.put(first, torch.cat((q, ys[:, None], raw[:, self.k+1, None], norms[:, None]), 1))
            spread.add(ys.abs().sum())
        self.absolute_spread, self.spread = float(spread.value), float(spread.value)/self.mass
        self.checksum = self.data.digest()

    def blocks(self):
        self.disk_passes += 1
        for first, data in self.data.blocks():
            self.state.file.seek(first*9*8)
            buf = bytearray(self.state.file.read(len(data)*9*8))
            if len(buf) != len(data)*9*8:
                raise AnalysisError("scratch_storage", "Quantile dual records changed or were truncated.")
            yield first, data[:, :self.k], data[:, self.k], data[:, self.k+1], data[:, self.k+2], torch.frombuffer(buf, dtype=torch.float64).reshape(len(data), 9)

    def basis_near(self, gamma):
        self.sql.execute("DELETE FROM ordered")
        for first, data in self.data.blocks():
            distance = (data[:, self.k]-data[:, :self.k]@gamma).abs()
            self.sql.executemany("INSERT INTO ordered VALUES (?,1,?)", ((float(v), first+i) for i,v in enumerate(distance)))
        directions, chosen = [], []
        # Row-at-a-time disk selection retains K directions, even when the
        # nearest millions of residuals are duplicate design rows.
        cursor = self.sql.execute("SELECT row_id FROM ordered ORDER BY value,row_id")
        for (index,) in cursor:
            row = self.data.select(torch.tensor([index]))[0, :self.k]
            norm = torch.linalg.vector_norm(row)
            work = row.clone()
            for direction in directions:
                work -= (work@direction)*direction
                work -= (work@direction)*direction
            remaining = torch.linalg.vector_norm(work)
            if norm > 0 and remaining > qr._INDEPENDENT*norm:
                directions.append(work/remaining)
                chosen.append(index)
                if len(chosen) == self.k:
                    break
        cursor.close()
        if len(chosen) != self.k:
            raise KernelError("singular_design", "No independent quantile basis exists.")
        return torch.tensor(chosen, dtype=torch.int64)

    def vertex(self, tau, basis, *, previous=None, perturbation_step=None):
        data = self.data.select(basis)
        factor = qr._basis_factor(data[:, :self.k])
        if previous is None:
            gamma = qr._basis_solve(factor, data[:, self.k])
            gamma += qr._basis_solve(factor, data[:, self.k]-data[:, :self.k]@gamma)
        else:
            gamma = previous.gamma
        _finite(gamma)
        target, ties = _sum((self.k,)), 0
        for first, q, ys, w, norm, state in self.blocks():
            ordinal = torch.arange(first, first+len(q))
            basic = (ordinal[:, None] == basis[None, :]).any(1)
            if previous is None:
                resid = ys-q@gamma
                resid[basic] = 0.
                bound = 64*qr._EPS*(ys.abs()+norm*torch.linalg.vector_norm(gamma))
                zero = resid.abs() <= torch.maximum(bound, qr._TIE*self.spread*w)
                resid[zero] = 0.
                virtual = 1.+torch.remainder(ordinal.to(torch.float64)*.6180339887498949, 1.)
            else:
                resid, zero = state[:, _RESID], state[:, _ZERO].bool()
                virtual = state[:, _VIRTUAL]-perturbation_step*state[:, _EDGE]
            virtual[basic] = 0.
            psi = _psi(resid,tau)
            psi[zero & (virtual < 0)] = tau-1.
            psi[basic] = 0.
            target.add(q.T@psi)
            ties += int(zero.sum())-int(basic.sum())
            state[:, _RESID], state[:, _ZERO], state[:, _PSI], state[:, _VIRTUAL] = resid, zero.to(torch.float64), psi, virtual
            self.state.put(first, state)
        lam = -qr._basis_solve(factor, target.value, transpose=True)
        _finite(lam)
        violation = float(torch.maximum(lam-tau, tau-1.-lam).max().clamp_min(0.))
        return _Vertex(basis, gamma, factor, lam, violation, ties)

    def tied_certificate(self, tau, vertex):
        if not vertex.ties:
            return None
        defect, gram = _sum((self.k,)), _sum((self.k,self.k))
        for _, q, _, _, _, state in self.blocks():
            zero = state[:, _ZERO].bool()
            psi = _psi(state[:, _RESID],tau)
            psi[zero] = 0.
            defect.add(-q.T@psi-q[zero].T@(state[zero, _A]-(1.-tau)))
            gram.add(q[zero].T@q[zero])
        factor, info = torch.linalg.cholesky_ex(gram.value)
        if int(info) != 0:
            return None
        shift = torch.cholesky_solve(defect.value[:, None], factor)[:, 0]
        inside, count, valid = _sum((self.k,self.k)), 0, True
        for _, q, _, _, _, state in self.blocks():
            zero = state[:, _ZERO].bool()
            rows = q[zero]
            d = state[zero, _A]-(1.-tau)+rows@shift
            valid &= bool(torch.isfinite(d).all()) and (not len(d) or float(torch.maximum(d-tau,tau-1.-d).max()) <= qr._DUAL_TOL)
            interior = rows[torch.minimum(tau-d,d-(tau-1.)) > qr._INTERIOR]
            count += len(interior)
            inside.add(interior.T@interior)
        if not valid:
            return None
        if count < self.k:
            return False
        spectrum = torch.linalg.eigvalsh(inside.value)
        return bool(spectrum[0] > qr._INDEPENDENT*spectrum[-1])

    def pivot(self, tau, vertex):
        over, under = vertex.lam-tau, tau-1.-vertex.lam
        j = int(torch.argmax(torch.maximum(over,under)))
        sigma = 1. if float(under[j]) > 0 else -1.
        unit = torch.zeros(self.k, dtype=torch.float64)
        unit[j] = sigma
        direction = qr._basis_solve(vertex.factor,unit)
        maximum = 0.
        for _, q, _, _, _, _ in self.blocks():
            maximum = max(maximum,float((q@direction).abs().max()))
        self.sql.execute("DELETE FROM ordered")
        self.sql.execute("DROP TABLE IF EXISTS real_breaks")
        self.sql.execute("CREATE TABLE real_breaks (value REAL NOT NULL, weight REAL NOT NULL,row_id INTEGER NOT NULL)")
        slope = _sum()
        for first,q,_,_,_,state in self.blocks():
            g = q@direction
            g[g.abs() <= qr._FLAT*maximum] = 0.
            ordinal = torch.arange(first,first+len(q))
            g[(ordinal[:,None] == vertex.basis[None,:]).any(1)] = 0.
            g[ordinal == vertex.basis[j]] = sigma
            slope.add(-(g@state[:, _PSI]))
            state[:, _EDGE] = g
            self.state.put(first,state)
            zero = state[:, _ZERO].bool()
            for table, values, mask in (("ordered",state[:,_VIRTUAL],zero),("real_breaks",state[:,_RESID],~zero)):
                valid = mask & (values*g > 0)
                ids = valid.nonzero().flatten()
                self.sql.executemany(f"INSERT INTO {table} VALUES (?,?,?)", ((float(values[i]/g[i]),float(g[i].abs()),first+int(i)) for i in ids))
        current = float(slope.value)+(1.-tau if sigma > 0 else tau)
        compensation = 0.
        for table in ("ordered","real_breaks"):
            cursor = self.sql.execute(f"SELECT value,weight,row_id FROM {table} ORDER BY value,row_id")
            for step, climb, entering in cursor:
                adjusted = climb-compensation
                updated = current+adjusted
                compensation = (updated-current)-adjusted
                current = updated
                if current >= 0:
                    cursor.close()
                    basis = vertex.basis.clone()
                    basis[j] = entering
                    return basis, step if table == "ordered" else None
            cursor.close()
        raise KernelError("numerical_failure", "The exact quantile line search found no breakpoint.")

    def unique(self,tau,vertex):
        boundary = torch.minimum(tau-vertex.lam,vertex.lam-(tau-1.)) <= qr._DUAL_TOL
        count = int(boundary.sum())
        if not count:
            return True
        if count > 1 or not vertex.ties:
            return False
        j = int(boundary.nonzero()[0])
        unit = torch.zeros(self.k,dtype=torch.float64)
        unit[j] = 1. if float(vertex.lam[j]) < tau-.5 else -1.
        direction = qr._basis_solve(vertex.factor,unit)
        maximum, crossing = 0.,0.
        for first,q,_,_,_,state in self.blocks():
            g = q@direction
            ordinal = torch.arange(first,first+len(q))
            g[(ordinal[:,None] == vertex.basis[None,:]).any(1)] = 0.
            maximum = max(maximum,float(g.abs().max()))
            moving = state[:,_ZERO].bool() & (state[:,_VIRTUAL]*g > 0)
            if bool(moving.any()):
                crossing = max(crossing,float(g[moving].abs().max()))
        return crossing > 1e-9*maximum

    @staticmethod
    def directions(q,state,dv,mu=None,affine=None):
        a,s,z,w = (state[:,i] for i in (_A,_S,_Z,_W))
        d,rr = 1./(z/a+w/s),z-w
        if mu is None:
            da = d*(q@dv-rr)
            return da,-z*(1.+da/a),-w*(1.-da/s)
        da0,dz0,dw0 = affine
        lower,upper = (mu-da0*dz0)/a,(mu+da0*dw0)/s
        da = d*(q@dv+lower-upper-rr)
        return da,lower-z-z*da/a,upper-w+w*da/s

    def steps(self,dv,mu=None,affine_dv=None,primal=0.,dual=0.):
        pr,du,complement,ahead = math.inf,math.inf,_sum(),_sum()
        for _,q,_,_,_,state in self.blocks():
            affine = self.directions(q,state,affine_dv) if mu is not None else None
            da,dz,dw = self.directions(q,state,dv,mu,affine)
            a,s,z,w = (state[:,i] for i in (_A,_S,_Z,_W))
            pr = min(pr,qr._ratio(a,da),qr._ratio(s,-da))
            du = min(du,qr._ratio(z,dz),qr._ratio(w,dw))
            if mu is None:
                complement.add(z@a+w@s)
                ahead.add((z+dual*dz)@(a+primal*da)+(w+dual*dw)@(s-primal*da))
        return min(1.,qr._STEP*pr),min(1.,qr._STEP*du),float(complement.value),float(ahead.value)

    def solve(self,tau,*,max_iterations=qr._MAX_ITERATIONS,max_pivots=None):
        tau = qr._check_tau(tau)
        if self.data.digest() != self.checksum:
            raise AnalysisError("scratch_storage", "Immutable quantile scratch data changed.")
        self.state.clear()
        exact = self.absolute_spread <= qr._EXACT_FIT*self.level or self.absolute_spread == 0.
        floor = 1e-3*self.absolute_spread/self.n
        for _,data in self.data.blocks():
            ys = data[:,self.k]
            state = torch.zeros((len(data),9),dtype=torch.float64)
            state[:,_A],state[:,_S] = 1.-tau,tau
            state[:,_Z],state[:,_W] = (-ys).clamp_min(0),ys.clamp_min(0)
            flat = ys.abs() < floor
            state[flat,_Z] += floor
            state[flat,_W] += floor
            self.state.append(state)
        v,vertex,certified = torch.zeros(self.k,dtype=torch.float64),None,None
        iterations,gap = 0,0. if exact else math.inf
        if not exact:
            for iterations in range(1,max_iterations+1):
                gram,rhs = _sum((self.k,self.k)),_sum((self.k,))
                for _,q,_,_,_,state in self.blocks():
                    a,s,z,w = (state[:,i] for i in (_A,_S,_Z,_W))
                    d,rr = 1./(z/a+w/s),z-w
                    gram.add(q.T@(q*d[:,None]))
                    rhs.add(q.T@(d*rr))
                factor,info = torch.linalg.cholesky_ex((gram.value+gram.value.T)/2)
                if int(info) != 0 or not bool(torch.isfinite(factor).all()):
                    iterations -= 1
                    break
                affine_dv = torch.cholesky_solve(rhs.value[:,None],factor)[:,0]
                primal,dual,_,_ = self.steps(affine_dv)
                mu,dv = None,affine_dv
                if min(primal,dual) < 1:
                    _,_,comp,ahead = self.steps(affine_dv,primal=primal,dual=dual)
                    mu = comp*(ahead/comp)**3/(2*self.n)
                    rhs = _sum((self.k,))
                    for _,q,_,_,_,state in self.blocks():
                        a,s,z,w = (state[:,i] for i in (_A,_S,_Z,_W))
                        da,dz,dw = self.directions(q,state,affine_dv)
                        xi = (mu-da*dz)/a-(mu+da*dw)/s
                        d,rr = 1./(z/a+w/s),z-w
                        rhs.add(q.T@(d*(rr-xi)))
                    dv = torch.cholesky_solve(rhs.value[:,None],factor)[:,0]
                    primal,dual,_,_ = self.steps(dv,mu,affine_dv)
                objective,complement = _sum(),_sum()
                for first,q,_,_,_,state in self.blocks():
                    affine = self.directions(q,state,affine_dv) if mu is not None else None
                    da,dz,dw = self.directions(q,state,dv,mu,affine)
                    state[:,_A] += primal*da
                    state[:,_S] -= primal*da
                    state[:,_Z] += dual*dz
                    state[:,_W] += dual*dw
                    self.state.put(first,state)
                    objective.add(qr.check_loss(state[:,_W]-state[:,_Z],tau))
                    complement.add(state[:,_Z]@state[:,_A]+state[:,_W]@state[:,_S])
                ratio = float(complement.value)/max(float(objective.value),1e-300)
                if not math.isfinite(ratio) or not bool(torch.isfinite(dv).all()):
                    iterations -= 1
                    break
                v += dual*dv
                gap = ratio
                if gap < qr._VERTEX_GAP:
                    vertex = self.vertex(tau,self.basis_near(-v))
                    if vertex.violation <= qr._DUAL_TOL:
                        break
                    certified = self.tied_certificate(tau,vertex)
                    if certified is not None:
                        break
                if gap < qr._GAP:
                    break
        if vertex is None:
            vertex = self.vertex(tau,self.basis_near(-v))
        pivots,limit = 0,200+20*self.k if max_pivots is None else max_pivots
        while not exact and certified is None and vertex.violation > qr._DUAL_TOL:
            if pivots == limit:
                raise KernelError("nonconvergence", "Quantile replay did not reach a certified LP vertex; rescale the outcome or reduce tied indicator terms.")
            pivots += 1
            basis,step = self.pivot(tau,vertex)
            vertex = self.vertex(tau,basis,previous=vertex if step is not None else None,perturbation_step=step)
            if step is None and vertex.violation > qr._DUAL_TOL:
                certified = self.tied_certificate(tau,vertex)
        objective = _sum()
        for _,_,_,_,_,state in self.blocks():
            objective.add(qr.check_loss(state[:,_RESID],tau))
        beta = self.map@(self.ols+vertex.gamma)
        _finite(beta)
        return _Fit(beta,vertex.gamma,vertex.basis,float(objective.value),iterations,pivots,gap,
                    True if exact else self.unique(tau,vertex) if certified is None else certified,exact)

    def residual_blocks(self,fit):
        for first,data in self.data.blocks():
            q,ys,w,norm = data[:,:self.k],data[:,self.k],data[:,self.k+1],data[:,self.k+2]
            residual = ys-q@fit.gamma
            ordinal = torch.arange(first,first+len(q))
            residual[(ordinal[:,None] == fit.basis[None,:]).any(1)] = 0.
            bound = 64*qr._EPS*(ys.abs()+norm*torch.linalg.vector_norm(fit.gamma))
            residual[residual.abs() <= torch.maximum(bound,qr._TIE*self.spread*w)] = 0.
            yield first,residual/w


def _store_order(sql,blocks):
    sql.execute("DELETE FROM ordered")
    index = 0
    for values,weights in blocks:
        _finite(values,weights)
        sql.executemany("INSERT INTO ordered VALUES (?,?,?)", ((float(value),float(weight),index+i)
                        for i,(value,weight) in enumerate(zip(values,weights,strict=True))))
        index += len(values)


def _empirical(sql,p,total,*,raw=False):
    slack = 8*qr._EPS*(total+1 if raw else total)
    target = p*(total+1)-1+slack if raw else p*total+slack
    previous,last,cumulative,comp = None,None,0.,0.
    for value,weight in sql.execute("SELECT value,weight FROM ordered ORDER BY value,row_id"):
        old = cumulative
        adjusted = weight-comp
        cumulative = old+adjusted
        comp = (cumulative-old)-adjusted
        last = value
        if cumulative > target:
            if not raw and previous is not None and abs(old-p*total) <= slack:
                return (previous+value)/2
            return value
        previous = value
    if last is None:
        raise AnalysisError("empty_sample", "An empirical quantile needs observations.")
    return last


def _raw_deviations(problem,tau):
    k = problem.k
    _store_order(problem.sql,((raw[:,k],raw[:,k+1]) for _,raw in problem.base.blocks()))
    raw = _empirical(problem.sql,tau,problem.mass,raw=True)
    loss = _sum()
    for _,data in problem.base.blocks():
        loss.add(qr.check_loss(data[:,k]-raw,tau,data[:,k+1]))
    return raw,float(loss.value)


def _kernel_scale(problem,fit,tau,h,nobs):
    moments = _WeightedMoments(1,intercept=True)
    def records():
        for (_,raw),(_,resid) in zip(problem.base.blocks(),problem.residual_blocks(fit),strict=True):
            weights = raw[:,problem.k+1]
            moments.add(resid[:,None],weights,False)
            yield resid,weights
    _store_order(problem.sql,records())
    q25 = _empirical(problem.sql,.25,problem.mass)
    q75 = _empirical(problem.sql,.75,problem.mass)
    variance = float(moments.m2.value[0])*float(moments.magnitude[0])**2*moments.weight_max/(nobs-1)
    return min(math.sqrt(max(variance,0.)),(q75-q25)/1.34)*(normal_ppf(tau+h)-normal_ppf(tau-h))


def _inverse(matrix,code="singular_design"):
    factor,info = torch.linalg.cholesky_ex((matrix+matrix.T)/2)
    if int(info) != 0 or not bool(torch.isfinite(factor).all()):
        raise AnalysisError(code,"The full-source quantile covariance design is singular; change density/bandwidth or use bootstrap covariance.")
    value = torch.cholesky_inverse(factor)
    _finite(value)
    return value


def _covariance(sample,notes,problem,fit,tau):
    spec,k,n = sample.spec,problem.k,sample.nobs
    kind = spec.covariance
    rule = notes.option("bandwidth")
    method,kernel = single._density_method(notes)
    h = kernel_call(qr.bandwidth,tau,n,rule,single._BANDWIDTH_ALPHA)
    single._window(tau,h,rule)
    record = {"density_method":method,"bandwidth_method":rule,"bandwidth":h,
              "kernel":kernel if method == "kernel" else None,"kernel_bandwidth":None,
              "zero_density_observations":None}
    difference,c = None,None
    if method == "fitted":
        low,high = kernel_call(problem.solve,tau-h),kernel_call(problem.solve,tau+h)
        difference = high.beta-low.beta
        mean,maximum = _sum((k,)),0.
        for _,raw in problem.base.blocks():
            x,w = raw[:,:k],raw[:,k+1]
            mean.add((x*w[:,None]).sum(0))
            maximum = max(maximum,float((x@difference).abs().max()))
        sparsity = float(mean.value@difference)/(problem.mass*2*h)
    elif method == "residual":
        _store_order(problem.sql,((resid,raw[:,k+1]) for (_,raw),(_,resid) in
                                 zip(problem.base.blocks(),problem.residual_blocks(fit),strict=True)))
        sparsity = (_empirical(problem.sql,tau+h,problem.mass)-_empirical(problem.sql,tau-h,problem.mass))/(2*h)
    else:
        c = _kernel_scale(problem,fit,tau,h,n)
        if not c > 0 or not math.isfinite(c):
            raise single._degenerate("kernel bandwidth of the residual density")
        mass = _sum()
        for (_,raw),(_,resid) in zip(problem.base.blocks(),problem.residual_blocks(fit),strict=True):
            density = kernel_call(qr.kernel_values,resid/c,kernel)/c
            mass.add((raw[:,k+1]*density).sum())
        density = float(mass.value)/n
        if not density > 0:
            raise single._degenerate("estimated residual density at zero")
        sparsity = 1/density
        record["kernel_bandwidth"] = c
    if not sparsity > 0 or not math.isfinite(sparsity):
        raise single._degenerate("estimated sparsity")
    record.update({"sparsity":sparsity,"density":1/sparsity})
    info = {"covariance":kind,"df_inference":n-k,"df_resid":n-k}
    gram,meat = _sum((k,k)),_sum((k,k))
    acc = ClusterAccumulator(k) if kind == "cluster" else None
    zeros = 0
    try:
        for (_,raw),(_,resid) in zip(problem.base.blocks(),problem.residual_blocks(fit),strict=True):
            x,w = raw[:,:k],raw[:,k+1]
            if kind == "nonrobust":
                gram.add(x.T@(x*w[:,None]))
                continue
            if method == "fitted":
                spacing = x@difference
                positive = spacing > 1e-10*maximum
                density = torch.where(positive,2*h/spacing.clamp_min(1e-300),torch.zeros_like(spacing))
                zeros += int((~positive).sum())
            else:
                density = kernel_call(qr.kernel_values,resid/c,kernel)/c
            gram.add(x.T@(x*(w*density)[:,None]))
            if kind == "robust":
                square = w if spec.weight_type == "fweight" else w.square()
                meat.add((x.T@(x*square[:,None]))*(tau*(1-tau)))
            else:
                psi = torch.full_like(resid,tau)
                psi[resid <= 0] = tau-1.
                acc.add([int(code).to_bytes(8,"little") for code in raw[:,k+3].tolist()],x*(w*psi)[:,None])
        bread = _inverse(gram.value,"singular_density_matrix" if kind != "nonrobust" else "singular_design")
        if kind == "nonrobust":
            covariance = bread*(tau*(1-tau)*sparsity**2)
            info["correction"] = f"i.i.d. errors: tau (1 - tau) s^2 (X'WX)^-1, {method} density, {rule} bandwidth"
        else:
            matrix = meat.value
            if acc is not None:
                matrix,groups = acc.finish()
                info.update({"cluster_count":groups,"cluster_df":groups-1,"cluster_columns":[spec.cluster],
                             "cluster_column":spec.cluster,"cluster_spill":acc.diagnostics})
            covariance = bread@matrix@bread
            info["correction"] = f"{method} density sandwich, {rule} bandwidth; no finite-sample factor"
            if method == "fitted":
                record["zero_density_observations"] = zeros
        _finite(covariance)
        return (covariance+covariance.T)/2,info,record
    finally:
        if acc is not None:
            acc.close()


def _bootstrap(sample,notes,root,sql,base,taus,k,groups):
    seed = sample.spec.options.get("seed")
    seed = secrets.randbelow(2**31) if seed is None else int(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    reps = int(notes.option("reps"))
    dimension = k*len(taus)
    centre = torch.zeros(dimension,dtype=torch.float64)
    covariance = _sum((dimension,dimension))
    accepted,passes = 0,0
    problem = _Problem(root,sql,base,k,bootstrap=True)
    try:
        for _ in range(reps):
            sql.execute("DELETE FROM counts")
            total = groups if sample.spec.cluster else sample.nrows
            for first in range(0,total,base.rows):
                draws = torch.randint(total,(min(base.rows,total-first),),generator=generator)
                sql.executemany("INSERT INTO counts VALUES (?,1) ON CONFLICT(row_id) DO UPDATE SET multiplicity=multiplicity+1", ((int(index),) for index in draws))
            problem.nbase = 0
            problem.cluster_bootstrap = sample.spec.cluster is not None
            try:
                problem.prepare()
                beta = torch.cat([problem.solve(tau).beta for tau in taus])
            except KernelError:
                continue  # Reject the entire JOINT draw, never one quantile alone.
            accepted += 1
            difference = beta-centre
            centre += difference/accepted
            covariance.add(difference[:,None]*(beta-centre)[None,:])
        passes = problem.disk_passes
    finally:
        problem.close()
    if accepted < 2:
        raise AnalysisError("bootstrap_failed","Fewer than two full-rank joint bootstrap samples could be estimated.")
    if accepted != reps:
        notes.warn(f"{reps-accepted} of {reps} bootstrap samples could not be estimated and were left out of the joint covariance.")
    record = {"reps":reps,"reps_used":accepted,"seed":seed,
              "resampling":"clusters" if sample.spec.cluster else "observations",
              "variance":"about the replicate mean, divisor R - 1",
              "draw_storage":"bounded global randint draws; exact SQLite multiplicities",
              "joint_draws":True,"disk_numerical_passes":passes}
    if sample.spec.cluster:
        record.update({"cluster_column":sample.spec.cluster,"cluster_count":groups})
    value = covariance.value/(accepted-1)
    return (value+value.T)/2,record


def _spool(sample,root,sql,k):
    base = _Rows(root/"source.bin",k+4,sample.rows)
    groups = 0
    try:
        for batch in sample.batches():
            x,y,w = batch.designs["x"],batch.numeric(sample.spec.outcome),_weights(sample,batch)
            if bool((batch.positions > 2**53).any()):
                raise AnalysisError("prediction_precision","Quantile physical row positions exceed exact float64 integer precision.")
            codes = torch.zeros(len(y),dtype=torch.float64)
            if sample.spec.cluster:
                keys = encode_cluster_labels(batch.frame[sample.spec.cluster])
                for i,key in enumerate(keys):
                    found = sql.execute("SELECT code FROM labels WHERE label=?",(key,)).fetchone()
                    if found is None:
                        code = groups
                        groups += 1
                        sql.execute("INSERT INTO labels VALUES (?,?)",(key,code))
                    else:
                        code = found[0]
                    codes[i] = code
            base.append(torch.cat((x,y[:,None],w[:,None],batch.positions.to(torch.float64)[:,None],codes[:,None]),1))
        if sample.spec.cluster and groups < 2:
            raise AnalysisError("insufficient_clusters","Cluster inference requires at least two observed clusters.")
        return base,groups
    except BaseException:
        base.close()
        raise


def _predictions(base,fit,k):
    result = []
    for _,raw in base.blocks():
        fitted = raw[:,:k]@fit.beta
        take = min(400-len(result),len(raw))
        for row,observed,predicted in zip(raw[:take,k+2].tolist(),raw[:take,k].tolist(),fitted[:take].tolist(),strict=True):
            result.append({"row":int(row),"observed":observed,"fitted":predicted,"residual":observed-predicted})
    return result


def fit_streaming_quantile(spec: ModelSpec,source: Dataset,*,batch_rows=None) -> ResultBundle:
    """Fit qreg/bsqreg/sqreg/iqreg with exact full-source LP certificates."""
    if spec.estimator not in SUPPORTED:
        raise AnalysisError("unsupported_streaming_estimator","This replay adapter is for qreg, bsqreg, sqreg and iqreg.")
    with torch.no_grad(),torch.device("cpu"):
        return _fit_quantile(spec,source,batch_rows)


def _fit_quantile(spec,source,batch_rows):
    notes = _Notes(spec)
    if spec.estimator in {"qreg","bsqreg"}:
        taus = [single.check_quantile(notes.option("quantile"))]
    else:
        taus = multiple._quantile_list(notes,exactly=2 if spec.estimator == "iqreg" else None)
        if spec.estimator == "iqreg" and not taus[0] < taus[1]:
            raise AnalysisError("invalid_quantile","Give the lower quantile before the upper quantile.")
    labels = [multiple.quantile_label(tau) for tau in taus]
    if len(set(labels)) != len(labels):
        raise AnalysisError("invalid_quantile","Requested quantiles have colliding equation labels.")
    if spec.weight_type and spec.estimator != "qreg":
        raise AnalysisError("unsupported_weights","Bootstrap quantile estimators do not accept user weights.")
    if spec.weight_type == "pweight" and spec.covariance == "nonrobust":
        raise AnalysisError("unsupported_covariance","Quantile pweights require robust or cluster covariance.")
    if spec.cluster and len(registry.cluster_columns(spec)) != 1:
        raise AnalysisError("unsupported_cluster_dimensions","Quantile replay accepts one cluster dimension.")
    sample = ReplaySample(spec,source,batch_rows=batch_rows)
    design = sample.add_design("x")
    sample.prepare()
    k,n = len(design.terms),sample.nobs
    if not k:
        raise AnalysisError("no_regressors","Quantile regression needs at least one estimable term.")
    if sample.nrows <= k:
        raise AnalysisError("insufficient_observations","Quantile regression needs more physical rows than parameters.")
    dimension = k*len(taus)
    resource = sample.plan_rows("exact full-source quantile regression", {
        "quantile_global_factors":640*(k+1)**2,
        "joint_bootstrap_covariance":128*(dimension+1)**2 if spec.estimator != "qreg" else 0,
        "quantile_SQLite_and_cluster_caches":16*1024**2,
    },256*(k+16)).record()
    with _scratch() as (root,sql):
        minimum_disk_bytes = sample.nrows*(2*k+16)*8
        if minimum_disk_bytes > shutil.disk_usage(root).free:
            raise AnalysisError("scratch_space","Exact quantile replay needs more local scratch space for its design and dual records; choose OPENECON_SCRATCH_DIRECTORY on a larger disk.")
        base,groups = _spool(sample,root,sql,k)
        checksum = base.digest()
        problem = _Problem(root,sql,base,k)
        try:
            kernel_call(problem.prepare)
            fits,raws = [],[]
            for tau in taus:
                fit = kernel_call(problem.solve,tau)
                raw = _raw_deviations(problem,tau)
                single.require_variation(fit,raw[1])
                fits.append(fit)
                raws.append(raw)
            if not all(fit.unique for fit in fits):
                notes.warn("The quantile-regression solution is not unique; one certified optimal vertex is reported.")
            extra,provenance = {},{}
            if spec.estimator == "qreg":
                covariance,info,density = _covariance(sample,notes,problem,fits[0],taus[0])
                beta,terms = fits[0].beta,design.terms
                extra.update({"density_method":density["density_method"],"bandwidth_method":density["bandwidth_method"],
                              "kernel":density["kernel"],"bandwidth_alpha":single._BANDWIDTH_ALPHA,
                              "unique_solution":fits[0].unique})
                if density["zero_density_observations"] is not None:
                    extra["zero_density_observations"] = density["zero_density_observations"]
                metrics = {"sparsity":density["sparsity"],"density":density["density"],"bandwidth":density["bandwidth"],
                           "kernel_bandwidth":density["kernel_bandwidth"]}
            else:
                # Point problem handles are closed before the bootstrap reuses
                # its owned filenames. Fit records contain only K-sized state.
                problem.close()
                problem = None
                covariance,record = _bootstrap(sample,notes,root,sql,base,taus,k,groups)
                info = multiple._inference(record,n-k)
                info["df_resid"] = n-k
                extra["bootstrap"],provenance["bootstrap"] = record,record
                metrics = {"reps":record["reps_used"]}
                if spec.estimator == "iqreg":
                    contrast = torch.cat((-torch.eye(k,dtype=torch.float64),torch.eye(k,dtype=torch.float64)),1)
                    covariance = contrast@covariance@contrast.T
                    beta,terms = fits[1].beta-fits[0].beta,design.terms
                    extra.update({"quantiles":taus,"coefficients_low":dict(zip(terms,(design.transform@fits[0].beta).tolist(),strict=True)),
                                  "coefficients_high":dict(zip(terms,(design.transform@fits[1].beta).tolist(),strict=True))})
                elif spec.estimator == "sqreg":
                    beta = torch.cat([fit.beta for fit in fits])
                    terms = [f"{label}:{term}" for label in labels for term in design.terms]
                    extra.update({"quantiles":taus,"equations":labels,
                                  "sum_adev":dict(zip(labels,[fit.objective for fit in fits],strict=True)),
                                  "sum_rdev":dict(zip(labels,[raw[1] for raw in raws],strict=True)),
                                  "raw_quantile":dict(zip(labels,[raw[0] for raw in raws],strict=True))})
                else:
                    beta,terms = fits[0].beta,design.terms
            metrics.update({"df_model":k-int(design.intercept),"df_resid":n-k})
            if spec.estimator in {"qreg","bsqreg"}:
                metrics.update({"quantile":taus[0],"pseudo_r_squared":multiple._pseudo(fits[0],raws[0]),
                                "sum_adev":fits[0].objective,"sum_rdev":raws[0][1],"raw_quantile":raws[0][0],"iterations":fits[0].iterations})
                title = single.quantile_title(taus[0])+("" if spec.estimator == "qreg" else ", bootstrap standard errors")
                diagnostics = single.solver_record(fits[0])
                predictions = _predictions(base,fits[0],k)
            elif spec.estimator == "iqreg":
                metrics.update({"quantile_low":taus[0],"quantile_high":taus[1],
                                "pseudo_r_squared_low":multiple._pseudo(fits[0],raws[0]),"pseudo_r_squared_high":multiple._pseudo(fits[1],raws[1])})
                title = f"{taus[0]:g}-{taus[1]:g} Interquantile regression, bootstrap standard errors"
                diagnostics = {"low":single.solver_record(fits[0]),"high":single.solver_record(fits[1])}
                predictions = []
            else:
                metrics.update({f"pseudo_r_squared_{label}":multiple._pseudo(fit,raw) for label,fit,raw in zip(labels,fits,raws,strict=True)})
                title = "Simultaneous quantile regression, bootstrap standard errors"
                diagnostics = dict(zip(labels,[single.solver_record(fit) for fit in fits],strict=True))
                predictions = []
            transform = torch.block_diag(*([design.transform]*len(taus))) if spec.estimator == "sqreg" else design.transform
            beta,covariance = transform@beta,transform@covariance@transform.T
            _finite(beta,covariance)
            if base.digest() != checksum:
                raise AnalysisError("scratch_storage","Immutable quantile source spool changed.")
            for _ in sample.batches():
                pass  # Verify raw input and exact missing/position contract before publishing.
            sql.commit()
            if problem is not None:
                problem.state.file.flush()
            extra["replay_solver"] = {"state":"owned sequential disk records","exact_lp_certificate":True,
                                      "row_state_resident":False,"source_spool_hash":checksum,
                                      "minimum_state_storage_bytes":minimum_disk_bytes,
                                      "disk_numerical_passes":problem.disk_passes if problem else record["disk_numerical_passes"],
                                      "scratch_bytes":sum(path.stat().st_size for path in root.iterdir() if path.is_file())}
            result = _result(sample,terms=terms,beta=beta,covariance=(covariance+covariance.T)/2,info=info,metrics=metrics,
                           notes=notes,predictions=predictions,tests=[],solver="frisch_newton_interior_point",
                           diagnostics=diagnostics,extra=extra,resource=resource,title=title,provenance_extra=provenance)
            if spec.estimator == "sqreg":
                for coefficient,label in zip(result.coefficients,[label for label in labels for _ in range(k)],strict=True):
                    coefficient.equation = label
            return result
        finally:
            if problem is not None:
                problem.close()
            base.close()
