"""Exact external midranks and disk Fenwick inversions, never an N-row sort."""
from __future__ import annotations

import math

import torch

from openecon.econometrics.core import TableSet
from openecon.engines import distributions as dist
from openecon.resources import plan_workspace
from . import common as c
from .correlation import _t_p_value, _Z_VARIANCE
from .replay import Moments, Replay
from .spill import Spill, blocks


def _ties(db, column):
    a = b = d = 0
    for (n,) in db.execute(f"SELECT COUNT(*) FROM pairs GROUP BY {column}"):
        a += n*(n-1)
        b += n*(n-1)*(n-2)
        d += n*(n-1)*(2*n+5)
    return a, b, d


def _kendall(db, n):
    if n < 2:
        return None, None
    t1, t2, t3 = _ties(db, "a")
    u1, u2, u3 = _ties(db, "b")
    joint = sum(k*(k-1) for (k,) in db.execute("SELECT COUNT(*) FROM pairs GROUP BY a,b"))
    pairs = n*(n-1)//2
    denominator = math.sqrt((pairs-t1/2)*(pairs-u1/2))
    if denominator == 0:
        return None, None
    db.execute("CREATE TABLE yr AS SELECT b, ROW_NUMBER() OVER (ORDER BY b) r FROM pairs GROUP BY b")
    db.execute("CREATE UNIQUE INDEX yr_b ON yr(b)")
    width = db.execute("SELECT COUNT(*) FROM yr").fetchone()[0]
    db.execute("CREATE TABLE tree (r INTEGER PRIMARY KEY, n INTEGER NOT NULL)")
    db.execute("CREATE TABLE joint AS SELECT a,b,COUNT(*) n,r FROM pairs JOIN yr USING(b) GROUP BY a,b")
    db.execute("CREATE INDEX joint_a ON joint(a)")

    def flush(value):
        total = 0
        for r, count in db.execute("SELECT r,n FROM joint WHERE a=?", (value,)):
            total += count
            while r <= width:
                db.execute("INSERT INTO tree VALUES (?,?) ON CONFLICT(r) DO UPDATE SET n=n+excluded.n", (r, count))
                r += r & -r
        return total

    previous, seen, discordant = None, 0, 0
    for a, r, count in db.execute("SELECT a,r,n FROM joint ORDER BY a,b"):
        if previous is not None and a != previous:
            seen += flush(previous)
        prefix = 0
        while r:
            row = db.execute("SELECT n FROM tree WHERE r=?", (r,)).fetchone()
            prefix += 0 if row is None else row[0]
            r -= r & -r
        discordant += count*(seen-prefix)
        previous = a
    score = pairs - t1/2 - u1/2 + joint/2 - 2*discordant
    variance = (n*(n-1)*(2*n+5)-t3-u3)/18 + t1*u1/(2*n*(n-1))
    if n > 2:
        variance += t2*u2/(9*n*(n-1)*(n-2))
    return max(-1., min(1., score/denominator)), score/math.sqrt(variance) if variance > 0 else None


def _spearman(db):
    state = Moments(2)
    query = """SELECT ra,rb FROM (
        SELECT RANK() OVER (ORDER BY a) + (COUNT(*) OVER (PARTITION BY a)-1)/2.0 ra,
               RANK() OVER (ORDER BY b) + (COUNT(*) OVER (PARTITION BY b)-1)/2.0 rb
        FROM pairs)"""
    for block in blocks(db.execute(query)):
        state.add(torch.tensor(block, dtype=torch.float64))
    denominator = c.root_product(float(state.sscp[0,0]), float(state.sscp[1,1]))
    return max(-1., min(1., float(state.sscp[0,1])/denominator)) if denominator else None


def correlate(data, names, *, method, pairwise, ci, alpha):
    sample = Replay(data, names, names, "drop")
    p = len(names)
    plan = plan_workspace("exact external rank correlations", {
        "pair_result_geometry": 512*p*p, "source_blocks": sample.plan.estimated_bytes,
        "SQLite_page_cache_and_rank_blocks": 4*1024**2})
    coefficients = [[None]*p for _ in names]
    counts = [[0]*p for _ in names]
    probabilities = [[None]*p for _ in names]
    states = [Moments(1) for _ in names]
    for frame in sample.batches(listwise=not pairwise):
        for name, state in zip(names, states, strict=True):
            values = c.values(frame, name, allow_missing=pairwise, check_scale=False)
            state.add(values[~torch.isnan(values), None])
    from openecon.analysis_contracts import AnalysisError
    if sample.n < 2:
        raise AnalysisError("insufficient_observations", "Correlations need at least two complete observations.")
    for i, state in enumerate(states):
        counts[i][i] = state.n
        coefficients[i][i] = 1. if state.n > 1 and float(state.sscp[0,0]) > 0 else None
        for j in range(i+1,p):
            with Spill() as db:
                db.execute("CREATE TABLE pairs(a REAL NOT NULL,b REAL NOT NULL)")
                n = 0
                for frame in sample.batches(listwise=not pairwise):
                    pair = frame[[names[i],names[j]]].dropna()
                    n += len(pair)
                    db.executemany("INSERT INTO pairs VALUES (?,?)", pair.itertuples(index=False,name=None))
                db.execute("CREATE INDEX pair_a ON pairs(a)")
                db.execute("CREATE INDEX pair_b ON pairs(b)")
                if method == "spearman":
                    value = _spearman(db) if n > 1 else None
                    probability = _t_p_value(value,n)
                else:
                    value,z = _kendall(db,n)
                    probability = None if z is None else min(1.,2*dist.normal_sf(abs(z)))
            counts[i][j] = counts[j][i] = n
            coefficients[i][j] = coefficients[j][i] = value
            probabilities[i][j] = probabilities[j][i] = probability
    tables = {"coefficients": c.frame(coefficients,columns=names,index=names),
        "p_values": c.frame(probabilities,columns=names,index=names),
        "n": c.frame(counts,columns=names,index=names)}
    if ci:
        factor,offset = _Z_VARIANCE[method]
        bound = dist.normal_isf(alpha/2)
        rows = []
        for i in range(p):
            for j in range(i+1,p):
                value,n = coefficients[i][j],counts[i][j]
                low = high = None
                if value is not None and n > offset:
                    if abs(value) == 1:
                        low = high = value
                    else:
                        half = bound*math.sqrt(factor/(n-offset))
                        low,high = math.tanh(math.atanh(value)-half),math.tanh(math.atanh(value)+half)
                rows.append([names[i],names[j],value,low,high,n])
        tables["intervals"] = c.frame(rows,columns=["var_i","var_j","coefficient","ci_low","ci_high","n"])
    return TableSet(tables,title=f"{method.capitalize()} correlations",method=method,
        missing="pairwise" if pairwise else "listwise",n=sample.raw_rows,
        n_complete=sample.raw_rows-sample.dropped,alpha=alpha,
        distribution="normal" if method == "kendall" else "t",**sample.attrs(),
        exact_external_order=True,full_source_collected=False,rank_resource_plan=plan.record())
