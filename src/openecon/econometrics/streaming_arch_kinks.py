"""Native EGARCH active-set Newton equations with disk-resident signs.

The smooth-piece likelihood and KKT equations are unchanged. Sign patterns and
crossed observation sets grow on owned scratch storage; active normals remain
at most parameter width. No preview substitutes for the complete sample.
"""
from __future__ import annotations

import math
from pathlib import Path
import sqlite3
import uuid

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.arch import kinks
from openecon.engines import optimize
from openecon.engines.contracts import KernelError


class Pattern:
    def __init__(self, path, n, overrides=None):
        self.path, self.n, self.overrides = Path(path), n, dict(overrides or {})

    def clone(self):
        return Pattern(self.path, self.n, self.overrides)

    def __getitem__(self, index):
        return self.block(index, 1)[0] if isinstance(index, int) else torch.tensor([float(self[position]) for position in index], dtype=torch.float64)

    def __setitem__(self, index, values):
        if isinstance(index, int):
            self.overrides[index] = float(values)
        else:
            for position, value in zip(index, values, strict=True):
                self.overrides[position] = float(value)

    def block(self, start, count):
        with self.path.open("rb") as handle:
            handle.seek(8*start)
            data = handle.read(8*count)
        if len(data) != 8*count:
            raise AnalysisError("state_spill_failed", "An owned EGARCH sign snapshot was truncated.")
        values = torch.frombuffer(bytearray(data), dtype=torch.float64)
        for position, value in self.overrides.items():
            if start<=position<start+count:
                values[position-start] = value
        return values


class Crossings:
    def __init__(self, path):
        self.path = Path(path)
        self.connection = sqlite3.connect(path)
        for pragma in ("journal_mode=OFF", "synchronous=OFF", "temp_store=FILE", "cache_size=-2048", "mmap_size=0"):
            self.connection.execute("PRAGMA "+pragma)
        self.connection.execute("CREATE TABLE indices(i INTEGER PRIMARY KEY)")
        Path(path).chmod(0o600)
        self.count = 0

    def add(self, values):
        self.connection.executemany("INSERT INTO indices VALUES(?)", ((int(value),) for value in values))
        self.count += len(values)

    def contained(self, previous):
        if previous is None:
            return False
        self.connection.commit()
        previous.connection.commit()
        self.connection.execute("ATTACH DATABASE ? AS frozen", (str(previous.path),))
        try:
            return self.connection.execute("SELECT 1 FROM indices a WHERE NOT EXISTS(SELECT 1 FROM frozen.indices b WHERE b.i=a.i) LIMIT 1").fetchone() is None
        finally:
            self.connection.execute("DETACH DATABASE frozen")

    def batches(self):
        query = self.connection.execute("SELECT i FROM indices ORDER BY i")
        while rows := query.fetchmany(4096):
            yield [int(row[0]) for row in rows]

    def in_block(self, start, count):
        return [int(row[0]) for row in self.connection.execute("SELECT i FROM indices WHERE i>=? AND i<?", (start, start+count))]

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def _path(like, suffix):
    return like.scratch/(uuid.uuid4().hex+suffix)


def freeze(like, theta, signs=None, active=None):
    path, on_kink = _path(like, ".signs"), []
    with path.open("wb") as handle:
        path.chmod(0o600)
        for row in like.blocks(theta, signs=signs):
            if row is None:
                return None, []
            start, residual, _, variance, *_ = row
            handle.write(kinks.piece(residual).contiguous().numpy().tobytes())
            if like.layout.dist == "ged":
                found = (residual.abs()<=kinks._ZERO*variance.sqrt()).nonzero().flatten().tolist()
                on_kink.extend(start+at for at in found)
                if len(on_kink)>=like.layout.k:
                    return None, []
    pattern = Pattern(path, like.n)
    for position in active or []:
        pattern[position] = signs[position]
    return pattern, on_kink


def piece_hessian(like, offset, transform, t, signs=None, active=None):
    generated = signs is None
    if signs is None:
        signs, _ = freeze(like, offset+transform@t)
        if signs is None:
            raise KernelError("numerical_failure", "EGARCH smooth-piece state is undefined.")
    flat, undefined = like.replay_flat_mask(active), torch.full_like(t, math.nan)
    def gradient(point):
        out = like.evaluate(offset+transform@point, signs=signs, flat=flat)
        return undefined if out is None else transform.T@out.gradient
    try:
        return optimize.numerical_hessian(gradient, t, relative_step=kinks.HESSIAN_STEP)
    finally:
        if generated:
            signs.path.unlink(missing_ok=True)


def _solve(like, offset, transform, t, signs, active, known=None):
    k, m = t.numel(), len(active)
    t, signs, start = t.clone(), signs.clone(), t.clone()
    flat = like.replay_flat_mask(active)
    def state(point, pattern):
        out = like.evaluate(offset+transform@point, signs=pattern, active=active, flat=flat)
        if out is None:
            return None
        gradient = transform.T@out.gradient
        if not m:
            return gradient, torch.empty(0, dtype=torch.float64), torch.empty((0, k), dtype=torch.float64)
        values = [out.selected[position] for position in active]
        return gradient, torch.stack([e/h.sqrt() for e, h, _ in values]), torch.stack([dz@transform for _, _, dz in values])
    def jacobian(point, normals):
        try:
            hessian = known if known is not None and torch.equal(point, start) else piece_hessian(like, offset, transform, point, signs, active)
            inverse = optimize.information_inverse(-hessian)
        except KernelError:
            return None
        matrix = torch.zeros((k+m, k+m), dtype=torch.float64)
        matrix[:k, :k], matrix[k:, :k] = hessian, normals
        for column, position in enumerate(active):
            sides = []
            for value in (1., -1.):
                pattern = signs.clone()
                pattern[position] = value
                out = state(point, pattern)
                if out is None:
                    return None
                sides.append(out[0])
            matrix[:k, k+column] = .5*(sides[0]-sides[1])
        return matrix, inverse
    system, previous, evaluations, here = None, math.inf, 0, False
    for iteration in range(kinks._MAX_NEWTON):
        current = state(t, signs)
        if current is None:
            return None
        gradient, z, normals = current
        if system is None:
            if evaluations>=kinks._MAX_JACOBIANS:
                return None
            system, evaluations, here = jacobian(t, normals), evaluations+1, True
            if system is None:
                return None
        matrix, inverse = system
        scaled, distance = float(gradient@inverse@gradient), float(z.abs().max()) if m else 0.
        if not (math.isfinite(scaled) and math.isfinite(distance)):
            return None
        if scaled<=kinks._SCALED_GRADIENT and distance<=kinks._ZERO:
            hessian = matrix[:k, :k].clone()
            if not here:
                try:
                    hessian = piece_hessian(like, offset, transform, t, signs, active)
                    optimize.information_inverse(-hessian)
                except KernelError:
                    return None
            return t, signs, hessian, [float(matrix[:k, k+j]@matrix[k+j, :k]) for j in range(m)], iteration
        measure = scaled+distance**2
        if measure>.25*previous and iteration:
            system, previous = None, math.inf
            continue
        previous = measure
        try:
            step = torch.linalg.solve(matrix, -torch.cat((gradient, z)))
        except RuntimeError:
            return None
        if not bool(torch.isfinite(step).all()):
            return None
        t += step[:k]
        if m:
            signs[active] = signs[active]+step[k:]
        here = False
    return None


def _crossings(like, theta, signs, active):
    result = Crossings(_path(like, ".sqlite"))
    like.kink_stores.append(result)
    for row in like.blocks(theta, signs=signs):
        if row is None:
            return None
        start, residual, *_ = row
        crossed = residual*signs.block(start, len(residual))<0.
        for position in active:
            if start<=position<start+len(residual):
                crossed[position-start] = False
        result.add([start+at for at in crossed.nonzero().flatten().tolist()])
    return result


def _first_crossing(like, before, after, signs, target_signs, crossings):
    best, first = math.inf, None
    for a, b in zip(like.blocks(before, signs=signs), like.blocks(after, signs=target_signs), strict=True):
        if a is None or b is None:
            return None
        start, old, *_ = a
        indices = crossings.in_block(start, len(old))
        if not indices:
            continue
        local = [position-start for position in indices]
        gap, residual = old[local]-b[1][local], old[local]
        fraction = torch.where(gap != 0., residual/gap, torch.ones_like(gap)).clamp(0., 1.)
        at = int(fraction.argmin())
        if float(fraction[at])<best:
            best, first = float(fraction[at]), indices[at]
    return best, first


def polish(like, offset, transform, start, hessian=None):
    try:
        return _polish(like, offset, transform, start, hessian)
    finally:
        # Only the current/refrozen sets can be live during a round. Their
        # files survive until the owned likelihood directory is removed.
        for store in like.kink_stores:
            store.close()
        like.kink_stores.clear()


def _polish(like, offset, transform, start, hessian=None):
    signs, on_kink = freeze(like, offset+transform@start)
    if signs is None:
        return None
    t, active = start.clone(), on_kink if like.layout.dist == "ged" else []
    if len(active)>=t.numel():
        return None
    refrozen, iterations = None, 0
    for number in range(kinks._MAX_ROUNDS):
        known = hessian if number == 0 and hessian is not None and not active and bool(torch.isfinite(hessian).all()) else None
        solved = _solve(like, offset, transform, t, signs, active, known)
        if solved is None:
            return None
        target, target_signs, hessian, ridge, steps = solved
        iterations += steps
        crossed = _crossings(like, offset+transform@target, target_signs, active)
        if crossed is None:
            return None
        if crossed.count>kinks._MAX_NEW_KINKS and not crossed.contained(refrozen):
            if refrozen is not None:
                refrozen.close()
            refrozen = crossed
            signs, _ = freeze(like, offset+transform@target, target_signs, active)
            if signs is None:
                return None
            t = target
            continue
        if crossed.count:
            first = _first_crossing(like, offset+transform@t, offset+transform@target, signs, target_signs, crossed)
            crossed.close()
            if first is None:
                return None
            share, position = first
            t += share*(target-t)
            if active:
                signs = signs.clone()
                signs[active] = signs[active]+share*(target_signs[active]-signs[active])
            active = sorted({*active, position})
            if len(active)>=t.numel():
                return None
            continue
        crossed.close()
        t, signs, released = target, target_signs, []
        for position, curvature in zip(active, ridge, strict=True):
            multiplier = float(signs[position])
            if curvature<0. and abs(multiplier)<=1.:
                continue
            side = math.copysign(1., multiplier)
            signs[position] = side if curvature<0. else -side
            released.append(position)
        if not released:
            return kinks.KinkSolution(t, signs, active, hessian, iterations)
        active = [position for position in active if position not in released]
    return None
