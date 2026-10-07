"""Exact Gaussian mixed likelihood on owned, bounded group QR factors.

No group holds its observations in RAM.  The QR of [Z, 1, X, y] is a
sufficient representation for its covariance action; zero directions have
unit relative variance and contribute zero to the relative determinant.
Two nested levels use a second QR with the whitened constant first, so the
top-level rank-one correction scales one row instead of subtracting Gram
matrices.  Original counts remain in the Gaussian density.
"""
from __future__ import annotations

from array import array
from contextlib import ExitStack
import math
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines import optimize
from openecon.engines.contracts import KernelError
from openecon.engines.linalg import collinear_columns
from openecon.engines.streaming_groups import ClusterAccumulator
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from openecon.linear_ols.streaming import _WeightedMoments
from openecon.resources import plan_workspace, workspace_budget_bytes
from openecon.streaming_design import MAX_CLUSTER_BYTES, encode_cluster_labels

from . import registry
from .core import information_criteria, wald_test
from .mixed import common
from .mixed.lmm import _BOUNDARY, _SINGULAR_CORRELATION, _reported, _variance_terms
from .mixed.lmm_kernels import CovStructure, MixedEvaluation, MixedLikelihood
from .replay_sample import ReplaySample, WORKING_BYTES
from .streaming_linear import _GroupMeans, _Notes, _finite, _result, _scratch_directory, _solve

_LOG_2PI = math.log(2 * math.pi)


def _blob(value):
    return array("d", value.reshape(-1).tolist()).tobytes()


def _matrix(value, width):
    if len(value) != 8 * width * width:
        raise KernelError("mixed_spill_failed", "An owned mixed factor record is malformed.")
    return torch.frombuffer(bytearray(value), dtype=torch.float64).reshape(width, width)


def _key_parts(keys):
    """Bound both SQLite's bind count and simultaneous encoded-key payload."""
    part, size = [], 0
    for key in keys:
        if part and (len(part) == 256 or size + len(key) > 1024**2):
            yield part
            part, size = [], 0
        part.append(key)
        size += len(key)
    if part:
        yield part


def _kernel_finite(*values):
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise KernelError("numerical_failure", "The mixed likelihood exceeds finite float64 precision.")


def _coordinate_scales(logarithms):
    value = torch.exp(torch.as_tensor(logarithms, dtype=torch.float64))
    if not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
        raise AnalysisError("mixed_precision", "Original-unit mixed parameter conversion exceeds float64 precision; rescale the variables.")
    return value


class _Factors:
    """Batched ragged TSQR updates; SQLite owns all low/top group factors."""

    def __init__(self, width, rows):
        self.width, self.rows = width, rows
        self.directory = tempfile.TemporaryDirectory(prefix="openecon-mixed-", dir=_scratch_directory())
        self.path = Path(self.directory.name) / "factors.sqlite3"
        try:
            self.connection = sqlite3.connect(self.path, isolation_level=None)
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA cache_size=-2048")
            self.connection.execute("PRAGMA mmap_size=0")
            self.connection.execute("PRAGMA temp_store=FILE")
            self.connection.execute("CREATE TABLE low(key BLOB PRIMARY KEY, parent BLOB, value BLOB, rows INTEGER, mass REAL) WITHOUT ROWID")
            self.connection.execute("CREATE INDEX low_parent ON low(parent,key)")
            self.connection.execute("CREATE TABLE parents(key BLOB PRIMARY KEY, cluster BLOB) WITHOUT ROWID")
            self.connection.execute("CREATE TABLE top(key BLOB PRIMARY KEY, value BLOB, mass REAL) WITHOUT ROWID")
            self.path.chmod(0o600)
        except BaseException:
            self.close()
            raise
        self.peak_groups = self.peak_packed_rows = self.updates = 0

    def close(self):
        connection = getattr(self, "connection", None)
        if connection is not None:
            connection.close()
            self.connection = None
        directory = getattr(self, "directory", None)
        if directory is not None:
            directory.cleanup()

    def add(self, keys, parents, clusters, values, weights):
        unique, codes = _GroupMeans._codes(keys)
        h = self.width
        old = torch.zeros((len(unique), h, h), dtype=torch.float64)
        physical = torch.zeros(len(unique), dtype=torch.int64)
        mass = torch.zeros(len(unique), dtype=torch.float64)
        mapping = {key: index for index, key in enumerate(unique)}
        for part in _key_parts(unique):
            query = "SELECT key,value,rows,mass FROM low WHERE key IN (" + ",".join("?" for _ in part) + ")"
            for key, raw, count, weight in self.connection.execute(query, part):
                index = mapping[key]
                old[index] = _matrix(raw, h)
                physical[index], mass[index] = count, weight
        order = torch.argsort(codes, stable=True)
        lengths = torch.bincount(codes, minlength=len(unique))
        starts = torch.cumsum(lengths, 0) - lengths
        first_rows = order[starts].tolist()
        parent_of = [parents[index] for index in first_rows]
        # Nesting is checked at the TOP level, including across separate low
        # groups and source blocks. Lowest labels are qualified by their parent.
        parent_labels = {}
        for parent, label in zip(parents, clusters or [None] * len(parents), strict=True):
            if parent in parent_labels and parent_labels[parent] != label:
                raise AnalysisError("cluster_not_nested", "Mixed top-level groups must nest within covariance clusters.")
            parent_labels[parent] = label
        for parent, label in parent_labels.items():
            previous = self.connection.execute("SELECT cluster FROM parents WHERE key=?", (parent,)).fetchone()
            if previous is not None and previous[0] != label:
                raise AnalysisError("cluster_not_nested", "Mixed top-level groups must nest within covariance clusters.")
        self.connection.executemany("INSERT OR IGNORE INTO parents VALUES(?,?)", parent_labels.items())
        weighted = values * weights.sqrt()[:, None]
        for power in range((int(lengths.max()) - 1).bit_length() + 1):
            padded = 1 << power
            selected = ((lengths <= padded) & (lengths > padded // 2)).nonzero().flatten()
            if not len(selected):
                continue
            packed = torch.zeros((len(selected), h + padded, h), dtype=torch.float64)
            packed[:, :h] = old[selected]
            local = torch.full((len(unique),), -1, dtype=torch.int64)
            local[selected] = torch.arange(len(selected))
            indices = order[local[codes[order]] >= 0]
            offsets = torch.arange(len(codes))[local[codes[order]] >= 0] - starts[codes[indices]]
            packed[local[codes[indices]], h + offsets] = weighted[indices]
            result = torch.linalg.qr(packed, mode="r")[1]
            old[selected] = result
            self.peak_packed_rows = max(self.peak_packed_rows, len(selected) * (h + padded))
            del packed, result
        physical += lengths
        mass.index_add_(0, codes, weights)
        _finite(old, mass)
        self.connection.executemany(
            "INSERT OR REPLACE INTO low VALUES(?,?,?,?,?)",
            ((key, parent_of[index], _blob(old[index]), int(physical[index]), float(mass[index]))
             for index, key in enumerate(unique)),
        )
        self.peak_groups = max(self.peak_groups, len(unique))
        self.updates += 1

    def batches(self):
        cursor = self.connection.execute("SELECT key,parent,value,rows,mass FROM low ORDER BY parent,key")
        records, encoded_bytes = [], 0
        for record in cursor:
            # A source block limits its encoded labels, but labels fetched
            # from many different old source blocks need a new payload bound.
            size = len(record[0]) + len(record[1])
            if records and (len(records) == self.rows or encoded_bytes + size > MAX_CLUSTER_BYTES):
                yield records, torch.stack([_matrix(item[2], self.width) for item in records])
                records, encoded_bytes = [], 0
            records.append(record)
            encoded_bytes += size
        if records:
            yield records, torch.stack([_matrix(item[2], self.width) for item in records])

    def levels(self, frequency):
        column = "mass" if frequency else "rows"
        low = self.connection.execute(f"SELECT COUNT(*),MIN({column}),SUM({column}),MAX({column}) FROM low").fetchone()
        top = self.connection.execute(f"SELECT COUNT(*),MIN(n),SUM(n),MAX(n) FROM (SELECT SUM({column}) n FROM low GROUP BY parent)").fetchone()
        def summary(row):
            return {"n_groups": row[0], "size_min": float(row[1]), "size_avg": float(row[2]) / row[0], "size_max": float(row[3])}
        return summary(top), summary(low)

    @property
    def diagnostics(self):
        return {"storage": "owned SQLite low-group joint TSQR factors", "group_factor_width": self.width,
                "maximum_group_factor_batch": self.peak_groups, "maximum_packed_factor_rows": self.peak_packed_rows,
                "factor_source_blocks": self.updates, "scratch_bytes": self.path.stat().st_size,
                "complete_group_observations_collected": False}


class _GeneralLikelihood(MixedLikelihood):
    def __init__(self, sample, store, structure, reml, two_level):
        self.sample, self.store, self.structure = sample, store, structure
        self.q, self.p = structure.q, store.width - structure.q - 2
        self.reml, self.two_level, self.nobs = reml, two_level, sample.nobs
        self.size = structure.size + int(two_level) + 1

    def _whiten(self, factors, theta):
        s2 = torch.exp(2 * theta[-1])
        if not bool(torch.isfinite(s2)) or not bool(s2 > 0):
            raise KernelError("numerical_failure", "Mixed residual variance exceeds float64 precision.")
        z = factors[:, :, :self.q]
        u = z @ (self.structure.factor(theta[:self.structure.size]) / s2.sqrt())
        covariance = torch.eye(self.store.width, dtype=torch.float64) + u @ u.transpose(1, 2)
        if not bool(torch.isfinite(covariance).all()):
            raise KernelError("numerical_failure", "Mixed covariance action exceeds float64 precision.")
        chol, code = torch.linalg.cholesky_ex(covariance)
        if bool((code != 0).any()):
            raise KernelError("numerical_failure", "Mixed compressed covariance is not positive definite.")
        white_data = torch.linalg.solve_triangular(chol, factors[:, :, self.q:], upper=False)
        white_z = torch.linalg.solve_triangular(chol, z, upper=False)
        logdet = 2 * torch.log(chol.diagonal(dim1=1, dim2=2)).sum(1)
        return white_data, white_z, logdet

    def _top_factors(self, theta):
        self.store.connection.execute("DELETE FROM top")
        logdet, global_tree = _CompensatedSum(()), _TSQRTree()
        parent, tree, mass = None, None, 0.
        pending, pending_bytes = [], 0
        d = torch.exp(2 * (theta[self.structure.size] - theta[-1])) if self.two_level else torch.zeros((), dtype=torch.float64)
        if not bool(torch.isfinite(d)):
            raise KernelError("numerical_failure", "Mixed top variance exceeds float64 precision.")

        def finish():
            nonlocal pending_bytes
            factor = tree.finish()
            # Every factor has the weighted constant first, hence column zero
            # has a single nonzero entry and the remainder is already orthogonal.
            a = factor[0, 0].square()
            denom = 1 + d * a
            if not bool(torch.isfinite(denom)) or not bool(denom > 0):
                raise KernelError("numerical_failure", "Mixed top covariance exceeds float64 precision.")
            transformed = factor[:, 1:].clone()
            transformed[0] /= denom.sqrt()
            global_tree.add(torch.linalg.qr(transformed, mode="r")[1])
            logdet.add(torch.log(denom))
            pending.append((parent, _blob(factor), mass))
            pending_bytes += len(parent)
            if len(pending) >= 256 or pending_bytes >= 1024**2:
                self.store.connection.executemany("INSERT INTO top VALUES(?,?,?)", pending)
                pending.clear()
                pending_bytes = 0

        for records, factors in self.store.batches():
            white, _, determinants = self._whiten(factors, theta)
            logdet.add(determinants.sum())
            for index, record in enumerate(records):
                if record[1] != parent:
                    if tree is not None:
                        finish()
                    parent, tree, mass = record[1], _TSQRTree(), 0.
                tree.add(torch.linalg.qr(white[index], mode="r")[1])
                mass += record[4]
        if tree is not None:
            finish()
        self.store.connection.executemany("INSERT INTO top VALUES(?,?,?)", pending)
        return global_tree.finish(), logdet.value

    def _top_batch(self, records):
        keys = list(dict.fromkeys(record[1] for record in records))
        found = {}
        for part in _key_parts(keys):
            query = "SELECT key,value,mass FROM top WHERE key IN (" + ",".join("?" for _ in part) + ")"
            for key, value, mass in self.store.connection.execute(query, part):
                found[key] = (_matrix(value, self.p + 2), mass)
        return torch.stack([found[record[1]][0] for record in records])

    def evaluate(self, theta, *, gradient=False, blups=False, sigma2=None, beta=None):
        if blups or sigma2 is not None:
            raise KernelError("unsupported_prediction", "Replay mixed evaluation retains group factors, not all BLUPs; sigma override is not an estimation option.")
        factor, logdet = self._top_factors(theta)
        if beta is None:
            try:
                beta, _, _ = _solve(factor, self.p)
            except AnalysisError as error:
                # An invalid optimizer trial must be rejected by the existing
                # native likelihood wrapper, rather than abort the line search.
                raise KernelError(error.code, str(error)) from error
        residual = factor[:, self.p] - factor[:, :self.p] @ beta
        rss = residual.square().sum()
        s2, pr = torch.exp(2 * theta[-1]), self.p if self.reml else 0
        xvx = factor[:, :self.p].T @ factor[:, :self.p]
        chol, code = torch.linalg.cholesky_ex(xvx)
        if int(code) or not bool(rss > 0):
            raise KernelError("singular_design", "The mixed global GLS design or residual quadratic form is not identified.")
        value = -.5 * ((self.nobs - pr) * (_LOG_2PI + torch.log(s2)) + logdet + rss / s2)
        if self.reml:
            value -= torch.log(chol.diagonal()).sum()
        result = MixedEvaluation(value, beta, xvx, rss, s2)
        if gradient:
            gd, gt = self._derivatives(theta, beta, torch.cholesky_inverse(chol) if self.reml else None)
            delta = self.structure.matrix(theta[:self.structure.size]) / s2
            trace = (gd * delta).sum()
            pieces = [torch.einsum("mab,ab->m", self.structure.jacobian(theta[:self.structure.size]), gd / s2)]
            if self.two_level:
                d = torch.exp(2 * (theta[self.structure.size] - theta[-1]))
                pieces.append((2 * d * gt).reshape(1))
                trace += d * gt
            pieces.append((-(self.nobs - pr) + rss / s2 - 2 * trace).reshape(1))
            result.gradient = torch.cat(pieces)
            _kernel_finite(result.gradient)
        _kernel_finite(result.value, result.beta, result.xvx, result.rvr)
        return result

    def profiled_sigma2(self, theta):
        return self.evaluate(theta).rvr / (self.nobs - (self.p if self.reml else 0))

    def _derivatives(self, theta, beta, cx=None, *, accumulator=None):
        s2 = torch.exp(2 * theta[-1])
        d = torch.exp(2 * (theta[self.structure.size] - theta[-1])) if self.two_level else torch.zeros((), dtype=torch.float64)
        total, top_total = _CompensatedSum((self.q, self.q)), _CompensatedSum(())
        parent, group = None, _CompensatedSum((self.q, self.q))
        top_factor, top_mass = None, None
        def finish():
            a = top_factor[0, 0].square()
            br = top_factor[0, 0] * (top_factor[0, self.p + 1] - top_factor[0, 1:self.p + 1] @ beta)
            bx = top_factor[0, 0] * top_factor[0, 1:self.p + 1]
            denom = 1 + d * a
            gt = -.5 * a / denom + .5 * br.square() / (s2 * denom.square()) if self.two_level else torch.zeros((), dtype=torch.float64)
            if cx is not None and self.two_level:
                gt += .5 * (bx @ cx @ bx) / denom.square()
            top_total.add(gt)
            if accumulator is not None:
                transformed = top_factor[:, 1:].clone()
                transformed[0] /= denom.sqrt()
                r = transformed[:, self.p] - transformed[:, :self.p] @ beta
                score_beta = transformed[:, :self.p].T @ r / s2
                delta = self.structure.matrix(theta[:self.structure.size]) / s2
                trace = (group.value * delta).sum() + d * gt
                pieces = [score_beta, torch.einsum("mab,ab->m", self.structure.jacobian(theta[:self.structure.size]), group.value / s2)]
                if self.two_level:
                    pieces.append((2 * d * gt).reshape(1))
                pieces.append((-top_mass + r.square().sum() / s2 - 2 * trace).reshape(1))
                label = self.store.connection.execute("SELECT cluster FROM parents WHERE key=?", (parent,)).fetchone()[0]
                accumulator.add([label if label is not None else parent], torch.cat(pieces)[None, :])
        for records, factors in self.store.batches():
            white, wz, _ = self._whiten(factors, theta)
            tops = self._top_batch(records)
            a = tops[:, 0, 0].square()
            denom = 1 + d * a
            b = tops[:, 0, 0, None] * tops[:, 0, 1:]
            t1 = wz.transpose(1, 2) @ white[:, :, :1]
            zvu = wz.transpose(1, 2) @ white[:, :, 1:] - (d / denom)[:, None, None] * t1 * b[:, None, :]
            zvz = wz.transpose(1, 2) @ wz - (d / denom)[:, None, None] * t1 @ t1.transpose(1, 2)
            zr = zvu[:, :, self.p] - zvu[:, :, :self.p] @ beta
            gd = -.5 * zvz + zr[:, :, None] * zr[:, None, :] / (2 * s2)
            if cx is not None:
                zx = zvu[:, :, :self.p]
                gd += .5 * (zx @ cx @ zx.transpose(1, 2))
            total.add(gd.sum(0))
            for index, record in enumerate(records):
                if record[1] != parent:
                    if parent is not None:
                        finish()
                    parent, group = record[1], _CompensatedSum((self.q, self.q))
                    top_factor, top_mass = self.store.connection.execute("SELECT value,mass FROM top WHERE key=?", (parent,)).fetchone()
                    top_factor = _matrix(top_factor, self.p + 2)
                group.add(gd[index])
        if parent is not None:
            finish()
        return (total.value + total.value.T) / 2, top_total.value


def _start(like, z2):
    best, best_value = None, -math.inf
    for scale in (.01, .1, 1., 10.):
        parts = [like.structure.start(.25 * scale / z2)]
        if like.two_level:
            parts.append(torch.tensor([.5 * math.log(.25 * scale)], dtype=torch.float64))
        theta = torch.cat([*parts, torch.zeros(1, dtype=torch.float64)])
        try:
            sigma = like.profiled_sigma2(theta)
            m = like.structure.size
            theta[:m] = like.structure.scaled(theta[:m], sigma)
            theta[m:] += .5 * torch.log(sigma)
            value = float(like.value(theta))
        except (KernelError, AnalysisError):
            continue
        if value > best_value:
            best, best_value = theta, value
    if best is None:
        raise AnalysisError("invalid_start", "The global mixed likelihood has no finite starting value.")
    return best


def fit_general(spec, source, *, batch_rows=None):
    notes = _Notes(spec)
    names, random = registry.role_columns(spec, "group"), registry.role_columns(spec, "random")
    intercept_re, reml = bool(notes.option("random_intercept")), notes.option("method") == "reml"
    if not 1 <= len(names) <= 2:
        raise AnalysisError("invalid_spec", "mixed takes one grouping column or two nested ones, top level first.")
    if not random and not intercept_re:
        raise AnalysisError("invalid_spec", "The model has no random effect; keep the intercept or give random slopes.")
    if any(name in {spec.outcome, *spec.predictors, *random} for name in names):
        raise AnalysisError("invalid_spec", "Grouping columns must differ from the outcome, regressors and random slopes.")
    if spec.covariance not in {"nonrobust", "robust", "cluster"} or (reml and spec.covariance != "nonrobust"):
        raise AnalysisError("unsupported_covariance", "Mixed supports model-based ML/REML and robust/cluster ML covariance only.")
    if spec.weights and spec.weight_type != "fweight":
        raise AnalysisError("unsupported_weights", "Mixed supports frequency weights only.")
    clusters = registry.cluster_columns(spec) if spec.covariance == "cluster" else []
    if len(clusters) > 1:
        raise AnalysisError("unsupported_cluster_dimensions", "Mixed requires one covariance cluster column containing top-level groups.")
    two_level, z_names = len(names) == 2, [*random, *(["_cons"] if intercept_re else [])]
    kind, q = notes.option("covstructure"), len(z_names)
    if kind not in {"identity", "independent", "exchangeable", "unstructured"}:
        raise AnalysisError("invalid_option", f"Unknown covariance structure '{kind}'.")
    # CovStructure itself creates O(q²) parameter/entry metadata, even for
    # diagonal structures. Refuse its minimum live geometry before that
    # constructor or any source pass, then refine with the encoded fixed width.
    # Global rank screening can omit fixed columns, so their raw count is
    # not a lower bound. The final encoded width is budgeted after prepare.
    min_p = 1
    min_m = 1 if q == 1 or kind == "identity" else q if kind == "independent" else 2 if kind == "exchangeable" else q * (q + 1) // 2
    min_h, min_k = q + min_p + 2, min_p + min_m + int(two_level) + 1
    plan_workspace("general mixed minimum random-structure geometry", {
        "random_structure_metadata": 128 * q**2,
        "minimum_joint_group_factor_workspace": 1536 * min_h**2,
        "minimum_optimizer_joint_information": 1024 * min_k**2,
    }, budget_bytes=min(WORKING_BYTES, workspace_budget_bytes()))
    structure = CovStructure(kind, q)
    sample = ReplaySample(spec, source, batch_rows=batch_rows)
    design = sample.add_design("mean")
    sample.prepare()
    p, q, m = len(design.terms), len(z_names), structure.size
    variance_size = m + int(two_level) + 1
    if not p or sample.nobs <= p + variance_size:
        raise AnalysisError("insufficient_observations", "Mixed needs an identified fixed design and more observations than parameters.")
    h, k = q + p + 2, p + variance_size
    resource = sample.plan_rows("general Gaussian mixed bounded group QR and variance information", {
        "factor_and_parent_SQLite_cache": 2 * 1024**2,
        "joint_score_SQLite_cache": 6 * 1024**2 if spec.covariance != "nonrobust" else 0,
        "global_TSQR_and_random_rank_factors": 1024 * h**2,
        "optimizer_Hessian_joint_covariance_and_transforms": 1024 * k**2,
        "random_and_outcome_moment_state": 1024 * (q + 2),
        "random_structure_metadata": 128 * q**2,
        "bounded_encoded_group_labels": MAX_CLUSTER_BYTES * (6 + 2 * int(two_level) + int(bool(clusters))),
    }, 512 * h**2 + 256 * (q**2 + q * (p + 1)))
    y_moments = _WeightedMoments(2, intercept=True)
    z_moments = _WeightedMoments(len(random) + 1, intercept=True)
    for batch in sample.batches():
        y = batch.numeric(spec.outcome)
        y_moments.add(torch.stack((torch.ones_like(y), y), 1), batch.weights, False)
        z = torch.stack([torch.ones_like(y), *(batch.numeric(name) for name in random)], 1)
        z_moments.add(z, batch.weights, False)
    variation = float(y_moments.m2.value[1] / y_moments.mass)
    if variation <= 0:
        raise AnalysisError("constant_outcome", "The mixed outcome does not vary.")
    scale = float(y_moments.magnitude[1]) * math.sqrt(variation)
    center = float(y_moments.anchor[1] + y_moments.magnitude[1] * y_moments.mean[1]) if spec.intercept else 0.
    means = z_moments.anchor + z_moments.magnitude * z_moments.mean
    std = z_moments.magnitude * (z_moments.m2.value / z_moments.mass).clamp_min(0).sqrt()
    rms = torch.hypot(means, std)
    zscale = torch.cat((rms[1:], torch.ones(1, dtype=torch.float64))) if intercept_re else rms[1:]
    if structure.kind in {"identity", "exchangeable"}:
        zscale = torch.full_like(zscale, float(zscale.max()))
    safe_scales = torch.where(zscale > 0, zscale, torch.ones_like(zscale))
    _finite(torch.tensor([scale, center], dtype=torch.float64), means, std, safe_scales)
    if scale <= 0:
        raise AnalysisError("mixed_precision", "Mixed outcome scaling is below float64 precision; rescale the outcome.")
    rank_tree, pooled_tree = _TSQRTree(), _TSQRTree()
    z2 = _CompensatedSum((q,))
    with ExitStack() as stack:
        store = _Factors(h, sample.rows)
        stack.callback(store.close)
        for batch in sample.batches():
            y = (batch.numeric(spec.outcome) - center) / scale
            raw_slopes = torch.stack([batch.numeric(name) for name in random], 1) if random else torch.empty((len(y), 0), dtype=torch.float64)
            raw_z = torch.cat((raw_slopes, torch.ones((len(y), 1), dtype=torch.float64)), 1) if intercept_re else raw_slopes
            z = raw_z / safe_scales
            z2.add((z.square() * batch.weights[:, None]).sum(0))
            if random:
                rank = (raw_slopes - means[1:]) / torch.where(std[1:] > 0, std[1:], torch.ones_like(std[1:])) if intercept_re else z
                rank_tree.add(torch.linalg.qr(rank * batch.weights.sqrt()[:, None], mode="r")[1])
            data = torch.cat((z, torch.ones((len(y), 1), dtype=torch.float64), batch.designs["mean"], y[:, None]), 1)
            _finite(data)
            parent = encode_cluster_labels(batch.frame[names[0]])
            if two_level:
                child = encode_cluster_labels(batch.frame[names[1]])
                keys = [b"P" + len(top).to_bytes(8, "big") + top + low for top, low in zip(parent, child, strict=True)]
            else:
                keys = parent
            cluster = encode_cluster_labels(batch.frame[clusters[0]]) if clusters else []
            store.add(keys, parent, cluster, data, batch.weights)
            pooled_tree.add(torch.linalg.qr(data[:, q + 1:] * batch.weights.sqrt()[:, None], mode="r")[1])
        if random:
            _, omitted = collinear_columns(rank_tree.finish())
            if omitted:
                raise AnalysisError("collinear_random_effects", "mixed: random effects are collinear; remove " + ", ".join(random[index] for index in omitted) + ".")
        z2_value = z2.value / sample.nobs
        _finite(z2_value)
        if bool((z2_value <= 0).any()):
            raise AnalysisError("collinear_random_effects", "A mixed random-effects column has zero norm.")
        top_level, low_level = store.levels(spec.weight_type == "fweight")
        if min(top_level["n_groups"], low_level["n_groups"]) < 2:
            raise AnalysisError("insufficient_groups", "Each mixed grouping level requires at least two groups.")
        if low_level["size_max"] <= 1:
            raise AnalysisError("insufficient_group_size", "Every lowest-level group has one observation; random and residual variances cannot be separated.")
        pooled = pooled_tree.finish()
        linear_beta, _, _ = _solve(pooled, p)
        linear_ssr = float((pooled[:, p] - pooled[:, :p] @ linear_beta).square().sum())
        if linear_ssr <= 1e-24 * sample.nobs:
            raise AnalysisError("perfect_fit", "The fixed regressors explain the mixed outcome exactly.")
        pr = p if reml else 0
        correction = -(sample.nobs - pr) * math.log(scale)
        if reml:
            correction += float(torch.linalg.slogdet(design.transform)[1])
        linear_ll = -(sample.nobs - pr) / 2 * (_LOG_2PI + math.log(linear_ssr / (sample.nobs - pr)) + 1)
        if reml:
            linear_ll -= float(torch.log(torch.linalg.cholesky(pooled[:, :p].T @ pooled[:, :p]).diagonal()).sum())
        linear_ll += correction
        like = _GeneralLikelihood(sample, store, structure, reml, two_level)
        result = optimize.maximize_bfgs(like, _start(like, z2_value),
            hessian_fn=lambda theta: optimize.numerical_hessian(like.gradient, theta), raise_on_failure=False)
        theta = result.theta
        g_work, s2 = structure.matrix(theta[:m]), torch.exp(2 * theta[-1])
        if bool((g_work.diagonal() * z2_value / s2 < _BOUNDARY).any()) or (two_level and float(torch.exp(2 * (theta[m] - theta[-1]))) < _BOUNDARY):
            raise AnalysisError("boundary_solution", "A mixed random variance is estimated at zero; remove that random effect.")
        if structure.kind in {"unstructured", "exchangeable"} and q > 1:
            sd = g_work.diagonal().sqrt()
            if float(torch.linalg.eigvalsh(g_work / sd[:, None] / sd).min()) < _SINGULAR_CORRELATION:
                raise AnalysisError("boundary_solution", "Mixed random effects are perfectly correlated; simplify the covariance structure.")
        if not result.converged:
            raise AnalysisError("nonconvergence", "The global mixed likelihood did not converge: " + str(result.diagnostics.get("message")))
        final = like.evaluate(theta)
        vtheta = optimize.information_inverse(-result.hessian)
        setup = SimpleNamespace(structure=structure, groups=names, z_names=z_names, two_level=two_level)
        reported, jacobian = _reported(setup, theta)
        log_y, log_z = math.log(scale), torch.log(safe_scales)
        row_logs = [2 * log_y] if two_level else []
        row_logs.extend(2 * log_y - float(log_z[a]) - float(log_z[b]) for a, b in structure.entries)
        row_logs.append(2 * log_y)
        row_scales = _coordinate_scales(row_logs)
        reported *= row_scales
        jacobian *= row_scales[:, None]
        transform = torch.zeros((p + len(reported), p + variance_size), dtype=torch.float64)
        transform[:p, :p], transform[p:, p:] = scale * design.transform, jacobian
        joint = torch.block_diag(optimize.information_inverse(final.xvx / final.sigma2), vtheta)
        if spec.covariance != "nonrobust":
            accumulator = ClusterAccumulator(p + variance_size, scratch_directory=_scratch_directory())
            stack.callback(accumulator.close)
            like._derivatives(theta, final.beta, accumulator=accumulator)
            meat, groups = accumulator.finish()
            joint = joint @ meat @ joint * groups / (groups - 1)
        covariance = transform @ joint @ transform.T
        beta = scale * design.transform @ final.beta
        if spec.intercept:
            beta[0] += center
        params = torch.cat((beta, reported))
        _finite(params, covariance)
        terms, equations, variances = _variance_terms(setup)
        if bool((reported[torch.tensor(variances, dtype=torch.bool)] <= 0).any()):
            raise AnalysisError("mixed_precision", "An original-unit mixed variance is below float64 precision; rescale the variables.")
        g_original = g_work * _coordinate_scales(2 * log_y - log_z[:, None] - log_z)
        original_theta = theta.clone()
        theta_transform = torch.eye(variance_size, dtype=torch.float64)
        if structure.kind == "unstructured":
            original_theta[:q] += math.log(scale) - torch.log(safe_scales)
            for index, (a, _) in enumerate(structure.lower, start=q):
                multiplier = float(_coordinate_scales(log_y - log_z[a]))
                original_theta[index] *= multiplier
                theta_transform[index, index] = multiplier
        elif structure.kind == "exchangeable":
            original_theta[0] += math.log(scale) - math.log(float(safe_scales[0]))
        elif structure.kind == "identity":
            original_theta[0] += math.log(scale) - math.log(float(safe_scales[0]))
        else:
            original_theta[:m] += math.log(scale) - torch.log(safe_scales)
        original_theta[m:] += math.log(scale)
        theta_se = (theta_transform @ vtheta @ theta_transform.T).diagonal().sqrt()
        _finite(g_original, original_theta, theta_se)
        icc = {}
        if not random and intercept_re:
            denominator = float(reported.sum())
            if two_level:
                icc = {names[0]: float(reported[0]) / denominator,
                       f"{names[1]}|{names[0]}": float(reported[0] + reported[1]) / denominator}
            else:
                icc = {names[0]: float(reported[0]) / denominator}
        ll = float(final.value) + correction
        criteria = information_criteria(ll, p + variance_size, sample.nobs)
        metrics = {"log_likelihood": ll, "aic": criteria["aic"], "bic": criteria["bic"],
                   "n_groups": top_level["n_groups"], "group_size_min": top_level["size_min"],
                   "group_size_avg": top_level["size_avg"], "group_size_max": top_level["size_max"]}
        if icc:
            metrics["icc"] = next(iter(icc.values()))
        tests = {"model": wald_test(beta, covariance[:p, :p], [i for i, term in enumerate(design.terms) if term != "Intercept"], label="Wald chi2 test of the fixed slopes")}
        if spec.covariance == "nonrobust":
            tests["lr_vs_linear"] = common.boundary_lr(2 * (ll - linear_ll), variance_size - 1, "linear model")
        levels = [top_level, low_level] if two_level else [top_level]
        extra = {"method": "reml" if reml else "ml", "likelihood": "restricted (REML)" if reml else "full (ML)",
                 "covstructure": structure.kind, "random_effects": {"group": names[-1], "effects": z_names},
                 "levels": [{"group": name, **level} for name, level in zip(names, levels, strict=True)],
                 "G": g_original.tolist(), "residual_variance": float(reported[-1]), "icc": icc,
                 "theta": original_theta.tolist(), "theta_std_error": theta_se.tolist(),
                 "theta_parameterization": "lowest-level covariance structure parameters (log sd / log-Cholesky), then ln sd of the top-level intercept, then ln sigma_e",
                 "linear_log_likelihood": linear_ll}
        if spec.covariance != "nonrobust":
            extra["notes"] = ["theta_std_error is model-based; the reported covariance is robust.", "No LR test vs. the linear model under robust/cluster covariance."]
        info = {"covariance": spec.covariance, "correction": "joint fixed/variance sandwich of top-level group scores with G/(G-1)" if spec.covariance != "nonrobust" else "model-based GLS and profiled variance information, block diagonal; delta method",
                "df_inference": None, "df_resid": None}
        if spec.covariance != "nonrobust":
            info.update({"n_clusters": groups, "cluster_column": clusters[0] if clusters else names[0]})
        predictions = []
        for batch in sample.batches():
            take = min(400 - len(predictions), len(batch.positions))
            observed = batch.numeric(spec.outcome)
            fitted = center + scale * (batch.designs["mean"] @ final.beta)
            for pos, y, fit in zip(batch.positions[:take].tolist(), observed[:take].tolist(), fitted[:take].tolist(), strict=True):
                predictions.append({"row": pos, "observed": y, "fitted": fit, "residual": y - fit})
        bundle = _result(sample, terms=[*design.terms, *terms], beta=params, covariance=covariance,
            info=info, metrics=metrics, notes=notes, predictions=predictions, tests=tests,
            solver="native_general_mixed_disk_group_TSQR", diagnostics={"converged": True, "iterations": result.iterations, "group_factors": store.diagnostics},
            extra=extra, resource=resource.record(), title=f"Mixed-effects {'REML' if reml else 'ML'} regression", use_t=False,
            provenance_extra={"gradient": "analytic global profiled fixed effects", "hessian": "numerical Ridders derivative of analytic gradient", "likelihood_evaluation": "low-group joint TSQR and positive top-group whitened QR; no observation covariance matrices"})
        for coefficient, equation in zip(bundle.coefficients, [spec.outcome] * p + equations, strict=True):
            coefficient.equation = equation
        common.variance_intervals(bundle, [p + i for i, flag in enumerate(variances) if flag], spec.alpha)
        if not all(math.isfinite(coefficient.ci_low) and math.isfinite(coefficient.ci_high) for coefficient in bundle.coefficients):
            raise AnalysisError("mixed_precision", "An original-unit mixed confidence interval exceeds float64 precision; rescale the variables.")
        return bundle
