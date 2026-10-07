"""Exact disk-backed matching and Abadie-Imbens inference.

One-dimensional searches use SQLite's coordinate index. Multidimensional
searches replay bounded candidate tiles twice (rank, then ties), retaining
the native work guard. Matched sets, usage and imputed outcomes live on disk.
"""
from __future__ import annotations

from array import array
import math
from pathlib import Path
import sqlite3
import tempfile

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.engines.execution import qr_factor
from openecon.engines.linalg import least_squares
from openecon.engines.optimize import information_inverse
from openecon.engines.streaming_ols import _CompensatedSum, _TSQRTree
from .core import ModelFrame, build_result
from .teffects.common import Treatment, coefficient_table
from .teffects.neighbors import MAX_PAIRS


def _encode(value):
    return array('d', value.tolist()).tobytes()


def _decode(value):
    return torch.frombuffer(bytearray(value), dtype=torch.float64)


def _propensity_rows(model, x, codes, gamma):
    """Canonical row reduction/SIMD shape preserves duplicate-score ties.

    BLAS and special-function tail kernels can round identical observations
    differently when raw reader batch sizes differ. Always evaluate the same
    padded 256-row shape; padding contributes no likelihood/information.
    """
    from .teffects.index import mean_pieces, residual_pieces
    for begin in range(0, len(x), 256):
        stop = min(begin+256, len(x))
        padded = torch.zeros((256, x.shape[1]), dtype=torch.float64)
        padded[:stop-begin] = x[begin:stop]
        y = torch.zeros(256, dtype=torch.float64)
        y[:stop-begin] = codes[begin:stop]
        eta = (padded*gamma).sum(dim=1)
        p, f = mean_pieces(model, eta)
        _, curvature = residual_pieces(model, y, eta)
        yield begin, stop, p[:stop-begin], f[:stop-begin], curvature[:stop-begin]


class MatchingStore:
    block_rows = 512

    def __init__(self, width, value_width, budget):
        from openecon.resources import plan_workspace
        self.plan = plan_workspace('disk-backed exact matching', {
            'candidate_tiles': 64*self.block_rows*(width+value_width+8),
            'small_factors': 128*(width+value_width+1)**2,
            'sqlite_cache': 2*1024**2,
        }, budget_bytes=budget)
        self.width, self.value_width = width, value_width
        self.scratch = self.db = None
        try:
            self.scratch = tempfile.TemporaryDirectory(prefix='openecon-matching-')
            self.path = Path(self.scratch.name)/'matching.sqlite3'
            self.db = sqlite3.connect(self.path)
            for pragma in ('journal_mode=OFF', 'synchronous=OFF', 'cache_size=-2048', 'mmap_size=0', 'temp_store=FILE'):
                self.db.execute('PRAGMA '+pragma)
            self.db.execute('CREATE TABLE rows(id INTEGER PRIMARY KEY, level INTEGER, c0 REAL, coord BLOB, value BLOB, p REAL, f REAL, usage REAL DEFAULT 0, usage2 REAL DEFAULT 0, count INTEGER DEFAULT 0, imputed REAL, sigma REAL, effect REAL)')
            self.db.execute('CREATE INDEX coordinate_search ON rows(level,c0,id)')
            self.db.execute('CREATE TABLE prefixes(level INTEGER, c0 REAL, endrank INTEGER, prefix BLOB, PRIMARY KEY(level,c0)) WITHOUT ROWID')
            self.db.execute('CREATE TABLE usage_events(level INTEGER, position INTEGER, amount REAL, square REAL, cover INTEGER)')
            self.path.chmod(0o600)
        except (OSError, sqlite3.Error) as exc:
            self.close()
            raise AnalysisError('matching_spill_failed', 'Matching needs writable temporary storage and enough free disk space.') from exc
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None

    def rows(self, level=None):
        sql = 'SELECT id,level,coord,value,p,f,usage,usage2,count,imputed,sigma,effect FROM rows'
        cursor = self.db.execute(sql+(' WHERE level=?' if level is not None else '')+' ORDER BY id', () if level is None else (level,))
        try:
            while records := cursor.fetchmany(self.block_rows):
                yield records
        finally:
            cursor.close()

    def _pool(self, level, query, coordinate, distance=None):
        sql = 'SELECT id,coord,value FROM rows WHERE level=? AND id!=?'
        params = [level, query]
        if distance is not None and self.width == 1:
            # Enlarge the SQL interval, then compare the actual float64
            # differences. This preserves native subtraction-based ties.
            low = math.nextafter(float(coordinate[0])-distance, -math.inf)
            high = math.nextafter(float(coordinate[0])+distance, math.inf)
            sql += ' AND c0 BETWEEN ? AND ?'
            params += [low, high]
        cursor = self.db.execute(sql+' ORDER BY id', params)
        try:
            while records := cursor.fetchmany(self.block_rows):
                yield records
        finally:
            cursor.close()

    def _distances(self, coordinate, records):
        pool = torch.stack([_decode(row[1]) for row in records])
        if self.width == 1:
            return (pool[:, 0]-coordinate[0]).abs()
        return torch.cdist(coordinate[None], pool, compute_mode='donot_use_mm_for_euclid_dist')[0]

    def prepare_index(self):
        """Disk prefix sums and range usage updates keep even huge ties linear."""
        if self.width != 1:
            return
        for level in (0, 1):
            cursor = self.db.execute('SELECT c0,value FROM rows WHERE level=? ORDER BY c0,id', (level,))
            total = _CompensatedSum((2*self.value_width-1,))
            group = _CompensatedSum((2*self.value_width-1,))
            previous, count = None, 0
            while records := cursor.fetchmany(self.block_rows):
                for coordinate, encoded in records:
                    if previous is not None and coordinate != previous:
                        total.add(group.value)
                        self.db.execute('INSERT INTO prefixes VALUES(?,?,?,?)', (level, previous, count, _encode(total.value)))
                        group = _CompensatedSum((2*self.value_width-1,))
                    values = _decode(encoded)
                    group.add(torch.cat((values, values[1:]*values[0])))
                    count += 1
                    previous = coordinate
            if previous is not None:
                total.add(group.value)
                self.db.execute('INSERT INTO prefixes VALUES(?,?,?,?)', (level, previous, count, _encode(total.value)))
            cursor.close()

    def _indexed_window(self, query, coordinate, level, reach):
        x = float(coordinate[0])
        low, high = math.nextafter(x-reach, -math.inf), math.nextafter(x+reach, math.inf)
        bounds = self.db.execute('SELECT MIN(c0),MAX(c0) FROM prefixes WHERE level=? AND c0 BETWEEN ? AND ?', (level, low, high)).fetchone()
        low, high = bounds
        # SQL bounds are deliberately loose; restore exactly the native
        # subtraction-based comparison at both endpoints before prefix lookup.
        while abs(low-x) > reach:
            low = self.db.execute('SELECT MIN(c0) FROM prefixes WHERE level=? AND c0>?', (level, low)).fetchone()[0]
        while abs(high-x) > reach:
            high = self.db.execute('SELECT MAX(c0) FROM prefixes WHERE level=? AND c0<?', (level, high)).fetchone()[0]
        stop, last = self.db.execute('SELECT endrank,prefix FROM prefixes WHERE level=? AND c0=?', (level, high)).fetchone()
        before = self.db.execute('SELECT endrank,prefix FROM prefixes WHERE level=? AND c0<? ORDER BY c0 DESC LIMIT 1', (level, low)).fetchone()
        start = before[0] if before else 0
        sums = _decode(last)-(_decode(before[1]) if before else 0.)
        count = stop-start
        own = self.db.execute('SELECT level,value FROM rows WHERE id=?', (query,)).fetchone()
        if own[0] == level:
            values = _decode(own[1])
            sums -= torch.cat((values, values[1:]*values[0]))
            count -= 1
        return count, sums, start, stop

    def finish_usage(self):
        if self.width != 1:
            return
        self.db.execute('CREATE INDEX usage_event_order ON usage_events(level,position)')
        for level in (0, 1):
            events = iter(self.db.execute('SELECT position,SUM(amount),SUM(square),SUM(cover) FROM usage_events WHERE level=? GROUP BY position ORDER BY position', (level,)))
            current = next(events, None)
            total, squared = _CompensatedSum(()), _CompensatedSum(())
            cover, start = 0, 0
            cursor = self.db.execute('SELECT c0,endrank FROM prefixes WHERE level=? ORDER BY c0', (level,))
            while groups := cursor.fetchmany(self.block_rows):
                updates = []
                for coordinate, stop in groups:
                    while current is not None and current[0] <= start:
                        total.add(torch.tensor(current[1], dtype=torch.float64))
                        squared.add(torch.tensor(current[2], dtype=torch.float64))
                        cover += current[3]
                        current = next(events, None)
                    updates.append((float(total.value) if cover else 0., float(squared.value) if cover else 0., level, coordinate))
                    start = stop
                self.db.executemany('UPDATE rows SET usage=?,usage2=? WHERE level=? AND c0=?', updates)
            cursor.close()

    def kth(self, query, coordinate, level, neighbours):
        if self.width == 1:
            below = self.db.execute('SELECT id,coord,value FROM rows WHERE level=? AND id!=? AND c0<=? ORDER BY c0 DESC,id LIMIT ?', (level, query, float(coordinate[0]), neighbours)).fetchall()
            above = self.db.execute('SELECT id,coord,value FROM rows WHERE level=? AND id!=? AND c0>? ORDER BY c0,id LIMIT ?', (level, query, float(coordinate[0]), neighbours)).fetchall()
            records = below+above
            if len(records) < neighbours:
                raise AnalysisError('insufficient_observations', 'The requested matching neighbourhood has too few candidate observations.')
            return float(torch.kthvalue(self._distances(coordinate, records), neighbours).values)
        nearest = torch.empty(0, dtype=torch.float64)
        for records in self._pool(level, query, coordinate):
            distances = self._distances(coordinate, records)
            combined = torch.cat((nearest, distances))
            nearest = torch.topk(combined, min(neighbours, len(combined)), largest=False).values
        if len(nearest) < neighbours:
            raise AnalysisError('insufficient_observations', 'The requested matching neighbourhood has too few candidate observations.')
        return float(nearest.max())

    def match(self, query, coordinate, level, neighbours, *, usage=False, covariance=False):
        reach = self.kth(query, coordinate, level, neighbours)
        if self.width == 1:
            count, sums, start, stop = self._indexed_window(query, coordinate, level, reach)
            if usage:
                share = 1/count
                self.db.executemany('INSERT INTO usage_events VALUES(?,?,?,?,?)',
                                    [(level, start, share, share**2, 1), (level, stop, -share, -share**2, -1)])
            return count, sums if covariance else sums[:self.value_width], reach
        total = _CompensatedSum((self.value_width if not covariance else self.value_width*2-1,))
        count = 0
        for records in self._pool(level, query, coordinate, reach):
            keep = self._distances(coordinate, records) <= reach
            values = torch.stack([_decode(row[2]) for row in records])[keep]
            if covariance:
                values = torch.cat((values, values[:, 1:]*values[:, :1]), dim=1)
            total.add(values.sum(dim=0))
            count += int(keep.sum())
        if usage:
            reciprocal = 1/count
            for records in self._pool(level, query, coordinate, reach):
                keep = self._distances(coordinate, records) <= reach
                self.db.executemany('UPDATE rows SET usage=usage+?,usage2=usage2+? WHERE id=?',
                                    ((reciprocal, reciprocal**2, row[0]) for row, chosen in zip(records, keep.tolist(), strict=True) if chosen))
        return count, total.value, reach


def fit_matching(sample, nuisance):
    from .streaming_teffects import _codes, _option, _role
    spec = sample.spec
    method, estimand = _option(spec, 'method'), _option(spec, 'estimand')
    matching = sample.designs.get('matching')
    bias = sample.designs.get('bias')
    treatment_design = sample.designs.get('treatment')
    kb = 0 if bias is None else len(bias.terms)
    kt = 0 if treatment_design is None else len(treatment_design.terms)
    width = len(matching.terms)-1 if matching is not None else 1
    neighbours = int(_option(spec, 'neighbors'))
    variance_neighbours = int(_option(spec, 'vce_neighbors'))
    sample.plan_rows('exact matching source projection', {'matching_candidate_tiles': 64*512*(width+kb+kt+8),
                     'matching_factors': 128*(width+kb+kt+1)**2, 'matching_sqlite_cache': 2*1024**2,
                     'ranked_candidates': 2*(512+32*(width+kb+kt+1))*max(neighbours, variance_neighbours)}, 64*(width+kb+kt+8))
    counts = sample.notes['treatment_counts']
    if min(counts) <= variance_neighbours or min(counts) < neighbours:
        raise AnalysisError('insufficient_observations', 'Treatment groups are too small for the requested matching/variance neighbourhoods.')
    if width > 1:
        comparisons = max(counts[0]*counts[1], max(k*(k-1) for k in counts))
        if comparisons > MAX_PAIRS:
            raise AnalysisError('matching_too_large', 'Exact covariate matching exceeds the native distance-work budget; use propensity-score matching or a smaller sample.')
    cross = _CompensatedSum((width, width))
    response = _CompensatedSum(())
    info = _CompensatedSum((kt, kt))
    for batch in sample.batches():
        response.add(batch.numeric(spec.outcome).sum())
        if matching is not None:
            cross.add(batch.designs['matching'][:, 1:].T@batch.designs['matching'][:, 1:])
    ybar = float(response.value/sample.nrows)
    metric = _option(spec, 'metric')
    whitener = None
    if matching is not None and metric != 'euclidean':
        scales = matching.scales[matching.kept][1:]
        covariance = cross.value*scales[:, None]*scales[None, :]/(sample.nrows-1)
        whitener = covariance.diagonal().sqrt() if metric == 'ivariance' else torch.linalg.cholesky(covariance)
    store = MatchingStore(width, 1+kb+kt, sample.working_bytes)
    try:
        for batch in sample.batches():
            codes = _codes(sample, batch)
            y = batch.numeric(spec.outcome)-ybar
            xb = batch.designs['bias'] if bias is not None else torch.empty((len(y), 0), dtype=torch.float64)
            xt = batch.designs['treatment'] if treatment_design is not None else torch.empty((len(y), 0), dtype=torch.float64)
            if matching is not None:
                # Recover native raw coordinates; mean shifts can change the
                # rounding of exact ties, so no arbitrary rotation is used.
                raw = sample._encode(matching, batch.frame)[:, matching.kept][:, 1:]
                coords = raw
                if whitener is not None:
                    coords = raw/whitener if metric == 'ivariance' else torch.linalg.solve_triangular(whitener, raw.T, upper=False).T
                p = f = torch.zeros(len(y), dtype=torch.float64)
            else:
                p, f = torch.empty(len(y), dtype=torch.float64), torch.empty(len(y), dtype=torch.float64)
                for begin, stop, probabilities, slopes, curvature in _propensity_rows(_option(spec, 'tmodel'), xt, codes, nuisance.treatment[0]):
                    p[begin:stop], f[begin:stop] = probabilities, slopes
                    info.add((xt[begin:stop]*(-curvature)[:, None]).T@xt[begin:stop])
                if bool((torch.minimum(p, 1-p) < _option(spec, 'pstolerance')).any()):
                    raise AnalysisError('overlap_violation', 'A canonical fitted propensity lies below pstolerance.')
                coords = p[:, None]
            values = torch.cat((y[:, None], xb, xt), dim=1)
            store.db.executemany('INSERT INTO rows(id,level,c0,coord,value,p,f) VALUES(?,?,?,?,?,?,?)',
                                ((int(position), int(level), float(coord[0]), _encode(coord), _encode(value), float(probability), float(slope))
                                 for position, level, coord, value, probability, slope in zip(batch.positions, codes, coords, values, p, f, strict=True)))
        store.db.commit()
        store.prepare_index()
        directions = [1] if estimand == 'atet' and method == 'nnmatch' else [1, 0]
        size_min, size_max, size_sum, size_count, ties = math.inf, 0, 0, 0, 0
        for level in directions:
            for records in store.rows(level):
                updates = []
                for record in records:
                    size, sums, distance = store.match(record[0], _decode(record[2]), 1-level, neighbours, usage=True)
                    caliper = _option(spec, 'caliper')
                    if caliper is not None and distance > caliper:
                        raise AnalysisError('caliper_violation', 'An observation has fewer than the requested matches within its caliper.')
                    size_min, size_max = min(size_min, size), max(size_max, size)
                    size_sum, size_count, ties = size_sum+size, size_count+1, ties+int(size > neighbours)
                    updates.append((size, float(sums[0]/size+ybar), record[0]))
                store.db.executemany('UPDATE rows SET count=?,imputed=? WHERE id=?', updates)
        store.finish_usage()
        bias_betas, bias_tables = {}, {}
        if bias is not None:
            for level in (1-k for k in directions):
                tree, used = _TSQRTree(), 0
                for records in store.rows(level):
                    values = torch.stack([_decode(row[3]) for row in records])
                    weights = torch.tensor([row[6] for row in records], dtype=torch.float64)
                    chosen = weights > 0
                    used += int(chosen.sum())
                    if bool(chosen.any()):
                        augmented = torch.cat((values[:, 1:1+kb], (values[:, 0]+ybar)[:, None]), 1)
                        tree.add(qr_factor(augmented[chosen]*weights[chosen, None].sqrt()))
                if used <= kb:
                    raise AnalysisError('insufficient_observations', 'A bias-adjustment regression has too few matched observations.')
                factor = tree.finish()
                fitted = least_squares(factor[:, :kb], factor[:, kb], drop_collinear=True)
                beta = torch.zeros(kb, dtype=torch.float64)
                beta[fitted.kept] = fitted.beta
                bias_betas[level] = beta
                converted = beta/bias.scales[bias.kept]
                bias_tables[level] = dict(zip(bias.terms, converted.tolist(), strict=True))
        effect_sum, effect_count = _CompensatedSum(()), 0
        for records in store.rows():
            updates = []
            for record in records:
                identity, level, coord = record[:3]
                values = _decode(record[3])
                size, sums, _ = store.match(identity, _decode(coord), level, variance_neighbours)
                sigma = size/(size+1)*float(values[0]-sums[0]/size)**2
                imputed = record[9]
                if imputed is not None and bias is not None:
                    _, matched, _ = store.match(identity, _decode(coord), 1-level, neighbours)
                    imputed += float((values[1:1+kb]-matched[1:1+kb]/record[8])@bias_betas[1-level])
                effect = None if imputed is None else (float(values[0]+ybar)-imputed)*(2*level-1)
                if effect is not None and (estimand == 'ate' or level == 1):
                    effect_sum.add(torch.tensor(effect, dtype=torch.float64))
                    effect_count += 1
                updates.append((sigma, effect, identity))
            store.db.executemany('UPDATE rows SET sigma=?,effect=? WHERE id=?', updates)
        tau = float(effect_sum.value/effect_count)
        variance_sum = _CompensatedSum(())
        c, d = _CompensatedSum((kt,)), _CompensatedSum((kt,))
        for records in store.rows():
            for record in records:
                identity, level, coord = record[:3]
                values = _decode(record[3])
                usage, usage2, sigma, effect = record[6], record[7], record[10], record[11]
                term = ((effect-tau)**2 if estimand == 'ate' or level == 1 else 0.)
                term += (usage**2+(2*usage if estimand == 'ate' else 0.)-usage2)*sigma if estimand == 'ate' or level == 0 else 0.
                variance_sum.add(torch.tensor(term, dtype=torch.float64))
                if treatment_design is not None:
                    covariance_rows = []
                    for group in (0, 1):
                        size, sums, _ = store.match(identity, _decode(coord), group, variance_neighbours, covariance=True)
                        # value = [y, bias, treatment]; final suffix contains
                        # all non-y columns multiplied by centred y.
                        begin = 1+kb
                        sx = sums[begin:begin+kt]
                        sxy = sums[store.value_width+kb:store.value_width+kb+kt]
                        covariance_rows.append((sxy-sx*sums[0]/size)/(size-1))
                    p, f = record[4], record[5]
                    vector = covariance_rows[1]/p+covariance_rows[0]/(1-p) if estimand == 'ate' else covariance_rows[1]+p/(1-p)*covariance_rows[0]
                    c.add(f*vector)
                    if estimand == 'atet':
                        d.add(f*(effect-tau)*values[1+kb:])
        ai_variance = float(variance_sum.value/effect_count**2)
        adjustment = 0.
        vg = information_inverse(info.value) if kt else None
        if kt:
            cv, dv = c.value/effect_count, d.value/effect_count
            adjustment = float(cv@vg@cv-(dv@vg@dv if estimand == 'atet' else 0.))
        variance = ai_variance-adjustment
        if not math.isfinite(variance) or variance <= 0:
            raise AnalysisError('invalid_covariance', 'The estimated matching variance is not positive after the native propensity-score correction.')
        treatment = Treatment(_role(spec, 'treatment')[0], sample.notes['treatment_labels'], torch.empty(0, dtype=torch.int64), counts)
        usage_max, usage_mean, usage_count = store.db.execute('SELECT MAX(usage),AVG(usage),COUNT(*) FROM rows WHERE usage>0').fetchone()
        extra = {'method': method, 'estimand': estimand, 'treatment': treatment.column, 'control': treatment.text(0),
                 'levels': [treatment.text(i) for i in (0, 1)], 'neighbors': neighbours, 'vce_neighbors': variance_neighbours,
                 'metric': metric if matching is not None else 'propensity score',
                 'matching_variables': matching.terms[1:] if matching is not None else ['propensity score'],
                 'matches': {'min': size_min, 'max': size_max, 'mean': size_sum/size_count, 'units_with_ties': ties},
                 'usage': {'max': usage_max, 'mean_used': usage_mean, 'units_used': usage_count},
                 'observations_by_level': {treatment.text(i): counts[i] for i in (0, 1)},
                 'variance': {'abadie_imbens': ai_variance, 'propensity_adjustment': adjustment},
                 'bias_adjustment': {f'group {treatment.text(i)}': v for i, v in bias_tables.items()} or None,
                 'caliper': _option(spec, 'caliper')}
        correction = f'Abadie-Imbens (2006) heteroskedasticity-robust variance with nn({variance_neighbours}) conditional variances'
        frame = ModelFrame(spec, sample.sample)
        if kt:
            design = treatment_design
            transform = design.transform
            extra['auxiliary_equations'] = {f'TME{treatment.text(1)}': coefficient_table([term.removeprefix('TME:') for term in design.terms], transform@nuisance.treatment[0], transform@vg@transform.T)}
            summaries = {}
            for group in (0, 1):
                count, low, high, mean = store.db.execute('SELECT COUNT(*),MIN(p),MAX(p),AVG(p) FROM rows WHERE level=?', (group,)).fetchone()
                summary = {'n': count, 'min': low, 'max': high, 'mean': mean}
                for name, quantile in [('p25', .25), ('median', .5), ('p75', .75)]:
                    position = (count-1)*quantile
                    left = store.db.execute('SELECT p FROM rows WHERE level=? ORDER BY p,id LIMIT 1 OFFSET ?', (group, math.floor(position))).fetchone()[0]
                    right = store.db.execute('SELECT p FROM rows WHERE level=? ORDER BY p,id LIMIT 1 OFFSET ?', (group, math.ceil(position))).fetchone()[0]
                    summary[name] = left+(right-left)*(position-math.floor(position))
                summaries[treatment.text(group)] = summary
            extra['overlap'] = {'pstolerance': _option(spec, 'pstolerance'), 'propensity': {f'P({treatment.column}={treatment.text(1)})': summaries}}
            correction += ', minus the Abadie-Imbens (2016) estimated-propensity-score adjustment'
            if bias is not None:
                frame.warn("Stata's teffects psmatch has no bias adjustment; the reported propensity-score correction is an approximation for the bias-adjusted estimate.")
        name = 'ATET' if estimand == 'atet' else 'ATE'
        provenance = sample.provenance()
        provenance.update({'method': method, 'estimand': estimand, 'neighbor_search': 'disk indexed one-dimensional' if width == 1 else 'bounded exact candidate replays',
                           'matching_resource_plan': store.plan.record(), 'matching_scratch_bytes': store.path.stat().st_size})
        result = build_result(frame, terms=[treatment.effect_term(name, 1)], params=torch.tensor([tau], dtype=torch.float64),
                              covariance=torch.tensor([[variance]], dtype=torch.float64), equations=[name],
                              title=f'Treatment-effects estimation: {"nearest-neighbor" if matching is not None else "propensity-score"} matching',
                              use_t=False, nobs=sample.nobs, metrics={'n_control': counts[0], 'n_treated': counts[1], 'matches_min': size_min, 'matches_max': size_max},
                              solver='disk_exact_nearest_neighbor_matching',
                              inference={'correction': correction, 'small_sample_correction': 1., 'df_inference': None, 'variance_method': 'Abadie-Imbens matching variance'},
                              extra=extra, provenance=provenance)
        result.nobs_original, result.dropped_rows = sample.original_count, sample.original_count-sample.nrows
        result.sample_positions = []
        return result
    except (OSError, sqlite3.Error) as exc:
        raise AnalysisError('matching_spill_failed', 'Matching needs writable temporary storage and enough free disk space.') from exc
    finally:
        store.close()
