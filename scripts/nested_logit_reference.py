"""Independent NumPy/SciPy scientific oracle; never imported by the runtime.

The production implementation and its private helpers are deliberately absent.
Scores are differentiated from the physical beta/dissimilarity probability
formula; information is a five-point difference of those independent scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def identity(value):
    return type(value).__name__, value


@dataclass
class Reference:
    data: object
    attributes: tuple = ("cost", "quality")
    fixed: object = None

    def __post_init__(self):
        self.frame = self.data.reset_index(drop=True).copy()
        self.fixed = {} if self.fixed is None else dict(self.fixed)
        self.nests = list(dict.fromkeys(self.frame["nest"].tolist()))
        self.free = [nest for nest in self.nests if nest not in self.fixed]
        self.cases = list(dict.fromkeys(self.frame["choice"].tolist()))
        self.groups = [np.flatnonzero(self.frame.choice.to_numpy() == case) for case in self.cases]
        self.x = self.frame[list(self.attributes)].to_numpy(dtype=float)
        self.available = self.frame.available.to_numpy(dtype=bool)
        self.nest_indices = np.array([self.nests.index(value) for value in self.frame.nest])
        self.q = len(self.attributes)
        self.k = self.q + len(self.free)
        self.free_indices = [self.nests.index(nest) for nest in self.free]
        self.max_rows = max(map(len, self.groups))
        self.rows = np.full((len(self.groups), self.max_rows), -1, dtype=int)
        self.mask = np.zeros_like(self.rows, dtype=bool)
        self.chosen_indices = []
        for ci, group in enumerate(self.groups):
            self.rows[ci, :len(group)] = group
            self.mask[ci, :len(group)] = self.available[group]
            if "chosen" in self.frame:
                self.chosen_indices.append(np.flatnonzero(self.frame.chosen.to_numpy()[group] == 1)[0])
        self.centered = self.x.copy()
        for group in self.groups:
            selected = group[self.available[group]]
            self.centered[group] -= self.x[selected[0]]

    def lambdas(self, params):
        result = np.array([self.fixed.get(nest, 1.) for nest in self.nests], dtype=float)
        result[self.free_indices] = np.asarray(params)[self.q:]
        return result

    def moments(self, params):
        params = np.asarray(params, dtype=float)
        lam = self.lambdas(params)
        utility = (self.centered @ params[:self.q])[self.rows]
        design = self.centered[self.rows]
        membership = self.nest_indices[self.rows]
        c, a = self.rows.shape
        conditional = np.zeros((c, a))
        log_conditional = np.full((c, a), -np.inf)
        logs = np.full((c, len(self.nests)), -np.inf)
        means_x = np.zeros((c, len(self.nests), self.q))
        means_v = np.zeros((c, len(self.nests)))
        entropy = np.zeros_like(means_v)
        for m in range(len(self.nests)):
            mask = self.mask & (membership == m)
            present = mask.any(axis=1)
            logits = np.where(mask, utility/lam[m], -np.inf)
            inclusive = logsumexp(logits, axis=1)
            subtract = np.where(present, inclusive, 0.)
            logq = np.where(mask, logits-subtract[:, None], -np.inf)
            q = np.exp(logq)
            conditional += q
            log_conditional[mask] = logq[mask]
            means_x[:, m] = np.einsum("ca,caq->cq", q, design)
            means_v[:, m] = np.einsum("ca,ca->c", q, utility)
            logs[:, m] = lam[m]*inclusive
            entropy[:, m] = np.where(present, subtract-means_v[:, m]/lam[m], 0.)
        log_nest = logs-logsumexp(logs, axis=1)[:, None]
        nest_probability = np.exp(log_nest)
        log_probability = log_conditional+np.take_along_axis(log_nest, membership, axis=1)
        probability = np.exp(log_probability)
        aggregate_x = np.einsum("cm,cmq->cq", nest_probability, means_x)
        chosen_means = means_x[np.arange(c)[:, None], membership]
        chosen_v = means_v[np.arange(c)[:, None], membership]
        chosen_lam = lam[membership]
        jac_logp = np.zeros((c, a, self.k))
        jac_logp[:, :, :self.q] = (design-chosen_means)/chosen_lam[:, :, None]+chosen_means-aggregate_x[:, None, :]
        for column, free_m in enumerate(self.free_indices, self.q):
            jac_logp[:, :, column] = -nest_probability[:, free_m, None]*entropy[:, free_m, None]
            jac_logp[:, :, column] += (membership == free_m)*(-(utility-chosen_v)/chosen_lam**2+entropy[np.arange(c)[:, None], membership])
        jac_logp[~self.mask] = 0.
        def flat(value, fill):
            result = np.full((len(self.frame), *value.shape[2:]), fill)
            physical = self.rows >= 0
            result[self.rows[physical]] = value[physical]
            return result
        if self.chosen_indices:
            selected = (np.arange(c), self.chosen_indices)
            case_scores = jac_logp[selected]
            case_ll = log_probability[selected]
        else:
            case_scores, case_ll = np.empty((0, self.k)), np.empty(0)
        return dict(probability=flat(probability, 0.), log_probability=flat(log_probability, -np.inf),
                    conditional=flat(conditional, 0.), log_conditional=flat(log_conditional, -np.inf),
                    nest_probability=nest_probability, log_nest_probability=log_nest,
                    log_probability_jacobian=flat(jac_logp, 0.),
                    probability_jacobian=flat(probability[:, :, None]*jac_logp, 0.),
                    case_scores=case_scores, case_loglikelihood=case_ll)

    def objective(self, params):
        moments = self.moments(params)
        return -moments["case_loglikelihood"].sum(), -moments["case_scores"].sum(axis=0)

    def fit(self, starts=None):
        if starts is None:
            starts = [np.r_[np.zeros(self.q), np.repeat(value, len(self.free))] for value in (.25, .55, .85)]
        bounds = [(None, None)]*self.q+[(.00101, .99899)]*len(self.free)
        results = [minimize(self.objective, np.asarray(start), jac=True, method="L-BFGS-B", bounds=bounds,
                            options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 1000, "maxls": 60})
                   for start in starts]
        finite = [result for result in results if np.isfinite(result.fun)]
        best = min(finite, key=lambda result: result.fun)
        # A physical-coordinate root refinement removes L-BFGS relative-loss
        # stopping noise, while retaining the independently selected basin.
        from scipy.optimize import root
        refined = root(lambda value: self.objective(value)[1], best.x, method="hybr", options={"xtol": 1e-10})
        if refined.success and np.all(refined.x[self.q:] > .001) and np.all(refined.x[self.q:] < .999):
            best.x = refined.x
            best.fun = self.objective(best.x)[0]
        return best, results

    def information(self, params):
        return -jacobian(lambda value: self.moments(value)["case_scores"].sum(axis=0), params)

    def covariance(self, params, vce="oim"):
        moments = self.moments(params)
        info = self.information(params)
        info = (info+info.T)/2
        bread = np.linalg.inv(info)
        score = moments["case_scores"]
        labels = []
        cluster_score = []
        if "respondent" in self.frame:
            labels = list(dict.fromkeys(self.frame.respondent.tolist()))
            for label in labels:
                indices = [j for j, group in enumerate(self.groups) if self.frame.respondent.iloc[group[0]] == label]
                cluster_score.append(score[indices].sum(axis=0))
        cluster_score = np.asarray(cluster_score)
        selected = cluster_score if vce == "cr0" else score
        meat = selected.T @ selected
        covariance = bread if vce == "oim" else bread @ meat @ bread
        return dict(information=info, bread=bread, meat=meat, covariance=covariance,
                    case_scores=score, cluster_scores=cluster_score)

    def target(self, params, target):
        moments = self.moments(params)
        ci = self.cases.index(target["case"])
        kind = target.get("kind", "probability")
        if "nest" in target:
            return moments[kind][ci, self.nests.index(target["nest"])]
        rows = self.groups[ci]
        matches = [i for i in rows if identity(self.frame.alternative.iloc[i]) == identity(target["alternative"])]
        return moments[kind][matches[0]]

    def effect(self, params, case, outcome, changed, attribute, kind="effect"):
        ci = self.cases.index(case)
        rows = self.groups[ci]
        outcomes = [row for row in rows if identity(self.frame.alternative.iloc[row]) == identity(outcome)]
        changes = [row for row in rows if identity(self.frame.alternative.iloc[row]) == identity(changed)]
        if not outcomes or not changes:
            return None
        i, j = outcomes[0], changes[0]
        if not self.available[i] or not self.available[j]:
            return None
        moments = self.moments(params)
        lam = self.lambdas(params)
        m = self.nest_indices[i]
        same = self.nest_indices[j] == m
        bracket = (i == j)/lam[m]+same*(1-1/lam[m])*moments["conditional"][j]-moments["probability"][j]
        beta = np.asarray(params)[self.attributes.index(attribute)]
        return beta*bracket*(moments["probability"][i] if kind == "effect" else self.x[j, self.attributes.index(attribute)])

    def ame(self, params, target, weights=None, kind="effect"):
        weights = dict.fromkeys(self.cases, 1.) if weights is None else weights
        values, selected = [], []
        for case in self.cases:
            value = self.effect(params, case, target["outcome"], target["changed"], target["attribute"], kind)
            if value is not None:
                values.append(value)
                selected.append(weights[case])
        return np.average(values, weights=selected)


def jacobian(function, params, relative_step=2e-4):
    """Independent five-point central differences in physical coordinates."""
    params = np.asarray(params, dtype=float)
    columns = []
    for j in range(len(params)):
        step = relative_step*max(1., abs(params[j]))
        direction = np.zeros_like(params)
        direction[j] = step
        columns.append((-np.asarray(function(params+2*direction))+8*np.asarray(function(params+direction))
                        -8*np.asarray(function(params-direction))+np.asarray(function(params-2*direction)))/(12*step))
    return np.stack(columns, axis=-1)
