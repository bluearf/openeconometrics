"""Anchored frequency-weight moments, without materializing replicated rows."""
from __future__ import annotations

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from . import common as c
from .summary import geometry


def frequency_moments(data, names, weights, missing):
    weights = c.check_name(weights, "weights")
    used = [*names, weights]
    if len(set(used)) != len(used):
        raise AnalysisError("invalid_spec", "Weight and variable roles must be distinct.")
    if isinstance(data, Dataset):
        from openecon.econometrics.stats.replay import Replay
        replay = Replay(data, used, used, missing, check_small=False)
        geometry(len(names), rows=replay.rows)
        blocks = replay.batches()
    else:
        raw = c.source(data)
        geometry(len(names), rows=len(raw))
        sample, _, dropped = c.select(raw, used, missing=missing)
        blocks = [sample]
        replay = None
    total = rows = zeros = 0
    anchor = mean = sscp = None
    for frame in blocks:
        if bool((frame[weights] > 2**53).any()):
            raise AnalysisError("invalid_weights", "Frequency weights exceed exact float64 integer precision.")
        w = c.column(frame, weights)
        if bool(((w < 0) | (w != w.round()) | (w > 2**53)).any()):
            raise AnalysisError("invalid_weights", "Frequency weights must be nonnegative exact integers up to 2^53.")
        x = c.matrix(frame, names)
        positive = w > 0
        zeros += int((~positive).sum())
        x, w = x[positive], w[positive]
        if not len(x):
            continue
        size = sum(int(value) for value in w.tolist())
        if total + size > 2**53:
            raise AnalysisError("invalid_weights", "Frequency weight total exceeds exact float64 integer precision.")
        if anchor is None:
            anchor = x[0].clone()
            mean = torch.zeros(len(names), dtype=c.FLOAT)
            sscp = torch.zeros((len(names), len(names)), dtype=c.FLOAT)
        centered = x - anchor
        location = (w[:, None] * centered).sum(0) / size
        location += (w[:, None] * (centered - location)).sum(0) / size
        dev = centered - location
        block = dev.T @ (w[:, None] * dev)
        delta = location - mean
        sscp += block + torch.outer(delta, delta) * (total * size / (total + size))
        mean += delta * (size / (total + size))
        total += size
        rows += len(x)
    if total < 2:
        raise AnalysisError("insufficient_observations", "Frequency moments need a weight total of at least two.")
    if not bool(torch.isfinite(sscp).all()) or not bool(torch.isfinite(mean).all()):
        raise AnalysisError("numerical_failure", "Weighted moments exceed float64 precision; rescale inputs.")
    if bool((sscp.diagonal() <= 0).any()):
        raise AnalysisError("constant_column", "Every variable must vary among positive-weight observations.")
    extra = replay.attrs() if replay is not None else {"resource_plan": geometry(len(names), rows=len(raw))}
    extra.update({"weights": weights, "weight_type": "fweight", "physical_rows": rows,
                  "n_zero_weight": zeros, "weight_sum": total, "covariance_divisor": total - 1,
                  "precision": "float64", "device": "cpu", "inference": "descriptive; loading SE/CI not provided"})
    return anchor + mean, sscp, total, replay.dropped if replay is not None else dropped, extra
