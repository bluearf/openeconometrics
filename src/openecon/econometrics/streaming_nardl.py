"""Global ordered positive/negative partial sums on bounded ARDL replays."""
from __future__ import annotations

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.streaming_design import numeric_values
from .core import kernel_call
from .ordered_replay import OrderedReplay
from .streaming_ardl import _ARDLReplay
from .streaming_linear import _Notes
from .tsmodels import ardl, nardl


class _NARDLReplay(_ARDLReplay):
    def __init__(self, ordered):
        super().__init__(ordered)
        original = list(self.spec.predictors)
        chosen = self.notes.option("asymmetric")
        asymmetric = original if chosen is None else list(chosen)
        if not asymmetric or len(asymmetric) != len(set(asymmetric)) or set(asymmetric)-set(original):
            raise AnalysisError("invalid_spec", "asymmetric must be a nonempty, unique subset of x.")
        if self.notes.option("trend") == "none":
            raise AnalysisError("invalid_option", "NARDL partial sums require a constant or trend.")
        delta = self.notes.option("time_delta")
        if ordered.date:
            try:
                if delta is None:
                    raise ValueError("missing interval")
                interval = pd.Timedelta(delta)
                if interval.value <= 0:
                    raise ValueError("invalid interval")
            except (TypeError, ValueError, OverflowError) as exc:
                raise AnalysisError("invalid_time_delta", "NARDL datetime periods require a positive fixed time_delta, for example '1D'.") from exc
            if ordered.db.execute("SELECT 1 FROM (SELECT t-LAG(t) OVER(ORDER BY t) AS gap FROM rows WHERE valid=1) WHERE gap!=? LIMIT 1", (interval.value,)).fetchone():
                raise AnalysisError("time_gaps", "NARDL datetime periods must be consecutive at time_delta.")
        elif delta is not None:
            raise AnalysisError("invalid_time_delta", "time_delta applies only to datetime periods.")
        self.origin = ordered.db.execute("SELECT pos FROM rows WHERE valid=1 ORDER BY t,pos LIMIT 1").fetchone()[0]
        self.pairs = {name: [name+"_positive", name+"_negative"] for name in asymmetric}
        self.names = [generated for name in original for generated in self.pairs.get(name, [name])]
        if len(self.names) != len(set(self.names)) or set(ordered.columns)&{generated for pair in self.pairs.values() for generated in pair}:
            raise AnalysisError("duplicate_terms", "NARDL partial-sum names collide with source columns.")
        self.imposed = {}
        for kind in ("long_run", "short_run"):
            chosen = list(self.notes.option(kind+"_symmetric") or [])
            if len(chosen) != len(set(chosen)) or set(chosen)-set(asymmetric):
                raise AnalysisError("invalid_spec", kind+"_symmetric must be a unique subset of asymmetric predictors.")
            self.imposed[kind] = chosen
        options = dict(self.spec.options)
        for key in ("lags", "maxlags"):
            if key in options:
                options[key] = nardl._expand_orders(options[key], original, set(asymmetric), name=key)
        if options.get("lags") is not None and options["lags"][0] < 1:
            raise AnalysisError("invalid_lags", "NARDL needs at least one outcome lag.")
        expanded = self.spec.model_copy(update={"options": options})
        self.notes = _Notes(expanded)
        self.frame.spec, self.frame.option, self.frame.warn, self.frame.warnings = expanded, self.notes.option, self.notes.warn, self.notes.warnings
        for _ in self.input_batches():
            pass  # Include changes crossing source blocks in sign validation.
        if not all(self.change_signs[name][0] and self.change_signs[name][1] for name in asymmetric):
            raise AnalysisError("unidentified_asymmetry", "Each asymmetric predictor needs both positive and negative changes.")

    def input_batches(self):
        previous, totals = {}, {name: torch.zeros(2, dtype=torch.float64) for name in self.pairs}
        self.change_signs = {name: [False, False] for name in self.pairs}
        for raw in self.ordered.source.iter_batches(batch_rows=self.ordered.rows):
            frame = raw.copy()
            for name, pair in self.pairs.items():
                values = numeric_values(raw[name], name)
                start = previous.get(name, values[:1])
                difference = torch.diff(values, prepend=start)
                self.change_signs[name][0] |= bool((difference > 0).any())
                self.change_signs[name][1] |= bool((difference < 0).any())
                partials = torch.stack((difference.clamp_min(0).cumsum(0), difference.clamp_max(0).cumsum(0)), dim=1)+totals[name]
                if not bool(torch.isfinite(partials).all()):
                    raise AnalysisError("non_finite_design", "NARDL cumulative changes exceed float64 units; rescale the predictor.")
                frame[pair[0]], frame[pair[1]] = partials[:, 0].numpy(), partials[:, 1].numpy()
                totals[name] = partials[-1].clone()
                previous[name] = values[-1:].clone()
            yield frame

    def restrictions(self, terms, orders):
        rows = nardl._symmetry_rows(terms, self.pairs, dict(zip(self.names, orders[1:], strict=True)))
        chosen = [rows[name][0] for name in self.imposed["long_run"]]
        chosen += [row for name in self.imposed["short_run"] for row in rows[name][1]]
        return torch.stack(chosen) if chosen else torch.empty((0, len(terms)), dtype=torch.float64)


def fit_streaming_nardl(spec, source):
    with OrderedReplay(spec, source) as ordered:
        replay = _NARDLReplay(ordered)
        result = kernel_call(ardl._fit_ardl_frame, replay.frame, names=replay.names, retain_levels=True,
                             constraint_builder=replay.restrictions if any(replay.imposed.values()) else None,
                             _replay=replay)
        result = nardl._finish_nardl(result, spec, replay.names, replay.pairs, replay.imposed, replay.origin)
        ordered.verify_original()
        return result
