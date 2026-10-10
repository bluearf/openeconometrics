"""Explicit physical-row partitions with group/time leakage checks."""

from __future__ import annotations

import json
import math

import pandas as pd
import torch

from openecon.resources import plan_workspace
from . import common as c

SCHEMA = "openecon.supervised.split.v1"
ROLES = ("train", "validation", "test")


def _plan(n, max_work):
    c.integer(n, 2, c.MAX_ROWS, "split rows")
    c.integer(max_work, 1, c.MAX_WORK, "max_work")
    work = 128 * n * (1 + math.ceil(math.log2(max(2, n))))
    if work > max_work:
        c.fail("system_budget", "Complete split validation work exceeds max_work.")
    return plan_workspace(
        "supervised row partition", {"identities, sorting and complete state": n * 8192 + 65536}
    ).record()


def _token(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _valid_label(value):
    raw = c.unlabel(value)
    if raw is None or raw is pd.NA or raw is pd.NaT or isinstance(raw, float) and math.isnan(raw):
        c.fail("invalid_split", "Group and time values must be complete.")
    return raw


def _time_key(value):
    raw = _valid_label(value)
    if type(raw) in (int, float) and type(raw) is not bool:
        return "numeric", raw
    if isinstance(raw, (pd.Timestamp,)):
        return "timestamp", raw.value
    import datetime as dt

    if isinstance(raw, (dt.datetime, dt.date)):
        return "timestamp", pd.Timestamp(raw).value
    c.fail("invalid_split", "Chronological time requires homogeneous numeric values or dates.")


def _allocate(units, fractions):
    m = len(units)
    active = [j for j, v in enumerate(fractions) if v > 0]
    if m < len(active):
        c.fail("insufficient_sample", "Too few atomic units for the requested nonempty partitions.")
    # Largest-remainder allocation by atomic unit count, with a declared one-unit
    # lower bound for each nonzero fraction. This is independent of outcomes.
    remaining = m - len(active)
    raw = [remaining * v for v in fractions]
    counts = [int(math.floor(v)) + int(j in active) for j, v in enumerate(raw)]
    for j in sorted(active, key=lambda j: (-(raw[j] - math.floor(raw[j])), j))[: m - sum(counts)]:
        counts[j] += 1
    roles = [None] * sum(len(unit) for unit in units)
    offset = 0
    for role, count in zip(ROLES, counts):
        for unit in units[offset : offset + count]:
            for row in unit:
                roles[row] = role
        offset += count
    return roles


def _derive(n, settings, groups, times, explicit):
    strategy = settings["strategy"]
    fractions = settings["fractions"]
    if strategy == "explicit":
        c.sequence(explicit, n, "explicit split roles")
        if any(type(v) is not str or v not in ROLES for v in explicit):
            c.fail("invalid_split", "Each explicit role must be train, validation or test.")
        roles = list(explicit)
    else:
        source = groups if strategy == "group" else times if strategy == "chronological" else None
        if source is None and strategy != "iid":
            c.fail("invalid_split", "The declared group/time source is absent.")
        if source is None:
            units = [[i] for i in range(n)]
        else:
            units_by_token = {}
            for i, value in enumerate(source):
                _valid_label(value)
                units_by_token.setdefault(_token(value), []).append(i)
            units = list(units_by_token.values())
        if strategy == "chronological":
            key_types = {_time_key(times[unit[0]])[0] for unit in units}
            if len(key_types) != 1:
                c.fail("invalid_split", "Chronological values must have one time type.")
            units.sort(key=lambda unit: _time_key(times[unit[0]])[1])
        else:
            generator = torch.Generator(device="cpu").manual_seed(settings["seed"])
            order = torch.randperm(len(units), generator=generator, device="cpu").tolist()
            units = [units[i] for i in order]
        roles = _allocate(units, fractions)
    if "train" not in roles:
        c.fail("invalid_split", "A nonempty training partition is required.")
    if groups is not None:
        seen = {}
        for value, role in zip(groups, roles):
            _valid_label(value)
            token = _token(value)
            if token in seen and seen[token] != role:
                c.fail("split_leakage", "An atomic group occurs in multiple partitions.")
            seen[token] = role
    if times is not None:
        keys = [_time_key(v) for v in times]
        if len({kind for kind, _ in keys}) != 1:
            c.fail("invalid_split", "Chronological values must have one time type.")
        extrema = [
            (
                min(v for (kind, v), role in zip(keys, roles) if role == r),
                max(v for (kind, v), role in zip(keys, roles) if role == r),
            )
            for r in ROLES
            if r in roles
        ]
        if any(left[1] >= right[0] for left, right in zip(extrema, extrema[1:])):
            c.fail("split_leakage", "Time partitions overlap, split an atomic date, or look ahead.")
    return {
        "roles": roles,
        "positions": {
            role: [i for i, actual in enumerate(roles) if role == actual] for role in ROLES
        },
    }


def _settings(strategy, fractions, seed, max_work):
    if type(strategy) is not str or strategy not in ("explicit", "iid", "group", "chronological"):
        c.fail("invalid_split", "Unsupported split strategy.")
    c.sequence(fractions, 3, "split fractions")
    fractions = [c.real(v, 0, 1, "split fraction") for v in fractions]
    if fractions[0] <= 0 or abs(sum(fractions) - 1) > 8 * c.EPS:
        c.fail("invalid_split", "Training fraction must be positive and all fractions sum to one.")
    c.integer(seed, 0, 2**63 - 1, "seed")
    c.integer(max_work, 1, c.MAX_WORK, "max_work")
    return {"strategy": strategy, "fractions": fractions, "seed": seed, "max_work": max_work}


@c.cpu_call
def replay(record):
    record = c.load(record)
    c.keys(
        record,
        (
            "schema",
            "n",
            "index",
            "settings",
            "group_name",
            "time_name",
            "groups",
            "times",
            "explicit_roles",
            "roles",
            "positions",
            "digest",
        ),
        "split state",
    )
    if record["schema"] != SCHEMA:
        c.fail("invalid_state", "Unsupported split schema.")
    c.keys(record["settings"], ("strategy", "fractions", "seed", "max_work"), "split settings")
    settings = _settings(**record["settings"])
    c.same(record["settings"], settings, "split settings")
    _plan(record["n"], settings["max_work"])
    c.unseal(record)
    c.restore_index(record["index"], record["n"])
    for name, values in (("group_name", "groups"), ("time_name", "times")):
        if record[name] is None:
            if record[values] is not None:
                c.fail("invalid_state", "Split role source name disagrees with its values.")
        else:
            c.columns([record[name]])
            c.sequence(record[values], record["n"], values)
            for value in record[values]:
                _valid_label(value)
    c.sequence(record["roles"], record["n"], "cached roles")
    c.keys(record["positions"], ROLES, "cached positions")
    for role in ROLES:
        positions = record["positions"][role]
        if not isinstance(positions, (list, tuple)) or any(
            type(i) is not int or not 0 <= i < record["n"] for i in positions
        ):
            c.fail("invalid_state", "Invalid cached split positions.")
    derived = _derive(
        record["n"], settings, record["groups"], record["times"], record["explicit_roles"]
    )
    c.same(record["roles"], derived["roles"], "split roles")
    c.same(record["positions"], derived["positions"], "split positions")
    return record


class SplitState(c.Transport):
    @classmethod
    def replay(cls, payload):
        return replay(payload)


@c.cpu_call
def prediction_split(
    data,
    *,
    roles=None,
    strategy="iid",
    fractions=(0.6, 0.2, 0.2),
    seed=0,
    group=None,
    time=None,
    max_work=c.DEFAULT_WORK,
):
    """Declare disjoint resident train/validation/test physical positions.

    Group randomization assigns complete first-seen groups; chronological splitting
    assigns complete sorted time blocks. Fractions refer to atomic unit counts.
    Explicit role labels can additionally be checked against group/time leakage.
    """
    names = []
    for name in (group, time):
        if name is not None:
            c.columns([name])
            if name not in names:
                names.append(name)
    if isinstance(data, pd.DataFrame) and not names:
        n = len(data)
    elif not names:
        if not isinstance(data, dict) or not data:
            c.fail(
                "unsupported_data", "Split requires a resident table or nonempty column mapping."
            )
        names = c.columns([next(iter(data))])
        n = c.probe(data, names)
    else:
        n = c.probe(data, names)
    settings = _settings("explicit" if roles is not None else strategy, fractions, seed, max_work)
    _plan(n, max_work)
    if roles is not None:
        c.sequence(roles, n, "explicit roles")
    if isinstance(data, pd.DataFrame) and not names:
        index = c.index_record(data.index)
        groups = times = None
    else:
        frame, index = c.resident(data, names, n)
        groups = [c.label(v) for v in frame[group]] if group is not None else None
        times = [c.label(v) for v in frame[time]] if time is not None else None
    derived = _derive(n, settings, groups, times, roles)
    record = c.seal(
        {
            "schema": SCHEMA,
            "n": n,
            "index": index,
            "settings": settings,
            "group_name": group,
            "time_name": time,
            "groups": groups,
            "times": times,
            "explicit_roles": list(roles) if roles is not None else None,
            **derived,
        }
    )
    return SplitState(payload=record)


def split_restore(state):
    return SplitState(payload=replay(state))
