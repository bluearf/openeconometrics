"""Complete declared Cartesian plans, regular binary fractions and design audits."""

from __future__ import annotations

import itertools
import math

import torch

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import table
from openecon.econometrics.resident_cpu import resident_cpu
from . import common as c


@resident_cpu
def conjoint_plan(attributes, *, profile="profile_id", max_work=100_000_000, device="cpu", weights=None):
    """Generate every declared Cartesian profile in ordered level order; no subsampling."""
    attributes, _, p = c.definition(attributes)
    c.name(profile)
    if profile in attributes:
        raise AnalysisError("invalid_spec", "Profile ID cannot be an attribute.")
    n = math.prod(len(levels) for levels in attributes.values())
    settings = c.guard("conjoint_plan", n, p, device=device, weights=weights, max_work=max_work,
                       work=n * len(attributes))
    rows = [[f"p{i+1:04d}", *values] for i, values in enumerate(itertools.product(*attributes.values()))]
    state = {"attributes": attributes, "attribute_order": list(attributes), "profile_column": profile,
             "design": "full_factorial", "profile_order": "last attribute varies fastest", "settings": settings}
    return c.seal("conjoint_plan", {"profiles": table(rows, columns=[profile, *attributes])}, state,
                  inference="descriptive", n_profiles=n, **settings)


@resident_cpu
def conjoint_orthogonal(attributes, base, generators, *, signs=None, profile="profile_id",
                       max_work=100_000_000, device="cpu", weights=None):
    """Generate an explicitly declared regular two-level fraction; audit aliases separately.

    base=['A','B','C'], generators={'D':['A','B','C']} declares D=A*B*C
    in -1/+1 character coding. Optional generator signs are +/-1. This is not
    automatic ORTHOPLAN, a mixed-level array search or an interaction-free design.
    """
    attributes, _, p = c.definition(attributes)
    base = c.c.name_list(base, "base")
    if any(len(v) != 2 for v in attributes.values()) or not set(base) <= set(attributes):
        raise AnalysisError("invalid_spec", "Regular fractions require exactly two levels and named independent base factors.")
    if not isinstance(generators, dict) or set(generators) != set(attributes) - set(base):
        raise AnalysisError("invalid_spec", "Declare generators for every non-base attribute, and no others.")
    signs = {} if signs is None else signs
    if not isinstance(signs, dict) or set(signs) - set(generators) or any(
        isinstance(v, bool) or v not in (-1, 1) for v in signs.values()
    ):
        raise AnalysisError("invalid_spec", "Generator signs must be +/-1 for generated attributes.")
    characters = {a: (1 << i, 1) for i, a in enumerate(base)}
    for attr, word in generators.items():
        word = c.c.name_list(word, "generator")
        if not set(word) <= set(base):
            raise AnalysisError("invalid_spec", "Each generator word must contain distinct independent base factors.")
        mask = sum(1 << base.index(a) for a in word)
        if mask in {v[0] for v in characters.values()}:
            raise AnalysisError("invalid_spec", "Generator characters must be nonconstant and distinct from every main effect.")
        characters[attr] = (mask, signs.get(attr, 1))
    c.name(profile)
    if profile in attributes:
        raise AnalysisError("invalid_spec", "Profile ID cannot be an attribute.")
    n = 2**len(base)
    settings = c.guard("conjoint_orthogonal", n, p, device=device, weights=weights,
                       max_work=max_work, work=n * len(attributes) * len(base))
    rows = []
    for i in range(n):
        row = [f"p{i+1:04d}"]
        for attr, levels in attributes.items():
            mask, sign = characters[attr]
            value = sign
            for j in range(len(base)):
                if mask & (1 << j):
                    value *= 1 if i & (1 << j) else -1
            row.append(levels[int(value > 0)])
        rows.append(row)
    state = {"attributes": attributes, "attribute_order": list(attributes), "profile_column": profile,
             "design": "regular_two_level_fraction", "base": base, "generators": generators,
             "signs": signs, "character_masks": characters, "settings": settings,
             "profile_order": "first base factor varies fastest", "higher_order_aliases_possible": True}
    return c.seal("conjoint_orthogonal", {"profiles": table(rows, columns=[profile, *attributes])},
                  state, inference="descriptive", n_profiles=n, **settings)


@resident_cpu
def conjoint_diagnostics(plan, attributes=None, *, factors=None, profile="profile_id",
                        max_work=100_000_000, device="cpu", weights=None):
    """Audit actual main-effect rank/condition, factor-block orthogonality and pairwise aliases.

    Complete main/two-factor +/-1 alias checks apply to all-discrete two-level
    factors. Mixed-level designs retain full main-effect Gram and Cramer V.
    """
    data, attributes = c.profiles(plan, attributes, profile=profile)
    attributes, modes, p = c.definition(attributes, factors)
    binary = all(len(attributes[a]) == 2 and modes[a] == "discrete" for a in attributes)
    alias_p = 1 + len(attributes) + len(attributes) * (len(attributes)-1)//2 if binary else 0
    n = len(data)
    settings = c.guard("conjoint_diagnostics", n, p, device=device, weights=weights, max_work=max_work,
                       work=n * (p*p + alias_p*alias_p + len(attributes)**2),
                       extra_buffers={"binary_alias_audit": 8 * (4*n*alias_p + 4*alias_p**2)})
    x, terms, mapping, anchors, _ = c.coding(data, attributes, modes)
    singular = torch.linalg.svdvals(x)
    tolerance = max(x.shape) * torch.finfo(torch.float64).eps * float(singular[0])
    rank = int((singular > tolerance).sum())
    condition = float(singular[0]/singular[-1]) if rank == p else None
    gram = x.T @ x
    counts, associations, blocks = [], [], []
    for a, levels in attributes.items():
        observed = [c.key(v) for v in data[a]]
        counts.extend([a, v, observed.count(c.key(v))] for v in levels)
    for first, second in itertools.combinations(mapping, 2):
        a, b = first["attribute"], second["attribute"]
        joint = torch.zeros((len(attributes[a]), len(attributes[b])), dtype=torch.float64)
        lookup_a = {c.key(v): i for i, v in enumerate(attributes[a])}
        lookup_b = {c.key(v): i for i, v in enumerate(attributes[b])}
        for va, vb in zip(data[a], data[b]):
            joint[lookup_a[c.key(va)], lookup_b[c.key(vb)]] += 1
        row, col = joint.sum(1), joint.sum(0)
        expected = row[:, None] * col[None, :] / n
        valid = expected > 0
        chi2 = float((((joint-expected)**2)[valid]/expected[valid]).sum())
        denominator = min(int((row>0).sum())-1, int((col>0).sum())-1)
        v = math.sqrt(chi2/(n*denominator)) if denominator > 0 else None
        associations.append([a, b, v])
        xa = x[:, first["start"]:first["stop"]]
        xb = x[:, second["start"]:second["stop"]]
        cross = (xa-xa.mean(0)).T @ (xb-xb.mean(0)) / n
        blocks.append([a, b, float(cross.abs().max())])
    aliases = []
    if binary:
        columns = [x[:, 0], *[x[:, item["start"]] for item in mapping]]
        labels = ["I", *attributes]
        for i, j in itertools.combinations(range(1, len(columns)), 2):
            columns.append(columns[i]*columns[j])
            labels.append(f"{labels[i]}*{labels[j]}")
        character_gram = torch.stack(columns, 1).T @ torch.stack(columns, 1) / n
        for i, j in itertools.combinations(range(len(labels)), 2):
            value = float(character_gram[i, j])
            if abs(abs(value)-1) <= 1e-12:
                aliases.append([labels[i], labels[j], int(round(value))])
    orthogonal = all(v[2] <= 1e-12 for v in blocks)
    state = {"attributes": attributes, "attribute_order": list(attributes), "modes": modes,
             "profile_ids": [c.label(v) for v in data[profile]], "anchors": anchors,
             "terms": terms, "rank_tolerance": tolerance, "settings": settings}
    return c.seal("conjoint_diagnostics", {
        "summary": table([[n, p, rank, condition, orthogonal, binary]],
                         columns=["profiles", "coefficients", "rank", "condition", "between_factor_orthogonal", "binary_alias_audit"]),
        "gram": table(gram.tolist(), columns=terms, index=terms),
        "levels": table(counts, columns=["attribute", "level", "count"]),
        "associations": table(associations, columns=["first", "second", "cramer_v"]),
        "factor_blocks": table(blocks, columns=["first", "second", "max_abs_centered_cross_product"]),
        "aliases": table(aliases, columns=["first", "second", "sign"]),
    }, state, inference="descriptive", main_effect_identified=rank == p,
       higher_order_aliases_not_excluded=True, **settings)
