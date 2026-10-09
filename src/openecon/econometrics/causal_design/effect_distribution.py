"""Pointwise sharp individual-effect bounds for two finite empirical marginals."""

from __future__ import annotations

from numbers import Integral

import pandas as pd
import torch

from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.econometrics.multivariate import common as c
from openecon.resources import plan_workspace

from . import common as m

_NAMES = ("treatment_effect_cdf_bounds", "treatment_effect_quantile_bounds")


def _fail(message, code="invalid_spec"):
    raise AnalysisError(code, message)


def _grid(values, name, maximum, *, quantiles=False):
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= maximum:
        _fail(f"{name} must be a fixed list of 1..{maximum} distinct finite numbers.")
    result = []
    for value in values:
        try:
            number = c.check_number(value, name)
        except OverflowError:
            _fail(f"A {name} value cannot be represented as finite float64.", "numerical_failure")
        if isinstance(value, Integral) and int(value) != int(number):
            _fail(
                f"An integer {name} value is not exactly representable as float64.",
                "numerical_failure",
            )
        if value != number:
            _fail(f"A {name} value changes during float64 conversion.", "numerical_failure")
        if quantiles and not 0 < number < 1:
            _fail("Quantiles must be strictly inside (0,1).", "invalid_option")
        result.append(number)
    if len(set(result)) != len(result):
        _fail(f"{name} must contain distinct represented values.")
    return result


def _differences(support):
    # TwoSum assumes round-to-nearest binary64, gradual underflow and finite
    # intermediates. The bounded support prevents all intermediate overflow.
    tiny = float.fromhex("0x0.0000000000001p-1022")
    probe = torch.tensor(tiny, dtype=torch.float64, device="cpu")
    if float(probe * torch.ones((), dtype=torch.float64, device="cpu")) != tiny:
        _fail(
            "CPU float64 must retain subnormal arithmetic; flush-to-zero is unsupported.",
            "numerical_failure",
        )
    values = torch.tensor(support, dtype=torch.float64, device="cpu")
    if not bool(torch.isfinite(values).all()) or bool((values.abs() > 1e150).any()):
        _fail("Prespecified support magnitudes must not exceed 1e150.", "numerical_failure")
    a, b = values[None, :], -values[:, None]
    total = a + b
    virtual = total - a
    error = (a - (total - virtual)) + (b - virtual)
    if (
        not bool(torch.isfinite(total).all())
        or not bool(torch.isfinite(error).all())
        or bool((error != 0).any())
    ):
        _fail(
            "Every treated-minus-control support difference must be exactly representable in float64.",
            "numerical_failure",
        )
    return total


def _label_size(value):
    # Admit caller-owned scalars before creating an escaped JSON copy.
    if isinstance(value, str) and len(value) > 256:
        _fail("Original text labels exceed the 256-byte escaped-label domain.", "resource_limit")
    if isinstance(value, Integral) and int(value).bit_length() > 850:
        _fail("Original integer labels exceed the 256-byte escaped-label domain.", "resource_limit")
    encoded = m.canonical(m._label(value)).encode("ascii")
    if len(encoded) > 256:
        _fail(
            "Original index and column labels are limited to 256 escaped JSON bytes.",
            "resource_limit",
        )
    return len(encoded)


def _prepare(
    data, y, treatment, support, targets, *, quantiles, design, missing, device, weights, max_work
):
    m.options(device, weights, max_work)
    if design != "randomized" or not isinstance(design, str):
        _fail(
            "Explicitly declare design='randomized' for the two empirical marginal laws.",
            "unsupported_design",
        )
    if missing != "raise":
        _fail(
            "Complete original empirical marginals require missing='raise'.", "unsupported_missing"
        )
    support = _grid(support, "support", 8)
    targets = _grid(
        targets,
        "quantiles" if quantiles else "thresholds",
        16 if quantiles else 64,
        quantiles=quantiles,
    )
    names = [c.check_name(y, "y"), c.check_name(treatment, "treatment")]
    if names[0] == names[1]:
        _fail("Outcome and treatment must be distinct columns.")
    if isinstance(data, Dataset):
        _fail(
            "Finite empirical bounds require a resident table, not Dataset replay.",
            "unsupported_dataset",
        )
    source = c.source(data)
    n, k = len(source), len(support)
    if n > 100000:
        _fail("At most 100000 original observations are supported.", "resource_limit")
    c.require_numeric(source, names)
    if pd.api.types.is_bool_dtype(source[y].dtype):
        _fail("Outcomes must be numeric support values, not Boolean labels.", "non_numeric_column")
    if pd.api.types.is_float_dtype(source[y].dtype) and getattr(source[y].dtype, "itemsize", 8) > 8:
        _fail("Floating outcome dtypes wider than binary64 are unsupported.", "unsupported_dtype")
    label_bytes = sum(_label_size(value) for value in source.index) + sum(
        _label_size(name) for name in names
    )
    v, e = 2 * k + 2, k * k + 2 * k
    runs = 2 * (k * k if quantiles else len(targets))
    # Edmonds-Karp: <=V*E augmentations per run; dense BFS scans <=V^2
    # entries, plus path updates, certificate validation, and complete copies.
    work = (
        512 * n
        + 16 * label_bytes
        + runs * ((v * e + 1) * (4 * v * v + 8 * v) + 1024 * v * v)
        + 2048 * k * k * len(targets)
    )
    m.work(
        work,
        max_work,
        "all exact transportation extrema, certificates, original sample and witnesses",
    )
    plan = plan_workspace(
        "finite empirical individual-effect bounds",
        {
            "selected_source_and_complete_provenance": 2048 * n,
            "subjects_counts_and_full_portable_copies": 1024 * n,
            "escaped_original_labels_repeated_copies": 16 * label_bytes,
            "every_flow_residual_cut_coupling_and_JSON_copy": 256 * runs * v * v,
            "all_quantile_witnesses_and_tables": 4096 * k * k * len(targets),
            "support_difference_and_solver_buffers": 65536,
        },
    )
    differences = _differences(support)
    # Refuse integer->float collisions before constructing the selected table.
    if pd.api.types.is_integer_dtype(source[y].dtype):
        for value in source[y].dropna():
            if int(value) != int(float(value)):
                _fail(
                    "An original integer outcome is not exactly representable in float64.",
                    "numerical_failure",
                )
    selected, metadata = m.sample(
        source, names, numeric=names, missing="raise", max_work=max_work, minimum=2
    )
    assignment, outcomes = m.binary(selected, treatment), m.tensor(selected, y)
    membership = (
        outcomes[:, None] == torch.tensor(support, dtype=torch.float64, device="cpu")[None, :]
    )
    if not bool(membership.any(1).all()):
        _fail(
            "Every original outcome must belong exactly to the prespecified finite support.",
            "invalid_support",
        )
    counts = torch.stack([membership[assignment == arm].sum(0) for arm in (0, 1)]).to(torch.int64)
    n0, n1 = (int(row.sum()) for row in counts)
    denominator = n0 * n1
    if denominator > 10_000_000_000:
        _fail("The common integer mass denominator exceeds 1e10.", "resource_limit")
    row_margins, column_margins = counts[0] * n1, counts[1] * n0
    metadata.update(
        planned_work=work,
        computation_resource_plan=plan.record(),
        original_escaped_label_bytes=label_bytes,
        input_schema={name: str(source[name].dtype) for name in names},
    )
    state = dict(
        support=support,
        pairwise_effects=differences.tolist(),
        outcomes=outcomes.tolist(),
        treatment=assignment.tolist(),
        arm_counts=counts.tolist(),
        arm_sizes=[n0, n1],
        denominator=denominator,
        row_mass_counts=row_margins.tolist(),
        column_mass_counts=column_margins.tolist(),
        inference=None,
        covariance=None,
        standard_error=None,
        confidence_interval=None,
        target="pointwise sharp bounds for individual difference Y(1)-Y(0) over couplings of the two supplied empirical marginal PMFs",
        population_sampling_inference=False,
        joint_curve_sharpness_claimed=False,
        assumptions=[
            "declared binary randomization identifies the relevant marginal potential-outcome laws",
            "consistency and no interference",
            "prespecified finite common outcome support",
            "the two empirical arm PMFs are the fixed input laws; no unknown population coverage is claimed",
            "no restriction on dependence between the two potential outcomes",
        ],
        numerical_domain="CPU IEEE754 binary64 round-to-nearest with gradual underflow; every support difference exact; Torch int64 transportation counts",
        flow_iteration_bound=v * e,
    )
    state["scientific_input_sha256"] = m.digest(
        dict(
            sample=metadata,
            support=support,
            outcomes=state["outcomes"],
            treatment=state["treatment"],
        )
    )
    return state, metadata, targets, row_margins, column_margins, differences


def _flow(rows, columns, allowed):
    k, mass = len(rows), int(rows.sum())
    v, source, sink = 2 * k + 2, 2 * k, 2 * k + 1
    capacities = torch.zeros((v, v), dtype=torch.int64, device="cpu")
    capacities[source, :k], capacities[k : 2 * k, sink] = rows, columns
    capacities[:k, k : 2 * k] = allowed.to(torch.int64) * mass
    flow = torch.zeros_like(capacities)
    iterations = 0
    while True:
        residual = capacities - flow
        parent = [-1] * v
        parent[source] = source
        queue = [source]
        for u in queue:
            for node in torch.nonzero(residual[u] > 0, as_tuple=False).flatten().tolist():
                if parent[node] == -1:
                    parent[node] = u
                    queue.append(node)
        if parent[sink] == -1:
            break
        if iterations >= v * (k * k + 2 * k):
            _fail(
                "The complete Edmonds-Karp iteration certificate exceeded its admitted bound.",
                "numerical_failure",
            )
        path, node = [], sink
        while node != source:
            path.append((parent[node], node))
            node = parent[node]
        amount = min(int(residual[u, node]) for u, node in path)
        for u, node in path:
            flow[u, node] += amount
            flow[node, u] -= amount
        iterations += 1
    value = int(flow[source].sum())
    cut = torch.tensor([item != -1 for item in parent], dtype=torch.bool, device="cpu")
    cut_capacity = int(capacities[cut][:, ~cut].sum())
    coupling = flow[:k, k : 2 * k].clone()
    left_rows, left_columns = rows - coupling.sum(1), columns - coupling.sum(0)
    for i in range(k):
        for j in range(k):
            amount = torch.minimum(left_rows[i], left_columns[j])
            coupling[i, j] += amount
            left_rows[i] -= amount
            left_columns[j] -= amount
    record = dict(
        allowed=allowed.tolist(),
        capacities=capacities.tolist(),
        flow=flow.tolist(),
        residual=residual.tolist(),
        reachable_cut=cut.tolist(),
        maximum_allowed_mass_count=value,
        cut_capacity=cut_capacity,
        coupling_mass_counts=coupling.tolist(),
        augmentations=iterations,
    )
    _certificate(record, rows, columns, allowed)
    return record


def _integer_matrix(value, shape):
    if (
        not isinstance(value, list)
        or len(value) != shape[0]
        or any(not isinstance(row, list) or len(row) != shape[1] for row in value)
    ):
        _fail("A transportation certificate has invalid dimensions.", "invalid_saved_result")
    if any(type(cell) is not int or abs(cell) > 10_000_000_000 for row in value for cell in row):
        _fail("A transportation certificate has invalid integer counts.", "invalid_saved_result")
    return torch.tensor(value, dtype=torch.int64, device="cpu")


def _certificate(record, rows, columns, allowed):
    k, mass = len(rows), int(rows.sum())
    v = 2 * k + 2
    capacities = _integer_matrix(record["capacities"], (v, v))
    flow = _integer_matrix(record["flow"], (v, v))
    residual = _integer_matrix(record["residual"], (v, v))
    coupling = _integer_matrix(record["coupling_mass_counts"], (k, k))
    expected = torch.zeros_like(capacities)
    expected[2 * k, :k], expected[k : 2 * k, 2 * k + 1] = rows, columns
    expected[:k, k : 2 * k] = allowed.to(torch.int64) * mass
    raw_cut = record["reachable_cut"]
    if (
        not isinstance(raw_cut, list)
        or len(raw_cut) != v
        or any(type(item) is not bool for item in raw_cut)
    ):
        _fail("Invalid source cut membership.", "invalid_saved_result")
    cut = torch.tensor(raw_cut, dtype=torch.bool, device="cpu")
    value = record["maximum_allowed_mass_count"]
    if (
        type(value) is not int
        or not 0 <= value <= mass
        or type(record["cut_capacity"]) is not int
        or type(record["augmentations"]) is not int
        or not 0 <= record["augmentations"] <= v * (k * k + 2 * k)
    ):
        _fail("Invalid flow value/cut/augmentation counts.", "invalid_saved_result")
    reachable = [False] * v
    reachable[2 * k] = True
    queue = [2 * k]
    for node in queue:
        for other in torch.nonzero(residual[node] > 0, as_tuple=False).flatten().tolist():
            if not reachable[other]:
                reachable[other] = True
                queue.append(other)
    balance = flow.sum(1)
    expected_balance = torch.zeros(v, dtype=torch.int64, device="cpu")
    expected_balance[2 * k] = value
    expected_balance[2 * k + 1] = -value
    checks = [
        torch.equal(capacities, expected),
        torch.equal(flow, -flow.T),
        torch.equal(residual, capacities - flow),
        bool((residual >= 0).all()),
        torch.equal(balance, expected_balance),
        raw_cut == reachable,
        bool(cut[2 * k]),
        not bool(cut[2 * k + 1]),
        not bool((residual[cut][:, ~cut] > 0).any()),
        int(capacities[cut][:, ~cut].sum()) == value == record["cut_capacity"],
        bool((coupling >= 0).all()),
        torch.equal(coupling.sum(1), rows),
        torch.equal(coupling.sum(0), columns),
        int(coupling[allowed].sum()) == value,
        record["allowed"] == allowed.tolist(),
    ]
    if not all(checks):
        _fail("The full coupling/flow/min-cut certificate is inconsistent.", "invalid_saved_result")


def _extrema(threshold, rows, columns, differences):
    allowed = differences <= threshold
    upper = _flow(rows, columns, allowed)
    lower = _flow(rows, columns, ~allowed)
    return dict(
        threshold=threshold,
        lower_mass_count=int(rows.sum()) - lower["maximum_allowed_mass_count"],
        upper_mass_count=upper["maximum_allowed_mass_count"],
        lower=lower,
        upper=upper,
    )


def _witness(coupling, grid, differences):
    counts = torch.tensor(coupling, dtype=torch.int64, device="cpu")
    pmf = [int(counts[differences == value].sum()) for value in grid]
    cumulative = []
    total = 0
    for count in pmf:
        total += count
        cumulative.append(total)
    return dict(coupling_mass_counts=coupling, effect_mass_counts=pmf, cdf_mass_counts=cumulative)


def _quantile_records(state, quantiles, differences):
    mass = state["denominator"]
    grid = state["effect_grid"]
    extrema = state["extrema"]
    records = []
    for q in quantiles:
        numerator, denominator = q.as_integer_ratio()
        required = (numerator * mass + denominator - 1) // denominator
        lo = next(i for i, value in enumerate(extrema) if value["upper_mass_count"] >= required)
        hi = next(i for i, value in enumerate(extrema) if value["lower_mass_count"] >= required)
        lower = extrema[lo]["upper"]["coupling_mass_counts"]
        if hi:
            upper = extrema[hi - 1]["lower"]["coupling_mass_counts"]
        else:
            counts = torch.tensor(state["arm_counts"], dtype=torch.int64, device="cpu")
            upper = (counts[0, :, None] * counts[1, None, :]).tolist()
        low_witness, high_witness = (
            _witness(lower, grid, differences),
            _witness(upper, grid, differences),
        )
        records.append(
            dict(
                quantile=q,
                quantile_numerator=str(numerator),
                quantile_denominator=str(denominator),
                required_mass_count=required,
                lower_bound=grid[lo],
                upper_bound=grid[hi],
                lower_grid_index=lo,
                upper_grid_index=hi,
                upper_certificate_predecessor_index=hi - 1 if hi else None,
                lower_witness=low_witness,
                upper_witness=high_witness,
            )
        )
    return records


def _tables(state, positions):
    k = len(state["support"])
    mass = state["denominator"]
    marginals = m.frame(
        [
            [
                arm,
                i,
                value,
                state["arm_counts"][arm][i],
                state["arm_sizes"][arm],
                state["arm_counts"][arm][i] / state["arm_sizes"][arm],
            ]
            for arm in (0, 1)
            for i, value in enumerate(state["support"])
        ],
        columns=["arm", "support_index", "outcome", "count", "arm_n", "probability"],
    )
    couplings = []
    pmfs = []
    quantiles = "quantile_records" in state
    if quantiles:
        rows = [
            [
                value["quantile"],
                value["lower_bound"],
                value["upper_bound"],
                value["required_mass_count"],
                mass,
            ]
            for value in state["quantile_records"]
        ]
        bounds = m.frame(
            rows,
            columns=[
                "quantile",
                "lower_bound",
                "upper_bound",
                "required_mass_count",
                "denominator",
            ],
        )
        witnesses = [
            [value[bound + "_witness"] for bound in ("lower", "upper")]
            for value in state["quantile_records"]
        ]
    else:
        bounds = m.frame(
            [
                [
                    value["threshold"],
                    value["lower_mass_count"] / mass,
                    value["upper_mass_count"] / mass,
                    value["lower_mass_count"],
                    value["upper_mass_count"],
                    mass,
                ]
                for value in state["extrema"]
            ],
            columns=[
                "threshold",
                "lower_bound",
                "upper_bound",
                "lower_mass_count",
                "upper_mass_count",
                "denominator",
            ],
        )
        witnesses = [[value[bound] for bound in ("lower", "upper")] for value in state["extrema"]]
    for target, values in enumerate(witnesses):
        for bound, value in zip(("lower", "upper"), values, strict=True):
            matrix = value["coupling_mass_counts"]
            for i in range(k):
                for j in range(k):
                    couplings.append(
                        [
                            target,
                            bound,
                            i,
                            j,
                            state["support"][i],
                            state["support"][j],
                            state["pairwise_effects"][i][j],
                            matrix[i][j],
                            mass,
                            matrix[i][j] / mass,
                        ]
                    )
            if quantiles:
                for effect, count, cdf in zip(
                    state["effect_grid"],
                    value["effect_mass_counts"],
                    value["cdf_mass_counts"],
                    strict=True,
                ):
                    pmfs.append([target, bound, effect, count, cdf, mass, count / mass, cdf / mass])
    tables = dict(
        bounds=bounds,
        marginals=marginals,
        couplings=m.frame(
            couplings,
            columns=[
                "target_index",
                "bound",
                "control_support_index",
                "treated_support_index",
                "control_outcome",
                "treated_outcome",
                "effect",
                "mass_count",
                "denominator",
                "mass",
            ],
        ),
        subjects=m.frame(
            list(zip(positions, state["treatment"], state["outcomes"], strict=True)),
            columns=["position", "treatment", "outcome"],
        ),
    )
    if quantiles:
        tables["witness_distributions"] = m.frame(
            pmfs,
            columns=[
                "target_index",
                "bound",
                "effect",
                "mass_count",
                "cdf_mass_count",
                "denominator",
                "probability",
                "cdf",
            ],
        )
    return tables


def validate_effect_distribution_result(result):
    """Validate full counts, attained witnesses and dual certificates; no solve/refit.

    The common artifact codec may dispatch here on save/load for these two APIs.
    Checksums alone are integrity checks, not semantic transportation proofs.
    """
    try:
        state = result.attrs["state"]
        settings = state["settings"]
        sample = state["sample"]
        if state["procedure"] not in _NAMES or settings["design"] != "randomized":
            _fail("Incorrect effect-distribution procedure/design.", "invalid_saved_result")
        support = _grid(state["support"], "support", 8)
        if (
            settings["support"] != support
            or settings["missing"] != "raise"
            or settings["device"] != "cpu"
            or settings["weights"] is not None
            or settings["y"] == settings["treatment"]
            or sample["columns"] != [settings["y"], settings["treatment"]]
            or sample["missing"] != "raise"
            or sample["device"] != "cpu"
            or sample["dtype"] != "float64"
            or state["inference"] is not None
            or state["covariance"] is not None
            or state["standard_error"] is not None
            or state["confidence_interval"] is not None
            or state["population_sampling_inference"] is not False
            or state["joint_curve_sharpness_claimed"] is not False
            or state["flow_iteration_bound"]
            != (2 * len(support) + 2) * (len(support) ** 2 + 2 * len(support))
        ):
            _fail(
                "Saved empirical-only domain and role claims are inconsistent.",
                "invalid_saved_result",
            )
        m.options(settings["device"], settings["weights"], settings["max_work"])
        if settings["max_work"] != sample["max_work"]:
            _fail("Saved work admission is inconsistent.", "invalid_saved_result")
        differences = _differences(support)
        if state["pairwise_effects"] != differences.tolist():
            _fail("Saved support differences are inconsistent.", "invalid_saved_result")
        n = len(state["outcomes"])
        if (
            not 2 <= n <= 100000
            or len(state["treatment"]) != n
            or sample["positions"] != list(range(n))
            or sample["n"] != n
            or sample["n_input"] != n
            or sample["n_missing"] != 0
            or len(sample["unit_labels"]) != n
        ):
            _fail("Saved original sample topology is inconsistent.", "invalid_saved_result")
        if state["scientific_input_sha256"] != m.digest(
            dict(
                sample=sample,
                support=support,
                outcomes=state["outcomes"],
                treatment=state["treatment"],
            )
        ):
            _fail("Saved scientific input identity is inconsistent.", "invalid_saved_result")
        counts = torch.zeros((2, len(support)), dtype=torch.int64, device="cpu")
        for y, d in zip(state["outcomes"], state["treatment"], strict=True):
            if (
                type(d) not in (float, int)
                or d not in (0, 1)
                or type(y) not in (float, int)
                or y not in support
            ):
                _fail("Saved outcome/assignment is inconsistent.", "invalid_saved_result")
            counts[int(d), support.index(y)] += 1
        sizes = [int(row.sum()) for row in counts]
        mass = sizes[0] * sizes[1]
        rows, columns = counts[0] * sizes[1], counts[1] * sizes[0]
        if (
            min(sizes) == 0
            or mass > 1e10
            or state["arm_sizes"] != sizes
            or state["denominator"] != mass
            or state["arm_counts"] != counts.tolist()
            or state["row_mass_counts"] != rows.tolist()
            or state["column_mass_counts"] != columns.tolist()
        ):
            _fail("Saved empirical marginal mass counts are inconsistent.", "invalid_saved_result")
        quantiles = state["procedure"] == _NAMES[1]
        targets = _grid(
            settings["quantiles"] if quantiles else settings["thresholds"],
            "targets",
            16 if quantiles else 64,
            quantiles=quantiles,
        )
        grid = sorted(set(differences.flatten().tolist())) if quantiles else targets
        if len(state["extrema"]) != len(grid):
            _fail("Saved complete extrema grid is inconsistent.", "invalid_saved_result")
        for threshold, record in zip(grid, state["extrema"], strict=True):
            allowed = differences <= threshold
            _certificate(record["upper"], rows, columns, allowed)
            _certificate(record["lower"], rows, columns, ~allowed)
            if (
                record["threshold"] != threshold
                or record["upper_mass_count"] != record["upper"]["maximum_allowed_mass_count"]
                or record["lower_mass_count"]
                != mass - record["lower"]["maximum_allowed_mass_count"]
            ):
                _fail(
                    "Saved CDF endpoints disagree with their optimality certificates.",
                    "invalid_saved_result",
                )
        if quantiles:
            if state["effect_grid"] != grid or state["quantile_records"] != _quantile_records(
                state, targets, differences
            ):
                _fail(
                    "Saved sharp quantile witnesses/crossings are inconsistent.",
                    "invalid_saved_result",
                )
            for value in state["quantile_records"]:
                for bound in ("lower", "upper"):
                    index = value[bound + "_grid_index"]
                    cdf = value[bound + "_witness"]["cdf_mass_counts"]
                    if cdf[index] < value["required_mass_count"] or (
                        index and cdf[index - 1] >= value["required_mass_count"]
                    ):
                        _fail(
                            "A witness does not attain its claimed left quantile.",
                            "invalid_saved_result",
                        )
        if m._tables(result) != m._tables(_tables(state, sample["positions"])):
            _fail(
                "Saved scientific tables disagree with complete transportation state.",
                "invalid_saved_result",
            )
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        _fail(f"Invalid effect-distribution saved state: {exc}", "invalid_saved_result")


def _run(
    data, y, treatment, *, support, targets, quantiles, design, missing, device, weights, max_work
):
    state, metadata, targets, rows, columns, differences = _prepare(
        data,
        y,
        treatment,
        support,
        targets,
        quantiles=quantiles,
        design=design,
        missing=missing,
        device=device,
        weights=weights,
        max_work=max_work,
    )
    settings = dict(
        y=y,
        treatment=treatment,
        support=state["support"],
        design=design,
        missing=missing,
        device=device,
        weights=None,
        max_work=max_work,
    )
    settings["quantiles" if quantiles else "thresholds"] = targets
    grid = sorted(set(differences.flatten().tolist())) if quantiles else targets
    state["extrema"] = [_extrema(threshold, rows, columns, differences) for threshold in grid]
    if quantiles:
        state["effect_grid"] = grid
        state["quantile_records"] = _quantile_records(state, targets, differences)
    result = m.result(
        _NAMES[int(quantiles)],
        _tables(state, metadata["positions"]),
        metadata,
        settings,
        state,
        notes=[
            "Bounds condition on the two empirical marginal PMFs; no population confidence interval, covariance or sampling certainty is asserted.",
            "Individual-effect quantiles are not differences of marginal outcome quantiles.",
            "Each endpoint is attained by its own complete coupling; no single joint curve or vector is asserted to attain all endpoints.",
            "Exact integer capacities, primal feasible witnesses and dual min-cut certificates prove pointwise optimality; no potential-outcome dependence restriction is imposed.",
        ],
    )
    validate_effect_distribution_result(result)
    return result


@m.procedure
def treatment_effect_cdf_bounds(
    data,
    y,
    treatment,
    *,
    support,
    thresholds,
    design,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Sharp empirical bounds for P(Y(1)-Y(0)<=threshold), with exact witnesses."""
    return _run(
        data,
        y,
        treatment,
        support=support,
        targets=thresholds,
        quantiles=False,
        design=design,
        missing=missing,
        device=device,
        weights=weights,
        max_work=max_work,
    )


@m.procedure
def treatment_effect_quantile_bounds(
    data,
    y,
    treatment,
    *,
    support,
    quantiles,
    design,
    missing="raise",
    device="cpu",
    weights=None,
    max_work=100_000_000,
):
    """Sharp empirical left-quantile bounds for individual effects, with witnesses."""
    return _run(
        data,
        y,
        treatment,
        support=support,
        targets=quantiles,
        quantiles=True,
        design=design,
        missing=missing,
        device=device,
        weights=weights,
        max_work=max_work,
    )
