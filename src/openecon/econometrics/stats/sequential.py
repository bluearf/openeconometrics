"""Prospective known-information Gaussian efficacy designs and portable replay.

Information is the reciprocal terminal variance, not an integer sample size.
All numerical work is cumulative within an operation. Saved calibration work
is a reproducible admission plan; restoration performs fixed-boundary walks,
never a boundary or information search.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Integral, Real

from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.core import TableSet, table
from openecon.econometrics.resident_cpu import resident_cpu
from . import sequential_kernels as k

SCHEMA = "openecon.sequential.design.v1"
LIMIT = 1024 * 1024
NOTES = (
    "Known-information canonical Gaussian independent increments; prespecified efficacy looks. "
    "Alpha is total across looks and directions. Two-sided lower exits are opposite-direction "
    "rejections, not futility. Information is continuous reciprocal variance, not integer N. "
    "Refinement is an empirical numerical admission check, not a rigorous quadrature error bound."
)


def _fail(code, message):
    raise AnalysisError(code, message)


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        _fail("invalid_sequential_input", f"{name} must be a finite real number.")
    try:
        value = float(value)
    except (OverflowError,TypeError,ValueError) as exc:
        raise AnalysisError("invalid_sequential_input", f"{name} must be a finite real number.") from exc
    if not math.isfinite(value):
        _fail("invalid_sequential_input", f"{name} must be a finite real number.")
    return value


def _information(value):
    value = _real(value, "information")
    if not 1e-12 <= value <= 1e6:
        _fail("unsupported_information", "Information must be in [1e-12,1e6].")
    return value


def _effect(value):
    value = _real(value, "effect")
    if abs(value) > 1e6:
        _fail("unsupported_effect", "Absolute effect must be <=1e6.")
    return value


def _drift(effect, information):
    drift = effect * math.sqrt(information)
    if abs(drift) > 8:
        _fail("unsupported_drift", "Canonical standardized drift must satisfy |effect*sqrt(information)|<=8.")
    return drift


class _Work:
    def __init__(self, limit):
        if isinstance(limit, bool) or not isinstance(limit, Integral) or not 1 <= limit <= k.MAX_WORK:
            _fail("invalid_resource_budget", "max_work must be an integer in [1,2000000000].")
        self.limit, self.used, self.prepared = int(limit), 0, set()

    def debit(self, amount):
        if self.used + amount > self.limit:
            _fail("resource_limit", f"Cumulative sequential work exceeds max_work={self.limit}.")
        self.used += amount

    def probability(self, settings, bounds, drift, order):
        receipt = k.probability_work_bound(settings["fractions"], bounds, drift, settings["sides"], order)
        cost = receipt["integration_work_bound"]
        if order not in self.prepared:
            cost += receipt["preparation_work_bound"]
        self.debit(cost)
        self.prepared.add(order)
        return k.probabilities(settings["fractions"], bounds, drift, settings["sides"], order,
                               max_work=k.MAX_WORK)


def _settings(fractions, spending, sides, alpha, param, effect, information):
    fractions = list(k._fractions(fractions))
    sides = k._sides(sides)
    spending, param = k._law(spending, param)
    alpha = _real(alpha, "alpha")
    k.spend(fractions, alpha, spending, param, sides)
    effect, information = _effect(effect), _information(information)
    _drift(effect, information)
    return {"fractions": fractions, "spending": spending, "sides": sides,
            "alpha": alpha, "param": param, "effect": effect, "information": information}


def _pair(settings, bounds, drift, work):
    coarse = work.probability(settings, bounds, drift, 128)
    fine = work.probability(settings, bounds, drift, 256)
    error = k.probability_refinement_error(coarse, fine)
    if error > k.DEFAULT_TOLERANCE:
        _fail("numerical_failure", "Fixed-boundary probability refinement failed.")
    return {"coarse": coarse, "fine": fine, "refinement_error": error}


def _encode(state):
    return json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _seal(state):
    state = copy.deepcopy(state)
    state.pop("checksum", None)
    state["checksum"] = hashlib.sha256(_encode(state).encode()).hexdigest()
    if len(_encode(state).encode()) > LIMIT:
        _fail("resource_limit", "Sequential state exceeds 1 MiB.")
    return state


def _same(left, right):
    if isinstance(right, dict):
        if set(right) == {"operation","estimated_workspace_bytes","budget_bytes","buffers","scope"}:
            if not isinstance(left,dict) or not isinstance(left.get("budget_bytes"),Integral) \
                    or isinstance(left.get("budget_bytes"),bool) \
                    or left["budget_bytes"] < right["estimated_workspace_bytes"]:
                return False
            # The previous admission budget is provenance. The numerical walk
            # has separately admitted its buffers against today's budget.
            right = {**right,"budget_bytes":left["budget_bytes"]}
        return isinstance(left, dict) and set(left) == set(right) and all(_same(left[x], right[x]) for x in right)
    if isinstance(right, (list, tuple)):
        return isinstance(left, (list, tuple)) and len(left) == len(right) and all(_same(a,b) for a,b in zip(left,right))
    if isinstance(right, bool) or right is None or isinstance(right, str):
        return type(left) is type(right) and left == right
    if isinstance(right,int):
        return isinstance(left,Integral) and not isinstance(left,bool) and left == right
    if isinstance(right,float):
        return isinstance(left, Real) and not isinstance(left, bool) and math.isfinite(float(left)) \
            and math.isclose(float(left), float(right), rel_tol=2e-10, abs_tol=2e-12)
    return type(left) is type(right) and left == right


def _bounded_state(state):
    if isinstance(state, str):
        if len(state.encode()) > LIMIT:
            _fail("invalid_state", "Sequential JSON exceeds 1 MiB.")
        try:
            state = json.loads(state, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
        except (ValueError, RecursionError) as exc:
            raise AnalysisError("invalid_state", "Sequential JSON is invalid or nonfinite.") from exc
    if not isinstance(state, dict):
        _fail("invalid_state", "Supply sequential state, its JSON, or a complete sequential TableSet.")
    pending, count = [(state, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 14 or count > 20000:
            _fail("invalid_state", "Sequential state is too deep or large.")
        if isinstance(item, dict):
            if len(item) > 100 or any(not isinstance(key, str) for key in item):
                _fail("invalid_state", "Sequential state keys are invalid.")
            pending.extend((v, depth+1) for v in item.values())
        elif isinstance(item, (list, tuple)):
            if len(item) > 100:
                _fail("invalid_state", "Sequential state arrays are too large.")
            pending.extend((v, depth+1) for v in item)
        elif item is not None and type(item) not in (bool, int, float, str):
            _fail("invalid_state", "Sequential state must contain finite JSON primitives.")
        elif isinstance(item, float) and not math.isfinite(item):
            _fail("invalid_state", "Sequential state numbers must be finite.")
        elif isinstance(item, str) and len(item) > 4096:
            _fail("invalid_state", "Sequential state strings are too large.")
    try:
        encoded = _encode(state)
    except (ValueError, TypeError, OverflowError) as exc:
        raise AnalysisError("invalid_state", "Sequential state is not finite JSON.") from exc
    if len(encoded.encode()) > LIMIT:
        _fail("invalid_state", "Sequential state exceeds 1 MiB.")
    if set(state) != {"schema", "settings", "calibration", "alternative", "inverse", "checksum"} \
            or state["schema"] != SCHEMA or not isinstance(state["checksum"], str):
        _fail("invalid_state", "Unknown sequential state schema or fields.")
    if _seal(state)["checksum"] != state["checksum"]:
        _fail("invalid_state", "Sequential state checksum is invalid.")
    return copy.deepcopy(state)


def _calibration_replay(state, work):
    settings, saved = state["settings"], state["calibration"]
    if not isinstance(saved, dict) or set(saved) != {
        "bounds", "bounds_coarse", "spending", "probabilities", "probabilities_coarse",
        "probabilities_fine_bounds_coarse", "order", "refine_order", "tolerance",
        "max_refinement_error", "boundary_refinement_error", "probability_refinement_error",
        "calibration_error", "numerical_receipt",
    } or saved["order"] != 128 or saved["refine_order"] != 256 or saved["tolerance"] != k.DEFAULT_TOLERANCE:
        _fail("invalid_state", "Saved calibration fields or numerical protocol are invalid.")
    for name in ("bounds", "bounds_coarse"):
        if not isinstance(saved[name], list) or len(saved[name]) != len(settings["fractions"]):
            _fail("invalid_state", "Saved boundaries have invalid dimensions.")
        k._settings(settings["fractions"], saved[name], 0., settings["sides"], 128)
    targets = k.spend(settings["fractions"], settings["alpha"], settings["spending"],
                      settings["param"], settings["sides"]).tolist()
    coarse = work.probability(settings, saved["bounds_coarse"], 0., 128)
    fine = work.probability(settings, saved["bounds"], 0., 256)
    fine_coarse = work.probability(settings, saved["bounds"], 0., 128)
    boundary_error = max(abs(a-b) for a,b in zip(saved["bounds"], saved["bounds_coarse"]))
    probability_error = max(k.probability_refinement_error(coarse, fine),
                            k.probability_refinement_error(fine_coarse, fine))
    calibration_error = max(abs(math.fsum(p["upper"][:i+1]+p["lower"][:i+1])-target)
                            for p in (coarse,fine) for i,target in enumerate(targets))
    if max(boundary_error, probability_error) > k.DEFAULT_TOLERANCE \
            or calibration_error > k.CALIBRATION_TOLERANCE:
        _fail("invalid_state", "Saved boundaries fail refinement or spending calibration.")
    receipt = saved["numerical_receipt"]
    if not isinstance(receipt, dict) or isinstance(receipt.get("max_work"), bool):
        _fail("invalid_state", "Saved calibration work protocol is invalid.")
    expected_receipt = k.calibration_work_bound(settings["fractions"], settings["sides"],
                                               max_work=receipt.get("max_work"))
    expected = {"bounds": saved["bounds"], "bounds_coarse": saved["bounds_coarse"],
        "spending": targets, "probabilities": fine, "probabilities_coarse": coarse,
        "probabilities_fine_bounds_coarse": fine_coarse, "order": 128, "refine_order": 256,
        "tolerance": k.DEFAULT_TOLERANCE, "max_refinement_error": max(boundary_error, probability_error),
        "boundary_refinement_error": boundary_error, "probability_refinement_error": probability_error,
        "calibration_error": calibration_error, "numerical_receipt": expected_receipt}
    if not _same(saved, expected):
        _fail("invalid_state", "Saved calibration receipts are inconsistent with fixed-boundary replay.")
    return expected


def _inverse_replay(state, work):
    inverse = state["inverse"]
    if inverse is None:
        return None
    keys = {"effect", "target_power", "max_information", "drift_low", "drift_high",
            "information", "power_low", "power_high", "coarse_power", "refinement_error"}
    if not isinstance(inverse, dict) or set(inverse) != keys:
        _fail("invalid_state", "Saved information inversion fields are invalid.")
    values = {key: _real(value,key) for key,value in inverse.items()}
    effect, target, max_information = values["effect"], values["target_power"], values["max_information"]
    settings = state["settings"]
    if effect != settings["effect"] or effect == 0 or settings["sides"] == 1 and effect < 0 \
            or not settings["alpha"] < target < 1:
        _fail("invalid_state", "Saved information inversion direction or target is invalid.")
    _information(max_information)
    lo, hi = values["drift_low"], values["drift_high"]
    if not 0 <= lo < hi <= min(8., abs(effect)*math.sqrt(max_information)) \
            or hi-lo > 2.1e-6 or not math.isclose((hi/abs(effect))**2, settings["information"], rel_tol=2e-12):
        _fail("invalid_state", "Saved continuous-information bracket or information is invalid.")
    sign = 1. if effect > 0 else -1.
    low = work.probability(settings,state["calibration"]["bounds"],sign*lo,256)
    high = state["alternative"]["fine"]
    coarse = work.probability(settings,state["calibration"]["bounds"],sign*hi,64)
    refinement = k.probability_refinement_error(coarse,high)
    if not low["power"] < target <= high["power"] or high["power"]-target > 2e-6 \
            or refinement > k.DEFAULT_TOLERANCE:
        _fail("invalid_state", "Saved inverse bracket or numerical probability refinement is invalid.")
    expected = {"effect": effect, "target_power": target, "max_information": max_information,
                "drift_low": lo, "drift_high": hi, "information": settings["information"],
                "power_low": low["power"], "power_high": high["power"],
                "coarse_power": coarse["power"], "refinement_error": refinement}
    if not _same(inverse, expected):
        _fail("invalid_state", "Saved information inverse receipts are inconsistent.")
    return expected


def _render(state):
    s, c, alt = state["settings"], state["calibration"], state["alternative"]
    null, alternative = c["probabilities"], alt["fine"]
    fractions, bounds = s["fractions"], c["bounds"]
    info, drift, sides = s["information"], _drift(s["effect"],s["information"]), s["sides"]
    labels = [f"look_{i+1}" for i in range(len(fractions))]
    summary = table([[s["spending"],sides,s["alpha"],s["alpha"]/sides,s["param"],
        s["effect"],info,drift,null["power"],alternative["power"],
        info*null["expectedfraction"],info*alternative["expectedfraction"]]], columns=[
        "spending", "sides", "total_alpha", "per_direction_alpha", "parameter", "effect",
        "maximum_information", "standardized_drift", "null_rejection_probability",
        "rejection_probability", "null_expected_information", "expected_information"])
    boundaries = table([[i+1,t,info*t,b,-b if sides==2 else None,sides==2,a,
        a-(c["spending"][i-1] if i else 0.)] for i,(t,b,a) in enumerate(zip(fractions,bounds,c["spending"]))],
        columns=["look","fraction","information","upper","lower","lower_present",
                 "cumulative_total_alpha","incremental_total_alpha"],index=labels)
    def stages(p):
        return table([[p["reach"][i],p["upper"][i],p["lower"][i],p["continuation"][i]]
                      for i in range(len(fractions))],index=labels,
                     columns=["reach_probability","upper_first_crossing","lower_first_crossing","continuation_probability"])
    covariance = [[math.sqrt(min(a,b)/max(a,b)) for b in fractions] for a in fractions]
    means = [[0.,drift*math.sqrt(t)] for t in fractions]
    accuracy = table([[128,256,c["boundary_refinement_error"],c["probability_refinement_error"],
        alt["refinement_error"],c["calibration_error"],null["error"],alternative["error"],
        null["tail_bound"],k.DEFAULT_TOLERANCE,k.CALIBRATION_TOLERANCE,k.CONSERVATION_TOLERANCE,False]],
        columns=["coarse_nodes_per_panel","fine_nodes_per_panel","boundary_refinement_error",
        "null_probability_refinement_error","alternative_refinement_error","calibration_error",
        "null_mass_error","alternative_mass_error","omitted_probability_bound","refinement_tolerance",
        "calibration_tolerance","conservation_tolerance","rigorous_quadrature_error_bound"])
    tables = {"summary":summary,"boundaries":boundaries,"null_stages":stages(null),
        "alternative_stages":stages(alternative),"canonical_covariance":table(covariance,index=labels,columns=labels),
        "canonical_means":table(means,index=labels,columns=["null_mean","alternative_mean"]),"numerical_accuracy":accuracy}
    if state["inverse"] is not None:
        inv = state["inverse"]
        fields = ["effect","target_power","max_information","drift_low","drift_high",
                  "information","power_low","power_high","coarse_power","refinement_error"]
        tables["information_inversion"] = table([[inv[x] for x in fields]], columns=fields)
    return TableSet(tables,title="Canonical Gaussian group-sequential efficacy design",
                    procedure="sequential_design",state=state,notes=NOTES)


def _tables_same(output, expected):
    if output.title != expected.title or set(output) != set(expected) or not _same(output.attrs,expected.attrs):
        return False
    for name in expected:
        left,right = output[name],expected[name]
        if not hasattr(left,"to_numpy") or not _same(list(left.columns),list(right.columns)) \
                or not _same(list(left.index),list(right.index)) or not _same(list(left.index.names),list(right.index.names)) \
                or not _same(list(left.columns.names),list(right.columns.names)) or not _same(left.attrs,right.attrs) \
                or not _same(left.to_numpy().tolist(),right.to_numpy().tolist()):
            return False
    return True


def _load(value, work):
    output = value if isinstance(value,TableSet) else None
    state = _bounded_state(output.attrs.get("state") if output is not None else value)
    try:
        settings = state["settings"]
        if not isinstance(settings,dict) or set(settings) != {"fractions","spending","sides","alpha","param","effect","information"}:
            _fail("invalid_state","Sequential settings are invalid.")
        canonical = _settings(**settings)
        if not _same(settings,canonical):
            _fail("invalid_state","Sequential primitive settings are inconsistent.")
        _calibration_replay(state,work)
        expected_alt = _pair(canonical,state["calibration"]["bounds"],_drift(canonical["effect"],canonical["information"]),work)
        if not _same(state["alternative"],expected_alt):
            _fail("invalid_state","Saved alternative probabilities are inconsistent.")
        _inverse_replay(state,work)
        expected = _render(state)
        if output is not None and not _tables_same(output,expected):
            _fail("invalid_state","Complete sequential tables, labels or metadata were altered.")
        return expected.attrs["state"]
    except AnalysisError as exc:
        if exc.code in ("resource_limit","workspace_limit","invalid_resource_budget"):
            raise
        raise AnalysisError("invalid_state","Sequential state failed semantic replay.") from exc
    except (TypeError,KeyError,ValueError,OverflowError,AttributeError) as exc:
        raise AnalysisError("invalid_state","Sequential state fields or dimensions are invalid.") from exc


@resident_cpu
def sequential_design(*,fractions,spending="ldof",sides=2,alpha=.05,param=None,
                      effect=.3,information=100.,max_work=2_000_000_000) -> TableSet:
    """Calibrate 2..6 prespecified Gaussian efficacy looks; alpha is TOTAL.

    Four spending laws (ldof, ldpocock, hsd, power), one upper or symmetric
    two-sided stopping. Known information is continuous reciprocal variance.
    Unknown variance, adaptive looks, futility and integer N are unsupported.
    """
    work = _Work(max_work)
    settings = _settings(fractions,spending,sides,alpha,param,effect,information)
    receipt = k.calibration_work_bound(settings["fractions"],settings["sides"],max_work=max_work)
    work.debit(receipt["work_bound"])
    work.prepared.update((128,256))
    calibration = k.calibrate(settings["fractions"],settings["alpha"],settings["spending"],settings["param"],
                              settings["sides"],max_work=max_work)
    alternative = _pair(settings,calibration["bounds"],_drift(settings["effect"],settings["information"]),work)
    return _render(_seal({"schema":SCHEMA,"settings":settings,"calibration":calibration,
                          "alternative":alternative,"inverse":None}))


@resident_cpu
def restore_sequential_design(state,*,max_work=2_000_000_000) -> TableSet:
    """Validate complete primitive/derived state and tables without any search."""
    return _render(_load(state,_Work(max_work)))


@resident_cpu
def sequential_power(result,effect,information=None,*,max_work=2_000_000_000) -> TableSet:
    """Evaluate saved efficacy boundaries at new signed effect/information."""
    work = _Work(max_work)
    state = _load(result,work)
    settings = state["settings"]
    settings["effect"] = _effect(effect)
    if information is not None:
        settings["information"] = _information(information)
    drift = _drift(settings["effect"],settings["information"])
    state["alternative"] = _pair(settings,state["calibration"]["bounds"],drift,work)
    state["inverse"] = None
    return _render(_seal(state))


@resident_cpu
def sequential_information(result,effect,*,power=.8,max_information=1e6,max_work=2_000_000_000) -> TableSet:
    """Invert continuous information for fixed boundaries with verified bracket.

    The target must exceed total alpha. One-sided upper inversion requires a
    positive effect; symmetric inversion admits either nonzero direction.
    An explicit work budget is checked before every quadrature evaluation.
    """
    work = _Work(max_work)
    state = _load(result,work)
    settings,bounds = state["settings"],state["calibration"]["bounds"]
    effect,target,max_info = _effect(effect),_real(power,"power"),_information(max_information)
    if effect == 0 or settings["sides"] == 1 and effect < 0:
        _fail("unsupported_effect","Information inversion requires a nonzero effect in a rejection direction.")
    if not settings["alpha"] < target < 1:
        _fail("unsupported_power","Target power must be between total alpha and 1.")
    sign = 1. if effect > 0 else -1.
    lo,hi = 0.,min(8.,abs(effect)*math.sqrt(max_info))
    ceiling = work.probability(settings,bounds,sign*hi,256)
    if ceiling["power"] < target:
        _fail("unattainable_power","Target power is unattainable within maximum information and drift limits.")
    for _ in range(32):
        mid = (lo+hi)/2
        probability = work.probability(settings,bounds,sign*mid,64)["power"]
        if probability < target:
            lo = mid
        else:
            hi = mid
    lo,hi = max(0.,lo-1e-6),min(hi+1e-6,abs(effect)*math.sqrt(max_info),8.)
    lower = work.probability(settings,bounds,sign*lo,256)
    upper = work.probability(settings,bounds,sign*hi,256)
    if not lower["power"] < target <= upper["power"]:
        _fail("numerical_failure","Independent fine quadrature did not confirm the information bracket.")
    for _ in range(8):
        mid = (lo+hi)/2
        current = work.probability(settings,bounds,sign*mid,256)
        if current["power"] < target:
            lo,lower = mid,current
        else:
            hi,upper = mid,current
    information = (hi/abs(effect))**2
    _information(information)
    settings["effect"],settings["information"] = effect,information
    coarse = work.probability(settings,bounds,sign*hi,128)
    check64 = work.probability(settings,bounds,sign*hi,64)
    refinement = k.probability_refinement_error(coarse,upper)
    inverse_refinement = k.probability_refinement_error(check64,upper)
    if max(refinement,inverse_refinement) > k.DEFAULT_TOLERANCE or upper["power"]-target > 2e-6:
        _fail("numerical_failure","Information inversion failed probability refinement or target accuracy.")
    state["alternative"] = {"coarse":coarse,"fine":upper,"refinement_error":refinement}
    state["inverse"] = {"effect":effect,"target_power":target,"max_information":max_info,
        "drift_low":lo,"drift_high":hi,"information":information,"power_low":lower["power"],
        "power_high":upper["power"],"coarse_power":check64["power"],"refinement_error":inverse_refinement}
    return _render(_seal(state))
