"""svycal: R survey/sampling and samplics references, independent NumPy/SciPy oracles, failure contract."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.econometrics.survey_calib.calibration import distance

# R 4.6.1, survey 4.5 calibrate(epsilon=1e-14) and sampling 2.11 calib():
# tests/reference/survey_calib_calibrate.R
R_CAL = {
    "linear_g": [
        1.025853189512965, 0.96053411476547279, 1.0359017634359926, 1.0340944094383131, 0.93758424187254497,
        1.0274768819180553, 1.2077795796722806, 1.1010371765908957, 0.83908305871650835, 1.0274768819180555,
        1.1092783965162436, 1.0127682289504756, 0.83908305871650835, 1.2829635667502108, 1.176221163668826,
        1.0127682289504756, 0.90602582586909086, 0.98782734046727894, 1.0043433474325383, 0.91426704579443896,
        1.1615125107012463, 0.98782734046727882, 1.0879522160284061, 0.98120981294702136, 1.1615125107012463,
        0.88289229138357384, 0.98945103287236946, 1.153087629183309, 1.0630113275452093, 0.96650115997944153,
        0.98945103287236935, 1.2366964977791768, 0.95807627846150423, 0.96650115997944164, 1.0316365731343446,
        1.1381953146231398
    ],
    "raking_g": [
        1.0216944752617356, 0.95960125320748135, 1.0300494377992602, 1.0287245984135323, 0.93748803002844605,
        1.0241966201216703, 1.2132735383031408, 1.0979729895197974, 0.85423378095980529, 1.0241966201216703,
        1.1055279734416723, 1.0074797895972423, 0.85423378095980529, 1.303855121282413, 1.1799463684152809,
        1.0074797895972425, 0.91173635750100412, 0.98413725237780658, 1.0017552143429056, 0.91800987569102921,
        1.1606874067264896, 0.98413725237780636, 1.08269705205314, 0.97980552732539561, 1.1606874067264898,
        0.89175909419046695, 0.98654741904419363, 1.1540922943726999, 1.0576117883129836, 0.9638132435925606,
        0.98654741904419363, 1.2473429706419732, 0.95833678490719332, 0.96381324359256049, 1.0273690461567249,
        1.136571847144328
    ],
    "logit_g": [
        1.0237011852169586, 0.95852798938465544, 1.0372070215943634, 1.0322929307940734, 0.93593112576000614,
        1.028028298494992, 1.208916326061199, 1.1028141224655996, 0.84520173938098142, 1.028028298494992,
        1.1112880196102983, 1.0128613296818512, 0.84520173938098142, 1.2736965940857463, 1.1787496633222276,
        1.0128613296818512, 0.90511265882511671, 0.98391136156230197, 1.0037478166751672, 0.91287933835507429,
        1.1646312107574792, 0.98391136156230208, 1.092041163113981, 0.97973650249589261, 1.1646312107574792,
        0.883614820024476, 0.98816328102208739, 1.1559912136715433, 1.0628139564560386, 0.96495752768428456,
        0.98816328102208728, 1.2349775596071955, 0.95612839813962702, 0.96495752768428467, 1.0297915119421968,
        1.1409358150776747
    ],
    "truncated_g": [
        1.0244053749965616, 0.96092261551200697, 1.0380099210163349, 1.0340693573340378, 0.93681815134549595,
        1.0286405746875251, 1.2123484325092049, 1.1017873165095562, 0.83592101768332339, 1.0286405746875253,
        1.1114512988470324, 1.0142000928584904, 0.83592101768332339, 1.25, 1.1791692580225508,
        1.0142000928584904, 0.90363897685884176, 0.98644970101834883, 1.0048307465296809, 0.91330295919631799,
        1.1647287761935159, 0.98644970101834872, 1.091582034371485, 0.98102091837183647, 1.1647287761935161,
        0.87982914870099749, 0.9906849007093127, 1.1553594298647063, 1.0638316425313434, 0.96658043654280157,
        0.99068490070931259, 1.2421107177065105, 0.95721109021399187, 0.96658043654280168, 1.0303578320360227,
        1.141213584044338
    ],
    "by_raking_g": [
        1.1974578151162198, 0.9823778846174942, 1.043271542129008, 1.0988734338849602, 1.2073375101532229,
        0.95738101793798402, 1.0084053137027857, 1.0709123069555064, 1.1766164859414217, 0.95738101793798402,
        0.98274617216444271, 1.0797479309503832, 1.0077390907065704, 0.9763682908455249, 1.0030631228816833,
        0.96964545482075171, 0.99615647816487185, 1.1002530939287105, 0.93299179988673242, 0.95850067765612523,
        1.0586623278972183, 1.1002530939287107, 0.92226830744476185, 0.94748399479383538, 0.92336106788057681,
        0.92563110788600855, 1.1769116782294375, 1.0092607534130795, 0.94833353220058703, 0.95066496584173987,
        1.1769116782294373, 1.0365563900073813, 1.0391047154188082, 0.95066496584173987, 0.83729124566244495,
        1.064590241948479
    ],
    "sampling_linear_g": [
        1.0258531895129599, 0.96053411476547301, 1.0359017634359919, 1.0340944094383104, 0.93758424187254219,
        1.0274768819180549, 1.207779579672279, 1.1010371765908922, 0.83908305871650579, 1.0274768819180549,
        1.1092783965162425, 1.0127682289504742, 0.83908305871650579, 1.2829635667502111, 1.1762211636688242,
        1.0127682289504742, 0.90602582586908742, 0.98782734046727527, 1.0043433474325372, 0.91426704579443796,
        1.1615125107012438, 0.98782734046727527, 1.0879522160284065, 0.98120981294701959, 1.1615125107012438,
        0.88289229138356995, 0.98945103287237002, 1.1530876291833068, 1.0630113275452073, 0.96650115997943919,
        0.98945103287237002, 1.2366964977791759, 0.95807627846150201, 0.96650115997943919, 1.0316365731343393,
        1.1381953146231396
    ],
    "sampling_raking_g": [
        1.0216944757961162, 0.95960125322350265, 1.0300494392387072, 1.0287245993273959, 0.93748803040143214,
        1.0241966205475121, 1.2132735422689713, 1.0979729909333626, 0.85423377991926197, 1.0241966205475121,
        1.1055279752688272, 1.0074797907681909, 0.85423377991926197, 1.3038551265409926, 1.1799463708363302,
        1.0074797907681909, 0.91173635675427767, 0.98413725278912423, 1.0017552145238044, 0.91800987527452516,
        1.160687409974446, 0.98413725278912423, 1.0826970541391217, 0.97980552727188097, 1.160687409974446,
        0.89175909325036096, 0.98654741935070878, 1.154092296469259, 1.0576117895634463, 0.96381324425938086,
        0.98654741935070878, 1.2473429750858842, 0.95833678462945082, 0.96381324425938086, 1.0273690467544825,
        1.1365718493569421
    ],

}

# samplics 0.6.0 (linear calibrate, rake = iterative proportional fitting, domain calibration):
# tests/reference/survey_calib_samplics.py
SAMPLICS = {
    "samplics_linear_g": [
        1.0258531895129648, 0.96053411476547301, 1.0359017634359926, 1.0340944094383129, 0.93758424187254497,
        1.0274768819180553, 1.2077795796722801, 1.1010371765908953, 0.83908305871650823, 1.0274768819180553,
        1.1092783965162436, 1.0127682289504756, 0.83908305871650812, 1.2829635667502108, 1.1762211636688258,
        1.0127682289504756, 0.90602582586909053, 0.98782734046727871, 1.0043433474325383, 0.91426704579443885,
        1.1615125107012461, 0.98782734046727871, 1.0879522160284063, 0.98120981294702114, 1.1615125107012461,
        0.8828922913835735, 0.98945103287236924, 1.1530876291833088, 1.0630113275452093, 0.96650115997944142,
        0.98945103287236935, 1.2366964977791766, 0.95807627846150401, 0.96650115997944153, 1.031636573134344,
        1.1381953146231398
    ],
    "samplics_rake_g": [
        1.1166775419325095, 0.91665245599644063, 1.0169699370376726, 1.0653385000840727, 1.0169699370376728,
        0.96082626441053587, 1.1819280254658646, 1.1166775419325092, 0.91665245599644063, 0.96082626441053598,
        1.0653385000840729, 1.0169699370376726, 0.91665245599644074, 1.1819280254658644, 1.1166775419325092,
        1.0169699370376728, 0.96082626441053576, 1.0653385000840723, 0.96082626441053609, 0.91665245599644063,
        1.1819280254658642, 1.0653385000840727, 1.0169699370376726, 0.96082626441053576, 1.181928025465864,
        0.9608262644105362, 0.91665245599644052, 1.1166775419325092, 1.0653385000840727, 1.0169699370376728,
        0.9166524559964403, 1.1819280254658644, 0.96082626441053609, 1.0169699370376728, 1.1166775419325097,
        1.0653385000840727
    ],
    "samplics_domain_g": [
        0.83821014067448107, 1.0979412282816674, 1.0306035389020263, 0.9344068397882539, 0.86706915040861299,
        1.1268002380157989, 1.0306035389020263, 0.96326584952238559, 0.89592816014274468, 1.1268002380157989,
        1.0594625486361582, 0.99212485925651728, 0.93487712865528383, 1.037945580654442, 1.0112241301361418,
        0.97305062939571263, 0.94632917887741241, 0.9196077283591122, 1.0112241301361418, 0.98450267961784144,
        0.95778122909954111, 0.91960772835911209, 1.0226761803582705, 0.99595472983997002, 1.027345564415685,
        1.0335375426650613, 1.0096541979888958, 1.0184998812022903, 1.0246918594516665, 1.0308838377010427,
        1.0096541979888958, 1.015846176238272, 1.0220381544876482, 1.0308838377010427, 1.037075815950419,
        1.0131924712742535
    ],
}

N = 460.0
CATEGORICAL = {"sex": {"F": 0.47 * N, "M": 0.53 * N}, "age": {"a": 0.33 * N, "b": 0.32 * N, "c": 0.35 * N}}
SPEC = dict(population=N, categorical=CATEGORICAL, totals={"x": 8600.0}, tol=1e-13)
BOUNDS = {"linear": None, "raking": None, "logit": (0.6, 1.5), "truncated": (0.8, 1.25)}


def fixture():
    i = np.arange(36)
    frame = pd.DataFrame({
        "h": i // 12 + 1, "psu": (i // 3) % 4 + 1, "d": 8 + (i * 7) % 9 + 0.25 * (i % 4),
        "sex": np.where((i * 5) % 7 < 3, "F", "M"), "age": np.array(["a", "b", "c"])[(i + i // 3) % 3],
        "x": 10 + (i * 13) % 17 + 0.5 * (i % 3), "z": 5.0 + (i * 5) % 6,
    })
    frame["Nh"] = np.array([10, 12, 8])[frame.h - 1]
    frame["y"] = 3 + 0.9 * frame.x + (i * 11) % 7 - 2 * (frame.sex == "F")
    return frame


def design_matrix(frame):
    """Full-rank basis (1, sexM, ageb, agec, x) with the totals implied by SPEC."""
    x = np.column_stack([np.ones(len(frame)), frame.sex == "M", frame.age == "b", frame.age == "c",
                         frame.x]).astype(float)
    return x, np.array([N, 0.53 * N, 0.32 * N, 0.35 * N, 8600.0])


def g_weights(result):
    return result["weights"]["g_weight"].to_numpy()


@pytest.mark.parametrize("method", list(BOUNDS))
def test_r_survey_reference_and_calibration_equations(method):
    frame = fixture()
    result = oe.svycal(frame, "d", method=method, bounds=BOUNDS[method], **SPEC)
    g = g_weights(result)
    np.testing.assert_allclose(g, R_CAL[f"{method}_g"], rtol=1e-11, atol=0)
    if method in ("linear", "raking"):
        # sampling::calib stops at a 1e-6 relative misfit; its Newton raking is accurate to about 1e-9.
        np.testing.assert_allclose(g, R_CAL[f"sampling_{method}_g"], rtol=1e-12 if method == "linear" else 1e-8)
    x, totals = design_matrix(frame)
    w = frame.d.to_numpy() * g
    np.testing.assert_allclose(x.T @ w, totals, rtol=1e-12)
    np.testing.assert_allclose(result["weights"]["calibrated_weight"], w, rtol=1e-15)
    table = result["totals"]
    np.testing.assert_allclose(table.calibrated_total, table.target, rtol=1e-12)
    assert list(table.status) == ["active", "active", "redundant", "active", "active", "redundant", "active"]
    assert result.attrs["n_redundant"] == 2 and result.attrs["converged"] is True
    low, high = BOUNDS[method] or (-np.inf, np.inf)
    assert low - 1e-12 <= g.min() and g.max() <= high + 1e-12
    json.dumps(result.attrs)


def test_linear_closed_form_and_samplics():
    frame = fixture()
    x, totals = design_matrix(frame)
    d = frame.d.to_numpy()
    lam = np.linalg.solve(x.T @ (x * d[:, None]), totals - x.T @ d)
    g = g_weights(oe.svycal(frame, "d", method="linear", **SPEC))
    np.testing.assert_allclose(g, 1 + x @ lam, rtol=1e-13)
    np.testing.assert_allclose(g, SAMPLICS["samplics_linear_g"], rtol=1e-12)


def ipf(d, margins, counts, sweeps=500):
    """Raking ratio: alternate post-stratification to each categorical margin."""
    w = d.copy()
    for _ in range(sweeps):
        for labels, target in zip(margins, counts):
            for level, count in target.items():
                cell = labels == level
                w[cell] *= count / w[cell].sum()
    return w / d


def test_raking_equals_iterative_proportional_fitting_and_samplics():
    frame = fixture()
    result = oe.svycal(frame, "d", method="raking", categorical=CATEGORICAL, tol=1e-14)
    g = g_weights(result)
    expected = ipf(frame.d.to_numpy(), [frame.sex.to_numpy(), frame.age.to_numpy()],
                   [CATEGORICAL["sex"], CATEGORICAL["age"]])
    np.testing.assert_allclose(g, expected, rtol=1e-12)
    np.testing.assert_allclose(g, SAMPLICS["samplics_rake_g"], rtol=1e-10)
    # Within each cell of the two margins the raked weight factor is constant (multiplicative model).
    assert frame.assign(g=g).groupby(["sex", "age"]).g.nunique().max() == 1


def logit_f(u, low, high):
    """Deville–Särndal (1992) logit distance in its original form."""
    a = (high - low) / ((1 - low) * (high - 1))
    e = np.exp(a * u)
    return (low * (high - 1) + high * (1 - low) * e) / (high - 1 + (1 - low) * e)


def test_logit_root_finding_oracle():
    frame = fixture()
    x, totals = design_matrix(frame)
    d = frame.d.to_numpy()
    root = optimize.root(lambda lam: x.T @ (d * logit_f(x @ lam, 0.6, 1.5)) - totals, np.zeros(5),
                         method="hybr", options={"xtol": 1e-14})
    assert root.success
    g = g_weights(oe.svycal(frame, "d", method="logit", bounds=(0.6, 1.5), **SPEC))
    np.testing.assert_allclose(g, logit_f(x @ root.x, 0.6, 1.5), rtol=1e-11)


def test_truncated_dual_minimisation_oracle():
    frame = fixture()
    x, totals = design_matrix(frame)
    d = frame.d.to_numpy()
    low, high = 0.8, 1.25

    def primitive(u):  # integral of clip(1 + s, low, high) from 0 to u
        v = 1 + u
        inner = np.clip(v, low, high)
        return (inner ** 2 - 1) / 2 + low * np.minimum(v - low, 0) + high * np.maximum(v - high, 0)

    scale = np.abs(x).T @ d
    scaled_x = x / scale

    def hessian(m):
        g = 1 + x @ (m / scale)
        interior = (g > low) & (g < high)
        return scaled_x.T @ ((d * interior)[:, None] * scaled_x)

    # The same independent convex-dual minimisation needs a converged solution.
    # BFGS with its approximate Hessian can stop with precision loss on this
    # piecewise quadratic; exact derivatives let the trust-region solver certify
    # stationarity without weakening the objective or comparison tolerances.
    dual = optimize.minimize(lambda m: d @ primitive(x @ (m / scale)) - (m / scale) @ totals, np.zeros(5),
                             jac=lambda m: (x.T @ (d * np.clip(1 + x @ (m / scale), low, high)) - totals) / scale,
                             hess=hessian, method="trust-exact", options={"gtol": 1e-13, "maxiter": 10_000})
    assert dual.success and dual.status == 0, dual.message
    assert np.max(np.abs(dual.jac)) <= 1e-13
    expected = np.clip(1 + x @ (dual.x / scale), low, high)
    result = oe.svycal(frame, "d", method="truncated", bounds=(low, high), **SPEC)
    np.testing.assert_allclose(g_weights(result), expected, rtol=1e-8)
    assert result.attrs["n_at_bounds"] == int(np.sum((expected <= low + 1e-9) | (expected >= high - 1e-9)))
    # The reported multipliers reproduce g = clip(1 + x'lambda) on the kept constraint basis.
    kept = result["totals"].query("status == 'active'")
    basis = np.column_stack([np.ones(36), frame.sex == "F", frame.age == "a", frame.age == "b", frame.x])
    np.testing.assert_allclose(g_weights(result), np.clip(1 + basis @ kept.multiplier.to_numpy(), low, high),
                               rtol=1e-12)


@pytest.mark.parametrize("method", list(BOUNDS))
def test_analytic_distance_derivatives(method):
    low, high = BOUNDS[method] or (-np.inf, np.inf)
    import torch
    link = distance(method, low, high)
    u = torch.linspace(-1.5, 1.5, 61, dtype=torch.float64)
    u = u[(1 + u - low).abs().gt(1e-3) & (1 + u - high).abs().gt(1e-3)]  # away from truncation kinks
    g, slope = link(u)
    h = 1e-6
    numeric = (link(u + h)[0] - link(u - h)[0]) / (2 * h)
    np.testing.assert_allclose(slope.numpy(), numeric.numpy(), rtol=1e-8, atol=1e-10)
    assert float(link(torch.zeros(1, dtype=torch.float64))[0]) == pytest.approx(1.0, abs=1e-15)
    if method == "logit":
        np.testing.assert_allclose(g.numpy(), logit_f(u.numpy(), low, high), rtol=1e-13)


def test_stratum_wise_calibration_references_and_separate_fits():
    frame = fixture()
    by = oe.svycal(frame, "d", method="raking", by="h", tol=1e-13,
                   categorical={1: {"sex": {"F": 66, "M": 90}}, 2: {"sex": {"F": 63, "M": 85}},
                                3: {"sex": {"F": 68, "M": 80}}},
                   totals={1: {"x": 2900}, 2: {"x": 2700}, 3: {"x": 2750}})
    np.testing.assert_allclose(g_weights(by), R_CAL["by_raking_g"], rtol=1e-11)
    assert list(by["totals"].index[:3]) == ["h=1: sex[F]", "h=1: sex[M]", "h=1: x"]
    controls = {1: (150.0, 2900.0), 2: (145.0, 2700.0), 3: (152.0, 2750.0)}
    linear = oe.svycal(frame, "d", by="h", population={h: c[0] for h, c in controls.items()},
                       totals={h: {"x": c[1]} for h, c in controls.items()})
    np.testing.assert_allclose(g_weights(linear), SAMPLICS["samplics_domain_g"], rtol=1e-12)
    for h, (count, total) in controls.items():
        part = frame[frame.h == h]
        alone = oe.svycal(part, "d", population=count, totals={"x": total})
        np.testing.assert_allclose(g_weights(alone), g_weights(linear)[frame.h == h], rtol=1e-13)


@pytest.mark.parametrize("method,bounds", [("linear", None), ("raking", None), ("logit", (0.2, 3.0)),
                                           ("truncated", (0.2, 3.0))])
def test_single_margin_is_poststratification(method, bounds):
    frame = fixture()
    counts = {"a": 150.0, "b": 140.0, "c": 170.0}
    g = g_weights(oe.svycal(frame, "d", method=method, bounds=bounds, categorical={"age": counts}, tol=1e-13))
    expected = frame.age.map(counts) / frame.groupby("age").d.transform("sum")
    np.testing.assert_allclose(g, expected, rtol=1e-11)


def test_invariances_spec_round_trip_and_index_alignment():
    frame = fixture()
    base = oe.svycal(frame, "d", method="raking", **SPEC)
    scaled = oe.svycal(frame.assign(d=frame.d * 4), "d", method="raking", population=4 * N, tol=1e-13,
                       categorical={c: {k: 4 * v for k, v in m.items()} for c, m in CATEGORICAL.items()},
                       totals={"x": 4 * 8600.0})
    np.testing.assert_allclose(g_weights(scaled), g_weights(base), rtol=1e-11)
    order = np.random.default_rng(5).permutation(len(frame))
    shuffled = frame.iloc[order].set_index(pd.Index([f"r{k}" for k in order]))
    moved = oe.svycal(shuffled, "d", method="raking", **SPEC)
    np.testing.assert_allclose(g_weights(moved), g_weights(base)[order], rtol=1e-11)
    assert list(moved["weights"].index) == [f"r{k}" for k in order]
    # Integer-coded levels match float columns; JSON keeps level types in [level, count] pairs.
    coded = frame.assign(age=frame.age.map({"a": 1.0, "b": 2.0, "c": 3.0}))
    numeric = oe.svycal(coded, "d", method="raking", population=N, totals={"x": 8600.0}, tol=1e-13,
                        categorical={"sex": CATEGORICAL["sex"], "age": {1: 0.33 * N, 2: 0.32 * N, 3: 0.35 * N}})
    np.testing.assert_allclose(g_weights(numeric), g_weights(base), rtol=1e-12)
    spec = json.loads(json.dumps(numeric.attrs["calibration"]))
    assert spec["groups"][0]["categorical"]["age"][0] == [1, pytest.approx(0.33 * N)]
    design = oe.survey_design(coded, weights="d", psu="psu", strata="h", fpc="Nh")
    a = oe.survey_calibrated_total(coded, design, ["y"], calibration=spec)
    b = oe.survey_calibrated_total(coded, design, ["y"], calibration=numeric)
    np.testing.assert_allclose(a.estimate, b.estimate, rtol=1e-14)
    assert a.attrs["calibration"]["spec"] == spec


@pytest.mark.parametrize("bounds", [(0.8, 1.12), (0.9, 1.12), (0.95, 1.118), (0.6, 1.118)])
@pytest.mark.parametrize("method", ["truncated", "logit"])
def test_bounded_feasibility_agrees_with_linear_programme(method, bounds):
    """Jointly infeasible bounds pass the per-constraint check but must not return weights."""
    frame = fixture()
    d, female = frame.d.to_numpy(), (frame.sex == "F").to_numpy(float)
    targets = np.array([N, 0.47 * N, 8600.0])
    lp = optimize.linprog(np.zeros(len(frame)), A_eq=np.vstack([d, d * female, d * frame.x]), b_eq=targets,
                          bounds=[bounds] * len(frame), method="highs")
    spec = dict(population=N, categorical={"sex": CATEGORICAL["sex"]}, totals={"x": 8600.0})
    if lp.status == 0:
        g = g_weights(oe.svycal(frame, "d", method=method, bounds=bounds, **spec))
        assert bounds[0] <= g.min() and g.max() <= bounds[1]
    else:
        assert lp.status == 2
        with pytest.raises(AnalysisError) as error:
            oe.svycal(frame, "d", method=method, bounds=bounds, **spec)
        assert error.value.code == "calibration_not_converged"


def test_negative_linear_weights_are_reported():
    frame = fixture()
    result = oe.svycal(frame, "d", population=N, totals={"x": 13000.0})
    assert result.attrs["n_negative_weights"] > 0 and result.attrs["g_min"] < 0
    assert "negative" in result.attrs["notes"][0]
    assert "negative" in str(result)


def test_redundant_duplicate_columns():
    frame = fixture().assign(x2=lambda f: 2 * f.x)
    same = oe.svycal(frame, "d", method="raking", population=N, totals={"x": 8600.0, "x2": 17200.0})
    assert list(same["totals"].status) == ["active", "active", "redundant"]
    with pytest.raises(AnalysisError) as error:
        oe.svycal(frame, "d", population=N, totals={"x": 8600.0, "x2": 17000.0})
    assert error.value.code == "calibration_infeasible" and "17200" in str(error.value)


@pytest.mark.parametrize("change,code", [
    (dict(categorical={"sex": {"F": 200.0, "M": 250.0}}), "calibration_infeasible"),          # margins sum to 450 != N
    (dict(categorical={"sex": {"F": 216.2, "M": 243.8, "X": 5.0}}), "calibration_infeasible"),  # absent level
    (dict(categorical={"sex": {"F": 216.2}}), "calibration_level_missing"),
    (dict(method="logit", bounds=(0.6, 1.5), totals={"x": 12500.0}), "calibration_infeasible"),  # > 1.5 sum d x
    (dict(method="truncated", bounds=(0.97, 1.03), totals={"x": 9300.0}), "calibration_infeasible"),
    (dict(method="raking", max_iter=1), "calibration_not_converged"),
    (dict(method="raking", totals={"x": 0.0}), "calibration_infeasible"),
    (dict(method="linear", bounds=(0.5, 2.0)), "invalid_calibration_spec"),
    (dict(method="logit"), "invalid_calibration_spec"),
    (dict(method="logit", bounds=(1.0, 2.0)), "invalid_calibration_spec"),
    (dict(method="truncated", bounds=(None, None)), "invalid_calibration_spec"),
    (dict(method="ipf"), "invalid_calibration_spec"),
    (dict(population=None, categorical=None, totals=None), "invalid_calibration_spec"),
    (dict(categorical={"sex": {"F": -1.0, "M": 243.8}}), "invalid_calibration_spec"),
    (dict(categorical={"sex": [["F", 1.0], ["F", 2.0]]}), "invalid_calibration_spec"),
    (dict(totals={"sex": 3.0}), "non_numeric_column"),
    (dict(totals={"absent": 3.0}), "missing_columns"),
    (dict(max_iter=0), "invalid_calibration_spec"),
    (dict(tol=0.1), "invalid_calibration_spec"),
])
def test_failure_contract(change, code):
    frame = fixture()
    spec = {**SPEC, "categorical": {"sex": CATEGORICAL["sex"]}, **change}
    with pytest.raises(AnalysisError) as error:
        oe.svycal(frame, "d", **spec)
    assert error.value.code == code


def test_input_failures():
    frame = fixture()
    with pytest.raises(AnalysisError, match="missing values") as error:
        oe.svycal(frame.assign(x=frame.x.where(frame.index != 3)), "d", **SPEC)
    assert error.value.code == "calibration_missing"
    for weights in (frame.d.where(frame.index != 0, 0.0), -frame.d):
        with pytest.raises(AnalysisError) as error:
            oe.svycal(frame.assign(d=weights), "d", **SPEC)
        assert error.value.code == "invalid_survey_weights"
    with pytest.raises(AnalysisError) as error:
        oe.survey_design(frame.assign(d=frame.d.where(frame.index != 0, 0.0)), weights="d", psu="psu", strata="h")
    assert error.value.code == "invalid_survey_weights"
    with pytest.raises(AnalysisError) as error:
        oe.svycal(frame, 3, **SPEC)
    assert error.value.code == "invalid_survey_design"
    groups = {1: {"x": 2900.0}, 2: {"x": 2700.0}}
    with pytest.raises(AnalysisError) as error:
        oe.svycal(frame, "d", by="h", totals=groups)
    assert error.value.code == "calibration_level_missing"
    with pytest.raises(AnalysisError) as error:
        oe.svycal(frame, "d", by="h", totals={**groups, 3: {"x": 1.0}, 4: {"x": 1.0}})
    assert error.value.code == "calibration_infeasible"
    with pytest.raises(AnalysisError) as error:
        oe.svycal(frame, "d", by="h", totals={**groups, 3: {"x": 1.0}}, population={1: 150.0})
    assert error.value.code == "invalid_calibration_spec"
    design = oe.survey_design(frame, weights="d", psu="psu", strata="h", fpc="Nh")
    with pytest.raises(AnalysisError) as error:
        oe.svycal(frame.assign(d=frame.d * 2), design, **SPEC)
    assert error.value.code == "survey_design_changed"
