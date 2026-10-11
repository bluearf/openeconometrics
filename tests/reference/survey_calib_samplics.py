"""Reference g-weights for tests/test_survey_calib_calibration.py from samplics 0.6.

Run offline with the test-only reference environment (samplics 0.6.0, numpy 2.5.3):

    <refvenv>/bin/python -I tests/reference/survey_calib_samplics.py

samplics ``SampleWeight.calibrate`` is linear (GREG) calibration, with ``domain=``
calibrating separately within each domain; ``SampleWeight.rake`` is iterative
proportional fitting (raking ratio) of categorical margins. The fixture uses the
integer formulas of ``fixture()`` in the tests. Numbers are printed with 17
significant digits and hard-coded in the tests.
"""

import warnings

import numpy as np

warnings.simplefilter("ignore")
from samplics.weighting import SampleWeight  # noqa: E402

i = np.arange(36)
h = i // 12 + 1
d = 8 + (i * 7) % 9 + 0.25 * (i % 4)
sex = np.where((i * 5) % 7 < 3, "F", "M")
age = np.array(["a", "b", "c"])[(i + i // 3) % 3]
x = 10 + (i * 13) % 17 + 0.5 * (i % 3)
N = 460.0


def emit(name, values):
    print(f"{name} = [{', '.join(format(float(v), '.17g') for v in np.ravel(values))}]")


# Linear calibration on (1, sexM, ageb, agec, x).
aux = np.column_stack([np.ones(36), sex == "M", age == "b", age == "c", x]).astype(float)
control = {"one": N, "sexM": 0.53 * N, "ageb": 0.32 * N, "agec": 0.35 * N, "x": 8600.0}
emit("samplics_linear_g", SampleWeight().calibrate(d, aux, control) / d)

# Raking ratio (IPF) of the sex and age margins to the population counts.
margins = {"sex": sex, "age": age}
counts = {"sex": {"F": 0.47 * N, "M": 0.53 * N}, "age": {"a": 0.33 * N, "b": 0.32 * N, "c": 0.35 * N}}
raked = SampleWeight().rake(d, margins, control=counts, tol=1e-14, ctrl_tol=1e-14, max_iter=1000)
emit("samplics_rake_g", raked / d)

# Linear calibration within strata (domain=h) on (1, x).
within = np.column_stack([np.ones(36), x])
controls = {1: {"one": 150.0, "x": 2900.0}, 2: {"one": 145.0, "x": 2700.0}, 3: {"one": 152.0, "x": 2750.0}}
emit("samplics_domain_g", SampleWeight().calibrate(d, within, controls, domain=h) / d)
