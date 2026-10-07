"""Adversarial inputs for the unit-root family.

Every call must either return a result whose numbers are finite (or explicitly
missing) or raise ``AnalysisError`` with a code and a message that says what
to change. Nothing here may surface as a raw exception, a NaN or an infinity,
and no input table may be modified.
"""

import json
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import openecon as oe
from openecon.analysis_contracts import AnalysisError

T = 60


def make_walk(seed=0, total=T):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"t": np.arange(total), "y": rng.standard_normal(total).cumsum(),
                         "x": rng.standard_normal(total).cumsum(),
                         "z": rng.standard_normal(total)})


def make_panel(n, periods, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"id": np.repeat(np.arange(n), periods),
                         "year": np.tile(np.arange(2000, 2000 + periods), n),
                         "y": rng.standard_normal((n, periods)).cumsum(axis=1).ravel(),
                         "x": rng.standard_normal((n, periods)).cumsum(axis=1).ravel()})


def rows_of(data):
    if isinstance(data, pd.DataFrame):
        return len(data)
    if isinstance(data, dict):
        return len(next(iter(data.values())))
    return len(data) if data is not None else 2


SERIES = {
    "dfuller": lambda d, **k: oe.dfuller(d, "y", **k),
    "pperron": lambda d, **k: oe.pperron(d, "y", **k),
    "dfgls": lambda d, **k: oe.dfgls(d, "y", **k),
    "kpss": lambda d, **k: oe.kpss(d, "y", **k),
    "zandrews": lambda d, **k: oe.zandrews(d, "y", **k),
    "egranger": lambda d, **k: oe.egranger(d, "y", ["x"], **k),
    "chow": lambda d, **k: oe.chow(d, "y", ["x"], rows_of(d) // 2, **k),
    "sbsingle": lambda d, **k: oe.sbsingle(d, "y", ["x"], **k),
    "cusum": lambda d, **k: oe.cusum(d, "y", ["x"], **k),
}
PANEL_TESTS = ("llc", "ips", "fisher", "hadri", "breitung", "ht")


def code_of(call):
    """The AnalysisError code raised by ``call``; any other outcome fails the test."""
    with pytest.raises(AnalysisError) as caught:
        call()
    message = str(caught.value)
    assert caught.value.code and caught.value.code == caught.value.code.lower()
    assert len(message) > 15 and "Traceback" not in message
    return caught.value.code


def _walk_numbers(value, path, out):
    if isinstance(value, dict):
        for key, item in value.items():
            _walk_numbers(item, f"{path}.{key}", out)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_numbers(item, f"{path}[{index}]", out)
    elif isinstance(value, float) and not math.isfinite(value):
        out.append(path)


def assert_clean(result):
    """attrs are JSON-safe and finite; numeric table cells are finite or missing."""
    attrs = dict(result.attrs)
    json.dumps(attrs, allow_nan=False)
    bad = []
    _walk_numbers(attrs, "attrs", bad)
    assert not bad, bad
    tables = [result] if isinstance(result, pd.DataFrame) else list(result.values())
    for table in tables:
        numbers = table.select_dtypes("number").to_numpy(dtype=float)
        assert not np.isinf(numbers).any()
    p_value = attrs.get("p_value")
    assert p_value is None or 0.0 <= p_value <= 1.0
    return result


# ---- degenerate samples ---------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SERIES))
def test_degenerate_samples(name):
    run = SERIES[name]
    walk = make_walk()
    assert code_of(lambda: run(walk.iloc[:0])) == "empty_data"
    assert code_of(lambda: run(walk.iloc[:1])) == "insufficient_observations"
    assert code_of(lambda: run(walk.iloc[:2])) == "insufficient_observations"
    assert code_of(lambda: run(walk.iloc[:4])) in {"insufficient_observations",
                                                   "singular_subsample"}
    assert code_of(lambda: run(walk.assign(y=np.nan))) == "missing_values"
    assert code_of(lambda: run(walk.assign(y=walk["y"].where(walk.index != 10)))) \
        == "missing_values"
    assert code_of(lambda: run(walk.assign(y="a"))) == "non_numeric_column"
    assert code_of(lambda: run(walk.assign(y=walk["y"].astype(str)))) == "non_numeric_column"
    assert code_of(lambda: run(walk.assign(y=walk["y"].where(walk.index != 5, np.inf)))) \
        == "non_finite_values"
    assert code_of(lambda: run(walk.assign(y=3.0))) in {"constant_series", "perfect_fit"}
    assert code_of(lambda: run(walk[["t", "x"]])) == "missing_columns"
    assert code_of(lambda: run(None)) == "invalid_data"
    assert code_of(lambda: run(pd.concat([walk, walk[["y"]]], axis=1))) == "duplicate_columns"
    # The smallest samples that are accepted give finite results.
    for rows in (13, 14, 20):
        assert_clean(run(walk.iloc[:rows]))


@pytest.mark.parametrize("name", list(SERIES))
def test_inputs_are_not_modified_and_results_are_clean(name):
    walk = make_walk(3)
    before = walk.copy(deep=True)
    result = assert_clean(SERIES[name](walk))
    pd.testing.assert_frame_equal(walk, before)
    # Mappings and records are accepted like frames.
    again = SERIES[name](walk.to_dict("list"))
    assert again.attrs["statistic"] == pytest.approx(result.attrs["statistic"], rel=1e-12)
    records = SERIES[name](walk.to_dict("records"))
    assert records.attrs["statistic"] == pytest.approx(result.attrs["statistic"], rel=1e-12)


def test_deterministic_series():
    walk = make_walk()
    trend = walk.assign(y=np.arange(T) * 2.0 + 1)
    assert code_of(lambda: oe.dfuller(trend, "y")) == "perfect_fit"
    assert code_of(lambda: oe.dfuller(trend, "y", trend="trend")) in {"perfect_fit",
                                                                      "collinear_regressors"}
    assert code_of(lambda: oe.pperron(trend, "y")) == "perfect_fit"
    assert code_of(lambda: oe.dfgls(trend, "y")) == "perfect_fit"
    assert code_of(lambda: oe.zandrews(trend, "y")) in {"perfect_fit", "collinear_regressors"}
    assert code_of(lambda: oe.kpss(trend, "y", trend=True)) == "constant_series"
    assert_clean(oe.kpss(trend, "y"))               # a trend is not level stationary
    alternating = walk.assign(y=np.tile([1.0, -1.0], T // 2))
    for name in ("dfuller", "pperron", "dfgls", "zandrews"):
        assert code_of(lambda name=name: SERIES[name](alternating)) \
            in {"perfect_fit", "collinear_regressors"}
    assert_clean(oe.kpss(alternating, "y", auto=True))
    step = walk.assign(y=(np.arange(T) >= 30).astype(float))
    assert code_of(lambda: oe.zandrews(step, "y")) == "perfect_fit"
    assert code_of(lambda: oe.chow(step, "y", ["x"], 30)) == "perfect_fit"
    assert code_of(lambda: oe.sbsingle(step, "y", ["x"])) == "perfect_fit"
    assert_clean(oe.dfuller(step, "y"))
    # y equal to a regressor, and regressors without variation.
    same = walk.assign(x=walk["y"])
    for name in ("egranger", "chow", "sbsingle", "cusum"):
        assert code_of(lambda name=name: SERIES[name](same)) == "perfect_fit"
    flat = walk.assign(x=1.0)
    assert code_of(lambda: oe.egranger(flat, "y", ["x"])) == "collinear_regressors"
    assert oe.chow(flat, "y", ["x"], 30).attrs["omitted"] == ["x"]


# ---- magnitudes ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SERIES))
def test_extreme_magnitudes(name):
    walk = make_walk(5)
    run = SERIES[name]
    base = run(walk).attrs["statistic"]
    for scale in (1e-8, 1e8, 1e-100, 1e100):
        scaled = walk.assign(y=walk["y"] * scale, x=walk["x"] * scale)
        assert assert_clean(run(scaled)).attrs["statistic"] == pytest.approx(base, rel=1e-6)
    # Squares that overflow or underflow float64 are refused with the way out, not
    # reported as a constant series or an exact fit.
    for scale in (1e200, 1e-200):
        with pytest.raises(AnalysisError) as caught:
            run(walk.assign(y=walk["y"] * scale))
        assert caught.value.code == "numerical_failure"
        assert "Rescale" in str(caught.value) and "'y'" in str(caught.value)
    if name in ("egranger", "chow", "sbsingle", "cusum"):
        with pytest.raises(AnalysisError) as caught:
            run(walk.assign(x=walk["x"] * 1e180))
        assert caught.value.code == "numerical_failure" and "'x'" in str(caught.value)


def test_extreme_magnitudes_in_panels():
    panel = make_panel(8, 30)
    for test in PANEL_TESTS:
        base = oe.xtunitroot(panel, "y", "id", "year", test=test).attrs["statistic"]
        for scale, shift in ((1e8, 0.0), (1e-8, 0.0), (1.0, 1e9)):
            moved = panel.assign(y=panel["y"] * scale + shift)
            got = assert_clean(oe.xtunitroot(moved, "y", "id", "year", test=test))
            assert got.attrs["statistic"] == pytest.approx(base, rel=1e-5)
        for scale in (1e200, 1e-200):
            assert code_of(lambda scale=scale, test=test: oe.xtunitroot(
                panel.assign(y=panel["y"] * scale), "y", "id", "year", test=test)) \
                == "numerical_failure"
    base = oe.xtcointtest(panel, "y", ["x"], "id", "year").attrs["statistic"]
    moved = panel.assign(y=panel["y"] * 1e8 + 5e9, x=panel["x"] * 1e-8)
    assert oe.xtcointtest(moved, "y", ["x"], "id", "year").attrs["statistic"] \
        == pytest.approx(base, rel=1e-5)
    assert code_of(lambda: oe.xtcointtest(panel.assign(x=panel["x"] * 1e200), "y", ["x"],
                                          "id", "year")) == "numerical_failure"


# ---- time columns ----------------------------------------------------------------------------


def test_time_columns():
    walk = make_walk()
    dates = pd.date_range("2000-01-01", periods=T, freq="QS")
    base = oe.dfuller(walk, "y").attrs["statistic"]

    def adf(frame, time="t"):
        return oe.dfuller(frame, "y", time=time)

    for frame in (walk, walk.sample(frac=1.0, random_state=2), walk.assign(t=dates),
                  walk.assign(t=dates).sample(frac=1.0, random_state=3),
                  walk.assign(t=dates.tz_localize("UTC")), walk.assign(t=walk["t"].astype(float)),
                  walk.assign(t=walk["t"] - 1000),
                  # Integer periods beyond 2^53 are still consecutive integers.
                  walk.assign(t=walk["t"] + 2 ** 53 + 1),
                  # Datetimes are taken as consecutive in sorted order, whatever their spacing.
                  walk.assign(t=dates.delete(5).insert(T - 1, pd.Timestamp("2030-01-01")))):
        assert adf(frame).attrs["statistic"] == pytest.approx(base, rel=1e-12)
    assert code_of(lambda: adf(walk.assign(t=walk["t"] + 0.5))) == "invalid_time"
    assert code_of(lambda: adf(walk.assign(t=walk["t"].astype(str)))) == "invalid_time"
    assert code_of(lambda: adf(walk.assign(t=walk["t"] > 3))) == "invalid_time"
    assert code_of(lambda: adf(walk.assign(t=pd.to_timedelta(walk["t"], unit="D")))) \
        == "invalid_time"
    assert code_of(lambda: adf(walk.assign(t=walk["t"].where(walk["t"] < 30, walk["t"] + 1)))) \
        == "time_gaps"
    assert code_of(lambda: adf(walk.assign(t=walk["t"].where(walk["t"] != 7, 6)))) \
        == "repeated_time_values"
    assert code_of(lambda: adf(walk.assign(t=walk["t"].where(walk["t"] != 7).astype(float)))) \
        == "missing_values"
    assert code_of(lambda: adf(walk, time="y")) == "invalid_spec"
    assert code_of(lambda: adf(walk, time=["t"])) == "invalid_spec"
    assert code_of(lambda: adf(walk, time="nope")) == "missing_columns"
    # The break date of zandrews is reported as a label of the time column.
    assert oe.zandrews(walk.assign(t=dates), "y", time="t").attrs["break_period"] \
        == dates[oe.zandrews(walk, "y").attrs["break_index"]].isoformat()


def test_chow_break_dates():
    walk = make_walk()
    expected = oe.chow(walk, "y", ["x"], 30).attrs["statistic"]
    for position in (np.int64(30), np.int32(30)):
        assert oe.chow(walk, "y", ["x"], position).attrs["statistic"] == expected
    for position in (30.0, "30", None, True):
        assert code_of(lambda position=position: oe.chow(walk, "y", ["x"], position)) \
            == "invalid_break"
    for position in (0, 60, -1, 10 ** 9):
        assert code_of(lambda position=position: oe.chow(walk, "y", ["x"], position)) \
            == "invalid_break"
    assert code_of(lambda: oe.chow(walk, "y", ["x"], 1)) == "singular_subsample"
    assert code_of(lambda: oe.chow(walk, "y", ["x"], 2)) == "singular_subsample"
    # A second regime that cannot be fitted leaves the forecast test only.
    short = assert_clean(oe.chow(walk, "y", ["x"], 59))
    assert short.attrs["statistic"] is None and short.attrs["p_value"] is None
    assert short.loc["forecast_f", "statistic"] > 0 and short.loc["forecast_f", "df"] == 1
    assert any("undefined" in note for note in short.attrs["notes"])
    # With a time column the break is a value of that column.
    for value in (30, np.int64(30), 30.0, 29.5):
        assert oe.chow(walk, "y", ["x"], value, time="t").attrs["statistic"] == expected
    for value in (None, "30", float("nan"), pd.Timestamp("2001-01-01"), True):
        with pytest.raises(AnalysisError) as caught:
            oe.chow(walk, "y", ["x"], value, time="t")
        assert caught.value.code == "invalid_break" and "period number" in str(caught.value)
    dates = pd.date_range("2000-01-01", periods=T, freq="QS")
    dated = walk.assign(t=dates)
    for value in ("2007-07-01", pd.Timestamp("2007-07-01"), np.datetime64("2007-07-01"),
                  dates[30].to_pydatetime(), "2007-05-15"):
        result = oe.chow(dated, "y", ["x"], value, time="t")
        assert result.attrs["statistic"] == expected
        assert result.attrs["break_period"] == "2007-07-01T00:00:00"
    # A number is not a date: pandas would read 30 as 30 nanoseconds after 1970.
    for value in (30, None, float("nan"), "not a date", pd.NaT):
        assert code_of(lambda value=value: oe.chow(dated, "y", ["x"], value, time="t")) \
            == "invalid_break"
    zoned = walk.assign(t=dates.tz_localize("UTC"))
    for value in ("2007-07-01", pd.Timestamp("2007-07-01", tz="UTC"),
                  pd.Timestamp("2007-07-01 02:00", tz="Europe/Istanbul")):
        assert oe.chow(zoned, "y", ["x"], value, time="t").attrs["break_index"] == 30


# ---- options at their bounds ----------------------------------------------------------------


def test_lag_options():
    walk = make_walk()
    for name in ("dfuller", "pperron", "kpss", "zandrews", "egranger"):
        run = SERIES[name]
        for lags in (-1, 1.0, 2.5, True, "x"):
            assert code_of(lambda lags=lags, run=run: run(walk, lags=lags)) \
                in {"invalid_lags", "invalid_option"}
        assert run(walk, lags=np.int64(2)).attrs["statistic"] \
            == run(walk, lags=2).attrs["statistic"]
        assert code_of(lambda run=run: run(walk, lags=100)) \
            in {"invalid_lags", "insufficient_observations"}
    for bad in (-1, 1.0, True, "aic", 29, 100):
        assert code_of(lambda bad=bad: oe.dfgls(walk, "y", maxlag=bad)) == "invalid_lags"
    assert oe.dfgls(walk, "y", maxlag=28).attrs["nobs"] == 31
    assert len(oe.dfgls(walk, "y", maxlag=0)) == 1
    # The largest lag orders that leave a regression still run.
    assert_clean(oe.dfuller(walk, "y", lags=28))
    assert code_of(lambda: oe.dfuller(walk, "y", lags=29)) == "insufficient_observations"
    assert_clean(oe.pperron(walk, "y", lags=58))
    assert code_of(lambda: oe.pperron(walk, "y", lags=59)) == "invalid_lags"
    assert_clean(oe.kpss(walk, "y", lags=59))
    assert code_of(lambda: oe.kpss(walk, "y", lags=60)) == "invalid_lags"
    assert code_of(lambda: oe.kpss(walk, "y", lags=2, auto=True)) == "invalid_option"
    assert code_of(lambda: oe.kpss(walk, "y", trend="yes")) == "invalid_option"
    assert code_of(lambda: oe.dfgls(walk, "y", trend="trend")) == "invalid_option"
    assert code_of(lambda: oe.dfuller(walk, "y", trend="ct")) == "invalid_option"
    assert code_of(lambda: oe.dfuller(walk, "y", regress="yes")) == "invalid_option"
    assert code_of(lambda: oe.pperron(walk, "y", trend="drift")) == "invalid_option"
    for name in ("zandrews", "egranger"):
        run = SERIES[name]
        for method in ("aic", "bic", "t"):
            assert_clean(run(walk, lags=method))
            assert_clean(run(walk, lags=method, maxlag=0))
        assert code_of(lambda run=run: run(walk, lags="hqic")) == "invalid_option"
        assert code_of(lambda run=run: run(walk, lags=2, maxlag=4)) == "invalid_option"
        assert code_of(lambda run=run: run(walk, lags="aic", maxlag=40)) == "invalid_lags"
        assert code_of(lambda run=run: run(walk, lags="aic", maxlag=-1)) == "invalid_lags"
    assert code_of(lambda: oe.zandrews(walk, "y", break_="level")) == "invalid_option"
    assert code_of(lambda: oe.egranger(walk, "y", ["x"], trend="drift")) == "invalid_option"
    assert code_of(lambda: oe.egranger(walk, "y", ["x"], ecm=1)) == "invalid_option"
    assert code_of(lambda: oe.cusum(walk, "y", ["x"], level="2%")) == "invalid_option"
    assert code_of(lambda: oe.cusum(walk, "y", ["x"], intercept="no")) == "invalid_option"
    assert code_of(lambda: oe.sbsingle(walk, "y", ["x"], test="supf")) == "invalid_option"


def test_trimming_bounds():
    walk = make_walk()
    for trim in (0.5, 0.7, -0.1, "a", None, True, float("nan")):
        assert code_of(lambda trim=trim: oe.zandrews(walk, "y", trim=trim)) == "invalid_option"
        assert code_of(lambda trim=trim: oe.sbsingle(walk, "y", ["x"], trim=trim)) \
            == "invalid_option"
    for trim in (0, 0.0, 0.01, 0.3, 0.45, 0.499, np.float64(0.2)):
        assert_clean(oe.zandrews(walk, "y", trim=trim))
        assert_clean(oe.sbsingle(walk, "y", ["x"], trim=trim))
    # Without trimming every date that leaves K + 1 observations per regime is a candidate.
    assert oe.sbsingle(walk, "y", ["x"], trim=0).attrs["candidates"] == T - 2 * 3 + 1
    # With one candidate the sup-Wald test is an ordinary Wald test.
    one = oe.sbsingle(walk, "y", ["x"], trim=0.499)
    assert one.attrs["candidates"] == 1
    assert one.attrs["p_value"] == pytest.approx(stats.chi2.sf(one.attrs["statistic"], 2),
                                                 rel=1e-8)
    for test in ("avewald", "expwald"):
        result = assert_clean(oe.sbsingle(walk, "y", ["x"], test=test))
        assert result.attrs["p_value"] is None and result.attrs["critical_values"] is None
    # Three parameters need four observations per regime: eight rows leave one candidate,
    # seven leave none.
    assert oe.sbsingle(make_walk(total=8), "y", ["x", "z"], trim=0.4).attrs["candidates"] == 1
    assert code_of(lambda: oe.sbsingle(make_walk(total=7), "y", ["x", "z"], trim=0.4)) \
        == "insufficient_observations"


def test_regressor_lists():
    walk = make_walk()
    for name in ("egranger", "chow", "sbsingle", "cusum"):
        call = {"egranger": lambda x: oe.egranger(walk, "y", x),
                "chow": lambda x: oe.chow(walk, "y", x, 30),
                "sbsingle": lambda x: oe.sbsingle(walk, "y", x),
                "cusum": lambda x: oe.cusum(walk, "y", x)}[name]
        for x in ("x", None, [1], ["x", "x"], ["y"]):
            assert code_of(lambda x=x, call=call: call(x)) == "invalid_spec"
        assert code_of(lambda call=call: call(["nope"])) == "missing_columns"
        assert call(("x",)).attrs["statistic"] == call(["x"]).attrs["statistic"]
        if name == "egranger":
            assert code_of(lambda call=call: call([])) == "invalid_spec"
        else:
            assert_clean(call([]))                  # a break in the mean
    assert code_of(lambda: oe.cusum(walk, "y", [], intercept=False)) == "invalid_spec"
    assert code_of(lambda: oe.dfuller(walk, ["y"])) == "invalid_spec"
    assert code_of(lambda: oe.egranger(walk, ["y"], ["x"])) == "invalid_spec"
    rng = np.random.default_rng(9)
    many = walk.assign(**{f"x{i}": rng.standard_normal(T).cumsum() for i in range(6)})
    assert_clean(oe.egranger(many, "y", [f"x{i}" for i in range(5)]))
    assert code_of(lambda: oe.egranger(many, "y", [f"x{i}" for i in range(6)])) \
        == "too_many_regressors"


def test_collinear_regressors_are_omitted_and_reported():
    walk = make_walk()
    data = walk.assign(x2=walk["x"] * 3 + 1, one=1.0, late=(walk["t"] >= 50).astype(float))
    calls = {"egranger": lambda x: oe.egranger(data, "y", x),
             "chow": lambda x: oe.chow(data, "y", x, 30),
             "sbsingle": lambda x: oe.sbsingle(data, "y", x),
             "cusum": lambda x: oe.cusum(data, "y", x)}
    for name, call in calls.items():
        base = call(["x"]).attrs["statistic"]
        for x, omitted in ((["x", "x2"], ["x2"]), (["one", "x"], ["one"]),
                           (["x", "x2", "one"], ["x2", "one"])):
            result = assert_clean(call(x))
            assert result.attrs["omitted"] == omitted, name
            assert result.attrs["statistic"] == pytest.approx(base, rel=1e-9)
            assert any("collinear" in note.lower() for note in result.attrs["notes"])
    # A dummy that is constant inside a regime cannot break there.
    assert code_of(lambda: calls["chow"](["x", "late"])) == "singular_subsample"
    assert code_of(lambda: calls["sbsingle"](["x", "late"])) == "singular_subsample"
    late = assert_clean(calls["cusum"](["x", "late"]))
    assert late.attrs["start"] > 3 and any("identify" in note for note in late.attrs["notes"])
    assert_clean(calls["egranger"](["x", "late"]))


def test_engle_granger_needs_a_minimum_sample():
    walk = make_walk()
    for rows in (3, 5, 7):
        assert code_of(lambda rows=rows: oe.egranger(walk.iloc[:rows], "y", ["x"])) \
            == "insufficient_observations"
    assert_clean(oe.egranger(walk.iloc[:8], "y", ["x"]))
    assert code_of(lambda: oe.egranger(walk.iloc[:9], "y", ["x", "z"], trend="trend")) \
        == "insufficient_observations"
    assert_clean(oe.egranger(walk.iloc[:10], "y", ["x", "z"], trend="trend"))
    assert code_of(lambda: oe.egranger(walk.iloc[:10], "y", ["x"], lags=5)) \
        == "insufficient_observations"


# ---- panels ---------------------------------------------------------------------------------


def panel_run(frame, test, **options):
    return oe.xtunitroot(frame, "y", "id", "year", test=test, **options)


@pytest.mark.parametrize("test", PANEL_TESTS)
def test_degenerate_panels(test):
    base = make_panel(8, 30)
    before = base.copy(deep=True)
    assert_clean(panel_run(base, test))
    assert_clean(panel_run(base, test, demean=True))
    pd.testing.assert_frame_equal(base, before)
    assert code_of(lambda: panel_run(base.iloc[:0], test)) == "empty_data"
    assert code_of(lambda: panel_run(make_panel(1, 30), test)) == "insufficient_panels"
    assert code_of(lambda: panel_run(make_panel(6, 3), test)) == "insufficient_observations"
    assert code_of(lambda: panel_run(base[~((base["id"] == 3) & (base["year"] == 2010))],
                                     test)) == "time_gaps"
    assert code_of(lambda: panel_run(pd.concat([base, base.iloc[[5]]]), test)) \
        == "repeated_time_values"
    assert code_of(lambda: panel_run(base.assign(y=base["y"].where(base.index != 7)), test)) \
        == "missing_values"
    assert code_of(lambda: panel_run(base.assign(
        id=base["id"].where(base.index != 3).astype(float)), test)) == "missing_values"
    assert code_of(lambda: panel_run(base.assign(y="a"), test)) == "non_numeric_column"
    assert code_of(lambda: panel_run(base.assign(year=base["year"] + 0.5), test)) \
        == "invalid_time"
    assert code_of(lambda: panel_run(base.assign(year=base["year"].astype(str)), test)) \
        == "invalid_time"
    assert code_of(lambda: oe.xtunitroot(base, "y", "id", "id", test=test)) == "invalid_spec"
    assert code_of(lambda: oe.xtunitroot(base, "id", "id", "year", test=test)) == "invalid_spec"
    assert code_of(lambda: oe.xtunitroot(base, ["y"], "id", "year", test=test)) == "invalid_spec"
    assert code_of(lambda: oe.xtunitroot(base, "y", ["id"], "year", test=test)) == "invalid_spec"
    # One panel without variation makes every test undefined.
    constant = base.assign(y=base["y"].where(base["id"] != 2, 5.0))
    assert code_of(lambda: panel_run(constant, test)) in {"constant_series", "perfect_fit",
                                                          "collinear_regressors"}
    trending = base.assign(y=base["y"].where(base["id"] != 2, base["year"] * 1.0))
    assert code_of(lambda: panel_run(trending, test, trend="trend")) \
        in {"constant_series", "perfect_fit", "collinear_regressors"}
    # Labels, row order and the type of the time column do not matter.
    statistic = panel_run(base, test).attrs["statistic"]
    relabelled = base.assign(id=base["id"].map(lambda i: f"unit {7 - i}"),
                             year=pd.to_datetime(base["year"].astype(str) + "-01-01"))
    shuffled = relabelled.sample(frac=1.0, random_state=4)
    assert panel_run(shuffled, test).attrs["statistic"] == pytest.approx(statistic, rel=1e-10)
    # Two panels are enough, also after removing cross-sectional means.
    assert_clean(panel_run(make_panel(2, 30), test, demean=True))
    unbalanced = base[~((base["id"] == 3) & (base["year"] < 2005))]
    if test in ("ips", "fisher"):
        assert assert_clean(panel_run(unbalanced, test)).attrs["balanced"] is False
    else:
        assert code_of(lambda: panel_run(unbalanced, test)) == "unbalanced_panel"


def test_panel_options_at_their_bounds():
    base = make_panel(8, 30)
    for test in PANEL_TESTS:
        for lags in (-1, 1.5, True, "x"):
            assert code_of(lambda lags=lags, test=test: panel_run(base, test, lags=lags)) \
                in {"invalid_lags", "invalid_option"}
        assert code_of(lambda test=test: panel_run(base, test, trend="drift")) \
            == "invalid_option"
        assert code_of(lambda test=test: panel_run(base, test, demean=1)) == "invalid_option"
        assert code_of(lambda test=test: panel_run(base, test, lags=0, maxlag=2)) \
            == "invalid_option"
    assert code_of(lambda: panel_run(base, "pp")) == "invalid_option"
    # Options that a test does not use are refused rather than ignored.
    unused = {"llc": {"method": "pperron", "robust": True, "altt": True},
              "ips": {"kernel": "parzen", "kernel_lags": 2, "robust": True, "lrv": "null"},
              "fisher": {"kernel_lags": 2, "altt": True, "lrv": "null"},
              "hadri": {"method": "pperron", "altt": True, "lrv": "null"},
              "breitung": {"kernel_lags": 2, "robust": True, "method": "pperron"},
              "ht": {"kernel": "parzen", "robust": True, "method": "pperron"}}
    for test, options in unused.items():
        for name, value in options.items():
            with pytest.raises(AnalysisError) as caught:
                panel_run(base, test, **{name: value})
            assert caught.value.code == "invalid_option" and name in str(caught.value)
    for test in ("hadri", "ht"):
        assert code_of(lambda test=test: panel_run(base, test, lags=1)) == "invalid_lags"
        assert code_of(lambda test=test: panel_run(base, test, lags="aic")) == "invalid_lags"
    for test in ("ips", "hadri"):
        assert code_of(lambda test=test: panel_run(base, test, trend="none")) == "invalid_option"
    for test in ("llc", "ips", "fisher", "breitung"):
        assert code_of(lambda test=test: panel_run(base, test, lags=20)) \
            in {"insufficient_observations", "ips_moments_unavailable"}
        assert code_of(lambda test=test: panel_run(base, test, lags="aic", maxlag=13)) \
            == "invalid_lags"
        for method in ("aic", "bic", "hqic"):
            assert_clean(panel_run(base, test, lags=method))
        assert panel_run(base, test, lags="aic", maxlag=0).attrs["lags_max"] == 0
    assert code_of(lambda: panel_run(base, "ips", lags=9)) == "ips_moments_unavailable"
    assert code_of(lambda: panel_run(make_panel(6, 9), "ips")) == "ips_moments_unavailable"
    assert code_of(lambda: panel_run(base, "fisher", method="pperron", lags="aic")) \
        == "invalid_lags"
    assert code_of(lambda: panel_run(base, "fisher", method="pperron", lags=29)) \
        == "invalid_lags"
    assert_clean(panel_run(base, "fisher", method="pperron", lags=28))
    # Kernel truncations: fewer than the differences (LLC) or the periods (Hadri).
    assert_clean(panel_run(base, "llc", kernel_lags=0))
    assert_clean(panel_run(base, "llc", kernel_lags=28))
    assert code_of(lambda: panel_run(base, "llc", kernel_lags=29)) == "invalid_lags"
    assert_clean(panel_run(base, "hadri", kernel_lags=29))
    assert code_of(lambda: panel_run(base, "hadri", kernel_lags=30)) == "invalid_lags"
    assert code_of(lambda: panel_run(base, "hadri", kernel_lags=-1)) == "invalid_lags"
    for kernel in ("parzen", "quadratic_spectral"):
        assert_clean(panel_run(base, "llc", kernel=kernel))
        assert_clean(panel_run(base, "hadri", kernel=kernel, kernel_lags=3))
    assert code_of(lambda: panel_run(base, "llc", kernel="qs")) == "invalid_option"
    # Short and wide panels.
    assert_clean(panel_run(make_panel(3000, 6), "ht"))
    assert_clean(panel_run(make_panel(400, 6), "llc"))
    for trend in ("none", "constant", "trend"):     # four periods are the smallest panels
        assert_clean(panel_run(make_panel(40, 4), "ht", trend=trend))
    assert code_of(lambda: panel_run(make_panel(6, 3), "ht")) == "insufficient_observations"
    with pytest.raises(AnalysisError, match="at least 4"):
        panel_run(make_panel(6, 3), "ht")


def test_panel_cointegration_adversarial():
    base = make_panel(8, 30)
    before = base.copy(deep=True)

    def kao(frame=base, x=("x",), **options):
        return oe.xtcointtest(frame, "y", list(x), "id", "year", **options)

    assert_clean(kao())
    assert_clean(kao(demean=True))
    pd.testing.assert_frame_equal(base, before)
    # These routes were added independently; retain rejection of unknown tests,
    # rather than requiring shipped Pedroni/Westerlund routes to be absent.
    assert_clean(kao(test="pedroni"))
    assert_clean(kao(test="westerlund"))
    assert code_of(lambda: kao(test="not_a_test")) == "invalid_option"
    assert code_of(lambda: kao(x=())) == "invalid_spec"
    assert code_of(lambda: kao(x=("y",))) == "invalid_spec"
    assert code_of(lambda: oe.xtcointtest(base, "y", "x", "id", "year")) == "invalid_spec"
    assert code_of(lambda: kao(x=("nope",))) == "missing_columns"
    assert code_of(lambda: kao(make_panel(1, 30))) == "insufficient_panels"
    assert code_of(lambda: kao(make_panel(6, 6))) == "insufficient_observations"
    assert code_of(lambda: kao(base[~((base["id"] == 3) & (base["year"] < 2005))])) \
        == "unbalanced_panel"
    assert code_of(lambda: kao(base[~((base["id"] == 3) & (base["year"] == 2010))])) \
        == "time_gaps"
    assert code_of(lambda: kao(base.assign(x=base["x"].where(base.index != 4)))) \
        == "missing_values"
    for lags in (-1, 1.5, "aic", True):
        assert code_of(lambda lags=lags: kao(lags=lags)) == "invalid_lags"
    assert_clean(kao(lags=0))
    assert_clean(kao(lags=24))
    assert code_of(lambda: kao(lags=25)) == "insufficient_observations"
    assert_clean(kao(kernel_lags=0))
    assert_clean(kao(kernel_lags=28))
    assert code_of(lambda: kao(kernel_lags=29)) == "invalid_lags"
    assert code_of(lambda: kao(kernel="qs")) == "invalid_option"
    assert code_of(lambda: kao(demean="yes")) == "invalid_option"
    doubled = base.assign(x2=base["x"] * 2)
    result = assert_clean(kao(doubled, x=("x", "x2")))
    assert result.attrs["omitted"] == ["x2"]
    assert result.attrs["statistic"] == pytest.approx(kao().attrs["statistic"], rel=1e-9)
    assert code_of(lambda: kao(base.assign(x=base["id"] * 1.0))) == "collinear_regressors"
    assert code_of(lambda: kao(base.assign(y=base["x"] * 2 + base["id"]))) == "perfect_fit"
    assert code_of(lambda: kao(base.assign(y=1.0))) == "perfect_fit"


def test_no_warnings_escape():
    walk, panel = make_walk(), make_panel(6, 25)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for run in SERIES.values():
            run(walk)
        for test in PANEL_TESTS:
            panel_run(panel, test)
        oe.xtcointtest(panel, "y", ["x"], "id", "year")
        oe.dfuller(walk, "y", regress=True)
        oe.egranger(walk, "y", ["x", "z"], ecm=True, regress=True, lags="aic")
