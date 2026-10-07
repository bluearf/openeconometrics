"""Replay helper contracts: independently checked moments, inference and source identity."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import openecon as oe
from openecon.analysis_contracts import AnalysisError
from openecon.dataset import Dataset
from openecon.resources import use_workspace_budget


def replay(frame: pd.DataFrame, rows: int = 7) -> Dataset:
    return Dataset.from_batches(lambda: (frame.iloc[start:start+rows] for start in range(0, len(frame), rows)),
                                list(frame), row_count=len(frame))


def assert_tables(actual, expected, *, atol=1e-10, rtol=1e-9):
    assert list(actual) == list(expected)
    for name in expected:
        pd.testing.assert_frame_equal(actual[name], expected[name], check_exact=False,
                                      atol=atol, rtol=rtol)
    for key in ["n", "n_missing", "df", "df1", "df2", "df_resid", "groups", "levels", "terms"]:
        if key in expected.attrs:
            assert actual.attrs[key] == expected.attrs[key]
    assert actual.attrs["streaming"] is True


def sample(n=113):
    rng = np.random.default_rng(144)
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n),
                          "a": rng.choice(["a", "b", "c"], n),
                          "b": rng.choice(["p", "q"], n)})
    frame["y"] = 4 + 0.7*frame.x + (frame.a == "a")*0.8 + rng.normal(size=n)
    return frame


@pytest.mark.parametrize("kwargs", [{}, {"mu": 0.35}, {"paired_with": "z", "mu": -0.2}, {"by": "b"}])
@pytest.mark.parametrize("rows", [1, 7, 1000])
def test_ttest_all_modes_match_dense_and_independent_moments(kwargs, rows):
    frame = sample()
    frame.loc[[1, 11], "x"] = np.nan
    result = oe.ttest(replay(frame, rows), "x", **kwargs)
    assert_tables(result, oe.ttest(frame, "x", **kwargs))
    if "by" not in kwargs:
        chosen = frame[["x", "z"]].dropna() if "paired_with" in kwargs else frame[["x"]].dropna()
        values = chosen.x.to_numpy() - chosen.z.to_numpy() if "paired_with" in kwargs else chosen.x.to_numpy()
        difference = values.mean() - kwargs.get("mu", 0)
        t = difference / (values.std(ddof=1) / np.sqrt(len(values)))
        assert result["test"].iloc[0].statistic == pytest.approx(t, rel=1e-12)


@pytest.mark.parametrize("kwargs", [{"sd": 0.7}, {"by": "a"}, {"by": "b", "center": "median"},
                                       {"by": "a", "center": "trimmed"}])
def test_variance_tests_use_exact_group_ranks(kwargs):
    frame = sample()
    frame.loc[[3, 4], "x"] = np.nan
    assert_tables(oe.sdtest(replay(frame, 3), "x", **kwargs), oe.sdtest(frame, "x", **kwargs))


def test_oneway_preserves_all_homogeneity_and_posthoc_tables():
    frame = sample(130)
    frame.loc[8, "a"] = None
    kwargs = {"posthoc": ["tukey", "bonferroni", "sidak", "scheffe", "lsd", "games_howell", "holm"]}
    assert_tables(oe.oneway(replay(frame, 13), "x", "a", **kwargs),
                  oe.oneway(frame, "x", "a", **kwargs), atol=1e-9)


@pytest.mark.parametrize("ss_type", [1, 2, 3])
@pytest.mark.parametrize("slopes", [False, True])
def test_factorial_ancova_qr_and_marginal_means(ss_type, slopes):
    frame = sample(189)
    frame.loc[7, "x"] = np.nan
    kwargs = {"covariates": ["x"], "ss_type": ss_type,
              "homogeneity_of_slopes": slopes, "emmeans": ["a", "b", ["a", "b"]]}
    actual = oe.anova(replay(frame, 11), "y", ["a", "b"], **kwargs)
    assert_tables(actual, oe.anova(frame, "y", ["a", "b"], **kwargs), atol=1e-9)
    # The compressed design produces the same residual SS as independent NumPy OLS.
    clean = frame.dropna()
    a = np.column_stack([np.where(clean.a == "a", 1, np.where(clean.a == "c", -1, 0)),
                         np.where(clean.a == "b", 1, np.where(clean.a == "c", -1, 0))])
    b = np.where(clean.b == "p", 1, -1)[:, None]
    x = clean.x.to_numpy()[:, None]
    pieces = [np.ones((len(clean), 1)), x, a, b, a*b]
    if slopes:
        pieces += [a*x, b*x]
    design = np.column_stack(pieces)
    residual = clean.y - design @ np.linalg.lstsq(design, clean.y, rcond=None)[0]
    assert actual["anova"].loc["error", "ss"] == pytest.approx(float(residual @ residual), rel=1e-12)


@pytest.mark.parametrize("matrix", ["correlation", "covariance"])
@pytest.mark.parametrize("rows", [1, 9, 1000])
def test_pca_moments_and_replay_scores_match_numpy_and_keep_missing_rows(matrix, rows):
    frame = sample()
    frame.index = np.repeat(np.arange(57), 2)[:len(frame)]
    frame.loc[5, "z"] = np.nan
    names = ["x", "z", "y"]
    result = oe.pca(replay(frame, rows), names, matrix=matrix, components=2)
    assert_tables(result, oe.pca(frame, names, matrix=matrix, components=2))
    clean = frame[names].dropna().to_numpy()
    target = np.corrcoef(clean, rowvar=False) if matrix == "correlation" else np.cov(clean, rowvar=False)
    assert result["eigenvalues"].eigenvalue.to_numpy() == pytest.approx(np.linalg.eigvalsh(target)[::-1], rel=1e-12)
    scores = oe.pca_scores(result, replay(frame, rows), normalize=True)
    assert isinstance(scores, Dataset)
    actual = pd.concat(scores.iter_batches(batch_rows=5))
    expected = oe.pca_scores(result, frame, normalize=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-10, rtol=1e-10)
    assert len(actual) == len(frame)
    assert actual.loc[5].isna().all().all()


def test_anchored_moments_retain_large_level_variation():
    frame = sample(151)
    frame["x"] = 1e9 + frame.x
    actual = oe.ttest(replay(frame, 3), "x", mu=1e9)
    expected = oe.ttest(frame, "x", mu=1e9)
    assert_tables(actual, expected, atol=1e-8, rtol=1e-8)
    centred = frame.x.to_numpy() - 1e9
    statistic = centred.mean() / (centred.std(ddof=1) / np.sqrt(len(centred)))
    assert actual.attrs["statistic"] == pytest.approx(statistic, rel=1e-10, abs=1e-10)
    frame["y"] += 1e8
    assert_tables(oe.anova(replay(frame, 3), "y", ["a", "b"], covariates=["x"]),
                  oe.anova(frame, "y", ["a", "b"], covariates=["x"]), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("procedure", ["ttest", "sdtest", "oneway", "anova"])
def test_changed_factory_fails_between_replay_passes(procedure):
    frame = sample()
    calls = 0
    def factory():
        nonlocal calls
        calls += 1
        changed = frame.copy()
        if calls > 1:
            changed.loc[0, "x"] += 1
            changed.loc[0, "y"] += 1
        yield from (changed.iloc[i:i+7] for i in range(0, len(changed), 7))
    data = Dataset.from_batches(factory, list(frame), row_count=len(frame))
    with pytest.raises(AnalysisError, match="changed"):
        if procedure == "ttest":
            oe.ttest(data, "x", by="b")
        elif procedure == "sdtest":
            oe.sdtest(data, "x", by="b")
        elif procedure == "oneway":
            oe.oneway(data, "x", "a")
        else:
            oe.anova(data, "y", ["a", "b"], covariates=["x"])


def test_pca_lazy_scores_reject_mutation_after_creation():
    frame = sample()
    source = Dataset.from_frame(frame)
    scores = oe.pca_scores(oe.pca(source, ["x", "z"]), source)
    frame.loc[0, "x"] += 1
    with pytest.raises(AnalysisError, match="changed"):
        list(scores.iter_batches())


@pytest.mark.parametrize("procedure", ["ttest", "sdtest", "oneway", "anova", "pca"])
def test_missing_raise_and_early_resource_guard(procedure):
    frame = sample()
    frame.loc[2, "x"] = np.nan
    def run(**kwargs):
        if procedure == "ttest":
            return oe.ttest(replay(frame), "x", **kwargs)
        if procedure == "sdtest":
            return oe.sdtest(replay(frame), "x", sd=1, **kwargs)
        if procedure == "oneway":
            return oe.oneway(replay(frame), "x", "a", **kwargs)
        if procedure == "anova":
            return oe.anova(replay(frame), "x", ["a", "b"], **kwargs)
        return oe.pca(replay(frame), ["x", "z"], **kwargs)
    with pytest.raises(AnalysisError, match="missing"):
        run(missing="raise")
    with use_workspace_budget(1), pytest.raises(AnalysisError, match="workspace"):
        run()


def test_factorial_native_restrictions_and_empty_cells_are_preserved():
    frame = sample()
    frame = frame.loc[~((frame.a == "a") & (frame.b == "p"))]
    with pytest.raises(AnalysisError, match="unobserved"):
        oe.anova(replay(frame), "y", ["a", "b"])
    # Main effects do not require all interactions cells to be present.
    assert_tables(oe.anova(replay(frame), "y", ["a", "b"], interactions="none"),
                  oe.anova(frame, "y", ["a", "b"], interactions="none"))


@pytest.mark.parametrize("method", ["pf", "ipf", "pcf", "ml"])
def test_factor_fitting_and_score_source_use_small_moment_matrix(method):
    rng = np.random.default_rng(71)
    latent = rng.normal(size=(183, 2))
    columns = ["x1", "x2", "x3", "x4", "x5", "x6"]
    loadings = np.array([[.8,.1],[.75,.1],[.7,.1],[.1,.8],[.1,.75],[.1,.7]])
    frame = pd.DataFrame(latent @ loadings.T + rng.normal(scale=.4, size=(183,6)), columns=columns)
    frame.loc[17, "x2"] = np.nan
    actual = oe.factor(replay(frame, 9), columns, method=method, factors=2, mineigen=0,
                       rotate="varimax")
    expected = oe.factor(frame, columns, method=method, factors=2, mineigen=0,
                         rotate="varimax")
    assert_tables(actual, expected, rtol=1e-6, atol=1e-6)
    scores = oe.factor_scores(actual, replay(frame, 9))
    assert isinstance(scores, Dataset)
    pd.testing.assert_frame_equal(pd.concat(scores.iter_batches()), oe.factor_scores(actual, frame),
                                  check_exact=False, atol=1e-9, rtol=1e-9)
    pd.testing.assert_frame_equal(oe.factortest(replay(frame, 9), columns),
                                  oe.factortest(frame, columns), check_exact=False, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("pairwise", [True, False])
def test_pearson_missing_pair_counts_and_intervals_match_dense(pairwise):
    frame = sample(137)
    frame.loc[[1, 4], "x"] = np.nan
    frame.loc[[4, 19], "z"] = np.nan
    frame.loc[[1, 21], "y"] = np.nan
    actual = oe.correlate(replay(frame, 3), ["x", "z", "y"], pairwise=pairwise, ci=True)
    expected = oe.correlate(frame, ["x", "z", "y"], pairwise=pairwise, ci=True)
    assert_tables(actual, expected)
    if pairwise:
        pair = frame[["x", "z"]].dropna()
        assert actual["coefficients"].loc["x", "z"] == pytest.approx(np.corrcoef(pair.x, pair.z)[0, 1], rel=1e-12)
        assert actual["n"].loc["x", "z"] == len(pair)


def test_partial_correlation_tsqr_matches_dense():
    frame = sample()
    frame.loc[2, "x"] = np.nan
    actual = oe.pcorr(replay(frame, 3), "y", ["x"], controls=["z"])
    expected = oe.pcorr(frame, "y", ["x"], controls=["z"])
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-10, rtol=1e-10)
    assert actual.attrs["streaming"] is True


def test_physical_parquet_helpers_and_paper_tables(tmp_path):
    frame = sample(357)
    frame.loc[21, "x"] = np.nan
    path = tmp_path / "source.parquet"
    frame.to_parquet(path, index=False, row_group_size=13)
    from openecon.dataset import scan
    source = scan(path)
    tests = [oe.ttest(source, "x", by="b"), oe.oneway(source, "y", "a"),
             oe.anova(source, "y", ["a", "b"], covariates=["x"]),
             oe.pca(source, ["x", "z", "y"]),
             oe.correlate(source, ["x", "z", "y"], ci=True)]
    dense = [oe.ttest(frame, "x", by="b"), oe.oneway(frame, "y", "a"),
             oe.anova(frame, "y", ["a", "b"], covariates=["x"]),
             oe.pca(frame, ["x", "z", "y"]),
             oe.correlate(frame, ["x", "z", "y"], ci=True)]
    for actual, expected in zip(tests, dense, strict=True):
        assert_tables(actual, expected)
        latex = actual.to_latex()
        assert "\\toprule" in latex and "\\bottomrule" in latex
    scores = oe.pca_scores(tests[3], source)
    pd.testing.assert_frame_equal(pd.concat(scores.iter_batches(), ignore_index=True),
                                  oe.pca_scores(dense[3], frame).reset_index(drop=True),
                                  check_exact=False, atol=1e-10, rtol=1e-10)


def test_owned_rank_scratch_is_removed_after_source_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path))
    frame = sample()
    calls = 0
    def factory():
        nonlocal calls
        calls += 1
        changed = frame.copy()
        if calls > 1:
            changed.loc[0, "x"] += .5
        yield from (changed.iloc[i:i+13] for i in range(0,len(changed),13))
    source = Dataset.from_batches(factory, list(frame), row_count=len(frame))
    with pytest.raises(AnalysisError, match="changed"):
        oe.oneway(source, "x", "a")
    assert not list(tmp_path.iterdir())


def test_exact_rank_storage_failure_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENECON_SCRATCH_DIRECTORY", str(tmp_path / "absent"))
    with pytest.raises(AnalysisError, match="scratch"):
        oe.oneway(replay(sample()), "x", "a")


def test_declared_categorical_order_matches_resident_outputs():
    frame = sample(149)
    frame['a'] = pd.Categorical(frame.a, categories=['c','a','b','unused'], ordered=True)
    frame['b'] = pd.Categorical(frame.b, categories=['q','p'], ordered=True)
    source = replay(frame, 3)
    assert_tables(oe.ttest(source,'x',by='b'),oe.ttest(frame,'x',by='b'))
    assert_tables(oe.sdtest(source,'x',by='a'),oe.sdtest(frame,'x',by='a'))
    assert_tables(oe.oneway(source,'x','a'),oe.oneway(frame,'x','a'))
    assert_tables(oe.anova(source,'y',['a','b'],covariates=['x'],emmeans=['a']),
                  oe.anova(frame,'y',['a','b'],covariates=['x'],emmeans=['a']))


def test_changed_category_declaration_is_detected_even_with_unchanged_values():
    frame = sample()
    calls = 0
    def factory():
        nonlocal calls
        calls += 1
        changed = frame.copy()
        categories = ['p','q'] if calls == 1 else ['q','p']
        changed['b'] = pd.Categorical(changed.b, categories=categories, ordered=True)
        yield from (changed.iloc[i:i+13] for i in range(0,len(changed),13))
    source = Dataset.from_batches(factory,list(frame),row_count=len(frame))
    with pytest.raises(AnalysisError,match='category ordering changed'):
        oe.ttest(source,'x',by='b')


def test_small_value_blocks_use_global_scale_checks():
    frame = pd.DataFrame({'x':[1e-110,2e-110,3e-110,1.,2.,3.],
                          'z':[3e-110,1e-110,2e-110,3.,1.,2.],
                          'g':['a','b','a','b','a','b']})
    assert_tables(oe.ttest(replay(frame,1),'x'),oe.ttest(frame,'x'))
    assert_tables(oe.oneway(replay(frame,1),'x','g'),oe.oneway(frame,'x','g'))
    assert_tables(oe.correlate(replay(frame,1),['x','z']),oe.correlate(frame,['x','z']))
    # Multivariate procedures have their own valid float64 scale domain and
    # permit these small but representable standardized covariance matrices.
    tiny = frame[['x','z']].iloc[3:].copy()*1e-110
    assert_tables(oe.pca(replay(tiny,1),['x','z']),oe.pca(tiny,['x','z']))
    all_tiny = frame[['x']].iloc[:3]
    with pytest.raises(AnalysisError,match='too small'):
        oe.ttest(replay(all_tiny,1),'x')


def test_physical_parquet_declared_categories_and_custom_index(tmp_path):
    from openecon.dataset import scan
    frame = sample(193)
    frame['a'] = pd.Categorical(frame.a.replace({'a':'A','b':'B','c':'C'}),
                                 categories=['B','A','C','UNUSED'], ordered=True)
    frame['b'] = pd.Categorical(frame.b, categories=['q','p','UNUSED'], ordered=True)
    frame.index = pd.Index([f'case-{i//2}' for i in range(len(frame))], name='case')
    frame.iloc[17, frame.columns.get_loc('x')] = np.nan
    path = tmp_path/'categories.parquet'
    frame.to_parquet(path, row_group_size=13)
    source = scan(path)
    assert list(source.columns) == list(frame.columns)
    assert_tables(oe.ttest(source,'x',by='b'),oe.ttest(frame,'x',by='b'))
    assert_tables(oe.sdtest(source,'x',by='a'),oe.sdtest(frame,'x',by='a'))
    assert_tables(oe.oneway(source,'x','a',posthoc=['bonferroni']),
                  oe.oneway(frame,'x','a',posthoc=['bonferroni']))
    assert_tables(oe.anova(source,'y',['a','b'],covariates=['x'],emmeans=['a']),
                  oe.anova(frame,'y',['a','b'],covariates=['x'],emmeans=['a']))
    result = oe.pca(source,['x','z','y'])
    scores = oe.pca_scores(result,source)
    actual, expected = pd.concat(scores.iter_batches()), oe.pca_scores(result,frame)
    assert actual.index.tolist() == expected.index.tolist()
    assert actual.index.name == expected.index.name
    # The reader retains precise Arrow storage for the string index. Index
    # values/order/duplicates are identical to the resident object index.
    pd.testing.assert_frame_equal(actual, expected, check_exact=False,
                                  check_index_type=False, rtol=1e-10, atol=1e-10)


def test_score_source_validates_numeric_input_before_returning_lazy_view():
    frame = sample()
    result = oe.pca(frame,['x','z'])
    invalid = frame.copy()
    invalid['x'] = 'invalid'
    with pytest.raises(AnalysisError,match='numeric'):
        oe.pca_scores(result,replay(invalid))


def test_posthoc_geometry_is_guarded_before_pair_allocation(tmp_path, monkeypatch):
    from openecon.econometrics.stats import posthoc
    monkeypatch.setenv('OPENECON_SCRATCH_DIRECTORY',str(tmp_path))
    groups = 700
    frame = pd.DataFrame({'y':np.repeat(np.arange(groups,dtype=float),2)+np.tile([0.,.5],groups),
                          'g':np.repeat(np.arange(groups),2)})
    def forbidden(*args,**kwargs):
        raise AssertionError('posthoc pair geometry was allocated before its resource check')
    monkeypatch.setattr(posthoc.torch,'triu_indices',forbidden)
    with use_workspace_budget(8),pytest.raises(AnalysisError,match='posthoc'):
        oe.oneway(replay(frame,23),'y','g',posthoc=['bonferroni','sidak'])
    assert not list(tmp_path.iterdir())


def test_shared_resident_posthoc_also_checks_numeric_geometry(monkeypatch):
    import torch
    from openecon.econometrics.stats import common,posthoc
    groups=700
    moments=common.GroupMoments(torch.full((groups,),2.,dtype=torch.float64),
                                torch.arange(groups,dtype=torch.float64),
                                torch.ones(groups,dtype=torch.float64),
                                torch.ones(groups,dtype=torch.float64))
    def forbidden(*args,**kwargs):
        raise AssertionError('unguarded pair allocation')
    monkeypatch.setattr(posthoc.torch,'triu_indices',forbidden)
    with use_workspace_budget(8),pytest.raises(AnalysisError,match='posthoc'):
        posthoc.compare('bonferroni',list(range(groups)),moments,1.,groups,.05,0)


@pytest.mark.parametrize('streamed',[False,True])
def test_marginal_mean_cell_geometry_is_guarded_before_cartesian_product(monkeypatch,streamed):
    from openecon.econometrics.stats import anova
    rng=np.random.default_rng(32)
    frame=pd.DataFrame({name:rng.integers(0,20,499) for name in ['a','b','c']})
    frame['y']=rng.normal(size=len(frame))
    def forbidden(*args,**kwargs):
        raise AssertionError('marginal-mean cells were created before their workspace guard')
    monkeypatch.setattr(anova.itertools,'product',forbidden)
    with pytest.raises(AnalysisError,match='marginal means'):
        oe.anova(replay(frame,13) if streamed else frame,'y',['a','b','c'],
                 interactions='none',emmeans=[['a','b','c']])
