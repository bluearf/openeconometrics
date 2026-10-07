"""Editor documentation must describe shipped APIs without starting Python jobs."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_editor_api.py"
spec = importlib.util.spec_from_file_location("editor_api_generator", SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def saved_catalog():
    entries = json.loads((ROOT / "web/src/editor-api.json").read_text(encoding="utf-8"))
    return {entry["name"]: entry for entry in entries}


def test_advanced_unitroot_help_preserves_discrete_inference_contracts():
    catalog = saved_catalog()
    for name in ("ngperron", "kss"):
        entry = catalog[f"openecon.{name}"]
        assert entry["returns"] == "openecon.DataFrame"
        assert "p_value=None" in entry["description"]
        params = {p["name"]: p for p in entry["parameters"]}
        assert params["trend"]["default"] == "'constant'"
        assert params["max_work"]["default"] == "100000000"
    assert {p["name"]: p for p in catalog["openecon.ngperron"]["parameters"]}["lags"]["default"] == "None"


def test_multiple_break_help_states_bounded_iid_inference():
    entry = saved_catalog()["openecon.bai_perron"]
    assert entry["returns"] == "openecon.TableSet"
    options = {p["name"]: p for p in entry["parameters"]}
    assert options["errors"]["default"] == "'iid_gaussian'"
    assert "fixed exogenous" in options["x"]["description"].lower()
    assert options["replications"]["default"] == "499"
    assert options["seed"]["default"] == "0"


def test_native_network_extensions_have_explicit_offline_contracts():
    catalog = saved_catalog()
    assert catalog["openecon.hypergraph"]["returns"] == "openecon.Hypergraph"
    assert catalog["openecon.Hypergraph.degree"]["returns"] == "openecon.DataFrame"
    for owner in ("Network", "MultiNetwork", "SignedNetwork"):
        for method in ("weighted_assignment", "general_matching"):
            assert catalog[f"openecon.{owner}.{method}"]["returns"] == "openecon.NetworkMatchingResult"
    for method in ("k_shortest_paths", "strong_bridges", "strong_articulation_points"):
        assert catalog[f"openecon.Network.{method}"]["returns"] == "openecon.DataFrame"
    parameters = {p["name"]: p for p in catalog["openecon.Network.qap_regression"]["parameters"]}
    assert parameters["method"]["default"] == "'freedman_lane'"
    assert parameters["adjustment"]["default"] == "'none'"
    assert "conditional" in parameters["joint"]["description"]


def test_dynamic_help_requires_explicit_selection_before_simple_projection():
    catalog = saved_catalog()
    for name in ('dynamic_network', 'read_dynamic_network'):
        assert catalog[f'openecon.{name}']['returns'] == 'openecon.DynamicNetwork'
    for name in ('at', 'window'):
        assert catalog[f'openecon.DynamicNetwork.{name}']['returns'] == 'openecon.MultiNetwork'
    window = {p['name']: p for p in catalog['openecon.DynamicNetwork.window']['parameters']}
    assert window['selection']['default'] == "'overlap'"
    assert window['attributes']['default'] == window['weight']['default'] == "'raise'"
    assert 'openecon.DynamicNetwork.pagerank' not in catalog


def test_hypergraph_and_bounded_path_help_exposes_only_supported_types():
    catalog = saved_catalog()
    assert catalog['openecon.hypergraph']['returns'] == 'openecon.Hypergraph'
    assert catalog['openecon.read_hypergraph']['returns'] == 'openecon.Hypergraph'
    for name in ['degree', 'strength', 'memberships', 'summary']:
        assert catalog[f'openecon.Hypergraph.{name}']['returns'] == 'openecon.DataFrame'
    for name in ['clique_projection', 'star_projection']:
        assert catalog[f'openecon.Hypergraph.{name}']['returns'] == 'openecon.Network'
    assert 'openecon.Hypergraph.pagerank' not in catalog
    for name in ['k_shortest_paths', 'strong_bridges', 'strong_articulation_points']:
        assert catalog[f'openecon.Network.{name}']['returns'] == 'openecon.DataFrame'
    options = {p['name']: p for p in catalog['openecon.Network.k_shortest_paths']['parameters']}
    assert options['max_path_length']['default'] == 'None'
    assert all(options[name]['kind'] == 'keyword-only' for name in ['k', 'max_path_length', 'max_output_nodes', 'max_frontier', 'max_work'])


def test_multigraph_help_requires_reducer_and_preserves_distinct_result_types():
    catalog = saved_catalog()
    assert catalog['openecon.multigraph']['returns'] == 'openecon.MultiNetwork'
    assert catalog['openecon.read_multigraph']['returns'] == 'openecon.MultiNetwork'
    assert catalog['openecon.MultiNetwork.filter']['returns'] == 'openecon.MultiNetwork'
    assert catalog['openecon.MultiNetwork.degree']['returns'] == 'openecon.DataFrame'
    projection = catalog['openecon.MultiNetwork.to_network']
    assert projection['returns'] == 'openecon.Network'
    reducer = next(p for p in projection['parameters'] if p['name'] == 'reducer')
    assert reducer['kind'] == 'keyword-only' and 'default' not in reducer
    assert 'openecon.MultiNetwork.pagerank' not in catalog


def test_catalog_regenerates_from_committed_source_without_importing_runtime():
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--revision", "HEAD", "--check"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "editor API entries from committed source" in result.stdout


def test_ols_signature_and_defaults_match_the_shipped_public_interface():
    source = subprocess.check_output(
        ["git", "show", "HEAD:src/openecon/analysis.py"], cwd=ROOT, text=True,
    )
    tree = ast.parse(source)
    ols = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "ols")
    entry = saved_catalog()["openecon.ols"]
    assert entry["signature"] == f"ols({ast.unparse(ols.args)})"
    assert [p["name"] for p in entry["parameters"]] == [arg.arg for arg in ols.args.kwonlyargs]
    defaults = {p["name"]: p.get("default") for p in entry["parameters"]}
    assert defaults["missing"] == "'drop'"
    assert defaults["intercept"] == "True"
    assert defaults["alpha"] == "0.05"
    assert entry["returns"] == "openecon.OLSResult"


def test_ols_only_postestimation_is_not_suggested_for_binary_model_results():
    catalog = saved_catalog()
    assert catalog["openecon.logit"]["returns"] == "openecon.ResultBundle"
    assert catalog["openecon.probit"]["returns"] == "openecon.ResultBundle"
    assert "openecon.OLSResult.predict" in catalog
    assert "openecon.OLSResult.vif" in catalog
    assert "openecon.ResultBundle.predict" not in catalog
    assert "openecon.ResultBundle.vif" not in catalog
    assert catalog["openecon.ResultBundle.summary"]["parameters"] == [
        {"name": "format", "kind": "positional-or-keyword", "default": "'text'", "description": "'text' or 'latex'."}
    ]


def test_table_and_chart_help_preserve_useful_actual_defaults():
    catalog = saved_catalog()
    assert catalog["openecon.DataFrame.head"]["parameters"][0]["default"] == "5"
    assert catalog["openecon.DataFrame.describe"]["returns"] == "openecon.DataFrame"
    assert catalog["openecon.DataFrame.to_latex"]["kind"] == "method"
    hist = catalog["openecon.plot.hist"]
    assert hist["parameters"][2] == {"name": "bins", "kind": "keyword-only", "default": "20"}
    assert hist["returns"] == "openecon.plot.PlotSpec"
    assert catalog["console.display"]["signature"] == "display(value)"
    assert catalog["console.display"]["returns"] == "None"


def test_prediction_dispatch_help_lists_source_derived_category_options_without_false_defaults():
    reader = generator.SourceReader("HEAD")
    catalog = saved_catalog()
    wrappers = reader.committed("src/openecon/analysis.py")
    delegated = reader.committed("src/openecon/econometrics/postest/prediction.py")
    for name in ("predict", "margins"):
        wrapper = generator.function_node(wrappers.body, name)
        item = catalog[f"openecon.{name}"]
        assert item["signature"] == f"{name}({ast.unparse(wrapper.args)})"
        options = {p["name"]: p for p in item["parameters"]}
        target = generator.function_node(delegated.body, name)
        assert {p["name"] for p in generator.parameter_list(target.args, {})} <= options.keys()
        assert options["outcome"]["default"] == "None"
        assert "Saved categorical" in options["outcome"]["description"]
        assert "default" not in options["kind"]
        assert "xb" in options["kind"]["description"] and "response" in options["kind"]["description"]


def test_documentation_is_bounded_unique_and_excludes_unpublished_models():
    catalog = saved_catalog()
    assert len(catalog) >= 280
    assert (ROOT / "web/src/editor-api.json").stat().st_size < 500_000
    assert all(entry["description"] and len(entry["description"]) <= 500 for entry in catalog.values())
    assert "openecon.panel_reg" not in catalog
    for entry in catalog.values():
        assert len({p["name"] for p in entry["parameters"]}) == len(entry["parameters"])
        if entry["kind"] in {"class", "method"}:
                assert all(p["name"] not in {"self", "cls"} for p in entry["parameters"])


def test_wave5_test_procedures_and_heteroskedastic_scale_are_documented():
    catalog = saved_catalog()
    assert "PANIC" in catalog["openecon.xtpanic"]["description"]
    assert "Toda" in catalog["openecon.tycausality"]["description"]
    for name, expected in (("xtpanic", {"factors": "1", "inference": "'asymptotic'"}),
                           ("tycausality", {"dmax": "1", "joint": "True"})):
        params = {p["name"]: p for p in catalog[f"openecon.{name}"]["parameters"]}
        for param, default in expected.items():
            assert params[param]["default"] == default
    for name in ("predict", "margins"):
        kind = next(p for p in catalog[f"openecon.{name}"]["parameters"] if p["name"] == "kind")
        assert "sigma" in kind["description"] and "default" not in kind


def test_export_reader_refuses_calls_or_user_code():
    expression = ast.parse("{'function': __import__('os').system('false')}", mode="eval").body
    try:
        generator.export_literal(expression)
    except ValueError as error:
        assert "Unsupported export syntax" in str(error)
    else:
        raise AssertionError("Export metadata cannot execute source expressions")


def test_econometrics_catalog_covers_lazy_public_targets_and_exact_parameters():
    from openecon.econometrics import registry

    reader = generator.SourceReader("HEAD")
    exports = generator.registry_exports(reader)
    assert {name: target[:2] for name, target in exports.items()} == registry.public_exports()
    catalog = saved_catalog()
    for name, (module, original, _) in exports.items():
        node = generator.function_node(
            reader.committed("src/" + module.replace(".", "/") + ".py").body, original,
        )
        item = catalog[f"openecon.{name}"]
        assert item["signature"] == f"{name}({ast.unparse(node.args)})"
        assert [parameter["name"] for parameter in item["parameters"]] == [
            parameter["name"] for parameter in generator.parameter_list(node.args, {})
        ]
    for name in ("nardl", "tobit", "truncreg", "intreg", "xtreg", "meprobit"):
        assert catalog[f"openecon.{name}"]["returns"] == "openecon.ResultBundle"


def test_export_reader_allows_literal_templates_but_never_evaluates_expressions():
    node = ast.parse("f'{package}.test:helper'", mode="eval").body
    assert generator.export_literal(node, {"package": "openecon.econometrics.panel"}) == (
        "openecon.econometrics.panel.test:helper"
    )
    for expression in ("f'{__import__(\"os\").system(\"false\")}'", "f'{package!r}'", "f'{package:10}'"):
        try:
            generator.export_literal(ast.parse(expression, mode="eval").body, {"package": "trusted"})
        except ValueError:
            pass
        else:
            raise AssertionError("Export templates cannot execute or format expressions")


def test_registry_reader_refuses_a_factory_with_executable_statements():
    class UntrustedReader:
        def committed(self, path):
            if path.endswith("registry.py"):
                return ast.parse("FAMILIES = ('example',)")
            return ast.parse('''
def unsafe(name):
    __import__('os').system('false')
    return EstimatorInfo(function=name, entry='openecon.econometrics.example:fit')
ESTIMATORS = (unsafe('model'),)
EXPORTS = {}
''')

    try:
        generator.registry_exports(UntrustedReader())
    except ValueError as error:
        assert "single-return" in str(error)
    else:
        raise AssertionError("Manifest factories cannot execute project code")


@pytest.mark.parametrize("call", [
    "factory('wrong', name='model')", "factory('model', unexpected=12)",
    "factory('model', 'ignored')", "factory('model', **__import__('os').environ)",
    "factory(*['model'])", "factory()", "factory(name='a', name='b')",
])
def test_registry_reader_rejects_invalid_factory_call_shapes(call):
    class Reader:
        def committed(self, path):
            if path.endswith("registry.py"):
                return ast.parse("FAMILIES = ('example',)")
            return ast.parse(f'''
def factory(name):
    return EstimatorInfo(function=name, entry='openecon.econometrics.example:fit')
ESTIMATORS = ({call},)
EXPORTS = {{}}
''')

    with pytest.raises(ValueError, match="factory"):
        generator.registry_exports(Reader())


def test_registry_reader_obeys_positional_only_keyword_only_and_literal_defaults():
    class Reader:
        def committed(self, path):
            if path.endswith("registry.py"):
                return ast.parse("FAMILIES = ('example',)")
            return ast.parse('''
def factory(name, /, unused='default', *, module='openecon.econometrics.example'):
    return EstimatorInfo(function=name, entry=f'{module}:fit')
ESTIMATORS = (factory('model'),)
EXPORTS = {}
''')

    assert generator.registry_exports(Reader())["model"][:2] == (
        "openecon.econometrics.example", "model",
    )


@pytest.mark.parametrize("name", ["ols", "predict", "ModelSpec"])
def test_registry_help_cannot_overwrite_a_core_export(monkeypatch, name):
    monkeypatch.setattr(generator, "registry_exports", lambda reader: {
        name: ("openecon.econometrics.example", "unused", None),
    })
    with pytest.raises(ValueError, match="collide with the core API"):
        generator.generate("HEAD")


def test_parameter_choices_follow_committed_statistical_contracts():
    reader = generator.SourceReader("HEAD")
    choices = generator.parameter_choices(reader)
    catalog = saved_catalog()
    constants = {
        target.id: ast.literal_eval(node.value)
        for node in reader.committed("src/openecon/linear_ols/spec.py").body
        if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
        and target.id in {"COVARIANCES", "WEIGHTS", "KERNELS"}
    }
    assert [ast.literal_eval(value) for value in choices["openecon.ols"]["covariance"]] == list(constants["COVARIANCES"])
    assert [ast.literal_eval(value) for value in choices["openecon.ols"]["kernel"]] == list(constants["KERNELS"])
    assert choices["openecon.logit"]["covariance"] == ["'cluster'", "'nonrobust'"]
    assert choices["openecon.probit"]["covariance"] == choices["openecon.logit"]["covariance"]
    assert choices["openecon.ols"]["missing"] == ["'raise'", "'drop'"]
    for name, parameters in choices.items():
        for parameter in catalog[name]["parameters"]:
            if parameter["name"] in parameters:
                assert parameter["choices"] == parameters[parameter["name"]]
                assert all(isinstance(ast.literal_eval(value), str) for value in parameter["choices"])


def test_network_workflow_completion_preserves_chained_result_types():
    catalog = saved_catalog()
    assert catalog['openecon.Network.max_flow']['returns'] == 'openecon.NetworkFlowResult'
    assert catalog['openecon.Network.min_cut']['returns'] == 'openecon.NetworkFlowResult'
    assert catalog['openecon.NetworkFlowResult.summary']['returns'] == 'openecon.DataFrame'
    assert catalog['openecon.Network.bipartite_projection']['returns'] == 'openecon.Network'
    assert catalog['openecon.Network.minimum_spanning_forest']['returns'] == 'openecon.Network'
    assert catalog['openecon.Network.link_prediction']['returns'] == 'openecon.DataFrame'
    assert catalog['openecon.Network.shortest_path']['returns'] == 'openecon.DataFrame'


def test_network_inference_completion_exposes_result_tables_and_bounded_cut_summary():
    catalog = saved_catalog()
    assert catalog['openecon.Network.global_min_cut']['returns'] == 'openecon.NetworkCutResult'
    assert catalog['openecon.NetworkCutResult.summary']['returns'] == 'openecon.DataFrame'
    assert catalog['openecon.Network.triad_census']['returns'] == 'openecon.DataFrame'
    assert catalog['openecon.Network.qap_correlation']['returns'] == 'openecon.DataFrame'


def test_network_models_have_exact_signatures_and_bounded_chain_returns():
    catalog = saved_catalog()
    assert catalog['openecon.network_snapshots']['returns'] == 'openecon.NetworkSnapshots'
    assert catalog['openecon.Network.qap_regression']['returns'] == 'openecon.DataFrame'
    assert catalog['openecon.Network.block_model']['returns'] == 'openecon.NetworkBlockResult'
    for cls in ('NetworkSnapshots','NetworkBlockResult'):
        assert catalog[f'openecon.{cls}.summary']['returns'] == 'openecon.DataFrame'
        assert catalog[f'openecon.{cls}.to_latex']['kind'] == 'method'
    assert catalog['openecon.NetworkSnapshots.aggregate']['returns'] == 'openecon.Network'
    for method in ('snapshot_summary','transitions','edge_persistence','temporal_path'):
        assert catalog[f'openecon.NetworkSnapshots.{method}']['returns'] == 'openecon.DataFrame'
    params = {p['name']:p for p in catalog['openecon.Network.qap_regression']['parameters']}
    assert params['permutations']['default'] == '999'
    assert params['intercept']['default'] == 'True'
    assert 'Freedman' in catalog['openecon.Network.qap_regression']['description']


def test_count_models_and_selected_pair_predictions_chain_to_tables():
    catalog = saved_catalog()
    for method in ('poisson_block_model', 'degree_corrected_block_model'):
        entry = catalog[f'openecon.Network.{method}']
        assert entry['returns'] == 'openecon.NetworkBlockResult'
        assert {p['name'] for p in entry['parameters']} == {
            'groups', 'initial', 'seed', 'starts', 'max_iter', 'tol', 'max_work'}
        assert 'Poisson' in entry['description']
    predicted = catalog['openecon.NetworkBlockResult.expected_edges']
    assert predicted['returns'] == 'openecon.DataFrame'
    parameters = {p['name']:p for p in predicted['parameters']}
    assert parameters['max_pairs']['default'] == '100000'
    assert parameters['max_memory_mb']['default'] == '256'
    assert 'presence' in predicted['description']


def test_advanced_method_help_has_explicit_offline_parameters():
    catalog = saved_catalog()
    for name in ("fmols", "dols", "ccr", "panel_fmols", "panel_dols", "pmg", "mg", "dfe", "svar", "lp", "lpiv", "panel_lp", "mediation", "oaxaca"):
        item = catalog[f"openecon.{name}"]
        assert item["returns"] == "openecon.ResultBundle"
        params = {p["name"] for p in item["parameters"]}
        assert {"data", "y"} <= params
        assert "kwargs" not in params
    assert "instruments" in {p["name"] for p in catalog["openecon.lpiv"]["parameters"]}
    assert "pooling" in {p["name"] for p in catalog["openecon.panel_dols"]["parameters"]}
