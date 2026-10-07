"""Disposable team QA checks real native exports without contacting Cloud."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import httpx

_SPEC = importlib.util.spec_from_file_location(
    'team_latex_verify', Path(__file__).parents[1] / 'scripts/verify_team_live.py')
verify = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify)


@pytest.fixture
def native_analysis(tmp_path, monkeypatch):
    from openecon.console_worker import _execute

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'team-data.csv').write_bytes(b'x,y\n1,2\n2,4\n3,6\n')
    result = _execute(verify.QA_ANALYSIS_CODE, {'__name__': '__main__'}, 'qa-local-latex')
    assert result['status'] == 'ok', result['error']
    result['state_reset'] = True
    result['artifacts'] = [
        {'name': 'qa-dataframe.tex', 'url': '/api/projects/own/workspace/artifacts/tex'},
        {'name': 'qa-output.csv', 'url': '/api/projects/own/workspace/artifacts/csv'},
        {'name': 'qa-regression.tex', 'url': '/api/projects/own/workspace/artifacts/regression'},
    ]
    return (result, (tmp_path / 'qa-output.csv').read_bytes(),
            (tmp_path / 'qa-dataframe.tex').read_bytes(),
            (tmp_path / 'qa-regression.tex').read_bytes())


def test_real_qa_code_produces_native_frame_model_figure_and_readable_tex_artifact(native_analysis):
    verify.verify_latex_analysis(*native_analysis)
    result, _, _, _ = native_analysis
    assert verify.artifact_named(result, 'qa-output.csv')['url'].endswith('/csv')
    assert verify.artifact_named(result, 'qa-dataframe.tex')['url'].endswith('/tex')


@pytest.mark.parametrize('name', ['missing.tex', 'qa-output.csv'])
def test_named_artifacts_reject_missing_or_duplicate_files_without_relying_on_order(name):
    result = {'artifacts': [{'name': 'qa-output.csv'}, {'name': 'qa-output.csv'}]}
    with pytest.raises(AssertionError):
        verify.artifact_named(result, name)


@pytest.mark.parametrize('field,value', [('latex', {}), ('latex', ''), ('latex_math', []), ('latex_math', 42)])
def test_qa_rejects_invalid_json_latex_field_types(native_analysis, field, value):
    result, csv_bytes, tex_bytes, regression_bytes = native_analysis
    result = deepcopy(result)
    result['outputs'][0][field] = value
    with pytest.raises(AssertionError):
        verify.verify_latex_analysis(result, csv_bytes, tex_bytes, regression_bytes)


def test_qa_rejects_truncated_changed_tex_artifact(native_analysis):
    result, csv_bytes, tex_bytes, regression_bytes = native_analysis
    with pytest.raises(AssertionError):
        verify.verify_latex_analysis(result, csv_bytes, tex_bytes[:-10], regression_bytes)


def test_qa_rejects_plot_coordinates_that_disagree_with_model(native_analysis):
    result, csv_bytes, tex_bytes, regression_bytes = native_analysis
    result = deepcopy(result)
    result['outputs'][2]['data']['data'][0]['estimate'] += 1
    with pytest.raises(AssertionError):
        verify.verify_latex_analysis(result, csv_bytes, tex_bytes, regression_bytes)


def test_qa_rejects_missing_math_preview_field(native_analysis):
    result, csv_bytes, tex_bytes, regression_bytes = native_analysis
    result = deepcopy(result)
    del result['outputs'][0]['latex_math']
    with pytest.raises(AssertionError):
        verify.verify_latex_analysis(result, csv_bytes, tex_bytes, regression_bytes)


def test_qa_rejects_changed_multi_model_artifact(native_analysis):
    result, csv_bytes, tex_bytes, regression_bytes = native_analysis
    with pytest.raises(AssertionError):
        verify.verify_latex_analysis(result, csv_bytes, tex_bytes, regression_bytes[:-10])


@pytest.mark.parametrize('method,code,content_type,retries', [
    ('GET', 401, 'text/html', True),
    ('GET', 401, 'application/json', False),
    ('GET', 403, 'text/html', False),
    ('POST', 401, 'text/html', False),
    ('PATCH', 401, 'text/html', False),
    ('DELETE', 401, 'text/html', False),
])
def test_platform_retry_never_repeats_a_mutation_or_bypasses_application_auth(
    monkeypatch, method, code, content_type, retries,
):
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(code, headers={'Content-Type': content_type},
                                  text='<title>401 Unauthorized</title>')
        return httpx.Response(200, json={'ok': True})

    monkeypatch.setattr(verify.time, 'sleep', lambda _: None)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        response = verify.platform_request(client, method, 'https://qa.example/api/me')
    assert response.status_code == (200 if retries else code)
    assert len(requests) == (2 if retries else 1)
