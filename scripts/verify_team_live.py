"""Authorized integration QA using disposable test identities, never an owner token.

Creates no emails. Test credentials live only in an ignored, mode-0600 local file
until explicit cleanup. All cloud deletion is restricted to these exact test IDs.
"""
import argparse
import json
from pathlib import Path
import secrets
import subprocess
import time
from uuid import uuid4

import firebase_admin
from firebase_admin import auth, credentials
from google.cloud import firestore, storage
from google.oauth2.credentials import Credentials
import httpx

PROJECT = 'openecon-workbench'
STATE = Path('artifacts/team-setup/qa-state.json')
QA_ANALYSIS_CODE = '''import openecon as oe
import openecon_charts as charts
import pandas as pd
from pathlib import Path
df = oe.example()
assert isinstance(df, pd.DataFrame) and isinstance(df, oe.DataFrame)
head = df.head()
assert isinstance(head, oe.DataFrame)
pd.testing.assert_frame_equal(pd.DataFrame(head), pd.DataFrame(df.iloc[:5]))
display(head)
model = oe.ols(data=df, y="wage", x=["education", "experience"], covariance="HC3")
display(model)
figure = charts.coefficients(model, title="Ekip analizi · %95 güven aralığı")
display(figure)
frame_source = head.to_latex(index=False)
assert frame_source == oe.to_latex(pd.DataFrame(head), index=False)
display(frame_source)
display(model.to_latex())
display(oe.Latex(figure.to_latex()))
reduced = oe.ols(data=df, y="wage", x=["education"], covariance="HC3")
comparison = oe.regression_table([model, reduced], caption="Wage regressions")
display(comparison)
assert oe.regression_table([model, reduced], "qa-regression.tex", caption="Wage regressions") is None
assert Path("qa-regression.tex").read_text(encoding="utf-8") == comparison
assert head.to_latex("qa-dataframe.tex", index=False) is None
assert Path("qa-dataframe.tex").read_text(encoding="utf-8") == frame_source
assert oe.read("team-data.csv").shape == (3, 2)
Path("qa-output.csv").write_text("result\\npassed\\n")
print("fresh-job-analysis-ok")
print("native-frame-latex-ok")
'''


def artifact_named(result, name):
    """Pick an exact, unique disposable artifact without relying on ordering."""
    artifacts = [artifact for artifact in result.get('artifacts', [])
                 if artifact.get('name') == name]
    assert len(artifacts) == 1, f'Expected one QA artifact named {name}.'
    return artifacts[0]


def verify_latex_analysis(result, csv_bytes, tex_bytes, regression_bytes):
    """Check actual model/data/figure export values and the durable .tex file."""
    from openecon.latex import escape_latex

    assert result['status'] == 'ok', result.get('error')
    assert result['state_reset'] is True
    assert 'native-frame-latex-ok' in result['stdout'].splitlines()
    assert csv_bytes == b'result\npassed\n'
    assert isinstance(tex_bytes, bytes)
    frame_source = tex_bytes.decode('utf-8')
    outputs = result['outputs']
    assert [item['type'] for item in outputs] == ['table', 'model', 'plot', 'latex', 'latex', 'latex', 'latex']
    for item in outputs:
        assert isinstance(item.get('latex'), str) and item['latex'].strip()
        assert 'latex_math' in item and (item['latex_math'] is None or isinstance(item['latex_math'], str))
    table, model, plot, frame_export, model_export, plot_export, comparison = outputs
    for item in [table, model]:
        assert isinstance(item['latex_math'], str) and r'\begin{array}' in item['latex_math']
    assert plot['latex_math'] is None
    for item in [frame_export, model_export, plot_export, comparison]:
        assert isinstance(item['data'], str) and item['data'] == item['latex']
    for item in [frame_export, model_export]:
        if item['latex_math'] is not None:
            assert r'\begin{array}' in item['latex_math']
    assert plot_export['latex_math'] is None
    assert frame_export['data'] == frame_source
    assert model_export['data'] == model['latex']
    assert plot_export['data'] == plot['latex']
    assert r'\begin{tabular}' in frame_source and r'\end{tabular}' in frame_source
    assert table['data']['total_rows'] == len(table['data']['rows']) == 5
    assert table['data']['total_columns'] == len(table['data']['columns'])
    assert 'wage' in table['data']['columns']
    for column in table['data']['columns']:
        assert escape_latex(column) in frame_source
    wage_column = table['data']['columns'].index('wage')
    for row in table['data']['rows']:
        assert f'{row[wage_column]:.4f}' in frame_source
    assert model['data']['nobs'] == 480
    assert '480' in model['latex']
    assert model['data']['spec']['covariance'] == 'HC3'
    assert 'HC3' in model['latex']
    for item in [table, model]:
        assert item['latex_style'] in ('publication-v1', 'publication-v2')
    for source in [frame_source, model['latex'], comparison['latex']]:
        assert r'\toprule' in source and r'\midrule' in source and r'\bottomrule' in source
        assert r'\hline' not in source
    assert isinstance(regression_bytes, bytes)
    assert comparison['latex'] == regression_bytes.decode('utf-8')
    assert 'Wage regressions' in comparison['latex']
    assert '(1)' in comparison['latex'] and '(2)' in comparison['latex']
    assert 'Dependent variable:' in comparison['latex']
    assert 'Standard errors are in parentheses.' in model['latex']
    assert '* p < 0.10; ** p < 0.05; *** p < 0.01' in model['latex']
    for coefficient in model['data']['coefficients']:
        term = coefficient['term']
        label = 'Constant' if term.casefold() in {'intercept', 'constant', 'const', '_cons'} else term
        assert escape_latex(label) in model['latex']
        assert f"{coefficient['estimate']:.4f}" in model['latex']
        assert f"({coefficient['std_error']:.4f})" in model['latex']
        p_value = coefficient['p_value']
        stars = '***' if p_value < .01 else '**' if p_value < .05 else '*' if p_value < .1 else ''
        expected = f"{coefficient['estimate']:.4f}" + ('^{' + stars + '}' if stars else '')
        assert expected in model['latex'] and expected in comparison['latex']
    assert plot['data']['kind'] == 'coefficients'
    assert r'\begin{tikzpicture}' in plot['latex'] and r'\end{tikzpicture}' in plot['latex']
    assert escape_latex(plot['data']['title']) in plot['latex']
    assert len(plot['data']['data']) == len(model['data']['coefficients'])
    for index, (point, coefficient) in enumerate(zip(plot['data']['data'], model['data']['coefficients'], strict=True)):
        for key in ['term', 'estimate', 'ci_low', 'ci_high']:
            assert point[key] == coefficient[key]
        assert f"({point['estimate']!r},{index})" in plot['latex']
        for endpoint in ['ci_low', 'ci_high']:
            assert f"axis cs:{point[endpoint]!r},{index}" in plot['latex']


def ensure_cleanup_idle(db, project_ids):
    """Read all disposable projects before deleting any project or identity."""
    terminal = {'finished', 'failed', 'cancelled'}
    for project_id in project_ids:
        ref = db.document('oe_projects/' + project_id)
        project = ref.get().to_dict() or {}
        if project.get('active_run'):
            raise SystemExit('QA cleanup refused: a disposable analysis is still active. Stop it and wait for completion.')
        for snapshot in ref.collection('runs').stream():
            run = snapshot.to_dict() or {}
            if run.get('state') not in terminal:
                raise SystemExit('QA cleanup refused: a disposable analysis is still active. Stop it and wait for completion.')


def _cleanup_ids(values):
    if not isinstance(values, list) or any(
        not isinstance(value, str) or len(value) != 32
        or any(character not in '0123456789abcdef' for character in value)
        for value in values
    ):
        raise SystemExit('QA cleanup refused: invalid disposable resource manifest.')
    return set(values)


def _cleanup_document(snapshot):
    document = snapshot.to_dict()
    if document is not None and not isinstance(document, dict):
        raise SystemExit('QA cleanup refused: resource ownership could not be established.')
    return document


def _inspect_cleanup_users(db, user_ids, project_ids):
    """Use both profile indexes and authoritative projects; neither is sufficient alone.

    A complete project scan also finds memberships whose profile index is stale or
    missing. Finish every read before returning, including the full stream, so a
    partial query result can never authorize identity deletion.
    """
    protected, profiles = set(), {}
    for user_id in user_ids:
        profile = _cleanup_document(db.document('oe_users/' + user_id).get())
        profiles[user_id] = profile
        indexed = profile.get('project_ids', []) if profile is not None else []
        indexed = _cleanup_ids(indexed)
        if indexed - project_ids:
            protected.add(user_id)
    for snapshot in db.collection('oe_projects').stream():
        project = _cleanup_document(snapshot)
        if project is None:
            raise SystemExit('QA cleanup refused: resource ownership could not be established.')
        members, owner = project.get('members'), project.get('owner_uid')
        if not isinstance(members, dict) or not isinstance(owner, str):
            raise SystemExit('QA cleanup refused: resource ownership could not be established.')
        project_id = snapshot.id
        _cleanup_ids([project_id])
        associated = set(members) | {owner}
        if project_id not in project_ids:
            protected.update(associated & user_ids)
        elif owner not in user_ids or associated - user_ids:
            # An adopted QA project may contain human data even if its ID was
            # originally recorded by this script. Require explicit review.
            raise SystemExit('QA cleanup refused: a disposable project has non-QA ownership or membership.')
    return protected, profiles


def cleanup_qa(db, blobs, app, state):
    """Delete only manifest resources; retain accounts adopted for other projects.

    This is an administrative cleanup, not an ownership transfer. A retained
    identity and its private credential manifest remain available for a later
    explicit transfer to the user's verified Firebase UID.
    """
    try:
        project_ids = _cleanup_ids(state['projects'])
        invitation_ids = _cleanup_ids(state['invites'])
        grant_ids = _cleanup_ids(state.get('desktop_login_ids', []))
        users = state['users']
        if not isinstance(users, dict) or any(
            not isinstance(user, dict) or not isinstance(user.get('uid'), str)
            or not user['uid'] or len(user['uid']) > 128
            or '/' in user['uid'] or any(ord(character) < 32 for character in user['uid'])
            for user in users.values()
        ):
            raise SystemExit('QA cleanup refused: invalid disposable identity manifest.')
        user_ids = {user['uid'] for user in users.values()}
        protected, _ = _inspect_cleanup_users(db, user_ids, project_ids)
        ensure_cleanup_idle(db, sorted(project_ids))
        # Validate every recorded related resource before making the first write.
        for invitation_id in invitation_ids:
            invitation = _cleanup_document(db.document('oe_invitations/' + invitation_id).get())
            if invitation is not None and invitation.get('project_id') not in project_ids:
                raise SystemExit('QA cleanup refused: an invitation belongs to a non-QA project.')
        for request_id in grant_ids:
            grant = _cleanup_document(db.document('oe_desktop_logins/' + request_id).get())
            if grant is not None and grant.get('uid') is not None and grant['uid'] not in user_ids:
                raise SystemExit('QA cleanup refused: a desktop grant belongs to a non-QA identity.')

        for request_id in sorted(grant_ids):
            db.document('oe_desktop_logins/' + request_id).delete()
        for project_id in sorted(project_ids):
            db.recursive_delete(db.document('oe_projects/' + project_id))
            for prefix in [f'projects/{project_id}/', f'staging/{project_id}/']:
                for blob in blobs.list_blobs(prefix=prefix):
                    blob.delete(if_generation_match=int(blob.generation))
        for invitation_id in sorted(invitation_ids):
            db.document('oe_invitations/' + invitation_id).delete()

        # Recheck before removing accounts, retaining anyone newly associated
        # since preflight as well as anyone protected by the original snapshot.
        newly_protected, profiles = _inspect_cleanup_users(db, user_ids, project_ids)
        protected.update(newly_protected)
        removed, pruned = 0, 0
        for user_id in sorted(user_ids):
            profile_ref = db.document('oe_users/' + user_id)
            if user_id in protected:
                profile = profiles[user_id]
                if profile is not None and project_ids.intersection(profile.get('project_ids', [])):
                    # A transform preserves concurrent additions and every other
                    # profile field; never replace a human's complete index.
                    profile_ref.update({'project_ids': firestore.ArrayRemove(sorted(project_ids))})
                    pruned += 1
                continue
            try:
                auth.delete_user(user_id, app=app)
            except auth.UserNotFoundError:
                pass
            profile_ref.delete()
            removed += 1
        return {'checks': state.get('checks', []), 'projects_removed': len(project_ids),
                'identities_removed': removed, 'identities_preserved': len(protected),
                'preserved_profiles_pruned': pruned, 'state_retained': bool(protected),
                'url': state.get('url')}
    except Exception:
        # SDK errors may contain private document paths or credential details.
        # Keep the private manifest on all failures and never print raw errors.
        raise SystemExit('QA cleanup stopped: resources could not be safely inspected or removed. Private QA state was retained.') from None


def platform_request(client, method, url, **kwargs):
    """Retry only read-only requests rejected by Cloud Run before the app.

    Firebase/application JSON errors and every mutation are returned immediately.
    This allows a fresh public service's IAM propagation to settle without
    repeating an accepted analysis, creating projects or altering membership.
    """
    for attempt in range(5):
        response = client.request(method, url, **kwargs)
        transient = (method == 'GET' and response.status_code == 401
                     and 'text/html' in response.headers.get('content-type', '')
                     and '<title>401 Unauthorized</title>' in response.text)
        if not transient or attempt == 4:
            return response
        time.sleep(2)
    raise AssertionError('Unreachable retry state.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'verify', 'cleanup'])
    parser.add_argument('--url', default='https://openecon-teams-preview-291739190496.us-central1.run.app')
    parser.add_argument('--resume-run', help='Read back an already accepted QA execution before member cleanup')
    args = parser.parse_args()
    token = subprocess.check_output(['gcloud', 'auth', 'print-access-token', '--project=' + PROJECT], text=True).strip()
    credential = Credentials(token, quota_project_id=PROJECT)

    class AdminCredential(credentials.Base):
        def get_credential(self):
            return credential

    app = firebase_admin.initialize_app(AdminCredential(), {'projectId': PROJECT})
    db = firestore.Client(project=PROJECT, credentials=credential)
    blobs = storage.Client(project=PROJECT, credentials=credential).bucket(PROJECT + '-projects')
    if args.action == 'prepare':
        if STATE.exists():
            raise SystemExit('Existing QA state must be inspected or cleaned before another prepare.')
        state = {'users': {}, 'projects': [], 'invites': [], 'checks': []}
        STATE.parent.mkdir(exist_ok=True, parents=True)
        STATE.touch(mode=0o600)
        def save():
            STATE.write_text(json.dumps(state, indent=2))
        save()
        for role in ['owner', 'editor', 'viewer', 'outsider', 'unverified']:
            email = f'openecon-qa-{role}-{uuid4().hex[:12]}@example.com'
            password = 'Qa9-' + secrets.token_urlsafe(30)
            user = auth.create_user(email=email, password=password, email_verified=role != 'unverified',
                                    display_name=f'QA {role.title()}', app=app)
            state['users'][role] = {'uid': user.uid, 'email': email, 'password': password}
            save()
        owner = state['users']['owner']
        db.document('oe_users/' + owner['uid']).set({'uid': owner['uid'], 'email': owner['email'],
            'name': 'QA Owner', 'enabled': True, 'project_ids': [], 'created_at': '2026-10-01T00:00:00+00:00'})
        print('Prepared five disposable QA identities; no emails sent.')
        return
    state = json.loads(STATE.read_text())
    if args.action == 'cleanup':
        report = cleanup_qa(db, blobs, app, state)
        report_path = Path('artifacts/verification/team-live-qa.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        if not report['state_retained']:
            STATE.unlink()
        print(json.dumps(report, indent=2))
        return

    if args.resume_run and (state.get('url') != args.url.rstrip('/') or not state['projects']):
        raise SystemExit('Resume must use the same owned QA project and origin.')
    state['url'] = args.url.rstrip('/')
    client = httpx.Client(timeout=330, follow_redirects=False)
    config = client.get(state['url'] + '/api/auth/config').json()['firebase']
    tokens = {}
    for role, user in state['users'].items():
        response = client.post('https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword',
            params={'key': config['apiKey']}, json={'email': user['email'], 'password': user['password'],
                                                 'returnSecureToken': True})
        if response.status_code != 200:
            raise RuntimeError(f'QA sign-in failed: {role}, HTTP {response.status_code}')
        tokens[role] = response.json()['idToken']

    def save():
        STATE.write_text(json.dumps(state, indent=2))

    def check(name):
        state['checks'].append(name)
        save()
        print('PASS', name, flush=True)

    def call(method, path, role='owner', status=200, **kwargs):
        response = platform_request(client, method, state['url'] + '/api' + path,
            headers={'Authorization': 'Bearer ' + tokens[role], 'Origin': state['url']}, **kwargs)
        if response.status_code != status:
            # Only our API's safe response is printed; never provider tokens.
            raise AssertionError(f'{method} {path}: HTTP {response.status_code}: {response.text[:600]}')
        return response.json() if 'application/json' in response.headers.get('content-type', '') else response.content

    started = time.monotonic()
    if args.resume_run:
        project_id = state['projects'][-1]
        workspace = f'/projects/{project_id}/workspace'
        result = call('GET', workspace + '/console', 'viewer')['history'][-1]
        assert result['id'] == args.resume_run and result['code'] == QA_ANALYSIS_CODE
        assert result['status'] == 'ok', result.get('error')
    else:
        assert client.get(state['url'] + '/api/projects').status_code == 401
        check('Anonymous project API denied; login page available')
        assert call('GET', '/me', 'unverified')['user']['email_verified'] is False
        call('GET', '/projects', 'unverified', status=403)
        check('Unverified email cannot access projects')
        for role in state['users']:
            call('GET', '/me', role)
        call('POST', '/projects', 'outsider', status=403, json={'name': 'Unauthorized'})
        check('Uninvited account cannot create a project')
        project = call('POST', '/projects', status=201, json={'name': 'Ekip doğrulama', 'description': 'Geçici QA projesi'})
        project_id = project['id']
        state['projects'].append(project_id)
        save()
        workspace = f'/projects/{project_id}/workspace'
        call('GET', workspace + '/console', 'outsider', status=404)
        for role in ['editor', 'viewer']:
            invite = call('POST', f'/projects/{project_id}/invitations', status=201,
                          json={'email': state['users'][role]['email'], 'role': role})
            state['invites'].append(invite['id'])
            save()
            call('POST', '/invitations/' + invite['id'] + '/accept', 'outsider', status=404)
            call('POST', '/invitations/' + invite['id'] + '/accept', role)
            call('POST', '/invitations/' + invite['id'] + '/accept', role, status=404)
        check('Email-bound invitations accepted once; cross-project access denied')
        script = 'import openecon as oe\ndf = oe.example()\ndisplay(df.head())\n'
        call('PUT', workspace + '/console/script', 'editor', json={'code': script, 'version': 0})
        call('PUT', workspace + '/console/script', 'owner', status=409, json={'code': 'stale', 'version': 0})
        call('PUT', workspace + '/console/script', 'viewer', status=403, json={'code': 'forbidden', 'version': 1})
        assert call('GET', workspace + '/console/script', 'viewer')['code'] == script
        check('Shared draft persists; stale writes and viewer edits denied')
        dataset = call('POST', workspace + '/datasets/upload', 'editor', status=201,
                       files={'file': ('team-data.csv', b'x,y\n1,2\n2,4\n3,6\n', 'text/csv')})
        assert call('GET', workspace + '/files/' + dataset['id'] + '/download', 'viewer') == b'x,y\n1,2\n2,4\n3,6\n'
        call('POST', workspace + '/console/execute', 'viewer', status=403, json={'code': 'print(1)'})
        check('Project upload/download persist; viewer cannot execute Python')
        started = time.monotonic()
        result = call('POST', workspace + '/console/execute', 'editor', json={'code': QA_ANALYSIS_CODE})
        assert result['status'] == 'ok', result.get('error')
    csv_artifact = artifact_named(result, 'qa-output.csv')
    tex_artifact = artifact_named(result, 'qa-dataframe.tex')
    regression_artifact = artifact_named(result, 'qa-regression.tex')
    csv_bytes = call('GET', csv_artifact['url'].removeprefix('/api'), 'viewer')
    tex_bytes = call('GET', tex_artifact['url'].removeprefix('/api'), 'viewer')
    regression_bytes = call('GET', regression_artifact['url'].removeprefix('/api'), 'viewer')
    verify_latex_analysis(result, csv_bytes, tex_bytes, regression_bytes)
    timing = ('saved execution readback' if args.resume_run else 'analysis and downloads')
    check(f'Isolated Cloud Run OLS + D3 + publication tables + multi-model TeX + uploaded input + durable exports passed ({timing}: {time.monotonic()-started:.1f}s)')
    saved = call('GET', workspace + '/console', 'viewer')['history'][-1]
    assert saved['id'] == result['id']
    verify_latex_analysis(saved, csv_bytes, tex_bytes, regression_bytes)
    check('Saved execution, LaTeX source, chart and CSV/TeX files read back by another member')
    call('PATCH', f'/projects/{project_id}/members/' + state['users']['editor']['uid'],
         json={'role': 'viewer'})
    call('POST', workspace + '/console/execute', 'editor', status=403, json={'code': 'print(1)'})
    call('DELETE', f'/projects/{project_id}/members/' + state['users']['viewer']['uid'])
    call('GET', workspace + '/console', 'viewer', status=404)
    check('Role changes and member removal take effect on the next request')
    print('Live integration finished; QA identities remain for UI verification and explicit cleanup.', flush=True)


if __name__ == '__main__':
    main()
