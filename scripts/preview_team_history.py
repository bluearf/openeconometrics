"""Render the real workbench against 137 owned synthetic persisted team runs."""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
from openecon.team_auth import TeamAuth, TeamIdentity
from openecon.team_server import create_team_app, json_bytes
from openecon.team_storage import MemoryStorage
from openecon.team_store import MemoryDocuments, TeamStore

ROOT = Path(__file__).resolve().parents[1]
ORIGIN, PROJECT = 'https://history-preview.example.com', 'openecon-history-preview'


def main():
    class Storage(MemoryStorage):
        reads = 0
        def get(self, reference, maximum=24 * 1024**2):
            self.reads += 1
            return super().get(reference, maximum)
    storage, documents = Storage(), MemoryDocuments()
    store = TeamStore(documents, owner_email='owner@example.com', public_origin=ORIGIN)
    owner = TeamIdentity('owner', 'owner@example.com', 'Owner', True)
    viewer = TeamIdentity('viewer', 'viewer@example.com', 'Viewer', True)
    store.me(owner)
    pid = store.create_project(owner, 'History verification')['id']
    invitation = store.invite(pid, owner, viewer.email, 'viewer')
    store.accept(invitation['id'], viewer)
    code = 'print("preserved editor buffer")'
    store.save_script(pid, owner, code, 0)
    for index in range(137):
        run_id = f'{index:032x}'
        run = dict(id=run_id, created_at=f'2026-10-01T00:{index // 60:02d}:{index % 60:02d}+00:00',
                   code=f'print("legacy_marker_{index}")', state='finished', generation=index + 1,
                   email=owner.email, record_summary={'status': 'ok'})
        record = dict(id=run_id, created_at=run['created_at'], code=run['code'], status='ok',
                      stdout=f'Original saved output #{index}\n', outputs=[{'type': 'text', 'data': f'Original saved output #{index}'}],
                      error=None, variables=[], duration_ms=12, session_generation=index + 1)
        run['result'] = storage.put(f'synthetic/record-{index}.json', json_bytes(record))
        documents.put(f'oe_projects/{pid}/runs/{run_id}', run)
    def verify(token):
        if token != 'fixture-viewer':
            raise ValueError('Invalid fixture token')
        return dict(uid=viewer.uid, sub=viewer.uid, email=viewer.email, name='Viewer', aud=PROJECT,
                    iss=f'https://securetoken.google.com/{PROJECT}', firebase={'sign_in_provider': 'password'}, email_verified=True)
    app = create_team_app(store=store, storage=storage, auth=TeamAuth(PROJECT, verifier=verify),
                          public_origin=ORIGIN,
                          firebase_config={'projectId': PROJECT, 'authDomain': f'{PROJECT}.firebaseapp.com'})
    with tempfile.TemporaryDirectory(prefix='openecon-history-preview-') as temporary, TestClient(app, base_url=ORIGIN) as client:
        directory = Path(temporary)
        entry = directory / 'entry.tsx'
        entry.write_text('import {createRoot} from "react-dom/client";\nimport App from ' + json.dumps(str(ROOT / 'web/src/App.tsx'))
            + ';\nimport {createWorkspaceClient} from ' + json.dumps(str(ROOT / 'web/src/api.ts'))
            + ';\ncreateRoot(document.getElementById("app")).render(<App client={createWorkspaceClient('
            + json.dumps(pid) + ',async()=>"fixture-viewer")} syncOnly readOnly projectName="History verification"/>);')
        builder = '''import {pathToFileURL} from "node:url";
const [root,entry,outfile]=process.argv.slice(2);
const {build}=await import(pathToFileURL(root+"/web/node_modules/esbuild/lib/main.js").href);
await build({entryPoints:[entry],outfile,bundle:true,jsx:"automatic",format:"esm",loader:{".woff2":"file",".woff":"file",".ttf":"file"},nodePaths:[root+"/web/node_modules"]});'''
        subprocess.run(['node', '--input-type=module', '-', str(ROOT), str(entry), str(directory / 'workbench.js')], input=builder, text=True, check=True)
        (directory / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>OpenEconometrics · history verification</title>'
            '<link rel="stylesheet" href="/styles.css"><link rel="stylesheet" href="/workbench.css"><div id="app"></div><script type="module" src="/workbench.js"></script></html>')
        pages, opened, mutations = [], [], []
        class Handler(BaseHTTPRequestHandler):
            def respond(self, body, status=200, media='application/json'):
                self.send_response(status)
                self.send_header('Content-Type', media)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_GET(self):
                path = urlsplit(self.path).path
                if path.startswith('/api/'):
                    before = storage.reads
                    result = client.get(self.path, headers={'authorization': 'Bearer fixture-viewer'})
                    if path.endswith('/console/history') and result.status_code == 200:
                        data = result.json()
                        pages.append(dict(ids=[run['id'] for run in data['runs']], bytes=len(result.content),
                                          count=len(data['runs']), scanned=data['scanned'], result_blob_reads=storage.reads-before))
                    if path.endswith('/record') and result.status_code == 200:
                        opened.append(dict(id=result.json()['id'], response_sha256=hashlib.sha256(result.content).hexdigest()))
                    return self.respond(result.content, result.status_code)
                if path == '/receipt.json':
                    return self.respond(json_bytes(dict(synthetic_runs=137, pages=pages, opened=opened, mutations=mutations,
                        editor_source_unchanged=store.script(pid, viewer)['code'] == code,
                        cloud_access=False, user_projects_accessed=False)))
                files = {'/': (directory/'index.html', 'text/html'), '/styles.css': (ROOT/'web/src/styles.css', 'text/css'),
                         '/workbench.js': (directory/'workbench.js', 'text/javascript'), '/workbench.css': (directory/'workbench.css', 'text/css'),
                         '/assets/Barlow.woff2': (ROOT/'desktop/ui/Barlow.woff2', 'font/woff2')}
                if path not in files:
                    return self.respond(b'{}', 404)
                file, media = files[path]
                if not file.exists():
                    return self.respond(b'', 404)
                self.respond(file.read_bytes(), media=media)
            def do_POST(self):
                mutations.append(self.path)
                self.respond(b'{"detail":{"message":"Read-only verification fixture."}}', 403)
            do_PUT = do_POST
            def log_message(self, *_):
                pass
        with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            print(json.dumps({'url': f'http://127.0.0.1:{server.server_port}', 'synthetic_runs': 137}), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
