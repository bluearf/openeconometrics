"""Preview the real project form against an owned, synthetic HTTP API fixture.

No Firebase account, cloud data or user project is accessed. The first valid
POST deliberately fails; retry reaches the real control API and memory store.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

from fastapi.testclient import TestClient
from openecon.team_auth import TeamAuth, TeamIdentity
from openecon.team_server import create_team_app
from openecon.team_storage import MemoryStorage
from openecon.team_store import MemoryDocuments, TeamStore

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = 'https://form-preview.example.com'
PROJECT = 'openecon-form-preview'


def main():
    documents = MemoryDocuments()
    store = TeamStore(documents, owner_email='owner@example.com', public_origin=ORIGIN)
    store.me(TeamIdentity('owner', 'owner@example.com', 'Owner', True))
    def verify(token):
        if token != 'fixture-owner':
            raise ValueError('Invalid fixture token')
        return dict(uid='owner', sub='owner', email='owner@example.com', name='Owner',
            aud=PROJECT, iss=f'https://securetoken.google.com/{PROJECT}',
            firebase={'sign_in_provider': 'password'}, email_verified=True)
    app = create_team_app(store=store, storage=MemoryStorage(), auth=TeamAuth(PROJECT, verifier=verify),
        runner=SimpleNamespace(), public_origin=ORIGIN,
        firebase_config={'projectId': PROJECT, 'authDomain': f'{PROJECT}.firebaseapp.com'})
    with tempfile.TemporaryDirectory(prefix='openecon-project-form-') as temporary, TestClient(app, base_url=ORIGIN) as client:
        directory = Path(temporary)
        entry = directory / 'entry.tsx'
        entry.write_text('''import {useState} from "react";
import {createRoot} from "react-dom/client";
import ProjectCreateForm from ''' + json.dumps(str(ROOT / 'web/src/ProjectCreateForm.tsx')) + ''';
function Preview(){
 const [name,onName]=useState(""),[description,onDescription]=useState(""),
 [busy,setBusy]=useState(false),[error,setError]=useState(""),[notice,setNotice]=useState("");
 async function create(input){setBusy(true);setError("");setNotice("");try{
  const response=await fetch("/api/projects",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(input)});
  const saved=await response.json();if(!response.ok)throw Error(saved.detail.message);
  const listed=await(await fetch("/api/projects")).json();
  const found=listed.projects.find(project=>project.id===saved.id);
  if(!found||found.name!==input.name||found.description!==input.description)throw Error("Readback did not match");
  setNotice("Saved and read back: "+Array.from(found.name).length+" name characters, "+Array.from(found.description).length+" description characters.");
 }catch(reason){setError(reason.message);}finally{setBusy(false);}}
 return <dialog open className="team-dialog"><header><h2>New project</h2></header>
 <ProjectCreateForm {...{name,description,busy,error,onName,onDescription}} onCreate={create}/>
 {notice&&<p role="status" className="team-notice">{notice}</p>}</dialog>;
}createRoot(document.getElementById("app")).render(<Preview/>);''')
        builder = '''import {pathToFileURL} from "node:url";
const [root,entry,outfile]=process.argv.slice(2);
const {build}=await import(pathToFileURL(root+"/web/node_modules/esbuild/lib/main.js").href);
await build({entryPoints:[entry],outfile,bundle:true,jsx:"automatic",format:"esm",nodePaths:[root+"/web/node_modules"]});'''
        subprocess.run(['node', '--input-type=module', '-', str(ROOT), str(entry), str(directory / 'form.js')],
                       input=builder, text=True, check=True)
        (directory / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>OpenEconometrics · project form verification</title><link rel="stylesheet" href="/team.css">
<style>body{margin:0;background:#f6f7f9;font-family:Arial,sans-serif}*{box-sizing:border-box}
.team-dialog{top:70px;width:480px;max-width:calc(100% - 32px)}
.team-dialog>.team-notice{margin:0 22px 22px}</style>
<div id="app"></div><script type="module" src="/form.js"></script></html>''')
        statuses = []
        class Handler(BaseHTTPRequestHandler):
            def respond(self, body, status=200, media='application/json'):
                self.send_response(status)
                self.send_header('Content-Type', media)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == '/api/projects':
                    result = client.get(self.path, headers={'authorization': 'Bearer fixture-owner'})
                    return self.respond(result.content, result.status_code)
                if self.path == '/receipt.json':
                    projects = store.projects(TeamIdentity('owner', 'owner@example.com', 'Owner', True))
                    return self.respond(json.dumps(dict(post_statuses=statuses, stored_projects=len(projects),
                        stored_text_lengths=[dict(name=len(p['name']), description=len(p['description'])) for p in projects],
                        cloud_access=False, user_projects_accessed=False)).encode())
                files = {'/': (directory / 'index.html', 'text/html'),
                         '/form.js': (directory / 'form.js', 'text/javascript'),
                         '/team.css': (ROOT / 'web/src/team-styles.css', 'text/css')}
                if self.path not in files:
                    return self.respond(b'{}', 404)
                path, media = files[self.path]
                self.respond(path.read_bytes(), media=media)

            def do_POST(self):
                if self.path != '/api/projects':
                    return self.respond(b'{}', 404)
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 512 * 1024:
                    return self.respond(b'{}', 413)
                body = self.rfile.read(size)
                if not statuses:
                    statuses.append(503)
                    return self.respond(b'{"detail":{"message":"Could not create the project right now. Your draft is preserved."}}', 503)
                result = client.post(self.path, content=body, headers={'authorization': 'Bearer fixture-owner',
                    'origin': ORIGIN, 'content-type': 'application/json'})
                statuses.append(result.status_code)
                self.respond(result.content, result.status_code)

            def log_message(self, *_):
                pass
        with ThreadingHTTPServer(('127.0.0.1', 0), Handler) as server:
            print(json.dumps({'url': f'http://127.0.0.1:{server.server_port}', 'synthetic_memory_store': True}), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()
