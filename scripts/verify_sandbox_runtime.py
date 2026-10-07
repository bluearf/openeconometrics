"""Disposable private Cloud Run runtime proof, excluded from deployment uploads.

The emitted private parent app exposes only fixed probes. Its child watchdog
proof accepts a strictly scoped, create-only capability for a disposable beacon;
it accepts no submitted code, arbitrary paths, general URLs or credentials.
Operator identity tokens and beacon capabilities stay in memory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

MANAGED_BOOTSTRAP = r'''import os,sys
from pathlib import Path
from openecon.sandbox_privileges import reduce_guest_privileges
reduce_guest_privileges()
native=os.getuid()
os.environ.clear()
os.environ.update({'PATH':'/opt/venv/bin:/usr/local/bin:/usr/bin:/bin','HOME':'/tmp','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','OPENECON_POC_NATIVE_UID':str(native)})
p='/tmp/openecon-fixed-probe.py'
Path(p).write_bytes(sys.stdin.buffer.read(65537))
os.execv('/opt/venv/bin/python',['/opt/venv/bin/python','-I',p])
'''

CHILD_NETWORK_DIAGNOSTIC = r'''
import ipaddress,json,socket,ssl
from pathlib import Path
from urllib.error import HTTPError,URLError
from urllib.request import Request,ProxyHandler,build_opener
from openecon.sandbox_privileges import capability_state,ALLOWED_BOUNDING
state=capability_state()
if any(state[key] for key in ('CapEff','CapPrm','CapInh','CapAmb')) or state['NoNewPrivs']!=1 or state['CapBnd']&~ALLOWED_BOUNDING:
    raise RuntimeError('Fixed managed guest privilege guard')
def error_fields(error):
    result={'exception_type':type(error).__name__}
    number=getattr(error,'errno',None)
    if type(number) is int:result['errno']=number
    verify=getattr(error,'verify_code',None)
    if type(verify) is int:result['certificate_verify_code']=verify
    return result
resolver=Path('/etc/resolv.conf')
nameservers=[]
try:
    with resolver.open() as stream:raw=stream.read(4097)
    if len(raw)<=4096:
        for line in raw.splitlines():
            fields=line.split()
            if len(fields)>=2 and fields[0]=='nameserver':
                try:nameservers.append(str(ipaddress.ip_address(fields[1])))
                except ValueError:pass
except OSError:pass
paths=ssl.get_default_verify_paths()
result={'resolver_file_present':resolver.exists(),'nameserver_count':len(nameservers),
        'nameserver_ips':nameservers[:8],'default_cafile_present':bool(paths.cafile and Path(paths.cafile).is_file()),
        'default_capath_present':bool(paths.capath and Path(paths.capath).is_dir()),
        'default_cafile_path':paths.cafile,'default_capath_path':paths.capath,'endpoints':[]}
opener=build_opener(ProxyHandler({}))
for host,url in [('storage.googleapis.com','https://storage.googleapis.com/'),
                 ('www.googleapis.com','https://www.googleapis.com/oauth2/v1/certs')]:
    item={'target':host,'dns_succeeded':False,'http_status':None}
    try:
        addresses=socket.getaddrinfo(host,443,0,socket.SOCK_STREAM)
        item['dns_succeeded']=bool(addresses)
        item['resolved_address_count']=len({entry[4][0] for entry in addresses})
    except OSError as error:item['dns_error']=error_fields(error)
    try:
        with opener.open(Request(url,headers={'Accept-Encoding':'identity'}),timeout=5) as response:
            item['http_status']=response.status
    except HTTPError as error:item['http_status']=error.code
    except URLError as error:
        reason=error.reason if isinstance(error.reason,BaseException) else error
        item['request_error']=error_fields(reason)
    except OSError as error:item['request_error']=error_fields(error)
    result['endpoints'].append(item)
print(json.dumps(result,allow_nan=False),flush=True)
'''

PARENT_NETWORK_ROUTE = r'''
@app.get('/network-diagnostic')
def network_diagnostic():
    if not LOCK.acquire(blocking=False):raise HTTPException(status_code=409,detail='Fixed proof busy')
    STATE['active']=True
    signal.setitimer(signal.ITIMER_REAL,40)
    try:
        return call(NETWORK_DIAGNOSTIC_LITERAL,timeout=30)
    except BaseException:
        raise HTTPException(status_code=500,detail='Fixed network diagnostic failed') from None
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        STATE['active']=False
        LOCK.release()
'''

DNS_CANDIDATE_PREFIX = r'''
import os
from pathlib import Path
from openecon.sandbox_privileges import capability_state,ALLOWED_BOUNDING
state=capability_state()
if any(state[key] for key in ('CapEff','CapPrm','CapInh','CapAmb')) or state['NoNewPrivs']!=1 or state['CapBnd']&~ALLOWED_BOUNDING:
    raise RuntimeError('Fixed managed guest privilege guard')
# These are guest-overlay writes. No parent or builder configuration is read.
os.chmod('/etc',0o755)
resolver=Path('/etc/resolv.conf')
resolver.write_text('nameserver 8.8.8.8\nnameserver 8.8.4.4\noptions timeout:1 attempts:2\n')
os.chmod(resolver,0o600)
hosts=Path('/etc/hosts')
os.chmod(hosts,0o600)
hosts.write_text('127.0.0.1 localhost\n::1 localhost\n169.254.169.254 metadata.google.internal\n')
'''

DNS_CANDIDATE_RESULT = r'''
result['fixed_public_resolver_candidate']=True
result['metadata_alias_resolves_to_fixed_ip']=False
try:
    alias={entry[4][0] for entry in socket.getaddrinfo('metadata.google.internal',80,0,socket.SOCK_STREAM)}
    result['metadata_alias_resolves_to_fixed_ip']=alias=={'169.254.169.254'}
except OSError as error:result['metadata_alias_error']=error_fields(error)
result['metadata_checks']=[]
for host in ('169.254.169.254','metadata.google.internal'):
    check={'target':host,'blocked':False,'http_status':None}
    try:
        request=Request('http://'+host+'/computeMetadata/v1/instance/id',headers={'Metadata-Flavor':'Google'})
        with opener.open(request,timeout=2) as response:check['http_status']=response.status
    except HTTPError as error:
        check['http_status']=error.code
        check['blocked']=error.code==403
    except URLError as error:
        reason=error.reason if isinstance(error.reason,BaseException) else error
        check['request_error']=error_fields(reason)
        check['blocked']=not isinstance(reason,socket.gaierror)
    except OSError as error:
        check['request_error']=error_fields(error)
        check['blocked']=not isinstance(error,socket.gaierror)
    result['metadata_checks'].append(check)
result['passed']=(result['nameserver_ips']==['8.8.8.8','8.8.4.4']
    and all(item['dns_succeeded'] and type(item['http_status']) is int for item in result['endpoints'])
    and result['metadata_alias_resolves_to_fixed_ip']
    and all(item['blocked'] for item in result['metadata_checks']))
print(json.dumps(result,allow_nan=False),flush=True)
'''

PARENT_DNS_CANDIDATE_ROUTE = r'''
@app.get('/dns-candidate')
def dns_candidate():
    if not LOCK.acquire(blocking=False):raise HTTPException(status_code=409,detail='Fixed proof busy')
    STATE['active']=True
    signal.setitimer(signal.ITIMER_REAL,45)
    root_etc=Path(ROOTFS)/'etc'
    mode_before=root_etc.stat().st_mode
    parent_mode_before=Path('/etc').stat().st_mode
    resolver_absent_before=not (root_etc/'resolv.conf').exists()
    try:
        result=call(DNS_CANDIDATE_LITERAL,timeout=35)
        result['parent_rootfs_resolver_absent_before']=resolver_absent_before
        result['parent_rootfs_resolver_still_absent']=not (root_etc/'resolv.conf').exists()
        result['parent_rootfs_etc_mode_unchanged']=root_etc.stat().st_mode==mode_before
        result['parent_etc_mode_unchanged']=Path('/etc').stat().st_mode==parent_mode_before
        result['passed']=bool(result.get('passed') and result.get('sandbox_deleted')
            and all(result[key] for key in ('parent_rootfs_resolver_absent_before',
                'parent_rootfs_resolver_still_absent','parent_rootfs_etc_mode_unchanged','parent_etc_mode_unchanged')))
        return result
    except BaseException:
        raise HTTPException(status_code=500,detail='Fixed DNS candidate failed') from None
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        STATE['active']=False
        LOCK.release()
'''


CHILD_PROBE = r'''
import json, os, socket, stat, time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, ProxyHandler, build_opener

def main():
    config = CONFIG_LITERAL
    root = Path('/tmp')
    checks = {
        'child_nonroot': os.getuid() == 10001,
        'parent_file_absent': not Path(config['sentinel_path']).exists(),
        'parent_proc_file_absent': not Path('/proc/1/root' + config['sentinel_path']).exists(),
        'parent_env_absent': 'OPENECON_POC_PARENT_SENTINEL' not in os.environ,
        'parent_env_proc_absent': True,
        'launcher_not_exposed': not Path('/usr/local/gcp/bin/sandbox').exists(),
    }
    # The image deliberately leaves /dev empty. The platform must supply only
    # its sandbox-local virtual devices; do not bind host devices into this tree.
    try:
        entropy_device = Path('/dev/urandom')
        with entropy_device.open('rb') as stream:
            entropy_available = (stat.S_ISCHR(os.fstat(stream.fileno()).st_mode)
                                 and len(stream.read(32)) == 32)
    except OSError:
        entropy_available = False
    try:
        null_device = Path('/dev/null')
        with null_device.open('rb') as stream:
            null_available = stat.S_ISCHR(os.fstat(stream.fileno()).st_mode) and stream.read(1) == b''
        with null_device.open('wb') as stream:
            null_available &= (stat.S_ISCHR(os.fstat(stream.fileno()).st_mode)
                               and stream.write(b'fixed-inert-probe') == 17)
    except OSError:
        null_available = False
    checks['sandbox_urandom_available'] = entropy_available
    checks['sandbox_null_available'] = null_available
    needle = config['sentinel'].encode()
    for environ in Path('/proc').glob('[0-9]*/environ'):
        try:
            with environ.open('rb') as stream:
                checks['parent_env_proc_absent'] &= needle not in stream.read(65536)
        except (OSError, PermissionError):
            pass
    for host in ['127.0.0.1', '::1']:
        try:
            with socket.create_connection((host, 8080), timeout=1):
                reachable = True
        except OSError:
            reachable = False
        checks['parent_loopback_' + ('ipv4' if host == '127.0.0.1' else 'ipv6') + '_blocked'] = not reachable
    opener = build_opener(ProxyHandler({}))
    for host, key in [('169.254.169.254', 'metadata_ip'), ('metadata.google.internal', 'metadata_dns')]:
        request = Request('http://' + host + '/computeMetadata/v1/instance/service-accounts/default/email',
                          headers={'Metadata-Flavor': 'Google'})
        try:
            with opener.open(request, timeout=2) as response:
                status = response.status
                response.read(1)
            blocked = False  # Any successful metadata response fails the gate.
        except HTTPError as error:
            status = error.code
            blocked = error.code == 403
        except OSError:
            status = None
            blocked = True
        checks[key + '_blocked'] = blocked
    try:
        with opener.open('https://storage.googleapis.com/', timeout=5) as response:
            external_status = response.status
    except HTTPError as error:
        external_status = error.code
    except OSError:
        external_status = None
    checks['public_gcs_egress_available'] = isinstance(external_status, int)
    checks['fresh_overlay'] = not (root / 'openecon-poc-overlay').exists()
    print(json.dumps({'checks': checks, 'child_uid': os.getuid(),
                      'native_child_uid': int(os.environ['OPENECON_POC_NATIVE_UID'])}, allow_nan=False), flush=True)

if __name__ == '__main__':
    main()
'''

CHILD_ANALYSIS = r'''
import json, time
from pathlib import Path

def main():
    started = time.monotonic()
    from openecon.console import ConsoleSession
    from openecon.team_job import _JobWorkspace
    imported = time.monotonic()
    folder = Path('/tmp/analysis-workspace')
    folder.mkdir()
    session = ConsoleSession(_JobWorkspace(folder))
    code = """df = oe.example()
display(df.head(3))
model = oe.ols(data=df, y='wage', x=['education', 'experience'])
display(model)
display(oe.plot.coefficients(model))
oe.plot.scatter(data=df, x='education', y='wage')
"""
    try:
        record = session.execute(code, timeout_seconds=10)
    finally:
        session.close()
    outputs = record.get('outputs', [])
    checks = {
        'analysis_ok': record.get('status') == 'ok',
        'table_model_two_plots': [x.get('type') for x in outputs] == ['table', 'model', 'plot', 'plot'],
        'observations_480': len(outputs) > 1 and outputs[1].get('data', {}).get('nobs') == 480,
    }
    print(json.dumps({'checks': checks, 'import_seconds': imported - started,
                      'analysis_seconds': time.monotonic() - imported}, allow_nan=False), flush=True)

if __name__ == '__main__':
    main()
'''

PARENT_TEMPLATE = r'''
import asyncio, hashlib, json, os, re, selectors, signal, subprocess, threading, time
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit
import uvicorn

ROOTFS = '/opt/openecon-sandbox-rootfs'
LAUNCHER = '/usr/local/gcp/bin/sandbox'
PYTHON = '/opt/venv/bin/python'
BOOT = uuid4().hex
LOCK = threading.Lock()
STATE = {'boot_nonce': BOOT, 'parent_pid': os.getpid(), 'active': False, 'disconnect_cleanup_confirmed': False,
         'stage': 'ready', 'launcher_exit': None}
SENTINEL = 'nonsecret-parent-probe-' + uuid4().hex
SENTINEL_PATH = '/tmp/openecon-parent-' + uuid4().hex
Path(SENTINEL_PATH).write_text(SENTINEL)
os.environ['OPENECON_POC_PARENT_SENTINEL'] = SENTINEL
signal.signal(signal.SIGALRM, signal.SIG_DFL)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

CHILD_PROBE = CHILD_PROBE_LITERAL
CHILD_ANALYSIS = CHILD_ANALYSIS_LITERAL
BOOTSTRAP = "import os,sys; from pathlib import Path; os.environ['OPENECON_POC_NATIVE_UID']=str(os.getuid()); os.environ.update({'PATH':'/opt/venv/bin:/usr/local/bin:/usr/bin:/bin','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'});\nif os.getuid()==0:\n os.setgroups([]); os.setgid(10001); os.setuid(10001)\nif os.getuid()!=10001: raise RuntimeError('Fixed probe identity unavailable')\np='/tmp/openecon-fixed-probe.py'; Path(p).write_bytes(sys.stdin.buffer.read(65537)); os.execv('/opt/venv/bin/python',['/opt/venv/bin/python','-I',p])"
BEACON_CHILD = "import json,time\nfrom urllib.request import Request,ProxyHandler,HTTPRedirectHandler,build_opener\nclass NoRedirect(HTTPRedirectHandler):\n def redirect_request(self,*args,**kwargs):raise ValueError('Fixed redirect refused')\nspec=CONFIG_LITERAL\npayload=json.dumps({'execution_id':spec['execution_id'],'nonce':spec['nonce'],'proof':'fixed-beacon'},sort_keys=True,separators=(',',':')).encode()\nif spec['mode']=='watchdog':\n print('READY',flush=True);time.sleep(20)\nrequest=Request(spec['upload']['url'],data=payload,method='PUT',headers=spec['upload']['headers'])\nwith build_opener(ProxyHandler({}),NoRedirect()).open(request,timeout=8) as response:\n success=response.status in (200,201,204);response.read(4096)\nprint(json.dumps({'beacon_uploaded':success}),flush=True)"

def command(name):
    return [LAUNCHER, 'run', name, '--rootfs', ROOTFS, '--write', '--allow-egress', '--workdir', '/tmp',
            '--', PYTHON, '-I', '-c', BOOTSTRAP]

def deleted(name):
    probe = subprocess.run([LAUNCHER, 'exec', name, '--', PYTHON, '-I', '-c', 'pass'],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=5)
    return probe.returncode != 0

def cleanup(name, process):
    # The fixed trusted CLI output is discarded; no sandbox output reaches logs.
    try:
        result = subprocess.run([LAUNCHER, 'delete', name, '--force'], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        return result.returncode == 0
    except BaseException:
        return False

def call(code, timeout=40):
    if not isinstance(code, str) or len(code.encode()) > 65536:
        raise ValueError('Fixed probe bound')
    name = 'oe-proof-' + uuid4().hex
    process = subprocess.Popen(command(name), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, close_fds=True, env={
                                   'PATH': '/opt/venv/bin:/usr/local/bin:/usr/bin:/bin'})
    started = time.monotonic()
    process.stdin.write(code.encode())
    process.stdin.close()
    capture = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, 'stdout')
    selector.register(process.stderr, selectors.EVENT_READ, 'stderr')
    aggregate = 0
    try:
        while selector.get_map():
            if time.monotonic() - started > timeout:
                raise TimeoutError('Fixed probe deadline')
            for key, _ in selector.select(.1):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                aggregate += len(chunk)
                if aggregate > 65536:
                    raise ValueError('Fixed probe stream bound')
                if key.data == 'stdout':
                    capture.extend(chunk)
        code_status = process.wait(timeout=2)
        STATE['launcher_exit'] = code_status
        if code_status != 0:
            raise ValueError('Fixed sandbox probe failed')
        result = json.loads(bytes(capture))
        if not isinstance(result, dict):
            raise ValueError('Fixed probe response shape')
        result['sandbox_seconds'] = time.monotonic() - started
        return result
    finally:
        selector.close()
        if not cleanup(name, process):
            os._exit(70)
        if 'result' in locals():
            result['sandbox_deleted'] = deleted(name)

@app.get('/state')
def state():
    return dict(STATE)

@app.get('/healthz')
def health():
    return {'ok': True}

@app.get('/launcher-help')
def launcher_help():
    result = subprocess.run([LAUNCHER, 'run', '--help'], stdin=subprocess.DEVNULL,
                            capture_output=True, timeout=5,
                            env={'PATH':'/usr/local/bin:/usr/bin:/bin'})
    raw = result.stdout
    # Only successful fixed help is returned; never return launcher errors.
    text = raw.decode('utf-8', errors='replace') if result.returncode == 0 and len(raw) <= 16384 else ''
    if 'Usage' not in text and 'usage' not in text:
        text = ''
    return {'launcher_exit':result.returncode,'help':text}

def fixed_error_category(raw):
    text = raw[:16384].decode('utf-8',errors='replace').lower()
    for needle,category in [('unknown flag','unknown_flag'),('permission denied','permission_denied'),
                            ('no such file','missing_file'),('socket','launcher_connection'),
                            ('environment','launcher_environment'),('config','launcher_configuration'),
                            ('failed to exec','execution_failed'),('uid','identity')]:
        if needle in text:return category
    return 'launcher_failure'

def safe_fixed_diagnostic(raw):
    # Only a fixed no-input identity command may return this provider diagnostic.
    # No user data or capabilities exist in that command. Redaction is still
    # conservative for a runtime accidentally emitting an ambient credential.
    text = raw[:4096].decode('utf-8',errors='replace')
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
    text = re.sub(r'(?i)bearer\s+\S+', 'Bearer [redacted]', text)
    text = re.sub(r'https?://\S+', '[url redacted]', text)
    text = re.sub(r'[A-Za-z0-9_+/=.-]{64,}', '[long value redacted]', text)
    return text.replace(SENTINEL,'[parent sentinel]').replace(SENTINEL_PATH,'[parent path]')

def native_identity_impl(environment_mode, rootfs=ROOTFS):
    if not LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail='Fixed probe busy')
    STATE['active'] = True
    STATE['stage'] = 'native_identity'
    signal.setitimer(signal.ITIMER_REAL, 20)
    name = 'oe-native-' + uuid4().hex
    environment = None if environment_mode == 'platform' else {'PATH':'/usr/local/bin:/usr/bin:/bin'}
    if environment_mode == 'home': environment['HOME'] = '/tmp'
    process = subprocess.Popen([LAUNCHER,'run',name,'--rootfs',rootfs,'--write','--allow-egress',
                               '--workdir','/tmp','--',PYTHON,'-I','-c',
                               "import os,json; print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),'groups':os.getgroups()}))"],
                              stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                              close_fds=True,env=environment)
    response = {'launcher_exit':None,'identity':None}
    try:
        raw,diagnostic = process.communicate(timeout=8)
        code_status = process.returncode
        STATE['launcher_exit'] = code_status
        response['launcher_exit'] = code_status
        if code_status != 0 or len(raw) > 4096:
            response['error_category'] = fixed_error_category(diagnostic)
            response['fixed_launcher_diagnostic'] = safe_fixed_diagnostic(diagnostic)
        else:
            identity = json.loads(raw)
            if (not isinstance(identity,dict) or set(identity)!={'uid','gid','groups'}
                    or type(identity['uid']) is not int or type(identity['gid']) is not int
                    or not isinstance(identity['groups'],list) or len(identity['groups'])>64
                    or any(type(value) is not int for value in identity['groups'])):
                raise ValueError('Fixed identity shape')
            response['identity'] = identity
    except subprocess.TimeoutExpired:
        response['error_category'] = 'launcher_timeout'
    finally:
        if not cleanup(name,process):
            # Only fixed safe diagnostic fields leave before destroying an
            # instance whose sandbox cleanup could not be confirmed.
            return JSONResponse(response,background=BackgroundTask(os._exit,70))
        signal.setitimer(signal.ITIMER_REAL,0)
        STATE['active'] = False
        LOCK.release()
    return response

@app.get('/native-identity')
def native_identity():
    return native_identity_impl('stripped')

@app.get('/native-identity-home')
def native_identity_home():
    return native_identity_impl('home')

@app.get('/native-identity-host-root')
def native_identity_host_root():
    # Diagnostic only: this fixed trusted identity command accesses no paths,
    # secrets, metadata or user code. Production always requires the clean root.
    return native_identity_impl('home','/')

@app.get('/native-identity-platform')
def native_identity_platform():
    # Parent environment reaches only the trusted launcher. No values or names
    # are returned; the full child isolation gate independently tests sentinel absence.
    return native_identity_impl('platform')

@app.get('/probe')
def probe():
    if not LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail='Fixed probe busy')
    STATE['active'] = True
    signal.setitimer(signal.ITIMER_REAL, 170)
    try:
        if not Path(LAUNCHER).is_file() or not Path(ROOTFS).is_dir():
            raise ValueError('Required platform runtime unavailable')
        config = {'sentinel': SENTINEL, 'sentinel_path': SENTINEL_PATH}
        STATE['stage'] = 'isolation_launch'
        isolation = call(CHILD_PROBE.replace('CONFIG_LITERAL', repr(config)))
        analyses = []
        for index in range(3):
            STATE['stage'] = 'analysis_' + str(index + 1)
            analyses.append(call(CHILD_ANALYSIS))
        # Same-instance overlay mutations cannot affect host or a fresh sandbox.
        host_library = Path(ROOTFS + '/opt/venv/pyvenv.cfg')
        host_hash = hashlib.sha256(host_library.read_bytes()).hexdigest()
        STATE['stage'] = 'overlay_write'
        first = call("from pathlib import Path\nimport json\np=Path('/opt/venv/pyvenv.cfg')\ntry:\n p.write_text('overlay-only')\n library_safe=p.read_text()=='overlay-only'\nexcept PermissionError:\n library_safe=True\nPath('/tmp/openecon-poc-overlay').write_text('overlay-only')\nprint(json.dumps({'library_write_denied_or_overlay_only':library_safe,'overlay_writable':Path('/tmp/openecon-poc-overlay').read_text()=='overlay-only'}))")
        STATE['stage'] = 'overlay_freshness'
        second = call("from pathlib import Path\nimport hashlib,json\nprint(json.dumps({'marker_absent':not Path('/tmp/openecon-poc-overlay').exists(),'library_hash':hashlib.sha256(Path('/opt/venv/pyvenv.cfg').read_bytes()).hexdigest()}))")
        STATE['stage'] = 'orphan_delete'
        orphan = call("import subprocess,sys,json\nsubprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(300)'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)\nprint(json.dumps({'spawned':True}))")
        checks = dict(isolation['checks'])
        checks.update(parent_nonroot=os.getuid() != 0, launcher_available=True,
                      overlay_mutation_worked=first['overlay_writable'],
                      library_write_denied_or_overlay_only=first['library_write_denied_or_overlay_only'],
                      cross_sandbox_marker_absent=second['marker_absent'],
                      cross_sandbox_library_unchanged=second['library_hash'] == host_hash,
                      host_library_unchanged=hashlib.sha256(host_library.read_bytes()).hexdigest() == host_hash,
                      orphan_sandbox_deleted=orphan['sandbox_deleted'],
                      all_analysis_checks=all(all(item['checks'].values()) and item['sandbox_deleted'] for item in analyses),
                      initial_sandbox_deleted=isolation['sandbox_deleted'])
        return {'schema_version': 1, 'passed': all(checks.values()), 'checks': checks,
                'parent_uid': os.getuid(), 'parent_pid': os.getpid(), 'child_uid': isolation['child_uid'],
                'native_child_uid': isolation['native_child_uid'],
                'boot_nonce': BOOT, 'analysis_timings': analyses,
                'isolation_sandbox_seconds': isolation['sandbox_seconds'],
                'signed_gcs_delivery_verified': False}
    except BaseException:
        # No tracebacks, CLI diagnostics, environment, or provider payloads.
        raise HTTPException(status_code=500, detail={'error':'Fixed sandbox proof failed',
                            'stage':STATE['stage'],'launcher_exit':STATE['launcher_exit']}) from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        STATE['active'] = False
        LOCK.release()

@app.get('/disconnect')
async def disconnect():
    if not LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail='Fixed probe busy')
    STATE['active'] = True
    STATE['disconnect_cleanup_confirmed'] = False
    # Kernel default-fatal alarm tests the backstop during request CPU throttling.
    signal.setitimer(signal.ITIMER_REAL, 12)
    name = 'oe-disconnect-' + uuid4().hex
    process = subprocess.Popen(command(name), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, close_fds=True,
                               env={'PATH': '/opt/venv/bin:/usr/local/bin:/usr/bin:/bin'})
    process.stdin.write(b"import time\nprint('READY',flush=True)\ntime.sleep(300)\n")
    process.stdin.close()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    events = selector.select(8)
    ready = bool(events) and os.read(process.stdout.fileno(), 64).startswith(b'READY\n')
    selector.close()
    if not ready:
        if not cleanup(name, process):
            os._exit(70)
        STATE['active'] = False
        signal.setitimer(signal.ITIMER_REAL, 0)
        LOCK.release()
        raise HTTPException(status_code=500, detail='Fixed disconnect probe failed')
    async def stream():
        try:
            yield json.dumps({'started': True, 'boot_nonce': BOOT}).encode() + b'\n'
            while True:
                await asyncio.sleep(1)
                yield b'{"running":true}\n'
        finally:
            # Synchronous forced cleanup runs before yielding from cancellation.
            if not cleanup(name, process):
                os._exit(70)
            STATE['disconnect_cleanup_confirmed'] = True
            STATE['active'] = False
            signal.setitimer(signal.ITIMER_REAL, 0)
            LOCK.release()
    return StreamingResponse(stream(), media_type='application/x-ndjson')

@app.get('/alarm')
async def alarm():
    if not LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail='Fixed probe busy')
    STATE['active'] = True
    signal.setitimer(signal.ITIMER_REAL, 4)
    async def stream():
        try:
            yield json.dumps({'started': True, 'boot_nonce': BOOT, 'parent_pid': os.getpid()}).encode() + b'\n'
            await asyncio.sleep(300)
        finally:
            # This fixed private proof intentionally leaves the kernel timer
            # armed after disconnect. It launches no code or child process.
            STATE['active'] = False
            LOCK.release()
    return StreamingResponse(stream(), media_type='application/x-ndjson')

def fixed_beacon_body(raw):
    def pairs(items):
        result = {}
        for key,value in items:
            if key in result: raise ValueError('Fixed duplicate field')
            result[key] = value
        return result
    spec = json.loads(raw,object_pairs_hook=pairs)
    if (not isinstance(spec,dict) or set(spec)!={'project_id','execution_id','nonce','mode','upload'}
            or spec['mode'] not in {'positive','watchdog'}
            or any(not isinstance(spec[key],str) or not re.fullmatch(r'[0-9a-f]{32}',spec[key])
                   for key in ('project_id','execution_id','nonce'))):
        raise ValueError('Fixed beacon shape')
    upload = spec['upload']
    if not isinstance(upload,dict) or set(upload)!={'url','headers'} or not isinstance(upload['url'],str) or len(upload['url'])>16384:
        raise ValueError('Fixed beacon upload shape')
    parsed = urlsplit(upload['url'])
    query = parse_qs(parsed.query,keep_blank_values=True)
    expected = {'X-Goog-Algorithm','X-Goog-Credential','X-Goog-Date','X-Goog-Expires','X-Goog-SignedHeaders','X-Goog-Signature'}
    payload = json.dumps({'execution_id':spec['execution_id'],'nonce':spec['nonce'],'proof':'fixed-beacon'},sort_keys=True,separators=(',',':')).encode()
    if (parsed.scheme!='https' or parsed.netloc!='storage.googleapis.com' or parsed.fragment
            or parsed.path!='/openecon-workbench-projects/staging/'+spec['project_id']+'/'+spec['execution_id']+'/beacon.json'
            or set(query)!=expected or any(len(values)!=1 for values in query.values())
            or query['X-Goog-Algorithm']!=['GOOG4-RSA-SHA256']
            or query['X-Goog-SignedHeaders']!=['content-length;content-type;host;x-goog-if-generation-match']
            or not re.fullmatch(r'openecon-transfer@openecon-workbench\.iam\.gserviceaccount\.com/[0-9]{8}/auto/storage/goog4_request',query['X-Goog-Credential'][0])
            or not re.fullmatch(r'[0-9a-f]{512}',query['X-Goog-Signature'][0])
            or upload['headers']!={'Content-Type':'application/json','Content-Length':str(len(payload)),'x-goog-if-generation-match':'0'}):
        raise ValueError('Fixed beacon scope')
    expires = int(query['X-Goog-Expires'][0])
    dated = datetime.strptime(query['X-Goog-Date'][0],'%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc)-dated).total_seconds()
    if not 30<=expires<=300 or not -60<=age<expires:
        raise ValueError('Fixed beacon expiry')
    return spec

@app.post('/alarm-child')
async def alarm_child(request: Request):
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw)>32768: raise HTTPException(status_code=413,detail='Fixed beacon bound')
    try:
        spec = fixed_beacon_body(bytes(raw))
    except (ValueError,TypeError,KeyError):
        raise HTTPException(status_code=400,detail='Fixed beacon rejected') from None
    if not LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409,detail='Fixed probe busy')
    STATE['active'] = True
    STATE['stage'] = 'fixed_beacon_'+spec['mode']
    signal.setitimer(signal.ITIMER_REAL,30)
    code = BEACON_CHILD.replace('CONFIG_LITERAL',repr(spec))
    if spec['mode']=='positive':
        try:
            result = call(code,timeout=20)
            return {'beacon_uploaded':result.get('beacon_uploaded') is True,
                    'sandbox_deleted':result.get('sandbox_deleted') is True,'boot_nonce':BOOT}
        except BaseException:
            raise HTTPException(status_code=500,detail='Fixed beacon positive proof failed') from None
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            STATE['active'] = False
            LOCK.release()
    name = 'oe-alarm-child-'+uuid4().hex
    process = subprocess.Popen(command(name),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL,close_fds=True,
                               env={'PATH':'/opt/venv/bin:/usr/local/bin:/usr/bin:/bin','HOME':'/tmp'})
    process.stdin.write(code.encode())
    process.stdin.close()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout,selectors.EVENT_READ)
    events = selector.select(8)
    ready = bool(events) and os.read(process.stdout.fileno(),64).startswith(b'READY\n')
    selector.close()
    if not ready:
        if not cleanup(name,process): os._exit(70)
        signal.setitimer(signal.ITIMER_REAL,0)
        STATE['active'] = False
        LOCK.release()
        raise HTTPException(status_code=500,detail='Fixed beacon child launch failed')
    signal.setitimer(signal.ITIMER_REAL,4)
    async def stream():
        yield json.dumps({'started':True,'boot_nonce':BOOT,'parent_pid':os.getpid()}).encode()+b'\n'
        # Deliberately no disconnect cleanup in this bounded private proof:
        # only the kernel watchdog and platform teardown can kill this child.
        await asyncio.sleep(300)
    return StreamingResponse(stream(),media_type='application/x-ndjson')

@app.get('/hold')
async def hold():
    async def stream():
        for _ in range(25):
            yield b'{"holding":true}\n'
            await asyncio.sleep(1)
    return StreamingResponse(stream(),media_type='application/x-ndjson')

uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', '8080')), access_log=False, log_level='warning')
'''


def parent_source(*, managed_guest: bool = False, dns_candidate: bool = False) -> str:
    if dns_candidate and not managed_guest:
        raise ValueError('The fixed DNS candidate requires the managed guest proof.')
    template, child = PARENT_TEMPLATE, CHILD_PROBE
    if managed_guest:
        start = template.index("@app.get('/launcher-help')")
        end = template.index("@app.get('/probe')", start)
        template = template[:start] + template[end:]
        start = template.index('BOOTSTRAP = ')
        end = template.index('\nBEACON_CHILD = ', start)
        template = template[:start] + 'BOOTSTRAP = ' + repr(MANAGED_BOOTSTRAP) + template[end:]
        template = template.replace("BOOT = uuid4().hex", "BOOT = uuid4().hex\nif os.getuid()!=0 or os.getpid()==1: raise RuntimeError('Trusted root Tini parent required')\nprint(json.dumps({'proof':'openecon-fixed-parent-identity','boot_nonce':BOOT}),flush=True)")
        template = template.replace('parent_nonroot=os.getuid() != 0',
                                    'trusted_root_parent=os.getuid() == 0 and os.getpid() != 1')
        template = template.replace("'signed_gcs_delivery_verified': False}",
                                    "'signed_gcs_delivery_verified': False,'managed_guest':True,'network_isolation_verified':False,'application_auth_boundary_verified':False}")
        child = child.replace('def main():\n',
                              "def main():\n    from openecon.sandbox_privileges import capability_state, ALLOWED_BOUNDING\n    privilege=capability_state()\n")
        child = child.replace("'child_nonroot': os.getuid() == 10001,",
                              "'managed_guest_privileges_reduced': (os.getuid()==0 and os.getgid()==0 and os.getgroups()==[] and all(privilege[key]==0 for key in ('CapEff','CapPrm','CapInh','CapAmb')) and privilege['NoNewPrivs']==1 and not privilege['CapBnd']&~ALLOWED_BOUNDING),")
        if dns_candidate:
            child_dns=DNS_CANDIDATE_PREFIX+CHILD_NETWORK_DIAGNOSTIC.replace(
                'from urllib.request import Request,ProxyHandler,build_opener',
                'from urllib.request import Request,ProxyHandler,HTTPRedirectHandler,build_opener')
            child_dns=child_dns.replace('opener=build_opener(ProxyHandler({}))',
                "class NoRedirect(HTTPRedirectHandler):\n    def redirect_request(self,*args,**kwargs):return None\nopener=build_opener(ProxyHandler({}),NoRedirect())")
            child_dns=child_dns.replace('print(json.dumps(result,allow_nan=False),flush=True)',DNS_CANDIDATE_RESULT)
            route=PARENT_DNS_CANDIDATE_ROUTE.replace('DNS_CANDIDATE_LITERAL',repr(child_dns))
        else:
            route=PARENT_NETWORK_ROUTE.replace('NETWORK_DIAGNOSTIC_LITERAL',repr(CHILD_NETWORK_DIAGNOSTIC))
        template=template.replace('uvicorn.run(app,',route+'\nuvicorn.run(app,')
    source = (template.replace('CHILD_PROBE_LITERAL', repr(child))
            .replace('CHILD_ANALYSIS_LITERAL', repr(CHILD_ANALYSIS)))
    if len(source.encode()) > 32768:
        raise ValueError('Fixed runtime source exceeds the Cloud Run proof budget.')
    return source


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Probe redirects are refused.')


class _InstanceLoggingGate:
    """Read only fixed private-probe startup and managed termination records."""
    project = 'openecon-workbench'
    service = 'openecon-sandbox-check'

    def __init__(self):
        import google.auth
        from google.auth.transport.requests import AuthorizedSession
        credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        self.session = AuthorizedSession(credentials)
        url = ('https://run.googleapis.com/v2/projects/' + self.project
               + '/locations/us-central1/services/' + self.service)
        body = self._json(self.session.get(url, timeout=15, allow_redirects=False, stream=True))
        ready = body.get('latestReadyRevision', '')
        prefix = f'projects/{self.project}/locations/us-central1/services/{self.service}/revisions/'
        if (not isinstance(ready, str) or not ready.startswith(prefix)
                or not re.fullmatch(re.escape(self.service) + r'-[a-z0-9-]{1,64}', ready[len(prefix):])):
            raise ValueError('Fixed probe revision unavailable.')
        self.revision = ready[len(prefix):]

    @staticmethod
    def _json(response):
        with response:
            if response.status_code != 200:
                raise ValueError('Fixed logging gate unavailable.')
            raw = bytearray()
            for chunk in response.iter_content(4096):
                raw.extend(chunk)
                if len(raw) > 262144:
                    raise ValueError('Fixed logging response exceeds its bound.')
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError('Fixed logging response shape.')
        return body

    def entries(self, expression):
        base = (f'resource.type="cloud_run_revision" AND resource.labels.project_id="{self.project}" '
                f'AND resource.labels.service_name="{self.service}" '
                f'AND resource.labels.revision_name="{self.revision}"')
        response = self.session.post('https://logging.googleapis.com/v2/entries:list',
                                     json={'resourceNames': ['projects/' + self.project],
                                           'filter': base + ' AND ' + expression,
                                           'orderBy': 'timestamp desc', 'pageSize': 20},
                                     timeout=15, allow_redirects=False, stream=True)
        entries = self._json(response).get('entries', [])
        if not isinstance(entries, list) or len(entries) > 20:
            raise ValueError('Fixed logging entry bound.')
        return entries

    def instance(self, boot):
        if not isinstance(boot, str) or not re.fullmatch(r'[0-9a-f]{32}', boot):
            raise ValueError('Fixed boot identity unavailable.')
        expression = ('jsonPayload.proof="openecon-fixed-parent-identity" '
                      f'AND jsonPayload.boot_nonce="{boot}"')
        for attempt in range(8):
            ids = set()
            for entry in self.entries(expression):
                value = entry.get('labels', {}).get('instanceId')
                if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{10,256}', value):
                    ids.add(value)
            if len(ids) == 1:
                return ids.pop()
            if len(ids) > 1:
                raise ValueError('Fixed boot identity is ambiguous.')
            if attempt < 7:
                time.sleep(2)
        raise ValueError('Fixed boot startup record unavailable.')

    def watchdog_termination(self, instance, since):
        expression = (f'labels.instanceId="{instance}" AND timestamp>="{since}" '
                      f'AND logName="projects/{self.project}/logs/run.googleapis.com%2Fvarlog%2Fsystem"')
        for attempt in range(10):
            for entry in self.entries(expression):
                text = entry.get('textPayload', '')
                if (isinstance(text, str) and len(text) <= 8192
                        and (re.search(r'\bcontainer\b.*\bexit\s*\(?142\)?', text, re.IGNORECASE)
                             or re.search(r'\bcontainer\b.*(?:SIGALRM|signal\s+14)\b', text, re.IGNORECASE))):
                    return True
            if attempt < 9:
                time.sleep(2)
        return False

    def death_proof(self, old_boot, new_boot, since):
        old_instance, new_instance = self.instance(old_boot), self.instance(new_boot)
        return {'old_instance_id': old_instance, 'new_instance_id': new_instance,
                'replacement_instance_confirmed': old_instance != new_instance,
                'managed_watchdog_termination_confirmed': self.watchdog_termination(old_instance, since),
                'revision': self.revision}


def verify(origin: str, report: Path, *, managed_guest: bool = False) -> dict:
    if not re.fullmatch(r'https://[a-z][a-z0-9-]*(?:\.[a-z0-9-]+)?\.run\.app', origin):
        raise ValueError('A canonical Cloud Run HTTPS origin is required.')
    # No shell, credential files, environment copy, or token-bearing diagnostics.
    token = subprocess.run(['gcloud', 'auth', 'print-identity-token'], check=True,
                           capture_output=True, text=True).stdout.strip()
    if not token or len(token) > 16384:
        raise ValueError('Operator identity unavailable.')
    opener = build_opener(ProxyHandler({}), _NoRedirect())

    def get(path, timeout=180):
        request = Request(origin + path, headers={'Authorization': 'Bearer ' + token})
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError('Probe response exceeds its bound.')
        return json.loads(raw)

    started = time.monotonic()
    proof = get('/probe')
    if managed_guest and proof.get('managed_guest') is not True:
        raise ValueError('The fixed managed-guest runtime is required.')
    logging = _InstanceLoggingGate() if proof.get('managed_guest') is True else None
    proof['host_request_seconds'] = time.monotonic() - started
    request = Request(origin + '/disconnect', headers={'Authorization': 'Bearer ' + token})
    with opener.open(request, timeout=30) as response:
        first = response.readline(4097)
        if len(first) > 4096:
            raise ValueError('Disconnect response exceeds its bound.')
        disconnect = json.loads(first)
    # The operator deliberately closes the active request. The default-fatal
    # kernel alarm or confirmed forced cleanup must prevent live orphan work.
    time.sleep(14)
    state = get('/state', timeout=60)
    cleanup = (state.get('boot_nonce') != disconnect.get('boot_nonce')
               or (state.get('active') is False and state.get('disconnect_cleanup_confirmed') is True))
    alarm_started = datetime.now(timezone.utc).isoformat()
    request = Request(origin + '/alarm', headers={'Authorization': 'Bearer ' + token})
    with opener.open(request, timeout=30) as response:
        first = response.readline(4097)
        if len(first) > 4096:
            raise ValueError('Alarm response exceeds its bound.')
        alarm = json.loads(first)
    time.sleep(6)
    alarm_state = None
    for attempt in range(3):
        try:
            alarm_state = get('/state', timeout=60)
            break
        except (HTTPError, OSError):
            if attempt == 2:
                raise
            time.sleep(1)
    alarm_killed_parent = alarm_state.get('boot_nonce') != alarm.get('boot_nonce')
    alarm_log = None
    if logging is not None:
        alarm_log = logging.death_proof(alarm.get('boot_nonce'), alarm_state.get('boot_nonce'), alarm_started)
        alarm_killed_parent &= (alarm_log['replacement_instance_confirmed']
                                and alarm_log['managed_watchdog_termination_confirmed'])
    # The harness is fixed trusted code; nevertheless whitelist every retained
    # output so provider diagnostics, arbitrary content and headers cannot leak.
    evidence = {
        'schema_version': 1, 'verified_at': datetime.now(timezone.utc).isoformat(),
        'origin': origin, 'checks': proof.get('checks', {}),
        'parent_uid': proof.get('parent_uid'), 'parent_pid': proof.get('parent_pid'),
        'child_uid': proof.get('child_uid'),
        'native_child_uid': proof.get('native_child_uid'),
        'analysis_timings': proof.get('analysis_timings', []),
        'host_request_seconds': proof['host_request_seconds'],
        'isolation_sandbox_seconds': proof.get('isolation_sandbox_seconds'),
        'disconnect_started': disconnect.get('started') is True,
        'disconnect_cleanup_or_instance_death_confirmed': cleanup,
        'kernel_alarm_after_disconnect_killed_parent': alarm_killed_parent,
        'signed_gcs_delivery_verified': False,
        'managed_guest': proof.get('managed_guest') is True,
        'network_isolation_verified': False if logging is not None else None,
        'application_auth_boundary_verified': False,
        'managed_parent_watchdog_instance_evidence': alarm_log,
    }
    evidence['passed'] = (proof.get('passed') is True and cleanup and evidence['disconnect_started']
                          and alarm.get('started') is True and alarm_killed_parent)
    report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        report.write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    save()
    if not evidence['passed']:
        return evidence
    # The previous parent-only alarm cannot establish child termination. This
    # exact-capability positive/negative control observes a namespace-local
    # child's external side effect after the parent watchdog kills its broker.
    from openecon.team_storage import TeamStorage
    from verify_sandbox_broker_live import (BUCKET, PROJECT, SIGNER, beacon_spec, verify_beacon)
    storage = TeamStorage(PROJECT, BUCKET, SIGNER)
    scope = {'project_id': __import__('uuid').uuid4().hex,
             'run_id': __import__('uuid').uuid4().hex}
    evidence['pending_beacon_scopes'] = [scope]
    evidence['kernel_watchdog_child_death_verified'] = False
    evidence['passed'] = False
    save()  # Only random IDs, before the first object or execution mutation.
    safe_cleanup = False
    positive = False
    try:
        spec, payload = beacon_spec(storage, scope['project_id'], scope['run_id'])
        spec.update(project_id=scope['project_id'], mode='positive')
        def beacon_request():
            return Request(origin + '/alarm-child',
                           data=json.dumps(spec, separators=(',', ':')).encode(), method='POST',
                           headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        with opener.open(beacon_request(), timeout=40) as response:
            raw = response.read(4097)
        if len(raw)>4096:
            raise ValueError('Fixed beacon positive response bound.')
        result = json.loads(raw)
        positive = (result.get('beacon_uploaded') is True and result.get('sandbox_deleted') is True
                    and verify_beacon(storage, scope['project_id'], scope['run_id'], payload, True))
        evidence['child_beacon_positive_control'] = positive
        if not positive:
            raise ValueError('Fixed beacon positive control failed.')
        blob = storage.bucket.blob(f"staging/{scope['project_id']}/{scope['run_id']}/beacon.json")
        blob.reload(timeout=15)
        blob.delete(if_generation_match=int(blob.generation), timeout=15, retry=None)
        if not verify_beacon(storage, scope['project_id'], scope['run_id'], payload, False):
            raise ValueError('Fixed beacon reset failed.')
        spec['mode'] = 'watchdog'
        child_alarm_started = datetime.now(timezone.utc).isoformat()
        with opener.open(beacon_request(), timeout=30) as response:
            raw = response.readline(4097)
            if len(raw)>4096:
                raise ValueError('Fixed beacon watchdog response bound.')
            child_alarm = json.loads(raw)
        if child_alarm.get('started') is not True:
            raise ValueError('Fixed beacon child unavailable.')
        time.sleep(6)
        child_alarm_state = None
        for attempt in range(3):
            try:
                child_alarm_state = get('/state', timeout=60)
                break
            except (HTTPError, OSError):
                if attempt == 2:
                    raise
                time.sleep(1)
        parent_died = child_alarm_state.get('boot_nonce') != child_alarm.get('boot_nonce')
        child_alarm_log = None
        if logging is not None:
            child_alarm_log = logging.death_proof(child_alarm.get('boot_nonce'),
                                                 child_alarm_state.get('boot_nonce'), child_alarm_started)
            parent_died &= (child_alarm_log['replacement_instance_confirmed']
                            and child_alarm_log['managed_watchdog_termination_confirmed'])
        # Hold an active request for 25s after restart. This allocates CPU to
        # the service during the delayed beacon window, avoiding an absence
        # observation made solely while an idle container was throttled.
        held = 0
        request = Request(origin+'/hold',headers={'Authorization':'Bearer '+token})
        with opener.open(request,timeout=35) as response:
            for _ in range(26):
                raw = response.readline(4097)
                if not raw:
                    break
                if raw != b'{"holding":true}\n':
                    raise ValueError('Fixed hold response shape.')
                held += 1
        absent = verify_beacon(storage, scope['project_id'], scope['run_id'], payload, False)
        evidence.update(child_watchdog_parent_death_confirmed=parent_died,
                        child_watchdog_observation_hold_completed=held == 25,
                        child_watchdog_beacon_absent=absent,
                        signed_gcs_delivery_verified=positive,
                        managed_child_watchdog_instance_evidence=child_alarm_log)
        safe_cleanup = parent_died and held == 25 and absent
        evidence['kernel_watchdog_child_death_verified'] = positive and safe_cleanup
        evidence['passed'] = evidence['kernel_watchdog_child_death_verified']
    except Exception:
        evidence['error'] = 'Fixed external child watchdog proof failed.'
    finally:
        if safe_cleanup:
            evidence['pending_beacon_scopes'] = []
        save()
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--emit-source', type=Path)
    mode.add_argument('--verify', metavar='PRIVATE_CLOUD_RUN_ORIGIN')
    parser.add_argument('--managed-guest', action='store_true',
                        help='Emit or require the fixed root-parent/helper-reduced guest proof.')
    parser.add_argument('--dns-candidate', action='store_true',
                        help='Emit the fixed guest-overlay public DNS and metadata-alias diagnostic.')
    parser.add_argument('--report', type=Path,
                        default=Path('artifacts/verification/sandbox-runtime-live.json'))
    args = parser.parse_args()
    if args.emit_source:
        source = parent_source(managed_guest=args.managed_guest,dns_candidate=args.dns_candidate)
        compile(source, '<fixed-sandbox-proof>', 'exec')
        args.emit_source.parent.mkdir(parents=True, exist_ok=True)
        args.emit_source.write_text(source)
        print(json.dumps({'emitted_source': str(args.emit_source), 'source_bytes': len(source.encode())}))
        return 0
    try:
        evidence = verify(args.verify, args.report, managed_guest=args.managed_guest)
    except (OSError, ValueError, HTTPError, subprocess.CalledProcessError):
        print(json.dumps({'passed': False, 'error': 'Private runtime verification unavailable.'}))
        return 1
    print(json.dumps({'passed': evidence['passed'], 'report': str(args.report)}))
    return 0 if evidence['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
