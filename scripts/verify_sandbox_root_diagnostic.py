"""Emit an ignored, private root-parent diagnostic; never production code.

There are no code, file, URL, token, capability or body inputs. Every sandbox
command and CLI variant is fixed. The trusted parent may run as UID 0 only in
this disposable diagnostic deployment. Production identity guards stay intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CHILD = r'''
import errno,json,os,stat
from pathlib import Path

def caps():
    result={}
    try:
        with open('/proc/self/status') as stream:raw=stream.read(8192)
        for line in raw.splitlines():
            key,sep,value=line.partition(':')
            if key in ('CapEff','CapPrm','CapBnd','CapInh','CapAmb') and sep:
                value=value.strip()
                if len(value)<=16 and all(char in '0123456789abcdefABCDEF' for char in value):
                    result[key]=value.lower()
    except OSError:pass
    return result

before={'uid':os.getuid(),'gid':os.getgid(),'groups':os.getgroups(),'capabilities':caps()}
drop={}
if DROP_LITERAL:
    for name,action in [('setgroups',lambda:os.setgroups([])),('setgid',lambda:os.setgid(10001)),
                        ('setuid',lambda:os.setuid(10001))]:
        try:action();drop[name]=0
        except OSError as error:drop[name]=error.errno or -1
devices={}
for name in ('urandom','null'):
    try:
        with open('/dev/'+name,'rb') as stream:
            kind=stat.S_ISCHR(os.fstat(stream.fileno()).st_mode)
            value=stream.read(32 if name=='urandom' else 1)
        devices[name]=kind and (len(value)==32 if name=='urandom' else value==b'')
    except OSError:devices[name]=False
marker=Path('/tmp')/MARKER_LITERAL
try:
    marker.write_text('fixed-inert-root-diagnostic')
    writable=marker.read_text()=='fixed-inert-root-diagnostic'
    write_errno=0
except OSError as error:writable=False;write_errno=error.errno or -1
print(json.dumps({'before':before,'after':{'uid':os.getuid(),'gid':os.getgid(),'groups':os.getgroups(),
      'capabilities':caps()},'drop_errno':drop,'devices':devices,'tmp_writable':writable,
      'tmp_write_errno':write_errno,'pid':os.getpid()},allow_nan=False),flush=True)
'''

HARDENED_CHILD = r'''
import ctypes,hashlib,ipaddress,json,os,socket,stat,struct,subprocess,sys,time
from pathlib import Path
config=CONFIG_LITERAL
def status():
    result={}
    with open('/proc/self/status') as stream:raw=stream.read(8192)
    for line in raw.splitlines():
        key,sep,value=line.partition(':')
        if key in ('CapEff','CapPrm','CapBnd','CapInh','CapAmb','NoNewPrivs') and sep:
            result[key]=value.strip()
    return result
before=status()
bootstrap=(sys.platform=='linux' and os.getuid()==0 and os.geteuid()==0 and os.getgid()==0
           and os.getegid()==0 and os.getgroups()==[] and os.getpid()==1
           and len(list(Path('/proc/self/task').iterdir()))==1
           and int(before.get('CapBnd','ffffffffffffffff'),16)&~0x20000420==0)
if not bootstrap:
    print(json.dumps({'checks':{'fixed_bootstrap_identity':False},'before':before}),flush=True)
    raise SystemExit(0)
libc=ctypes.CDLL(None,use_errno=True)
class Header(ctypes.Structure):_fields_=[('version',ctypes.c_uint32),('pid',ctypes.c_int)]
class Data(ctypes.Structure):_fields_=[('effective',ctypes.c_uint32),('permitted',ctypes.c_uint32),('inheritable',ctypes.c_uint32)]
header=Header(0x20080522,0)
nnpret=libc.prctl(38,1,0,0,0)
ambientret=libc.prctl(47,4,0,0,0)
data=(Data*2)()
capret=libc.capset(ctypes.byref(header),ctypes.byref(data))
after=status()
guard=lambda value: all(int(value.get(key,'1'),16)==0 for key in ('CapEff','CapPrm','CapInh','CapAmb')) and value.get('NoNewPrivs')=='1'
checks={'capset_succeeded':capret==0,'no_new_privs_set':nnpret==0,'ambient_clear_succeeded':ambientret==0,
        'guest_capabilities_disabled':guard(after),'supplementary_groups_empty':os.getgroups()==[],
        'fixed_bootstrap_identity':bootstrap,'known_bounding_only':int(after.get('CapBnd','ffffffffffffffff'),16)&~0x20000420==0}
if not all(checks.values()):
    print(json.dumps({'checks':checks,'before':before,'after':after}),flush=True)
    raise SystemExit(0)
attempt=(Data*2)();attempt[0].effective=32;attempt[0].permitted=32
ctypes.set_errno(0)
returned=libc.capset(ctypes.byref(header),ctypes.byref(attempt))
regain=0 if returned==0 else ctypes.get_errno()
checks['capability_regain_denied']=returned==-1 and regain==1 and guard(status())
parent_env_absent='OPENECON_ROOT_DIAG_SENTINEL' not in os.environ
os.environ.clear()
os.environ.update({'PATH':'/opt/venv/bin:/usr/local/bin:/usr/bin:/bin','HOME':'/tmp',
                   'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
exec_code="import json,os;raw=open('/proc/self/status').read(8192);v={k:t.strip() for k,s,t in (line.partition(':') for line in raw.splitlines()) if k in ('CapEff','CapPrm','CapBnd','CapInh','CapAmb','NoNewPrivs')};print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),'groups':os.getgroups(),'status':v}))"
executed=subprocess.run([sys.executable,'-I','-c',exec_code],capture_output=True,timeout=3,check=True)
exec_status=json.loads(executed.stdout)
checks['fork_exec_capabilities_disabled']=guard(exec_status['status']) and exec_status['groups']==[]
path=Path(config['library_path'])
original=path.read_bytes()
checks['library_original_matches']=hashlib.sha256(original).hexdigest()==config['library_hash']
marker=Path('/tmp')/config['marker']
checks['fresh_tmp_marker_absent']=not marker.exists()
if config['mode']=='fresh':
    print(json.dumps({'checks':checks,'before':before,'after':after,'exec_status':exec_status,
                      'regain_errno':regain}),flush=True)
    raise SystemExit(0)
checks.update(parent_env_absent=parent_env_absent,parent_tmp_file_absent=not Path(config['sentinel_path']).exists(),
              parent_proc_tmp_absent=not Path('/proc/1/root'+config['sentinel_path']).exists(),
              parent_env_proc_absent=True,parent_fd_absent=True,launcher_absent=not Path('/usr/local/gcp/bin/sandbox').exists())
needle=config['sentinel'].encode()
for folder in list(Path('/proc').glob('[0-9]*'))[:128]:
    try:
        with (folder/'environ').open('rb') as stream:checks['parent_env_proc_absent'] &= needle not in stream.read(65536)
    except OSError:pass
    try:
        for fd in list((folder/'fd').iterdir())[:128]:
            try:checks['parent_fd_absent'] &= config['sentinel_path'] not in os.readlink(fd)
            except OSError:pass
    except OSError:pass
    checks['parent_proc_tmp_absent'] &= not (folder/'root'/config['sentinel_path'].lstrip('/')).exists()
checks['tmp_writable_after_cap_clear']=False
try:marker.write_text('fixed-inert-hardened-diagnostic');checks['tmp_writable_after_cap_clear']=marker.read_text()=='fixed-inert-hardened-diagnostic'
except OSError:pass
targets=set(config['parent_ips'])|{'127.0.0.1','::1'}
gateways=set()
try:
    for line in Path('/proc/net/route').read_text()[:32768].splitlines()[1:33]:
        fields=line.split()
        if len(fields)>3 and fields[1]=='00000000' and fields[2]!='00000000':
            gateways.add(socket.inet_ntoa(struct.pack('<I',int(fields[2],16))))
except OSError:pass
try:
    for line in Path('/proc/net/ipv6_route').read_text()[:32768].splitlines()[:32]:
        fields=line.split()
        if len(fields)>=10 and fields[0]=='0'*32 and fields[1]=='00' and fields[4]!='0'*32:
            address=str(ipaddress.IPv6Address(int(fields[4],16)))
            gateways.add(address+('%'+fields[-1] if ipaddress.ip_address(address).is_link_local else ''))
except OSError:pass
targets.update(gateways)
resolver="import json,socket,sys;print(json.dumps(sorted(set(item[4][0] for item in socket.getaddrinfo(sys.argv[1],None,0,socket.SOCK_STREAM)))))"
resolved=0
for name in config['parent_names'][:3]:
    try:
        value=subprocess.run([sys.executable,'-I','-c',resolver,name],capture_output=True,timeout=2,check=True)
        addresses=json.loads(value.stdout)
        for address in addresses[:8]:targets.add(str(ipaddress.ip_address(address)))
        resolved+=1
    except (OSError,ValueError,subprocess.SubprocessError):pass
network=[]
for host in sorted(targets)[:20]:
    for port in sorted({8080,*config['nonce_ports']}):
        reachable=False;http=False
        try:
            with socket.create_connection((host,port),timeout=.35) as connection:
                reachable=True
                connection.settimeout(.35)
                connection.sendall(b'GET /healthz HTTP/1.0\r\nHost: fixed-probe\r\n\r\n')
                http=bool(connection.recv(512))
        except OSError:pass
        network.append({'target':host,'port':port,'tcp_reachable':reachable,'http_response':http})
checks['parent_and_gateway_tcp_blocked']=bool(network) and all(not value['tcp_reachable'] for value in network)
checks['network_target_bound_complete']=len(targets)<=20
checks['parent_interfaces_present']=bool(config['parent_ips'])
checks['parent_nonce_listener_positive_control']=config['listener_positive']
from urllib.error import HTTPError
from urllib.request import Request,ProxyHandler,build_opener
opener=build_opener(ProxyHandler({}))
for host,key in [('169.254.169.254','metadata_ip_blocked'),('metadata.google.internal','metadata_dns_blocked')]:
    blocked=False
    try:
        with opener.open(Request('http://'+host+'/computeMetadata/v1/instance/id',headers={'Metadata-Flavor':'Google'}),timeout=1) as response:response.read(1)
    except HTTPError as error:blocked=error.code==403
    except OSError:blocked=True
    checks[key]=blocked
devices={}
for name in ('urandom','null'):
    try:
        with open('/dev/'+name,'rb') as stream:
            kind=stat.S_ISCHR(os.fstat(stream.fileno()).st_mode)
            value=stream.read(32 if name=='urandom' else 1)
        devices[name]=kind and (len(value)==32 if name=='urandom' else value==b'')
        if name=='null':
            with open('/dev/null','wb') as stream:devices[name] &= stat.S_ISCHR(os.fstat(stream.fileno()).st_mode) and stream.write(b'fixed-inert-probe')==17
    except OSError:devices[name]=False
checks.update(sandbox_urandom_available=devices['urandom'],sandbox_null_available=devices['null'])
started=time.monotonic()
from openecon.console import ConsoleSession
from openecon.team_job import _JobWorkspace
imported=time.monotonic()
folder=Path('/tmp')/('analysis-'+config['marker']);folder.mkdir()
session=ConsoleSession(_JobWorkspace(folder))
analysis="""from pathlib import Path
status = {key: value.strip() for key, sep, value in (line.partition(':') for line in Path('/proc/self/status').read_text().splitlines()) if key in ('CapEff', 'CapPrm', 'CapInh', 'CapAmb', 'NoNewPrivs')}
assert all(int(status[key], 16) == 0 for key in ('CapEff', 'CapPrm', 'CapInh', 'CapAmb')) and status['NoNewPrivs'] == '1'
df = oe.example()
display(df.head(3))
model = oe.ols(data=df, y='wage', x=['education', 'experience'])
display(model)
display(oe.plot.coefficients(model))
oe.plot.scatter(data=df, x='education', y='wage')
"""
try:record=session.execute(analysis,timeout_seconds=10)
finally:session.close()
outputs=record.get('outputs',[])
checks.update(native_analysis_ok=record.get('status')=='ok',native_table_model_two_plots=[value.get('type') for value in outputs]==['table','model','plot','plot'],
              native_observations_480=len(outputs)>1 and outputs[1].get('data',{}).get('nobs')==480)
timing={'import_seconds':imported-started,'analysis_seconds':time.monotonic()-imported}
checks['overlay_chmod_write_succeeded']=False
try:
    os.chmod(path,0o600);path.write_bytes(original+b'\n# fixed-overlay-only-diagnostic\n')
    checks['overlay_chmod_write_succeeded']=path.read_bytes()!=original and path.stat().st_mode&0o777==0o600
except OSError:pass
print(json.dumps({'checks':checks,'before':before,'after':after,'exec_status':exec_status,
                 'regain_errno':regain,'network':network,
                 'gateway_count':len(gateways),'resolved_parent_names':resolved,'analysis_timings':timing},allow_nan=False),flush=True)
'''

PARENT = r'''
import fcntl,hashlib,ipaddress,json,os,re,selectors,signal,socket,struct,subprocess,threading,time
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
import uvicorn

if os.getuid()!=0 or os.getpid()==1:raise RuntimeError('Root diagnostic requires a trusted Tini parent')
signal.signal(signal.SIGALRM,signal.SIG_DFL)
BIN='/usr/local/gcp/bin/sandbox'
PYTHON='/opt/venv/bin/python'
ROOTFS='/opt/openecon-sandbox-rootfs'
CHILD=CHILD_LITERAL
HARDENED_CHILD=HARDENED_CHILD_LITERAL
ENV={'PATH':'/opt/venv/bin:/usr/local/bin:/usr/bin:/bin','HOME':'/tmp'}
LOCK=threading.Lock()
BOOT=uuid4().hex
app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)

def parent_caps():
    result={}
    try:
        with open('/proc/self/status') as stream:raw=stream.read(8192)
        for line in raw.splitlines():
            key,sep,value=line.partition(':')
            if key in ('CapEff','CapPrm','CapBnd','CapInh','CapAmb') and sep:
                value=value.strip()
                if re.fullmatch(r'[0-9a-fA-F]{1,16}',value):result[key]=value.lower()
    except OSError:pass
    return result

def diagnostic(raw):
    value=raw[:4096].decode('utf-8',errors='replace')
    value=re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',value)
    value=re.sub(r'(?i)bearer\s+\S+','Bearer [redacted]',value)
    value=re.sub(r'https?://\S+','[url redacted]',value)
    return re.sub(r'[A-Za-z0-9_+/=.-]{64,}','[long value redacted]',value)

def bounded(process,timeout=12):
    stdout=bytearray();stderr=bytearray();total=0;started=time.monotonic()
    selector=selectors.DefaultSelector()
    selector.register(process.stdout,selectors.EVENT_READ,stdout)
    selector.register(process.stderr,selectors.EVENT_READ,stderr)
    try:
        while selector.get_map():
            if time.monotonic()-started>timeout:raise TimeoutError('Fixed deadline')
            for key,_ in selector.select(.1):
                part=os.read(key.fileobj.fileno(),4096)
                if not part:selector.unregister(key.fileobj);continue
                total+=len(part)
                if total>65536:raise ValueError('Fixed diagnostic output bound')
                if len(key.data)<16384:key.data.extend(part[:16384-len(key.data)])
        return bytes(stdout),bytes(stderr),process.wait(timeout=2)
    finally:selector.close()

def delete(name):
    process=subprocess.Popen([BIN,'delete',name,'--force'],stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True,env=ENV)
    result={'cleanup_confirmed':False,'delete_exit':None,'delete_timeout':False}
    try:
        _,error,code=bounded(process,timeout=10)
        result.update(cleanup_confirmed=code==0,delete_exit=code)
        if code:result['fixed_delete_diagnostic']=diagnostic(error)
    except TimeoutError:result['delete_timeout']=True
    except ValueError:result['delete_output_bound']=True
    finally:
        if process.poll() is None:process.kill()
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:pass
    return result

def help_result():
    process=subprocess.Popen([BIN,'run','--help'],stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,close_fds=True,env=ENV)
    try:
        raw,error,code=bounded(process,timeout=5)
        text=raw.decode('utf-8',errors='replace') if code==0 else ''
        flags=sorted(set(re.findall(r'--[a-z][a-z-]*',text)))
        return {'exit':code,'flags':flags,'template_var_supported':'--template-var' in flags,
                'fixed_diagnostic':diagnostic(error) if code else ''}
    finally:
        if process.poll() is None:process.kill()
        process.wait(timeout=2)

@app.get('/healthz')
def health():return {'ok':True}

@app.get('/state')
def state():return {'boot_nonce':BOOT,'parent_uid':os.getuid(),'parent_pid':os.getpid(),'parent_capabilities':parent_caps()}

@app.get('/launcher-help')
def help_route():return help_result()

def probe(rootfs=ROOTFS,drop=False,template=False,unknown=False,live=False):
    if not LOCK.acquire(blocking=False):raise HTTPException(status_code=409,detail='Fixed diagnostic busy')
    signal.setitimer(signal.ITIMER_REAL,40)
    name='oe-root-diag-'+uuid4().hex
    marker='oe-fixed-diag-'+uuid4().hex
    result={'parent_uid':os.getuid(),'parent_pid':os.getpid(),'launcher_exit':None,'child':None,
            'parent_capabilities':parent_caps(),'rootfs':'host' if rootfs=='/' else 'custom',
            'template_var':template,'live_delete':live}
    process=None;attempted=False;cleanup={'cleanup_confirmed':True}
    started=time.monotonic()
    try:
        if template:
            help=help_result()
            if not help['template_var_supported']:
                result['template_flag_unavailable']=True
                return result
        arguments=[BIN,'run',name,'--rootfs',rootfs,'--write','--allow-egress','--workdir','/tmp']
        if template:
            arguments+=['--template-var','openecon_unknown_fixed_key=10001'] if unknown else ['--template-var','uid=10001','--template-var','gid=10001']
        code=CHILD.replace('DROP_LITERAL',repr(drop)).replace('MARKER_LITERAL',repr(marker))
        if live:code="import json,time;print(json.dumps({'ready':True}),flush=True);time.sleep(300)"
        arguments+=['--',PYTHON,'-I','-c',code]
        attempted=True
        process=subprocess.Popen(arguments,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,close_fds=True,env=ENV)
        if live:
            selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
            ready=bool(selector.select(8));selector.close()
            raw=os.read(process.stdout.fileno(),4097) if ready else b''
            if raw.strip()==b'{"ready": true}':result['live_child_ready']=True
            else:
                raw,error,code=bounded(process,timeout=2)
                result.update(launcher_exit=code,fixed_launcher_diagnostic=diagnostic(error))
        else:
            raw,error,code=bounded(process)
            result['launcher_exit']=code
            if code:result['fixed_launcher_diagnostic']=diagnostic(error)
            else:
                child=json.loads(raw)
                if not isinstance(child,dict) or len(raw)>8192:raise ValueError('Fixed child shape')
                result['child']=child
        result['host_tmp_marker_absent']=not (Path(rootfs)/'tmp'/marker).exists()
    except (ValueError,TimeoutError,subprocess.TimeoutExpired):
        result['fixed_error']='Fixed diagnostic deadline or output shape'
    finally:
        if attempted:cleanup=delete(name)
        if process is not None and process.poll() is None:process.kill()
        if process is not None:
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:cleanup['cleanup_confirmed']=False
        result.update(cleanup)
        result['request_seconds']=time.monotonic()-started
        if not cleanup['cleanup_confirmed']:
            # Unknown deletion is never converted to success. Report only fixed
            # evidence, then destroy this disposable parent with its descendants.
            return JSONResponse(result,background=BackgroundTask(os._exit,70))
        try:(Path(rootfs)/'tmp'/marker).unlink(missing_ok=True)
        except OSError:pass
        signal.setitimer(signal.ITIMER_REAL,0)
        LOCK.release()
    return result

@app.get('/native-identity')
def identity():return probe()

@app.get('/native-identity-host-root')
def identity_host():return probe(rootfs='/')

@app.get('/drop-test')
def drop_test():return probe(drop=True)

@app.get('/template-identity')
def template_identity():return probe(template=True)

@app.get('/template-unknown-key')
def template_unknown():return probe(template=True,unknown=True)

@app.get('/delete-live')
def delete_live():return probe(live=True)

def parent_addresses():
    found=set()
    for _,name in socket.if_nameindex()[:16]:
        try:
            with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as stream:
                raw=fcntl.ioctl(stream.fileno(),0x8915,struct.pack('256s',name.encode()[:15]))
            value=ipaddress.ip_address(socket.inet_ntoa(raw[20:24]))
            if not value.is_loopback and not value.is_unspecified:found.add(str(value))
        except OSError:pass
    try:
        for line in Path('/proc/net/if_inet6').read_text()[:16384].splitlines()[:16]:
            value=ipaddress.IPv6Address(int(line.split()[0],16))
            if not value.is_loopback and not value.is_unspecified and not value.is_link_local:found.add(str(value))
    except (OSError,ValueError):pass
    return sorted(found)[:8]

def listener(family,nonce,stop):
    stream=socket.socket(family,socket.SOCK_STREAM)
    stream.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    if family==socket.AF_INET6:stream.setsockopt(socket.IPPROTO_IPV6,socket.IPV6_V6ONLY,1)
    stream.bind(('::' if family==socket.AF_INET6 else '0.0.0.0',0))
    stream.listen(8);stream.settimeout(.2)
    def serve():
        while not stop.is_set():
            try:connection,_=stream.accept()
            except OSError:continue
            try:
                with connection:
                    connection.settimeout(.5);connection.recv(2048)
                    connection.sendall(b'HTTP/1.0 200 OK\r\nContent-Length: '+str(len(nonce)).encode()+b'\r\n\r\n'+nonce.encode())
            except OSError:pass
    thread=threading.Thread(target=serve,daemon=True);thread.start()
    return stream,thread

@app.get('/secure-isolation')
def secure_isolation():
    if not LOCK.acquire(blocking=False):raise HTTPException(status_code=409,detail='Fixed diagnostic busy')
    signal.setitimer(signal.ITIMER_REAL,90)
    started=time.monotonic();stop=threading.Event();listeners=[];names=[];processes=[];deletion_done={}
    result={'passed':False,'parent_uid':os.getuid(),'checks':{},
            'proof_scope':'guest_privileges_filesystem_devices_native_analysis',
            'production_auth_boundary_verified':False}
    sentinel='fixed-parent-'+uuid4().hex
    sentinel_path=Path('/tmp')/('oe-parent-sentinel-'+uuid4().hex)
    sentinel_path.write_text(sentinel)
    sentinel_fd=sentinel_path.open('rb')
    env={**ENV,'OPENECON_ROOT_DIAG_SENTINEL':sentinel}
    root_library=next(Path(ROOTFS).glob('opt/venv/lib/python*/site-packages/torch/__init__.py'))
    if root_library.stat().st_size>2*1024**2:os._exit(70)
    library_path='/'+str(root_library.relative_to(ROOTFS))
    library_hash=hashlib.sha256(root_library.read_bytes()).hexdigest()
    library_mode=root_library.stat().st_mode
    library_inode=root_library.stat().st_ino
    package_library=Path(library_path)
    package_hash=hashlib.sha256(package_library.read_bytes()).hexdigest()
    package_mode=package_library.stat().st_mode
    package_inode=package_library.stat().st_ino
    package_device=package_library.stat().st_dev
    result['initial_rootfs_package_inode_pair_equal']=(root_library.stat().st_dev,library_inode)==(package_device,package_inode)
    cleanup_all=True;host_clean=True
    try:
        addresses=parent_addresses()
        nonce=uuid4().hex
        ipv4=listener(socket.AF_INET,nonce,stop);listeners.append(ipv4)
        try:listeners.append(listener(socket.AF_INET6,nonce,stop))
        except OSError:pass
        positive=True
        for address in addresses or ['127.0.0.1']:
            family=socket.AF_INET6 if ':' in address else socket.AF_INET
            matching=[item for item in listeners if item[0].family==family]
            if not matching:positive=False;continue
            try:
                with socket.create_connection((address,matching[0][0].getsockname()[1]),timeout=1) as connection:
                    connection.sendall(b'GET /nonce HTTP/1.0\r\nHost: fixed-probe\r\n\r\n')
                    raw=bytearray()
                    while len(raw)<2048:
                        part=connection.recv(2048-len(raw))
                        if not part:break
                        raw.extend(part)
                    positive &= nonce.encode() in raw
            except OSError:positive=False
        parent_names=[socket.gethostname()]
        service=os.environ.get('K_SERVICE','')
        if re.fullmatch(r'[a-z][a-z0-9-]{0,62}',service):parent_names.append(service)
        config={'mode':'active','library_path':library_path,'library_hash':library_hash,
                'marker':'oe-hardened-overlay-'+uuid4().hex,'parent_ips':addresses,
                'parent_names':parent_names,'nonce_ports':[item[0].getsockname()[1] for item in listeners],
                'listener_positive':positive,'sentinel':sentinel,'sentinel_path':str(sentinel_path)}
        results=[]
        for mode in ('active','fresh'):
            config['mode']=mode
            name='oe-secure-diag-'+uuid4().hex;names.append(name)
            code=HARDENED_CHILD.replace('CONFIG_LITERAL',repr(config))
            command=[BIN,'run',name,'--rootfs',ROOTFS,'--write','--allow-egress','--workdir','/tmp','--',PYTHON,'-I','-c',code]
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,close_fds=True,env=env)
            processes.append(process)
            raw,error,status=bounded(process,timeout=60)
            result['launcher_exit']=status
            if status:
                result['fixed_launcher_diagnostic']=diagnostic(error)
                raise ValueError('Fixed hardened child failed')
            child=json.loads(raw)
            if not isinstance(child,dict) or len(raw)>16384:raise ValueError('Fixed hardened child shape')
            result[mode]=child;results.append(child)
            cleanup=delete(name)
            deletion_done[name]=cleanup['cleanup_confirmed']
            cleanup_all &= cleanup['cleanup_confirmed']
            result[mode+'_cleanup']=cleanup
            if not cleanup_all:raise ValueError('Fixed cleanup failed')
            host_clean=(hashlib.sha256(root_library.read_bytes()).hexdigest()==library_hash
                        and root_library.stat().st_mode==library_mode and root_library.stat().st_ino==library_inode)
            if not host_clean:raise ValueError('Fixed host mutation detected')
        active_checks=results[0].get('checks',{})
        # Routed outbound networking can reach parent sockets on this platform.
        # Preserve this observed failure explicitly; app authentication must be
        # independently proven before any production execution is authorized.
        result['network_isolation_verified']=active_checks.get('parent_and_gateway_tcp_blocked') is True
        result['application_auth_boundary_required']=not result['network_isolation_verified']
        result['checks']={'active_nonnetwork_child_checks':all(value for key,value in active_checks.items() if key!='parent_and_gateway_tcp_blocked') and bool(active_checks),
                          'fresh_child_checks':all(results[1].get('checks',{}).values()) and bool(results[1].get('checks')),
                          'rootfs_library_content_unchanged':hashlib.sha256(root_library.read_bytes()).hexdigest()==library_hash,
                          'rootfs_library_mode_unchanged':root_library.stat().st_mode==library_mode,
                          'rootfs_library_inode_unchanged':root_library.stat().st_ino==library_inode,
                          'parent_package_content_unchanged':hashlib.sha256(package_library.read_bytes()).hexdigest()==package_hash,
                          'parent_package_mode_unchanged':package_library.stat().st_mode==package_mode,
                          'parent_package_inode_unchanged':package_library.stat().st_ino==package_inode,
                          'explicit_deletion_confirmed':cleanup_all}
        result['passed']=all(result['checks'].values())
    except (OSError,ValueError,TimeoutError,subprocess.TimeoutExpired):
        result['fixed_error']='Fixed hardened isolation proof failed'
    finally:
        for name,process in zip(names,processes):
            if name not in deletion_done:deletion_done[name]=delete(name)['cleanup_confirmed']
            cleanup_all &= deletion_done[name]
            if process.poll() is None:
                process.kill()
                try:process.wait(timeout=2)
                except subprocess.TimeoutExpired:cleanup_all=False
        host_clean=(hashlib.sha256(root_library.read_bytes()).hexdigest()==library_hash
                    and root_library.stat().st_mode==library_mode and root_library.stat().st_ino==library_inode
                    and hashlib.sha256(package_library.read_bytes()).hexdigest()==package_hash
                    and package_library.stat().st_mode==package_mode and package_library.stat().st_ino==package_inode)
        stop.set()
        for stream,thread in listeners:stream.close();thread.join(timeout=1)
        sentinel_fd.close();sentinel_path.unlink(missing_ok=True)
        result['request_seconds']=time.monotonic()-started
        if not cleanup_all or not host_clean:return JSONResponse(result,background=BackgroundTask(os._exit,70))
        signal.setitimer(signal.ITIMER_REAL,0);LOCK.release()
    return result

uvicorn.run(app,host='0.0.0.0',port=int(os.environ.get('PORT','8080')),access_log=False,log_level='critical')
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-source', type=Path, required=True)
    args = parser.parse_args()
    source = (PARENT.replace('HARDENED_CHILD_LITERAL', repr(HARDENED_CHILD))
              .replace('CHILD_LITERAL', repr(CHILD)))
    compile(source, '<fixed-root-parent-diagnostic>', 'exec')
    compile(CHILD.replace('DROP_LITERAL', 'False').replace('MARKER_LITERAL', repr('inert')),
            '<fixed-root-child-diagnostic>', 'exec')
    compile(HARDENED_CHILD.replace('CONFIG_LITERAL', repr({'mode': 'fresh'})),
            '<fixed-hardened-child-diagnostic>', 'exec')
    if len(source.encode()) > 32768:
        raise ValueError('Fixed root diagnostic must fit its source bound.')
    args.emit_source.parent.mkdir(parents=True, exist_ok=True)
    args.emit_source.write_text(source)
    print(json.dumps({'emitted_source': str(args.emit_source), 'source_bytes': len(source.encode()),
                      'sha256': hashlib.sha256(source.encode()).hexdigest(), 'cloud_calls': False}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
