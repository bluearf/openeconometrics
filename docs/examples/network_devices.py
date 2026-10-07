"""Explicit sparse float64 devices; safe to run inside the installed Mac app."""
import json
import torch
import openecon as oe
from openecon.analysis_contracts import AnalysisError

show = globals().get('display', print)
g = oe.network([dict(source=u, target=v, weight=w) for u, v, w in
                [(0, 1, 2), (1, 0, 1), (1, 2, 3), (2, 0, 2), (2, 2, 1)]],
               directed=True, weight='weight', nodes=range(4))
devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
results = []
for device in devices:
    for method in ('degree', 'pagerank', 'eigenvector', 'hits', 'katz'):
        result = getattr(g, method)(device=device)
        assert result.attrs['dtype'] == 'float64' and not result.attrs['device_fallback']
        assert result.node.tolist() == list(range(4))
        show(result)
        results.append(dict(method=method, device=result.attrs['device'],
                            iterations=result.attrs.get('iterations'),
                            workspace_estimate_bytes=result.attrs['workspace_estimate_bytes']))
    assert abs(g.modularity([0, 0, 1, 2], device=device) + 2 / 27) < 1e-12
try:
    g.pagerank(device='mps')
except AnalysisError as error:
    assert error.code == 'network_device' and 'float64' in str(error)
else:
    raise AssertionError('Metal float64 must be refused')
if not torch.cuda.is_available():
    try:
        g.degree(device='cuda')
    except AnalysisError as error:
        assert error.code == 'network_device'
    else:
        raise AssertionError('Absent CUDA must be refused')
device_proof = dict(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                    mps_available=torch.backends.mps.is_available(),
                    device_name=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                    metal_float64_refused=True, absent_cuda_refused=not torch.cuda.is_available(),
                    graph_storage_device=g._edges.device.type, results=results)
print('NETWORK_DEVICE_PROOF:' + json.dumps(device_proof, allow_nan=False))
