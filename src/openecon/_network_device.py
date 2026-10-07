"""Explicit float64 execution and allocation policy for resident sparse graphs."""
from contextlib import contextmanager

import torch

from openecon.analysis_contracts import AnalysisError


def resolve(device):
    if not isinstance(device, str) or not (device in ('cpu', 'cuda') or device.startswith('cuda:')):
        raise AnalysisError('network_device', 'Use CPU or CUDA float64; Metal does not support float64.')
    try:
        selected = torch.device(device)
        if selected.type == 'cuda':
            if not torch.cuda.is_available():
                raise AnalysisError('network_device', 'The requested CUDA device is unavailable.')
            index = torch.cuda.current_device() if selected.index is None else selected.index
            if index >= torch.cuda.device_count():
                raise AnalysisError('network_device', 'The requested CUDA device index is unavailable.')
            selected = torch.device('cuda', index)
    except AnalysisError:
        raise
    except (RuntimeError, ValueError) as exc:
        raise AnalysisError('network_device', 'Invalid or inaccessible network device.') from exc
    return selected


@contextmanager
def execution(graph, device, workspace):
    """Check host and device buffers before upload; refuse allocation failure.

    Graph storage remains on CPU. Algorithms upload sparse indices/weights once
    per call and retain iteration vectors on the selected device. Convergence
    diagnostics synchronize scalar reductions; no persistent GPU cache exists.
    The estimate excludes the CUDA context/allocator cache and other processes.
    """
    selected = resolve(device)
    graph._guard(workspace)
    try:
        if selected.type == 'cuda':
            try:
                free, _ = torch.cuda.mem_get_info(selected)
            except (RuntimeError, ValueError) as exc:
                raise AnalysisError('network_device', 'Cannot inspect the requested CUDA device.') from exc
            if workspace > free:
                raise AnalysisError('network_memory_budget', 'Network buffers exceed available CUDA memory.')
        with torch.device(selected), torch.no_grad():
            yield selected
    except torch.OutOfMemoryError as exc:
        raise AnalysisError('network_memory_budget',
                            'Network allocation failed; no partial result or CPU fallback is returned.') from exc


def metadata(selected, workspace):
    return dict(device=str(selected), dtype='float64', workspace_estimate_bytes=workspace,
                transfer_policy='CPU graph; one sparse upload per call; device-local iteration; CPU result',
                device_fallback=False)
