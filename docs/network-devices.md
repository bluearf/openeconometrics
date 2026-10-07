# Sparse network devices

Resident `Network` graphs use CPU sparse storage. The following operations accept
an explicit `device='cpu'`, `'cuda'`, or `'cuda:<index>'` argument. CPU remains the
default. Each call admits its host workspace and checks free device memory before
upload. An absent/invalid device raises `network_device`; an insufficient budget
or allocator OOM raises `network_memory_budget`. Neither case falls back to CPU
or returns an incomplete result. CUDA is optional and has no runtime dependency
on SciPy, NetworkX, or statsmodels.

| Operation | CPU | CUDA | Arithmetic | Meaning |
| --- | --- | --- | --- | --- |
| `degree` | Yes | Implemented; hardware validation pending | int64 counts, float64 strengths | Unique edges, aggregate strengths; undirected loops count twice |
| `pagerank` | Yes | Implemented; hardware validation pending | float64 | Weighted sparse power iteration, personalized dangling mass, contraction bound |
| `eigenvector` | Yes | Implemented; hardware validation pending | float64 | Shifted incoming eigenvector, step and relative eigen-residual checks |
| `hits` | Yes | Implemented; hardware validation pending | float64 | Hub/authority iteration, both singular equations checked |
| `katz` | Yes | Implemented; hardware validation pending | float64 | Explicit orientation, sufficient contraction bound, exact zero-alpha path |
| `modularity` | Yes | Implemented; hardware validation pending | float64 | Score a complete partition, directed null model, doubled undirected loops |
| Louvain/Leiden optimization, block-model fitting | Yes | Unavailable | float64 | Scalar move/restart orchestration remains on CPU |
| Other resident methods and disk-backed algorithms | Yes | Unavailable | Method-specific | No implicit dispatch to these new kernels |
| All listed methods on Metal/MPS | Unavailable | — | — | Explicit error: Metal does not support float64 |

GPU community support is the modularity **scoring primitive**, not a GPU Louvain,
Leiden, or block-fitting implementation. Local move/sweep loops synchronize many
small scalar decisions; batching these methods without changing their objective,
restart independence and convergence contracts requires separate work. No GPU
block-fitting claim is made.

```python
scores = graph.hits(device='cuda:0')
display(scores)
print(scores.attrs['device'], scores.attrs['dtype'])
quality = graph.modularity(membership, device='cuda:0')
```

The graph retains its CPU indices and weights. Each call uploads the required
sparse indices/weights once; iteration vectors stay on the requested device.
Personalization/Katz baseline maps are validated on CPU and uploaded once.
Convergence scalar reductions synchronize the device. Final table columns are
copied to CPU. There is no persistent GPU graph cache. `modularity` returns a
Python float; its explicit device argument controls scoring while
`graph.metadata['device']` continues to describe CPU graph storage. CUDA indexed
reductions may vary in their final rounding; tolerances and residuals remain
required. GPU execution is not assumed to be faster.

`workspace_estimate_bytes` covers planned algorithm buffers, excluding the CUDA
context, caching allocator reserve, other processes and caller input. The free
memory check cannot prevent a later allocation failure; that failure is caught.
Degree counts remain int64; the reported float64 dtype refers to strength and
centrality arithmetic.

## Validation and profiling

`tests/test_network_device.py` checks independent linear-system, eigensystem,
singular-vector and exact rational modularity references. It also tests invalid
devices, missing CUDA, preflight memory refusal and injected allocator OOM. Real
CPU/CUDA comparisons have explicit hardware skips on this Mac, not passing
emulated results.

Run the scale script with the project Python and source paths configured:

```sh
python scripts/benchmark_network_devices.py --require-cuda --output /tmp/network-cuda.json
```

The script creates a fresh process per method/device/size, measures the first
complete call and three subsequent calls, synchronizes CUDA timing, measures
separate sparse upload/edge-value return times, records peak allocated/reserved
GPU bytes and NVIDIA driver/device inventory, and compares CPU/GPU values. The
weighted regular circulant has independent uniform-score formulas. These sizes
measure one regular workload; they do not characterize all graph structures or
convergence rates. First-call timing follows graph construction/device inspection
and includes upload and result conversion. It is not a cold driver startup
measurement. A separate PyTorch Profiler call records the twelve operators with
largest CPU total time and their device time. Profiling is outside timed calls.

On this Apple Silicon Mac, CUDA is unavailable. The
[CPU measurement receipt](evidence/market-53-devices-2026-10-07.json) records that
limitation explicitly. MARKET-53 remains open until the real CUDA comparison and
installed application device-recognition evidence are collected. The runnable
[Mac example](examples/network_devices.py) displays the five supported tables and
tests explicit Metal/absent-CUDA refusal.

PyTorch references: [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda),
[free/total device memory](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.mem_get_info.html),
and [allocator OOM](https://docs.pytorch.org/docs/stable/generated/torch.cuda.OutOfMemoryError.html).
