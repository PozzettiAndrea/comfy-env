# CUDA IPC across every torch on this machine

Generated 2026-09-13 03:01 by `matrix_torch_ipc.py`. GPU/driver: NVIDIA GeForce RTX 3090, 580.126.20. Kernel 6.8.0-107-generic.

| env | torch | cuda | py | native → async (receiver on cudaMallocAsync) | async → native (sender on cudaMallocAsync) | native → native (control) | mempool import 5248 / 5264 MiB |
|---|---|---|---|---|---|---|---|
| 3d-pack-enved-nodes-py310-torch2-10-cu128 | 2.10.0+cu128 | 12.8 | 3.10 | RAISES: does not yet support getIpcDevPtr | RAISES: does not yet support shareIpcHandle | OK (bytes match) | 5248 MiB  imported OK in   6.61 ms / 5264 MiB  CHILD DIED, exit status -11  (SIGSEGV) |
| 3d-pack-enved-nodes-py313-torch2-10-cu128 | 2.10.0+cu128 | 12.8 | 3.13 | RAISES: does not yet support getIpcDevPtr | RAISES: does not yet support shareIpcHandle | OK (bytes match) | 5248 MiB  imported OK in   6.61 ms / 5264 MiB  CHILD DIED, exit status -11  (SIGSEGV) |
| 3d-pack-enved-nodes-py313-torch2-8-cu128 | 2.8.0+cu128 | 12.8 | 3.13 | RAISES: does not yet support getIpcDevPtr | RAISES: does not yet support shareIpcHandle | OK (bytes match) | 5248 MiB  imported OK in   6.61 ms / 5264 MiB  CHILD DIED, exit status -11  (SIGSEGV) |
| depthanythingv3-nodes | 2.10.0+cu130 | 13.0 | 3.11 | RAISES: does not yet support getIpcDevPtr | RAISES: does not yet support shareIpcHandle | OK (bytes match) | 5248 MiB  imported OK in   6.64 ms / 5264 MiB  CHILD DIED, exit status -11  (SIGSEGV) |
| sam3dbody-nodes | 2.11.0+cu128 | 12.8 | 3.13 | RAISES: does not yet support getIpcDevPtr | RAISES: does not yet support shareIpcHandle | OK (bytes match) | 5248 MiB  imported OK in   6.65 ms / 5264 MiB  CHILD DIED, exit status -11  (SIGSEGV) |
