# Why does `cuMemPoolImportPointer` die at ~5248 MiB? -- literature survey

Scope: literature only (web, public NVIDIA docs, public driver source at tag
580, GitHub code/issue search, on-box configuration reads). No GPU work.
Box: RTX 3090 24576 MiB, driver 580.126.20 (open kernel modules,
`nvidia-driver-580-open`), CUDA driver API 13.0, Linux 6.8.0-107, x86_64.
Repro: `/home/work/utils/comfy-env/research/pool-ipc/repro_mempool_import_segv.py`;
writeup: `/home/work/utils/comfy-env/research/pool-ipc/NVIDIA-BUG-REPORT.md`.

Bottom line up front:

1. **No public source documents or explains a ~5248 MiB (5,502,926,848 B)
   limit.** Not in the driver-API reference, runtime reference, programming
   guide, CUDA 13.0/13.1/13.2/13.3/13.4 release notes (new features, resolved,
   known issues), the R580 data-center driver notes (580.105.08 through
   580.178.04), R590 notes, NVIDIA forums, or GitHub issues in pytorch /
   cupy / jax / rmm / tensorrt / triton / ucx / cuda-python / nccl /
   open-gpu-kernel-modules. The number does not appear anywhere.
2. **The API is essentially unexercised at scale by anyone public.**
   NVIDIA's own samples import 64 MiB (`streamOrderedAllocationIPC`),
   4 MiB (`memMapIPCDrv`), 64 MiB (`simpleIPC`); cuda.core's IPC test suite
   allocates **64 bytes**; the python `ipcMemoryPool` sample takes a
   user-supplied element count. PyTorch's cudaMallocAsync backend hard-fails
   `shareIpcHandle`/`getIpcDevPtr` ("does not yet support"); UCX has an open
   issue that it cannot IPC cudaMallocAsync memory at all. So the silence in
   bug trackers is absence of use, not evidence of correctness.
3. **The bug report's strongest "kernel is clean" inference is wrong.** The
   UVM and RM ioctl paths return 0 from the syscall and put the real
   `NV_STATUS` inside the user struct (`params.rmStatus` /
   `NVOS*.status`). This is visible in source
   (`kernel-open/nvidia-uvm/uvm_api.h` `__UVM_ROUTE_CMD_STACK`) and was
   independently observed by a forum poster on 580.105.08 who caught
   `UVM_MAP_EXTERNAL_ALLOCATION` returning 0 with `rmStatus =
   NV_ERR_INSUFFICIENT_RESOURCES`. "Every ioctl returned 0 under strace"
   therefore proves nothing about where the NULL came from. This reopens
   the kernel side and is the first thing to discriminate.
4. **There is a concrete, size-dependent kernel rejection on the import
   path**: `nvGpuOpsBuildExternalAllocPtes` (RM, `nv_gpu_ops.c`) returns
   `NV_ERR_INVALID_LIMIT` (0x2E) when `offset + size >
   RM_ALIGN_UP(pMemDesc->ActualSize, pageSize)`, plus `NV_ERR_INVALID_ARGUMENT`
   for page-size/alignment mismatches and `NV_ERR_NOT_SUPPORTED` for a
   mapping-page-size override on compressed kinds. Whether any of these
   fires at 5264 MiB is unknown; it is exactly the class of "size-dependent
   error status that libcuda then mishandles into a NULL deref" the
   writeup hypothesised, and it is cheaply testable (see Experiments).

Everything below is what each source actually says, then the ranked
hypotheses with discriminating experiments. Guesses are labelled.

---

## 1. Official documentation -- what is (and is not) documented

### 1.1 Driver API reference, Stream Ordered Memory Allocator
URL: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html

- `cuMemPoolExportPointer`: "Data can be shared through any IPC mechanism";
  errors `CUDA_SUCCESS`, `CUDA_ERROR_INVALID_VALUE`, `CUDA_ERROR_NOT_INITIALIZED`,
  `CUDA_ERROR_OUT_OF_MEMORY`. No size language.
- `cuMemPoolImportPointer`: "The imported memory must not be accessed before
  the allocation operation completes in the exporting process. The imported
  memory must be freed from all importing processes before being freed in the
  exporting process." Same error set. **No size limit, no page-count limit,
  no per-allocation maximum.**
- `cuMemPoolExportToShareableHandle`: pool must be created with a handle type
  other than `CU_MEM_HANDLE_TYPE_NONE`; `flags` must be 0.
- `cuMemPoolImportFromShareableHandle`: "Imported memory pools do not support
  creating new allocations." Fabric handles require IMEX; otherwise
  `CUDA_ERROR_NOT_PERMITTED`.
- **No documented ordering requirement** between `ExportToShareableHandle`
  and `ExportPointer` (the writeup's secondary finding stands: undocumented).

### 1.2 Runtime API reference, memory pools
URL: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html

- `cudaMemPoolAttrMaxPoolSize`: "A value of 0 indicates no maximum size. For
  cudaMemAllocationTypeManaged **and IPC imported pools this value will be
  system dependent**." This is the only place the docs admit an
  implementation-defined ceiling on the *import* side. It is about pool size,
  not a single allocation, and no number is given. Still, it is the closest
  thing to an admission that the import side has a resource-bounded limit.

### 1.3 Programming guide, Stream-Ordered Memory Allocator (13.x)
URLs: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html
and https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html

- "Setting handleTypes to a non zero value will make the pool exportable (IPC
  capable)."
- "For cudaMemPoolExportToShareableHandle to succeed, the memory pool must
  have been created with the requested handle type."
- "Imported memory pools are initially only accessible from their resident
  device." (hence the `cuMemPoolSetAccess` in the repro's child)
- Export-pool limitation: "IPC pools currently do not support releasing
  physical blocks back to the OS. As a result the cudaMemPoolTrimTo API has
  no effect and the cudaMemPoolAttrReleaseThreshold is effectively ignored."
  (Consistent with the writeup's observation that pre-warming with
  `RELEASE_THRESHOLD = UINT64_MAX` changed nothing -- it is a no-op for IPC
  pools by spec.)
- Import-pool limitation: "Allocating from an import pool is not allowed";
  "The resource usage stat attribute queries only reflect the allocations
  imported into the process and the associated physical memory."
- No size limits, no page-count limits.

### 1.4 Programming guide, Interprocess Communication (legacy cudaIpc*)
URL: https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/inter-process-communication.html

- Legacy IPC: not for `cudaMallocManaged`; Linux only; "it is recommended to
  only share allocations with a 2MiB aligned size" (information-disclosure
  rationale, not a functional limit). Nothing about memory pools, nothing
  about size.

### 1.5 NVIDIA technical blog, stream-ordered allocator part 2
URL: https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2/

- Describes the export FD / import FD / `cudaMallocFromPoolAsync` /
  `ExportPointer` / `ImportPointer` flow and the free-ordering rule. Calls the
  pointer export "opaque data" without describing its contents. No limits.

### 1.6 CUDA Toolkit release notes (all 13.x), what they say about pool IPC
- 13.0: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html
  -- new: `cuMemCreate`/`cudaMallocAsync` on host (`CU_MEM_LOCATION_TYPE_HOST`),
  managed-memory discard APIs. Known issues: only the HMM/KASLR init
  failure. **Nothing on pool IPC, shareable handles, or large allocations.**
- 13.1: https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html -- nothing.
- 13.2 (ships driver 595.45.04): https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html -- nothing.
- 13.3 (ships driver 610.43.02): https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html -- nothing.
- 13.4 (current): https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
  -- new: "CUDA memory pool IPC is now supported on Windows WDDM systems for
  allocations created with the stream-ordered allocator." Nothing on Linux
  fixes.
- Compute Sanitizer 13.x notes (via search; the 13.4 dev-preview page 404s):
  "Compute Sanitizer tools do not support IPC memory pools, and using it will
  result in false positives" (so memcheck will not help), and a fix for "a
  crash when using cudaMallocAsync and cuMemmap in the same program with
  enabled peer-GPU access ... requires a driver version of 590 or more
  recent". That is a sanitizer-side crash, not this one, but it shows the
  pool/VMM interaction code was being touched in the 590 timeframe.

### 1.7 R580 data-center driver notes
- 580.126.20: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-126-20/index.html
  -- fixed: IMEX log-rotation handle staleness; an IMEX deadlock. Known:
  Hopper VBIOS, ClosedRM->OpenRM nvidia-smi, TCC/IOMMU on L40/L4, DCGM,
  RHEL10 ib_umad, fabricmanager-devel. **Nothing about CUDA memory.**
- 580.178.04: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-178-04/index.html
  -- fixed: FM MC-team leak, HBM3E repair, WPR mapping/speculative access
  kernel panic, GSP unload hang, Xid 32 with `CUDA_SCALE_LAUNCH_QUEUES`.
  Nothing about pools/IPC.
- 590.48.01: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-590-48-01/index.html
  -- `migrate_vma_state_t` kvmalloc fallback. Nothing relevant.
- The full `NVIDIA_Changelog` is only inside the `.run` installer and is not
  mirrored; not consulted. `dpkg -L` shows no changelog shipped by the Ubuntu
  580 packages either. **Gap: the consumer-driver changelog was not read.**

## 2. Forum / GitHub reports

### 2.1 The one directly relevant forum thread: status hidden in ioctl struct
URL: https://forums.developer.nvidia.com/t/580-105-08-cumemsetaccess-returns-oom-near-512k-aggregate-vmm-mappings-across-gpus-with-free-vram/382585

- On 580.105.08 / CUDA 13.0 (open kernel module), `cuMemSetAccess` returns
  `CUDA_ERROR_OUT_OF_MEMORY` with ~60 GiB free once ~524,288 aggregate VMM
  mappings exist across GPUs (1 GPU x 262,144 ok; 2 x 253,952 ok; 2 x 262,144
  fail).
- "The `UVM_MAP_EXTERNAL_ALLOCATION` ioctl returns syscall value 0 but
  contains NV_STATUS field 0x1a (`NV_ERR_INSUFFICIENT_RESOURCES`)".
- Poster correlates with `RS_UNIQUE_HANDLE_RANGE = 0x00080000` (524,288 RM
  handle slots). No NVIDIA reply.
- Relevance: (a) proves the strace inference in the bug report is invalid;
  (b) shows the kernel can hand libcuda an error on the very ioctl the
  pool import uses. It is a *count* limit, not a *size* limit, so it does not
  explain 5248 MiB directly (a single 5 GiB import is ~2624 x 2 MiB PTEs, or
  ~84k x 64 KiB, far from 512K), but it documents the failure *shape*.

### 2.2 Other forum threads (checked, not explanatory)
- https://forums.developer.nvidia.com/t/how-to-correctly-export-and-import-cuda-memory-pool/300607
  -- CUDA 11.8, V100: "double free detected in tcache 2" right after
  `cudaMemPoolImportFromShareableHandle`; suspected sender exiting early.
  No sizes, no resolution, no NVIDIA reply. Shows the import path can
  corrupt host heap state rather than return an error -- same genre as ours.
- https://forums.developer.nvidia.com/t/why-exporting-and-importing-cuda-ipc-handles-in-the-scope-of-the-same-linux-process-is-not-supported/252737
  -- `cudaIpcOpenMemHandle` same-process import unsupported (Robert
  Crovella: "it works the way it works"); VMM/mempool FD same-process import
  not tested there. Matters for the "single-process gdb" experiment below.
- https://forums.developer.nvidia.com/t/11-2-cudamempool-t-and-peer2peer/164070,
  https://forums.developer.nvidia.com/t/cudamempool-and-cuda-memcheck/197873
  -- unrelated (peer access; memcheck false positives on pools).
- VMM FD import threads (INVALID_DEVICE on import across processes, D3D12
  sub-page imports, DRIVE leak):
  https://forums.developer.nvidia.com/t/cuda-ipc-virtual-memory-api-cumemimportfromshareablehandle-cuda-error-invalid-device-cuda-11-3/176092,
  https://forums.developer.nvidia.com/t/cuda-ipc-low-level-vmm-cumemimportfromshareablehandle-returns-invalid-device-ordinal/303595,
  https://forums.developer.nvidia.com/t/memory-leak-when-using-virtual-memory-api-cumemimportfromshareablehandle/303267
  -- none size-related.
- NVIDIA-forum-restricted searches for "cuMemPoolImportPointer segmentation
  fault" and "cudaMallocAsync IPC large allocation crash": no hits.

### 2.3 GitHub issue search (gh search issues, all public repos)
- `cuMemPoolImportPointer`: 3 hits, none relevant (ZLUDA arch error;
  cuda-python #1074 "IPC tests do not exit cleanly" -- `CUDA_ERROR_INVALID_VALUE`
  during buffer close, 64-byte buffers; iree feature request).
- `cudaMemPoolImportPointer`: 1 hit (cuda-api-wrappers feature request #249).
- cuda-python IPC issues (40 scanned):
  https://github.com/NVIDIA/cuda-python/issues/2784 (SEGV in `cuMemPoolDestroy`
  under concurrent export/destroy, CUDA 12.9.1/13.0.2/13.3.0, aarch64),
  #2568 (`register()` segfault when IPC disabled), #2840 (import mutex/GIL
  deadlock), #2004, #1603, #1040... **No size-related report; the test suite
  uses NBYTES = 64** (`cuda_core/tests/memory_ipc/test_memory_ipc.py`,
  `test_send_buffers.py`).
- open-gpu-kernel-modules issue search for external allocation / mempool /
  MallocAsync: no hits.
- pytorch: https://github.com/pytorch/pytorch/issues/186213 -- expandable
  segments IPC receiver reserves 9/8 x totalGlobalMem of VA per imported
  handle and exhausts the 47-bit/128 TiB VA space after ~2,620 handles.
  Uses `cuMemImportFromShareableHandle`, not pool import; failure is a clean
  `invalid argument`, not a segfault. Relevant only as a reminder that
  importers over-reserve VA and that VA exhaustion surfaces as an error, not
  NULL.
- ucx: https://github.com/openucx/ucx/issues/7110 -- cudaMallocAsync memory
  cannot be IPC'd through legacy handles; open.
- PyTorch `c10/cuda/CUDAMallocAsyncAllocator.cpp`: `shareIpcHandle` and
  `getIpcDevPtr` both `TORCH_CHECK(false, "cudaMallocAsync does not yet
  support ...")`. So PyTorch never exercises pool IPC.

### 2.4 GitHub code search: who calls `cuMemPoolImportPointer` at all
`gh search code cuMemPoolImportPointer` / `cudaMemPoolImportPointer`: ~80
hits, almost all API stubs/shims (tinygrad autogen, HAMi-core, vcuda,
DeepRec/TF `.inc` tables, FEX thunks, PhoenixOS interceptors, ZLUDA-likes).
Real users: NVIDIA `cuda-samples` (64 MiB), NVIDIA `cuda-python` cuda.core
(64 B tests), `0h-n0/rucuda`, `etienne0114/vgre`. No large-buffer user found.

## 3. NVIDIA samples -- sizes and call order
- `cpp/2_Concepts_and_Techniques/streamOrderedAllocationIPC/streamOrderedAllocationIPC.cu`
  (https://github.com/NVIDIA/cuda-samples): `DATA_SIZE = 64 MiB`. Order on the
  exporter: `cudaMemPoolCreate` -> `cudaMallocAsync(DATA_SIZE, pool)` ->
  `cudaMemPoolExportToShareableHandle` -> `cudaMemPoolExportPointer`.
  Importer: `ImportFromShareableHandle` -> `GetAccess`/`SetAccess` ->
  `ImportPointer` -> kernel -> `cudaFreeAsync` (importer first). Comment at
  the memset: they zero the export blob "to make sure call to
  cudaMemPoolImportPointer ... fails" when export wasn't done -- i.e. NVIDIA
  expects a *failure return*, not a crash, from a bad blob.
  Note the sample allocates **before** exporting the pool handle and still
  exports the pointer fine, so the undocumented constraint is precisely
  "handle export must precede pointer export", not "precede allocation".
- `cpp/3_CUDA_Features/memMapIPCDrv/memMapIpc.cpp`: `DATA_BUF_SIZE = 4 MiB`
  (VMM path, `cuMemExportToShareableHandle`).
- `cpp/0_Introduction/simpleIPC/simpleIPC.cu`: `DATA_SIZE = 64 MiB` (legacy).
- `python/4_DistributedComputing/ipcMemoryPool/`: cuda.core, size from
  `--elements`; README says "Minimum GPU memory: 512 MB".
- Nobody at NVIDIA publicly imports a multi-GiB pool allocation.

## 4. NCCL as the control for "big single-handle FD import works"
`NVIDIA/nccl src/allocator.cc` `ncclMemAlloc`: one `cuMemCreate(handleSize)`
for the whole request (aligned to `cuMemGetAllocationGranularity`), one
`cuMemAddressReserve`, one `cuMemMap`; the P2P transport exports/imports that
single handle (`ncclP2pImportShareableBuffer`, issue #1647 shows the path).
Multi-GiB user buffers are routinely registered this way on data-center
GPUs. That is the VMM path, not the pool path, and not a 3090 -- but it
means "FD import of a >5 GiB single physical allocation" is not inherently
broken in the driver stack. Discriminating experiment E3 below runs the
same thing on this box.

## 5. Kernel-side source at tag 580 (open-gpu-kernel-modules)

### 5.1 ioctl status plumbing (the strace gap)
`kernel-open/nvidia-uvm/uvm_api.h`, `__UVM_ROUTE_CMD_STACK`:
```
params.rmStatus = uvm_global_get_status();
... params.rmStatus = function_name(&params, filp);
if (copy_to_user(...)) return -EFAULT;
return 0;
```
Every UVM ioctl returns 0 unless the copy itself faults; the driver's verdict
lives in `rmStatus`. `UVM_MAP_EXTERNAL_ALLOCATION` is `UVM_IOCTL_BASE(33)` =
plain `33` on Linux (`uvm_ioctl.h` line 40/491), so the request number is
literally 33 (no `_IOC` size encoding). Its params struct:
`base u64, length u64, offset u64, perGpuAttributes[UVM_MAX_GPUS], gpuAttributesCount u64,
rmCtrlFd s32, hClient u32, hMemory u32, rmStatus u32` with
`UVM_MAX_GPUS = NV_MAX_DEVICES(32) * UVM_PARENT_ID_MAX_SUB_PROCESSORS(8) = 256`
and `sizeof(UvmGpuMappingAttributes) = 16 + 5*4 = 36` -> **`rmStatus` at byte
offset 24 + 256*36 + 8 + 4 + 4 + 4 = 9260**, struct size 9264.
RM escapes (`src/nvidia/arch/nvalloc/unix/src/escape.c`) do the same:
`pApi->status = rmStatus` inside the NVOS struct, syscall returns 0.

### 5.2 UVM external mapping (`uvm_map_external.c`)
- Size checks: `uvm_api_range_invalid_4k(base, length)` ->
  `NV_ERR_INVALID_ADDRESS`; `gpuAttributesCount` in (0, UVM_MAX_GPUS] ->
  `NV_ERR_INVALID_ARGUMENT`; `uvm_gpu_can_address(gpu, base, length)` ->
  `NV_ERR_OUT_OF_RANGE`; `map_offset + node_size > mem_info->size` ->
  `NV_ERR_INVALID_OFFSET`. PTE staging loops in `MAX_PTE_BUFFER_SIZE = 96 KiB`
  chunks (`buffer_size = min(96K, num_all_ptes * pte_size)`), so it is
  size-independent, as the writeup said. No 32-bit page counts on the size
  path (`NvU32` only for `pte_size` / `num_ptes` per chunk).
- `uvm_va_range.c`: no length cap on `uvm_va_range_create_external`; only
  `NV_ERR_NO_MEMORY` on allocation failure.

### 5.3 RM side, `src/nvidia/src/kernel/rmapi/nv_gpu_ops.c`, `nvGpuOpsBuildExternalAllocPtes`
(lines ~3884-4000 at tag 580):
```
allocSize = RM_ALIGN_UP(pMemDesc->ActualSize, pageSize);
if (offset >= allocSize)            return NV_ERR_INVALID_BASE;     // 0x20
if ((offset + size) > allocSize)    return NV_ERR_INVALID_LIMIT;    // 0x2E
if (size   & (mappingPageSize-1))   return NV_ERR_INVALID_ARGUMENT; // 0x1F
if (offset & (mappingPageSize-1))   return NV_ERR_INVALID_ARGUMENT;
if (mappingPageSize > pageSize || pageSize % mappingPageSize) return NV_ERR_INVALID_ARGUMENT;
if (isCompressedKind && mappingPageSize != 0 && mappingPageSize != pageSize) return NV_ERR_NOT_SUPPORTED; // 0x56
pteCount = min(pteBufferSize/entrySize, mappingSize/mappingPageSize); if (!pteCount) return NV_ERR_BUFFER_TOO_SMALL; // 0x02
```
plus, higher up in `nvGpuOpsGetExternalAllocPtesOrPhysAddrs`, address-space
and peer/SLI checks returning `NV_ERR_NOT_SUPPORTED`. These are all things
that depend on how libcuda chose `offset`, `size`, `mappingPageSize` and on
how the pool's backing memdesc was created -- i.e. on the pool's physical
chunking for a given allocation size. If libcuda's pool importer walks the
allocation as a series of (memdesc, offset, size) pieces and one piece is
computed wrong for large allocations, this is where the kernel says no and
`rmStatus` comes back nonzero while the syscall returns 0.

### 5.4 `kernel-open/nvidia/os-mlock.c`
`os_lock_user_pages` does **not** check `RLIMIT_MEMLOCK`; page counts are
`NvU64`; no overflow check on `page_count * sizeof(*user_pages)`. So a
memlock rlimit cannot bite through the nvidia driver, only through direct
`mlock()`/`MAP_LOCKED` from libcuda (unknown; not traced -- the writeup only
traced ioctls).

## 6. On-box facts read during this survey (no GPU work)
- `nvidia-smi -q -d MEMORY`: FB 24576 MiB total, **454 MiB reserved**,
  BAR1 **256 MiB** (no resizable BAR). DMA size 47 bits. IOMMU groups present
  (22), no `iommu=` on the cmdline. `vm.max_map_count = 1048576`,
  `vm.overcommit_memory = 0`, RAM 32010 MiB.
- `ulimit -l = 8192 KiB` (RLIMIT_MEMLOCK 8 MiB). The writeup tested
  `ulimit -s unlimited` only. Given 5.4 this is low-prior, but it is the one
  rlimit not yet varied.
- `strings libcuda.so.580.126.20`: pool/IPC diagnostics include "Can't do
  IPC on device %d", "Can't export a pointer that doesn't belong to
  IPC-capable CUmemoryPool", "File descriptor to import must originate from
  CUDA Mempool APIs!", "Cannot import onto a CUmemoryPool of this type",
  "Handle type was not requested during pool creation". **No string about a
  maximum import size** -- consistent with the limit being an accident, not
  a checked constant. Env knobs found: `CUDA_DISABLE_MEMPOOL_REUSE_INTERNAL_DEPS`,
  `..._OPPORTUNISTIC`, `..._VIA_EVENT_DEPS` (allocator policy; not import).
- The number: 5248 MiB = 0x1_4800_0000 = 2^32 + 2^30 + 2^27; 5264 MiB =
  0x1_4900_0000 adds 2^24. In pages: 2624 x 2 MiB; 83,968 x 64 KiB;
  1,343,488 x 4 KiB. None is a power of two or a recognisable constant;
  5248 MiB is 21.4% of 24576 MiB, 21.8% of the usable 24122 MiB. No source
  gives any of these ratios meaning.

---

## 7. Ranked hypotheses and the experiments that separate them

Ranking is by (evidence found) x (cheapness of the decisive test).

### H1 -- kernel returns an `NV_STATUS` error inside an ioctl struct; libcuda's pool importer dereferences the NULL it then holds. (Most likely; directly source-backed mechanism.)
Support: 5.1 (status in struct, syscall 0), 2.1 (seen in the wild on 580 on
this exact ioctl), 5.3 (size-dependent `NV_ERR_INVALID_LIMIT` /
`INVALID_ARGUMENT` / `NOT_SUPPORTED` on the external-PTE path), fault
signature (NULL base + 0xd4, `SEGV_MAPERR`, "three frames below
`cuMemPoolImportPointer`", import walks the allocation).
Sub-variants: (a) `offset+size > RM_ALIGN_UP(ActualSize, pageSize)` because
libcuda computes the piece geometry wrongly past some chunk count; (b)
mapping page size (2 MiB vs 64 KiB) mismatch on a compressed PTE kind
(GA102 uses compressible kinds for ordinary vidmem); (c) an
`NV_ERR_NO_MEMORY`/`INSUFFICIENT_RESOURCES` from RM for page-table or
handle resources; (d) `uvm_gpu_can_address` -> `OUT_OF_RANGE` (unlikely).
Experiments:
- **E1 (decisive, 30 min, no root):** `LD_PRELOAD` an `ioctl()` shim in the
  child that, for fd == `/dev/nvidia-uvm` and request 33, reads the u32 at
  byte 9260 of `arg` after the call and logs `base,length,offset,hMemory,
  rmStatus`; for `/dev/nvidiactl` requests, log `_IOC_NR`, `_IOC_SIZE`, and
  the trailing u32 of the struct (NVOS54/NVOS21/NVOS64 put `status` last;
  for others dump the full struct and diff). Run 5248 vs 5264 and diff the
  sequences. Nonzero `rmStatus` (0x2E / 0x1F / 0x56 / 0x51 / 0x1A / 0x5B, see
  `nvstatuscodes.h`) on the last ioctl before the fault confirms H1 and
  names the sub-variant; identical all-zero traces with a divergent *count*
  of ioctls points at H3. Also log `mmap/munmap/mlock/madvise` returns in
  the same shim (the writeup traced ioctls only).
- **E2 (5 min, root):** `modprobe nvidia_uvm uvm_debug_prints=1
  uvm_release_asserts=1` and `NVreg_RmMsg=":"` (or `NVreg_ResmanDebugLevel=0`)
  then rerun 5264 and read `dmesg`; UVM prints `UVM_ERR_PRINT_NV_STATUS`
  lines on failing external maps.
- **E3 (control, 20 min):** same parent/child, but VMM path: `cuMemCreate(N,
  handleTypes=POSIX_FD)` -> `cuMemExportToShareableHandle` -> child
  `cuMemImportFromShareableHandle` -> `cuMemAddressReserve` -> `cuMemMap` ->
  `cuMemSetAccess` at 4096/5248/5264/8192 MiB. If VMM imports 8 GiB fine, the
  fault is in libcuda's pool-import geometry, not the kernel's ability to
  map 8 GiB (still compatible with H1a/H1b, but rules out H1c/H1d and H2 on
  the kernel side). NCCL (section 4) predicts VMM will pass.

### H2 -- the threshold is not a constant but proportional to an environmental resource (free VRAM, usable VRAM after the 454 MiB reservation, host RAM, an rlimit), so 5248 MiB is specific to this box. (Plausible; only indirect support.)
Support: docs call the imported-pool max "system dependent" (1.2); the
number has no clean bit pattern (6); none of the release notes give a
constant (1.6/1.7). Against: the writeup found the same threshold with
four 4 GiB allocations coexisting (16 GiB imported), which argues against
"free VRAM at import time" but not against "free VRAM at *allocation* time
in the parent" or "VA budget per allocation".
Experiments:
- **E4:** parent pre-occupies VRAM with a plain `cuMemAlloc` of 8 GiB before
  creating the pool; re-bisect. Threshold moves -> resource-proportional
  (and E1 will show which status). Threshold static -> H2 weakened.
- **E5:** `ulimit -l unlimited` (or `LimitMEMLOCK=infinity` in a systemd
  scope) and separately `ulimit -v`, `-d` variations; rerun 5264. Cheap;
  5.4 says nvidia.ko ignores memlock, so a change here would implicate a
  direct `mlock` in libcuda (E1's syscall log will show it).
- **E6:** same script on a different VRAM size (12 GiB or 48 GiB card, or
  the 3090 with `nvidia-smi --lock-gpu-clocks` irrelevant; use MIG-less
  cards only). Threshold scales with VRAM -> H2; constant -> H3/H1a.
- **E7:** newer driver on the same box (595.45.04 with CUDA 13.2 or 610.43.02
  with CUDA 13.3, both in Ubuntu's graphics-drivers PPA). A moved or
  vanished threshold says "libcuda bookkeeping changed", which is H3-flavoured
  evidence and is also what NVIDIA support will ask first.

### H3 -- pure userland: libcuda's pool-import bookkeeping (chunk table / lookup keyed by the export blob) returns NULL for allocations beyond some internal chunk count or blob field width, with no kernel error at all. (Plausible; fault site is consistent; no source evidence either way.)
Support: fault in libcuda with `rbx == 0`; export is O(1) but import is
O(size) so the importer iterates something per unit; forum thread 2.2 shows
the same import code corrupting heap state; the 64-byte
`CUmemPoolPtrExportData` must encode (pool id, chunk/handle id, offset,
size) and some field could be narrow.
Experiments:
- **E8 (userland only, no crash risk):** dump the 64-byte export blob for
  a ladder of sizes (1, 2, 4, 5120, 5248, 5264, 6144, 8192 MiB) in the
  *parent* and diff. Look for a field that stops growing / wraps / changes
  representation between 5248 and 5264; also check whether the blob for a
  second allocation in the same pool differs only by offset. If a field
  saturates at 5248, H3 is essentially proven.
- **E9 (gdb, 20 min):** `gdb --args python3 repro.py 5264`, `set
  follow-fork-mode child`, `set detach-on-fork off`, run to the SIGSEGV,
  `x/40i $pc-100` in frame 0 and `disassemble` frame 1 around the return
  address 0x...5a12 to see whether `rbx` came from a `call` (lookup /
  allocation) or a load from a table indexed by a size-derived value.
  Combine with `catch syscall ioctl` to correlate with E1. Also `break
  malloc` conditional on return NULL (or `MALLOC_PERTURB_`) to test the
  "internal allocation returned NULL" reading.
- **E10:** vary the parent-side ordering and chunking: (i) allocate
  *before* `ExportToShareableHandle` (the NVIDIA sample's order); (ii)
  two allocations of 2640 MiB each then import both (writeup shows 4 x
  4096 imports fine -- confirm the per-allocation limit is truly on the
  single allocation, not on the largest physical chunk, by pre-warming with
  a single 5264 MiB alloc, freeing it, then allocating 2 x 2632 MiB); (iii)
  `cuMemPoolSetAttribute(CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH...)` is
  read-only; skip. Any shift in the boundary implicates chunk geometry
  (H1a/H3) over a byte constant.
- **E11:** same-process import (parent imports its own FD and blob). The
  forum says legacy `cudaIpcOpenMemHandle` refuses this; pool FDs are not
  documented either way. If it works and still crashes at 5264, you get a
  one-process gdb target with symbols for everything but libcuda.

### H4 -- a 32-bit or narrow-field overflow somewhere in the import path. (Weak as stated; the boundary is not at any 2^n.)
The writeup already rules out a 2^32-byte boundary (4608 and 5248 MiB
import). No 2^n page-count boundary matches either (see 6). Only a
non-power-of-two capacity (a table sized from another quantity, e.g.
`totalGlobalMem / k`) would fit, which collapses into H2/H3. E8 (blob diff)
and E6 (other VRAM size) discriminate.

### H5 -- numerology (guesses, listed only so they are on record)
- 5248 MiB = 2^32+2^30+2^27 vs 5264 adds 2^24: a buddy-style split into
  power-of-two chunks would have 3 chunks vs 4. But 6144 MiB (2 chunks) also
  crashes, so chunk *count* is not the trigger. Discard unless E8 shows
  chunk lists in the blob.
- 6144 MiB / (7/6) = 5266 MiB lands in the [5248, 5264) gap; no source gives
  a 7/6 ratio anything. Guess, no support.
- 5248 MiB is ~21.4% of 24576 MiB. No source. Guess; E6 tests it.

### Explicitly not supported by any source found
- BAR1 (256 MiB here) is irrelevant: same-GPU device-memory imports do not
  go through BAR1 and 1-5 GiB imports work.
- `vm.max_map_count` (1,048,576 here) is far above any plausible per-import
  host mapping count.
- No GeForce/consumer-specific restriction on memory-pool IPC is documented
  anywhere (the 13.4 note about WDDM is Windows-only); no 3090/GA102-specific
  report exists.
- IOMMU: nothing found linking IOMMU to pool IPC; all mappings here are
  GPU-side vidmem PTEs, not DMA mappings.

## 8. What the bug report should change before filing
1. Replace "every ioctl returns 0 so the kernel reported no error" with the
   E1 result (decoded `rmStatus` / `NVOS*.status`). As written, an NVIDIA
   triager who knows `__UVM_ROUTE_CMD_STACK` will discount the whole "closed
   source libcuda" attribution.
2. Add the VMM control (E3) and the blob diff (E8); together they say whether
   the bug is pool-specific, which is what decides the owning team.
3. Note the RLIMIT_MEMLOCK = 8 MiB on the test box, and the driver flavour
   (`nvidia-driver-580-open`, GSP firmware 580.126.20).
4. Keep the per-allocation-vs-cumulative finding (4 x 4096 MiB imports OK);
   it is the strongest evidence against a simple resource ceiling and the
   thing E10 sharpens.

## 9. All URLs consulted
Docs: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html ;
https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html ;
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html ;
https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html ;
https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/inter-process-communication.html ;
https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html ;
https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html ;
https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html ;
https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html ;
https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html ;
https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-126-20/index.html ;
https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-580-178-04/index.html ;
https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-590-48-01/index.html ;
https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-2/ ;
https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/ .
Forums: https://forums.developer.nvidia.com/t/580-105-08-cumemsetaccess-returns-oom-near-512k-aggregate-vmm-mappings-across-gpus-with-free-vram/382585 ;
https://forums.developer.nvidia.com/t/how-to-correctly-export-and-import-cuda-memory-pool/300607 ;
https://forums.developer.nvidia.com/t/why-exporting-and-importing-cuda-ipc-handles-in-the-scope-of-the-same-linux-process-is-not-supported/252737 ;
https://forums.developer.nvidia.com/t/cuda-ipc-virtual-memory-api-cumemimportfromshareablehandle-cuda-error-invalid-device-cuda-11-3/176092 ;
https://forums.developer.nvidia.com/t/cuda-ipc-low-level-vmm-cumemimportfromshareablehandle-returns-invalid-device-ordinal/303595 ;
https://forums.developer.nvidia.com/t/memory-leak-when-using-virtual-memory-api-cumemimportfromshareablehandle/303267 ;
https://forums.developer.nvidia.com/t/11-2-cudamempool-t-and-peer2peer/164070 ;
https://forums.developer.nvidia.com/t/cudamempool-and-cuda-memcheck/197873 .
GitHub: https://github.com/pytorch/pytorch/issues/186213 ;
https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDAMallocAsyncAllocator.cpp ;
https://github.com/openucx/ucx/issues/7110 ;
https://github.com/NVIDIA/cuda-python/issues/2784 ; https://github.com/NVIDIA/cuda-python/issues/1074 ;
https://github.com/NVIDIA/cuda-python/tree/main/cuda_core/tests/memory_ipc ;
https://github.com/NVIDIA/cuda-samples/blob/master/cpp/2_Concepts_and_Techniques/streamOrderedAllocationIPC/streamOrderedAllocationIPC.cu ;
https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/memMapIPCDrv/memMapIpc.cpp ;
https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleIPC/simpleIPC.cu ;
https://github.com/NVIDIA/cuda-samples/tree/master/python/4_DistributedComputing/ipcMemoryPool ;
https://github.com/NVIDIA/nccl/blob/master/src/allocator.cc ; https://github.com/NVIDIA/nccl/issues/1647 ;
https://github.com/NVIDIA/open-gpu-kernel-modules (tag 580): kernel-open/nvidia-uvm/uvm_api.h,
uvm_ioctl.h, uvm_types.h, uvm_map_external.c, uvm_va_range.c, uvm.c;
kernel-open/nvidia/os-mlock.c, nv.c; kernel-open/common/inc/nvlimits.h;
src/nvidia/src/kernel/rmapi/nv_gpu_ops.c; src/nvidia/arch/nvalloc/unix/src/escape.c;
src/common/sdk/nvidia/inc/nvstatuscodes.h .
Local copies of the kernel sources used are in this scratchpad directory.
