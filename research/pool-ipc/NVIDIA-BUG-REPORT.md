# cuMemPoolImportPointer SIGSEGVs the importing process for single allocations above 5248 MiB

**Component:** CUDA driver (stream-ordered memory allocator / memory-pool IPC)
**Severity:** crash (SIGSEGV) in the calling process, no error code returned
**Driver:** 580.126.20 (CUDA driver API 13.0)
**GPU:** NVIDIA GeForce RTX 3090 (GA102, 24576 MiB)
**OS:** Linux 6.8.0 x86_64
**Repro:** `repro_mempool_import_segv.py` — standalone, Python 3.9+ only, no CUDA
toolkit and no third-party packages (ctypes against `libcuda.so.1`)

## Summary

A process that imports a memory-pool allocation shared over
`CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` is killed by SIGSEGV inside
`cuMemPoolImportPointer()` when the exported allocation is larger than
5248 MiB. No error code is returned — the process dies. Exporting works
at every size; only the import crashes.

## Reproduction

    python3 repro_mempool_import_segv.py

Parent: `cuMemPoolCreate(handleTypes=POSIX_FILE_DESCRIPTOR)` →
`cuMemPoolExportToShareableHandle` → `cuMemAllocFromPoolAsync(N)` →
`cuMemPoolExportPointer`. The FD goes to a child over SCM_RIGHTS.
Child: `cuMemPoolImportFromShareableHandle` → `cuMemPoolSetAccess` →
`cuMemPoolImportPointer`.

## Measured boundary

| allocation | result |
|---|---|
| 1024 MiB | imports OK, 1.34 ms |
| 4096 MiB | imports OK, 5.16 ms |
| 5120 MiB | imports OK, 6.64 ms |
| **5248 MiB** | **imports OK, 6.58 ms — last good size** |
| **5264 MiB** | **SIGSEGV in the child — first bad size** |
| 6144 / 8192 / 16384 MiB | SIGSEGV |

Bisected at 16 MiB granularity: the threshold sits between 5248 MiB
(5,502,926,848 bytes) and 5264 MiB (5,519,704,064 bytes). Import time is
otherwise linear in allocation size at roughly 1.25 ms/GiB, while export
is O(1) — the import path evidently walks the allocation.

## Fault site

    Thread 1 received signal SIGSEGV
    #0  0x00007ffff1ca0ac8 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
    #1  0x00007ffff19c5a12 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
    #2  0x00007ffff1b5ad32 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
    #3  0x00007ffff1b4e610 in cuMemPoolImportPointer () from libcuda.so.1

    => 0x7ffff1ca0ac8:  mov 0xd4(%rbx),%r12d
    rbx  0x0

The kernel agrees on the faulting address:

    SIGSEGV {si_signo=SIGSEGV, si_code=SEGV_MAPERR, si_addr=0xd4}

`si_addr` is exactly the structure offset applied to a NULL base, and
`SEGV_MAPERR` means the address is simply unmapped. An internal
allocation or lookup returns NULL on a size-dependent path and the
result is dereferenced without a check.

Under `strace -ff -e trace=ioctl`, **every ioctl against the NVIDIA
device nodes returns 0** right up to the fault — the kernel driver
reports no error at any point, so the NULL originates in `libcuda`'s own
bookkeeping rather than a failed kernel request.

An audit of `NVIDIA/open-gpu-kernel-modules` at tag 580.126.20 found no
size threshold that could explain this: `uvm_map_external.c` bounds its
PTE staging buffer at `MAX_PTE_BUFFER_SIZE` (96 KiB) and loops, so
external mapping is size-independent, and `memdescCreate` explicitly
guards its own 32-bit truncation. That points at closed-source
`libcuda.so` rather than the kernel module.

## Ruled out

- **Not host OOM.** 27 GiB of host RAM was free.
- **Not device OOM.** The allocation itself succeeds on the exporter side;
  only the import crashes, and the device has 24576 MiB with the pool
  otherwise empty.
- **Not stack exhaustion.** Tested two independent ways: running the
  import on a pthread with a 512 MiB stack, and running the whole
  process under `ulimit -s unlimited`. Both crash identically at the
  same threshold, and a stack overflow would fault adjacent to the stack
  mapping rather than at address 0xd4.
- **Not a 2^32 boundary.** 4608 MiB and 5248 MiB both import fine.
- **Not pool segmentation.** Pre-warming the pool into a single 8 GiB
  segment, with `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = UINT64_MAX`, does
  not change the threshold.
- **Not cumulative.** Four separate 4096 MiB allocations (16 GiB total)
  from the same pool all import successfully into one child. The limit
  applies per allocation.
- **No lasting corruption.** A 5248 MiB import succeeds normally after a
  crash in a previous child.

## Expected behaviour

Either the import succeeds, or `cuMemPoolImportPointer` returns an error
code. Segfaulting the caller gives an application no way to detect or
recover from the condition. If a per-allocation size limit for pool IPC
is intended, it should be documented and reported as
`CUDA_ERROR_INVALID_VALUE` or `CUDA_ERROR_OUT_OF_MEMORY`.

## Secondary issue: undocumented ordering requirement

`cuMemPoolExportPointer()` returns `CUDA_ERROR_INVALID_VALUE` (1) for every
allocation unless `cuMemPoolExportToShareableHandle()` has already been
called on that pool. Neither the Stream-Ordered Memory Allocation section
of the programming guide nor the driver API reference for
`cuMemPoolExportPointer` mentions this ordering constraint, and the error
code gives no hint. Suggest documenting it, or removing the dependency.
