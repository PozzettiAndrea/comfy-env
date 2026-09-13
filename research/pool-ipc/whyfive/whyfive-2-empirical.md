# Why the cuMemPoolImportPointer boundary is ~5248 MiB — empirical

**Machine:** RTX 3090 (GA102, 24576 MiB total, ~24122 usable), driver 580.126.20,
CUDA driver API 13.0, Linux 6.8.0, x86_64. GPU idle apart from these trials.
All GPU commands wrapped in `flock .../gpu.lock`. Scripts in scratchpad:
`exp.py` (instrumented harness derived from the repro), `hog.py`, `hog_run.sh`.

## TL;DR — root cause

The limit is **not** a memory/VA/resource limit and **not** a byte value like
5250/5252. It is a **fixed-capacity 164-entry buffer overflow inside
`libcuda.so`'s mempool-IPC pointer-import path.** The importer splits the
allocation into **32 MiB chunks** (`ceil(size/32MiB)`, allocation rounds up to
32 MiB granularity) and builds one 0xB0-byte descriptor per chunk into a buffer
sized for **164 chunks**. 164 × 32 MiB = 5248 MiB = 5,502,926,848 bytes exactly.
The **165th** chunk descriptor overruns the buffer, smashes an adjacent object
pointer to NULL, and libcuda then dereferences it (`mov 0xd4(%rbx),%r12d`,
rbx=0) → SIGSEGV at `si_addr=0xd4`. The VMM (cuMemCreate/Map) IPC path has no
such bug and imports 8192/16384 MiB fine.

## 1. Exact byte boundary (priority 1)

`exp.py --unit bytes` around 5248 MiB (5248 MiB = 5,502,926,848 B):

| bytes | = | reserved | result |
|---|---|---|---|
| 5,501,878,272 | 5247 MiB | 5248 | OK |
| 5,502,926,847 | 5248 MiB − 1 B | 5248 | OK |
| **5,502,926,848** | **5248 MiB exactly** | **5248** | **OK (last good)** |
| **5,502,926,849** | **5248 MiB + 1 B** | **5280** | **SIGSEGV (first bad)** |
| 5,536,481,280 | 5280 MiB | 5280 | SIGSEGV |

The boundary is exactly **5248.000000 MiB**, i.e. **164 × 32 MiB**. One byte over
rounds the allocation up to 5280 MiB = **165 × 32 MiB** and crashes. So it is NOT
5250/5252 — it is precisely the 164→165 chunk-count step. 2-MiB bisect
(5250,5252,…,5262) all crash, consistent (all round to ≥165 chunks).

The chunk count is directly visible in the crash registers (gdb, `run9`):
plain 5264 MiB and `--first 16 --sizes 5248` both crash with **r13 = 0xa5 = 165**.

## 2. GPU memory pressure — boundary does NOT move (priority 2)

`hog.py` holding device memory via cuMemAlloc, then the ladder:

| hog | free at import | 5248 | 5264 | 5120 | 4096 |
|---|---|---|---|---|---|
| 8 GiB | ~9985 MiB | OK | SIGSEGV | OK | OK |
| 12 GiB | ~5857 MiB | OK | SIGSEGV | OK | OK |
| 16 GiB | ~1761 MiB | OK | SIGSEGV | OK | OK |

Boundary is pinned at 164/165 chunks regardless of free memory (down to <2 GiB
free). Also `--parent-extra 8192/12288` (exporter-side alloc) and
`--child-prealloc 1024/8192/14336` (importer-side alloc, child had only 3841 MiB
free) leave the boundary unchanged. → **constant, not a resource limit.**

## 3. Per-allocation, not total (priority 3)

`exp.py --multi` exports N allocations from one pool, imports all in one child:

- `3072+3072` (6 GiB): OK  ·  `4096+4096+4096` (12 GiB): OK
- `5248+1024`: OK  ·  `1024+5248`: OK  ·  `2048+5264`: **SIGSEGV** (the 5264 one)

Importing well past 12 GiB total is fine; only a single allocation of >164 chunks
crashes. Twice-importing the same 5248 blob: **both ImportPointer calls succeed**
(the exit-1 there is a double-`cuMemFreeAsync` bug in my harness, not a crash;
5264-twice still SIGSEGVs on the first). → **strictly per single allocation's
chunk count.**

## 4. Pool state / release threshold / pre-reserve (priority 4)

- `--release-max` (RELEASE_THRESHOLD=UINT64_MAX): no change.
- `--prewarm 8192 --release-max` (pool holds one 8 GiB reserved segment): 5248 OK,
  5264 SIGSEGV. No change.
- `--first 1024` then 5232/5234: both OK (5234 alone-ish still ≤164 chunks).

Subtle confirmation of "chunks, including straddle from misalignment":

| preceding alloc | exported | ptr offset | result |
|---|---|---|---|
| first 32 MiB | 5248 | +32 MiB (aligned) | **OK** (164 chunks) |
| first 64 MiB | 5248 | +64 MiB (aligned) | **OK** (164 chunks) |
| first 16 MiB | 5248 | +16 MiB (unaligned) | **SIGSEGV** (165 chunks) |
| first 48 MiB | 5248 | +48 MiB (unaligned) | **SIGSEGV** (165 chunks) |
| first 2 MiB | 5246 | +2 MiB | OK |
| first 16 MiB | 5232 | +16 MiB | OK; 5234 → SIGSEGV |

An unaligned start makes a 5248 MiB allocation straddle a 165th 32-MiB chunk →
crash; a 32-MiB-aligned start keeps it at 164 → OK. Pure chunk-count behaviour.

## 5. Which call dies, and timing (priority 5)

Child prints confirm `cuMemPoolImportFromShareableHandle` (**OK, ~0.05 ms**) and
`cuMemPoolSetAccess` (**OK, ~0.02 ms**) always succeed; only
`cuMemPoolImportPointer` dies. It does not crash instantly: good imports are
linear at ~1.25 ms/GiB (5248 MiB ≈ 6.6 ms; under strace ~11 ms) — the path walks
the allocation chunk-by-chunk. The 5264 child dies ~6 ms in, inside the
descriptor-building loop, **before** issuing the per-chunk map ioctls: strace
shows a good 5248 import issues ~500 ioctls (all return 0), while the 5264 child
issues only **2** ioctls after the pre-import marker, then SIGSEGVs.

## 6. VMM path at the same sizes — imports FINE (priority 6)

`exp.py --vmm` (cuMemCreate + POSIX-FD cuMemExportToShareableHandle → child
cuMemImportFromShareableHandle + AddressReserve + Map + SetAccess):

| size | mempool IPC | VMM IPC |
|---|---|---|
| 5248 MiB | OK | OK (0.26 ms) |
| 5264 MiB | **SIGSEGV** | **OK (0.27 ms)** |
| 8192 MiB | SIGSEGV | OK (0.39 ms) |
| 16384 MiB | SIGSEGV | OK (0.68 ms) |
| 5264+5264 | — | OK |

The VMM import is O(1) (maps a single handle, no per-chunk descriptor walk) and
has no boundary. **The bug is specific to the stream-ordered mempool import
path.**

## 7. FABRIC / PINNED-via-fabric (priority 7)

`--handle fabric` for both pool and VMM export → `CUDA_ERROR_NOT_PERMITTED (800)`
on this single-GPU host (no NVLink/fabric/IMEX). Not testable here — skipped as
allowed.

## 8. Kernel / limits / maps (priority 8)

- **dmesg at our crash timestamps:** only
  `python3[…]: segfault at d4 ip …libcuda.so.580.126.20…` — **no Xid, no NVRM
  message.** (The `NVRM … Out of memory` lines in the log are stale, ~3 h older
  than our runs, from another session.) Kernel driver reports nothing → userspace
  libcuda fault, matching the bug report.
- `ulimit`: stack 8192 KiB, max locked mem 8192 KiB, virtual mem unlimited,
  open files 524288. `vm.max_map_count = 1048576`, overcommit=0.
- Child `/proc/self` right before ImportPointer: **maps ≈ 136–140**, VmData 56 MB,
  4 threads — three orders of magnitude below max_map_count and unrelated to the
  stack. Even with a 14 GiB child prealloc the map count is unchanged. Not a
  maps/stack/ulimit condition.

## Disassembly — the actual bug site

Faulting function: `libcuda.so.580.126.20` `.text` **0x4a09d0** (three frames
below `cuMemPoolImportPointer` @ 0x34e5f0; callers 0x35ad32, 0x1c5a12). Signature
`f(rdi=obj, rsi, rdx=chunkCount, rcx=handleType, r8=destBuf)`:

```
4a09e3  mov  %rdi,%rbx                 ; rbx = obj (the allocation handle)
4a09dd  mov  %rdx,%r13                 ; r13 = chunk count (=165 when it crashes)
4a0a0e  mov  $0x4,%esi ; call calloc   ; r14 = calloc(count, 4)  temp int[]
4a0a30  loop count× : call 0x3deeb0    ; per-chunk: mutex-guarded bitmap handle
        r14[i] = handle
4a0a88  loop count× :                  ; write count × 0xB0-byte descriptors
        movd (r14[i]),xmm2 ; ... ; movq xmm0,-0x68(%r15) ; r15 += 0xB0
        movb $1,-0xB0(%r15)            ; into caller buffer r8+0x40  <-- OVERRUN
4a0abc  mov  %r14,%rdi ; call free
4a0ac8  mov  0xd4(%rbx),%r12d          ; <-- SIGSEGV: rbx==0, si_addr=0xd4
```

The per-chunk descriptor loop (`0x4a0a88`) writes `count` × 0xB0 bytes into a
buffer provisioned by the caller (0x1c5a12 does a `calloc` sized for the expected
chunk count). At **count = 165** the write runs one 0xB0-byte record past the
164-entry buffer and clobbers the neighbouring object pointer to NULL;
`0x4a0ac8` then dereferences it (`obj->field_0xd4`), faulting at offset 0xd4 off
NULL — exactly the reported `si_addr=0xd4`, `SEGV_MAPERR`.

Confirmed severity gradient with size (`run9`, gdb):
- **5264 / first16+5248** (165 chunks): rbx=0, r13=0xa5=165, faults at
  `0x4a0ac8`, si_addr=0xd4 — the mildest overrun (one record over).
- **6144 / 8192** (192/256 chunks): far larger overrun — **rip=0**, whole
  register file zeroed, dies via `call *0x4f8(%rax)` (a trashed vtable) with
  **rsi=0xc020462a** (an NVIDIA `_IOWR` ioctl request code): it corrupted the
  chunk-map vtable and jumped through NULL while trying to issue the map ioctl.

164 × 32 MiB = 5248 MiB is the last count the 164-entry buffer holds.

## Ranked conclusion

1. **Fixed 164-entry (32-MiB-chunk) descriptor buffer overflow in libcuda's
   `cuMemPoolImportPointer` path.** Boundary = 164 × 32 MiB = 5,502,926,848 B
   exactly; the 165th chunk overruns and NULLs an adjacent pointer that is then
   dereferenced. (Highest confidence: exact byte boundary, r13=165 register,
   disassembly, and the offset/straddle experiments all agree.)
2. **Per single-allocation chunk count**, independent of pool total, other
   allocations, pool segmentation, release threshold, and — via misalignment —
   sensitive to 32-MiB straddle, not raw bytes.
3. **Not a resource limit**: invariant under 16 GiB device pressure, 12 GiB
   exporter pressure, 14 GiB importer pressure; not host OOM; not stack/maps.
4. **Mempool-IPC-specific**: the VMM FD IPC path imports 5264/8192/16384 MiB and
   5264+5264 with no boundary and O(1) cost.
5. Kernel driver is blameless (no Xid/NVRM at crash time; every ioctl that does
   run returns 0). The bug is entirely in closed-source `libcuda.so`.

### Fix / workaround guidance
Cap any single pool-IPC-exported allocation at **≤ 5248 MiB (164 × 32 MiB)** and
32-MiB-align it; split larger buffers into ≤5248-MiB allocations (multi-import is
proven safe), or use the VMM (cuMemCreate/Map) IPC path for large single buffers.
