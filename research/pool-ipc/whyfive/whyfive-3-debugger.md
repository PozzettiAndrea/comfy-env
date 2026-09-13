# Why cuMemPoolImportPointer segfaults above 5248 MiB — root cause

Machine: RTX 3090, driver 580.126.20 (libcuda.so.580.126.20, text base file
offset 0x166000), Linux 6.8, python3 3.12 (non-PIE). All GPU runs under
`flock gpu.lock`. Working copy of the repro: `scratchpad/repro.py` (adds a
`CHILD_WRAPPER` env hook so the child runs under gdb/strace).

## TL;DR

**Stack-buffer overflow inside libcuda's user-mode wrapper for the RM control
`NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECTS_FROM_FD` (0x3d0c).** The wrapper
(libcuda file offset `0x479c50`) keeps the 652-byte
`NV0000_CTRL_OS_UNIX_IMPORT_OBJECTS_FROM_FD_PARAMS` on its stack and imports
handles in batches of 128 (`numObjects=min(128, count-index)`, `index += 128`
are computed correctly), but the `memcpy` that fills `params.objects[]` uses
the **total** handle count (`count*4` bytes) for every batch instead of the
batch length. The imported allocation is split into 32 MiB chunks, one RM
object per chunk, so:

| chunks (`ceil(MiB/32)`) | size | effect |
|---|---|---|
| <= 128 | <= 4096 MiB | fits, correct |
| 129..164 | 4097..5248 MiB | overflow into `objectTypes[]`, `numObjects/index` (rewritten afterwards) and 12 bytes of frame padding — silent |
| 165 | 5249..5280 MiB | `objects[164]` = **saved rbx** of the wrapper frame → caller's object pointer corrupted → `mov 0xd4(%rbx)` with rbx NULL |
| 166..176 | up to 5632 MiB | saved r12/r13/r14/r15/rbp also clobbered |
| >= 177 | > 5632 MiB | **return address** overwritten → `ret` to 0 (verified at 6144 and 8192 MiB: pc=0, all regs 0) |

164 * 32 MiB = 5248 MiB, exactly the measured last-good size. Every ioctl
returns 0; the kernel is not involved.

## 1. Backtrace (5264 MiB, child under gdb)

    #0  0x00007ffff1ca0ac8 in ?? () from libcuda.so.1     (file off 0x4a0ac8)  "F0"
    #1  0x00007ffff19c5a12 in ?? () from libcuda.so.1     (file off 0x1c5a12)  "F1"
    #2  0x00007ffff1b5ad32 in ?? () from libcuda.so.1
    #3  0x00007ffff1b4e610 in cuMemPoolImportPointer ()
    #4..#6 libffi / _ctypes
    => 0x7ffff1ca0ac8:  mov 0xd4(%rbx),%r12d        rbx=0  rdi=0  r13=0xa5 (=165)
    kernel: segfault at d4 ip ...a0ac8 error 4 (user read, unmapped)   SEGV_MAPERR si_addr=0xd4
    dmesg/journalctl -k: no NVRM / nvidia-uvm / Xid lines at all, only the segfault lines.

Crucial detail from the same kernel log: the other agent's earlier crashes
(PIE `python`) show `segfault at 5c6c000000d4`, `5e57000000d4`, ... — i.e.
rbx = (heap pointer) with its **low 32 bits zeroed**. In non-PIE python3 the
heap pointer is 0x01942320, and zeroing the low dword gives exactly 0. That is
a 4-byte store over the low half of a saved 8-byte register slot.

F0 (0x4a09d0) does `mov %rdi,%rbx` in its prologue and dereferences `rdi`
immediately (`mov (%rdi),%rax`, `cmpb $0,0x8c(%rdi)`) without faulting, and
nothing on the executed path (mode==1: 0x4a0d30 -> 0x4a0d50 -> two RM calls
`call *0x4a8(%r11)` at 0x4a0d99/0x4a0ded -> 0x4a0e17 loop -> free -> `jmp
0x4a0ac8`) writes rbx. So a callee's saved copy of rbx on the stack was
clobbered. gdb with breakpoints on the whole path confirmed: rbx == 0x1942320
at F0 entry, rbx == 0 at the `jmp 0x4a0ac8` (0x4a0e5c) for 5264, still
0x1942320 for 5248.

## 2. strace diff (child, `-f -e trace=ioctl,mmap,munmap,openat,...`)

    5248: 950 ioctls (167x NV_ESC_RM_CONTROL 0x2a, 100x 0x2b, 412 on /dev/nvidia-uvm), 121 mmap, 13 munmap -> imported OK
    5264: 452 ioctls (166x 0x2a,  99x 0x2b,  81 uvm),                             118 mmap, 12 munmap -> SIGSEGV
    Every ioctl on fd 8 (/dev/nvidiactl) and fd 9 (/dev/nvidia-uvm) returns 0 in both traces.
    The only nonzero ioctl returns are Python's TCGETS/ENOTTY on stdio.

Both traces are identical up to and including
`mmap(0x302000000, 50600083456 /*47.1 GiB VA*/, PROT_NONE, ...)` and one
uvm ioctl. Then:

    5264 (tail):  ioctl(8, RM_CONTROL) = 0 ; ioctl(8, RM_CONTROL) = 0 ; --- SIGSEGV si_addr=0xd4
    5248 (same spot): the same two RM_CONTROLs = 0, then ~330 more ioctl(9, UVM 0x21 ...) = 0 (the chunk-by-chunk external mapping), success.

So the last two syscalls before death are two RM controls; decoded with gdb
`catch syscall ioctl` (NVOS54_PARAMETERS at the ioctl arg):

    IOCTL#1 hClient=0xc1d01b6a hObject=0xc1d01b6a cmd=0x3d0c size=652 params=0x7fffffffcb10 (stack)  -> rax=0 status=0
    IOCTL#2 same, cmd=0x3d0c size=652 params=0x7fffffffcb10                                         -> rax=0 status=0

0x3d0c = NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECTS_FROM_FD. paramsSize 652 =
4 (fd) + 4 (hParent) + 4*N (objects) + N (objectTypes) + 2 + 2  ->  N = 128
(`NV0000_CTRL_OS_UNIX_IMPORT_OBJECTS_TO_FD_MAX_OBJECTS`, upstream
ctrl0000unix.h). Decoded params:

    5264 call1: fd=35 hParent=0x5c000002 numObjects=128 index=0   objects[0..127]=0x5c00007f..0x5c0000fe
    5264 call2: fd=35 hParent=0x5c000002 numObjects=37  index=128 objects[0..36]=0x5c0000ff..  objects[38]=0x89941 (heap garbage: over-read, see below)
    5248 call1: numObjects=128 index=0 ; call2: numObjects=36 index=128

Sizes 5248 vs 5264 differ only in the second batch: 36 vs 37 objects (164 vs
165 chunks of 32 MiB). The kernel path is identical and error-free — this
rules out items 3 (unchecked ioctl failure) and 4 (kernel/Xid) of the brief.

## 3. The stack, before and after (gdb `dump binary memory` at the syscall)

Stack snapshot at the entry of IOCTL#1, region around the end of the params
struct (params = 0x7fffffffcb10, ends at 0x7fffffffcd9c):

    5248 (164 chunks)                                5264 (165 chunks)
    cd8c: 5c00011c 5c00011d 5c00011e 00000080        cd8c: 5c00011c 5c00011d 5c00011e 00000080   <- objects[157..159], numObjects|index
    cd9c: 5c000120 5c000121 5c000122 01942320        cd9c: 5c000120 5c000121 5c000122 5c000123   <- objects[161..163] PAST THE STRUCT; then saved rbx low dword
    cdac: 00000000 000000a4 00000000 000000a4        cdac: 00000000 000000a5 00000000 000000a5   <- rbx high, saved r12, r13
    cdbc: 00000000 01957380 00000000 019502b0        cdbc: 00000000 01957430 00000000 019502b0   <- saved r14 (ids array), r15 (entries)
    cdcc: 00000000 ffffce50 00007fff f1ca0da0        (same)                                       <- saved rbp, RETURN ADDRESS 0x7ffff1ca0da0 (F0)

At 5248 the last handle written, objects[163], lands at 0x7fffffffcda4 —
4 bytes short of the saved rbx (0x7fffffffcda8 = 0x01942320 = F0's object).
At 5264 objects[164] = 0x5c000123 overwrites the low dword of saved rbx.
Batch 2 then re-runs the same oversized memcpy from `ids+128`, i.e. reads
past the end of the 660-byte `calloc(165,4)` handle array (heap over-read;
the 0x89941 above is what it picked up) and re-writes the tail with what it
found — zeros here — so the slot becomes 00000000 00000000 and, after the
wrapper's `pop %rbx`, F0 continues with rbx = 0 (or 0x5c6c00000000 in a PIE
process where the high half survives). Hence `mov 0xd4(%rbx)` -> SEGV at 0xd4.

## 4. The bug in the disassembly (libcuda.so.580.126.20, file offsets)

Wrapper 0x479c50 (frame #4 of the ioctl backtrace; called from F0 through
the RM-interface vtable slot `*0x4a8`):

    479c5c: lea  -0x2c0(%rbp),%r13          ; params struct (652 B) at rbp-0x2c0, objects[] at rbp-0x2b8
    479c6c: sub  $0x2b8,%rsp
    479c7f: rep stos (0x51 qwords)          ; zero it
    479cb8: mov  %r8d,%r14d                 ; r8d = total object count (165)
    479cc1: lea  0x0(,%r14,4),%rax
    479ccc: mov  %rax,-0x2d8(%rbp)          ; <<< memcpy length = count*4, computed ONCE from the TOTAL count
    ... per batch:
    479d02: mov  -0x2c8(%rbp),%rdx          ; ids array
    479d14: lea  (%rdx,%rax,4),%rsi         ; src = ids + index
    479d0d: mov  -0x2e0(%rbp),%rdi          ; dst = params.objects
    479d18: mov  -0x2d8(%rbp),%rdx          ; len = count*4   (should be min(128, count-index)*4)
    479d1f: call memcpy@plt
    479d2f: sub  %r13d,%eax ; cmp $0x80 ; cmova   ; numObjects = min(128, count-index)  <- correct
    479d5c: mov  %eax,-0x38(%rbp)                  ; numObjects | index<<16
    479d46: mov  $0x3d0c,%edx ; mov $0x28c,%r8d    ; cmd, paramsSize 652
    479d5f: call 495140                            ; -> NV_ESC_RM_CONTROL ioctl
    479cf0: sub  $0xff80,%r14w                     ; index += 128, loop while index < count

Frame layout: objects[] at rbp-0x2b8, so objects[i] is at rbp-0x2b8+4i and
the callee-saved slots are objects[164..165]=rbx (rbp-0x28), [166..167]=r12,
[168..169]=r13, [170..171]=r14, [172..173]=r15, [174..175]=saved rbp,
[176..177]=return address. `memcpy(dst, src, 165*4)` writes objects[0..164].

Confirmation of the frame model at larger sizes (gdb): 6144 MiB (192 chunks)
and 8192 MiB (256 chunks) both die with pc=0, rbx=r12=..=rbp=0,
rsp=0x7fffffffcde0 (the wrapper's post-`ret` rsp) — the return address was
overwritten (with zeros by batch 2). The kernel log from earlier runs shows the
same signature: `segfault at 0 ip 0000000000000000 ... in python3.12`.

## 5. The 32 MiB granularity / why 5248

F1 (0x1c5700, frame #1) computes `count = ceil(size / chunk)` and allocates
`calloc(count, 0xb0)` entries; F0 allocates one RM handle per chunk from
libcuda's handle bitmap (0x3deeb0; capacity 2^32, 1M in use — not a limit)
and passes all `count` handles to the wrapper. Observed count = 164 for
5248 MiB and 165 for 5264 MiB, so the chunk is exactly 32 MiB
(5248/164 = 32; ceil(5264/32) = 165). 4096 MiB = 128 chunks is the last size
that fits the struct; 4097..5248 MiB already overflow (silently, into
`objectTypes[]`, the `numObjects/index` word — which the wrapper rewrites
after the memcpy — and 12 bytes of alignment padding), 5249+ reach the saved
registers. The bisection boundary at 16 MiB granularity between 5248 and
5264 is therefore precise: the true boundary is 5248 MiB, i.e. 164 chunks.

## Ranked conclusion

1. (certain, directly observed) libcuda user-mode bug: the
   `IMPORT_OBJECTS_FROM_FD` wrapper memcpy's `count*4` bytes into a
   128-entry on-stack `objects[]` for every 128-object batch. Chunk = 32 MiB,
   so >4 GiB overflows, >5248 MiB (165 chunks) corrupts the saved rbx of the
   caller's object pointer -> NULL deref at +0xd4; >5632 MiB corrupts the
   return address -> jump to 0. Also a heap over-read of `ids[]` on batch 2.
   Fix on NVIDIA's side: length = min(128, count-index)*4 (or just
   numObjects*4). Workaround for us: never export a single pool allocation
   larger than 4096 MiB (128 chunks) over POSIX-FD pool IPC; split at 4 GiB.
   Note: 4097..5248 MiB "works" only because the overflowed bytes happen to
   be dead — it is still a stack smash and should be treated as broken.
2. (ruled out) kernel/RM: all ioctls return 0/status 0; no NVRM/UVM/Xid
   messages; the size-dependent syscall sequences are identical up to the
   crash.
3. (ruled out) stack exhaustion, 32-bit truncation, OOM — consistent with the
   writeup; the fault address is a struct offset from a smashed register, not
   a guard page.

Artifacts in scratchpad: gdb_5264.log (bt/regs/maps), st_5248.txt /
st_5264.txt (strace), loop_*.log (per-chunk handle trace), ioctl_*.log +
dumps_{5248,5264}/ (params + stack snapshots), big_{6144,8192}.log,
F0.asm / F1.asm / wrap.asm / alloc_id.asm (objdump extracts), gdb scripts
gdbcmds / gdbloop / gdbioctl / gdbbig.
