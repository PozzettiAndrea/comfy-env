#!/usr/bin/env python3
"""Minimal repro: cuMemPoolImportPointer SIGSEGVs for allocations > 5248 MiB.

Standalone -- Python 3.9+ only, no CUDA toolkit, no pip packages. Talks to
libcuda.so.1 through ctypes.

    python3 repro_mempool_import_segv.py            # full ladder
    python3 repro_mempool_import_segv.py 5264       # one size, in MiB

A parent process creates a memory pool with
handleTypes=CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, allocates a single
buffer from it, exports the pool FD (SCM_RIGHTS) and the pointer, and a
child process imports both. The import succeeds up to 5248 MiB and
segfaults the child inside libcuda from 5264 MiB upward.

Observed: RTX 3090 (GA102, 24576 MiB), driver 580.126.20, Linux 6.8, x86_64.
Fault site: mov 0xd4(%rbx),%r12d with %rbx == NULL, three frames below
cuMemPoolImportPointer.
"""
import ctypes
import json
import os
import socket
import subprocess
import sys
import time

CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
MIB = 1 << 20
LADDER = [1024, 4096, 5120, 5248, 5264, 6144, 8192]


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUmemPoolProps(ctypes.Structure):
    _fields_ = [("allocType", ctypes.c_int), ("handleTypes", ctypes.c_int),
                ("location", CUmemLocation), ("win32SecurityAttributes", ctypes.c_void_p),
                ("reserved", ctypes.c_ubyte * 64)]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


class CUmemPoolPtrExportData(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * 64)]


cu = ctypes.CDLL("libcuda.so.1")
_u64p = ctypes.POINTER(ctypes.c_ulonglong)
_vp = ctypes.c_void_p
for _name, _args in {
    "cuInit": [ctypes.c_uint],
    "cuDeviceGet": [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    "cuDeviceGetName": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
    "cuDriverGetVersion": [ctypes.POINTER(ctypes.c_int)],
    "cuDevicePrimaryCtxRetain": [ctypes.POINTER(_vp), ctypes.c_int],
    "cuCtxSetCurrent": [_vp],
    "cuMemPoolCreate": [ctypes.POINTER(_vp), ctypes.POINTER(CUmemPoolProps)],
    "cuMemPoolDestroy": [_vp],
    "cuMemAllocFromPoolAsync": [_u64p, ctypes.c_size_t, _vp, _vp],
    "cuMemFreeAsync": [ctypes.c_ulonglong, _vp],
    "cuStreamSynchronize": [_vp],
    "cuMemcpyDtoH_v2": [_vp, ctypes.c_ulonglong, ctypes.c_size_t],
    "cuMemPoolExportToShareableHandle": [_vp, _vp, ctypes.c_int, ctypes.c_ulonglong],
    "cuMemPoolImportFromShareableHandle": [ctypes.POINTER(_vp), _vp, ctypes.c_int, ctypes.c_ulonglong],
    "cuMemPoolExportPointer": [ctypes.POINTER(CUmemPoolPtrExportData), ctypes.c_ulonglong],
    "cuMemPoolImportPointer": [_u64p, _vp, ctypes.POINTER(CUmemPoolPtrExportData)],
    "cuMemPoolSetAccess": [_vp, ctypes.POINTER(CUmemAccessDesc), ctypes.c_size_t],
    "cuGetErrorString": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
}.items():
    _fn = getattr(cu, _name)
    _fn.restype = ctypes.c_int
    _fn.argtypes = _args


def check(res, what):
    if res != 0:
        s = ctypes.c_char_p()
        cu.cuGetErrorString(res, ctypes.byref(s))
        raise RuntimeError("%s -> %d (%s)" % (what, res, (s.value or b"?").decode()))


def init_ctx():
    check(cu.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev.value), "cuDevicePrimaryCtxRetain")
    check(cu.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
    return dev.value


def child_main():
    dev = init_ctx()
    sock = socket.socket(fileno=int(sys.argv[2]))
    payload, fds, _flags, _addr = socket.recv_fds(sock, 4096, 1)
    msg = json.loads(payload)

    pool = ctypes.c_void_p()
    check(cu.cuMemPoolImportFromShareableHandle(
        ctypes.byref(pool), ctypes.c_void_p(fds[0]),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0), "cuMemPoolImportFromShareableHandle")

    desc = CUmemAccessDesc()
    desc.location.type, desc.location.id = CU_MEM_LOCATION_TYPE_DEVICE, dev
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    check(cu.cuMemPoolSetAccess(pool, ctypes.byref(desc), 1), "cuMemPoolSetAccess")

    blob = CUmemPoolPtrExportData()
    ctypes.memmove(blob.reserved, bytes.fromhex(msg["blob"]), 64)

    dptr = ctypes.c_ulonglong()
    t0 = time.perf_counter()
    check(cu.cuMemPoolImportPointer(ctypes.byref(dptr), pool, ctypes.byref(blob)),
          "cuMemPoolImportPointer")           # <-- SIGSEGV here above 5248 MiB
    ms = (time.perf_counter() - t0) * 1e3

    probe = (ctypes.c_ubyte * 8)()
    check(cu.cuMemcpyDtoH_v2(probe, dptr.value, 8), "cuMemcpyDtoH")
    check(cu.cuMemFreeAsync(dptr.value, None), "cuMemFreeAsync")   # importer frees first
    check(cu.cuStreamSynchronize(None), "cuStreamSynchronize")
    sock.sendall((json.dumps({"import_ms": round(ms, 2), "byte0": probe[0]}) + "\n").encode())
    return 0


def trial(mib):
    dev = init_ctx()
    props = CUmemPoolProps()
    props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED
    props.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    props.location.type, props.location.id = CU_MEM_LOCATION_TYPE_DEVICE, dev
    pool = ctypes.c_void_p()
    check(cu.cuMemPoolCreate(ctypes.byref(pool), ctypes.byref(props)), "cuMemPoolCreate")

    # NOTE: the pool must be exported to a shareable handle BEFORE any
    # cuMemPoolExportPointer call, otherwise that returns CUDA_ERROR_INVALID_VALUE.
    # This ordering requirement is undocumented.
    fd = ctypes.c_int()
    check(cu.cuMemPoolExportToShareableHandle(
        ctypes.byref(fd), pool, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
        "cuMemPoolExportToShareableHandle")

    dptr = ctypes.c_ulonglong()
    check(cu.cuMemAllocFromPoolAsync(ctypes.byref(dptr), mib * MIB, pool, None),
          "cuMemAllocFromPoolAsync(%d MiB)" % mib)
    check(cu.cuStreamSynchronize(None), "cuStreamSynchronize")

    blob = CUmemPoolPtrExportData()
    check(cu.cuMemPoolExportPointer(ctypes.byref(blob), dptr.value), "cuMemPoolExportPointer")

    parent_sock, child_sock = socket.socketpair()
    parent_sock.settimeout(120)
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--child", str(child_sock.fileno())],
        pass_fds=[child_sock.fileno()])
    child_sock.close()
    socket.send_fds(parent_sock, [json.dumps({"blob": bytes(blob.reserved).hex()}).encode()],
                    [fd.value])

    reply = None
    try:
        line = parent_sock.makefile().readline()
        if line:
            reply = json.loads(line)
    except Exception:
        pass
    rc = proc.wait(timeout=30)

    check(cu.cuMemFreeAsync(dptr.value, None), "cuMemFreeAsync")
    check(cu.cuStreamSynchronize(None), "cuStreamSynchronize")
    cu.cuMemPoolDestroy(pool)
    os.close(fd.value)
    return reply, rc


def main():
    dev = init_ctx()
    name = ctypes.create_string_buffer(128)
    cu.cuDeviceGetName(name, 128, dev)
    ver = ctypes.c_int()
    cu.cuDriverGetVersion(ctypes.byref(ver))
    print("device %s, CUDA driver API %d.%d"
          % (name.value.decode(), ver.value // 1000, ver.value % 1000 // 10))

    sizes = [int(a) for a in sys.argv[1:]] or LADDER
    for mib in sizes:
        try:
            reply, rc = trial(mib)
        except RuntimeError as e:
            print("  %6d MiB  setup failed: %s" % (mib, e))
            continue
        if reply:
            print("  %6d MiB  imported OK in %6.2f ms" % (mib, reply["import_ms"]))
        else:
            print("  %6d MiB  CHILD DIED, exit status %d%s"
                  % (mib, rc, "  (SIGSEGV)" if rc == -11 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(child_main() if "--child" in sys.argv else main())
