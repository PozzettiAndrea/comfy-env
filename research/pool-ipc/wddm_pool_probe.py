r"""wddm_pool_probe.py -- can a WDDM GeForce export a shareable CUDA mempool?

Probes the question comfy-env's Windows zero-copy path hinges on:

  stage 1  device attributes: WDDM vs TCC, mempool support, and whether
           the mempool supported-handle-types mask carries the WIN32 bit
  stage 2  create a pool with handleTypes=WIN32, swap it in with
           cuDeviceSetMemPool, allocate with plain cuMemAllocAsync
           (mirroring torch's cudaMallocAsync backend, which allocates
           from the *current* pool), export the pool handle + pointer
  stage 3  child process: DuplicateHandle'd pool handle -> import pool,
           set access, import pointer, read + checksum, and free its
           import BEFORE the exporter frees (the mandated order)

Run on Windows: 64-bit Python 3.8+, NVIDIA driver >= 466 (mid-2021).
No CUDA toolkit, no pip packages -- talks to nvcuda.dll directly.

    python wddm_pool_probe.py
"""

import ctypes
import json
import os
import subprocess
import sys

CUDA_SUCCESS = 0
ATTR_TCC_DRIVER = 35
ATTR_VMM_SUPPORTED = 102
ATTR_HANDLE_TYPE_WIN32_SUPPORTED = 104          # VMM (cuMemCreate) path
ATTR_MEMORY_POOLS_SUPPORTED = 115
ATTR_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119       # needs driver >= 466 (CUDA 11.3)
HANDLE_TYPE_WIN32 = 2
ALLOC_TYPE_PINNED = 1
LOCATION_TYPE_DEVICE = 1
ACCESS_PROT_READWRITE = 3

PATTERN = 0xAB
SIZE = 1 << 20  # 1 MiB


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUmemPoolProps(ctypes.Structure):
    # 11.x layout; 12.2's maxSize and 12.6's usage were carved out of
    # reserved[], so total size (88) and field offsets are unchanged.
    _fields_ = [
        ("allocType", ctypes.c_int),
        ("handleTypes", ctypes.c_int),
        ("location", CUmemLocation),
        ("win32SecurityAttributes", ctypes.c_void_p),
        ("reserved", ctypes.c_ubyte * 64),
    ]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


class CUmemPoolPtrExportData(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * 64)]


class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", ctypes.c_ulong),
        ("lpSecurityDescriptor", ctypes.c_void_p),  # NULL = default owner ACL
        ("bInheritHandle", ctypes.c_int),
    ]


class Cuda:
    def __init__(self):
        self.lib = ctypes.WinDLL("nvcuda.dll")
        p, up, sz = ctypes.c_void_p, ctypes.POINTER, ctypes.c_size_t
        u64 = ctypes.c_ulonglong
        sigs = {
            "cuInit": [ctypes.c_uint],
            "cuDriverGetVersion": [up(ctypes.c_int)],
            "cuDeviceGet": [up(ctypes.c_int), ctypes.c_int],
            "cuDeviceGetAttribute": [up(ctypes.c_int), ctypes.c_int, ctypes.c_int],
            "cuDeviceGetName": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
            "cuDevicePrimaryCtxRetain": [up(p), ctypes.c_int],
            "cuCtxSetCurrent": [p],
            "cuMemPoolCreate": [up(p), up(CUmemPoolProps)],
            "cuMemPoolDestroy": [p],
            "cuDeviceSetMemPool": [ctypes.c_int, p],
            "cuMemAllocAsync": [up(u64), sz, p],
            "cuMemFreeAsync": [u64, p],
            "cuMemsetD8Async": [u64, ctypes.c_ubyte, sz, p],
            "cuMemcpyDtoH_v2": [p, u64, sz],
            "cuStreamSynchronize": [p],
            "cuMemPoolExportToShareableHandle": [p, p, ctypes.c_int, u64],
            "cuMemPoolImportFromShareableHandle": [up(p), p, ctypes.c_int, u64],
            "cuMemPoolExportPointer": [up(CUmemPoolPtrExportData), u64],
            "cuMemPoolImportPointer": [up(u64), p, up(CUmemPoolPtrExportData)],
            "cuMemPoolSetAccess": [p, up(CUmemAccessDesc), sz],
            "cuGetErrorString": [ctypes.c_int, up(ctypes.c_char_p)],
        }
        for name, argtypes in sigs.items():
            fn = getattr(self.lib, name)
            fn.restype = ctypes.c_int
            fn.argtypes = argtypes
            setattr(self, name, fn)

    def check(self, res, what):
        if res != CUDA_SUCCESS:
            s = ctypes.c_char_p()
            self.cuGetErrorString(res, ctypes.byref(s))
            raise RuntimeError("%s -> %d (%s)" % (what, res, (s.value or b"?").decode()))

    def init_ctx(self, dev_ordinal=0):
        self.check(self.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        self.check(self.cuDeviceGet(ctypes.byref(dev), dev_ordinal), "cuDeviceGet")
        ctx = ctypes.c_void_p()
        self.check(self.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev.value),
                   "cuDevicePrimaryCtxRetain")
        self.check(self.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")
        return dev.value

    def attr(self, attr_id, dev):
        v = ctypes.c_int()
        res = self.cuDeviceGetAttribute(ctypes.byref(v), attr_id, dev)
        return v.value if res == CUDA_SUCCESS else None


def child_main():
    cu = Cuda()
    dev = cu.init_ctx()
    msg = json.loads(sys.stdin.readline())

    pool = ctypes.c_void_p()
    cu.check(cu.cuMemPoolImportFromShareableHandle(
        ctypes.byref(pool), ctypes.c_void_p(msg["handle"]), HANDLE_TYPE_WIN32, 0),
        "cuMemPoolImportFromShareableHandle")

    desc = CUmemAccessDesc()
    desc.location.type, desc.location.id = LOCATION_TYPE_DEVICE, dev
    desc.flags = ACCESS_PROT_READWRITE
    res = cu.cuMemPoolSetAccess(pool, ctypes.byref(desc), 1)
    if res != CUDA_SUCCESS:
        print("CHILD cuMemPoolSetAccess -> %d (continuing)" % res, flush=True)

    blob = CUmemPoolPtrExportData()
    ctypes.memmove(blob.reserved, bytes.fromhex(msg["blob"]), 64)
    dptr = ctypes.c_ulonglong()
    cu.check(cu.cuMemPoolImportPointer(ctypes.byref(dptr), pool, ctypes.byref(blob)),
             "cuMemPoolImportPointer")

    host = (ctypes.c_ubyte * msg["size"])()
    cu.check(cu.cuMemcpyDtoH_v2(host, dptr.value, msg["size"]), "cuMemcpyDtoH")
    ok = all(b == msg["pattern"] for b in host)

    # importer frees BEFORE exporter -- the mandated teardown order
    cu.check(cu.cuMemFreeAsync(dptr.value, None), "cuMemFreeAsync(import)")
    cu.check(cu.cuStreamSynchronize(None), "cuStreamSynchronize")
    print("CHILD-RESULT " + json.dumps({"checksum_ok": ok}), flush=True)
    return 0 if ok else 1


def parent_main():
    results = {}
    cu = Cuda()
    dev = cu.init_ctx()

    ver = ctypes.c_int()
    cu.cuDriverGetVersion(ctypes.byref(ver))
    name = ctypes.create_string_buffer(128)
    cu.cuDeviceGetName(name, 128, dev)
    tcc = cu.attr(ATTR_TCC_DRIVER, dev)
    pools = cu.attr(ATTR_MEMORY_POOLS_SUPPORTED, dev)
    mask = cu.attr(ATTR_MEMPOOL_SUPPORTED_HANDLE_TYPES, dev)
    vmm_w32 = cu.attr(ATTR_HANDLE_TYPE_WIN32_SUPPORTED, dev)

    print("=== stage 1: attributes ===")
    print("device            : %s" % name.value.decode())
    print("driver CUDA ver   : %d.%d" % (ver.value // 1000, ver.value % 1000 // 10))
    print("driver model      : %s" % ("TCC" if tcc else "WDDM"))
    print("mempools supported: %s" % pools)
    print("pool handle mask  : %s  (WIN32 bit: %s)"
          % (mask, "YES" if mask is not None and mask & HANDLE_TYPE_WIN32 else "no"))
    print("VMM WIN32 support : %s  (fallback path)" % vmm_w32)
    results["wddm"] = not tcc
    results["pool_win32_bit"] = bool(mask is not None and mask & HANDLE_TYPE_WIN32)

    print("\n=== stage 2: shareable pool + swap + async alloc + export ===")
    sa = SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES), None, 0)
    props = CUmemPoolProps()
    props.allocType = ALLOC_TYPE_PINNED
    props.handleTypes = HANDLE_TYPE_WIN32
    props.location.type, props.location.id = LOCATION_TYPE_DEVICE, dev
    props.win32SecurityAttributes = ctypes.cast(ctypes.byref(sa), ctypes.c_void_p)

    pool = ctypes.c_void_p()
    dptr = ctypes.c_ulonglong()
    blob = CUmemPoolPtrExportData()
    handle = ctypes.c_void_p()
    try:
        cu.check(cu.cuMemPoolCreate(ctypes.byref(pool), ctypes.byref(props)),
                 "cuMemPoolCreate(handleTypes=WIN32)")
        results["pool_create"] = True
        cu.check(cu.cuDeviceSetMemPool(dev, pool), "cuDeviceSetMemPool")
        results["pool_swap"] = True
        # plain cuMemAllocAsync == what torch's cudaMallocAsync backend does;
        # it must draw from the swapped-in current pool for export to work
        cu.check(cu.cuMemAllocAsync(ctypes.byref(dptr), SIZE, None), "cuMemAllocAsync")
        cu.check(cu.cuMemsetD8Async(dptr.value, PATTERN, SIZE, None), "cuMemsetD8Async")
        cu.check(cu.cuStreamSynchronize(None), "cuStreamSynchronize")
        cu.check(cu.cuMemPoolExportToShareableHandle(
            ctypes.byref(handle), pool, HANDLE_TYPE_WIN32, 0),
            "cuMemPoolExportToShareableHandle")
        results["pool_handle_export"] = True
        cu.check(cu.cuMemPoolExportPointer(ctypes.byref(blob), dptr.value),
                 "cuMemPoolExportPointer(async-alloc'd ptr)")
        results["pointer_export"] = True
        print("all stage-2 calls: OK")
    except RuntimeError as e:
        print("FAILED: %s" % e)
        results.setdefault("stage2_error", str(e))
        return report(results)

    print("\n=== stage 3: cross-process import (DuplicateHandle) ===")
    try:
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        k32 = ctypes.windll.kernel32
        dup = ctypes.c_void_p()
        if not k32.DuplicateHandle(k32.GetCurrentProcess(), handle,
                                   ctypes.c_void_p(proc._handle), ctypes.byref(dup),
                                   0, False, 2):  # DUPLICATE_SAME_ACCESS
            raise RuntimeError("DuplicateHandle failed: winerror %d" % k32.GetLastError())
        proc.stdin.write(json.dumps({
            "handle": dup.value, "blob": bytes(blob.reserved).hex(),
            "size": SIZE, "pattern": PATTERN}) + "\n")
        proc.stdin.flush()
        out, _ = proc.communicate(timeout=90)
        print(out.strip())
        for line in out.splitlines():
            if line.startswith("CHILD-RESULT "):
                results["cross_process"] = json.loads(line[13:])["checksum_ok"]
        # exporter frees only after the importer freed (order verified above)
        cu.check(cu.cuMemFreeAsync(dptr.value, None), "cuMemFreeAsync(export)")
        cu.check(cu.cuStreamSynchronize(None), "cuStreamSynchronize")
        cu.cuMemPoolDestroy(pool)
    except Exception as e:
        print("FAILED: %s" % e)
        results.setdefault("stage3_error", str(e))
    return report(results)


def report(results):
    print("\n=== verdict ===")
    print(json.dumps(results, indent=2))
    if results.get("cross_process"):
        print("\nWIN32 mempool export VERIFIED on this WDDM device:")
        print("the symmetric pool-swap architecture is viable on Windows.")
        return 0
    if not results.get("pool_win32_bit"):
        print("\nWIN32 bit absent from the pool handle mask:")
        print("pool export unsupported here -> VMM worker-arena fallback path.")
    else:
        print("\nWIN32 bit advertised but the canary failed at a later stage:")
        print("see the first FAILED line above -- that call is the blocker.")
    return 1


if __name__ == "__main__":
    if os.name != "nt":
        sys.exit("This probe must run on Windows (WDDM is the question).")
    sys.exit(child_main() if "--child" in sys.argv else parent_main())
