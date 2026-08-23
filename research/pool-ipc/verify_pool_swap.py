r"""End-to-end verification of zero-copy transfer under cudaMallocAsync.

Answers: can a worker process read AND write a parent's torch tensors
with no copy, while the parent keeps the `cudaMallocAsync` allocator
backend that ComfyUI enables by default in `cuda_malloc.py`?

Parent (stands in for ComfyUI plus a comfy-env prestartup script):

  1. set PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync, as cuda_malloc.py does
  2. BEFORE importing torch, create a pool with handleTypes=POSIX_FD via the
     driver API and make it current with cuDeviceSetMemPool  <-- the swap
  3. import torch, confirm the backend, allocate a tensor
  4. export the pool to an FD (required before any pointer export)
  5. cuMemPoolExportPointer on the tensor's data_ptr
  6. hand the FD to a torch-free child over SCM_RIGHTS

Child (stands in for an isolated worker):

  imports the pool and the pointer, checksums the tensor (the
  parent->worker read direction), writes a new pattern IN PLACE (the
  direction results travel: the parent preallocates, the worker fills),
  records an interprocess event, frees its import first, acks.

Parent then verifies the worker's write is visible through its own
torch.Tensor, reports the allocator-statistics skew, and frees last.

Requires torch and an NVIDIA GPU. Prints a VERDICT line.
"""
import ctypes, json, os, socket, struct, subprocess, sys

CUDA_SUCCESS = 0
HT_POSIX_FD = 1
ALLOC_PINNED, LOC_DEVICE, PROT_RW = 1, 1, 3
EV_DISABLE_TIMING, EV_INTERPROCESS = 2, 4
PATTERN, PATTERN2 = 0xAB, 0xCD
SIZE = 1 << 20

class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

class CUmemPoolProps(ctypes.Structure):
    _fields_ = [("allocType", ctypes.c_int), ("handleTypes", ctypes.c_int),
                ("location", CUmemLocation), ("win32SecurityAttributes", ctypes.c_void_p),
                ("reserved", ctypes.c_ubyte * 64)]

class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]

class Blob64(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * 64)]  # PtrExportData / CUipcEventHandle

class CU:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        p, up, sz, u64 = ctypes.c_void_p, ctypes.POINTER, ctypes.c_size_t, ctypes.c_ulonglong
        sigs = {
            "cuInit": [ctypes.c_uint],
            "cuDeviceGet": [up(ctypes.c_int), ctypes.c_int],
            "cuDevicePrimaryCtxRetain": [up(p), ctypes.c_int],
            "cuCtxSetCurrent": [p],
            "cuMemPoolCreate": [up(p), up(CUmemPoolProps)],
            "cuMemPoolDestroy": [p],
            "cuDeviceSetMemPool": [ctypes.c_int, p],
            "cuDeviceGetMemPool": [up(p), ctypes.c_int],
            "cuMemFreeAsync": [u64, p],
            "cuMemsetD8Async": [u64, ctypes.c_ubyte, sz, p],
            "cuMemcpyDtoH_v2": [p, u64, sz],
            "cuStreamSynchronize": [p],
            "cuMemPoolExportToShareableHandle": [p, p, ctypes.c_int, u64],
            "cuMemPoolImportFromShareableHandle": [up(p), p, ctypes.c_int, u64],
            "cuMemPoolExportPointer": [up(Blob64), u64],
            "cuMemPoolImportPointer": [up(u64), p, up(Blob64)],
            "cuMemPoolSetAccess": [p, up(CUmemAccessDesc), sz],
            "cuEventCreate": [up(p), ctypes.c_uint],
            "cuEventRecord": [p, p],
            "cuStreamWaitEvent": [p, p, ctypes.c_uint],
            "cuIpcGetEventHandle": [up(Blob64), p],
            "cuIpcOpenEventHandle": [up(p), Blob64],
            "cuGetErrorString": [ctypes.c_int, up(ctypes.c_char_p)],
        }
        for n, a in sigs.items():
            f = getattr(self.lib, n); f.restype = ctypes.c_int; f.argtypes = a
            setattr(self, n, f)

    def check(self, res, what):
        if res != CUDA_SUCCESS:
            s = ctypes.c_char_p(); self.cuGetErrorString(res, ctypes.byref(s))
            raise RuntimeError("%s -> %d (%s)" % (what, res, (s.value or b"?").decode()))

    def init_ctx(self):
        self.check(self.cuInit(0), "cuInit")
        d = ctypes.c_int(); self.check(self.cuDeviceGet(ctypes.byref(d), 0), "cuDeviceGet")
        c = ctypes.c_void_p()
        self.check(self.cuDevicePrimaryCtxRetain(ctypes.byref(c), d.value), "cuDevicePrimaryCtxRetain")
        self.check(self.cuCtxSetCurrent(c), "cuCtxSetCurrent")
        return d.value


def child_main():
    cu = CU(); dev = cu.init_ctx()
    sock = socket.socket(fileno=int(sys.argv[2]))
    msg, fds, *_ = socket.recv_fds(sock, 4096, 1)
    m = json.loads(msg)
    out = {"errors": []}

    pool = ctypes.c_void_p()
    cu.check(cu.cuMemPoolImportFromShareableHandle(
        ctypes.byref(pool), ctypes.c_void_p(fds[0]), HT_POSIX_FD, 0), "ImportFromShareableHandle")
    desc = CUmemAccessDesc(); desc.location.type, desc.location.id = LOC_DEVICE, dev
    desc.flags = PROT_RW
    r = cu.cuMemPoolSetAccess(pool, ctypes.byref(desc), 1)
    if r != CUDA_SUCCESS: out["errors"].append("SetAccess->%d" % r)

    blob = Blob64(); ctypes.memmove(blob.reserved, bytes.fromhex(m["blob"]), 64)
    dptr = ctypes.c_ulonglong()
    cu.check(cu.cuMemPoolImportPointer(ctypes.byref(dptr), pool, ctypes.byref(blob)), "ImportPointer")

    host = (ctypes.c_ubyte * m["size"])()
    cu.check(cu.cuMemcpyDtoH_v2(host, dptr.value, m["size"]), "DtoH")
    out["read_checksum_ok"] = all(b == m["pattern"] for b in host)

    # result-inversion direction: write in place into the parent's tensor
    cu.check(cu.cuMemsetD8Async(dptr.value, PATTERN2, m["size"], None), "MemsetD8Async")

    # interprocess event: record write-done (legacy-IPC family, allocator-orthogonal?)
    try:
        ev_h = Blob64(); ctypes.memmove(ev_h.reserved, bytes.fromhex(m["event"]), 64)
        ev = ctypes.c_void_p()
        cu.check(cu.cuIpcOpenEventHandle(ctypes.byref(ev), ev_h), "IpcOpenEventHandle")
        cu.check(cu.cuEventRecord(ev, None), "EventRecord")
        out["event_ok"] = True
    except RuntimeError as e:
        out["event_ok"] = False; out["errors"].append(str(e))
        cu.check(cu.cuStreamSynchronize(None), "sync-fallback")

    # importer frees FIRST (mandated order); parent still holds the tensor ref
    cu.check(cu.cuMemFreeAsync(dptr.value, None), "FreeAsync(import)")
    cu.check(cu.cuStreamSynchronize(None), "sync")
    sock.sendall((json.dumps(out) + "\n").encode())
    return 0


def parent_main():
    results = {}
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    cu = CU(); dev = cu.init_ctx()

    # --- the swap, pre-torch (mimics prestartup script) ---
    props = CUmemPoolProps()
    props.allocType, props.handleTypes = ALLOC_PINNED, HT_POSIX_FD
    props.location.type, props.location.id = LOC_DEVICE, dev
    pool = ctypes.c_void_p()
    cu.check(cu.cuMemPoolCreate(ctypes.byref(pool), ctypes.byref(props)), "cuMemPoolCreate")
    cu.check(cu.cuDeviceSetMemPool(dev, pool), "cuDeviceSetMemPool")
    print("[1] pool created (handleTypes=POSIX_FD) and swapped in, pre-torch: OK")

    import torch
    assert torch.cuda.get_allocator_backend() == "cudaMallocAsync", torch.cuda.get_allocator_backend()
    print("[2] torch %s imported over the swap, backend=%s: OK"
          % (torch.__version__, torch.cuda.get_allocator_backend()))

    t = torch.full((SIZE,), PATTERN, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    results["swap_survives_torch"] = True

    # is the current pool still ours after torch init?
    cur = ctypes.c_void_p()
    cu.check(cu.cuDeviceGetMemPool(ctypes.byref(cur), dev), "cuDeviceGetMemPool")
    results["current_pool_is_ours"] = (cur.value == pool.value)
    print("[3] current device pool after torch alloc: %s (ours=%s)"
          % (hex(cur.value or 0), results["current_pool_is_ours"]))

    # pool FD export MUST precede pointer export (measured precondition)
    fd_out = ctypes.c_int()
    cu.check(cu.cuMemPoolExportToShareableHandle(
        ctypes.byref(fd_out), pool, HT_POSIX_FD, 0), "ExportToShareableHandle")
    print("[4] pool exported to FD %d (before any pointer export): OK" % fd_out.value)

    # THE blocker call
    blob = Blob64()
    try:
        cu.check(cu.cuMemPoolExportPointer(ctypes.byref(blob), t.data_ptr()),
                 "cuMemPoolExportPointer(torch tensor)")
        results["export_pointer_on_torch_alloc"] = True
        print("[5] cuMemPoolExportPointer on a torch-allocated tensor: OK  <-- THE blocker")
    except RuntimeError as e:
        results["export_pointer_on_torch_alloc"] = False
        print("[5] FAILED: %s" % e); return report(results)

    ev = ctypes.c_void_p()
    ev_hex = None
    try:
        cu.check(cu.cuEventCreate(ctypes.byref(ev), EV_INTERPROCESS | EV_DISABLE_TIMING), "EventCreate")
        ev_h = Blob64()
        cu.check(cu.cuIpcGetEventHandle(ctypes.byref(ev_h), ev), "IpcGetEventHandle")
        ev_hex = bytes(ev_h.reserved).hex()
        print("[6] interprocess event created+exported under async backend: OK")
        results["ipc_event_export"] = True
    except RuntimeError as e:
        results["ipc_event_export"] = False; print("[6] event export FAILED: %s" % e)

    ps, cs = socket.socketpair()
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child", str(cs.fileno())],
                            pass_fds=[cs.fileno()])
    socket.send_fds(ps, [json.dumps({"blob": bytes(blob.reserved).hex(), "size": SIZE,
                                     "pattern": PATTERN, "event": ev_hex}).encode()],
                    [fd_out.value])
    reply = json.loads(ps.makefile().readline())
    proc.wait(timeout=90)
    results["child"] = reply
    print("[7] child: %s" % reply)

    if reply.get("event_ok"):
        cu.check(cu.cuStreamWaitEvent(None, ev, 0), "StreamWaitEvent")
    torch.cuda.synchronize()
    results["inplace_write_visible"] = bool((t == PATTERN2).all().item())
    print("[8] worker's in-place write visible through parent torch tensor: %s"
          % results["inplace_write_visible"])

    free_b, total_b = torch.cuda.mem_get_info()
    results["stats"] = {
        "memory_allocated": torch.cuda.memory_allocated(),
        "memory_reserved": torch.cuda.memory_reserved(),
        "mem_get_info_used_MiB": round((total_b - free_b) / 2**20),
    }
    try:
        torch.cuda.empty_cache(); results["stats"]["empty_cache"] = "no-crash"
    except Exception as e:
        results["stats"]["empty_cache"] = "raised: %s" % e
    print("[9] stats skew: %s" % results["stats"])

    del t; torch.cuda.synchronize()   # exporter frees AFTER importer freed
    print("[10] exporter-side free after importer free: OK")
    return report(results)


def report(results):
    print("\n=== RESULTS ===\n" + json.dumps(results, indent=2))
    ok = (results.get("export_pointer_on_torch_alloc")
          and results.get("child", {}).get("read_checksum_ok")
          and results.get("inplace_write_visible"))
    print("\nVERDICT: %s" % ("zero-copy VERIFIED end-to-end" if ok else "NOT verified — see failures"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(child_main() if "--child" in sys.argv else parent_main())
