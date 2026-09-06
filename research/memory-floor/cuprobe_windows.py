"""Does cuMemGetInfo on WDDM report the CARD or the CALLING PROCESS?

The whole Windows branch in comfy-env rests on the answer. Driver level on
purpose: torch.cuda.mem_get_info is a thin wrapper over this call, and going
straight to nvcuda.dll removes torch's caching allocator as a variable.

  python cuprobe.py hold <MiB> <seconds>   allocate and sit there
  python cuprobe.py probe                  report what THIS process can see
  python cuprobe.py ctxcost                cost of a bare CUDA context
"""
import ctypes
import subprocess
import sys
import time

cu = ctypes.WinDLL("nvcuda.dll")
CUdeviceptr = ctypes.c_ulonglong


def ck(rc, what):
    if rc != 0:
        p = ctypes.c_char_p()
        cu.cuGetErrorName(rc, ctypes.byref(p))
        raise RuntimeError("%s -> %s %s" % (what, rc, p.value))


def make_ctx():
    ck(cu.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    ck(cu.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    c = ctypes.c_void_p()
    ck(cu.cuCtxCreate_v2(ctypes.byref(c), 0, dev), "cuCtxCreate")
    return c


def meminfo():
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    ck(cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
    return free.value // 1048576, total.value // 1048576


def smi():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free,memory.used",
         "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
    f, u = [int(x) for x in out.split(",")]
    return f, u


mode = sys.argv[1]

if mode == "hold":
    mib, secs = int(sys.argv[2]), float(sys.argv[3])
    make_ctx()
    before_free, _ = meminfo()
    p = CUdeviceptr()
    ck(cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(mib * 1048576)),
       "cuMemAlloc")
    after_free, _ = meminfo()
    sf, su = smi()
    print("HOLD %d MiB | own cuMemGetInfo free %d -> %d | smi free %d used %d"
          % (mib, before_free, after_free, sf, su), flush=True)
    time.sleep(secs)

elif mode == "ctxcost":
    sf0, su0 = smi()
    make_ctx()
    time.sleep(1.5)
    sf1, su1 = smi()
    own_free, own_total = meminfo()
    print("CTXCOST smi_used %d -> %d (delta %d MiB) | own_free %d own_total %d"
          % (su0, su1, su1 - su0, own_free, own_total), flush=True)

else:
    make_ctx()
    free, total = meminfo()
    sf, su = smi()
    print("PROBE own_free=%d own_total=%d smi_free=%d smi_used=%d"
          % (free, total, sf, su), flush=True)
