"""Linux port of cuprobe_windows.py.

Does cuMemGetInfo on Linux report the CARD or the CALLING PROCESS?
Only change from the Windows original: libcuda.so.1 via CDLL instead of
nvcuda.dll via WinDLL, plus a VMM allocation mode (that is what
comfy-aimdo actually pages with).

  python cuprobe_linux.py probe
  python cuprobe_linux.py hold  <MiB> <seconds>   legacy cuMemAlloc_v2
  python cuprobe_linux.py holdv <MiB> <seconds>   VMM cuMemCreate + cuMemMap
  python cuprobe_linux.py ctxcost
"""
import ctypes
import subprocess
import sys
import time

cu = ctypes.CDLL("libcuda.so.1")
CUdeviceptr = ctypes.c_ulonglong

cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t]
cu.cuMemGetInfo_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                               ctypes.POINTER(ctypes.c_size_t)]
cu.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                              ctypes.c_uint, ctypes.c_int]


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUmemAllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte),
                ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort),
                ("reserved", ctypes.c_ubyte * 4)]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int),
                ("requestedHandleTypes", ctypes.c_int),
                ("location", CUmemLocation),
                ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", CUmemAllocFlags)]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", ctypes.c_int)]


CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1

cu.cuMemAddressReserve.argtypes = [ctypes.POINTER(CUdeviceptr), ctypes.c_size_t,
                                   ctypes.c_size_t, CUdeviceptr, ctypes.c_ulonglong]
cu.cuMemCreate.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t,
                           ctypes.POINTER(CUmemAllocationProp), ctypes.c_ulonglong]
cu.cuMemMap.argtypes = [CUdeviceptr, ctypes.c_size_t, ctypes.c_size_t,
                        ctypes.c_ulonglong, ctypes.c_ulonglong]
cu.cuMemSetAccess.argtypes = [CUdeviceptr, ctypes.c_size_t,
                              ctypes.POINTER(CUmemAccessDesc), ctypes.c_size_t]
cu.cuMemGetAllocationGranularity.argtypes = [ctypes.POINTER(ctypes.c_size_t),
                                             ctypes.POINTER(CUmemAllocationProp),
                                             ctypes.c_int]


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
    return c, dev.value


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


def vmm_alloc(dev, nbytes):
    prop = CUmemAllocationProp()
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleTypes = 0
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = dev
    gran = ctypes.c_size_t()
    ck(cu.cuMemGetAllocationGranularity(ctypes.byref(gran), ctypes.byref(prop),
                                        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
       "cuMemGetAllocationGranularity")
    g = gran.value
    size = ((nbytes + g - 1) // g) * g
    ptr = CUdeviceptr()
    ck(cu.cuMemAddressReserve(ctypes.byref(ptr), size, g, CUdeviceptr(0), 0),
       "cuMemAddressReserve")
    h = ctypes.c_ulonglong()
    ck(cu.cuMemCreate(ctypes.byref(h), size, ctypes.byref(prop), 0), "cuMemCreate")
    ck(cu.cuMemMap(ptr, size, 0, h, 0), "cuMemMap")
    acc = CUmemAccessDesc()
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
    acc.location.id = dev
    acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    ck(cu.cuMemSetAccess(ptr, size, ctypes.byref(acc), 1), "cuMemSetAccess")
    return ptr, size, g


mode = sys.argv[1]

if mode in ("hold", "holdv"):
    mib, secs = int(sys.argv[2]), float(sys.argv[3])
    _, dev = make_ctx()
    before_free, _ = meminfo()
    sf0, su0 = smi()
    if mode == "hold":
        p = CUdeviceptr()
        ck(cu.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(mib * 1048576)),
           "cuMemAlloc")
        extra = ""
    else:
        p, size, g = vmm_alloc(dev, mib * 1048576)
        extra = " | vmm granularity %d B rounded size %d MiB" % (g, size // 1048576)
    after_free, _ = meminfo()
    sf, su = smi()
    print("HOLD(%s) %d MiB | own cuMemGetInfo free %d -> %d | smi free %d -> %d "
          "used %d -> %d%s"
          % (mode, mib, before_free, after_free, sf0, sf, su0, su, extra), flush=True)
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
    print("PROBE own_free=%d own_total=%d smi_free=%d smi_used=%d gap_own_minus_smi=%d"
          % (free, total, sf, su, free - sf), flush=True)
