#!/usr/bin/env python
"""Pinned vs pageable host RAM for H2D / D2H copies. stdlib + torch only.

Measures, on one GPU:
  1. H2D GB/s: pageable (warm), pageable (never-touched zero pages), pinned
     (pin_memory=True), pageable registered in place (cudaHostRegister),
     pinned non_blocking (launch time and completion time).
  2. D2H GB/s: same variants (dest side; no zero-page variant).
  3. Pin cost: pin_memory=True alloc, cudaHostRegister on existing buffer,
     .pin_memory() (alloc+copy), at 512 MiB and 4 GiB.
  4. Break-even: pin cost / (pageable_t - pinned_t) per transfer.
Budget: <= 8 GiB VRAM, <= 12 GiB host RAM at any moment.
"""
import json, statistics, subprocess, sys, time, gc
import torch

MiB = 1 << 20
GiB = 1 << 30
SIZES = [64 * MiB, 256 * MiB, 1 * GiB, 4 * GiB]
PIN_SIZES = [512 * MiB, 4 * GiB]
REPS = 5
dev = torch.device("cuda:0")
cudart = torch.cuda.cudart()


def smi():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    return out


def host_rss_gib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1 << 20)


def drop_caches():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()


def hreg(t):
    ptr, n = t.data_ptr(), t.numel() * t.element_size()
    r = cudart.cudaHostRegister(ptr, n, 0)
    assert r == 0, f"cudaHostRegister failed: {r}"
    assert t.is_pinned(), "is_pinned() not True after cudaHostRegister"


def hunreg(t):
    r = cudart.cudaHostUnregister(t.data_ptr())
    assert r == 0, f"cudaHostUnregister failed: {r}"


def timed(fn):
    """Return seconds for fn() with device syncs around it."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench(fn, reps=REPS):
    fn()  # warm-up
    torch.cuda.synchronize()
    ts = [timed(fn) for _ in range(reps)]
    return statistics.median(ts), min(ts), max(ts)


def gbs(nbytes, secs):
    return nbytes / secs / 1e9


def h2d_variants(n):
    """Return dict variant -> (median_s, min_s, max_s, launch_s or None)."""
    res = {}
    numel = n // 4

    def run_h2d(src, non_blocking=False):
        # fresh device buffer each rep
        def fn():
            d = torch.empty(numel, dtype=torch.float32, device=dev)
            d.copy_(src, non_blocking=non_blocking)
            del d
        return fn

    # (a0) pageable, never touched -> zero pages
    src = torch.empty(numel, dtype=torch.float32, pin_memory=False)
    # a single warm-up copy would touch nothing (reads only), pages stay zero-page mapped
    med, lo, hi = bench(run_h2d(src))
    res["pageable_zero"] = (med, lo, hi, None)
    # (a) pageable, warm (written)
    src.fill_(1.0)
    med, lo, hi = bench(run_h2d(src))
    res["pageable_warm"] = (med, lo, hi, None)
    # (c) same pageable buffer, cudaHostRegister in place
    hreg(src)
    med, lo, hi = bench(run_h2d(src))
    res["registered"] = (med, lo, hi, None)
    hunreg(src)
    assert not src.is_pinned()
    del src
    drop_caches()
    # (b) pinned allocation
    src = torch.empty(numel, dtype=torch.float32, pin_memory=True)
    src.fill_(1.0)
    med, lo, hi = bench(run_h2d(src))
    res["pinned"] = (med, lo, hi, None)
    # (d) pinned non_blocking: measure launch return time and time to completion
    d = torch.empty(numel, dtype=torch.float32, device=dev)
    d.copy_(src, non_blocking=True); torch.cuda.synchronize()
    launches, totals = [], []
    for _ in range(REPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d.copy_(src, non_blocking=True)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        launches.append(t1 - t0); totals.append(t2 - t0)
    res["pinned_nonblocking"] = (statistics.median(totals), min(totals), max(totals),
                                 statistics.median(launches))
    # pageable non_blocking for contrast (should behave like blocking)
    del d, src
    drop_caches()
    src = torch.empty(numel, dtype=torch.float32).fill_(1.0)
    d = torch.empty(numel, dtype=torch.float32, device=dev)
    d.copy_(src, non_blocking=True); torch.cuda.synchronize()
    launches, totals = [], []
    for _ in range(REPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        d.copy_(src, non_blocking=True)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        launches.append(t1 - t0); totals.append(t2 - t0)
    res["pageable_nonblocking"] = (statistics.median(totals), min(totals), max(totals),
                                   statistics.median(launches))
    del d, src
    drop_caches()
    return res


def d2h_variants(n):
    res = {}
    numel = n // 4
    d = torch.empty(numel, dtype=torch.float32, device=dev).fill_(2.0)

    def run_d2h(dst, non_blocking=False):
        def fn():
            dst.copy_(d, non_blocking=non_blocking)
        return fn

    # pageable dest, never touched (first write faults pages in)
    dst = torch.empty(numel, dtype=torch.float32, pin_memory=False)
    t_first = timed(lambda: dst.copy_(d))  # single cold write, pages fault in
    res["pageable_firsttouch"] = (t_first, t_first, t_first, None)
    med, lo, hi = bench(run_d2h(dst))
    res["pageable_warm"] = (med, lo, hi, None)
    hreg(dst)
    med, lo, hi = bench(run_d2h(dst))
    res["registered"] = (med, lo, hi, None)
    hunreg(dst)
    del dst
    drop_caches()
    dst = torch.empty(numel, dtype=torch.float32, pin_memory=True)
    med, lo, hi = bench(run_d2h(dst))
    res["pinned"] = (med, lo, hi, None)
    dst.copy_(d, non_blocking=True); torch.cuda.synchronize()
    launches, totals = [], []
    for _ in range(REPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.copy_(d, non_blocking=True)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        launches.append(t1 - t0); totals.append(t2 - t0)
    res["pinned_nonblocking"] = (statistics.median(totals), min(totals), max(totals),
                                 statistics.median(launches))
    del dst, d
    drop_caches()
    return res


def pin_costs(n):
    numel = n // 4
    res = {}
    # pin_memory=True allocation (fresh: host cache emptied before each)
    ts = []
    for _ in range(3):
        drop_caches()
        t0 = time.perf_counter()
        p = torch.empty(numel, dtype=torch.float32, pin_memory=True)
        ts.append(time.perf_counter() - t0)
        del p
    res["alloc_pinned"] = statistics.median(ts)
    drop_caches()
    # alloc pinned + first-touch fill (what you pay before you can even use it)
    ts = []
    for _ in range(3):
        drop_caches()
        t0 = time.perf_counter()
        p = torch.empty(numel, dtype=torch.float32, pin_memory=True)
        p.fill_(1.0)
        ts.append(time.perf_counter() - t0)
        del p
    res["alloc_pinned_and_fill"] = statistics.median(ts)
    drop_caches()
    # cudaHostRegister on an existing warm pageable buffer (+ unregister)
    src = torch.empty(numel, dtype=torch.float32).fill_(1.0)
    reg, unreg = [], []
    for _ in range(3):
        t0 = time.perf_counter(); hreg(src); reg.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); hunreg(src); unreg.append(time.perf_counter() - t0)
    res["host_register"] = statistics.median(reg)
    res["host_unregister"] = statistics.median(unreg)
    # .pin_memory() = alloc pinned + memcpy
    ts = []
    for _ in range(3):
        drop_caches()
        t0 = time.perf_counter()
        p = src.pin_memory()
        ts.append(time.perf_counter() - t0)
        assert p.is_pinned()
        del p
    res["pin_memory_copy"] = statistics.median(ts)
    # plain host memcpy for reference (pageable->pageable clone)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); c = src.clone(); ts.append(time.perf_counter() - t0); del c
    res["host_clone_ref"] = statistics.median(ts)
    del src
    drop_caches()
    return res


def main():
    torch.cuda.init()
    info = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "smi_before": smi(), "rss_before_gib": host_rss_gib(),
    }
    print(json.dumps(info), flush=True)
    out = {"info": info, "h2d": {}, "d2h": {}, "pin": {}}
    for n in SIZES:
        r = h2d_variants(n)
        out["h2d"][n] = r
        print(f"H2D {n // MiB:5d} MiB: " + "  ".join(
            f"{k}={gbs(n, v[0]):6.2f}GB/s" for k, v in r.items()), flush=True)
        print(f"    smi={smi()} rss={host_rss_gib():.2f}GiB", flush=True)
    for n in SIZES:
        r = d2h_variants(n)
        out["d2h"][n] = r
        print(f"D2H {n // MiB:5d} MiB: " + "  ".join(
            f"{k}={gbs(n, v[0]):6.2f}GB/s" for k, v in r.items()), flush=True)
        print(f"    smi={smi()} rss={host_rss_gib():.2f}GiB", flush=True)
    for n in PIN_SIZES:
        r = pin_costs(n)
        out["pin"][n] = r
        print(f"PIN {n // MiB:5d} MiB: " + "  ".join(
            f"{k}={v * 1e3:8.1f}ms({gbs(n, v):5.2f}GB/s)" for k, v in r.items()), flush=True)
        print(f"    smi={smi()} rss={host_rss_gib():.2f}GiB", flush=True)
    drop_caches()
    out["info"]["smi_after"] = smi()
    out["info"]["rss_after_gib"] = host_rss_gib()
    print("smi_after:", out["info"]["smi_after"], "rss_after:", f"{out['info']['rss_after_gib']:.2f}GiB")
    with open(sys.argv[1] if len(sys.argv) > 1 else "pinned_vs_pageable.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
