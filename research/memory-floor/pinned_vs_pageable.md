# Pinned vs pageable host RAM for H2D / D2H copies — measured

Date 2026-09-15. Script: `pinned_vs_pageable.py` (stdlib + torch), raw data
`pinned_vs_pageable.json`. Run as the only GPU job.

Machine: RTX 3090 24 GiB, driver 580.126.20, torch 2.8.0+cu128 / CUDA 12.8,
Ryzen 5 5600 (6c/12t, 1 NUMA node), 31 GiB DDR4. The pinned ceiling of
~25 GB/s is a PCIe 4.0 x16 link running near its practical peak.

Method: `d.copy_(src)` / `dst.copy_(d)` with `torch.cuda.synchronize()` on both
sides of `perf_counter`, 1 warm-up then median of 5, fresh device tensor per rep
for H2D. `registered` = the *same* pageable buffer after
`cudart.cudaHostRegister(ptr, nbytes, 0)` (verified `is_pinned()` flips True,
unregistered afterwards). All host/device buffers freed and both caching
allocators emptied between sizes. Peak usage: ~4.5 GiB VRAM, ~8.6 GiB host.

nvidia-smi memory.used: before 174 MiB, during run 436-478 MiB (the CUDA
context of the live process), after process exit 172 MiB. Host RSS ended at
0.60 GiB. Cleanup confirmed.

## 1. Host to device, GB/s (median of 5)

| size | pageable, never touched | pageable, warm | registered in place | pinned alloc | pinned non_blocking | pageable non_blocking |
|---|---|---|---|---|---|---|
| 64 MiB | 18.0 | 12.2 | 24.6 | 24.0 | 24.7 | 13.4 |
| 256 MiB | 18.2 | 12.6 | 26.0 | 24.6 | 24.9 | 12.6 |
| 1 GiB | 18.2 | 12.3 | 24.6 | 25.1 | 25.2 | 12.4 |
| 4 GiB | 18.2 | 12.2 | 25.2 | 25.2 | 25.2 | 12.3 |

Wall time at 4 GiB: pageable warm 351 ms, pinned 171 ms, registered 171 ms.
Min/max spread within each cell is under 3%.

`non_blocking=True` launch-return time: pinned 5-20 us (truly async, the copy
completes 171 ms later at the sync); pageable 5 / 21 / 87 / 348 ms, i.e. the
call blocks for the whole transfer. `non_blocking` on a pageable source is a
no-op for asynchrony.

## 2. Device to host, GB/s (median of 5)

| size | pageable, first touch (single cold write) | pageable, warm | registered in place | pinned alloc | pinned non_blocking |
|---|---|---|---|---|---|
| 64 MiB | 1.9 | 10.3 | 26.2 | 26.2 | 26.2 |
| 256 MiB | 2.0 | 11.9 | 26.2 | 25.3 | 25.3 |
| 1 GiB | 2.0 | 11.9 | 24.8 | 25.2 | 25.1 |
| 4 GiB | 2.0 | 11.7 | 25.0 | 25.1 | 25.1 |

Wall time at 4 GiB: pageable warm 366 ms, pinned 171 ms, first-touch pageable
2140 ms.

## 3. Cost of pinning (median of 3, host caching allocator emptied before each)

| size | `torch.empty(pin_memory=True)` | same + `fill_` | `cudaHostRegister` on warm buffer | `cudaHostRegister` on untouched buffer | `cudaHostUnregister` | `.pin_memory()` (alloc + copy) | plain host `clone()` (reference) |
|---|---|---|---|---|---|---|---|
| 512 MiB | 267 ms (2.0 GB/s) | 295 ms | 28-38 ms (14-19 GB/s); first call in process 189 ms | n/m | 11 ms | 313 ms (1.7 GB/s) | 55 ms (9.8 GB/s) |
| 4 GiB | 2065 ms (2.1 GB/s) | 2332 ms | 230-280 ms (15-19 GB/s) | 1379 ms (3.1 GB/s) | 94-100 ms | 2547 ms (1.7 GB/s) | 449 ms (9.6 GB/s) |

Reading: `cudaHostAlloc` runs at ~2 GB/s and is dominated by the driver
faulting in and zeroing the pages, not by the pinning itself. Registering
pages that are already resident costs ~15-19 GB/s equivalent (~0.06 s/GiB);
registering pages that are not resident falls back to the ~3 GB/s regime
because the driver has to populate them first. `.pin_memory()` is
`cudaHostAlloc` plus a ~10 GB/s memcpy and is the most expensive route.

## 4. Break-even (pin cost / per-transfer saving vs warm pageable)

Per-transfer saving (pageable warm minus pinned): H2D 44 ms/GiB, D2H 48 ms/GiB.

| buffer | route | pin cost | saving / transfer | transfers to pay back |
|---|---|---|---|---|
| 512 MiB | `pin_memory=True` alloc | 267 ms | 22-24 ms | 11-12 |
| 512 MiB | alloc + fill | 295 ms | 22-24 ms | 12-13 |
| 512 MiB | `cudaHostRegister` (warm pages) | 28 ms | 22-24 ms | 1.2-1.3 |
| 512 MiB | `.pin_memory()` copy | 313 ms | 22-24 ms | 13-14 |
| 4 GiB | `pin_memory=True` alloc | 2065 ms | 180-195 ms | 11-12 |
| 4 GiB | alloc + fill | 2332 ms | 180-195 ms | 12-13 |
| 4 GiB | `cudaHostRegister` (warm pages) | 230 ms | 180-195 ms | 1.2-1.3 |
| 4 GiB | `cudaHostRegister` (untouched pages) | 1379 ms | 180-195 ms | 7-8 |
| 4 GiB | `.pin_memory()` copy | 2547 ms | 180-195 ms | 13-14 |

The ratio is size-independent from 512 MiB up: allocating pinned pays back
after ~12 transfers, registering an already-resident buffer after ~1.3, i.e.
on the second transfer. A buffer that is transferred exactly once is cheapest
left pageable unless it is already resident and can be registered in place.

## 5. Pageable source state (warm vs zero pages)

A pageable source that was allocated but never written copies at 18.2 GB/s
instead of 12.2 GB/s. Every untouched 4 KiB maps to the kernel's shared zero
page, so the driver's staging memcpy reads one cache-hot page; this is a
measurement artefact, not a usable speed, and real model weights are always
warm. For D2H the analogous cold case is the opposite: a never-written
destination pays the page faults inside the copy and runs at 2.0 GB/s
(2.1 s for 4 GiB), 6x slower than warm pageable and 12x slower than pinned.

## 6. What the docs should say

1. "Pinned costs the RAM-to-VRAM transfer and nothing else" is fair for the
   transfer itself: pinned and in-place-registered buffers both hit 25 GB/s
   (the PCIe 4.0 x16 ceiling) in both directions at every size from 64 MiB up,
   and `non_blocking` returns in microseconds.
2. It is not fair for the whole lifecycle: obtaining pinned memory costs
   2 GB/s via `cudaHostAlloc` (2.1 s for 4 GiB) or 1.7 GB/s via
   `.pin_memory()`, so a pinned buffer must be reused ~12 times to break even;
   only `cudaHostRegister` on already-resident pages (~0.06 s/GiB) is nearly
   free, breaking even on the second transfer.
3. The pageable path measured costs exactly 2x the pinned time (12.2 vs
   25.2 GB/s H2D, 11.7 vs 25.1 GB/s D2H): the driver's staged copy through its
   own pinned bounce buffer is bounded by a ~12 GB/s host memcpy, not by the
   link, and it also blocks the calling thread for the full duration even with
   `non_blocking=True`.
4. The relative difference never stops mattering: it is a constant 2x from
   64 MiB to 4 GiB. What stops mattering is the absolute cost at small sizes
   (64 MiB: 5.5 ms vs 2.8 ms).
5. For a load-once model of N GiB, pageable H2D costs ~87 ms/GiB and pinned
   ~43 ms/GiB, but preparing a fresh pinned buffer first costs ~500 ms/GiB, so
   pin only buffers that will be moved repeatedly (offload/reload cycles), and
   prefer registering resident pages over allocating pinned.
6. A cold pageable *destination* is the real trap: D2H into never-written RAM
   runs at 2 GB/s; VRAM-to-host offload targets should be pre-touched or
   pinned.
