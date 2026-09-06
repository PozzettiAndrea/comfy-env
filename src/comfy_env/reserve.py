"""How much of the card comfy-env asks ComfyUI to leave alone.

The model is deliberately small: comfy-env stops re-deriving ComfyUI's
arithmetic and instead tells ComfyUI the card is smaller by what workers
hold, so ComfyUI's own logic backs off using its own formulas.

Measured constraints this encodes, all from research/memory-floor/p2:

* Publishing works on the LEGACY path only. With a 20.13 GiB reserve a 6 GiB
  model went from fully resident to 1.38 GiB. Under aimdo the same reserve
  changed nothing, because ModelPatcherDynamic ignores lowvram_model_memory
  and pages at fault time. The paged path therefore relies on the reactive
  route (asking the host to free) plus a second write, straight into the
  pager's own headroom.

  This paragraph used to end "and aimdo's own headroom is fixed once its
  devices are initialised". That was wrong, and it was wrong because p2
  exercised plain nn.Linear modules, which never page, so the setter had
  nothing to steer. comfy-aimdo #107 settles it the other way and makes it
  a contract: the setter is documented as taking effect at the next VBAR
  fault, with an upstream test asserting exactly that.
* On a device-wide platform the host ALREADY sees resident worker VRAM,
  because cudaMemGetInfo reports the whole card. Publishing residency again
  double books it, which cost 8.9 GiB of idle card in measurement. So the
  charge is what a worker will take BEYOND what it already holds. On
  Windows WDDM the host cannot see the worker at all and the full
  entitlement is charged.

Pure module: no imports, no I/O, no globals that outlive a call. Everything
here is exercised by bare CI with no torch and no comfy. Host side only:
workers report what they hold, the host decides what to publish, so this is
deliberately NOT in STAGED_WORKER_MODULES.
"""

#: Per-worker VRAM that exists outside any allocator: the CUDA context plus
#: cuBLAS/cuDNN handles. Measured 276 to 300 MiB on Linux/RTX 3090. Booked
#: for every live worker, busy or idle, because it is there either way.
CONTEXT_FLOOR_BYTES = 300 * 1024 * 1024

#: comfy-aimdo's compile time simple headroom (VRAM_HEADROOM in its plat.h):
#: what a host that never passed --reserve-vram runs its pager with. The
#: pager's budget is "this process may use capacity minus headroom", so the
#: reserve reaches the paged path only by being ADDED to this seed.
AIMDO_DEFAULT_HEADROOM = 256 * 1024 * 1024


def charge(residency, process_local_free, floor=CONTEXT_FLOOR_BYTES):
    """What to add to the reserve for one worker.

    Only what the host physically CANNOT SEE. That is the whole rule, and it
    makes the answer a measurement rather than a prediction.

    On a device wide platform (Linux, macOS) ``cudaMemGetInfo`` reports free
    memory for the whole card, so every byte a worker holds is already
    missing from the host's own reading: its models AND its CUDA context.
    There is nothing left to declare, so the charge is zero and the host
    simply sees the truth.

    On Windows WDDM the reading is the calling process's own budget and
    shows nothing of any sibling, so the charge is what the worker holds
    right now plus its context, measured, not forecast.

    There used to be a high water term here, reserving what a worker had
    peaked at on the theory that it would want that much again. It was a
    forecast from one observation, wrong in both directions: a pack that
    spiked once had that space held against it until it went idle, and a
    pack about to need far more got nothing. What replaces it is not a
    better forecast, it is the model proxy: the host can take memory back
    from a worker when it actually needs it, so it does not have to be
    stopped from taking it in advance.
    """
    if not process_local_free:
        return 0
    return int(floor) + max(0, int(residency or 0))


def total_reserve(base, charges, device_total=None):
    """The number to publish: the host's own base plus every worker's charge.

    ``base`` is whatever ComfyUI resolved at startup, usually from
    --reserve-vram. It is preserved rather than overwritten, because it is
    the operator's own instruction and comfy-env is adding to it, not
    replacing it.

    Capped below the device total when known: a reserve at or above the
    whole card makes every host load impossible, turning an over-book into a
    hard stop. The cap leaves a quarter of the card usable.
    """
    total = max(0, int(base or 0)) + sum(max(0, int(c)) for c in charges)
    if device_total:
        total = min(total, int(int(device_total) * 0.75))
    return total


def next_reserve(current, proposed, shrink_allowed):
    """Apply the one safety rule: grow before the memory is taken, shrink
    only once it is provably back.

    A reserve that drops before a worker has actually released hands the
    host space that is still occupied, and the host then loads into it. So
    raising is always allowed and lowering needs evidence, which the caller
    supplies as a measured release receipt rather than an expectation.
    """
    current = max(0, int(current or 0))
    proposed = max(0, int(proposed or 0))
    if proposed >= current:
        return proposed
    return proposed if shrink_allowed else current


def ask_target(weights, slack, min_inference, extra_reserved, want_inference=0):
    """How much to ask the host to free for an incoming worker load.

    Reproduces upstream's own expression rather than inventing one: ComfyUI
    frees ``total_memory_required * 1.1 + extra_mem`` where ``extra_mem`` is
    ``max(inference_memory, memory_required + extra_reserved_memory())``.
    Keeping the same shape is the point; comfy-env booking its own smaller
    number is what made the host free less for a worker load than for an
    identical in-process one.
    """
    weights = max(0, int(weights or 0))
    return int(weights * float(slack)) + max(
        int(min_inference or 0),
        int(want_inference or 0) + int(extra_reserved or 0),
    )


def aimdo_headroom(seed, published, base):
    """The pager headroom that mirrors the published reserve.

    ComfyUI seeds the pager once at startup from --reserve-vram and never
    again, and the pager ignores EXTRA_RESERVED_VRAM entirely, so on the
    paged path the published reserve is inert unless it is forwarded. What
    is forwarded is the host's own seed plus what comfy-env ADDED on top of
    ComfyUI's base: never the base itself, which the seed already carries.

    Catches the wrong implementation of forwarding ``published`` as is,
    which double books the operator's reserve on the paged path.
    """
    if seed is None:
        seed = AIMDO_DEFAULT_HEADROOM
    added = max(0, int(published or 0) - max(0, int(base or 0)))
    return max(0, int(seed)) + added


def reserve_for_requester(published, own_charge):
    """The part of the published reserve that applies to a worker's own load.

    The reserve holds space for every worker's growth, the requester's
    included. When the requester loads, that growth is the load itself, so
    asking the host to keep the requester's own charge free ON TOP of the
    weights being loaded evicts host models for bytes counted twice.

    Catches the wrong implementation of passing ``extra_reserved_memory()``
    straight into the ask.
    """
    return max(0, int(published or 0) - max(0, int(own_charge or 0)))
