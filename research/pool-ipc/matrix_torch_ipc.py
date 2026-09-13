#!/usr/bin/env python3
"""Every CUDA torch on this machine, versus every claim in the PyTorch issue.

For each interpreter that can `import torch` with CUDA available (the comfy-env
envs under ~/.ce/envs, plus any extra pythons given on the command line), run
the real two-process CUDA-IPC handoff three ways and record what happens:

  native -> async     sender on torch's native caching allocator exports a
                      handle with torch.multiprocessing.reductions.reduce_tensor;
                      receiver on backend:cudaMallocAsync rebuilds it.
                      The issue claims this raises "does not yet support
                      getIpcDevPtr".
  async  -> native    sender on cudaMallocAsync exports. The issue claims
                      "does not yet support shareIpcHandle".
  native -> native    the control: must succeed, and the received tensor must
                      hold the sender's bytes, or the harness proves nothing.

and, through the same interpreter but with no torch at all, the driver-level
mempool boundary from repro_mempool_import_segv.py at the two sizes either
side of it (5248 MiB imports, 5264 MiB kills the importer).

Usage:  python3 matrix_torch_ipc.py [extra_python ...]
Writes matrix_results.md next to this file and prints it.

Sender and receiver are separate interpreters started with a clean
PYTORCH_CUDA_ALLOC_CONF, because the allocator backend is chosen when
libc10_cuda loads: it cannot be switched inside a process that already
imported torch, which is exactly why ComfyUI's cuda_malloc.py sets it before
anything else runs.
"""

import glob
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPRO = HERE / "repro_mempool_import_segv.py"

# ---------------------------------------------------------------- child roles

SENDER = r'''
import os, pickle, sys, time
import torch
import torch.multiprocessing.reductions as red
out, done = sys.argv[1], sys.argv[2]
t = torch.arange(1024, dtype=torch.float32, device="cuda")
try:
    fn, args = red.reduce_tensor(t)
    with open(out, "wb") as f:
        pickle.dump({"ok": True, "args": args, "checksum": float(t.sum())}, f)
except Exception as e:
    with open(out, "wb") as f:
        pickle.dump({"ok": False, "error": f"{type(e).__name__}: {e}"}, f)
    sys.exit(0)
# the exporter must outlive the importer's mapping
for _ in range(600):
    if os.path.exists(done):
        break
    time.sleep(0.05)
'''

RECEIVER = r'''
import pickle, sys
import torch
import torch.multiprocessing.reductions as red
src = sys.argv[1]
with open(src, "rb") as f:
    payload = pickle.load(f)
if not payload.get("ok"):
    print(json.dumps({"ok": False, "error": "sender failed: " + payload["error"]}))
    sys.exit(0)
try:
    t = red.rebuild_cuda_tensor(*payload["args"])
    got = float(t.sum())
    print(json.dumps({"ok": got == payload["checksum"], "checksum": got,
                      "expected": payload["checksum"], "backend": torch.cuda.get_allocator_backend()}))
except Exception as e:
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}",
                      "backend": torch.cuda.get_allocator_backend()}))
'''.replace("import pickle, sys", "import json, pickle, sys")

PROBE = r'''
import json, sys
try:
    import torch
    print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
                      "available": torch.cuda.is_available(),
                      "python": "%d.%d" % sys.version_info[:2]}))
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
'''


def _env(backend):
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
    env.pop("PYTORCH_ALLOC_CONF", None)
    if backend == "async":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    return env


def probe(python):
    try:
        out = subprocess.run([python, "-c", PROBE], capture_output=True, text=True,
                             timeout=120, env=_env("native"))
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        return json.loads(line[-1]) if line else {"error": out.stderr.strip()[-200:]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def handoff(python, sender_backend, receiver_backend, attempts=3):
    """Back-to-back imports of a block the driver just unmapped can fail with
    cudaErrorMapBufferObjectFailed (2 of 60 control runs here). That is a
    driver race in the harness, not a torch behaviour, so retry after a pause
    and say how many attempts it took."""
    for attempt in range(1, attempts + 1):
        result = _handoff_once(python, sender_backend, receiver_backend)
        if result.get("ok") or "mapping of buffer object failed" not in result.get("error", ""):
            break
        time.sleep(1.0)
    result["attempt"] = attempt
    return result


def _handoff_once(python, sender_backend, receiver_backend):
    with tempfile.TemporaryDirectory(prefix="ipc-matrix-") as d:
        blob = os.path.join(d, "handle.pkl")
        done = os.path.join(d, "done")
        sender = subprocess.Popen([python, "-c", SENDER, blob, done],
                                  env=_env(sender_backend),
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        deadline = time.time() + 120
        while not os.path.exists(blob) and time.time() < deadline and sender.poll() is None:
            time.sleep(0.05)
        if not os.path.exists(blob):
            sender.kill()
            return {"ok": False, "error": "sender produced nothing: " + (sender.stderr.read() or "")[-300:]}
        try:
            rec = subprocess.run([python, "-c", RECEIVER, blob], capture_output=True, text=True,
                                 timeout=120, env=_env(receiver_backend))
            line = [l for l in rec.stdout.splitlines() if l.startswith("{")]
            result = json.loads(line[-1]) if line else {"ok": False, "error": rec.stderr.strip()[-300:] or f"exit {rec.returncode}"}
        finally:
            open(done, "w").close()
            try:
                sender.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sender.kill()
        return result


def mempool_boundary(python):
    """The driver bug, through this interpreter (stdlib only, no torch)."""
    try:
        out = subprocess.run([python, str(REPRO), "5248", "5264"], capture_output=True,
                             text=True, timeout=300)
        lines = [l.strip() for l in out.stdout.splitlines() if "MiB" in l]
        return " / ".join(lines) if lines else (out.stderr.strip()[-200:] or f"exit {out.returncode}")
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def short(r):
    if r.get("ok"):
        return "OK (bytes match)" + (f", attempt {r['attempt']}" if r.get("attempt", 1) > 1 else "")
    err = r.get("error", "?")
    for needle in ("does not yet support getIpcDevPtr", "does not yet support shareIpcHandle"):
        if needle in err:
            return "RAISES: " + needle
    return "FAIL: " + err[:140]


def main():
    pythons = sorted(glob.glob(os.path.expanduser("~/.ce/envs/*/.pixi/envs/default/bin/python")))
    pythons += sys.argv[1:]
    rows = []
    seen = set()
    for py in pythons:
        info = probe(py)
        if "error" in info or not info.get("available"):
            continue
        key = (info["torch"], info["python"])
        if key in seen:
            continue                         # one interpreter per torch build is enough
        seen.add(key)
        name = Path(py).parts[-6] if "/.ce/envs/" in py else py
        print(f"== {name}: torch {info['torch']} (cuda {info['cuda']}, py {info['python']})", flush=True)
        row = {
            "env": name, "torch": info["torch"], "cuda": info["cuda"], "python": info["python"],
            "native->async": short(handoff(py, "native", "async")),
            "async->native": short(handoff(py, "async", "native")),
            "native->native": short(handoff(py, "native", "native")),
            "mempool 5248/5264 MiB": mempool_boundary(py),
        }
        for k in ("native->async", "async->native", "native->native", "mempool 5248/5264 MiB"):
            print(f"   {k:22s} {row[k]}", flush=True)
        rows.append(row)

    import platform
    drv = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    md = [f"# CUDA IPC across every torch on this machine",
          f"",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `matrix_torch_ipc.py`. GPU/driver: {drv}. Kernel {platform.release()}.",
          f"",
          f"| env | torch | cuda | py | native → async (receiver on cudaMallocAsync) | async → native (sender on cudaMallocAsync) | native → native (control) | mempool import 5248 / 5264 MiB |",
          f"|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['env']} | {r['torch']} | {r['cuda']} | {r['python']} | {r['native->async']} | {r['async->native']} | {r['native->native']} | {r['mempool 5248/5264 MiB']} |")
    text = "\n".join(md) + "\n"
    (HERE / "matrix_results.md").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
