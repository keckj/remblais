"""Benchmark of remblais against GeomLoss (see docs/performance.md).

Debiased Sinkhorn divergence, p = 2, blur = 0.05, scaling = 0.5, FP32, N = M,
forward and backward passes (gradient with respect to x), on a GPU or on the
CPU (GeomLoss then runs on the CPU backend of PyKeOps). Each measurement
runs in a fresh process with a fresh PyKeOps build folder, so that the PyKeOps
kernel being measured is the first one compiled: GeomLoss's best case. The
table gives the times and the ratios GeomLoss / remblais (above 1: remblais is
faster).

    python tools/bench_geomloss.py                       # every GPU case of the docs
    python tools/bench_geomloss.py --cases online:2:1e4 multiscale:3:1e5
    OMP_PROC_BIND=close OMP_PLACES=cores python tools/bench_geomloss.py --device cpu --threads 192
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

CASES = ([f"multiscale:{d}:{n}" for d in (2, 3) for n in ("1e4", "1e5", "1e6")]
         + [f"online:{d}:{n}" for d in (2, 3) for n in ("1e3", "3e3", "1e4", "3e4", "1e5", "1e6")]
         + [f"online:64:{n}" for n in ("1e3", "3e3", "1e4", "3e4", "1e5")])
CPU_CASES = ([f"multiscale:{d}:{n}" for d in (2, 3) for n in ("1e4", "1e5")]
             + [f"online:{d}:{n}" for d in (2, 3) for n in ("1e3", "1e4", "1e5")]
             + [f"online:64:{n}" for n in ("1e3", "1e4")])
# IEEE: neither approximate exp/log nor flushed subnormal numbers
RUNS = {"remblais": {}, "ieee": {"fast_math": False, "flush_denormals": False}, "geomloss": None}


def child(a):
    if a.run == "geomloss":
        # before torch: otherwise PyKeOps links its JIT binaries against the NVRTC
        # wheel loaded by torch, which has no libnvrtc.so, and fails in a fresh folder
        import pykeops  # noqa: F401
    import torch

    if a.device == "cpu":
        torch.set_num_threads(a.threads)
    backend, d, n = a.case.split(":")
    d, n = int(d), int(float(n))
    g = torch.Generator().manual_seed(0)
    x = torch.rand(n, d, generator=g)
    y = 0.05 + 0.9 * torch.rand(n, d, generator=g)
    y[:, 0] += 0.2
    wx, wy = torch.rand(n, generator=g), torch.rand(n, generator=g)
    x, y, wx, wy = (t.to(a.device) for t in (x, y, wx / wx.sum(), wy / wy.sum()))
    x.requires_grad_(True)
    if a.run == "geomloss":
        import pykeops
        from geomloss import SamplesLoss

        pykeops.set_verbose(False)
        if os.environ.get("KEOPS_CACHE_FOLDER"):
            pykeops.clean_pykeops()   # fresh build folder: build the JIT binaries there
        loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, backend=backend)
    else:
        from remblais import SamplesLoss

        loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, backend=backend, **RUNS[a.run])

    def step():
        L = loss(wx, x, wy, y)
        [gx] = torch.autograd.grad(L, [x])
        return L

    sync = torch.cuda.synchronize if a.device == "cuda" else (lambda: None)
    warm, reps = (1, 1) if n >= 10**6 else (2, 5)
    for _ in range(warm):
        step()
    sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        L = step()
        sync()
        times.append(time.perf_counter() - t0)
    print(json.dumps({"ms": 1e3 * statistics.median(times), "loss": L.item()}))


def driver(a):
    env = dict(os.environ)
    if a.device == "cpu":
        env["OMP_NUM_THREADS"] = str(a.threads)
    print(f"device {a.device}" + (f", {a.threads} threads" if a.device == "cpu" else "") + ", times in ms\n")
    print("| Case | remblais | remblais IEEE | GeomLoss | GeomLoss / remblais | GeomLoss / IEEE |")
    print("| --- | --- | --- | --- | --- | --- |", flush=True)
    for case in a.cases:
        ms = {}
        for run in RUNS:
            with tempfile.TemporaryDirectory() as build:
                cmd = [sys.executable, __file__, "--child", "--run", run, "--case", case,
                       "--device", a.device, "--threads", str(a.threads)]
                p = subprocess.run(cmd, capture_output=True, text=True,
                                   env=env | {"KEOPS_CACHE_FOLDER": build})
            if p.returncode:
                sys.stderr.write(p.stdout + p.stderr)
                ms[run] = float("nan")
                continue
            ms[run] = json.loads(p.stdout.strip().splitlines()[-1])["ms"]
        r, i, gl = ms["remblais"], ms["ieee"], ms["geomloss"]
        print(f"| {case} | {r:.3g} | {i:.3g} | {gl:.3g} | × {gl / r:.2f} | × {gl / i:.2f} |", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=None, help="backend:D:N")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--run", choices=list(RUNS))
    ap.add_argument("--case")
    a = ap.parse_args()
    a.cases = a.cases or (CASES if a.device == "cuda" else CPU_CASES)
    child(a) if a.child else driver(a)
