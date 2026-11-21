"""Strong scaling of SamplesLoss(distributed=True) (see docs/guides/distributed.md).

Debiased Sinkhorn divergence, p = 2, blur = 0.05, scaling = 0.5, FP32, N = M,
D = 3, forward and backward passes (gradient with respect to x), with the data
of tools/bench_geomloss.py. Launch one process per GPU or per node (or
socket); with a single process, the loss is computed locally and gives the
reference time. Rank 0 prints one JSON line per case.

    torchrun --nproc-per-node 4 tools/bench_distributed.py --cases online:1e6 multiscale:3e6
    mpirun -np 4 --map-by ppr:1:node python tools/bench_distributed.py --cases online:3e5
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.distributed as dist

import kernox.distributed as kd
from remblais import SamplesLoss


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", default=["online:1e5"], help="backend:N")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--min-reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="untimed runs first (0 for long cases: "
                    "the kernels are then compiled or loaded from the cache during the first timed run)")
    ap.add_argument("--min-time", type=float, default=2.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    info = kd.init_from_env()
    device = info.device
    for case in a.cases:
        backend, n = case.split(":")
        n = int(float(n))
        g = torch.Generator().manual_seed(0)
        x = torch.rand(n, a.dim, generator=g)
        y = 0.05 + 0.9 * torch.rand(n, a.dim, generator=g)
        y[:, 0] += 0.2
        wx, wy = torch.rand(n, generator=g), torch.rand(n, generator=g)
        x, y, wx, wy = (t.to(device) for t in (x, y, wx / wx.sum(), wy / wy.sum()))
        loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, backend=backend, distributed=True)

        def step():
            xg = x.clone().requires_grad_()
            L = loss(wx, xg, wy, y)
            (gx,) = torch.autograd.grad(L, xg)
            return L.detach(), gx

        value = None
        for _ in range(a.warmup):  # compilation, caches
            value = float(step()[0])
        times = []
        start = time.perf_counter()
        while len(times) < a.min_reps or time.perf_counter() - start < a.min_time:
            if dist.is_initialized():
                dist.barrier()
            _sync(device)
            t = time.perf_counter()
            L = step()[0]
            _sync(device)
            times.append(time.perf_counter() - t)
            value = float(L) if value is None else value
        if info.rank == 0:
            print(json.dumps({"backend": backend, "N": n, "D": a.dim, "P": info.world_size,
                              "device": str(device), "threads": torch.get_num_threads(), "tag": a.tag,
                              "time": statistics.median(times), "reps": len(times), "value": value}),
                  flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
