# remblais

<p align="center">
  <img src="docs/images/remblais-banner.png" alt="remblais logo" width="560">
</p>

remblais computes geometric losses between measures on NVIDIA (CUDA), AMD
(ROCm/HIP) and Intel (SYCL) GPUs, and on x86-64 CPUs: Sinkhorn divergences (entropic optimal
transport), kernel norms (MMD) and Hausdorff divergences between point clouds,
images and volumes. It is a drop-in
replacement for [GeomLoss](https://www.kernel-operations.io/geomloss/) 0.3.1,
built on [kernox](https://github.com/keckj/kernox) instead of PyKeOps: write
`import remblais as geomloss` and your code runs unchanged. A large loss can
also be split across several GPUs or nodes. Support for other architectures
and accelerators may be added later.

![A Sinkhorn gradient flow computed with remblais moves a blob of points onto a ring](docs/images/flow.svg)

Images are measures too. Here, `ImagesBarycenter` computes the Wasserstein barycenters of four shapes:

![Wasserstein barycenters of a disk, a ring, a square and a cross, computed with remblais](docs/images/barycenters.svg)

- Documentation: <https://keckj.github.io/remblais/>
- Getting started: <https://keckj.github.io/remblais/getting-started/>

## Why remblais

- **Same API as GeomLoss.** `SamplesLoss`, `sinkhorn_divergence`,
  `ImagesBarycenter` and the `ot` module have the GeomLoss 0.3.1 signatures,
  calling conventions and output shapes. With `backend="tensorized"`, values
  and gradients match GeomLoss bit for bit, apart from the documented fixes.
- **Faster.** Every softmin is one fused kernox reduction, compiled once and
  reused for all values of ε; the debiased Sinkhorn loop fuses its four
  softmins into two reductions per iteration.
- **NVIDIA, AMD and Intel GPUs, and CPUs.** The `online` and `multiscale`
  backends, which GeomLoss runs through PyKeOps on NVIDIA GPUs only, run here
  on CUDA, ROCm/HIP and SYCL through kernox, and on x86-64 Linux CPUs as
  multithreaded, vectorized C++. remblais's own test suite runs on the CPU,
  NVIDIA and Intel GPUs; on AMD GPUs it relies on kernox's tested HIP backend.
- **Several GPUs or nodes.** `SamplesLoss(..., distributed=True)` splits the
  softmins of the `online` and `multiscale` backends across the processes of
  a `torch.distributed` group (NCCL between GPUs, Gloo between CPU nodes),
  with the same value and gradients on every process. The same training
  script runs alone or under `torchrun`, `mpirun` or `srun`: see
  [Several GPUs or nodes](docs/guides/distributed.md).
- **Fixes, not surprises.** A few GeomLoss bugs are fixed (the gradient of
  unbalanced transport, `"hausdorff"` without a kernel, ...). Every
  difference is listed in [Coming from GeomLoss](docs/geomloss.md).

## Installation

remblais needs Python 3.10 or later, PyTorch and kernox. Install PyTorch for
your platform, then kernox with the extra that matches your GPU, then remblais,
both from PyPI. For an NVIDIA GPU:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "kernox[cuda]"
pip install remblais
```

On the CPU only (x86-64 Linux, with `g++` 12 or later for the fast path):

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install remblais
```

For AMD and Intel GPUs, install PyTorch from the `rocm` or `xpu` index and
kernox with the `rocm` or `xpu` extra. Intel GPUs also need a SYCL runtime
matching the host `icpx` compiler, see
[Intel GPUs](https://keckj.github.io/kernox/getting-started/#intel-gpus). The
details are in the
[kernox installation guide](https://keckj.github.io/kernox/getting-started/)
and in [Getting started](docs/getting-started.md).

## Performance

The goal of remblais is to be **at least as fast as GeomLoss** wherever
GeomLoss runs. In every case measured so far, on an NVIDIA GPU and on the
CPU, it is faster.

Sinkhorn divergence, p = 2, FP32, forward and backward passes, on an RTX
5090; remblais with its default settings, against GeomLoss 0.3.1 with PyKeOps
2.3 in its best configuration (each PyKeOps kernel compiled alone in a fresh
process). Speed-up of remblais (GeomLoss time / remblais time):

![Speed-up of remblais over GeomLoss 0.3.1 at N = 10⁵ (10⁴ in D = 64), on an RTX 5090 and on 2 × AMD EPYC 9654](docs/images/speedup-geomloss.svg)

All the sizes measured:

| Case | N = 10³ | N = 10⁴ | N = 10⁵ | N = 10⁶ |
| --- | --- | --- | --- | --- |
| multiscale, D = 2 | | × 2.0 | × 1.9 | × 1.8 |
| multiscale, D = 3 | | × 2.4 | × 1.3 | × 1.15 |
| online, D = 2 | × 2.6 | × 5.1 | × 2.1 | × 1.4 |
| online, D = 3 | × 2.6 | × 5.6 | × 1.7 | × 1.2 |
| online, D = 64 | × 4.8 | × 7.1 | × 2.6 | not measured |

On the CPU (2 × AMD EPYC 9654, 192 threads), against GeomLoss on the CPU
backend of PyKeOps:

| Case | N = 10³ | N = 10⁴ | N = 10⁵ |
| --- | --- | --- | --- |
| multiscale, D = 2 | | × 1.3 | × 1.6 |
| multiscale, D = 3 | | × 22 | × 13 |
| online, D = 2 | × 2.4 | × 7.7 | × 10 |
| online, D = 3 | × 2.4 | × 8.2 | × 11 |
| online, D = 64 | × 2.7 | × 9.0 | not measured |

These figures cover only the cases listed; they are not a general claim of
superiority over GeomLoss on every workload or machine. The full tables and
the protocol are on the [Performance](docs/performance.md) page.

## Several GPUs or nodes

A large loss can be split across several GPUs or CPU nodes. Every process
runs the same script on the same clouds and gets the same value and the same
gradients, so a plain optimizer keeps the parameters identical everywhere:

```python
# fit.py
import torch
import kernox.distributed as kd
from remblais import SamplesLoss

info = kd.init_from_env()                 # first thing: joins the other processes
g = torch.Generator().manual_seed(0)      # the same clouds on every process
x = torch.rand(1_000_000, 3, generator=g).to(info.device).requires_grad_()
y = (torch.rand(1_000_000, 3, generator=g) + 0.5).to(info.device)

loss = SamplesLoss("sinkhorn", blur=0.05, backend="multiscale", distributed=True)
L = loss(x, y)
L.backward()                              # the full gradient, on every process
if info.rank == 0:                        # print and save on one process only
    print(L.item())
```

Test it on your machine, then launch it on the cluster with the tool you
already use; `init_from_env` finds the rank, the GPU and the first node by
itself, and no `mpi4py` is needed:

```sh
python fit.py                                                       # 1 process
torchrun --nproc-per-node 2 fit.py                                  # 2 processes, same result
torchrun --nproc-per-node 4 fit.py                                  # the 4 GPUs of a node
srun --nodes 2 --ntasks-per-node 4 --gpus-per-node 4 python fit.py  # SLURM, 2 nodes of 4 GPUs
mpirun -np 8 --map-by ppr:1:node --bind-to none python fit.py       # 8 CPU nodes, all their cores
```

Debiased Sinkhorn divergence, p = 2, blur = 0.05, D = 3, FP32, forward and
backward passes, one process per GPU or per CPU node:

| Case | 1 device | 8 devices | Speed-up |
| --- | --- | --- | --- |
| online, N = 10⁶, AMD MI210 (2 nodes of 4) | 176 s | 21.3 s | × 8.3 |
| multiscale, N = 10⁷, AMD MI210 (2 nodes of 4) | 1391 s | 188 s | × 7.4 |
| online, N = 10⁶, CPU nodes (2 × EPYC 9654) | 157 s | 20.7 s | × 7.6 |
| multiscale, N = 10⁷, CPU nodes (2 × EPYC 9654) | 1453 s | 185 s | × 7.9 |

The guide [Several GPUs or nodes](docs/guides/distributed.md) goes through
it step by step, with [all the measurements](docs/guides/distributed.md#scaling)
and when splitting pays. Batch scripts for SLURM and OAR, and
troubleshooting: [Running on a cluster](https://keckj.github.io/kernox/guides/cluster/).

## About the name and logo

In 1781, Gaspard Monge asked how to move a pile of earth, the *déblais*, into
a hole of the same volume, the *remblais*, at the lowest cost: the founding
problem of optimal transport. remblais is where the mass ends up. The logo
shows the ox of symox and kernox, harnessed this time to a scraper, carrying
the déblais to the remblais: Monge's optimal transport.

## License

At your option, [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE). Code
derived from GeomLoss stays under the MIT license of GeomLoss; its provenance
and notice are in [`THIRD_PARTY.md`](THIRD_PARTY.md).

## Related projects

- GeomLoss, the original library: <https://github.com/jeanfeydy/geomloss>
- kernox, fused GPU and CPU reductions: <https://github.com/keckj/kernox>
- symox, symbolic engine and code generation: <https://github.com/keckj/symox>
