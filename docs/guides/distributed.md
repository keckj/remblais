# Several GPUs or nodes

With `distributed=True`, `SamplesLoss` splits its kernox reductions across
the processes of a `torch.distributed` group: several GPUs of a node with
NCCL, several CPU nodes with Gloo. Every process passes the same point
clouds and receives the same value and the same gradients, so the rest of
the training loop does not change. Each process only computes a share of
the rows of every softmin.

## A distributed loss

```python
import torch
from remblais import SamplesLoss

x = torch.rand(2_000, 3, requires_grad=True)
y = torch.rand(3_000, 3) + 0.2

loss = SamplesLoss("sinkhorn", blur=0.05, backend="online", distributed=True)
L = loss(x, y)          # the same value on every process
L.backward()            # x.grad: the full gradient on every process
```

Without an initialized group, or with a single process, `distributed=True`
computes locally: the same script runs on a laptop and on a cluster. Pass a
process group instead of `True` to split across that group.

It applies to the `online` and `multiscale` backends, and to every loss
that uses them (`"sinkhorn"`, `"hausdorff"`, `"energy"`, `"gaussian"`,
`"laplacian"`). The `tensorized` backend stays local. It needs kernox 1.2
or later.

## Step by step

### 1. A training script

Each process runs the whole script. It joins the others with
`kernox.distributed.init_from_env()`, at the start, and passes the same
clouds to the loss:

<!-- skip-test -->
```python
# fit.py: moves a cloud x towards a cloud y by gradient descent
import torch
import kernox.distributed as kd
from remblais import SamplesLoss

info = kd.init_from_env()      # rank, world_size, local_rank, device
device = info.device           # this process's GPU, or the CPU

g = torch.Generator().manual_seed(0)     # the same clouds on every process
x = torch.rand(1_000_000, 3, generator=g).to(device).requires_grad_()
y = (torch.rand(1_000_000, 3, generator=g) + 0.5).to(device)

loss = SamplesLoss("sinkhorn", blur=0.05, backend="multiscale", distributed=True)
opt = torch.optim.SGD([x], lr=len(x))    # the gradient of each point is weighted by 1/N

for it in range(10):
    opt.zero_grad()
    L = loss(x, y)
    L.backward()                         # the full gradient, on every process
    opt.step()                           # the same step everywhere: x stays identical
    if info.rank == 0:                   # print and save on one process only
        print(f"{it}: {L.item():.6f}")
```

Every process computes the same gradient, so a plain optimizer keeps the
parameters identical on every process: no `DistributedDataParallel` and no
extra communication. Three rules:

- every process calls the loss the same number of times, with clouds of the
  same sizes (a call inside `if info.rank == 0:` hangs the others);
- every process passes the same clouds: the same seed, the same file, or
  `torch.distributed.broadcast(x, src=0)`;
- logs and files on one process only (`if info.rank == 0`).

If your program already trains with `DistributedDataParallel` on a different
batch per process, keep `distributed=False`: each process then computes the
loss of its own batch.

### 2. Test it on your machine

<!-- skip-test -->
```bash
python fit.py                          # 1 process
torchrun --nproc-per-node 2 fit.py     # 2 processes: the same values, up to rounding
```

### 3. Launch it on the cluster

<!-- skip-test -->
```bash
torchrun --nproc-per-node 4 fit.py                                  # one node, 4 GPUs
srun --nodes 2 --ntasks-per-node 4 --gpus-per-node 4 python fit.py  # SLURM, 2 nodes of 4 GPUs
mpirun -np 8 --map-by ppr:1:node --bind-to none python fit.py       # 8 CPU nodes, all their cores
```

`init_from_env` reads the environment of `torchrun`, `mpirun` (Open MPI,
MPICH, Intel MPI), PMIx or SLURM `srun`; it uses NCCL between GPUs (one GPU
per process) and Gloo between CPU processes, and divides the cores of a node
between its processes. No `mpi4py` is needed. Batch scripts for SLURM and
OAR, a check of the setup and the usual problems (a run that hangs, a slow
network, one thread per process): [Running on a cluster](https://keckj.github.io/kernox/guides/cluster/)
in the kernox documentation.

## What is split

Every softmin of the Sinkhorn loop is one kernox reduction over the points
of one cloud for each point of the other; each process computes a
contiguous span of these rows, and an `all_gather` of the result (one value
per point) gives every process the full softmin. With the `multiscale`
backend, the processes share the pairs kept by the block-sparse ranges,
not the rows, so that each one gets the same amount of work. The rest of
the computation (clustering, ε-scaling, the final sums) is cheap and runs
on every process.

In the backward pass, the gradients of the point positions and weights are
summed over the processes, so every process ends with the full gradients.
The CUDA graphs of the Sinkhorn loop are not used in distributed mode.

The value is the same on every process and from one run to the next with
the same number of processes; it differs from a single-process run by
rounding only, since the sums are done in another order.

## Scaling

Strong scaling measured with `tools/bench_distributed.py`: debiased Sinkhorn
divergence, p = 2, blur = 0.05, scaling = 0.5, D = 3, FP32, N = M, forward
and backward passes (gradient with respect to `x`), in seconds.

![Strong scaling of SamplesLoss(distributed=True): speed-up against the number of devices](../images/scaling.svg)

Speed-up = time on one device / time on P devices; the times are in the
tables below.

AMD Instinct MI210, one process per GPU, 4 GPUs per node, RCCL:

| Case | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs (2 nodes) | Speed-up at 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| online, N = 10⁶ | 176 | 87.9 | 43.6 | 21.3 | × 8.3 |
| multiscale, N = 3 × 10⁶ | 128 | 66.6 | 34.1 | 18.0 | × 7.1 |
| multiscale, N = 10⁷ | 1391 | 704 | 363 | 188 | × 7.4 |

CPU nodes of 2 × AMD EPYC 9654 (192 cores), one process per node with all
its cores, Gloo over TCP:

| Case | 1 node | 2 nodes | 4 nodes | 8 nodes | Speed-up at 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| online, N = 10⁶ | 157 | 78.7 | 40.3 | 20.7 | × 7.6 |
| multiscale, N = 3 × 10⁶ | 130 | 65.1 | 33.5 | 17.4 | × 7.4 |
| multiscale, N = 10⁷ | 1453 | 720 | 364 | 185 | × 7.9 |

## When it pays

Each softmin exchanges one vector the size of a cloud, while its work is the
product of the two sizes: for large clouds, the speed-up is close to the
number of processes, on GPUs as on CPU nodes. Splitting pays when one loss
takes seconds or more on one device, from about 10⁵ points with `online` and
10⁶ with `multiscale`. Below that, the latency of the collectives (tens of
microseconds within a node, more across nodes, at every softmin of every
iteration) dominates: keep `distributed=False`.

To measure on your machine:

<!-- skip-test -->
```bash
torchrun --nproc-per-node 4 tools/bench_distributed.py --cases online:1e6 multiscale:3e6
```
