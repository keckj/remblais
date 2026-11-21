# Performance

The goal of remblais is to be **at least as fast as GeomLoss** wherever
GeomLoss runs, and to bring the same speed to AMD and Intel GPUs, which
PyKeOps does not support. On an NVIDIA GPU, remblais is faster than GeomLoss
in every case measured so far.

## Where the speed comes from

- **One compiled reduction per softmin, reused everywhere.** $\varepsilon$ (and every
  constant that depends on it) is a parameter of the kernox reduction, not
  part of the formula. The whole $\varepsilon$-scaling loop, the four softmins of the
  debiased divergence, the coarse and fine levels of `multiscale` and every
  later call share one compiled kernel. The host overhead of a kernox call
  is also lower than that of a PyKeOps `Genred` call, which matters for
  small clouds.
- **Two reductions per iteration instead of four.** The debiased Sinkhorn
  loop needs four softmins per iteration (x to y, y to x, x to x, y to y).
  Those that share the same set of columns are fused into one reduction over
  concatenated rows.
- **Fewer operations per pair.** Constant factors such as $\frac{1}{2}$, $1/\varepsilon$ or
  $1/(2\,\texttt{blur}^2)$ are folded into a single multiplication per pair.
- **Dense fallback for tiny clusters.** In 3D with few points per cluster,
  block-sparse computation wastes most of each GPU tile; remblais detects it
  and computes the fine level densely when that is cheaper.
- **One-sided gradients.** When only one of the two clouds requires a
  gradient, the final extrapolation only differentiates the pairs it needs.
- **Fast math by default on NVIDIA,** like PyKeOps.

## Measurements

Debiased Sinkhorn divergence, $p = 2$, $\texttt{blur} = 0.05$, $\texttt{scaling} = 0.5$, FP32, $N = M$.
The time is that of a forward and a backward pass (gradient with respect to
`x`), median of 5 calls after 2 warm-up calls (1 + 1 at $N = 10^6$). Data: `x`
uniform in $[0, 1]^D$, `y` uniform in $[0.05, 0.95]^D$ and shifted by 0.2 along the
first axis, random normalized weights.

remblais runs with its default settings; the `IEEE` column is
`fast_math=False, flush_denormals=False` (no approximate `exp`/`log`, subnormal
numbers kept). GeomLoss 0.3.1 runs with PyKeOps 2.3 in its best configuration
(see [Measuring PyKeOps](#measuring-pykeops)). Times in milliseconds; the
ratios are GeomLoss time / remblais time, above 1 when remblais is faster.
To reproduce: `python tools/bench_geomloss.py` (GPU) or
`python tools/bench_geomloss.py --device cpu --threads 192` (CPU).

![Speed-up of remblais over GeomLoss 0.3.1 at N = 10⁵ (10⁴ in D = 64), on an RTX 5090 and on 2 × AMD EPYC 9654](images/speedup-geomloss.svg)

### RTX 5090

PyTorch 2.11 (CUDA 12.8), kernox 1.1.0, remblais 2.1.0.

#### multiscale, D = 2

| $N$ | remblais | remblais IEEE | GeomLoss | GeomLoss / remblais | GeomLoss / IEEE |
| --- | --- | --- | --- | --- | --- |
| $10^4$ | 8.34 | 9.14 | 16.7 | × 2.00 | × 1.83 |
| $10^5$ | 8.86 | 9.01 | 17.1 | × 1.93 | × 1.89 |
| $10^6$ | 12.0 | 12.7 | 21.5 | × 1.80 | × 1.70 |

#### multiscale, D = 3

| $N$ | remblais | remblais IEEE | GeomLoss | GeomLoss / remblais | GeomLoss / IEEE |
| --- | --- | --- | --- | --- | --- |
| $10^4$ | 11.1 | 11.7 | 26.9 | × 2.43 | × 2.29 |
| $10^5$ | 50.4 | 52.7 | 64.8 | × 1.29 | × 1.23 |
| $10^6$ | 1820 | 2140 | 2100 | × 1.15 | × 0.98 |

#### online, D = 2

| $N$ | remblais | remblais IEEE | GeomLoss | GeomLoss / remblais | GeomLoss / IEEE |
| --- | --- | --- | --- | --- | --- |
| $10^3$ | 5.21 | 5.22 | 13.4 | × 2.58 | × 2.57 |
| $3 \cdot 10^3$ | 5.42 | 5.34 | 16.4 | × 3.02 | × 3.07 |
| $10^4$ | 5.43 | 5.50 | 27.4 | × 5.05 | × 4.99 |
| $3 \cdot 10^4$ | 13.0 | 16.3 | 58.7 | × 4.53 | × 3.59 |
| $10^5$ | 143 | 177 | 297 | × 2.07 | × 1.68 |
| $10^6$ | 14300 | 17900 | 20100 | × 1.41 | × 1.13 |

#### online, D = 3

| $N$ | remblais | remblais IEEE | GeomLoss | GeomLoss / remblais | GeomLoss / IEEE |
| --- | --- | --- | --- | --- | --- |
| $10^3$ | 5.64 | 5.64 | 14.7 | × 2.60 | × 2.60 |
| $3 \cdot 10^3$ | 5.59 | 5.69 | 18.3 | × 3.28 | × 3.22 |
| $10^4$ | 5.60 | 5.75 | 31.4 | × 5.62 | × 5.47 |
| $3 \cdot 10^4$ | 17.2 | 20.2 | 68.7 | × 4.00 | × 3.40 |
| $10^5$ | 190 | 221 | 330 | × 1.73 | × 1.50 |
| $10^6$ | 19400 | 23100 | 23300 | × 1.20 | × 1.01 |

#### online, D = 64

| $N$ | remblais | remblais IEEE | GeomLoss | GeomLoss / remblais | GeomLoss / IEEE |
| --- | --- | --- | --- | --- | --- |
| $10^3$ | 6.53 | 6.64 | 31.2 | × 4.78 | × 4.70 |
| $3 \cdot 10^3$ | 6.88 | 6.87 | 61.8 | × 8.98 | × 9.00 |
| $10^4$ | 23.8 | 24.0 | 169 | × 7.11 | × 7.05 |
| $3 \cdot 10^4$ | 208 | 211 | 490 | × 2.36 | × 2.32 |
| $10^5$ | 2350 | 2420 | 6140 | × 2.61 | × 2.54 |

For small clouds, the time of remblais is almost constant (about 5 ms): it is
the host overhead of the Sinkhorn loop, not the GPU work. At large $N$ in
dimension 3 or less, almost all the time is spent in the log-sum-exp
reductions themselves, so the margin follows the throughput of the kernox
reduction against that of PyKeOps: × 1.15 to × 1.4 at $N = 10^6$. With IEEE
arithmetic, remblais is on par with GeomLoss (which uses fast math) in
multiscale $D = 3$ at $N = 10^6$ (× 0.98) and faster everywhere else.

### CPU: 2 × AMD EPYC 9654

192 cores, 192 threads pinned (`OMP_PROC_BIND=close OMP_PLACES=cores`),
PyTorch 2.15.0+cpu, kernox 1.1.1 with its CPU backend (with kernox 1.1.0, the
$N = 10^3$ cases are about three times slower at 192 threads: use 1.1.1 or later
on large nodes); GeomLoss runs on the CPU
backend of PyKeOps 2.3, which compiles without fast math: the IEEE column is
the like-for-like comparison.

#### multiscale, D = 2

| $N$ | remblais | remblais IEEE | GeomLoss | GeomLoss / remblais | GeomLoss / IEEE |
| --- | --- | --- | --- | --- | --- |
| $10^4$ | 23.6 | 22.6 | 31.4 | × 1.33 | × 1.39 |
| $10^5$ | 55.9 | 56.0 | 88.5 | × 1.58 | × 1.58 |

#### multiscale, D = 3

| $N$ | remblais | remblais IEEE | GeomLoss | GeomLoss / remblais | GeomLoss / IEEE |
| --- | --- | --- | --- | --- | --- |
| $10^4$ | 40.9 | 44.1 | 901 | × 22.0 | × 20.4 |
| $10^5$ | 347 | 348 | 4590 | × 13.2 | × 13.2 |

#### online, D = 2

| $N$ | remblais | remblais IEEE | GeomLoss | GeomLoss / remblais | GeomLoss / IEEE |
| --- | --- | --- | --- | --- | --- |
| $10^3$ | 3.78 | 3.86 | 8.94 | × 2.37 | × 2.32 |
| $10^4$ | 20.7 | 21.5 | 159 | × 7.68 | × 7.40 |
| $10^5$ | 1410 | 1500 | 14400 | × 10.2 | × 9.60 |

#### online, D = 3

| $N$ | remblais | remblais IEEE | GeomLoss | GeomLoss / remblais | GeomLoss / IEEE |
| --- | --- | --- | --- | --- | --- |
| $10^3$ | 4.16 | 4.06 | 9.97 | × 2.40 | × 2.45 |
| $10^4$ | 23.0 | 24.3 | 188 | × 8.18 | × 7.74 |
| $10^5$ | 1630 | 1740 | 17300 | × 10.6 | × 9.89 |

#### online, D = 64

| $N$ | remblais | remblais IEEE | GeomLoss | GeomLoss / remblais | GeomLoss / IEEE |
| --- | --- | --- | --- | --- | --- |
| $10^3$ | 9.84 | 9.98 | 26.1 | × 2.66 | × 2.62 |
| $10^4$ | 154 | 154 | 1380 | × 8.95 | × 8.96 |

For $N \le 10^4$, the FP32 value of remblais is within $9 \cdot 10^{-10}$ to $1 \cdot 10^{-7}$ (relative)
of an FP64 reference, with or without fast math.

### Multiscale in 2D and in 3D

On the CPU, `multiscale` gains much less in $D = 2$ (× 1.3 to × 1.6) than in
$D = 3$ (× 13 to × 22). The $\varepsilon$ schedule explains it:

![The ε schedule of the multiscale backend in 2D and 3D, against the size of the clusters](images/eps-schedule.svg)

The default cell size is $\texttt{diameter} / (\sqrt{D} \cdot 2000^{1/D})$: 0.024 in 2D but 0.084
in 3D for these clouds. With $\texttt{blur} = 0.05$, every iteration in 2D stays above
the cell size and runs on the centroids; only the final extrapolation touches
the points. The time is then mostly fixed costs (clustering, small
reductions, host overhead), which both libraries pay. In 3D, the last
iterations run on the points, block by block, and take most of the time:
this is where the block-sparse reductions of kernox are more than ten times
faster than those of PyKeOps on the CPU. With a blur below the cell size, the
fine iterations come back in 2D too.

These figures cover only the cases listed; they are not a general claim of
superiority over GeomLoss on every workload or machine. A case where remblais
is slower than GeomLoss is treated as a bug.

## Measuring PyKeOps

PyKeOps does not always produce the same kernel for the same formula (seen
with PyKeOps 2.4b on an RTX 2080 Ti).
When the $D = 3$ log-sum-exp kernel is the first one compiled in a process, it
uses an approximate `exp` (26 ms at $N = 10^5$); when it is compiled after the
$D = 2$ kernel in the same process, it uses the IEEE `expf` (39 ms). The
`use_fast_math` flag itself does not change the generated code.

The GeomLoss figures above are its best case: one fresh process and one fresh
build folder for each configuration, so that the kernel being measured is
always the first one compiled, as for a user who works in a single dimension.
This is why remblais is compared with its fast math setting on. On the CPU,
PyKeOps compiles with `-O3 -fopenmp`, without fast math: the IEEE column is
then the like-for-like comparison.

## Tuning

A few environment variables, read when remblais is imported, change the
strategy. They are off by default because they only help in specific cases.

- `REMBLAIS_CUDA_GRAPHS=1`: the Sinkhorn iterations of the `online` backend
  are captured in a CUDA graph on the first call for a given shape, then
  replayed. This saves host overhead for repeated calls with the same small
  $N$ (7.5 to 5.7 ms at $N = 10^3$, $D = 2$); there is no gain beyond $10^4$ points.
- `REMBLAIS_DOT_MIN_DIM=16`: for the $\frac{1}{2}\|x - y\|^2$ cost in dimension 16 or more,
  computes $\langle x, y \rangle$ after centering instead of $\|x - y\|^2$. About 23 % faster in
  $D = 64$ for $N \ge 10^4$, but slower at $N = 10^3$, and the rounding error grows with
  $\|x\|\|y\|$ instead of $\|x - y\|^2$: use it on centered, well-scaled data.
- `REMBLAIS_DENSE_RATIO`: the threshold of the dense fallback of the
  `multiscale` backend, as a fraction of $N \cdot M$ (default 0.75). `inf` always
  uses the block-sparse computation, exactly as GeomLoss does.

## Timing it yourself

Synchronize the GPU around the timed region, and warm up first: the first
call compiles the kernels (they are then cached on disk).

<!-- skip-test -->
```python
import time

import torch
from remblais import SamplesLoss

N, D = 100_000, 3
x = torch.rand(N, D, device="cuda", requires_grad=True)
y = torch.rand(N, D, device="cuda")
loss = SamplesLoss("sinkhorn", p=2, blur=0.05, backend="online")

def step():
    [g] = torch.autograd.grad(loss(x, y), [x])
    return g

for _ in range(2):
    step()
torch.cuda.synchronize()
times = []
for _ in range(5):
    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
print(f"median: {sorted(times)[2] * 1e3:.1f} ms")
```
