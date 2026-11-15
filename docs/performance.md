# Performance

The goal of remblais is to be **at least as fast as GeomLoss** wherever
GeomLoss runs, and to bring the same speed to AMD and Intel GPUs, which
PyKeOps does not support. In every case measured so far, on an NVIDIA GPU,
remblais is already faster than GeomLoss.

## Where the speed comes from

- **One compiled reduction per softmin, reused everywhere.** ε (and every
  constant that depends on it) is a parameter of the kernox reduction, not
  part of the formula. The whole ε-scaling loop, the four softmins of the
  debiased divergence, the coarse and fine levels of `multiscale` and every
  later call share one compiled kernel. The host overhead of a kernox call
  is also lower than that of a PyKeOps `Genred` call, which matters for
  small clouds.
- **Two reductions per iteration instead of four.** The debiased Sinkhorn
  loop needs four softmins per iteration (x to y, y to x, x to x, y to y).
  Those that share the same set of columns are fused into one reduction over
  concatenated rows.
- **Fewer operations per pair.** Constant factors such as ½, 1/ε or
  1/(2 blur²) are folded into a single multiplication per pair.
- **Dense fallback for tiny clusters.** In 3D with few points per cluster,
  block-sparse computation wastes most of each GPU tile; remblais detects it
  and computes the fine level densely when that is cheaper.
- **One-sided gradients.** When only one of the two clouds requires a
  gradient, the final extrapolation only differentiates the pairs it needs.
- **Fast math by default on NVIDIA,** like PyKeOps.

## Measurements

Debiased Sinkhorn divergence, p = 2, blur = 0.05, scaling = 0.5, FP32, N = M.
The time is that of a forward and a backward pass (gradient with respect to
`x`), median of 5 calls after 2 warm-up calls (1 + 1 at N = 10⁶), on an RTX
2080 Ti with PyTorch 2.11 (CUDA 12.8). Data: `x` uniform in [0, 1]ᴰ, `y`
uniform in [0.05, 0.95]ᴰ and shifted by 0.2 along the first axis, random
normalized weights.

remblais runs with its default `fast_math` setting; the `IEEE` column is
`fast_math=False`. GeomLoss 0.3.1 runs with PyKeOps 2.4b in its best
configuration (see [Measuring PyKeOps](#measuring-pykeops)). Times in
milliseconds; ratios lower than 1 mean that remblais is faster.

### multiscale, D = 2

| N | remblais | remblais IEEE | GeomLoss | remblais / GeomLoss | IEEE / GeomLoss |
| --- | --- | --- | --- | --- | --- |
| 10⁴ | 11.0 | 11.4 | 24.4 | ×0.45 | ×0.47 |
| 10⁵ | 10.7 | 11.0 | 26.4 | ×0.40 | ×0.42 |
| 10⁶ | 34.7 | 39.8 | 43.9 | ×0.79 | ×0.91 |

### multiscale, D = 3

| N | remblais | remblais IEEE | GeomLoss | remblais / GeomLoss | IEEE / GeomLoss |
| --- | --- | --- | --- | --- | --- |
| 10⁴ | 14.2 | 13.9 | 54.5 | ×0.26 | ×0.26 |
| 10⁵ | 150 | 171 | 162 | ×0.92 | ×1.06 |
| 10⁶ | 8795 | 10675 | 9165 | ×0.96 | ×1.16 |

### online, D = 2

| N | remblais | remblais IEEE | GeomLoss | remblais / GeomLoss | IEEE / GeomLoss |
| --- | --- | --- | --- | --- | --- |
| 10³ | 11.5 | 8.87 | 19.5 | ×0.59 | ×0.46 |
| 3·10³ | 9.92 | 8.22 | 22.8 | ×0.44 | ×0.36 |
| 10⁴ | 10.3 | 12.1 | 31.2 | ×0.33 | ×0.39 |
| 3·10⁴ | 65.8 | 83.9 | 112 | ×0.59 | ×0.75 |
| 10⁵ | 713 | 906 | 844 | ×0.84 | ×1.07 |
| 10⁶ | 73206 | 92872 | 81799 | ×0.89 | ×1.14 |

### online, D = 3

| N | remblais | remblais IEEE | GeomLoss | remblais / GeomLoss | IEEE / GeomLoss |
| --- | --- | --- | --- | --- | --- |
| 10³ | 9.66 | 9.07 | 19.7 | ×0.49 | ×0.46 |
| 3·10³ | 9.84 | 8.28 | 24.7 | ×0.40 | ×0.33 |
| 10⁴ | 11.1 | 13.5 | 35.6 | ×0.31 | ×0.38 |
| 3·10⁴ | 87.6 | 106 | 143 | ×0.61 | ×0.75 |
| 10⁵ | 944 | 1137 | 1104 | ×0.85 | ×1.03 |
| 10⁶ | 96878 | 116037 | 107123 | ×0.90 | ×1.08 |

### online, D = 64

| N | remblais | remblais IEEE | GeomLoss | remblais / GeomLoss | IEEE / GeomLoss |
| --- | --- | --- | --- | --- | --- |
| 10³ | 10.4 | 10.2 | 55.0 | ×0.19 | ×0.19 |
| 3·10³ | 16.4 | 18.6 | 96.4 | ×0.17 | ×0.19 |
| 10⁴ | 131 | 135 | 292 | ×0.45 | ×0.46 |
| 3·10⁴ | 1127 | 1163 | 2508 | ×0.45 | ×0.46 |
| 10⁵ | 12737 | 13026 | 23700 | ×0.54 | ×0.55 |

For N ≤ 10⁴, the FP32 value of remblais is within 9·10⁻¹⁰ to 1·10⁻⁷ (relative)
of an FP64 reference, with or without fast math.

These figures cover only the cases listed; they are not a general claim of
superiority over GeomLoss on every workload or card. A case where remblais is
slower than GeomLoss is treated as a bug.

At large N in dimension 3 or less, almost all the time is spent in the
log-sum-exp reductions themselves, so the margin follows the throughput of
the kernox reduction against that of PyKeOps. With IEEE arithmetic
(`fast_math=False`), remblais is up to 16 % slower than GeomLoss in some of
these cases (D ≤ 3, N ≥ 10⁵), and faster everywhere else.

## Measuring PyKeOps

PyKeOps 2.4b does not always produce the same kernel for the same formula.
When the D = 3 log-sum-exp kernel is the first one compiled in a process, it
uses an approximate `exp` (26 ms at N = 10⁵); when it is compiled after the
D = 2 kernel in the same process, it uses the IEEE `expf` (39 ms). The
`use_fast_math` flag itself does not change the generated code.

The GeomLoss figures above are its best case: one fresh process and one fresh
build folder for each configuration, so that the kernel being measured is
always the first one compiled, as for a user who works in a single dimension.
This is why remblais is compared with its fast math setting on.

## Tuning

A few environment variables, read when remblais is imported, change the
strategy. They are off by default because they only help in specific cases.

- `REMBLAIS_CUDA_GRAPHS=1`: the Sinkhorn iterations of the `online` backend
  are captured in a CUDA graph on the first call for a given shape, then
  replayed. This saves host overhead for repeated calls with the same small
  N (7.5 to 5.7 ms at N = 10³, D = 2); there is no gain beyond 10⁴ points.
- `REMBLAIS_DOT_MIN_DIM=16`: for the ½‖x − y‖² cost in dimension 16 or more,
  computes ⟨x, y⟩ after centering instead of ‖x − y‖². About 23 % faster in
  D = 64 for N ≥ 10⁴, but slower at N = 10³, and the rounding error grows with
  ‖x‖‖y‖ instead of ‖x − y‖²: use it on centered, well-scaled data.
- `REMBLAIS_DENSE_RATIO`: the threshold of the dense fallback of the
  `multiscale` backend, as a fraction of N · M (default 0.75). `inf` always
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
