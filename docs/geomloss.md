# Coming from GeomLoss

remblais is a drop-in replacement for GeomLoss 0.3.1. It has the same
functions, the same signatures, the same calling conventions and the same
output shapes; it computes with kernox instead of PyKeOps. Most code only
needs its import changed:

<!-- skip-test -->
```python
# import geomloss
import remblais as geomloss

loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05)
```

or, function by function:

<!-- skip-test -->
```python
# from geomloss import SamplesLoss, ImagesBarycenter, sinkhorn_divergence
# from geomloss import ot
from remblais import SamplesLoss, ImagesBarycenter, sinkhorn_divergence
from remblais import ot
```

As in GeomLoss, `ot` is a submodule: import it explicitly.

## What is the same

- `SamplesLoss`: every loss (`"sinkhorn"`, `"hausdorff"`, `"energy"`,
  `"gaussian"`, `"laplacian"`), every parameter, the three calling
  conventions, batches, `potentials=True`, the `backend="auto"` rule (except
  for custom costs, see below), custom kernels, explicit cluster labels, and the messages of `verbose=True`. Inputs of invalid shapes raise
  the same errors with the same messages.
- `backend="tensorized"` is pure PyTorch, as in GeomLoss: values and
  gradients are identical bit for bit, apart from the fixes listed below.
  The `online` and `multiscale` backends compute the same algorithm with
  kernox; their results agree with GeomLoss up to rounding.
- `sinkhorn_divergence` (images and volumes) and `ImagesBarycenter`.
- The `ot` module: `solve`, `solve_batch`, `solve_sample`, `OTResult`,
  `OTResultMatrix`, `OTResultSample`, `LinearOperator`. The test suite of
  `geomloss.ot` runs on remblais and passes.

## What you do not need any more

- PyKeOps, and its compilation toolchain. kernox compiles its kernels at
  runtime with NVRTC (NVIDIA), hipRTC (AMD) or the oneAPI compiler (Intel),
  and caches them on disk.
- Recompilations when $\varepsilon$ changes: in remblais, $\varepsilon$ and the blur are parameters
  of the compiled reduction, so the whole $\varepsilon$-scaling loop, every call and
  every batch reuse one kernel.

## Things that work differently

- **GPU vendors.** remblais runs on NVIDIA, AMD and Intel GPUs, wherever
  kernox does.
- **CPU.** On CPU tensors, the `online` and `multiscale` backends run on the
  kernox CPU backend: multithreaded, vectorized C++ on x86-64 Linux with
  `g++`. On other hosts (macOS, ARM), they fall back to the kernox oracle, a
  slow PyTorch reference. `tensorized` is unchanged.
- **Custom costs are kernox formulas.** A function given as `cost` (or as
  `kernel`) receives symbolic kernox formulas, not PyKeOps `LazyTensor`s, and
  `backend="auto"` runs it with the `online` backend whatever the size; see
  [Concepts](concepts.md#costs). KeOps formula strings such as `"Norm2(X-Y)"`
  are **not supported**: `cost="Norm2(X-Y)"` becomes
  `cost=lambda x, y: (x - y).norm2()`; see
  [Translating KeOps cost strings](#translating-keops-cost-strings) for the
  whole correspondence. The `tensorized` and `multiscale`
  backends need a `(formula, dense routine)` pair; with
  `backend="tensorized"`, a function on tensors works as in GeomLoss.
- **`LazyTensor`.** The `ot` module uses `remblais.lazy.LazyTensor`, a small
  kernox-based replacement for the PyKeOps `LazyTensor`. It covers what
  `geomloss.ot` needs: i and j variables and parameters, `+ − × ÷`, `**2`,
  `exp`, `log`, `sqrt`, `abs`, `sum(-1)`, `(x|y)`, the `sum`, `logsumexp`,
  `min`, `max` and `arg*` reductions, `@`, `.T` and NumPy inputs. Python
  numbers become parameters of the compiled reduction.

## Translating KeOps cost strings

In GeomLoss, a custom cost is often a KeOps formula string in the variables
`X` and `Y`. In remblais, it is a Python function `cost(x, y)` that receives
two kernox formulas, `x` (indexed by `i`) and `y` (indexed by `j`), and
returns a scalar formula. The translation is mechanical:

| KeOps string | kernox formula |
| --- | --- |
| `X`, `Y` | `x`, `y` (the arguments of the function) |
| `IntCst(3)`, `Var(...)` constants | Python numbers: `3`, `0.5`, ... |
| `X + Y`, `X - Y`, `X * Y`, `X / Y`, `-X` | the same operators |
| `Square(F)`, `Pow(F, 3)`, `Powf(F, G)` | `F ** 2` (or `kp.square(F)`), `F ** 3`, `F ** G` |
| `Inv(F)` | `1 / F` |
| `Exp`, `Log`, `Sqrt`, `Rsqrt`, `Sin`, `Cos`, `Asin`, `Acos`, `Atan`, `Abs`, `Sign` | `kp.exp`, `kp.log`, `kp.sqrt`, `kp.rsqrt`, `kp.sin`, `kp.cos`, `kp.asin`, `kp.acos`, `kp.atan`, `abs`, `kp.sign` |
| `Clamp(F, a, b)` | `kp.clamp(F, a, b)` |
| `Min(F, G)`, `Max(F, G)` (element-wise) | `kp.minimum(F, G)`, `kp.maximum(F, G)` |
| `Elem(X, k)` | `x[k]` |
| `Extract(X, k, d)` | `x[k:k + d]` |
| `Concat(F, G)` | `kp.cat(F, G)` |
| `Sum(F)` | `F.sum()` |
| `(X\|Y)` | `x.dot(y)` |
| `Norm2(F)`, `SqNorm2(F)` | `F.norm2()`, `F.sqnorm2()` |
| `SqDist(X, Y)` | `x.sqdist(y)` |
| `Normalize(F)` | `F.normalize()` |
| `WeightedSqNorm(S, F)`, `WeightedSqDist(S, X, Y)` | `F.weightedsqnorm(S)`, `x.weightedsqdist(y, S)` (the weights come last) |
| `Step(F)`, `ReLU(F)` | `F.step()`, `F.relu()` |
| `IfElse(F, A, B)` | `F.ifelse(A, B)`, or `kp.where(F >= 0, A, B)` |

The full list is in [kernox operations](https://keckj.github.io/kernox/operations/).
A few costs, before and after:

```python
import kernox as kp
from remblais import SamplesLoss

# cost="Norm2(X-Y)"
cost = lambda x, y: (x - y).norm2()

# cost="Asin(Norm2(Normalize(X)-Normalize(Y))/IntCst(2))": half the angle between directions
cost = lambda x, y: kp.asin((x.normalize() - y.normalize()).norm2() / 2)

# cost=f"Log(Clamp(1-IntCst({k})*(X|Y), IntCst(1e-15), IntCst(1e15)))"
k = 0.5
cost = lambda x, y: kp.log(kp.clamp(1 - k * x.dot(y), 1e-15, 1e15))

# cost="SqNorm2(Extract(X,0,2) - Extract(Y,0,2)) + Elem(X,2) + Elem(Y,2)"
cost = lambda x, y: (x[0:2] - y[0:2]).sqnorm2() + x[2] + y[2]

loss = SamplesLoss("sinkhorn", blur=0.05, cost=cost)
```

Since the cost is plain Python, intermediate variables, loops over
components and helper functions all work. Python numbers, such as `k` above,
are baked into the compiled kernel.

To reuse a cost across `SamplesLoss` instances without compiling it again,
wrap it in a `remblais.Cost` with a hashable key that identifies it, including
its constants: `Cost(("log_one_minus_kxy", k), cost)`. Two costs with the same
key share their compiled reductions.

## Deliberate differences

These are the places where remblais does not reproduce GeomLoss 0.3.1 on
purpose, usually because GeomLoss has a bug there.

1. **Gradient of unbalanced transport.** GeomLoss defines the correction
   of the unbalanced gradient (weight $\rho + \varepsilon$) in the `backward` method of an
   `nn.Module`, which PyTorch never calls. Its effective gradient uses
   $\rho + \varepsilon/2$ and is off by about $\varepsilon/(2\rho)$ (3 % for $\rho = 1$, $\varepsilon = 0.25$). remblais
   applies the intended correction with a `torch.autograd.Function`: at
   convergence, the gradients agree with finite differences to 1e-9. Values
   and potentials are unchanged, bit for bit.
2. **`"hausdorff"` without a kernel.** GeomLoss routes `"hausdorff"` to the
   kernel norms, so the call raises `KeyError` without a `kernel`. remblais
   computes the documented Hausdorff divergence; see
   [Hausdorff divergence](guides/hausdorff.md). With a `kernel`, the
   behavior is the same as in GeomLoss.
3. **Grad mode.** GeomLoss forces `torch.set_grad_enabled(True)` at the end
   of its Sinkhorn loop, even when called under `torch.no_grad()`. remblais
   restores the state of the caller.
4. **Potentials of kernel losses with `multiscale`.** GeomLoss returns them
   in the order of the sorted clusters; remblais returns them in the order of
   the input points, as GeomLoss already does for Sinkhorn. Likewise,
   `"energy"` or `truncate=None` with `multiscale` return a scalar instead of
   a tensor of shape `(1,)`.
5. **Unbalanced multiscale.** GeomLoss reuses the name `eps` by mistake
   while looking for the coarse-to-fine jump, so its final value uses
   $\rho + \varepsilon'/2$ with $\varepsilon' \neq \texttt{blur}^p$. remblais uses $\varepsilon = \texttt{blur}^p$.
6. **Zero distances.** Like GeomLoss with KeOps, the `online` and
   `multiscale` backends compute $\|x - y\| = 0$ exactly, with a zero gradient at
   $x = y$; the `tensorized` backend keeps the 1e-8 floor of GeomLoss under the
   square root.
7. **Grid clustering.** `grid_cluster` labels the cells by a dense rank
   obtained by sorting, in lexicographic order like PyKeOps. It has no
   $D \le 3$ limit and no collisions of the 32-bit integer encoding of PyKeOps.
   The centroid of a cluster with a single point is that point, exactly.
8. **Dense fallback of multiscale.** When the clusters are so small that the
   kept blocks would cost more than $0.75 \times N \cdot M$ pair evaluations (counting
   each cluster as whole tiles of 128 rows), the fine level is computed on
   all pairs, without truncation; the coarse levels are those of GeomLoss.
   No term is dropped, so the result is at least as accurate. This happens
   typically in 3D with few points per cluster. Set
   `REMBLAIS_DENSE_RATIO=inf` to always truncate, as GeomLoss does.
9. **Hard C-transform with `p=1`.** `C_transform` on grids raises
   `NotImplementedError` for `p=1` in GeomLoss (an `if` that should be an
   `elif`); remblais computes it.
10. **Unimplemented `ot` skeletons.** `ot.barycenter`,
    `ot.barycenter_sample`, `ot.solve_grid` and `ot.barycenter_grid` have no
    working body in GeomLoss 0.3.1 (they fail with a `NameError`); remblais
    raises an explicit `NotImplementedError`.

## Extensions

- **Batched custom costs.** With `backend="online"`, a custom cost works with
  batched inputs `(B, N, D)`; GeomLoss raises `ValueError`.
- **`fast_math`** and **`flush_denormals`.** New `SamplesLoss` arguments,
  described below.
- **`distributed`.** New `SamplesLoss` argument: the softmins are split
  across the processes of a `torch.distributed` group. See
  [Several GPUs or nodes](guides/distributed.md).

## Fast math

PyKeOps compiles its kernels with `-use_fast_math` by default, so the
reductions of GeomLoss use an approximate `exp` on NVIDIA GPUs. remblais does
the same by default, and makes it explicit with the `fast_math` argument of
`SamplesLoss`:

- `fast_math=None` (the default): approximate `exp` and `log` in the kernox
  reductions on NVIDIA GPUs in FP32; IEEE arithmetic everywhere else (FP64,
  FP16, AMD, Intel, CPU).
- `fast_math=False`: IEEE arithmetic everywhere. Results are then
  reproducible bit for bit from one run to the next.
- `fast_math=True`: asks kernox for fast math explicitly. It has no effect in
  FP64 and on the CPU, and is refused by the kernox backends that do
  not offer it.

The relative error of the approximate `exp` is at most $2^{-21} + |x| \cdot 2^{-24}$
(the kernox contract). `sinkhorn_divergence` on images, `ImagesBarycenter`
and the `ot` module always use IEEE arithmetic.

```python
import torch
from remblais import SamplesLoss

device = "cuda"
x = torch.rand(1_000, 3, device=device)
y = torch.rand(1_000, 3, device=device)

ieee = SamplesLoss("sinkhorn", blur=0.05, fast_math=False)
print(ieee(x, y))
```


## Subnormal numbers

Subnormal numbers are the tiny values below about $10^{-38}$ in FP32 ($10^{-308}$ in
FP64). An operation that produces one is many times slower on CPUs, and the
Gaussian kernels and softmins of remblais produce many when the blur is small.
They never change a loss or a gradient in practice: a softmin sums terms whose
largest is 1, and a kernel norm holds terms equal to 1 ($k(x, x)$).

PyKeOps' `-use_fast_math` also flushes subnormal numbers to zero on NVIDIA
GPUs. remblais does the same by default, with the `flush_denormals` argument of
`SamplesLoss`:

- `flush_denormals=None` (the default): subnormal numbers are flushed to zero
  in the kernox reductions on the CPU and on NVIDIA GPUs in FP32; they are
  kept elsewhere (AMD, Intel GPUs, FP64 on NVIDIA GPUs).
- `flush_denormals=False`: kept everywhere, as IEEE arithmetic requires.
- `flush_denormals=True`: asks kernox to flush them explicitly. It has no
  effect in FP64 on NVIDIA GPUs, and is refused by the kernox backends that
  do not offer it.
