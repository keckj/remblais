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
  conventions, batches, `potentials=True`, the `backend="auto"` rule, custom
  costs given as KeOps formula strings, custom kernels, explicit cluster
  labels, and the messages of `verbose=True`. Inputs of invalid shapes raise
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
- Recompilations when ε changes: in remblais, ε and the blur are parameters
  of the compiled reduction, so the whole ε-scaling loop, every call and
  every batch reuse one kernel.

## Things that work differently

- **GPU vendors.** remblais runs on NVIDIA, AMD and Intel GPUs, wherever
  kernox does.
- **CPU.** kernox has no native CPU backend. On CPU tensors, the `online` and
  `multiscale` backends use the kernox oracle, a slow PyTorch reference: fine
  for tests, not for speed. `tensorized` is unchanged.
- **Custom costs as Python functions.** With the `online` and `multiscale`
  backends, a function given as `cost` (or as `kernel`) receives kernox
  formulas, not PyKeOps `LazyTensor`s; see [Concepts](concepts.md#costs).
  KeOps formula strings such as `"Norm2(X-Y)"` keep working, with every
  backend.
- **`LazyTensor`.** The `ot` module uses `remblais.lazy.LazyTensor`, a small
  kernox-based replacement for the PyKeOps `LazyTensor`. It covers what
  `geomloss.ot` needs: i and j variables and parameters, `+ − × ÷`, `**2`,
  `exp`, `log`, `sqrt`, `abs`, `sum(-1)`, `(x|y)`, the `sum`, `logsumexp`,
  `min`, `max` and `arg*` reductions, `@`, `.T` and NumPy inputs. Python
  numbers become parameters of the compiled reduction.

## Deliberate differences

These are the places where remblais does not reproduce GeomLoss 0.3.1 on
purpose, usually because GeomLoss has a bug there.

1. **Gradient of unbalanced transport.** GeomLoss defines the correction
   of the unbalanced gradient (weight ρ + ε) in the `backward` method of an
   `nn.Module`, which PyTorch never calls. Its effective gradient uses
   ρ + ε/2 and is off by about ε/(2ρ) (3 % for ρ = 1, ε = 0.25). remblais
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
   ρ + ε′/2 with ε′ ≠ blurᵖ. remblais uses ε = blurᵖ.
6. **Zero distances.** Like GeomLoss with KeOps, the `online` and
   `multiscale` backends compute ‖x − y‖ = 0 exactly, with a zero gradient at
   x = y; the `tensorized` backend keeps the 1e-8 floor of GeomLoss under the
   square root.
7. **Grid clustering.** `grid_cluster` labels the cells by a dense rank
   obtained by sorting, in lexicographic order like PyKeOps. It has no
   D ≤ 3 limit and no collisions of the 32-bit integer encoding of PyKeOps.
   The centroid of a cluster with a single point is that point, exactly.
8. **Dense fallback of multiscale.** When the clusters are so small that the
   kept blocks would cost more than 0.75 × N · M pair evaluations (counting
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
- **`fast_math`.** A new `SamplesLoss` argument, described below.

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
  FP64 and on the CPU oracle, and is refused by the kernox backends that do
  not offer it.

The relative error of the approximate `exp` is at most 2⁻²¹ + |x| · 2⁻²⁴
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
