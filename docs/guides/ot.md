# The ot API

`remblais.ot` is the port of `geomloss.ot`, the newer solver interface of
GeomLoss 0.3.1. Where `SamplesLoss` returns a loss value for training, the
`ot` solvers return a result object with the transport plan, the dual
potentials, the marginals and the transported positions, computed on demand.

Like GeomLoss, these solvers accept NumPy arrays as well as PyTorch tensors,
and return results of the same kind as their inputs. GeomLoss marks this
interface as still in development; remblais follows it as it is in 0.3.1.

## Transport between point clouds

`ot.solve_sample` takes the coordinates of the two clouds and computes the
cost on the fly, without storing the $N \times M$ matrix: in remblais, the
reductions run on kernox.

```python
from remblais import ot

solution = ot.solve_sample(
    X_a=[[0, 0], [0, 2]],
    X_b=[[2, 1], [2, 2]],
    reg=0.001,
    max_iter=100,
)
print(solution.plan)               # [[0.5 0. ] [0.  0.5]]
print(f"{solution.value:.3f}")     # 4.501
```

With PyTorch tensors on a GPU, the same call returns tensors:

```python
import torch

device = "cuda"

X = torch.rand(1_000, 2, device=device)
Y = torch.rand(1_500, 2, device=device) + 0.3

result = ot.solve_sample(X, Y, blur=0.05, max_iter=100)
print(result.value)
print(result.potential_a.shape, result.potential_b.shape)   # (1000,) and (1500,)
```

`blur` and `reach` are the same lengths as for `SamplesLoss`, but the `ot`
module uses the cost $C(x, y) = \|x - y\|^2$, without the $\frac{1}{2}$ of `SamplesLoss`
(see [Concepts](../concepts.md#blur-reach-and-scaling)). The same `blur`
therefore gives $\varepsilon = 2\,\texttt{blur}^2$ here instead of $\varepsilon = \texttt{blur}^2$, and the same kernel
$\exp(-C/\varepsilon)$. `reg` sets $\varepsilon$ directly. Likewise, `reach` ($\rho = 2\,\texttt{reach}^2$) and
`unbalanced` ($\rho$ directly) enable unbalanced transport.

## Reading the result

The attributes are computed the first time you read them, then cached:

- `value`: the regularized transport cost;
- `potential_a`, `potential_b`: the dual potentials $f$ and $g$;
- `plan`: the dense $N \times M$ transport plan (avoid it for large clouds);
- `lazy_plan`: the same plan as a lazy, kernox-backed `LazyTensor`, never
  stored;
- `plan_operator`: the plan as a linear operator, to multiply with `@`
  without storing it;
- `marginal_a`, `marginal_b`: the marginals of the plan, close to the input
  weights in balanced transport.

The plan operator gives, for instance, the barycentric map: where each point
of `X` is sent, on average.

```python
moved = (result.plan_operator @ Y) / result.marginal_a[:, None]
print(moved.shape)                 # (1000, 2)
print(result.marginal_a.sum())     # ≈ 1
```

The `a_to_b` and `b_to_a` attributes exist in GeomLoss 0.3.1 but are not
implemented there (they return `None`), and remblais keeps them as they are.

## Transport with a cost matrix

When the cost matrix is small enough to store, or is not a function of
coordinates, use `ot.solve` with the matrix itself:

```python
C = torch.cdist(X, Y) ** 2 / 2     # (1000, 1500)
result = ot.solve(C, reg=0.01, max_iter=100)
print(result.value, result.plan.shape)
```

`ot.solve_batch` does the same for a batch of matrices `(B, N, M)`.

## What is not implemented

GeomLoss 0.3.1 ships `ot.barycenter`, `ot.barycenter_sample`,
`ot.solve_grid` and `ot.barycenter_grid` as skeletons whose bodies refer to
undefined names, so calling them fails with a `NameError`. remblais raises an
explicit `NotImplementedError` instead. `ot.solve_sample_batch` raises
`NotImplementedError` in both libraries. For images, use
[`remblais.sinkhorn_divergence` and `ImagesBarycenter`](images.md).
