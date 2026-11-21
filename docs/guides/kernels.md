# Kernel MMD

Kernel norms, also known as maximum mean discrepancies (MMD), compare two
measures through a kernel $k$:

$$
\text{Loss}(\alpha, \beta) = \frac{1}{2}\langle \alpha, k \star \alpha \rangle + \frac{1}{2}\langle \beta, k \star \beta \rangle - \langle \alpha, k \star \beta \rangle
$$

They need a single pass over the $N \times M$ pairs (no iterations), which makes
them the cheapest losses of remblais. Their weakness is the gradient: far
from the target, a Gaussian kernel barely pulls the points.

## Built-in kernels

```python
import torch
from remblais import SamplesLoss

device = "cuda"

x = torch.randn(2_000, 3, device=device)
y = torch.randn(2_500, 3, device=device) + 0.5

for name in ("gaussian", "laplacian", "energy"):
    loss = SamplesLoss(name, blur=0.5)
    print(name, loss(x, y).item())
```

- `"gaussian"`: $k(x, y) = \exp(-\|x - y\|^2 / 2\sigma^2)$, with $\sigma = \texttt{blur}$;
- `"laplacian"`: $k(x, y) = \exp(-\|x - y\| / \sigma)$;
- `"energy"`: $k(x, y) = -\|x - y\|$, the energy distance. It has no scale, so
  `blur` is ignored.

Weights, batches and gradients work as for the Sinkhorn divergence:

```python
a = torch.rand(2_000, device=device)
a = a / a.sum()
b = torch.full((2_500,), 1 / 2_500, device=device)
x.requires_grad_()
L = SamplesLoss("gaussian", blur=0.5)(a, x, b, y)
[g] = torch.autograd.grad(L, [x])
```

## Choosing the blur

`blur` sets the scale at which the kernel compares the measures. Details
smaller than the blur are smoothed out; structures much larger than the blur
are seen only through their tails. A good starting point is the typical
distance between the two clouds.

## Custom kernels

A custom kernel goes through `loss="hausdorff"` with a `kernel` argument, as
in GeomLoss. With the `tensorized` backend, the kernel receives tensors of
shape `(B, N, D)` and `(B, M, D)` and returns the `(B, N, M)` kernel matrix:

```python
def cauchy(x, y, blur=0.5, **kwargs):
    d2 = ((x.unsqueeze(-2) - y.unsqueeze(-3)) ** 2).sum(-1)
    return 1 / (1 + d2 / blur**2)

loss = SamplesLoss("hausdorff", kernel=cauchy, blur=0.5, backend="tensorized")
print(loss(x.detach(), y))
```

With the `online` backend, it receives kernox formulas instead (`x` indexed
by `i`, `y` by `j`), and `blur` as a number:

```python
def cauchy_formula(x, y, blur):
    d2 = ((x - y) ** 2).sum()
    return 1 / (1 + d2 / blur**2)

loss = SamplesLoss("hausdorff", kernel=cauchy_formula, blur=0.5, backend="online")
print(loss(x.detach(), y))
```

The two values agree up to rounding. The kernel is compiled into a single
fused reduction for the given blur.

## Large clouds

`backend="online"` never stores the $N \times M$ kernel matrix. In dimension 1 to 3,
`backend="multiscale"` also skips the blocks of pairs farther apart than
`truncate` times the blur (default 5). Pass `truncate=None` to keep every
pair.
