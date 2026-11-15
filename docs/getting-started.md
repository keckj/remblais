# Getting started

This page installs remblais and computes a first Sinkhorn divergence between
two point clouds, then its gradient.

## Requirements

- Python 3.10 or later, PyTorch, NumPy and SciPy.
- [kernox](https://github.com/keckj/kernox), the library of fused GPU
  reductions remblais is built on, and [symox](https://github.com/keckj/symox),
  the symbolic engine of kernox. Both are on PyPI.
- A GPU supported by kernox for speed: NVIDIA (CUDA), AMD (ROCm) or Intel
  (XPU). Without one, remblais still runs on the CPU, slowly (see
  [Running without a GPU](#running-without-a-gpu)).

## Install kernox

Follow the
[kernox installation guide](https://keckj.github.io/kernox/getting-started/):
it installs PyTorch from the index that matches your GPU, then kernox with the
matching extra (symox, the symbolic engine of kernox, comes from PyPI with
it). For an NVIDIA GPU, this comes down to:

```sh
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install "kernox[cuda]"
python -m kernox.diagnostics --check
```

The last command checks that kernox sees your GPU and can compile and run a
few reductions on it. For AMD and Intel GPUs, use the `rocm` or `xpu` extra
and the PyTorch index given in the kernox guide. Intel GPUs need one more step,
a SYCL runtime matching the host `icpx` compiler: see
[Intel GPUs](https://keckj.github.io/kernox/getting-started/#intel-gpus) in the
kernox guide.

## Install remblais

```sh
pip install remblais
```

From a source checkout, use an editable install with the test extra, which
also installs GeomLoss 0.3.1 as the reference the tests compare against:

```sh
pip install -e ".[test]"
pytest
```

## A first Sinkhorn divergence

We compare two clouds of points in 3D. Pick the device first: `"cuda"` for
NVIDIA and AMD GPUs, `"xpu"` for Intel GPUs.

```python
import torch
from remblais import SamplesLoss

device = "cuda"

x = torch.rand(2_000, 3, device=device)          # N points in [0, 1]^3
y = torch.rand(3_000, 3, device=device) + 0.5    # M points, shifted
```

`SamplesLoss` builds a loss function. Here, the debiased Sinkhorn divergence
for the cost ½‖x − y‖² (`p=2`), with a blur of 0.05: details smaller than the
blur are ignored.

```python
loss = SamplesLoss("sinkhorn", p=2, blur=0.05)
L = loss(x, y)
print(L)        # a scalar tensor, close to ½‖0.5 · (1, 1, 1)‖² = 0.375
```

Without weights, each point carries the same mass, 1/N in `x` and 1/M in `y`.
To give your own weights, pass four arguments: `loss(a, x, b, y)`, where `a`
and `b` are nonnegative vectors of sizes N and M that each sum to 1.

```python
a = torch.rand(2_000, device=device)
a = a / a.sum()
b = torch.full((3_000,), 1 / 3_000, device=device)
L = loss(a, x, b, y)
```

## Gradients

The loss is an ordinary differentiable PyTorch function of the positions and
the weights:

```python
x.requires_grad_()
L = loss(x, y)
[g] = torch.autograd.grad(L, [x])
print(g.shape)  # torch.Size([2000, 3])
```

`-g / a` is the direction in which each point of `x` should move to get closer
to `y`. The [gradient flow guide](guides/gradient-flow.md) uses it to
transport one cloud onto the other.

## Running without a GPU

kernox has no native CPU backend. On CPU tensors, remblais uses the dense
PyTorch implementation (`backend="tensorized"`) for small problems, and the
kernox *oracle*, a slow blockwise PyTorch reference, for the `online` and
`multiscale` backends. Results are correct, but do not expect GPU speed.
Every example of this documentation runs this way in the test suite.

## Next steps

- [Concepts](concepts.md) explains measures, costs, the blur, reach and
  scaling parameters, and the three backends.
- The how-to guides start with
  [Sinkhorn divergences between point clouds](guides/sinkhorn.md).
- Already using GeomLoss? Read [Coming from GeomLoss](geomloss.md).
