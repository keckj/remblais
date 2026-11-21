# Hausdorff divergence

The Hausdorff divergence measures how far each point of one cloud is from the
other cloud, in both directions. At small blur it is the symmetric
nearest-neighbour distance used by ICP (iterative closest point)
registration; at large blur it becomes a kernel distance. It does not try to
match masses: it only cares about supports, which makes it robust to uneven
sampling.

## Definition

remblais implements the divergence of [Feydy et al. (AISTATS 2019)](https://proceedings.mlr.press/v89/feydy19a.html), section 2. With
$\varepsilon = \texttt{blur}^p$ and the softmin

$$
\operatorname{softmin}_\varepsilon(\mu)(z) = -\varepsilon \log \sum_w \mu(w) \exp(-C(z, w) / \varepsilon),
$$

let $a = \operatorname{softmin}_\varepsilon(\alpha)$ and $b = \operatorname{softmin}_\varepsilon(\beta)$. Then

$$
H_\varepsilon(\alpha, \beta) = \frac{1}{2} \langle \alpha, b - a \rangle + \frac{1}{2} \langle \beta, a - b \rangle.
$$

As the blur goes to 0, $H_\varepsilon$ tends to $\frac{1}{2} \langle \alpha, d(\cdot, \operatorname{supp} \beta) \rangle + \frac{1}{2} \langle \beta, d(\cdot, \operatorname{supp} \alpha) \rangle$,
where $d(z, S)$ is the cost from $z$ to the nearest point of $S$: the symmetric ICP
cost.

## Usage

```python
import torch
from remblais import SamplesLoss

device = "cuda"

x = torch.rand(2_000, 2, device=device)
y = torch.rand(1_000, 2, device=device) + 0.1

hausdorff = SamplesLoss("hausdorff", p=2, blur=0.01)
print(hausdorff(x, y))
```

The gradient is exact (no iterations, no extrapolation), so it suits
registration loops:

```python
x.requires_grad_()
[g] = torch.autograd.grad(hausdorff(x, y), [x])
print(g.shape)
```

With `potentials=True`, the loss returns $F = b - a$ on `x` and $G = a - b$ on
`y`, so that $H = \frac{1}{2}(\langle \alpha, F \rangle + \langle \beta, G \rangle)$.

## Difference with GeomLoss

GeomLoss 0.3.1 documents `"hausdorff"` but routes it to the kernel norms:
without a `kernel` argument, the call raises `KeyError`. remblais keeps this
routing when `kernel` is given (see [custom kernels](kernels.md#custom-kernels))
and computes the divergence above otherwise.

The `multiscale` backend has no truncation for this loss: it computes the
exact result with the `online` backend.
