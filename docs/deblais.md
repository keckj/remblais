# remblais and deblais

In 1781, Gaspard Monge asked how to move a pile of earth, the *déblais*, into
a hole of the same volume, the *remblais*, at the lowest cost. The two words
name the two ends of the same transport, and two twin libraries:

| | **remblais** (this project) | **[deblais](https://github.com/keckj/deblais)** |
| --- | --- | --- |
| Problem | entropic optimal transport: Sinkhorn divergences, and kernel norms (MMD), Hausdorff divergences | exact optimal transport |
| Algorithm | Sinkhorn iterations with $\varepsilon$-scaling, multiscale | network simplex, multiscale, domain decomposition |
| Result | a smooth loss, with a blur parameter that trades accuracy for speed | an optimal vertex: sparse plan, exact cost, dual certificate |
| Runs on | NVIDIA, AMD and Intel GPUs, several GPUs or nodes, and CPUs | CPUs (OpenMP) |
| Cost functions | any cost or kernel formula | squared Euclidean |
| Data | point clouds in any dimension, images and volumes | point clouds, dimension $1 \le d \le 8$ |
| Interface | Python for PyTorch, drop-in replacement of GeomLoss | C++17 headers; Python for NumPy and PyTorch |
| Unbalanced transport | yes | no |

## Which one?

Use **remblais** when you need a loss to minimize, many times, on a GPU:

- training a model with a geometric loss, where a smooth, positive definite
  divergence behaves better than a raw transport cost;
- millions of points per evaluation, in single precision;
- costs other than the squared distance, unbalanced measures, images and
  volumes, high dimension.

Use **deblais** when you need the transport plan itself, or the true
Wasserstein distance:

- an assignment or a matching between two sets of points, with a proof of
  optimality;
- a reference value to validate an approximate solver;
- a sparse plan to interpolate between two measures (displacement
  interpolation), or a transport map without blur;
- exact integer flows.

The two meet when the blur of the Sinkhorn divergence goes to zero: with
$\varepsilon = \texttt{blur}^p$, the divergence $S_\varepsilon(\alpha, \beta)$
converges to the transport cost, $\frac{1}{2} W_2^2(\alpha, \beta)$ for
`p=2` in the convention of remblais, half the value $W_2^2$ that deblais
returns. The blur and the $\varepsilon$ schedule are described in
[Concepts](concepts.md).

## Together

Both libraries take PyTorch tensors and can share a process. A typical use is
to check, on a sample, how far an entropic loss is from the exact transport
cost:

<!-- skip-test -->
```python
import torch
import deblais
import remblais

x, y = torch.rand(50_000, 3), torch.rand(50_000, 3)

loss = remblais.SamplesLoss("sinkhorn", p=2, blur=0.01)  # cost |x - y|^2 / 2
entropic = 2 * loss(x, y)

exact = deblais.wasserstein2(x, y)                       # W2^2, squared distances

print(float(entropic), float(exact))
```

!!! note
    remblais follows the convention of GeomLoss: with `p=2` its cost is
    $\frac{1}{2} |x - y|^2$, half the cost of deblais. The factor 2 above
    accounts for it.

- deblais: <https://github.com/keckj/deblais>
- Documentation of deblais: <https://keckj.github.io/deblais/>
- The same comparison, seen from deblais:
  <https://keckj.github.io/deblais/remblais/>
