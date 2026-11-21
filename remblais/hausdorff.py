r"""Weighted Hausdorff divergence (``loss="hausdorff"`` without a kernel).

GeomLoss v0.3.1 documents ``"hausdorff"`` ("interpolates between ICP, $\texttt{blur} = 0$,
and a kernel distance, $\texttt{blur} = +\infty$") but routes this name to the kernel norms:
without a ``kernel`` argument, the call raises ``KeyError`` there. remblais
keeps this routing when ``kernel`` is given, and otherwise implements the
definition of Feydy et al. (AISTATS 2019, §2):

$$
a = \operatorname{softmin}_\varepsilon(\alpha), \quad b = \operatorname{softmin}_\varepsilon(\beta), \quad
\operatorname{softmin}_\varepsilon(\mu)(z) = -\varepsilon \log \int \exp(-C(z, w)/\varepsilon)\, d\mu(w)
$$

$$
H_\varepsilon(\alpha, \beta) = \frac{1}{2} \langle \alpha, b - a \rangle + \frac{1}{2} \langle \beta, a - b \rangle, \quad \varepsilon = \texttt{blur}^p.
$$

As $\varepsilon \to 0$, $H_\varepsilon$ tends to $\frac{1}{2} \langle \alpha, d(\cdot, \operatorname{supp} \beta) \rangle + \frac{1}{2} \langle \beta, d(\cdot, \operatorname{supp} \alpha) \rangle$ (symmetric
ICP). The returned potentials are ($b - a$ on $x$, $a - b$ on $y$), so that
$H = \frac{1}{2}(\langle \alpha, F \rangle + \langle \beta, G \rangle)$. The gradient is exact (no extrapolation).
"""

from . import _kp
from .costs import resolve_cost
from .sinkhorn_divergence import log_weights
from .sinkhorn_samples import softmin_tensorized
from .utils import scal


def _hausdorff(softmin, a, x, b, y, eps, potentials, batch):
    la, lb = log_weights(a), log_weights(b)
    a_x, b_x = softmin(eps, x, x, la), softmin(eps, x, y, lb)
    a_y, b_y = softmin(eps, y, x, la), softmin(eps, y, y, lb)
    F, G = b_x - a_x, a_y - b_y
    if potentials:
        return F, G
    return 0.5 * (scal(a, F, batch=batch) + scal(b, G, batch=batch))


def hausdorff_tensorized(a, x, b, y, p=2, blur=0.05, cost=None, potentials=False, **kwargs):
    dense = resolve_cost(cost, p, "tensorized").dense

    def softmin(eps, u, v, h):
        return softmin_tensorized(eps, dense(u, v), h)

    return _hausdorff(softmin, a, x, b, y, blur**p, potentials, batch=True)


def hausdorff_online(a, x, b, y, p=2, blur=0.05, cost=None, potentials=False, **kwargs):
    c = resolve_cost(cost, p, "online")
    batch = x.dim() == 3

    def softmin(eps, u, v, h):
        if batch and u.shape[0] == 1:
            return -eps * _kp.lse(c, u[0], v[0], h[0], 1.0 / eps).unsqueeze(0)
        return -eps * _kp.lse(c, u, v, h, 1.0 / eps)

    return _hausdorff(softmin, a, x, b, y, blur**p, potentials, batch=batch)


def hausdorff_multiscale(a, x, b, y, potentials=False, **kwargs):
    """No multiscale truncation for Hausdorff: exact online computation."""
    kwargs.pop("labels_x", None), kwargs.pop("labels_y", None)
    return hausdorff_online(a, x, b, y, potentials=potentials, **kwargs)
