"""Weighted Hausdorff divergence (``loss="hausdorff"`` without a kernel).

GeomLoss v0.3.1 documents ``"hausdorff"`` ("interpolates between ICP, blur = 0,
and a kernel distance, blur = +∞") but routes this name to the kernel norms:
without a ``kernel`` argument, the call raises ``KeyError`` there. remblais
keeps this routing when ``kernel`` is given, and otherwise implements the
definition of Feydy et al. (AISTATS 2019, §2):

    a = softmin_ε(α), b = softmin_ε(β),  softmin_ε(μ)(z) = −ε log ∫ exp(−C(z, w)/ε) dμ(w)
    H_ε(α, β) = ½ ⟨α, b − a⟩ + ½ ⟨β, a − b⟩,   ε = blur^p.

As ε → 0, H_ε tends to ½ ⟨α, d(·, supp β)⟩ + ½ ⟨β, d(·, supp α)⟩ (symmetric
ICP). The returned potentials are (b − a on x, a − b on y), so that
H = ½(⟨α, F⟩ + ⟨β, G⟩). The gradient is exact (no extrapolation).
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
