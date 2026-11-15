# Derived from GeomLoss v0.3.1 (geomloss/_legacy/sinkhorn_images.py, utils.py),
# Copyright (c) 2019 Jean Feydy, MIT License (see THIRD_PARTY.md).
"""Sinkhorn divergence between images and volumes (regular grids), multiscale.

The Gibbs kernel of a separable cost on a grid factorizes along the axes: each
softmin is a sequence of 1D LOGSUMEXP reductions (one per axis), run by kernox
as batched reductions (one N×N reduction per line of the image).
"""

import numpy as np
import torch
from torch.nn.functional import avg_pool2d, avg_pool3d, interpolate

import kernox as kp

from . import _kp
from .sinkhorn_divergence import scaling_parameters, sinkhorn_cost, sinkhorn_loop

BATCH, CHANNEL, HEIGHT, WIDTH, DEPTH = 0, 1, 2, 3, 4


def dimension(I):
    """2 for images (B, K, N, N), 3 for volumes (B, K, N, N, N)."""
    return I.dim() - 2


subsample = {2: lambda x: 4 * avg_pool2d(x, 2), 3: lambda x: 8 * avg_pool3d(x, 2)}
upsample_mode = {2: "bilinear", 3: "trilinear"}


def pyramid(I):
    """[1×1, 2×2, …, N×N] by summing 2×2 (or 2×2×2) blocks."""
    D = dimension(I)
    I_s = [I]
    for _ in range(int(np.log2(I.shape[HEIGHT]))):
        I = subsample[D](I)
        I_s.append(I)
    I_s.reverse()
    return I_s


def upsample(I):
    return interpolate(I, scale_factor=2, mode=upsample_mode[dimension(I)], align_corners=False)


def log_dens(α):
    α_log = α.log()
    α_log[α <= 0] = -10000.0
    return α_log


# ---------------------------------------------------------------------------------
# Réductions 1D batchées
# ---------------------------------------------------------------------------------
def _line_plan(kind, p, dtype, backend):
    """1D LSE (soft) or MAX (hard): (h_j − s·|x_i − x_j|^p), s a parameter."""

    def build():
        xi, xj, h, s = kp.Vi("x", 1), kp.Vj("y", 1), kp.Vj("h", 1), kp.Param("s")
        d = xi - xj
        c = kp.abs(d) if p == 1 else d * d
        f = h - s * c
        return f.logsumexp(over="j") if kind == "lse" else f.max(over="j")

    return _kp._plan(("grid", kind, p), build, dtype, backend)


def _lines(kind, h, p, s):
    """Reduction along the last axis of ``h`` (…, N); x = (0, …, N−1)/N."""
    shape = h.shape
    N = shape[-1]
    run = _line_plan(kind, p, _kp.dtype_name(h), _kp.backend_for(h.device))
    x = (torch.arange(N, device=h.device, dtype=h.dtype) / N).view(1, N, 1)
    rows = h.reshape(-1, N, 1)
    out = run(x=x.expand(rows.shape[0], N, 1), y=x.expand(rows.shape[0], N, 1),
              h=rows.contiguous(), s=s)
    return out.view(shape)


def _separable(kind, h, p, s):
    D = dimension(h)
    h = _lines(kind, h, p, s)
    if D == 2:
        h = _lines(kind, h.transpose(-1, -2), p, s).transpose(-1, -2)
    elif D == 3:
        h = _lines(kind, h.transpose(-1, -2), p, s).transpose(-1, -2)
        h = _lines(kind, h.transpose(-1, -3), p, s).transpose(-1, -3)
    return h


def softmin_grid(eps, C_xy, h_y):
    """Separable soft C-transform on the grid [0, 1)^D of step 1/N.

    ``C_xy`` is the exponent p: cost |x − y|²/2 (p = 2) or Σ_d |x_d − y_d| (p = 1,
    the L1 distance, the only separable version; it is the one of GeomLoss).
    """
    p = C_xy
    if p not in (1, 2):
        raise NotImplementedError()
    s = 1.0 / eps if p == 1 else 1.0 / (2 * eps)
    return -eps * _separable("lse", h_y, p, s)


def C_transform(G, tau=1, p=2):
    """The "hard" C-transform F(x_i) = max_j G(x_j) − |x_i − x_j|^p / (p·τ), with pixels of step 1.

    G has shape (B, N), (B, N, N) or (B, N, N, N). GeomLoss v0.3.1 raises
    ``NotImplementedError`` for p = 1 (``if`` instead of ``elif``); remblais
    computes both cases.
    """
    D = G.ndim - 1
    N = G.shape[1]
    if p not in (1, 2):
        raise NotImplementedError()
    # x = arange(N) sans division par N : on reporte le facteur N^p dans s.
    s = (N if p == 1 else N * N) / (tau if p == 1 else 2 * tau)
    G = G.unsqueeze(1)  # (B, 1, …) : même découpage en lignes que les images
    out = _lines("max", G, p, s)
    if D >= 2:
        out = _lines("max", out.transpose(-1, -2), p, s).transpose(-1, -2)
    if D == 3:
        out = _lines("max", out.transpose(-1, -3), p, s).transpose(-1, -3)
    return out.squeeze(1)


# ---------------------------------------------------------------------------------
# Divergence de Sinkhorn entre images
# ---------------------------------------------------------------------------------
def extrapolate(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_fine):
    return upsample(f_ba)


def kernel_truncation(C_xy, C_yx, C_xy_fine, C_yx_fine, f_ba, g_ab, eps, truncate=None, cost=None, verbose=False):
    return C_xy_fine, C_yx_fine


def sinkhorn_divergence(a, b, p=2, blur=None, reach=None, axes=None, scaling=0.5, cost=None, debias=True,
                        potentials=False, verbose=False, **kwargs):
    """Sinkhorn divergence between densities on a regular grid, multiscale.

    Exported as ``remblais.sinkhorn_divergence``. The pixels sit on the grid
    [0, 1)ᴰ with a step of 1/N.

    :param a: source densities, (B, K, N, N) for images or (B, K, N, N, N) for
        volumes; N must be a power of 2.
    :param b: target densities, same shape as ``a``.
    :param p: exponent of the cost: |x − y|²/2 for p = 2, the L1 distance for
        p = 1.
    :param blur: length scale σ, ε = blurᵖ; one pixel (1/N) when ``None``.
    :param reach: ``None`` for balanced transport, or a length scale for
        unbalanced transport.
    :param scaling: ratio of the ε-scaling, in [0.5, 1).
    :param debias: subtract the self-transport terms.
    :param potentials: return the dual potentials (F, G) instead of the value.
    :param verbose: print the pyramid and the ε-scaling schedule.
    :return: a (B,) tensor, summed over the K channels, or (F, G).
    """
    if blur is None:
        blur = 1 / a.shape[-1]

    a_s, b_s = pyramid(a)[1:], pyramid(b)[1:]
    a_logs = list(map(log_dens, a_s))
    b_logs = list(map(log_dens, b_s))
    depth = len(a_logs)
    if cost is None:
        C_s = [p] * depth
    else:
        raise NotImplementedError()

    if scaling < 0.5:
        raise ValueError(f"Scaling value of {scaling} is too small: please use a number in [0.5, 1).")
    diameter, eps, eps_list, rho = scaling_parameters(None, None, p, blur, reach, 1, scaling)

    pyramid_scales = [diameter / a.shape[-1] for a in a_s]
    if verbose:
        print("Pyramid scales:", pyramid_scales)
    current_scale = pyramid_scales.pop(0)
    jumps = []
    for i, eps_i in enumerate(eps_list[1:]):
        if current_scale**p > eps_i:
            jumps.append(i + 1)
            current_scale = pyramid_scales.pop(0)
    if verbose:
        print("Temperatures: ", eps_list)
        print("Jumps: ", jumps)
    assert len(jumps) == len(a_s) - 1, "There's a bug in the multicale pre-processing..."

    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(softmin_grid, a_logs, b_logs, C_s, C_s, C_s, C_s, eps_list, rho,
                                           jumps=jumps, kernel_truncation=kernel_truncation,
                                           extrapolate=extrapolate, debias=debias)
    return sinkhorn_cost(eps, rho, a, b, f_aa, g_bb, g_ab, f_ba, batch=True, debias=debias, potentials=potentials)
