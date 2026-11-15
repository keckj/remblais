# Derived from GeomLoss v0.3.1 (geomloss/_legacy/kernel_samples.py),
# Copyright (c) 2019 Jean Feydy, MIT License (see THIRD_PARTY.md).
"""Kernel norms (MMD) between point clouds: energy, gaussian, laplacian, custom kernel.

    Loss(α, β) = ½⟨α, k⋆α⟩ + ½⟨β, k⋆β⟩ − ⟨α, k⋆β⟩

Backends: ``tensorized`` (dense Torch matrices), ``online`` (fused kernox SUM
reductions, 1/blur as a parameter), ``multiscale`` (block-sparse truncation
at ``truncate`` standard deviations, kernox ranges).
"""

from functools import partial

import numpy as np
import torch

import kernox as kp

from . import _kp
from .clustering import Clusters, from_matrix_or_dense, grid_cluster
from .utils import distances, max_diameter, scal, squared_distances


class DoubleGrad(torch.autograd.Function):
    """Identity in the forward pass, doubled gradient in the backward pass (symmetric ½⟨α, k⋆α⟩ terms)."""

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return 2 * grad_output


def double_grad(x):
    return DoubleGrad.apply(x)


# ---------------------------------------------------------------------------------
# Noyaux : formule kernox (s = 1/blur) et version dense (GeomLoss)
# ---------------------------------------------------------------------------------
def _norm(d):
    d2 = d.sqnorm2()
    return kp.where(d2 > 0, kp.sqrt(d2), 0.0)


def gaussian_kernel(x, y, blur=0.05, use_keops=False, ranges=None):
    C2 = squared_distances(x / blur, y / blur)
    return (-C2 / 2).exp()


def laplacian_kernel(x, y, blur=0.05, use_keops=False, ranges=None):
    return (-distances(x / blur, y / blur)).exp()


def energy_kernel(x, y, blur=None, use_keops=False, ranges=None):
    return -distances(x, y)


kernel_routines = {"gaussian": gaussian_kernel, "laplacian": laplacian_kernel, "energy": energy_kernel}

# Le paramètre s porte toute l'échelle (une multiplication par paire, pas par coordonnée).
kernel_formulas = {
    "gaussian": lambda x, y, s: kp.exp(-s * (x - y).sqnorm2()),  # s = 1/(2 blur²)
    "laplacian": lambda x, y, s: kp.exp(-s * _norm(x - y)),      # s = 1/blur
    "energy": lambda x, y, s: -_norm(x - y),
}
kernel_params = {
    "gaussian": lambda blur: 1.0 / (2.0 * blur * blur),
    "laplacian": lambda blur: 1.0 / blur,
    "energy": lambda blur: 1.0,
}


class Kernel:
    """Kernel: ``formula(x, y, s)`` (kernox), ``param(blur)`` → s, and ``dense(x, y, blur=...)``."""

    def __init__(self, key, formula, dense, param=None):
        self.key, self.formula, self.dense = key, formula, dense
        self.param = param or (lambda blur: 1.0 / blur)


def resolve_kernel(kernel, name, blur, online):
    if isinstance(kernel, Kernel):
        return kernel
    if kernel is None:
        if name is None:
            raise ValueError("loss='hausdorff' avec un noyau : argument 'kernel' requis")
        if name not in kernel_routines:
            raise ValueError(f"noyau {name!r} inconnu (gaussian, laplacian, energy)")
        return Kernel(("named", name), kernel_formulas[name], kernel_routines[name], kernel_params[name])
    # Noyau personnalisé : sur tenseurs (tensorized, signature GeomLoss) ou sur
    # formules kernox (online/multiscale), blur figé dans la formule.
    if online:
        return Kernel(("custom", kernel, float(blur)), lambda x, y, s: kernel(x, y, blur=blur), None)
    return Kernel(("custom", kernel), None, kernel)


def _kprod(K, x, y, b, blur, ranges=None):
    """Online Σ_j k(x_i, y_j) b_j; handles B = 1 without a batch dimension."""
    if x.dim() == 3 and x.shape[0] == 1 and ranges is None:
        return _kprod(K, x[0], y[0], b[0], blur).unsqueeze(0)
    run = _kp.kernel_plan(K, x.shape[-1], _kp.dtype_name(x), _kp.backend_for(x.device))
    kw = dict(x=x, y=y, b=b.unsqueeze(-1))
    if "s" in run.input_names:
        kw["s"] = K.param(blur)
    return _kp._call(run, ranges, **kw).squeeze(-1)


def kernel_loss(α, x, β, y, blur=0.05, kernel=None, name=None, potentials=False, use_keops=False,
                ranges_xx=None, ranges_yy=None, ranges_xy=None, **kwargs):
    K = resolve_kernel(kernel, name, blur, online=use_keops)
    if use_keops:
        a_x = _kprod(K, double_grad(x), x.detach(), α.detach(), blur, ranges_xx)
        b_y = _kprod(K, double_grad(y), y.detach(), β.detach(), blur, ranges_yy)
        b_x = _kprod(K, x, y, β, blur, ranges_xy)
        if potentials:
            a_y = _kprod(K, y, x, α, blur, None if ranges_xy is None else _kp.swap_axes(ranges_xy))
            return a_x - b_x, b_y - a_y
    else:
        K_xx = K.dense(double_grad(x), x.detach(), blur=blur, use_keops=False, ranges=None)
        K_yy = K.dense(double_grad(y), y.detach(), blur=blur, use_keops=False, ranges=None)
        K_xy = K.dense(x, y, blur=blur, use_keops=False, ranges=None)
        a_x = (K_xx @ α.detach().unsqueeze(-1)).squeeze(-1)
        b_y = (K_yy @ β.detach().unsqueeze(-1)).squeeze(-1)
        b_x = (K_xy @ β.unsqueeze(-1)).squeeze(-1)
        if potentials:
            a_y = (K_xy.transpose(-1, -2) @ α.unsqueeze(-1)).squeeze(-1)
            return a_x - b_x, b_y - a_y
    batch = x.dim() > 2
    return (0.5 * scal(double_grad(α), a_x, batch=batch) + 0.5 * scal(double_grad(β), b_y, batch=batch)
            - scal(α, b_x, batch=batch))


kernel_tensorized = partial(kernel_loss, use_keops=False)
kernel_online = partial(kernel_loss, use_keops=True)


def kernel_multiscale(α, x, β, y, blur=0.05, kernel=None, name=None, truncate=5, diameter=None,
                      cluster_scale=None, potentials=False, verbose=False, **kwargs):
    """Kernel truncated at ``truncate`` blur (blocks of clusters), on unbatched measures."""
    if truncate is None or name == "energy":
        out = kernel_online(α.unsqueeze(0), x.unsqueeze(0), β.unsqueeze(0), y.unsqueeze(0), blur=blur,
                            kernel=kernel, name=name, potentials=potentials)
        return (out[0][0], out[1][0]) if potentials else out[0]

    center = (x.mean(-2, keepdim=True) + y.mean(-2, keepdim=True)) / 2
    x, y = x - center, y - center
    x_, y_ = x / blur, y / blur

    if cluster_scale is None:
        D = x.shape[-1]
        if diameter is None:
            diameter = max_diameter(x_.view(-1, D), y_.view(-1, D))
        else:
            diameter = diameter / blur
        cluster_scale = diameter / (np.sqrt(D) * 2000 ** (1 / D))
    cell_diameter = cluster_scale * np.sqrt(x_.shape[-1])

    cx = Clusters(α, x, labels=grid_cluster(x_, cluster_scale))
    cy = Clusters(β, y, labels=grid_cluster(y_, cluster_scale))
    if verbose:
        print(f"{len(cx)}x{len(cy)} clusters, computed at scale = {cluster_scale:2.3f}")

    with torch.no_grad():
        x_c, y_c = cx.centroids / blur, cy.centroids / blur
        bound = (truncate + cell_diameter) ** 2
        ranges_xx = from_matrix_or_dense(cx, cx, squared_distances(x_c, x_c) <= bound)
        ranges_yy = from_matrix_or_dense(cy, cy, squared_distances(y_c, y_c) <= bound)
        ranges_xy = from_matrix_or_dense(cx, cy, squared_distances(x_c, y_c) <= bound)

    out = kernel_loss(cx.weights, cx.points, cy.weights, cy.points, blur=blur, kernel=kernel, name=name,
                      potentials=potentials, use_keops=True, ranges_xx=ranges_xx, ranges_yy=ranges_yy,
                      ranges_xy=ranges_xy)
    if potentials:  # retour à l'ordre initial (GeomLoss rend l'ordre trié)
        F, G = out
        f, g = F.clone(), G.clone()
        f[cx.perm], g[cy.perm] = F, G
        return f, g
    return out
