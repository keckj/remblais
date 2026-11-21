# Derived from GeomLoss v0.3.1 (geomloss/_legacy/wasserstein_barycenter_images.py),
# Copyright (c) 2019 Jean Feydy, MIT License (see THIRD_PARTY.md).
"""Wasserstein barycenter of images (multiscale debiased Sinkhorn, Janati et al. 2020)."""

import torch

from .sinkhorn_images import log_dens, pyramid, upsample
from .sinkhorn_images import softmin_grid as softmin


def barycenter_iteration(f_k, g_k, d_log, eps, p, ak_log, w_k):
    # Barycenter update (geometric mean)
    ft_k = softmin(eps, p, ak_log + g_k / eps) / eps
    bar_log = d_log - (ft_k * w_k[:, :, None, None]).sum(1, keepdim=True)

    # Symmetric updates of the potentials
    ft_k = softmin(eps, p, ak_log + g_k / eps)
    gt_k = softmin(eps, p, bar_log + f_k / eps)
    f_k = (f_k + ft_k) / 2
    g_k = (g_k + gt_k) / 2

    # Barycenter and debiasing term
    ft_k = softmin(eps, p, ak_log + g_k / eps) / eps
    bar_log = d_log - (ft_k * w_k[:, :, None, None]).sum(1, keepdim=True)
    d_log = 0.5 * (d_log + bar_log + softmin(eps, p, d_log) / eps)
    return f_k, g_k, d_log, bar_log


def ImagesBarycenter(measures, weights, blur=0, p=2, scaling_N=10, backward_iterations=5):
    r"""Wasserstein barycenter of images (debiased Sinkhorn,
    [Janati et al., ICML 2020](https://proceedings.mlr.press/v119/janati20a.html)).

    Same parameters and same algorithm as GeomLoss 0.3.1.

    :param measures: (B, K, N, N) densities, $N$ a power of 2.
    :param weights: (B, K) barycentric weights, summing to 1.
    :param blur: length scale; ``0`` means one pixel, $1/N$.
    :param p: exponent of the cost.
    :param scaling_N: number of iterations per level of the image pyramid.
    :param backward_iterations: number of final iterations that are
        differentiated when an input requires a gradient.
    :return: the (B, 1, N, N) barycenter.
    """
    a_k = measures
    w_k = weights
    if blur == 0:
        blur = 1 / measures.shape[-1]

    with torch.set_grad_enabled(backward_iterations == 0):
        ak_s = pyramid(a_k)[1:]
        ak_log_s = list(map(log_dens, ak_s))

        sigma = 1
        eps = sigma**p
        f_k, g_k = softmin(eps, p, ak_log_s[0]), softmin(eps, p, ak_log_s[0])
        d_log = torch.ones_like(ak_log_s[0]).sum(dim=1, keepdim=True)
        d_log = d_log - d_log.logsumexp([2, 3], keepdim=True)

        for n, ak_log in enumerate(ak_log_s):
            for _ in range(scaling_N):
                eps = sigma**p
                f_k, g_k, d_log, bar_log = barycenter_iteration(f_k, g_k, d_log, eps, p, ak_log, w_k)
                sigma = max(sigma * 2 ** (-1 / scaling_N), blur)
            if n + 1 < len(ak_s):
                f_k, g_k, d_log = upsample(f_k), upsample(g_k), upsample(d_log)

    if (measures.requires_grad or weights.requires_grad) and backward_iterations > 0:
        for _ in range(backward_iterations):
            f_k, g_k, d_log, bar_log = barycenter_iteration(f_k, g_k, d_log, eps, p, ak_log, w_k)

    return bar_log.exp()
