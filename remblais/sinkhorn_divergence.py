# Derived from GeomLoss v0.3.1 (geomloss/_legacy/sinkhorn_divergence.py),
# Copyright (c) 2019 Jean Feydy, MIT License (see THIRD_PARTY.md).
r"""Symmetric Sinkhorn loop with $\varepsilon$-scaling, shared by all the backends.

Same algorithm as GeomLoss v0.3.1 (Feydy et al., "Interpolating between
Optimal Transport and MMD using Sinkhorn Divergences", AISTATS 2019,
https://proceedings.mlr.press/v89/feydy19a.html, and J. Feydy's PhD thesis, 2020): damped symmetric updates, $\varepsilon$ decreasing from
$\texttt{diameter}^p$ to $\texttt{blur}^p$, differentiable final extrapolation (envelope theorem),
debiased and unbalanced formulas.

Deliberate difference: the caller's "grad enabled" state is restored at the
end of the loop (GeomLoss forces it to ``True``, even under
``torch.no_grad()``).
"""

import numpy as np
import torch

from .utils import max_diameter, scal


def dampening(eps, rho):
    r"""Dampening factor $1/(1 + \varepsilon/\rho)$ of unbalanced transport."""
    return 1 if rho is None else 1 / (1 + eps / rho)


def log_weights(a):
    r"""$\log a$, with $\log 0$ replaced by $-100000$ (zero masses)."""
    a_log = a.log()
    a_log[a <= 0] = -100000
    return a_log


class UnbalancedWeight(torch.nn.Module):
    r"""$(\rho + \varepsilon/2) \cdot x$ in the forward pass, $(\rho + \varepsilon) \cdot g$ in the backward pass.

    This is the correction intended by GeomLoss: with potentials approximated at
    temperature $\varepsilon$, it makes the gradient of the unbalanced divergence exact at
    convergence. GeomLoss v0.3.1 declares this ``backward`` on an ``nn.Module``,
    where Torch never calls it: its effective gradient is the one of
    $(\rho + \varepsilon/2) \cdot x$, wrong by about $\varepsilon/(2\rho)$. remblais applies the correction (a
    deliberate difference, see "Coming from GeomLoss" in the documentation).
    """

    def __init__(self, eps, rho):
        super().__init__()
        self.eps, self.rho = eps, rho

    def forward(self, x):
        return _UnbalancedWeightFn.apply(x, self.rho + self.eps / 2, self.rho + self.eps)


class _UnbalancedWeightFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fwd, w_bwd):
        ctx.w_bwd = w_bwd
        return w_fwd * x

    @staticmethod
    def backward(ctx, g):
        return ctx.w_bwd * g, None, None


def epsilon_schedule(p, diameter, blur, scaling):
    r"""[$\texttt{diameter}^p$] + geometric sequence of ratio $\texttt{scaling}^p$ + [$\texttt{blur}^p$] (GeomLoss)."""
    return ([diameter**p]
            + [np.exp(e) for e in np.arange(p * np.log(diameter), p * np.log(blur), p * np.log(scaling))]
            + [blur**p])


def scaling_parameters(x, y, p, blur, reach, diameter, scaling):
    r"""(diameter, $\varepsilon$, list of the $\varepsilon$ values, $\rho$)."""
    if diameter is None:
        D = x.shape[-1]
        diameter = max_diameter(x.reshape(-1, D), y.reshape(-1, D))
    eps = blur**p
    rho = None if reach is None else reach**p
    eps_list = epsilon_schedule(p, diameter, blur, scaling)
    return diameter, eps, eps_list, rho


def sinkhorn_cost(eps, rho, a, b, f_aa, g_bb, g_ab, f_ba, batch=False, debias=True, potentials=False):
    """Value of the divergence (or potentials) from the dual potentials."""
    if potentials:
        return (f_ba - f_aa, g_ab - g_bb) if debias else (f_ba, g_ab)
    if debias:
        if rho is None:
            return scal(a, f_ba - f_aa, batch=batch) + scal(b, g_ab - g_bb, batch=batch)
        uw = UnbalancedWeight(eps, rho)
        return (scal(a, uw((-f_aa / rho).exp() - (-f_ba / rho).exp()), batch=batch)
                + scal(b, uw((-g_bb / rho).exp() - (-g_ab / rho).exp()), batch=batch))
    if rho is None:
        return scal(a, f_ba, batch=batch) + scal(b, g_ab, batch=batch)
    uw = UnbalancedWeight(eps, rho)
    return (scal(a, uw(1 - (-f_ba / rho).exp()), batch=batch)
            + scal(b, uw(1 - (-g_ab / rho).exp()), batch=batch))


def sinkhorn_loop(softmin, a_logs, b_logs, C_xxs, C_yys, C_xys, C_yxs, eps_list, rho,
                  jumps=(), kernel_truncation=None, truncate=5, cost=None, extrapolate=None,
                  debias=True, last_extrapolation=True):
    r"""Symmetric Sinkhorn iterations; returns (f_aa, g_bb, g_ab, f_ba).

    ``softmin(eps, C, h)`` computes $-\varepsilon \log \sum_j \exp(h_j - C_{ij}/\varepsilon)$. The lists
    (``a_logs``, ``C_xys``, ...) describe the coarse to fine levels of the
    multiscale scheme.
    """
    grad_state = torch.is_grad_enabled()
    if type(a_logs) is not list:
        a_logs, b_logs = [a_logs], [b_logs]
        C_xys, C_yxs = [C_xys], [C_yxs]
        if debias:
            C_xxs, C_yys = [C_xxs], [C_yys]

    torch.set_grad_enabled(False)
    try:
        k = 0
        eps = eps_list[k]
        damping = dampening(eps, rho)
        a_log, b_log = a_logs[k], b_logs[k]
        C_xy, C_yx = C_xys[k], C_yxs[k]
        if debias:
            C_xx, C_yy = C_xxs[k], C_yys[k]

        g_ab = damping * softmin(eps, C_yx, a_log)
        f_ba = damping * softmin(eps, C_xy, b_log)
        if debias:
            f_aa = damping * softmin(eps, C_xx, a_log)
            g_bb = damping * softmin(eps, C_yy, b_log)

        for i, eps in enumerate(eps_list):
            damping = dampening(eps, rho)
            ft_ba = damping * softmin(eps, C_xy, b_log + g_ab / eps)
            gt_ab = damping * softmin(eps, C_yx, a_log + f_ba / eps)
            if debias:
                ft_aa = damping * softmin(eps, C_xx, a_log + f_aa / eps)
                gt_bb = damping * softmin(eps, C_yy, b_log + g_bb / eps)

            f_ba, g_ab = 0.5 * (f_ba + ft_ba), 0.5 * (g_ab + gt_ab)
            if debias:
                f_aa, g_bb = 0.5 * (f_aa + ft_aa), 0.5 * (g_bb + gt_bb)

            if i in jumps:
                if i == len(eps_list) - 1:
                    C_xy_fine, C_yx_fine = C_xys[k + 1], C_yxs[k + 1]
                    if debias:
                        C_xx_fine, C_yy_fine = C_xxs[k + 1], C_yys[k + 1]
                    last_extrapolation = False
                    torch.set_grad_enabled(grad_state)
                else:
                    C_xy_fine, C_yx_fine = kernel_truncation(
                        C_xy, C_yx, C_xys[k + 1], C_yxs[k + 1], f_ba, g_ab, eps, truncate=truncate, cost=cost)
                    if debias:
                        C_xx_fine, _ = kernel_truncation(
                            C_xx, C_xx, C_xxs[k + 1], C_xxs[k + 1], f_aa, f_aa, eps, truncate=truncate, cost=cost)
                        C_yy_fine, _ = kernel_truncation(
                            C_yy, C_yy, C_yys[k + 1], C_yys[k + 1], g_bb, g_bb, eps, truncate=truncate, cost=cost)

                f_ba, g_ab = (extrapolate(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_fine),
                              extrapolate(g_ab, f_ba, eps, damping, C_yx, a_log, C_yx_fine))
                if debias:
                    f_aa = extrapolate(f_aa, f_aa, eps, damping, C_xx, a_log, C_xx_fine)
                    g_bb = extrapolate(g_bb, g_bb, eps, damping, C_yy, b_log, C_yy_fine)

                k = k + 1
                a_log, b_log = a_logs[k], b_logs[k]
                C_xy, C_yx = C_xy_fine, C_yx_fine
                if debias:
                    C_xx, C_yy = C_xx_fine, C_yy_fine
    finally:
        torch.set_grad_enabled(grad_state)

    if last_extrapolation:
        f_ba, g_ab = (damping * softmin(eps, C_xy, (b_log + g_ab / eps).detach()),
                      damping * softmin(eps, C_yx, (a_log + f_ba / eps).detach()))
        if debias:
            f_aa = damping * softmin(eps, C_xx, (a_log + f_aa / eps).detach())
            g_bb = damping * softmin(eps, C_yy, (b_log + g_bb / eps).detach())

    if debias:
        return f_aa, g_bb, g_ab, f_ba
    return None, None, g_ab, f_ba
