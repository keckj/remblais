# Derived from GeomLoss v0.3.1 (geomloss/_legacy/sinkhorn_samples.py),
# Copyright (c) 2019 Jean Feydy, MIT License (see THIRD_PARTY.md).
"""Sinkhorn divergences between point clouds: tensorized, online and multiscale backends.

- ``tensorized``: dense Torch cost matrices (quadratic memory), like GeomLoss;
- ``online``: one fused kernox LOGSUMEXP reduction per softmin, a plan
  compiled once (ε as a parameter), native batches, batched custom costs;
- ``multiscale``: grid clustering, coarse iterations on the centroids, then
  block-sparse truncation (kernox ranges) at the fine level.
"""

import os
from functools import partial

import numpy as np
import torch

from . import _kp
from .clustering import Clusters, from_matrix_or_dense, ranged_cost_ratio
from .costs import resolve_cost
from .sinkhorn_divergence import log_weights, scaling_parameters, sinkhorn_cost, sinkhorn_loop


# ---------------------------------------------------------------------------------
# Tensorized
# ---------------------------------------------------------------------------------
def softmin_tensorized(eps, C_xy, h_y):
    """−ε log Σ_j exp(h_j − C_ij/ε) on a dense (B, N, M) matrix."""
    B = C_xy.shape[0]
    return -eps * (h_y.view(B, 1, -1) - C_xy / eps).logsumexp(2).view(B, -1)


def sinkhorn_tensorized(a, x, b, y, p=2, blur=0.05, reach=None, diameter=None, scaling=0.5, cost=None,
                        debias=True, potentials=False, **kwargs):
    cost = resolve_cost(cost, p, "tensorized").dense
    C_xy = cost(x, y.detach())
    C_yx = cost(y, x.detach())
    C_xx = cost(x, x.detach()) if debias else None
    C_yy = cost(y, y.detach()) if debias else None

    diameter, eps, eps_list, rho = scaling_parameters(x, y, p, blur, reach, diameter, scaling)
    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(softmin_tensorized, log_weights(a), log_weights(b),
                                           C_xx, C_yy, C_xy, C_yx, eps_list, rho, debias=debias)
    return sinkhorn_cost(eps, rho, a, b, f_aa, g_bb, g_ab, f_ba, batch=True, debias=debias,
                         potentials=potentials)


# ---------------------------------------------------------------------------------
# Online (kernox)
# ---------------------------------------------------------------------------------
def softmin_online(eps, C_xy, h_y, cost=None):
    """Softmin as a kernox LOGSUMEXP reduction; C_xy = (x, y), (B, N, D) / (B, M, D)."""
    x, y = C_xy
    if x.dim() == 3 and x.shape[0] == 1:  # B = 1: call without a batch dimension
        return -eps * _kp.lse(cost, x[0], y[0], h_y.reshape(-1), 1.0 / eps).unsqueeze(0)
    return -eps * _kp.lse(cost, x, y, h_y, 1.0 / eps)


# Dot-product form: disabled by default until benchmarks settle the question;
# REMBLAIS_DOT_MIN_DIM=16 enables it for D ≥ 16 (experimental).
DOT_MIN_DIM = int(os.environ.get("REMBLAIS_DOT_MIN_DIM", 10**9))


def sinkhorn_loop_paired(cost, a_log, b_log, x, y, eps_list, rho, dot=False, debias=True):
    """Debiased Sinkhorn loop with two kernox reductions per iteration (instead of four).

    Same algorithm as :func:`sinkhorn_loop` (same updates, same order, same final
    extrapolation): the softmins that share their j set are fused into one
    reduction over concatenated rows,

    - A: rows [x ; y], j = y: (f_ba, g_bb);
    - B: rows [y ; x], j = x: (g_ab, f_aa);

    each row selecting exactly its own scores (see :func:`_kp.lse2`). The
    potentials are kept concatenated: P_A = [f_ba ; g_bb], P_B = [g_ab ; f_aa].
    x, y: (…, N, D), (…, M, D); a_log, b_log: (…, N), (…, M).

    ``debias=False``: only (f_ba, g_ab); A = rows x, j = y; B = rows y, j = x
    (two reductions per iteration, like GeomLoss).

    ``dot=True`` (½‖x − y‖² cost only): after centering, ½‖x − y‖² =
    ½‖x‖² + ½‖y‖² − ⟨x, y⟩; the norms go into the j scores and the output, and
    the pair only computes ⟨x_i, y_j⟩ (D FMAs instead of D subtractions + D
    FMAs). Rounding error in u·‖x‖‖y‖/ε instead of u·‖x − y‖²/ε: reserved for
    high dimensions, where both are of the same order.
    """
    from .sinkhorn_divergence import dampening

    N, M = x.shape[-2], y.shape[-2]
    batch = x.shape[:-2]
    dt, dev = x.dtype, x.device

    def sel(n1, n2):  # q = 1 on the first n1 rows, r = 1 on the next n2
        q = torch.cat((torch.ones(n1, dtype=dt, device=dev), torch.zeros(n2, dtype=dt, device=dev)))
        return q.expand(*batch, n1 + n2).contiguous(), (1 - q).expand(*batch, n1 + n2).contiguous()

    if debias:
        rows_A = torch.cat((x, y), dim=-2)  # gradients flow to x and y through torch.cat
        rows_B = torch.cat((y, x), dim=-2)
    else:
        rows_A, rows_B = x, y
    yd, xd = y.detach(), x.detach()
    if dot:
        with torch.no_grad():
            center = 0.5 * (xd.mean(-2, keepdim=True) + yd.mean(-2, keepdim=True))
        rows_A, rows_B = rows_A - center, rows_B - center
        yd, xd = yd - center, xd - center
    factor = 1.0 if dot else cost.factor
    ws = dict(a_log=a_log, b_log=b_log, rows_A=rows_A.detach().contiguous(), rows_B=rows_B.detach().contiguous(),
              cols_A=yd.contiguous(), cols_B=xd.contiguous(),
              s=torch.tensor([factor / e for e in eps_list], dtype=dt, device=dev))
    ws["qA"], ws["rA"] = sel(N, M) if debias else sel(N, 0)
    ws["qB"], ws["rB"] = sel(M, N) if debias else sel(M, 0)
    if dot:
        ws["n_A"], ws["n_B"] = 0.5 * (yd * yd).sum(-1), 0.5 * (xd * xd).sum(-1)
    consts = (N, M, tuple(eps_list), rho, dot, debias)

    grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        if USE_CUDA_GRAPHS and dev.type == "cuda":
            P_A, P_B = _graph_iterations(cost, ws, consts)
        else:
            P_A, P_B = _run_iterations(cost, ws, consts)
    finally:
        torch.set_grad_enabled(grad_state)

    # Differentiable final extrapolation (envelope theorem):
    eps = eps_list[-1]
    damping = dampening(eps, rho)
    k = len(eps_list) - 1
    hA, hB = _scores(a_log, b_log, P_A, P_B, eps, N, M, debias)
    hA, hB = _detach_pair(hA), _detach_pair(hB)
    wsg = dict(ws, rows_A=rows_A, rows_B=rows_B)
    if dot:
        wsg["n_rows_A"], wsg["n_rows_B"] = 0.5 * (rows_A * rows_A).sum(-1), 0.5 * (rows_B * rows_B).sum(-1)
    if debias and not dot and torch.is_grad_enabled() and x.requires_grad != y.requires_grad:
        # A single differentiable cloud (common case: fixed target y): four
        # separate LSEs, so that the gradient (one N·M pass per differentiable
        # LSE) does not go over the rows of the fixed cloud (2 passes instead of 4).
        s_k = ws["s"][k:k + 1]
        P_A = (-damping * eps) * torch.cat((_kp.lse_full(cost, x, ws["cols_A"], hA[0], s_k),
                                            _kp.lse_full(cost, y, ws["cols_A"], hA[1], s_k)), dim=-1)
        P_B = (-damping * eps) * torch.cat((_kp.lse_full(cost, y, ws["cols_B"], hB[0], s_k),
                                            _kp.lse_full(cost, x, ws["cols_B"], hB[1], s_k)), dim=-1)
    else:
        P_A = (-damping * eps) * _lse_side(cost, wsg, "A", k, *hA, dot)
        P_B = (-damping * eps) * _lse_side(cost, wsg, "B", k, *hB, dot)
    if not debias:
        return None, None, P_B, P_A
    f_ba, g_bb = P_A[..., :N], P_A[..., N:]
    g_ab, f_aa = P_B[..., :M], P_B[..., M:]
    return f_aa, g_bb, g_ab, f_ba


def _scores(a_log, b_log, P_A, P_B, eps, N, M, debias):
    """j scores (ha, hb) of sides A and B from the current potentials."""
    if not debias:  # P_A = f_ba (N), P_B = g_ab (M)
        hA = _h(b_log, P_B, eps)
        hB = _h(a_log, P_A, eps)
        return (hA, hA), (hB, hB)
    return ((_h(b_log, P_B[..., :M], eps), _h(b_log, P_A[..., N:], eps)),
            (_h(a_log, P_A[..., :N], eps), _h(a_log, P_B[..., M:], eps)))


def _detach_pair(h):
    """Detaches (ha, hb) while keeping ha is hb (without debiasing: a single softmin, see _lse_side)."""
    d = h[0].detach()
    return (d, d) if h[1] is h[0] else (d, h[1].detach())


def _h(log_w, P, eps):
    """log_w + P/ε in a single operation."""
    return torch.add(log_w, P, alpha=1.0 / eps)


def _lse_side(cost, ws, side, k, ha, hb, dot):
    """log Σ_j exp(score) for side A (rows [x ; y], j = y) or B (rows [y ; x], j = x)."""
    rows, cols, q, r = ws["rows_" + side], ws["cols_" + side], ws["q" + side], ws["r" + side]
    s = ws["s"][k:k + 1]
    if not dot:
        n, m = rows.shape[-2], cols.shape[-2]
        if ha is hb:  # without debiasing: a single softmin
            return _kp.lse_full(cost, rows, cols, ha, s)
        if STACKED and n == 2 * m:  # N = M: rows [x ; y] viewed as (…, 2, N, D), broadcast columns (stride 0)
            batch = rows.shape[:-2]
            out = _kp.lse_full(cost, rows.view(*batch, 2, m, rows.shape[-1]),
                               cols.unsqueeze(-3).expand(*batch, 2, m, cols.shape[-1]),
                               torch.stack((ha, hb), dim=-2), s)
            return out.reshape(*batch, n)
        return _kp.lse2(cost, rows, q, r, cols, ha, hb, s)
    n_cols = ws["n_" + side]
    sk = ws["s"][k]
    out = _kp.lse2dot(rows, q, r, cols, ha - sk * n_cols, hb - sk * n_cols, s)
    n_rows = ws.get("n_rows_" + side)
    if n_rows is None:
        n_rows = 0.5 * (rows * rows).sum(-1)
    return out - sk * n_rows


USE_CUDA_GRAPHS = os.environ.get("REMBLAIS_CUDA_GRAPHS", "0") == "1"
# N = M: paired softmins in one LSE with a batch dimension of 2 (without the q, r
# selectors of lse2, ~8% less work per pair); REMBLAIS_STACKED=0 disables it.
STACKED = os.environ.get("REMBLAIS_STACKED", "1") != "0"
_GRAPHS = {}           # key (cost, shapes, dtype, device, constants) -> (graph, inputs, outputs)
_GRAPH_FAILED = set()  # keys whose capture failed: eager execution


def _graph_iterations(cost, ws, consts):
    """Iterations replayed from a CUDA graph (captured on the first call for a key).

    The iterations depend only on the tensors of ``ws`` (copied into static
    buffers) and on Python constants (ε, ρ, sizes) that are part of the key: no
    host cost per iteration on replay.
    """
    key = (cost.key, consts, _kp.fast_math_enabled(), str(ws["a_log"].device), ws["a_log"].dtype,
           tuple((k, tuple(v.shape)) for k, v in sorted(ws.items())))
    if key in _GRAPH_FAILED:
        return _run_iterations(cost, ws, consts)
    hit = _GRAPHS.get(key)
    if hit is None:
        static = {k: v.clone() for k, v in ws.items()}
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):  # compilation, decisions and caches outside the capture
                    _run_iterations(cost, static, consts)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = _run_iterations(cost, static, consts)
        except Exception:  # noqa: BLE001  capture failed: eager execution, remembered
            _GRAPH_FAILED.add(key)
            return _run_iterations(cost, ws, consts)
        if len(_GRAPHS) >= 16:
            _GRAPHS.pop(next(iter(_GRAPHS)))
        hit = _GRAPHS[key] = (graph, static, outs)
    graph, static, outs = hit
    for k, v in ws.items():
        static[k].copy_(v)
    graph.replay()
    return outs[0].clone(), outs[1].clone()


def _run_iterations(cost, ws, consts):
    """Gradient-free iterations of :func:`sinkhorn_loop_paired`; returns (P_A, P_B)."""
    from .sinkhorn_divergence import dampening

    N, M, eps_list, rho, dot, debias = consts
    a_log, b_log = ws["a_log"], ws["b_log"]
    eps = eps_list[0]
    damping = dampening(eps, rho)
    P_A = _lse_side(cost, ws, "A", 0, b_log, b_log, dot).mul_(-damping * eps)
    P_B = _lse_side(cost, ws, "B", 0, a_log, a_log, dot).mul_(-damping * eps)
    for k, eps in enumerate(eps_list):
        damping = dampening(eps, rho)
        hA, hB = _scores(a_log, b_log, P_A, P_B, eps, N, M, debias)
        L_A = _lse_side(cost, ws, "A", k, *hA, dot)
        L_B = _lse_side(cost, ws, "B", k, *hB, dot)
        # P ← ½ (P + damping · (−ε L)), in place on the fresh output L
        P_A = L_A.mul_(-0.5 * damping * eps).add_(P_A, alpha=0.5)
        P_B = L_B.mul_(-0.5 * damping * eps).add_(P_B, alpha=0.5)
    return P_A, P_B


def sinkhorn_online(a, x, b, y, p=2, blur=0.05, reach=None, diameter=None, scaling=0.5, cost=None,
                    debias=True, potentials=False, paired=True, dot=None, **kwargs):
    cost = resolve_cost(cost, p, "online")
    if paired:
        diameter, eps, eps_list, rho = scaling_parameters(x, y, p, blur, reach, diameter, scaling)
        xs, ys, la, lb = x, y, log_weights(a), log_weights(b)
        squeeze = x.shape[0] == 1
        if squeeze:  # B = 1: calls without a batch dimension
            xs, ys, la, lb = x[0], y[0], la[0], lb[0]
        if dot is None:
            dot = cost.key == ("p", 2) and x.shape[-1] >= DOT_MIN_DIM
        f_aa, g_bb, g_ab, f_ba = sinkhorn_loop_paired(cost, la, lb, xs, ys, eps_list, rho, dot=dot, debias=debias)
        if squeeze:
            f_aa, g_bb, g_ab, f_ba = (None if t is None else t.unsqueeze(0) for t in (f_aa, g_bb, g_ab, f_ba))
        return sinkhorn_cost(eps, rho, a, b, f_aa, g_bb, g_ab, f_ba, batch=True, debias=debias,
                             potentials=potentials)
    softmin = partial(softmin_online, cost=cost)

    C_xx, C_yy = ((x, x.detach()), (y, y.detach())) if debias else (None, None)
    C_xy, C_yx = (x, y.detach()), (y, x.detach())

    diameter, eps, eps_list, rho = scaling_parameters(x, y, p, blur, reach, diameter, scaling)
    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(softmin, log_weights(a), log_weights(b),
                                           C_xx, C_yy, C_xy, C_yx, eps_list, rho, debias=debias)
    return sinkhorn_cost(eps, rho, a, b, f_aa, g_bb, g_ab, f_ba, batch=True, debias=debias,
                         potentials=potentials)


# ---------------------------------------------------------------------------------
# Multiscale (kernox + ranges block-sparse)
# ---------------------------------------------------------------------------------
# A multiscale "C" is a 5-tuple (x, y, clusters_x, clusters_y, ranges_xy):
# coarse level (centroids, Clusters, Clusters, None) or fine level (sorted points,
# None, None, ranges or None = all pairs).
def softmin_multiscale(eps, C_xy, f_y, cost=None):
    x, y, _, _, ranges_xy = C_xy
    return -eps * _kp.lse(cost, x, y, f_y.reshape(-1), 1.0 / eps, ranges=ranges_xy)


def kernel_truncation(C_xy, C_yx, C_xy_, C_yx_, f_ba, g_ab, eps, truncate=None, cost=None, verbose=False):
    """Kept fine blocks: (ci, cj) such that f_ci + g_cj > C(ci, cj) − truncate·ε (on the centroids)."""
    if truncate is None:
        return C_xy_, C_yx_
    x, yd, clusters_x, clusters_y, _ = C_xy
    x_, yd_, _, _, _ = C_xy_
    y_, xd_, _, _, _ = C_yx_
    with torch.no_grad():
        C = cost(x, yd)
        keep = f_ba.view(-1, 1) + g_ab.view(1, -1) > C - truncate * eps
        ratio = ranged_cost_ratio(clusters_x, clusters_y, keep)
        ranges_xy_ = from_matrix_or_dense(clusters_x, clusters_y, keep, ratio)
        if verbose:
            ks, Cs = int(keep.sum()), C.shape[0] * C.shape[1]
            print(f"Keep {ks}/{Cs} = {100 * ks / Cs:2.1f}% of the coarse cost matrix.")
            print(f"Estimated block-sparse cost: {ratio:.2f} x dense"
                  + (" -> dense fine level." if ranges_xy_ is None else "."))
    ranges_yx_ = None if ranges_xy_ is None else _kp.swap_axes(ranges_xy_)
    return (x_, yd_, None, None, ranges_xy_), (y_, xd_, None, None, ranges_yx_)


def extrapolate_samples(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_, softmin=None):
    """Fine potential on x from the coarse potential on y (dense softmin from centroids to points)."""
    yd = C_xy[1]
    x_ = C_xy_[0]
    return damping * softmin(eps, (x_, yd, None, None, None), (b_log + g_ab / eps).detach())


def sinkhorn_multiscale(a, x, b, y, p=2, blur=0.05, reach=None, diameter=None, scaling=0.5, truncate=5,
                        cost=None, cluster_scale=None, debias=True, potentials=False, labels_x=None,
                        labels_y=None, verbose=False, **kwargs):
    N, D = x.shape
    cost = resolve_cost(cost, p, "multiscale")
    softmin = partial(softmin_multiscale, cost=cost)
    extrapolate = partial(extrapolate_samples, softmin=softmin)

    diameter, eps, eps_list, rho = scaling_parameters(x, y, p, blur, reach, diameter, scaling)
    if cluster_scale is None:
        cluster_scale = diameter / (np.sqrt(D) * 2000 ** (1 / D))

    cx = Clusters(a, x, scale=cluster_scale, labels=labels_x)
    cy = Clusters(b, y, scale=cluster_scale, labels=labels_y)
    a_c, x_c, a, x = cx.masses, cx.centroids, cx.weights, cx.points
    b_c, y_c, b, y = cy.masses, cy.centroids, cy.weights, cy.points

    jumps = [len(eps_list) - 1]
    for i, eps_i in enumerate(eps_list[2:]):
        if cluster_scale**p > eps_i:
            jumps = [i + 1]
            break

    if verbose:
        print(f"{len(x_c)}x{len(y_c)} clusters, computed at scale = {cluster_scale:2.3f}")
        print("Successive scales : ", ", ".join(f"{e ** (1 / p):.3f}" for e in eps_list))
        if jumps[0] >= len(eps_list) - 1:
            print("Extrapolate from coarse to fine after the last iteration.")
        else:
            print(f"Jump from coarse to fine between indices {jumps[0]} (σ={eps_list[jumps[0]] ** (1 / p):2.3f}) "
                  f"and {jumps[0] + 1} (σ={eps_list[jumps[0] + 1] ** (1 / p):2.3f}).")

    a_logs = [log_weights(a_c), log_weights(a)]
    b_logs = [log_weights(b_c), log_weights(b)]
    C_xxs = [(x_c, x_c.detach(), cx, cx, None), (x, x.detach(), None, None, None)] if debias else None
    C_yys = [(y_c, y_c.detach(), cy, cy, None), (y, y.detach(), None, None, None)] if debias else None
    C_xys = [(x_c, y_c.detach(), cx, cy, None), (x, y.detach(), None, None, None)]
    C_yxs = [(y_c, x_c.detach(), cy, cx, None), (y, x.detach(), None, None, None)]

    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
        softmin, a_logs, b_logs, C_xxs, C_yys, C_xys, C_yxs, eps_list, rho, jumps=jumps, cost=cost.dense,
        kernel_truncation=partial(kernel_truncation, verbose=verbose), truncate=truncate,
        extrapolate=extrapolate, debias=debias)

    out = sinkhorn_cost(eps, rho, a, b, f_aa, g_bb, g_ab, f_ba, debias=debias, potentials=potentials)
    if potentials:  # back to the input order of the points
        F_x, G_y = out
        f_x, g_y = F_x.clone(), G_y.clone()
        f_x[cx.perm], g_y[cy.perm] = F_x, G_y
        return f_x, g_y
    return out
