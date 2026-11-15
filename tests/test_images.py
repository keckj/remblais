"""Divergence de Sinkhorn entre images et barycentres : port de GeomLoss et softmin séparable.

Sans KeOps, GeomLoss ne peut pas calculer ses softmins sur grille : la
référence exécute l'algorithme de GeomLoss 0.3.1 avec une softmin dense Torch
(monkeypatch), puis compare à remblais (softmin kernox).
"""

import numpy as np
import pytest
import torch

import remblais
from remblais.sinkhorn_images import C_transform, softmin_grid

try:
    import geomloss
    import geomloss._legacy.sinkhorn_images as gl_images
    import geomloss._legacy.wasserstein_barycenter_images as gl_bary
except ImportError:  # pragma: no cover
    geomloss = None

needs_geomloss = pytest.mark.skipif(geomloss is None, reason="geomloss 0.3.1 absent")


def dense_softmin_grid(eps, p, h):
    """Référence dense : LSE explicite (N × N) le long de chaque axe."""
    D = h.dim() - 2
    N = h.shape[-1]
    x = torch.arange(N, dtype=h.dtype, device=h.device) / N
    d = x[:, None] - x[None, :]
    C = d.abs() / eps if p == 1 else d**2 / (2 * eps)

    def lines(g):
        return (g.unsqueeze(-2) - C).logsumexp(-1)

    h = lines(h)
    if D >= 2:
        h = lines(h.transpose(-1, -2)).transpose(-1, -2)
    if D == 3:
        h = lines(h.transpose(-1, -3)).transpose(-1, -3)
    return -eps * h


def images(B=2, K=1, N=16, D=2, seed=0, dtype=torch.float64, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    shape = (B, K) + (N,) * D
    a = torch.rand(shape, generator=g, dtype=torch.float64) ** 4
    b = torch.rand(shape, generator=g, dtype=torch.float64) ** 4
    a[..., : N // 4] = 0  # des pixels vides
    dims = tuple(range(2, 2 + D))
    a, b = a / a.sum(dims, keepdim=True), b / b.sum(dims, keepdim=True)
    return a.to(device=device, dtype=dtype), b.to(device=device, dtype=dtype)


@pytest.mark.parametrize("D", [2, 3])
@pytest.mark.parametrize("p", [1, 2])
def test_softmin_grid_matches_dense(p, D, device):
    a, _ = images(N=8, D=D, device=device)
    h = torch.where(a > 0, a.log(), torch.full_like(a, -1e4))
    for eps in (1.0, 0.05, 0.003):
        torch.testing.assert_close(softmin_grid(eps, p, h).cpu(), dense_softmin_grid(eps, p, h.cpu()),
                                   rtol=1e-11, atol=1e-11)


@needs_geomloss
@pytest.mark.parametrize("kw", [dict(), dict(p=1), dict(reach=0.5), dict(debias=False), dict(potentials=True),
                                dict(blur=0.1, scaling=0.7)], ids=str)
def test_sinkhorn_images_matches_geomloss_algorithm(kw, monkeypatch, device):
    monkeypatch.setattr(gl_images, "softmin_grid", dense_softmin_grid)
    a, b = images(N=16)
    a_, b_ = [t.to(device).requires_grad_() for t in (a, b)]
    out = remblais.sinkhorn_divergence(a_, b_, **kw)
    ref = gl_images.sinkhorn_divergence(a.requires_grad_(), b.requires_grad_(), **kw)
    outs, refs = (out, ref) if isinstance(out, tuple) else ((out,), (ref,))
    for u, v in zip(outs, refs):
        torch.testing.assert_close(u.detach().cpu(), v.detach(), rtol=1e-10, atol=1e-12)
    if not kw.get("potentials"):
        g = torch.autograd.grad(out.sum(), (a_, b_))
        g_ref = torch.autograd.grad(ref.sum(), (a, b))
        for u, v in zip(g, g_ref):
            torch.testing.assert_close(u.cpu(), v, rtol=1e-9, atol=1e-11)


@needs_geomloss
@pytest.mark.parametrize("grad", [False, True])
def test_images_barycenter_matches_geomloss_algorithm(grad, monkeypatch, device):
    monkeypatch.setattr(gl_bary, "softmin", dense_softmin_grid)
    meas, _ = images(B=1, K=3, N=16, seed=2)
    w = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)
    m_, w_ = meas.to(device), w.to(device)
    if grad:  # GeomLoss : les itérations finales ne dérivent que par rapport aux poids
        w_.requires_grad_(), w.requires_grad_()
    out = remblais.ImagesBarycenter(m_, w_, scaling_N=3)
    ref = gl_bary.ImagesBarycenter(meas, w, scaling_N=3)
    assert out.shape == ref.shape == (1, 1, 16, 16)
    torch.testing.assert_close(out.detach().cpu(), ref.detach(), rtol=1e-9, atol=1e-12)
    if grad:
        torch.testing.assert_close(torch.autograd.grad((out**2).sum(), w_)[0].cpu(),
                                   torch.autograd.grad((ref**2).sum(), w)[0], rtol=1e-8, atol=1e-12)


def test_sinkhorn_images_close_to_samples():
    """La divergence sur grille approche celle de SamplesLoss entre les pixels pondérés (p = 2).

    L'approximation multi-échelle de la grille disparaît quand scaling → 1
    (écart 32 % pour scaling = 0,5, 0,3 % pour 0,97).
    """
    a, b = images(B=1, N=16, seed=5)
    N = a.shape[-1]
    t = torch.arange(N, dtype=torch.float64) / N
    X = torch.stack(torch.meshgrid(t, t, indexing="ij"), -1).view(-1, 2)
    val_grid = remblais.sinkhorn_divergence(a, b, blur=0.1, scaling=0.97)[0]
    val_pts = remblais.SamplesLoss(blur=0.1, scaling=0.97, backend="tensorized", diameter=1.0)(
        a.view(-1), X, b.view(-1), X)
    assert abs(val_grid - val_pts) < 1e-2 * abs(val_pts)


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("D", [1, 2, 3])
def test_c_transform(p, D, device):
    N, tau = 6, 0.7
    g = torch.Generator().manual_seed(1)
    G = torch.randn((2,) + (N,) * D, generator=g, dtype=torch.float64)
    idx = torch.stack(torch.meshgrid(*[torch.arange(N, dtype=torch.float64)] * D, indexing="ij"), -1).view(-1, D)
    diff = idx[:, None, :] - idx[None, :, :]
    C = diff.abs().sum(-1) / tau if p == 1 else (diff**2).sum(-1) / (2 * tau)
    ref = (G.view(2, 1, -1) - C).max(-1).values.view(G.shape)
    out = C_transform(G.to(device), tau=tau, p=p)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-12, atol=1e-12)
    assert np.isfinite(out.cpu().numpy()).all()
