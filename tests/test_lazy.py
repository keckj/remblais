"""LazyTensor kernox : sémantique KeOps (formes, réductions, transposition, NumPy, gradients)."""

import numpy as np
import pytest
import torch

from remblais.lazy import LazyTensor


def _data(N=7, M=5, D=3, device="cpu"):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, D, generator=g, dtype=torch.float64).to(device)
    y = torch.randn(M, D, generator=g, dtype=torch.float64).to(device)
    h = torch.randn(M, generator=g, dtype=torch.float64).to(device)
    return x, y, h


def test_shapes_and_reductions(device):
    x, y, h = _data(device=device)
    N, D = x.shape
    M = y.shape[0]
    x_i, y_j = LazyTensor(x.view(N, 1, D)), LazyTensor(y.view(1, M, D))
    h_j = LazyTensor(h.view(1, M, 1))
    C = ((x_i - y_j) ** 2).sum(-1)
    assert C.shape == (N, M)
    assert (x_i - y_j).shape == (N, M, D)
    Cd = ((x[:, None] - y[None]) ** 2).sum(-1)
    eps = 0.7
    S = h_j - C / eps
    torch.testing.assert_close(S.logsumexp(dim=1).view(-1), (h[None] - Cd / eps).logsumexp(1))
    torch.testing.assert_close(S.logsumexp(dim=0).view(-1), (h[None] - Cd / eps).logsumexp(0))
    torch.testing.assert_close(C.min(dim=1).view(-1), Cd.min(1).values)
    torch.testing.assert_close((C * h_j).sum(dim=1).view(-1), (Cd * h[None]).sum(1))
    K = (-C).exp()
    v = torch.randn(M, 2, dtype=torch.float64, device=device)
    torch.testing.assert_close(K @ v, (-Cd).exp() @ v)
    u = torch.randn(N, 2, dtype=torch.float64, device=device)
    torch.testing.assert_close(K.T @ u, (-Cd).exp().T @ u)
    torch.testing.assert_close((x_i | y_j).sum(1).view(-1), (x @ y.T).sum(1))
    torch.testing.assert_close((x_i - y_j).norm2().sum(1).view(-1), Cd.sqrt().sum(1))


def test_scalar_parameters_do_not_recompile():
    from remblais import _kp

    x, y, h = _data()
    x_i, y_j = LazyTensor(x[:, None, :]), LazyTensor(y[None, :, :])
    n0 = len(_kp._PLANS)
    for eps in (1.0, 0.5, 0.25):
        (-(((x_i - y_j) ** 2).sum(-1)) / eps).logsumexp(1)
    assert len(_kp._PLANS) - n0 <= 1


def test_numpy_and_degenerate_sizes():
    x, y, h = _data(N=1)
    xn, yn = x.numpy(), y.numpy()
    C = ((LazyTensor(xn.reshape(1, 1, 3)) - LazyTensor(yn.reshape(1, -1, 3))) ** 2).sum(-1)
    out = C.min(dim=1)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out.reshape(-1), ((xn[:, None] - yn[None]) ** 2).sum(-1).min(1))
    x1, y1, _ = _data(N=1, M=1)
    C1 = ((LazyTensor(x1.view(1, 1, 3)) - LazyTensor(y1.view(1, 1, 3))) ** 2).sum(-1)
    torch.testing.assert_close(C1.sum(dim=1).view(-1), ((x1 - y1) ** 2).sum(-1))


def test_gradients():
    x, y, h = _data()
    x.requires_grad_(), h.requires_grad_()
    S = LazyTensor(h.view(1, -1, 1)) - ((LazyTensor(x[:, None]) - LazyTensor(y[None])) ** 2).sum(-1)
    out = S.logsumexp(dim=1).sum()
    gx, gh = torch.autograd.grad(out, (x, h))
    ref = (h[None] - ((x[:, None] - y[None]) ** 2).sum(-1)).logsumexp(1).sum()
    rx, rh = torch.autograd.grad(ref, (x, h))
    torch.testing.assert_close(gx, rx)
    torch.testing.assert_close(gh, rh)


def test_errors():
    x, y, _ = _data()
    with pytest.raises(ValueError):
        LazyTensor(x)
    with pytest.raises(ValueError):
        LazyTensor(torch.zeros(3, 4, 2))
