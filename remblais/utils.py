# Derived from GeomLoss v0.3.1 (geomloss/_legacy/utils.py),
# Copyright (c) 2019 Jean Feydy, MIT License (see THIRD_PARTY.md).
"""Shared tools: dot product, dense distances (tensorized backend)."""

import torch


def scal(a, f, batch=False):
    """⟨a, f⟩, per batch element if ``batch``."""
    if batch:
        B = a.shape[0]
        return (a.reshape(B, -1) * f.reshape(B, -1)).sum(1)
    return torch.dot(a.reshape(-1), f.reshape(-1))


def squared_distances(x, y):
    """Dense ‖x_i − y_j‖², (N, M) or (B, N, M), with the GeomLoss identity."""
    if x.dim() == 2:
        D_xx = (x * x).sum(-1).unsqueeze(1)
        D_xy = torch.matmul(x, y.permute(1, 0))
        D_yy = (y * y).sum(-1).unsqueeze(0)
    elif x.dim() == 3:
        D_xx = (x * x).sum(-1).unsqueeze(2)
        D_xy = torch.matmul(x, y.permute(0, 2, 1))
        D_yy = (y * y).sum(-1).unsqueeze(1)
    else:
        raise ValueError(f"Nombre de dimensions incorrect : x.shape = {tuple(x.shape)}")
    return D_xx - 2 * D_xy + D_yy


def distances(x, y):
    """Dense ‖x_i − y_j‖, with the 1e-8 floor under the square root (like GeomLoss)."""
    return torch.sqrt(torch.clamp_min(squared_distances(x, y), 1e-8))


def max_diameter(x, y):
    """Diagonal of the common bounding box of ``x`` and ``y`` (N, D)."""
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    return (maxs - mins).norm().item()
