# Derived from GeomLoss v0.3.1 (geomloss/_backends/keops.py),
# Copyright (c) 2019 Jean Feydy, MIT License (see THIRD_PARTY.md).
"""The "lazy" backend: the kernox-based LazyTensor of remblais replaces KeOps."""

from ..lazy import LazyTensor as torch_LazyTensor

keops_available = True
numpy_LazyTensor = torch_LazyTensor  # NumPy inputs: Torch computation on the CPU, NumPy output


def sum(K_ij, axis=None, keepdims=False):
    assert axis is not None
    assert not keepdims
    return K_ij.sum(dim=axis)


def logsumexp(K_ij, axis=None, keepdims=False):
    assert axis is not None
    assert not keepdims
    return K_ij.logsumexp(dim=axis)


def amin(K_ij, axis=None, keepdims=False):
    assert axis is not None
    assert not keepdims
    return K_ij.min(dim=axis)
