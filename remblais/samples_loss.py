# Derived from GeomLoss v0.3.1 (geomloss/_legacy/samples_loss.py),
# Copyright (c) 2019 Jean Feydy, MIT License (see THIRD_PARTY.md).
"""``SamplesLoss``: same signature, same input and output shapes as GeomLoss v0.3.1."""

import warnings
from functools import partial

import torch
from torch.nn import Module

from . import _kp
from .costs import is_symbolic_only
from .hausdorff import hausdorff_multiscale, hausdorff_online, hausdorff_tensorized
from .kernel_samples import kernel_multiscale, kernel_online, kernel_tensorized
from .sinkhorn_samples import sinkhorn_multiscale, sinkhorn_online, sinkhorn_tensorized

routines = {
    "sinkhorn": {
        "tensorized": sinkhorn_tensorized,
        "online": sinkhorn_online,
        "multiscale": sinkhorn_multiscale,
    },
    # "hausdorff" with ``kernel``: kernel norm (GeomLoss routing); without
    # ``kernel``: weighted Hausdorff divergence (see remblais.hausdorff).
    "hausdorff": {
        "tensorized": hausdorff_tensorized,
        "online": hausdorff_online,
        "multiscale": hausdorff_multiscale,
    },
    "hausdorff_kernel": {
        "tensorized": kernel_tensorized,
        "online": kernel_online,
        "multiscale": kernel_multiscale,
    },
}
for _name in ("energy", "gaussian", "laplacian"):
    routines[_name] = {
        "tensorized": partial(kernel_tensorized, name=_name),
        "online": partial(kernel_online, name=_name),
        "multiscale": partial(kernel_multiscale, name=_name),
    }

BACKENDS = ("auto", "tensorized", "online", "multiscale")


def _is_process_group(group):
    try:
        import torch.distributed as dist
    except ImportError:
        return False
    return dist.is_available() and isinstance(group, dist.ProcessGroup)


class SamplesLoss(Module):
    r"""Loss between two weighted point clouds, compatible with GeomLoss 0.3.1.

    Create the loss once, then call it like a function: ``loss(x, y)`` (uniform
    weights), ``loss(a, x, b, y)`` (explicit weights) or
    ``loss(l_x, a, x, l_y, b, y)`` (explicit cluster labels, ``multiscale``
    backend). Points have shape (N, D), or (B, N, D) for a batch; weights have
    shape (N,) or (B, N). The result is a scalar, or a (B,) tensor for a batch,
    differentiable with respect to the points and the weights.

    :param loss: ``"sinkhorn"`` (debiased Sinkhorn divergence), ``"hausdorff"``,
        ``"energy"``, ``"gaussian"`` or ``"laplacian"``.
    :param p: exponent of the default cost: $\|x - y\|$ for $p = 1$, $\frac{1}{2}\|x - y\|^2$ for
        $p = 2$.
    :param blur: length scale $\sigma$. Entropic regularization uses $\varepsilon = \texttt{blur}^p$; for
        kernel losses, it is the width of the kernel.
    :param reach: ``None`` for balanced transport, or a length scale for
        unbalanced transport ($\rho = \texttt{reach}^p$).
    :param diameter: an upper bound on the distance between points, where the
        $\varepsilon$-scaling starts; estimated from the data when ``None``.
    :param scaling: ratio between successive blur values of the $\varepsilon$-scaling,
        between 0 and 1. Larger is slower and more accurate.
    :param truncate: ``multiscale`` backend: pairs of clusters whose interaction
        is below $\exp(-\texttt{truncate})$ are skipped; ``None`` keeps every pair.
    :param cost: a custom cost: a function of kernox formulas ``x`` (``Vi``)
        and ``y`` (``Vj``) returning a scalar formula, such as
        ``lambda x, y: (x - y).norm2()`` (``backend="auto"`` then selects
        ``online``); a ``(formula, dense routine)`` pair for the ``tensorized``
        and ``multiscale`` backends; a function on tensors with
        ``backend="tensorized"``; or a ``remblais.Cost``. KeOps formula strings
        are not supported.
    :param kernel: a custom kernel, for ``loss="hausdorff"``.
    :param cluster_scale: ``multiscale`` backend: size of the grid cells;
        estimated from the data when ``None``.
    :param debias: subtract the self-transport terms, so that the loss of a
        measure with itself is zero.
    :param potentials: return the dual potentials $(F, G)$ instead of the value.
    :param verbose: print the clustering and the $\varepsilon$-scaling schedule.
    :param backend: ``"auto"``, ``"tensorized"`` (dense PyTorch matrices),
        ``"online"`` (fused kernox reductions) or ``"multiscale"`` (clustering
        and block-sparse kernox reductions, dimension 3 or less).
    :param fast_math: specific to remblais. ``None`` (default): approximate
        ``exp`` and ``log`` in the kernox reductions on NVIDIA GPUs in FP32, as
        PyKeOps does under GeomLoss, and IEEE arithmetic elsewhere. ``False``:
        IEEE arithmetic everywhere. ``True``: ask kernox for fast math (no effect
        in FP64 and on the CPU, refused by the backends that do not offer it).
    :param flush_denormals: specific to remblais. ``None`` (default): subnormal
        numbers flushed to zero in the kernox reductions on the CPU and on NVIDIA
        GPUs in FP32 (as PyKeOps does under GeomLoss), kept elsewhere. ``False``:
        kept everywhere. ``True``: ask kernox to flush them (no effect in FP64 on
        NVIDIA GPUs, refused by the backends that do not offer it).
    :param distributed: specific to remblais. ``None`` (default): local
        reductions. ``True``: the kernox reductions of the ``online`` and
        ``multiscale`` backends are split across the processes of the default
        ``torch.distributed`` group (each process computes a share of the rows,
        see ``kernox.distributed``); a process group: across that group. Every
        process passes the same inputs and receives the same result, with
        gradients. The ``tensorized`` backend stays local.
    """

    def __init__(self, loss="sinkhorn", p=2, blur=0.05, reach=None, diameter=None, scaling=0.5, truncate=5,
                 cost=None, kernel=None, cluster_scale=None, debias=True, potentials=False, verbose=False,
                 backend="auto", fast_math=None, flush_denormals=None, distributed=None):
        super().__init__()
        if loss not in routines or loss == "hausdorff_kernel":
            raise ValueError(f"unknown loss {loss!r}: expected sinkhorn, hausdorff, energy, gaussian or laplacian")
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}: expected one of {', '.join(BACKENDS)}")
        self.loss = loss
        self.backend = backend
        self.p = p
        self.blur = blur
        self.reach = reach
        self.truncate = truncate
        self.diameter = diameter
        self.scaling = scaling
        self.cost = cost
        self.kernel = kernel
        self.cluster_scale = cluster_scale
        self.debias = debias
        self.potentials = potentials
        self.verbose = verbose
        if fast_math not in (None, True, False):
            raise ValueError(f"fast_math: expected None, True or False, got {fast_math!r}")
        self.fast_math = fast_math
        if flush_denormals not in (None, True, False):
            raise ValueError(f"flush_denormals: expected None, True or False, got {flush_denormals!r}")
        self.flush_denormals = flush_denormals
        if distributed not in (None, False, True) and not _is_process_group(distributed):
            raise ValueError(f"distributed: expected None, True or a torch.distributed process group, got {distributed!r}")
        if distributed not in (None, False):
            try:
                import kernox.distributed  # noqa: F401
            except ImportError as e:
                raise ImportError("SamplesLoss(distributed=...) needs kernox 1.2 or later (kernox.distributed)") from e
        self.distributed = distributed

    def select_backend(self, B, N, M, D, labelled):
        """Rule used by ``backend="auto"`` (the one of GeomLoss v0.3.1)."""
        backend = self.backend
        if labelled:
            if backend in ("auto", "multiscale"):
                return "multiscale"
            raise ValueError('Explicit cluster labels are only supported with the "auto" and "multiscale" backends.')
        if backend == "auto":
            if is_symbolic_only(self.cost):  # no dense routine: only the online backend applies
                return "online"
            if M * N <= 5000**2:
                return "tensorized"
            if D <= 3 and self.loss == "sinkhorn" and M * N > 10000**2 and self.p == 2:
                return "multiscale"
            return "online"
        return backend

    def forward(self, *args):
        l_x, α, x, l_y, β, y = self.process_args(*args)
        B, N, M, D, l_x, α, l_y, β = self.check_shapes(l_x, α, x, l_y, β, y)

        backend = self.select_backend(B, N, M, D, l_x is not None or l_y is not None)

        if backend == "multiscale":  # the multiscale backend handles unbatched measures
            if B == 1:
                α, x, β, y = α.squeeze(0), x.squeeze(0), β.squeeze(0), y.squeeze(0)
            elif B > 1:
                warnings.warn("The 'multiscale' backend do not support batchsize > 1. "
                              "Using 'tensorized' instead: beware of memory overflows!")
                backend = "tensorized"

        if B == 0 and backend in ("tensorized", "online"):
            α, x, β, y = α.unsqueeze(0), x.unsqueeze(0), β.unsqueeze(0), y.unsqueeze(0)

        loss = self.loss
        if loss == "hausdorff" and self.kernel is not None:
            loss = "hausdorff_kernel"

        fm = self.fast_math
        if fm is None:
            fm = _kp.default_fast_math(x.device, x.dtype)
        fd = self.flush_denormals
        if fd is None:
            fd = _kp.default_flush_denormals(x.device, x.dtype)
        group = None if self.distributed in (None, False) else (None if self.distributed is True else self.distributed,)
        with _kp.fast_math(fm), _kp.flush_denormals(fd), _kp.distributed(group):
            values = routines[loss][backend](
                α, x, β, y, p=self.p, blur=self.blur, reach=self.reach, diameter=self.diameter,
                scaling=self.scaling, truncate=self.truncate, cost=self.cost, kernel=self.kernel,
                cluster_scale=self.cluster_scale, debias=self.debias, potentials=self.potentials,
                labels_x=l_x, labels_y=l_y, verbose=self.verbose)

        if self.potentials:
            F, G = values
            return F.view_as(α), G.view_as(β)
        if backend == "multiscale":
            return values if B == 0 else values.view(-1)
        return values[0] if B == 0 else values

    def process_args(self, *args):
        if len(args) == 6:
            return args
        if len(args) == 4:
            α, x, β, y = args
            return None, α, x, None, β, y
        if len(args) == 2:
            x, y = args
            return None, self.generate_weights(x), x, None, self.generate_weights(y), y
        raise ValueError("A SamplesLoss accepts two (x, y), four (α, x, β, y) or six (l_x, α, x, l_y, β, y)  arguments.")

    def generate_weights(self, x):
        if x.dim() == 2:
            return torch.ones(x.shape[0]).type_as(x) / x.shape[0]
        if x.dim() == 3:
            B, N, _ = x.shape
            return torch.ones(B, N).type_as(x) / N
        raise ValueError("Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors.")

    def check_shapes(self, l_x, α, x, l_y, β, y):
        if α.dim() != β.dim():
            raise ValueError("Input weights 'α' and 'β' should have the same number of dimensions.")
        if x.dim() != y.dim():
            raise ValueError("Input samples 'x' and 'y' should have the same number of dimensions.")
        if x.shape[-1] != y.shape[-1]:
            raise ValueError("Input samples 'x' and 'y' should have the same last dimension.")

        if x.dim() == 2:
            B = 0
            N, D = x.shape
            M, _ = y.shape
            if α.dim() not in (1, 2):
                raise ValueError("Without batches, input weights 'α' and 'β' should be encoded as (N,) or (N,1) tensors.")
            if α.dim() == 2:
                if α.shape[1] > 1:
                    raise ValueError("Without batches, input weights 'α' should be encoded as (N,) or (N,1) tensors.")
                if β.shape[1] > 1:
                    raise ValueError("Without batches, input weights 'β' should be encoded as (M,) or (M,1) tensors.")
                α, β = α.view(-1), β.view(-1)
            l_x = self._check_labels(l_x, N, "l_x", "x", "N")
            l_y = self._check_labels(l_y, M, "l_y", "y", "M")
            N2, M2 = α.shape[0], β.shape[0]
        elif x.dim() == 3:
            B, N, D = x.shape
            B2, M, _ = y.shape
            if B != B2:
                raise ValueError("Samples 'x' and 'y' should have the same batchsize.")
            if α.dim() not in (2, 3):
                raise ValueError("With batches, input weights 'α' and 'β' should be encoded as (B,N) or (B,N,1) tensors.")
            if α.dim() == 3:
                if α.shape[2] > 1:
                    raise ValueError("With batches, input weights 'α' should be encoded as (B,N) or (B,N,1) tensors.")
                if β.shape[2] > 1:
                    raise ValueError("With batches, input weights 'β' should be encoded as (B,M) or (B,M,1) tensors.")
                α, β = α.squeeze(-1), β.squeeze(-1)
            if l_x is not None or l_y is not None:
                raise NotImplementedError('The "multiscale" backend has not been implemented with batches.')
            B2, N2 = α.shape
            B3, M2 = β.shape
            if B != B2:
                raise ValueError("Samples 'x' and weights 'α' should have the same batchsize.")
            if B != B3:
                raise ValueError("Samples 'y' and weights 'β' should have the same batchsize.")
        else:
            raise ValueError("Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors.")

        if N != N2:
            raise ValueError("Weights 'α' and samples 'x' should have compatible shapes.")
        if M != M2:
            raise ValueError("Weights 'β' and samples 'y' should have compatible shapes.")
        return B, N, M, D, l_x, α, l_y, β

    @staticmethod
    def _check_labels(l, n, name, pts, size):
        if l is None:
            return None
        if l.dim() not in (1, 2):
            raise ValueError(f"Without batches, the vector of labels '{name}' should be encoded as an ({size},) or ({size},1) tensor.")
        if l.dim() == 2:
            if l.shape[1] > 1:
                raise ValueError(f"Without batches, the vector of labels '{name}' should be encoded as ({size},) or ({size},1) tensors.")
            l = l.view(-1)
        if len(l) != n:
            raise ValueError(f"The vector of labels '{name}' should have the same length as the point cloud '{pts}'.")
        return l
