"""Bridge to kernox: costs, kernels, reduction plans compiled once and reused.

Each reduction is compiled **once** per (cost, D, dtype, backend): ε (or
1/blur) is a parameter of the plan, so all the iterations of ε-scaling, the
four softmins (xy, yx, xx, yy), the coarse and fine levels of the multiscale
backend and successive calls to the loss share the same kernel.

On the CPU, kernox has no native backend: the blockwise Torch oracle is used
instead (an explicit reference, without a full M×N matrix).
"""

import contextlib
import contextvars
import os
import threading

import torch

import kernox as kp

_DTYPES = {torch.float16: "float16", torch.float32: "float32", torch.float64: "float64"}


def dtype_name(t):
    try:
        return _DTYPES[t.dtype]
    except KeyError:
        raise TypeError(f"remblais: unsupported dtype {t.dtype} (float16/32/64)") from None


def backend_for(device):
    """``oracle`` on the CPU (kernox has no native CPU backend), ``auto`` otherwise."""
    return "oracle" if torch.device(device).type == "cpu" else "auto"


# ---------------------------------------------------------------------------------
# Symbolic costs
# ---------------------------------------------------------------------------------
def _norm(d):
    """‖d‖ with a zero gradient at d = 0 (like KeOps ``Norm2``, where Rsqrt(0) = 0)."""
    d2 = d.sqnorm2()
    return kp.where(d2 > 0, kp.sqrt(d2), 0.0)


def cost_p1(x, y):
    return _norm(x - y)


def cost_p2(x, y):
    return (x - y).sqnorm2() / 2


def sqdist(x, y):
    return (x - y).sqnorm2()


DEFAULT_COSTS = {1: cost_p1, 2: cost_p2}


class Cost:
    """Transport cost: a kernox formula (online backends) and a dense matrix (tensorized backend).

    ``formula(x, y)`` receives two ``Formula`` objects (``Vi``, ``Vj``) and returns a
    scalar formula; ``dense(x, y)`` receives (B, N, D), (B, M, D) or (N, D), (M, D)
    tensors and returns the cost matrix.

    ``kernel_formula`` and ``factor`` (optional): C = factor · kernel_formula. The
    online plans then use ``kernel_formula`` and the parameter ``s = factor/ε``
    (one multiplication per pair instead of one per coordinate: symox distributes
    constants over sums).
    """

    def __init__(self, key, formula=None, dense=None, kernel_formula=None, factor=1.0):
        self.key = key
        self.formula = formula
        self.dense = dense
        self.kernel_formula = kernel_formula or formula
        self.factor = factor if kernel_formula is not None else 1.0


# ---------------------------------------------------------------------------------
# Plan cache
# ---------------------------------------------------------------------------------
_PLANS = {}
_LOCK = threading.Lock()


# Approximate exp/log (kernox fast_math option, CUDA FP32) for the plans compiled
# in this context; set by SamplesLoss around each call. Gradient plans inherit
# it (kernox), even outside the context.
_FAST_MATH = contextvars.ContextVar("remblais_fast_math", default=False)


@contextlib.contextmanager
def fast_math(flag):
    """Context manager: kernox plans compiled in it use ``fast_math=flag``."""
    token = _FAST_MATH.set(bool(flag))
    try:
        yield
    finally:
        _FAST_MATH.reset(token)


def fast_math_enabled():
    return _FAST_MATH.get()


def default_fast_math(device, dtype):
    """Default setting: fast math on NVIDIA GPUs in FP32 (like PyKeOps, hence
    GeomLoss), IEEE elsewhere (FP64, FP16, ROCm, XPU, CPU)."""
    device = torch.device(device)
    return device.type == "cuda" and torch.version.hip is None and dtype == torch.float32


def _plan(key, build, dtype, backend):
    fm = _FAST_MATH.get() and backend != "oracle"
    full = key + (dtype, backend, fm)
    run = _PLANS.get(full)
    if run is None:
        with _LOCK:
            run = _PLANS.get(full)
            if run is None:
                run = kp.compile(build(), dtype=dtype, backend=backend, **({"fast_math": True} if fm else {}))
                _PLANS[full] = run
    return run


def lse_plan(cost, D, dtype, backend):
    """``log Σ_j exp(h_j − s·C(x_i, y_j))``; ``s = 1/ε`` is a parameter."""

    def build():
        x, y, h, s = kp.Vi("x", D), kp.Vj("y", D), kp.Vj("h", 1), kp.Param("s")
        return (h - s * cost.kernel_formula(x, y)).logsumexp(over="j")

    return _plan(("lse", cost.key, D), build, dtype, backend)


def lse2_plan(cost, D, dtype, backend):
    """LSE with two sets of scores: ``log Σ_j exp(q_i·a_j + r_i·b_j − s·C(x_i, y_j))``.

    With (q_i, r_i) ∈ {(1, 0), (0, 1)}, the selection is exact (products by 0 or
    1, sum with a zero): two softmins with the same j set and different rows run
    in a single launch.
    """

    def build():
        x, q, r = kp.Vi("x", D), kp.Vi("q", 1), kp.Vi("r", 1)
        y, ha, hb, s = kp.Vj("y", D), kp.Vj("ha", 1), kp.Vj("hb", 1), kp.Param("s")
        return (q * ha + r * hb - s * cost.kernel_formula(x, y)).logsumexp(over="j")

    return _plan(("lse2", cost.key, D), build, dtype, backend)


def lse2(cost, x, q, r, y, ha, hb, s):
    """x (…, N, D), q/r (…, N), y (…, M, D), ha/hb (…, M); returns (…, N).

    ``s``: the **full** parameter (``cost.factor/ε``), a number or a (1,) tensor.
    """
    run = lse2_plan(cost, x.shape[-1], dtype_name(x), backend_for(x.device))
    out = _call(run, None, x=x, q=q.unsqueeze(-1), r=r.unsqueeze(-1), y=y, ha=ha.unsqueeze(-1),
                hb=hb.unsqueeze(-1), s=s)
    return out.squeeze(-1)


def lse2dot_plan(D, dtype, backend):
    """Dot-product variant of :func:`lse2_plan` for ½‖x − y‖²:
    ``log Σ_j exp(q_i·a_j + r_i·b_j + s⟨x_i, y_j⟩)`` (the norms go into a, b and the output)."""

    def build():
        x, q, r = kp.Vi("x", D), kp.Vi("q", 1), kp.Vi("r", 1)
        y, ha, hb, s = kp.Vj("y", D), kp.Vj("ha", 1), kp.Vj("hb", 1), kp.Param("s")
        return (q * ha + r * hb + s * x.dot(y)).logsumexp(over="j")

    return _plan(("lse2dot", D), build, dtype, backend)


def lse2dot(x, q, r, y, ha, hb, s):
    run = lse2dot_plan(x.shape[-1], dtype_name(x), backend_for(x.device))
    out = _call(run, None, x=x, q=q.unsqueeze(-1), r=r.unsqueeze(-1), y=y, ha=ha.unsqueeze(-1),
                hb=hb.unsqueeze(-1), s=s)
    return out.squeeze(-1)


def kernel_plan(kernel, D, dtype, backend):
    """``Σ_j k(x_i, y_j) b_j``; ``s = 1/blur`` is a parameter."""

    def build():
        x, y, b, s = kp.Vi("x", D), kp.Vj("y", D), kp.Vj("b", 1), kp.Param("s")
        return (kernel.formula(x, y, s) * b).sum_over("j")

    return _plan(("kernel", kernel.key, D), build, dtype, backend)


def _contig(t):
    """Contiguous, except for broadcast batch dimensions (stride 0), which kernox accepts as they are."""
    if t.is_contiguous():
        return t
    if t.dim() >= 3 and t.stride(-1) == 1 and t.stride(-2) == t.shape[-1]:
        inner = t.shape[-1] * t.shape[-2]
        expected = inner
        for size, stride in zip(reversed(t.shape[:-2]), reversed(t.stride()[:-2])):
            if stride not in (0, expected) and size != 1:
                return t.contiguous()
            expected *= size if stride else 1
        return t
    return t.contiguous()


# Range validation (bounds, overlaps; one host synchronization per call):
# REMBLAIS_VALIDATE_RANGES=1, enabled by the tests.
VALIDATE_RANGES = os.environ.get("REMBLAIS_VALIDATE_RANGES", "0") == "1"


def _call(run, ranges, **kw):
    if ranges is not None:
        run = run.with_ranges(ranges, validate=VALIDATE_RANGES)
    return run(**{k: (_contig(v) if torch.is_tensor(v) else v) for k, v in kw.items()})


def lse(cost, x, y, h, inv_eps, ranges=None):
    """Online LSE; x (…, N, D), y (…, M, D), h (…, M). Returns (…, N)."""
    run = lse_plan(cost, x.shape[-1], dtype_name(x), backend_for(x.device))
    out = _call(run, ranges, x=x, y=y, h=h.unsqueeze(-1), s=cost.factor * inv_eps)
    return out.squeeze(-1)


def lse_full(cost, x, y, h, s):
    """Like :func:`lse`, with the **full** parameter ``s`` (``cost.factor/ε``, a number or a (1,) tensor)."""
    run = lse_plan(cost, x.shape[-1], dtype_name(x), backend_for(x.device))
    return _call(run, None, x=x, y=y, h=h.unsqueeze(-1), s=s).squeeze(-1)


def kernel_product(kernel, x, y, b, inv_blur, ranges=None):
    """Online Σ_j k(x_i, y_j) b_j; returns (…, N)."""
    run = kernel_plan(kernel, x.shape[-1], dtype_name(x), backend_for(x.device))
    out = _call(run, ranges, x=x, y=y, b=b.unsqueeze(-1), s=inv_blur)
    return out.squeeze(-1)


def swap_axes(ranges):
    """Swaps the i and j roles of a complete :class:`kernox.Ranges`."""
    return kp.Ranges(ranges.ranges_j, ranges.slices_j, ranges.redranges_i,
                     ranges.ranges_i, ranges.slices_i, ranges.redranges_j)
