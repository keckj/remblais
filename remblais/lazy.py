r"""Minimal ``LazyTensor`` on kernox (a subset of ``pykeops.torch.LazyTensor``).

It serves the ``remblais.ot`` API (a port of ``geomloss.ot``), written for
KeOps, and GeomLoss-style custom kernels. KeOps conventions:

- ``LazyTensor(t)`` with ``t`` of shape ``(*B, N, 1, D)``: an i variable;
  ``(*B, 1, M, D)``: a j variable; ``(*B, 1, 1, D)``: a parameter;
- a Python number becomes a **parameter** of the plan (its value is not part
  of the compilation key: ``C / eps`` does not recompile when $\varepsilon$ changes);
- ``shape`` is ``(*B, N, M)`` for a scalar formula, ``(*B, N, M, D)`` otherwise;
- reductions ``sum``, ``logsumexp``, ``min``, ``max`` over ``dim`` = the i axis
  (-3 + ...) or j, returning ``(*B, N, D)`` (or ``(*B, M, D)``); ``K @ v``
  (v dense (M, V)); ``K.T`` swaps i and j.

NumPy inputs are accepted (computed with Torch on the CPU, NumPy output).
Gradients: those of kernox (``torch.autograd``) with respect to the leaf
tensors.
"""

import numpy as np
import torch

import kernox as kp

from . import _kp

_UNARY = {
    "neg": lambda f: -f, "exp": kp.exp, "log": kp.log, "sqrt": kp.sqrt, "abs": kp.abs,
    "sin": kp.sin, "cos": kp.cos, "square": lambda f: f * f, "rsqrt": kp.rsqrt, "relu": lambda f: kp.relu(f),
    "sum": lambda f: f.sum(), "sqnorm2": lambda f: f.sqnorm2(),
    "norm2": lambda f: kp.where(f.sqnorm2() > 0, kp.sqrt(f.sqnorm2()), 0.0),
}
_BINARY = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b, "div": lambda a, b: a / b,
    "dot": lambda a, b: a.dot(b),
}


class LazyTensor:
    """Lazy symbolic (i, j) expression; see the module."""

    def __init__(self, x=None, *, _node=None, _leaves=None, _shape=None, _numpy=False):
        if _node is not None:
            self._node, self._leaves, self._shape, self._numpy = _node, _leaves, _shape, _numpy
            return
        self._numpy = isinstance(x, np.ndarray)
        t = torch.from_numpy(x) if self._numpy else x
        if not torch.is_tensor(t) or t.dim() < 3:
            raise ValueError("LazyTensor expects a (…, N, 1, D), (…, 1, M, D) or (…, 1, 1, D) tensor")
        ni, nj, d = t.shape[-3:]
        if ni != 1 and nj != 1:
            raise ValueError(f"LazyTensor: shape {tuple(t.shape)} is neither of type i (N, 1, D) nor j (1, M, D)")
        kind = "i" if ni != 1 else ("j" if nj != 1 else "p")
        self._leaves = [(kind, t)]
        self._node = ("leaf", 0, kind, d)
        self._shape = (tuple(t.shape[:-3]), ni, nj, d)

    # ------------------------------------------------------------------ shapes
    @property
    def shape(self):
        b, n, m, d = self._shape
        return b + (n, m) + ((d,) if d > 1 else ())

    @property
    def ndim(self):
        return len(self.shape)

    def dim(self):
        return self.ndim

    @property
    def dtype(self):
        return self._leaves[0][1].dtype

    # ------------------------------------------------------------- construction
    @staticmethod
    def _wrap(o, like):
        if isinstance(o, LazyTensor):
            return o
        if isinstance(o, (int, float, np.floating, np.integer)):
            return LazyTensor(_node=("scalar", 0), _leaves=[("s", float(o))], _shape=((), 1, 1, 1),
                              _numpy=like._numpy)
        if torch.is_tensor(o) or isinstance(o, np.ndarray):
            return LazyTensor(o)
        return NotImplemented

    def _combine(self, op, other, reverse=False):
        other = self._wrap(other, self)
        if other is NotImplemented:
            return NotImplemented
        a, b = (other, self) if reverse else (self, other)
        node = (op, a._node, _shift(b._node, len(a._leaves)))
        (ba, na, ma, da), (bb, nb, mb, db) = a._shape, b._shape
        d = max(da, db) if op != "dot" else 1
        if op != "dot" and da != db and 1 not in (da, db):
            raise ValueError(f"LazyTensor: incompatible dimensions {da} and {db}")
        shape = (tuple(torch.broadcast_shapes(ba, bb)), max(na, nb), max(ma, mb), d)
        return LazyTensor(_node=node, _leaves=a._leaves + b._leaves, _shape=shape, _numpy=a._numpy or b._numpy)

    def _unary(self, op, d=None):
        b, n, m, dd = self._shape
        return LazyTensor(_node=(op, self._node), _leaves=self._leaves, _shape=(b, n, m, dd if d is None else d),
                          _numpy=self._numpy)

    __add__ = lambda s, o: s._combine("add", o)  # noqa: E731
    __radd__ = lambda s, o: s._combine("add", o, True)  # noqa: E731
    __sub__ = lambda s, o: s._combine("sub", o)  # noqa: E731
    __rsub__ = lambda s, o: s._combine("sub", o, True)  # noqa: E731
    __mul__ = lambda s, o: s._combine("mul", o)  # noqa: E731
    __rmul__ = lambda s, o: s._combine("mul", o, True)  # noqa: E731
    __truediv__ = lambda s, o: s._combine("div", o)  # noqa: E731
    __rtruediv__ = lambda s, o: s._combine("div", o, True)  # noqa: E731

    def __neg__(self):
        return self._unary("neg")

    def __pow__(self, k):
        if k == 2:
            return self._unary("square")
        if k == 0.5:
            return self._unary("sqrt")
        if k == -0.5:
            return self._unary("rsqrt")
        if isinstance(k, int) and k > 0:
            out = self
            for _ in range(k - 1):
                out = out * self
            return out
        raise NotImplementedError(f"LazyTensor ** {k!r}: only 2, 0.5, −0.5 and positive integers are supported")

    def exp(self):
        return self._unary("exp")

    def log(self):
        return self._unary("log")

    def sqrt(self):
        return self._unary("sqrt")

    def rsqrt(self):
        return self._unary("rsqrt")

    def abs(self):
        return self._unary("abs")

    def __abs__(self):
        return self._unary("abs")

    def sin(self):
        return self._unary("sin")

    def cos(self):
        return self._unary("cos")

    def relu(self):
        return self._unary("relu")

    def square(self):
        return self._unary("square")

    def sqnorm2(self):
        return self._unary("sqnorm2", 1)

    def norm2(self):
        r"""$\|\cdot\|$ with a zero gradient at 0 (like KeOps ``Norm2``)."""
        return self._unary("norm2", 1)

    def sqdist(self, other):
        return (self - other).sqnorm2()

    def __or__(self, other):  # KeOps dot product (x | y)
        return self._combine("dot", other)

    @property
    def T(self):
        b, n, m, d = self._shape
        leaves = [("j" if k == "i" else "i" if k == "j" else k, _swap_tensor(t) if k in "ij" else t)
                  for k, t in self._leaves]
        return LazyTensor(_node=_swap_kinds(self._node), _leaves=leaves, _shape=(b, m, n, d), _numpy=self._numpy)

    def t(self):
        return self.T

    # ---------------------------------------------------------------- reductions
    def _axis(self, dim):
        nb = len(self._shape[0])
        if dim is None:
            raise ValueError("LazyTensor: reduction without 'dim'")
        if dim < 0:
            dim += nb + 2 + (1 if self._shape[3] > 1 else 0)
        if dim == nb:
            return "i"
        if dim == nb + 1:
            return "j"
        if dim == nb + 2 and self._shape[3] > 1:
            return "c"
        raise ValueError(f"LazyTensor: invalid reduction axis {dim} for shape {self.shape}")

    def sum(self, dim=None, axis=None, **kw):
        ax = self._axis(axis if dim is None else dim)
        if ax == "c":
            return self._unary("sum", 1)
        return self._reduce("sum", ax)

    def logsumexp(self, dim=None, axis=None, weight=None, **kw):
        ax = self._axis(axis if dim is None else dim)
        return self._reduce("lse", ax, weight=weight)

    def min(self, dim=None, axis=None, **kw):
        return self._reduce("min", self._axis(axis if dim is None else dim))

    def max(self, dim=None, axis=None, **kw):
        return self._reduce("max", self._axis(axis if dim is None else dim))

    def argmin(self, dim=None, axis=None, **kw):
        return self._reduce("argmin", self._axis(axis if dim is None else dim))

    def argmax(self, dim=None, axis=None, **kw):
        return self._reduce("argmax", self._axis(axis if dim is None else dim))

    def __matmul__(self, v):
        r"""$\sum_j K_{ij} v_j$, with $v$ dense (M,) or (M, V) (or (*B, M, V))."""
        numpy = isinstance(v, np.ndarray)
        vt = torch.from_numpy(v) if numpy else v
        squeeze = vt.dim() == 1
        if squeeze:
            vt = vt[:, None]
        vj = LazyTensor(vt.unsqueeze(-3))
        out = (self * vj)._reduce("sum", "j")
        if squeeze:
            out = out[..., 0]
        return out.numpy() if numpy and torch.is_tensor(out) else out

    def _reduce(self, kind, axis, weight=None):
        src = self if axis == "j" else self.T
        if weight is not None:
            raise NotImplementedError("LazyTensor.logsumexp(weight=) is not supported")
        out = _run(src, kind)
        if self._numpy:
            return out.detach().numpy()
        return out


def _shift(node, k):
    if node[0] == "leaf":
        return ("leaf", node[1] + k) + node[2:]
    if node[0] == "scalar":
        return ("scalar", node[1] + k)
    return (node[0],) + tuple(_shift(c, k) for c in node[1:])


def _swap_kinds(node):
    if node[0] == "leaf":
        k = node[2]
        return ("leaf", node[1], "j" if k == "i" else "i" if k == "j" else k, node[3])
    if node[0] == "scalar":
        return node
    return (node[0],) + tuple(_swap_kinds(c) for c in node[1:])


def _swap_tensor(t):
    return t.transpose(-3, -2)


def _build(node, vars_):
    tag = node[0]
    if tag in ("leaf", "scalar"):
        return vars_[node[1]]
    if tag in _UNARY:
        return _UNARY[tag](_build(node[1], vars_))
    return _BINARY[tag](_build(node[1], vars_), _build(node[2], vars_))


def _run(lt, kind):
    """Reduction of ``lt`` over j; returns (*B, N, D) (including D = 1)."""
    leaves = lt._leaves
    tensors = [t for k, t in leaves if k != "s"]
    ref = tensors[0]
    dtype = ref.dtype
    for t in tensors:
        if t.dtype != dtype:
            raise ValueError("LazyTensor: the operands have different dtypes")
    dims = [(k, (t.shape[-1] if k != "s" else 1)) for k, t in leaves]
    key = ("lazy", kind, lt._node, tuple(dims))
    nbatch = len(lt._shape[0])

    def build():
        vars_ = []
        for idx, (k, d) in enumerate(dims):
            name = f"v{idx}"
            vars_.append(kp.Vi(name, d) if k == "i" else kp.Vj(name, d) if k == "j" else kp.Param(name, d))
        f = _build(lt._node, vars_)
        if kind == "sum":
            return f.sum_over("j")
        if kind == "lse":
            return f.logsumexp(over="j")
        if kind in ("min", "max", "argmin", "argmax"):
            return getattr(f, kind)(over="j")
        raise ValueError(kind)

    if not any(k == "i" for k, _ in leaves) or not any(k == "j" for k, _ in leaves):
        return _run_dense(lt, kind)
    try:
        run = _kp._plan(key, build, _kp.dtype_name(ref), _kp.backend_for(ref.device))
    except kp.UnsupportedError as e:
        msg = str(e)  # kernox: the formula needs at least one i variable and one j variable
        if "variable j" not in msg and "j variable" not in msg:
            raise
        return _run_dense(lt, kind)  # i or j variable removed by symbolic simplification
    kwargs = {}
    names = set(run.input_names)
    for idx, (k, t) in enumerate(leaves):
        name = f"v{idx}"
        if name not in names:
            continue  # variable simplified away by symox (e.g. multiplied by 0)
        if k == "s":
            kwargs[name] = t
        elif k == "p":
            kwargs[name] = t.reshape(t.shape[:-3] + (t.shape[-1],)) if t.dim() > 3 else t.reshape(-1)
        else:
            u = t.squeeze(-2) if k == "i" else t.squeeze(-3)
            if nbatch and u.dim() - 2 < nbatch:
                u = u.reshape((1,) * (nbatch - (u.dim() - 2)) + tuple(u.shape))
            kwargs[name] = u
    for name, u in kwargs.items():
        if torch.is_tensor(u) and not _inner_contiguous(u):
            kwargs[name] = u.contiguous()
    if nbatch:
        _check_batch_ranks(kwargs, nbatch)
    return run(**kwargs)


def _inner_contiguous(u):
    if u.dim() < 2:
        return u.is_contiguous()
    return u.stride(-1) == 1 and (u.shape[-2] <= 1 or u.stride(-2) == u.shape[-1])


def _check_batch_ranks(kwargs, nbatch):
    for name, u in kwargs.items():
        if torch.is_tensor(u) and u.dim() not in (1, nbatch + 2):
            raise ValueError(f"LazyTensor: inconsistent batch rank for {name}")


# ---------------------------------------------------------------------------------
# Dense fallback: formulas without an i or j variable (N = 1, M = 1, or simplification)
# ---------------------------------------------------------------------------------
_DENSE_UNARY = {
    "neg": lambda t: -t, "exp": torch.exp, "log": torch.log, "sqrt": torch.sqrt, "abs": torch.abs,
    "sin": torch.sin, "cos": torch.cos, "square": lambda t: t * t, "rsqrt": torch.rsqrt, "relu": torch.relu,
    "sum": lambda t: t.sum(-1, keepdim=True), "sqnorm2": lambda t: (t * t).sum(-1, keepdim=True),
    "norm2": lambda t: (t * t).sum(-1, keepdim=True).sqrt(),
}
_DENSE_BINARY = dict(_BINARY, dot=lambda a, b: (a * b).sum(-1, keepdim=True))


def _eval_dense(node, leaves):
    tag = node[0]
    if tag in ("leaf", "scalar"):
        return leaves[node[1]][1]
    if tag in _DENSE_UNARY:
        return _DENSE_UNARY[tag](_eval_dense(node[1], leaves))
    return _DENSE_BINARY[tag](_eval_dense(node[1], leaves), _eval_dense(node[2], leaves))


def _run_dense(lt, kind):
    """Torch evaluation by broadcasting (…, N|1, M|1, D), then reduction over j."""
    b, n, m, d = lt._shape
    v = _eval_dense(lt._node, lt._leaves)
    if not torch.is_tensor(v):
        ref = next(t for k, t in lt._leaves if k != "s")
        v = torch.full((1, 1, 1), float(v), dtype=ref.dtype, device=ref.device)
    v = v.expand(b + (v.shape[-3], m, v.shape[-1]))
    if kind == "sum":
        out = v.sum(-2)
    elif kind == "lse":
        out = v.logsumexp(-2)
    elif kind in ("min", "max"):
        out = getattr(v, "amin" if kind == "min" else "amax")(-2)
    else:
        out = getattr(v, kind)(-2)
    return out.expand(b + (n, out.shape[-1])).contiguous()
