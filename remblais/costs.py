"""Transport costs: defaults for p = 1, 2, KeOps formula strings, kernox formulas.

A custom cost can be given:

- as a **KeOps string** (``"Norm2(X-Y)"``, ``"(SqDist(X,Y) / IntCst(2))"``, ...),
  the GeomLoss syntax for the ``online`` and ``multiscale`` backends; it is
  parsed here (common subset of the KeOps grammar, see :data:`FUNCTIONS`) and
  works with **all** backends;
- as a **Python function**: for the ``tensorized`` backend, it receives
  (B, N, D), (B, M, D) tensors and returns (B, N, M) (GeomLoss semantics); for
  ``online`` and ``multiscale``, it receives kernox formulas ``x`` (Vi) and
  ``y`` (Vj) and returns a scalar formula;
- as a ``(formula, dense routine)`` pair (the GeomLoss multiscale convention),
  the formula being a KeOps string or a kernox function;
- as a :class:`remblais.Cost`.
"""

import re

import torch

import kernox as kp

from ._kp import DEFAULT_COSTS, Cost, sqdist
from .utils import distances, squared_distances

_DENSE_DEFAULTS = {1: lambda x, y: distances(x, y), 2: lambda x, y: squared_distances(x, y) / 2}


# ---------------------------------------------------------------------------------
# Formules KeOps en chaîne
# ---------------------------------------------------------------------------------
_TOKEN = re.compile(r"\s*(?:(\d+\.\d*(?:[eE][-+]?\d+)?|\d*\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?)"
                    r"|([A-Za-z_][A-Za-z_0-9]*)|(\S))")


def _tokens(s):
    pos, out = 0, []
    s = s.strip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if m is None:
            break
        num, name, op = m.groups()
        out.append(("num", float(num)) if num else ("name", name) if name else ("op", op))
        pos = m.end()
    return out


class _Parser:
    def __init__(self, s):
        self.toks = _tokens(s)
        self.i = 0
        self.src = s

    def error(self, msg):
        return ValueError(f"formule KeOps {self.src!r} : {msg}")

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self, kind=None, val=None):
        tok = self.peek()
        if (kind and tok[0] != kind) or (val is not None and tok[1] != val):
            raise self.error(f"attendu {val or kind}, trouvé {tok[1]!r}")
        self.i += 1
        return tok

    def parse(self):
        node = self.expr()
        if self.i != len(self.toks):
            raise self.error(f"jeton inattendu {self.peek()[1]!r}")
        return node

    def expr(self):
        node = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            node = ("add" if op == "+" else "sub", node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            node = ("mul" if op == "*" else "div", node, self.unary())
        return node

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return ("neg", self.unary())
        return self.atom()

    def atom(self):
        kind, val = self.peek()
        if kind == "num":
            self.take()
            return ("const", val)
        if kind == "op" and val == "(":
            self.take()
            node = self.expr()
            if self.peek() == ("op", "|"):  # produit scalaire KeOps (a|b)
                self.take()
                node = ("Scalprod", node, self.expr())
            self.take("op", ")")
            return node
        if kind == "name":
            self.take()
            if self.peek() == ("op", "("):
                self.take()
                args = [self.expr()]
                while self.peek() == ("op", ","):
                    self.take()
                    args.append(self.expr())
                self.take("op", ")")
                if val not in FUNCTIONS:
                    raise self.error(f"fonction {val!r} non prise en charge ({', '.join(sorted(FUNCTIONS))})")
                if val == "IntCst":
                    return ("const", args[0][1])
                return (val, *args)
            if val in ("X", "Y"):
                return ("var", val)
            raise self.error(f"variable {val!r} inconnue (seules X et Y sont permises)")
        raise self.error(f"jeton inattendu {val!r}")


FUNCTIONS = {
    "SqDist", "Norm2", "SqNorm2", "Sqrt", "Rsqrt", "Exp", "Log", "Abs", "Sum", "Square",
    "IntCst", "Pow", "Powf", "Scalprod", "Inv", "Sin", "Cos", "Minus", "Normalize",
}


class _FormulaOps:
    """Interpretation on kernox formulas."""

    reduce_sum = staticmethod(lambda a: a.sum())

    @staticmethod
    def norm(a):
        d2 = a.sqnorm2()
        return kp.where(d2 > 0, kp.sqrt(d2), 0.0)

    sqrt, rsqrt, exp, log, abs = kp.sqrt, kp.rsqrt, kp.exp, kp.log, kp.abs
    sin, cos = kp.sin, kp.cos


class _DenseOps:
    """Interpretation on (…, N, 1, D) / (…, 1, M, D) tensors."""

    reduce_sum = staticmethod(lambda a: a.sum(-1, keepdim=True))

    @staticmethod
    def norm(a):
        return (a * a).sum(-1, keepdim=True).sqrt()

    sqrt, rsqrt, exp, log, abs = torch.sqrt, torch.rsqrt, torch.exp, torch.log, torch.abs
    sin, cos = torch.sin, torch.cos


def _eval(node, X, Y, ops):
    tag = node[0]
    ev = lambda n: _eval(n, X, Y, ops)  # noqa: E731
    if tag == "const":
        return node[1]
    if tag == "var":
        return X if node[1] == "X" else Y
    if tag == "add":
        return ev(node[1]) + ev(node[2])
    if tag == "sub":
        return ev(node[1]) - ev(node[2])
    if tag == "mul":
        return ev(node[1]) * ev(node[2])
    if tag == "div":
        return ev(node[1]) / ev(node[2])
    if tag in ("neg", "Minus"):
        return -ev(node[1])
    a = ev(node[1])
    if tag == "SqDist":
        d = a - ev(node[2])
        return ops.reduce_sum(d * d)
    if tag == "SqNorm2":
        return ops.reduce_sum(a * a)
    if tag == "Norm2":
        return ops.norm(a)
    if tag == "Normalize":
        return a / ops.norm(a)
    if tag == "Scalprod":
        return ops.reduce_sum(a * ev(node[2]))
    if tag == "Sum":
        return ops.reduce_sum(a)
    if tag == "Square":
        return a * a
    if tag == "Inv":
        return 1.0 / a
    if tag == "Pow":
        n = node[2]
        if n[0] != "const" or n[1] != int(n[1]):
            raise ValueError("Pow(a, n) : n doit être un entier constant")
        return a ** int(n[1])
    if tag == "Powf":
        return a ** ev(node[2])
    return getattr(ops, tag.lower())(a)


def keops_string_cost(s):
    """``Cost`` from a KeOps formula string, with variables ``X`` and ``Y``.

    The string is parsed by remblais, with the common subset of the KeOps
    grammar listed in ``remblais.costs.FUNCTIONS`` (``SqDist``, ``Norm2``,
    ``SqNorm2``, ``Sum``, ``Square``, ``Sqrt``, ``Exp``, ``Log``, ``IntCst``,
    ``Pow``, ``Powf``, ...), ``(a|b)`` and arithmetic, and works with every
    backend. ``SamplesLoss(cost="...")`` calls it for you.
    """
    tree = _Parser(s).parse()

    def formula(x, y):
        out = _eval(tree, x, y, _FormulaOps)
        if not isinstance(out, kp.Formula):
            raise ValueError(f"formule KeOps {s!r} : le coût ne dépend ni de X ni de Y")
        if out.dim != 1:
            raise ValueError(f"formule KeOps {s!r} : coût scalaire attendu (dimension {out.dim})")
        return out

    def dense(x, y):
        if x.dim() == 2:
            return dense(x[None], y[None])[0]
        out = _eval(tree, x[:, :, None, :], y[:, None, :, :], _DenseOps)
        if out.shape[-1] != 1:
            raise ValueError(f"formule KeOps {s!r} : coût scalaire attendu")
        return out.squeeze(-1)

    return Cost(("keops", s), formula, dense)


def _formula_only(fn):
    def dense(x, y):
        raise ValueError("ce coût est une formule kernox : utilisez backend='online' ou 'multiscale', "
                         "ou donnez une routine dense (couple (formule, routine))")
    return Cost(("callable", fn), fn, dense)


def resolve_cost(cost, p, backend):
    """Normalizes the ``cost`` argument of SamplesLoss into a :class:`Cost`."""
    if isinstance(cost, Cost):
        return cost
    if cost is None:
        if p not in DEFAULT_COSTS:
            raise ValueError(f"p = {p} non pris en charge sans coût personnalisé (p = 1 ou 2)")
        if p == 2:  # ½‖x − y‖² : le ½ passe dans le paramètre s = 1/(2ε)
            return Cost(("p", p), DEFAULT_COSTS[p], _DENSE_DEFAULTS[p], kernel_formula=sqdist, factor=0.5)
        return Cost(("p", p), DEFAULT_COSTS[p], _DENSE_DEFAULTS[p])
    if isinstance(cost, str):
        return keops_string_cost(cost)
    if isinstance(cost, tuple) and len(cost) == 2:
        formula, routine = cost
        base = keops_string_cost(formula) if isinstance(formula, str) else _formula_only(formula)
        return Cost(base.key + ("dense", routine), base.formula, routine)
    if callable(cost):
        if backend == "tensorized":
            return Cost(("dense", cost), None, cost)
        return _formula_only(cost)
    raise TypeError(f"cost : chaîne KeOps, fonction, couple (formule, routine) ou remblais.Cost attendu, reçu {cost!r}")


__all__ = ["Cost", "keops_string_cost", "resolve_cost", "FUNCTIONS"]
